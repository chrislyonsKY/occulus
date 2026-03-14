# ai-dev/architecture.md — Occulus

> This is the **AI-facing** architecture document. The human-facing version is `ARCHITECTURE.md`.
> This document includes implementation detail, decision rationale, and agent guidance
> that would clutter the public-facing doc.

---

## System Overview

Occulus is a pure-Python point cloud analysis library targeting aerial LiDAR, terrestrial laser
scanning (TLS), and UAV/photogrammetric point clouds. It exposes a platform-aware type hierarchy
(`PointCloud`, `AerialCloud`, `TerrestrialCloud`, `UAVCloud`) and a processing pipeline covering
I/O, filtering, normal estimation, registration, segmentation, meshing, feature extraction, and
metrics.

The library is **synchronous** (Option B). All I/O is blocking; no `asyncio` anywhere. Optional
dependencies (`laspy`, `open3d`) are lazy-imported at call time with clear install hints.

The C++ backend (`occulus._cpp`) is scaffolded for future acceleration but is **not required** for
any v1.0.0 operation. All algorithms run on pure NumPy + SciPy.

---

## Module Responsibilities

### `occulus/__init__.py`
- Exposes the public API surface.
- Imports are intentional — only what users should call directly.
- Do NOT re-export everything blindly.

### `occulus/exceptions.py`
- Base class: `OcculusError(Exception)`.
- All domain exceptions inherit from this base.
- Never raise bare `Exception` or `ValueError` for domain errors.
- Hierarchy: `OcculusIOError`, `OcculusValidationError`, `OcculusRegistrationError`,
  `OcculusSegmentationError`, `OcculusMeshError`, `OcculusFeatureError`, `OcculusCppError`,
  `OcculusNetworkError`, `UnsupportedPlatformError`.

### `occulus/types.py`
- **Responsibility**: Core data model. Platform-aware PointCloud hierarchy.
- **Key types**: `Platform` (enum), `ScanPosition` (dataclass), `AcquisitionMetadata` (dataclass),
  `PointCloud` (base), `AerialCloud`, `TerrestrialCloud`, `UAVCloud`.
- **Invariants**: `xyz` is always contiguous float64 (N, 3). Optional arrays must match `n_points`.
- **Open3D interop**: `to_open3d()` / `from_open3d()` — lazy-import `open3d`.

### `occulus/config.py`
- Constants and environment variable loading.
- All `OCCULUS_*` env vars read here; no `os.environ` access elsewhere.

### `occulus/io/`
- `readers.py`: `read()` dispatcher → `_read_las`, `_read_ply`, `_read_pcd`, `_read_xyz`.
- `writers.py`: `write()` dispatcher → `_write_las`, `_write_ply`, `_write_xyz`.
- LAS/LAZ: `laspy` (lazy import). PLY/PCD: `open3d` (lazy import). XYZ: NumPy only.
- `_make_cloud()` constructs the correct subtype from a platform hint.

### `occulus/filters/`
- Pure NumPy/SciPy. No optional deps.
- `voxel_downsample`: grid-hash approach — fastest for large clouds.
- `statistical_outlier_removal`: KDTree neighbor distances, z-score threshold.
- `radius_outlier_removal`: KDTree radius query, count threshold.
- `crop`: axis-aligned bounding box mask.

### `occulus/normals/`
- `estimate_normals`: PCA on KDTree neighborhoods; eigenvector corresponding to smallest eigenvalue.
- `orient_normals_to_viewpoint`: flip normals toward a given 3D point (scanner position).

### `occulus/registration/`
- `icp.py`: iterative closest point — point-to-point (SVD) and point-to-plane (linear system).
  Returns `RegistrationResult(transformation, fitness, inlier_rmse, converged, n_iterations)`.
- `global_registration.py`: FPFH features (Fast Point Feature Histograms) + RANSAC correspondence.
  Used to produce a good initial transformation before ICP refinement.

### `occulus/segmentation/`
- `ground.py`: CSF (Cloth Simulation Filter) and PMF (Progressive Morphological Filter).
  Both return updated cloud with `classification` array set (ASPRS class 2 = ground).
- `objects.py`: DBSCAN clustering (scipy); tree segmentation via CHM-based marker watershed.

### `occulus/mesh/`
- All methods require `cloud.has_normals == True`; raise `OcculusMeshError` otherwise.
- Delegates to Open3D (lazy import): Poisson, BPA, Alpha Shape.
- Returns `open3d.geometry.TriangleMesh`.

### `occulus/features/`
- `detect_planes`: iterative RANSAC — finds dominant planes, returns `(plane_model, inlier_mask)` list.
- `detect_cylinders`: RANSAC cylinder fitting.
- `compute_geometric_features`: eigenvalue decomposition on local neighborhoods → linearity,
  planarity, sphericity, omnivariance, anisotropy, eigenentropy, curvature.

### `occulus/metrics/`
- `point_density`: 2D histogram → density raster (pts/m²) as NumPy array + geotransform tuple.
- `canopy_height_model`: max-Z raster (aerial/UAV only, raises `UnsupportedPlatformError` for TLS).
- `coverage_statistics`: dict of mean/min/max/std density and gap fraction.

### `occulus/viz/`
- Lazy-import `open3d`. All functions are no-ops if Open3D unavailable.
- `visualize`: Open3D draw window or headless off-screen render.
- `visualize_registration`: side-by-side source/target with transformation applied.
- `visualize_segments`: colorize cloud by integer label array.

---

## Interface Contracts

### Public API

```python
def read(
    path: str | Path,
    *,
    platform: Platform | str = Platform.UNKNOWN,
    subsample: float | None = None,
) -> PointCloud: ...

def write(
    cloud: PointCloud,
    path: str | Path,
    *,
    compress: bool | None = None,
) -> Path: ...

def estimate_normals(
    cloud: PointCloud,
    *,
    radius: float = 0.1,
    max_nn: int = 30,
) -> PointCloud: ...

def icp(
    source: PointCloud,
    target: PointCloud,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    method: str = "point_to_point",
    initial_transform: NDArray[np.float64] | None = None,
) -> RegistrationResult: ...
```

---

## Error Handling Patterns

```python
# Pattern: wrap external calls, reraise as domain error
try:
    result = external_lib.do_thing(...)
except external_lib.SomeError as exc:
    raise OcculusIOError(f"Failed to read {path}: {exc}") from exc
```

```python
# Pattern: validate inputs early, fail fast
def crop(cloud: PointCloud, bbox: tuple[float, ...]) -> PointCloud:
    if len(bbox) != 6:
        raise OcculusValidationError(f"bbox must have 6 elements, got {len(bbox)}")
    ...
```

---

## Async Strategy

Synchronous API throughout. No `asyncio`. Network operations (future STAC integration) will
use `httpx` in sync mode or be wrapped in `async_*` variants if needed.

---

## Testing Patterns

```python
# Unit test — use tmp_path and synthetic arrays; never real files or network
def test_voxel_downsample(tmp_path):
    rng = np.random.default_rng(42)
    cloud = PointCloud(rng.standard_normal((10_000, 3)))
    result = voxel_downsample(cloud, voxel_size=0.1)
    assert result.n_points < cloud.n_points

# Integration test — real files, opt-in
@pytest.mark.integration
def test_read_las_live(real_las_file):
    cloud = read(real_las_file)
    assert cloud.n_points > 0
```

---

## Configuration

- Config via `occulus/config.py`
- No hardcoded URLs or credentials anywhere else
- Environment variables: `OCCULUS_LOG_LEVEL`, `OCCULUS_MAX_THREADS`, `OCCULUS_VOXEL_SIZE_DEFAULT`
