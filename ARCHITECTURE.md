# ARCHITECTURE.md — Occulus

> **Audience:** Developers, AI agents, and contributors.
> Read `CLAUDE.md` first for workflow protocol and conventions.

---

## Problem Statement

LiDAR point cloud analysis requires handling millions of 3D points across
different acquisition platforms (aerial, terrestrial, UAV) with varying
density, noise characteristics, and coordinate systems. Existing tools are
either closed-source, require heavy C/C++ toolchains, or lack a unified API
that works across platform types. Occulus provides a pure-Python analysis
library with platform-aware types and a consistent pipeline from raw point
cloud I/O through segmentation, registration, meshing, and feature extraction.

---

## Design Goals

1. **Platform-aware types** — A single type hierarchy (`PointCloud`, `AerialCloud`, `TerrestrialCloud`, `UAVCloud`) that carries acquisition metadata and enforces platform-specific constraints (e.g., CHM only for aerial/UAV).
2. **Minimal required dependencies** — Core functionality runs on NumPy + SciPy only; heavier libraries (laspy, Open3D, matplotlib) are optional and lazy-imported.
3. **Complete analysis pipeline** — I/O, filtering, normal estimation, registration (ICP + FPFH/RANSAC), ground classification (CSF/PMF), segmentation (DBSCAN/watershed), meshing (Poisson/BPA), feature extraction (eigenvalues, planes, cylinders), and canopy metrics — all in one package.

---

## Non-Goals

- Real-time streaming or online processing of point cloud data
- Full GIS/raster stack (use rasterio/GDAL for raster workflows)
- GUI application — Occulus is a library, not an interactive tool

---

## Module Map

```
src/occulus/
├── __init__.py          ← Public API surface: read, write, types
├── _version.py          ← Single source of truth for version string
├── _cpp/                ← C++ backend bindings (scaffolded, not required for v1.0)
├── config.py            ← Constants and OCCULUS_* environment variable loading
├── exceptions.py        ← Exception hierarchy rooted at OcculusError
├── types.py             ← Platform-aware PointCloud hierarchy + dataclasses
├── io/                  ← LAS/LAZ, PLY, PCD, XYZ readers and writers
│   ├── readers.py       ← read() dispatcher → format-specific readers
│   └── writers.py       ← write() dispatcher → format-specific writers
├── filters/             ← Voxel downsample, SOR, radius outlier, crop
├── normals/             ← PCA normal estimation + viewpoint orientation
├── registration/        ← Point cloud alignment
│   ├── icp.py           ← Point-to-point and point-to-plane ICP
│   └── global_registration.py  ← FPFH features + RANSAC correspondence
├── segmentation/        ← Classification and clustering
│   ├── ground.py        ← CSF and PMF ground classification
│   └── objects.py       ← DBSCAN clustering + CHM-watershed tree segmentation
├── mesh/                ← Surface reconstruction (Poisson, BPA, Alpha Shape)
├── features/            ← RANSAC plane/cylinder detection + geometric eigenfeatures
├── metrics/             ← Point density, CHM, coverage statistics
└── viz/                 ← Open3D visualization (clouds, registration, segments)
```

---

## Data Flow

```
LAS/LAZ/PLY/XYZ file
    ↓  occulus.read()
PointCloud (AerialCloud / TerrestrialCloud / UAVCloud)
    ↓  filters.voxel_downsample() / filters.statistical_outlier_removal()
Filtered PointCloud
    ↓  normals.estimate_normals()
PointCloud with normals
    ↓  segmentation.classify_ground_csf()    ← sets classification array
Classified PointCloud
    ├→ metrics.canopy_height_model()          ← 2D raster (aerial/UAV only)
    ├→ segmentation.segment_trees()           ← CHM-watershed tree labels
    ├→ features.compute_geometric_features()  ← eigenvalue-based descriptors
    ├→ registration.icp() / global_registration.fpfh_ransac()
    ├→ mesh.poisson() / mesh.ball_pivoting()  ← surface reconstruction
    └→ occulus.write()                        ← serialize back to file
```

---

## Key Interfaces

### `PointCloud` Type Hierarchy

```python
class PointCloud:
    xyz: NDArray[np.float64]       # (N, 3) contiguous
    normals: NDArray | None        # (N, 3) if estimated
    colors: NDArray | None         # (N, 3) RGB uint8
    classification: NDArray | None # (N,) ASPRS class codes

class AerialCloud(PointCloud):
    metadata: AcquisitionMetadata  # flight altitude, pulse density, etc.

class TerrestrialCloud(PointCloud):
    scan_positions: list[ScanPosition]

class UAVCloud(PointCloud):
    metadata: AcquisitionMetadata
```

### Registration Result

```python
@dataclass
class RegistrationResult:
    transformation: NDArray[np.float64]  # 4×4 rigid transform
    fitness: float                       # inlier fraction
    inlier_rmse: float
    converged: bool
    n_iterations: int
```

---

## Dependency Graph

```
occulus.io
    → laspy (optional, LAS/LAZ)
    → open3d (optional, PLY/PCD)
    → occulus.types
    → occulus.exceptions

occulus.filters
    → scipy.spatial (KDTree)
    → occulus.types

occulus.normals
    → scipy.spatial (KDTree)
    → numpy.linalg (eigh)

occulus.registration
    → scipy.spatial (KDTree)
    → occulus.normals
    → occulus.features (FPFH)

occulus.segmentation
    → scipy.ndimage (morphological ops)
    → scipy.spatial (KDTree)
    → occulus.metrics (CHM for watershed)

occulus.mesh
    → open3d (required for meshing)
    → occulus.normals

occulus.metrics
    → numpy, scipy
    → occulus.types (platform checks)

occulus.viz
    → open3d (required)
    → matplotlib (optional)
```

---

## External Dependencies

| Dependency | Version | Why |
|---|---|---|
| numpy | >=1.24 | Core array operations, linear algebra |
| scipy | >=1.11 | KDTree, morphological filters, spatial algorithms |
| pyproj | >=3.6 | CRS transformations for georeferenced clouds |
| laspy | >=2.5 | LAS/LAZ I/O (optional: `occulus[las]`) |
| open3d | >=0.17 | PLY/PCD I/O, meshing, visualization (optional: `occulus[viz]`) |
| matplotlib | >=3.8 | 2D plotting for metrics output (optional: `occulus[viz]`) |

---

## Performance Considerations

- **Voxel downsample** uses a hash-grid approach for O(N) complexity — preferred over KDTree-based methods for large clouds (>10M points).
- **KDTree construction** (scipy.spatial.cKDTree) is the bottleneck for SOR, normal estimation, and ICP. Building it once and reusing across operations is recommended.
- **CHM rasterization** and morphological filters operate on 2D grids — memory scales with (extent / resolution)², not point count.
- **Open3D meshing** (Poisson, BPA) loads the full cloud into Open3D's C++ data structures — expect 2–3× memory overhead.
- Point clouds exceeding available RAM should be tiled before processing; Occulus does not implement out-of-core streaming.

---

## Testing Strategy

- **Unit tests**: Mock all I/O. No real network calls. Synthetic arrays via `np.random.default_rng(42)`.
- **Integration tests**: Tagged `@pytest.mark.integration`. Require network access and real LAS/LAZ files.
- **Fixtures**: Live in `tests/conftest.py`. Provide sample bounding boxes, synthetic point clouds, and temporary directories.

---

## Future Work

| Milestone | Description | Status |
|---|---|---|
| v1.0.0 | Core library — types, I/O, filters, normals, registration, segmentation, meshing, features, metrics, viz, real-world examples, WCAG 2.1 AA output, CI/CD, PyPI publishing, C++ backend scaffold | In progress |
| v1.2 | STAC catalog integration for cloud-native LiDAR discovery | Planned |
| v2.0 | Out-of-core tiled processing for TB-scale datasets | Planned |
