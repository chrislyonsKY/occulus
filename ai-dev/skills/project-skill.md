---
name: occulus-skill
description: "Project-specific patterns, conventions, and domain knowledge for Occulus. Read before writing any code for this package."
---

# Occulus — Project Skill

> Read `CLAUDE.md` and `ai-dev/architecture.md` before this file.

---

## Package Layout

```
src/occulus/
├── __init__.py          ← Public API
├── _version.py          ← Single version source (1.0.0)
├── exceptions.py        ← OcculusError hierarchy
├── config.py            ← Constants, env var loading
├── types.py             ← PointCloud, AerialCloud, TerrestrialCloud, UAVCloud
├── io/
│   ├── __init__.py      ← re-exports read, write
│   ├── readers.py       ← read() dispatcher + format readers
│   └── writers.py       ← write() dispatcher + format writers
├── filters/
│   └── __init__.py      ← voxel_downsample, SOR, radius, crop
├── normals/
│   └── __init__.py      ← estimate_normals, orient_normals_to_viewpoint
├── registration/
│   ├── __init__.py      ← re-exports icp, ransac_registration, RegistrationResult
│   ├── icp.py           ← Point-to-point and point-to-plane ICP
│   └── global_registration.py ← FPFH + RANSAC
├── segmentation/
│   ├── __init__.py      ← re-exports classify_ground_csf, cluster_dbscan, etc.
│   ├── ground.py        ← CSF, PMF ground classification
│   └── objects.py       ← DBSCAN, tree segmentation
├── mesh/
│   └── __init__.py      ← poisson_mesh, ball_pivoting_mesh, alpha_shape_mesh
├── features/
│   └── __init__.py      ← detect_planes, detect_cylinders, compute_geometric_features
├── metrics/
│   └── __init__.py      ← point_density, canopy_height_model, coverage_statistics
├── viz/
│   └── __init__.py      ← visualize, visualize_registration, visualize_segments
└── _cpp/
    └── __init__.py      ← pybind11 bindings (optional, not built in v1.0.0)
```

---

## Public API Surface

```python
from occulus import read, write, PointCloud, AerialCloud, TerrestrialCloud, UAVCloud
from occulus import Platform, ScanPosition, AcquisitionMetadata

from occulus.filters import voxel_downsample, statistical_outlier_removal
from occulus.normals import estimate_normals, orient_normals_to_viewpoint
from occulus.registration import icp, ransac_registration, RegistrationResult
from occulus.segmentation import classify_ground_csf, classify_ground_pmf, cluster_dbscan
from occulus.mesh import poisson_mesh, ball_pivoting_mesh, alpha_shape_mesh
from occulus.features import detect_planes, compute_geometric_features
from occulus.metrics import point_density, canopy_height_model, coverage_statistics
from occulus.viz import visualize, visualize_registration, visualize_segments
```

---

## Domain Vocabulary

| Term | Definition |
|---|---|
| `LAS / LAZ` | ASPRS standard binary format for point clouds. LAZ is compressed LAS. |
| `TLS` | Terrestrial Laser Scanning — ground-based, stationary scanner. High density (1000s pts/m²). |
| `ALS` | Aerial Laser Scanning — airborne, nadir-looking. Lower density (10-50 pts/m²) but wide area. |
| `UAV LiDAR` | Drone-mounted scanner. Oblique angles, variable density, often oblique. |
| `SfM` | Structure from Motion — photogrammetric point cloud from overlapping photos. |
| `ICP` | Iterative Closest Point — local registration algorithm. Needs good initial alignment. |
| `FPFH` | Fast Point Feature Histograms — 33D descriptor for global registration. |
| `RANSAC` | Random Sample Consensus — robust estimator for models with outliers. |
| `CSF` | Cloth Simulation Filter — drapes virtual cloth on inverted cloud to find ground. |
| `PMF` | Progressive Morphological Filter — morphological opening with growing window. |
| `CHM` | Canopy Height Model — raster of max vegetation height (ALS minus ground). |
| `ASPRS class 2` | Ground points in LAS classification standard. |
| `EPSG:3089` | Kentucky Single Zone (US feet) — common for KY state data. |
| `EPSG:4326` | WGS84 lat/lon — public API default for bbox parameters. |

---

## Data Sources

| Source | Format | Auth |
|---|---|---|
| Kentucky 5cm LiDAR (KyFromAbove) | LAS/LAZ tiles | None (public) |
| USGS 3DEP (via STAC) | COG / LAZ | None (public) |
| OpenTopography | LAZ | API key (env var `OCCULUS_OPENTOPO_KEY`) |

---

## Known Gotchas

- **LAS scale/offset**: `laspy` returns raw integers; multiply by scale and add offset to get
  real-world coordinates. Always use `las.x`, `las.y`, `las.z` (not `las.X`, `las.Y`, `las.Z`).
- **Open3D axis order**: `o3d.utility.Vector3dVector` expects (N, 3) float64. Copy arrays with
  `np.ascontiguousarray` before passing.
- **ICP needs initial alignment**: Point-to-point ICP diverges if clouds are far apart. Always
  run RANSAC global registration first for unknown initial positions.
- **CSF inverts Z**: The cloth simulation works on an inverted (upside-down) point cloud.
  Re-invert before returning results.
- **Normals for meshing**: Poisson and BPA both require consistent outward normals. Run
  `orient_normals_to_viewpoint` before meshing TLS data.
- **EPSG:3089 is US feet**: Distances in Kentucky state plane are in US survey feet, not meters.
  Convert with `* 0.3048006096` for meter-based algorithms.
- **Empty classification array**: Many LAS files have a classification array of all zeros.
  Do not assume ground points exist; always check `np.any(cloud.classification == 2)`.

---

## Code Patterns Specific to This Project

### Lazy optional dependency import

```python
def poisson_mesh(cloud: PointCloud, depth: int = 8) -> object:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for mesh reconstruction: pip install occulus[viz]"
        ) from exc
    ...
```

### Platform guard

```python
from occulus.exceptions import UnsupportedPlatformError
from occulus.types import Platform

def canopy_height_model(cloud: PointCloud, resolution: float = 1.0) -> ...:
    if cloud.platform == Platform.TERRESTRIAL:
        raise UnsupportedPlatformError(
            "canopy_height_model requires aerial or UAV data, not terrestrial"
        )
    ...
```

### Subsampling helper

```python
def _subsample(xyz: NDArray[np.float64], fraction: float) -> NDArray[np.bool_]:
    """Return a boolean mask selecting `fraction` of points uniformly at random."""
    rng = np.random.default_rng()
    mask = np.zeros(len(xyz), dtype=bool)
    idx = rng.choice(len(xyz), size=int(len(xyz) * fraction), replace=False)
    mask[idx] = True
    return mask
```
