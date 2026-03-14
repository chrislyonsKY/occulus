# Occulus

**Multi-platform point cloud analysis — registration, segmentation, meshing, and feature extraction**


![Occulus banner](assets/banner.png)

Occulus is a pure-Python library for analysing aerial (ALS), terrestrial (TLS), and UAV LiDAR point clouds. It provides platform-aware types, format I/O, filtering, normal estimation, ICP and FPFH+RANSAC registration, CSF/PMF ground classification, DBSCAN/CHM-watershed segmentation, Poisson/BPA surface meshing, eigenvalue feature extraction, canopy metrics, and Open3D visualisation — with NumPy and SciPy as the only required runtime dependencies.

## Install

```bash
pip install occulus
```

Optional extras:

```bash
pip install occulus[las]   # LAS/LAZ I/O via laspy
pip install occulus[viz]   # Open3D visualization + matplotlib
pip install occulus[all]   # Everything
```

## Quick Start

```python
from occulus.io import read
from occulus.segmentation import classify_ground_csf
from occulus.metrics import canopy_height_model

cloud = read("scan.laz", platform="aerial", subsample=0.5)
classified = classify_ground_csf(cloud)
chm, xe, ye = canopy_height_model(classified, resolution=1.0)
```

## What's Inside

| Module | What it does |
|---|---|
| `occulus.io` | Read/write LAS, LAZ, PLY, PCD, and XYZ point clouds |
| `occulus.types` | Platform-aware `PointCloud`, `AerialCloud`, `TerrestrialCloud`, `UAVCloud` |
| `occulus.filters` | Voxel downsample, statistical/radius outlier removal, crop |
| `occulus.normals` | PCA normal estimation and viewpoint orientation |
| `occulus.registration` | ICP (point-to-point, point-to-plane) and FPFH+RANSAC global registration |
| `occulus.segmentation` | CSF/PMF ground classification, DBSCAN clustering, CHM-watershed trees |
| `occulus.mesh` | Poisson, Ball Pivoting, and Alpha Shape surface reconstruction |
| `occulus.features` | RANSAC plane/cylinder detection, eigenvalue geometric features |
| `occulus.metrics` | Point density, canopy height model, coverage statistics |
| `occulus.viz` | Open3D 3D visualization for clouds, registration results, and segments |

## License

GPL-3.0 — see [LICENSE](https://github.com/chrislyonsKY/occulus/blob/main/LICENSE).
