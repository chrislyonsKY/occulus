# occulus

**Multi-platform point cloud analysis — registration, segmentation, meshing, and feature extraction.**

One library for aerial, terrestrial, and UAV LiDAR. Platform-aware types, C++ performance, NumPy arrays in and out.

---

## Install

=== "pip"

    ```bash
    pip install occulus
    ```

=== "With LAS/LAZ support"

    ```bash
    pip install occulus[las]
    ```

=== "With visualization"

    ```bash
    pip install occulus[viz]
    ```

=== "Everything"

    ```bash
    pip install occulus[all]
    ```

---

## Quick Start

```python
from occulus.io import read
from occulus.segmentation import classify_ground_csf
from occulus.metrics import canopy_height_model

# Read with platform awareness
cloud = read("survey.laz", platform="aerial", subsample=0.5)

# Classify ground returns
classified = classify_ground_csf(cloud)

# Generate a Canopy Height Model raster
chm, x_edges, y_edges = canopy_height_model(classified, resolution=1.0)
```

---

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

---

## Real-World Examples

All examples use **real USGS 3DEP and international LiDAR data** with WCAG 2.1 AA compliant output.

| Region | Terrain | Script |
|---|---|---|
| Eastern Kentucky | Appalachian forest/coal | `kentucky_ground_classification.py` |
| Kentucky (statewide) | KY From Above terrain survey | `kyfromabove_terrain_survey.py` |
| Colorado Front Range | Rocky Mountain high-relief | `colorado_rocky_mountain_terrain.py` |
| Sonoran Desert, AZ | Basin and range arid terrain | `arizona_desert_terrain.py` |
| Utah Canyonlands | 300–500 m canyon walls | `utah_canyon_geology.py` |
| Oregon Coast | Sea cliffs + coastal rainforest | `oregon_coast_terrain.py` |
| Louisiana Delta | Near-zero deltaic wetlands | `louisiana_wetlands_delta.py` |
| Houston, TX | Dense urban + bayous | `houston_urban_density.py` |
| Sabzevar, Iran | Active fault geomorphology | `iran_fault_geomorphology.py` |
| Delft, Netherlands | Below-sea-level polder | `netherlands_ahn4_polder.py` |

---

## What occulus Is Not

occulus is **not** a GIS application, real-time streaming engine, or sensor control toolkit.

If you need raster processing, use [rasterio](https://rasterio.readthedocs.io/).
If you need STAC catalog access, use [pystac-client](https://pystac-client.readthedocs.io/).
If you need Kentucky-specific LiDAR data access, use [AbovePy](https://github.com/chrislyonsKY/AbovePy).

---

[Get Started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/chrislyonsKY/occulus){ .md-button }
