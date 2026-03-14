# Architecture

!!! note
    This is the human-facing architecture overview.
    The AI-facing version with implementation detail is in `ai-dev/architecture.md`.

## Problem Statement

LiDAR point cloud analysis requires handling millions of 3D points across
different acquisition platforms (aerial, terrestrial, UAV) with varying
density, noise, and coordinate systems. Occulus provides a unified pure-Python
library with platform-aware types and a consistent pipeline from I/O through
segmentation, registration, meshing, and feature extraction.

## Design Goals

1. **Platform-aware types** — `PointCloud` hierarchy carries acquisition metadata and enforces platform constraints.
2. **Minimal required dependencies** — Core runs on NumPy + SciPy; heavier libraries (laspy, Open3D) are optional.
3. **Complete analysis pipeline** — I/O, filtering, normals, registration, segmentation, meshing, features, and metrics in one package.

## Module Map

```
src/occulus/
├── __init__.py       ← Public API surface (read, write, types)
├── config.py         ← Constants and environment config
├── exceptions.py     ← Exception hierarchy (OcculusError base)
├── types.py          ← PointCloud, AerialCloud, TerrestrialCloud, UAVCloud
├── io/               ← LAS/LAZ, PLY, PCD, XYZ readers and writers
├── filters/          ← Voxel downsample, SOR, radius outlier, crop
├── normals/          ← PCA normal estimation + viewpoint orientation
├── registration/     ← ICP + FPFH/RANSAC global registration
├── segmentation/     ← CSF/PMF ground classification, DBSCAN, watershed
├── mesh/             ← Poisson, BPA, Alpha Shape surface reconstruction
├── features/         ← RANSAC plane/cylinder + eigenvalue features
├── metrics/          ← Point density, CHM, coverage statistics
└── viz/              ← Open3D 3D visualization
```

## Data Flow

```
LAS/LAZ/PLY/XYZ file
    ↓  read()
PointCloud
    ↓  filters (downsample, outlier removal)
Filtered PointCloud
    ↓  estimate_normals()
PointCloud with normals
    ↓  classify_ground_csf() / classify_ground_pmf()
Classified PointCloud
    ├→ canopy_height_model()     ← 2D raster
    ├→ segment_trees()           ← individual tree labels
    ├→ icp() / fpfh_ransac()    ← registration
    ├→ poisson() / bpa()        ← surface mesh
    └→ write()                  ← serialize to file
```

## Dependencies

| Dependency | Why |
|---|---|
| numpy >=1.24 | Core array operations and linear algebra |
| scipy >=1.11 | KDTree, morphological filters, spatial algorithms |
| pyproj >=3.6 | CRS transformations for georeferenced clouds |
| laspy >=2.5 | LAS/LAZ I/O (optional) |
| open3d >=0.17 | PLY/PCD I/O, meshing, visualization (optional) |

## Key Decisions

See [`ai-dev/decisions/`](https://github.com/chrislyonsKY/occulus/tree/main/ai-dev/decisions)
for the full decision log.
