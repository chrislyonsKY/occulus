 # Release Notes

---

## [1.0.0] — 2026-03-14

The first production release of Occulus — a complete point cloud analysis library for aerial, terrestrial, and UAV LiDAR.

### Core Library

- **I/O** — LAS/LAZ (laspy), PLY/PCD (Open3D), XYZ (NumPy). Subsample-on-read.
- **Types** — Platform-aware hierarchy: `PointCloud`, `AerialCloud`, `TerrestrialCloud`, `UAVCloud`. Open3D interop.
- **Filters** — Voxel downsample, statistical outlier removal, radius outlier removal, axis-aligned crop.
- **Normals** — PCA normal estimation on KDTree neighborhoods, viewpoint-based orientation.
- **Registration** — Point-to-point and point-to-plane ICP, FPFH+RANSAC global registration.
- **Segmentation** — CSF and PMF ground classification, DBSCAN clustering, CHM-watershed tree segmentation.
- **Mesh** — Poisson, Ball Pivoting, Alpha Shape surface reconstruction (via Open3D).
- **Features** — Iterative RANSAC plane/cylinder detection, eigenvalue geometric features (7 descriptors).
- **Metrics** — Point density raster, Canopy Height Model, coverage statistics.
- **Visualization** — Open3D 3D viewer for clouds, registration results, and labeled segments.

### C++ Backend

- Full C++17 implementation with pybind11 bindings: k-d tree, filters, normals, ICP, CSF/PMF, RANSAC, eigenvalue features.
- Pure-Python fallbacks used when the C++ extension is not compiled.

### Examples

- 36 runnable scripts using real USGS 3DEP, KY From Above, OpenTopography, and AHN4 data.
- Terrain analysis across 10 geographic regions (KY, CO, AZ, UT, OR, LA, TX, Iran, Netherlands).
- All outputs WCAG 2.1 AA compliant with alt-text metadata and cross-section reference lines.
- Full toolkit demonstration on Appalachian coal mine terrain.

### Infrastructure

- CI: ruff lint/format, mypy strict typecheck, pytest on Python 3.11/3.12/3.13.
- PyPI publish via GitHub Actions trusted publisher.
- mkdocs-material documentation site.
- Full test suite with mocked I/O.

### Dependencies

| Package | Version | Required |
|---|---|---|
| numpy | >=1.24 | Yes |
| scipy | >=1.11 | Yes |
| pyproj | >=3.6 | Yes |
| laspy | >=2.5 | Optional (`occulus[las]`) |
| open3d | >=0.17 | Optional (`occulus[viz]`) |
| matplotlib | >=3.8 | Optional (`occulus[viz]`) |

---

## [0.1.0] — 2026-01-01

- Initial project scaffold: type system, exception hierarchy, I/O dispatcher, build infrastructure.
