# Changelog

All notable changes to Occulus will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2026-03-14

### Added

- **I/O**: Full LAS/LAZ reader and writer via `laspy`; PLY and PCD via Open3D (optional); XYZ/CSV/TXT via NumPy
- **Types**: Platform-aware `PointCloud` hierarchy — `AerialCloud`, `TerrestrialCloud`, `UAVCloud`; Open3D interop (`to_open3d`, `from_open3d`); `AerialCloud.ground_points()`, `AerialCloud.first_returns()`, `TerrestrialCloud.viewpoint_mask()`
- **Filters**: `voxel_downsample`, `statistical_outlier_removal`, `radius_outlier_removal`, `crop`
- **Normals**: `estimate_normals` (PCA-based), `orient_normals_to_viewpoint`
- **Registration**: Point-to-point and point-to-plane ICP; FPFH feature computation; RANSAC global registration; `RegistrationResult` dataclass
- **Segmentation**: CSF ground classification, Progressive Morphological Filter (PMF), DBSCAN clustering, tree segmentation via CHM watershed
- **Mesh**: Poisson surface reconstruction, Ball Pivoting Algorithm, Alpha Shape — all via Open3D (optional)
- **Features**: RANSAC plane detection, cylinder fitting, eigenvalue-based geometric features (linearity, planarity, sphericity)
- **Metrics**: 2D point density raster, Canopy Height Model, coverage statistics
- **Visualization**: Open3D window rendering, registration overlay, per-segment colorization
- **Exceptions**: Complete hierarchy including `OcculusNetworkError`
- **Config**: `occulus.config` constants; environment variable overrides via `OCCULUS_*`
- **Dependencies**: `scipy>=1.11` and `pyproj>=3.6` added as core dependencies

### Changed

- Version bumped from 0.1.0 to 1.0.0
- Classifier updated to `Production/Stable`

---

## [0.1.0] — 2026-01-01

### Added

- Initial project scaffold: type system, exception hierarchy, I/O dispatcher, build infrastructure
