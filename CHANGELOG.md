# Changelog

All notable changes to Occulus will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-03-15

### Added

- **CLI** — `occulus` command with subcommands: `info`, `classify`, `filter`, `convert`, `dem`, `register`, `tile`. Entry point via `[project.scripts]`.
- **DEM/DSM/DTM rasterization** (`occulus.raster`) — IDW and nearest-neighbor interpolation, GeoTIFF export via rasterio (optional). `create_dsm`, `create_dtm`, `create_dem`, `RasterResult` dataclass.
- **M3C2 change detection** (`occulus.change`) — Multi-epoch signed distance computation with Level of Detection (Lague et al. 2013). `m3c2()` function, `M3C2Result` dataclass.
- **Volume computation** (`occulus.analysis.volume`) — Cut/fill analysis between two surfaces or a surface and reference elevation. `compute_volume()`, `VolumeResult` dataclass.
- **Cross-section extraction** (`occulus.analysis.cross_section`) — Profile generation along polylines. `extract_cross_section()`, `extract_profiles()`, `CrossSection` dataclass.
- **CRS transforms** (`occulus.crs`) — `reproject()` and `transform_coordinates()` via pyproj. `PointCloud.reproject()` convenience method added to types.
- **Tiling** (`occulus.tiling`) — Spatial grid tiling for large datasets. `tile_point_cloud()`, `iter_tiles()`, `process_tiles()`, `Tile` dataclass.
- **RGB colorization** (`occulus.colorize`) — Drape georeferenced orthoimagery onto point clouds. `colorize_from_raster()`, `colorize_from_array()`.
- **3D Tiles / Potree export** (`occulus.export`) — `export_3dtiles()` writes `.pnts` binary + `tileset.json`. `export_potree()` writes octree hierarchy + `metadata.json`.
- **ML semantic segmentation** (`occulus.ml`) — Inference wrappers for ONNX Runtime and PyTorch backends. `predict_semantic()`, `prepare_features()`, `SegmentationPrediction` dataclass.
- **COPC streaming** (`occulus.io.copc`) — Read Cloud Optimized Point Cloud files with spatial bbox filtering. `read_copc()`, `read_copc_metadata()`, `COPCMetadata` dataclass.
- **Powerline detection** (`occulus.segmentation.powerlines`) — Wire and pylon detection via linearity/verticality features, DBSCAN clustering, optional catenary fitting and clearance analysis. `detect_powerlines()`, `PowerlineResult` dataclass.
- **Exceptions** — `OcculusCRSError`, `OcculusChangeDetectionError`, `OcculusRasterError`, `OcculusExportError`, `OcculusMLError`.

### New optional dependencies

- `raster` — `rasterio>=1.3`
- `copc` — `httpx>=0.27`
- `web` — `py3dtiles>=7.0`
- `ml` — `onnxruntime>=1.16`
- `ml-torch` — `torch>=2.0`, `onnxruntime>=1.16`

---

## [1.0.0] — 2026-03-14

### Added

- **I/O** — Full LAS/LAZ reader and writer via `laspy`; PLY and PCD via Open3D (optional); XYZ/CSV/TXT via NumPy. Subsample-on-read for large datasets.
- **Types** — Platform-aware `PointCloud` hierarchy: `AerialCloud`, `TerrestrialCloud`, `UAVCloud`. Open3D interop via `to_open3d()` / `from_open3d()`. Platform-specific helpers: `ground_points()`, `first_returns()`, `viewpoint_mask()`.
- **Filters** — `voxel_downsample` (hash-grid O(N)), `statistical_outlier_removal`, `radius_outlier_removal`, `crop` (AABB).
- **Normals** — `estimate_normals` (PCA on KDTree neighborhoods), `orient_normals_to_viewpoint`.
- **Registration** — Point-to-point and point-to-plane ICP; FPFH feature computation; RANSAC global registration; `RegistrationResult` dataclass.
- **Segmentation** — CSF ground classification (Cloth Simulation Filter), PMF (Progressive Morphological Filter), DBSCAN clustering, CHM-watershed individual tree segmentation.
- **Mesh** — Poisson surface reconstruction, Ball Pivoting Algorithm, Alpha Shape — all via Open3D (optional).
- **Features** — Iterative RANSAC plane detection, cylinder fitting, eigenvalue-based geometric features (linearity, planarity, sphericity, omnivariance, anisotropy, eigenentropy, curvature).
- **Metrics** — 2D point density raster, Canopy Height Model (aerial/UAV only), coverage statistics (gap fraction, mean density).
- **Visualization** — Open3D window rendering, registration overlay, per-segment colorization.
- **Exceptions** — Complete hierarchy: `OcculusError`, `OcculusIOError`, `OcculusValidationError`, `OcculusRegistrationError`, `OcculusSegmentationError`, `OcculusMeshError`, `OcculusFeatureError`, `OcculusCppError`, `OcculusNetworkError`, `UnsupportedPlatformError`.
- **Config** — `occulus.config` constants; environment variable overrides via `OCCULUS_*`.
- **C++ backend** — Full C++17 source with pybind11 bindings: k-d tree, voxel downsample, SOR, PCA normals, point-to-point ICP, CSF/PMF ground classification, RANSAC plane detection, eigenvalue geometric features. Pure-Python fallbacks used when C++ extension is not compiled.
- **Examples** — 36 runnable scripts covering terrain analysis (KY, CO, AZ, UT, OR, LA, TX, Iran, Netherlands), forest inventory, ground classification, registration, segmentation, meshing, and more. All outputs WCAG 2.1 AA compliant with alt-text metadata and cross-section reference lines.
- **CI/CD** — GitHub Actions: ruff lint + format, mypy typecheck, pytest on Python 3.11/3.12/3.13, PyPI publish via trusted publisher.
- **Documentation** — mkdocs-material site with tabbed install, API reference via mkdocstrings, architecture overview.
- **Tests** — Full test suite: types, I/O, filters, normals, registration, segmentation, features, metrics, config, exceptions. Mocked I/O for unit tests; `@pytest.mark.integration` for real-file tests.

### Dependencies

- `numpy>=1.24` (required)
- `scipy>=1.11` (required)
- `pyproj>=3.6` (required)
- `laspy[lazrs]>=2.5` (optional: `occulus[las]`)
- `open3d>=0.17` (optional: `occulus[viz]`)
- `matplotlib>=3.8` (optional: `occulus[viz]`)

---

## [0.1.0] — 2026-01-01

### Added

- Initial project scaffold: type system, exception hierarchy, I/O dispatcher, build infrastructure.
