# ai-dev/spec.md — Occulus

> Requirements, acceptance criteria, and milestone breakdown.
> Updated as scope clarifies. Changes require a decision record if they affect architecture.

---

## Problem Statement

Geospatial teams working with LiDAR data — aerial surveys, terrestrial laser scanning, UAV
photogrammetry — lack a single Python library that is platform-aware, standards-compliant, and
production-ready without requiring a GDAL system installation or a compiled C++ toolchain.

Occulus solves this by providing a clean Python API over NumPy/SciPy for the full point cloud
processing pipeline: read → filter → register → segment → mesh → extract features → export.

---

## Target Users

- **Primary**: Geospatial professionals and GIS analysts working with LiDAR data (aerial, TLS, UAV)
  who need reproducible, scriptable processing pipelines in Python.
- **Secondary**: Researchers and developers building point cloud applications who need a
  well-typed, extensible foundation without vendor lock-in.

---

## Functional Requirements

### Must Have (v1.0.0)

- [x] Read LAS/LAZ files (laspy); XYZ/CSV (NumPy); PLY/PCD (Open3D optional)
- [x] Write LAS/LAZ, XYZ, PLY output
- [x] Platform-aware type hierarchy (AerialCloud, TerrestrialCloud, UAVCloud)
- [x] Voxel downsampling, SOR, radius, and crop filters
- [x] PCA-based normal estimation and viewpoint orientation
- [x] Point-to-point and point-to-plane ICP registration
- [x] FPFH + RANSAC global registration
- [x] CSF and PMF ground classification
- [x] DBSCAN object clustering
- [x] Poisson, BPA, and Alpha Shape surface reconstruction (Open3D, optional)
- [x] RANSAC plane and cylinder detection
- [x] Eigenvalue-based geometric feature extraction
- [x] 2D point density raster and Canopy Height Model
- [x] Open3D visualization helpers

### Should Have

- [ ] STAC catalog search for public LiDAR datasets
- [ ] Streaming/chunked reads for files > available RAM
- [ ] Multi-scan TLS registration (sequential ICP with loop closure)
- [ ] GeoParquet export with EPSG metadata

### Won't Have (v1.0.0)

- Real-time streaming point cloud processing
- GPU-accelerated algorithms (CUDA)
- Compiled C++ backend (scaffolded but not built)
- Web-based visualization

---

## Acceptance Criteria

### Read/Write Round-Trip

**Given** a PointCloud with known xyz coordinates
**When** written to XYZ then read back
**Then** coordinates match to float64 precision and point count is identical

### ICP Convergence

**Given** a source cloud and a target cloud offset by a known rigid transform
**When** ICP is run with sufficient iterations
**Then** `result.converged == True` and recovered transform is within 1mm of ground truth

### Ground Classification

**Given** an aerial point cloud with mixed ground and vegetation
**When** CSF classification is run
**Then** `classification == 2` for ground points; recall > 90% on synthetic terrain

### Voxel Downsample

**Given** a dense cloud with 100,000 points
**When** voxel_downsample is called with voxel_size = 0.1
**Then** output has fewer points and one representative per voxel cell

---

## Technical Constraints

- Python >=3.11 (Hatch-managed environment)
- Must work on macOS, Linux, Windows
- No hard dependency on GDAL system installation
- `laspy` and `open3d` are optional — core pipeline runs on NumPy + SciPy only
- All public functions pass `mypy --strict`
- All public functions have NumPy-style docstrings

---

## Milestones

| Milestone | Scope | Target |
|---|---|---|
| v0.1.0 | Type system, exception hierarchy, I/O dispatcher scaffold | 2026-01-01 |
| v1.0.0 | Full processing pipeline, all modules implemented, 80%+ test coverage | 2026-03-14 |
| v1.1.0 | STAC search, GeoParquet export | TBD |
| v2.0.0 | C++ backend, streaming reads | TBD |

---

## Open Questions

- [ ] Coordinate system storage — should PointCloud carry a `crs` string attribute? (DL candidate)
- [ ] LAS 1.4 point format 6-10 support in writers — currently only formats 0-3
