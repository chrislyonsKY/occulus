# Getting Started

## Requirements

- Python >=3.11
- No system-level libraries required — all dependencies install from wheels

## Installation

### Basic install

```bash
pip install occulus
```

### With optional extras

```bash
pip install occulus[las]   # LAS/LAZ I/O via laspy + lazrs
pip install occulus[viz]   # Open3D visualization + matplotlib plotting
pip install occulus[all]   # All optional dependencies
```

### Development install

```bash
git clone https://github.com/chrislyonsKY/occulus.git
cd occulus
pip install -e ".[dev]"
```

## Verify the Install

```python
import occulus
print(occulus.__version__)
```

## First Example

```python
from occulus.io import read
from occulus.segmentation import classify_ground_csf
from occulus.metrics import canopy_height_model

# Read a LiDAR file (LAS, LAZ, PLY, or XYZ)
cloud = read("scan.laz", platform="aerial", subsample=0.5)
print(f"Loaded {cloud.n_points:,} points")

# Classify ground returns using Cloth Simulation Filter
classified = classify_ground_csf(cloud)

# Generate a Canopy Height Model raster
chm, x_edges, y_edges = canopy_height_model(classified, resolution=1.0)
print(f"CHM max height: {chm[~np.isnan(chm)].max():.1f} m")
```

## Next Steps

- [API Reference](api/reference.md) — full function documentation
- [Architecture](architecture.md) — how the package is structured
- [Examples](https://github.com/chrislyonsKY/occulus/tree/main/examples) — runnable scripts with real-world LiDAR data
