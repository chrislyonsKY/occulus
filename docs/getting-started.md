# Getting Started

## Requirements

- Python >=3.11
- No system-level libraries required — all dependencies install from wheels

## Installation

=== "pip"

    ```bash
    pip install occulus
    ```

=== "With LAS/LAZ"

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

=== "Development"

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

---

## First Example

```python
from occulus.io import read
from occulus.segmentation import classify_ground_csf
from occulus.metrics import canopy_height_model
import numpy as np

# Read a LiDAR file (LAS, LAZ, PLY, or XYZ)
cloud = read("scan.laz", platform="aerial", subsample=0.5)
print(f"Loaded {cloud.n_points:,} points")

# Classify ground returns using Cloth Simulation Filter
classified = classify_ground_csf(cloud)

# Generate a Canopy Height Model raster
chm, x_edges, y_edges = canopy_height_model(classified, resolution=1.0)
print(f"CHM max height: {chm[~np.isnan(chm)].max():.1f} m")
```

---

## Platform-Aware Types

occulus knows where your data came from and adapts defaults accordingly:

```python
from occulus.io import read

# Each platform gets the right subtype and smart defaults
aerial = read("survey.laz", platform="aerial")      # → AerialCloud
scan   = read("bridge.laz", platform="terrestrial")  # → TerrestrialCloud
drone  = read("uav.laz", platform="uav")             # → UAVCloud
```

| Platform | Type | Typical density | Default CSF cloth |
|---|---|---|---|
| Aerial (ALS) | `AerialCloud` | 2–25 pts/m² | 2.0 m |
| Terrestrial (TLS) | `TerrestrialCloud` | 1,000–100,000 pts/m² | 0.5 m |
| UAV | `UAVCloud` | 20–500 pts/m² | 1.5 m |

---

## Next Steps

- [API Reference](api/reference.md) — full function documentation
- [Architecture](architecture.md) — how the package is structured
- [Examples](https://github.com/chrislyonsKY/occulus/tree/main/examples) — 36 runnable scripts with real-world LiDAR data

---

[API Reference :material-arrow-right:](api/reference.md){ .md-button .md-button--primary }
[View Examples :material-github:](https://github.com/chrislyonsKY/occulus/tree/main/examples){ .md-button }
