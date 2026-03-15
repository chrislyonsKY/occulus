# Command-Line Interface

Occulus provides a CLI for common point cloud operations.

```bash
pip install occulus
occulus --help
```

## Commands

| Command | Description |
|---|---|
| `occulus info <file>` | Print cloud statistics |
| `occulus classify <file> -o <out>` | Ground classification (CSF or PMF) |
| `occulus filter <file> -o <out>` | Voxel downsample, SOR, radius filter |
| `occulus convert <file> -o <out>` | Format conversion (LAS/LAZ/PLY/XYZ) |
| `occulus dem <file> -o <out>` | DEM rasterization |
| `occulus register <src> <tgt> -o <out>` | ICP registration |
| `occulus tile <file> -o <dir>` | Spatial tiling |

## Examples

```bash
# Print info about a LAS file
occulus info survey.laz

# Classify ground points
occulus classify survey.laz -o ground.laz --algorithm csf --platform aerial

# Generate a DEM at 1m resolution
occulus dem ground.laz -o terrain.npy --resolution 1.0 --method idw

# Tile a large dataset into 500m chunks
occulus tile survey.laz -o tiles/ --tile-size 500
```
