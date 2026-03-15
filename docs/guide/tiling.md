# Tiling Large Datasets

For datasets that exceed available RAM, use the tiling module to split
into spatial chunks and process independently.

```python
from occulus.tiling import tile_point_cloud, process_tiles

# Split into 500m tiles
tiles = tile_point_cloud("huge_survey.laz", "tiles/", tile_size=500)

# Process each tile
from occulus.segmentation import classify_ground_csf
results = process_tiles(tiles, classify_ground_csf, output_dir="classified/")
```

Or iterate lazily:

```python
from occulus.tiling import iter_tiles

for tile, cloud in iter_tiles("huge_survey.laz", tile_size=500):
    print(f"Tile {tile.index}: {tile.point_count} points")
```
