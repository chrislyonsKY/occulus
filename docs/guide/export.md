# Web Export (3D Tiles & Potree)

Export point clouds for browser-based 3D viewers.

## 3D Tiles (Cesium)

```python
from occulus.export import export_3dtiles

path = export_3dtiles(cloud, "output/cesium/", max_points_per_tile=50_000)
# Produces tileset.json + .pnts binary files
```

## Potree

```python
from occulus.export import export_potree

path = export_potree(cloud, "output/potree/", max_depth=10)
# Produces metadata.json + octree .bin files
```
