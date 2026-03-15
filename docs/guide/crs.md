# CRS & Reprojection

Reproject point clouds between coordinate reference systems using pyproj.

```python
import occulus

cloud = occulus.read("survey.laz", platform="aerial")

# Reproject to WGS84
wgs84 = cloud.reproject("EPSG:4326")

# Or with explicit source CRS
utm = cloud.reproject("EPSG:32617", source_crs="EPSG:3089")
```

The source CRS is inferred from `cloud.metadata.coordinate_system` when available.
