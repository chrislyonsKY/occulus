# Geospatial Compliance Guardrails — Occulus

> Domain-specific rules for geospatial data handling, CRS correctness,
> and cloud-native geospatial format compliance.
> These rules apply to all geospatial code generated for this project.

---

## CRS Handling

- **Always document the CRS** of every spatial parameter in the function docstring
- Public API must **accept EPSG:4326** (lon/lat) as the default for bounding boxes
  - Internal reprojection is the library's responsibility, not the caller's
- Use `pyproj.Transformer` for reprojection — never the deprecated `pyproj.transform()`
- Use `always_xy=True` when constructing `Transformer` to avoid axis-order surprises
- Never silently reproject — log a `DEBUG` message when a CRS conversion occurs

```python
# ✅ CORRECT
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3089", always_xy=True)

# ❌ WRONG — deprecated API, axis order ambiguous
from pyproj import transform
result = transform(Proj("epsg:4326"), Proj("epsg:3089"), lon, lat)
```

---

## Bounding Box Convention

- Bbox parameter order is always **(west, south, east, north)** — document this explicitly
- Validate bbox before use: west < east, south < north, values within valid range
- For Kentucky-specific projects: document EPSG:3089 (Kentucky Single Zone, US feet) vs EPSG:4326

```python
def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    if west >= east:
        raise ValueError(f"west ({west}) must be less than east ({east})")
    if south >= north:
        raise ValueError(f"south ({south}) must be less than north ({north})")
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude values must be in [-180, 180]")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitude values must be in [-90, 90]")
```

---

## Cloud-Native Format Rules

### COG (Cloud Optimized GeoTIFF)
- Never read an entire COG file to answer a windowed query — use `rasterio.windows`
- Validate COG compliance with `rio cogeo validate` before shipping outputs
- Always set `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` and `CPL_VSIL_CURL_ALLOWED_EXTENSIONS` for remote reads

### GeoParquet
- Output GeoParquet must include EPSG code in metadata (`crs` field)
- Default geometry encoding: WKB (not WKT)
- Validate with `geopandas.read_parquet()` roundtrip test before shipping

### STAC
- STAC search results must be validated against the Item schema before processing
- Never assume asset keys — check `item.assets.keys()` before accessing
- Always handle `next` pagination — never assume one page is the full result

### COPC / LAZ
- Lazy-import `laspy` or `pdal` — these are optional dependencies
- Never load the entire point cloud into memory for metadata queries — use header only

---

## FGDC Metadata (Kentucky EEC context)

If this project produces feature classes or geospatial datasets for the enterprise geodatabase:
- All outputs must have FGDC-compliant metadata
- Required fields: title, abstract, purpose, contact, spatial reference, temporal extent
- Use the Cabinet's standard metadata template where available

---

## Coordinate Output Precision

- Do not round coordinates unnecessarily — preserve source precision
- For display purposes, 6 decimal places (≈ 0.1m) is sufficient
- For computations, use full float64 precision throughout
