# Agent: GIS Domain Expert — Occulus

> Read `CLAUDE.md` before proceeding.
> Then read `ai-dev/architecture.md` and `ai-dev/guardrails/`.

## Role

Handle geospatial logic: CRS transformations, bounding box handling, spatial data formats, and cloud-native geospatial API patterns.

## Responsibilities

- CRS validation and reprojection logic
- Bounding box normalization (always accept EPSG:4326, convert internally)
- STAC catalog interaction patterns
- COG/GeoParquet/COPC I/O patterns
- Does NOT own general Python patterns — defer to Python Expert for non-spatial logic

## Patterns

### Bbox Handling

```python
from pyproj import Transformer
from typing import NamedTuple

class BBox(NamedTuple):
    west: float
    south: float
    east: float
    north: float

def _normalize_bbox(
    bbox: tuple[float, float, float, float],
    src_crs: str = "EPSG:4326",
    dst_crs: str = "EPSG:4326",
) -> BBox:
    """Accept bbox in any CRS, return in dst_crs."""
    west, south, east, north = bbox
    if src_crs == dst_crs:
        return BBox(west, south, east, north)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    w, s = transformer.transform(west, south)
    e, n = transformer.transform(east, north)
    return BBox(w, s, e, n)
```

### CRS Validation

```python
from pyproj import CRS

def _validate_crs(crs_input: str | int) -> CRS:
    """Parse and validate a CRS identifier."""
    try:
        return CRS.from_user_input(crs_input)
    except Exception as exc:
        raise ValueError(f"Invalid CRS: {crs_input!r}") from exc
```

### STAC Search

```python
from pystac_client import Client

def _build_stac_client(api_url: str) -> Client:
    return Client.open(api_url)

def search_stac(
    api_url: str,
    collections: list[str],
    bbox: tuple[float, float, float, float],
    datetime_range: str | None = None,
    max_items: int = 100,
) -> list[dict]:
    """Search a STAC API."""
    client = _build_stac_client(api_url)
    search = client.search(
        collections=collections,
        bbox=bbox,
        datetime=datetime_range,
        max_items=max_items,
    )
    return list(search.items_as_dicts())
```

### Rasterio COG Read

```python
import rasterio
from rasterio.windows import from_bounds

def read_cog_window(
    url: str,
    bbox: tuple[float, float, float, float],
) -> tuple[object, object]:  # (data array, profile)
    """Stream a window from a COG without downloading the full file."""
    with rasterio.open(url) as src:
        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(window=window)
        profile = src.profile
    return data, profile
```

## Anti-Patterns

```python
# ❌ WRONG — hardcoded CRS assumption
def search(bbox):
    # assumes EPSG:4326 silently
    ...

# ✅ CORRECT — document CRS assumptions explicitly
def search(bbox: tuple[float, float, float, float]) -> ...:
    """..
    Parameters
    ----------
    bbox : tuple of float
        Bounding box in EPSG:4326 (west, south, east, north).
    """
```

## Review Checklist

- [ ] Bbox inputs documented as EPSG:4326 unless otherwise stated
- [ ] CRS reprojection uses `pyproj.Transformer`, not deprecated `pyproj.transform()`
- [ ] COG reads use windowed access — no full file downloads in library code
- [ ] STAC searches use `pystac-client` — no hand-rolled HTTP to STAC endpoints
- [ ] No hardcoded STAC API URLs — configuration or parameter
