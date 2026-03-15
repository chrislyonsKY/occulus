"""DEM / DSM / DTM rasterization and export.

Create regular-grid elevation models from point clouds and export
them as GeoTIFF files.

Functions
---------
- :func:`create_dsm` — Digital Surface Model (max-Z, all points)
- :func:`create_dtm` — Digital Terrain Model (ground-only)
- :func:`create_dem` — Alias for :func:`create_dtm`
- :func:`idw_interpolate` — Inverse Distance Weighting interpolation
- :func:`nearest_interpolate` — Nearest-neighbour interpolation
- :func:`export_geotiff` — Write raster to GeoTIFF (requires rasterio)

Types
-----
- :class:`RasterResult` — Container for rasterized elevation data
"""

from occulus.raster.dem import RasterResult, create_dem, create_dsm, create_dtm
from occulus.raster.export import export_geotiff
from occulus.raster.interpolation import idw_interpolate, nearest_interpolate

__all__ = [
    "RasterResult",
    "create_dem",
    "create_dsm",
    "create_dtm",
    "export_geotiff",
    "idw_interpolate",
    "nearest_interpolate",
]
