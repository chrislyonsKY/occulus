"""GeoTIFF export for rasterized elevation models.

Provides :func:`export_geotiff` to write :class:`~occulus.raster.dem.RasterResult`
objects to Cloud Optimized GeoTIFF (COG) files. Requires ``rasterio`` as an
optional dependency — it is lazy-imported at call time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from occulus.exceptions import OcculusExportError, OcculusRasterError
from occulus.raster.dem import RasterResult

logger = logging.getLogger(__name__)

__all__ = [
    "export_geotiff",
]


def export_geotiff(
    raster: RasterResult,
    path: str | Path,
    *,
    crs: str = "",
) -> Path:
    """Write a RasterResult to a GeoTIFF file.

    Uses ``rasterio`` to produce a single-band Float64 GeoTIFF with the
    appropriate affine transform derived from the raster's edge coordinates
    and resolution. The CRS can be overridden via the ``crs`` parameter;
    otherwise the raster's own CRS is used.

    Parameters
    ----------
    raster : RasterResult
        Raster data to export.
    path : str or Path
        Output file path. Parent directory must exist.
    crs : str, optional
        CRS override as an EPSG string (e.g. ``"EPSG:4326"``) or WKT.
        If empty, uses ``raster.crs``. If both are empty, no CRS is
        written to the file.

    Returns
    -------
    Path
        Absolute path to the written GeoTIFF file.

    Raises
    ------
    OcculusRasterError
        If the raster data is invalid.
    OcculusExportError
        If the file cannot be written (I/O error, rasterio failure).
    ImportError
        If ``rasterio`` is not installed.
    """
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.transform import from_bounds  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("rasterio is required for GeoTIFF export: pip install rasterio") from exc

    out_path = Path(path).resolve()

    if raster.data.ndim != 2:
        raise OcculusRasterError(f"Raster data must be 2D, got {raster.data.ndim}D")

    ny, nx = raster.data.shape
    if nx == 0 or ny == 0:
        raise OcculusRasterError("Cannot export an empty raster (zero cells)")

    # Determine CRS to use
    effective_crs = crs or raster.crs or None

    # Compute affine transform from edge coordinates
    x_min = float(raster.x_edges[0])
    x_max = float(raster.x_edges[-1])
    y_min = float(raster.y_edges[0])
    y_max = float(raster.y_edges[-1])

    transform = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

    try:
        if not out_path.parent.exists():
            raise OcculusExportError(f"Parent directory does not exist: {out_path.parent}")

        profile = {
            "driver": "GTiff",
            "dtype": "float64",
            "width": nx,
            "height": ny,
            "count": 1,
            "nodata": raster.nodata,
            "transform": transform,
        }
        if effective_crs:
            profile["crs"] = effective_crs

        with rasterio.open(out_path, "w", **profile) as dst:
            # Rasterio expects (row 0 = north), so flip if y_edges are ascending
            if raster.y_edges[0] < raster.y_edges[-1]:
                dst.write(np.flipud(raster.data).astype(np.float64), 1)
            else:
                dst.write(raster.data.astype(np.float64), 1)

    except OcculusExportError:
        raise
    except Exception as exc:
        raise OcculusExportError(f"Failed to write GeoTIFF to {out_path}: {exc}") from exc

    logger.info(
        "export_geotiff: wrote %dx%d raster to %s (crs=%s)",
        nx,
        ny,
        out_path,
        effective_crs or "none",
    )
    return out_path
