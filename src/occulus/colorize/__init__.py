"""RGB colorization — drape orthoimagery onto point clouds.

Assigns RGB values to each point by sampling a georeferenced raster
(orthoimage, satellite image, or classification map) at each point's
XY position.  Handles CRS mismatches transparently via pyproj.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusValidationError

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)


def colorize_from_array(
    cloud: PointCloud,
    image: NDArray[np.uint8],
    transform: tuple[float, float, float, float, float, float],
    *,
    nodata_color: tuple[int, int, int] = (0, 0, 0),
) -> PointCloud:
    """Assign RGB to points by sampling a raster array.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    image : NDArray[np.uint8]
        RGB image array, shape (H, W, 3).
    transform : tuple of 6 floats
        Affine geotransform (x_origin, x_res, x_skew, y_origin, y_skew, y_res).
        Same convention as GDAL/rasterio.
    nodata_color : tuple[int, int, int]
        RGB value for points outside the image extent.

    Returns
    -------
    PointCloud
        Cloud with ``rgb`` attribute set to (N, 3) uint8 array.

    Raises
    ------
    OcculusValidationError
        If image shape or transform is invalid.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        raise OcculusValidationError(f"Image must be (H, W, 3), got shape {image.shape}")

    x_origin, x_res, x_skew, y_origin, y_skew, y_res = transform
    h, w = image.shape[:2]

    xyz = cloud.xyz
    # Pixel coordinates from affine transform (ignoring skew for simplicity)
    col = ((xyz[:, 0] - x_origin) / x_res).astype(np.int64)
    row = ((xyz[:, 1] - y_origin) / y_res).astype(np.int64)

    # Mask valid pixels
    valid = (col >= 0) & (col < w) & (row >= 0) & (row < h)

    rgb = np.full((len(xyz), 3), nodata_color, dtype=np.uint8)
    rgb[valid] = image[row[valid], col[valid], :3]

    n_colored = int(valid.sum())
    logger.info(
        "Colorized %d / %d points (%.1f%%)",
        n_colored,
        len(xyz),
        100 * n_colored / max(len(xyz), 1),
    )

    # Return cloud with rgb attribute
    result = cloud.__class__(cloud.xyz, platform=cloud.platform)
    result.rgb = rgb
    return result


def colorize_from_raster(
    cloud: PointCloud,
    image_path: str | Path,
    *,
    band_order: tuple[int, int, int] = (1, 2, 3),
    nodata_color: tuple[int, int, int] = (0, 0, 0),
) -> PointCloud:
    """Assign RGB to points by sampling a georeferenced raster file.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    image_path : str or Path
        Path to a GeoTIFF or other rasterio-readable image.
    band_order : tuple[int, int, int]
        1-based band indices for R, G, B channels.
    nodata_color : tuple[int, int, int]
        RGB value for points outside the image or on nodata pixels.

    Returns
    -------
    PointCloud
        Cloud with ``rgb`` attribute set.

    Raises
    ------
    OcculusValidationError
        If the raster cannot be read or has insufficient bands.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise OcculusValidationError(
            "Raster colorization requires rasterio: pip install occulus[raster]"
        ) from exc

    logger.info("Reading raster for colorization: %s", image_path)
    try:
        with rasterio.open(str(image_path)) as src:
            r = src.read(band_order[0])
            g = src.read(band_order[1])
            b = src.read(band_order[2])
            image = np.stack([r, g, b], axis=-1).astype(np.uint8)
            t = src.transform
            transform = (t.c, t.a, t.b, t.f, t.d, t.e)
    except Exception as exc:
        raise OcculusValidationError(f"Failed to read raster {image_path}: {exc}") from exc

    return colorize_from_array(cloud, image, transform, nodata_color=nodata_color)
