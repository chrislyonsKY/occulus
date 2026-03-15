"""Spatial interpolation methods for point cloud rasterization.

Provides IDW (Inverse Distance Weighting) and nearest-neighbor interpolation
on irregular point sets to regular grids. Both methods use ``scipy.spatial.KDTree``
for efficient neighbor lookup.

Functions
---------
- :func:`idw_interpolate` — IDW interpolation with configurable power and search radius
- :func:`nearest_interpolate` — nearest-neighbor assignment to grid cells
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusRasterError

logger = logging.getLogger(__name__)

__all__ = [
    "idw_interpolate",
    "nearest_interpolate",
]


def idw_interpolate(
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    *,
    power: float = 2.0,
    max_dist: float | None = None,
    k: int = 12,
    nodata: float = -9999.0,
) -> NDArray[np.float64]:
    """Inverse Distance Weighting interpolation onto a regular grid.

    For each grid cell centre, the ``k`` nearest source points are found via
    KDTree and their Z values are combined using inverse-distance weights
    raised to ``power``.

    Parameters
    ----------
    xy : NDArray[np.float64]
        Source point XY coordinates, shape (N, 2).
    z : NDArray[np.float64]
        Source point Z values, shape (N,).
    grid_x : NDArray[np.float64]
        Grid cell centre X coordinates, shape (nx,).
    grid_y : NDArray[np.float64]
        Grid cell centre Y coordinates, shape (ny,).
    power : float, optional
        Distance weighting exponent, by default 2.0.
    max_dist : float or None, optional
        Maximum search distance. Grid cells with no neighbours within this
        radius are set to ``nodata``. ``None`` disables the distance limit.
    k : int, optional
        Number of nearest neighbours to use, by default 12.
    nodata : float, optional
        Value assigned to cells with no data, by default -9999.0.

    Returns
    -------
    NDArray[np.float64]
        Interpolated grid of shape (ny, nx).

    Raises
    ------
    OcculusRasterError
        If inputs are invalid or interpolation fails.
    """
    try:
        from scipy.spatial import KDTree  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("scipy is required for IDW interpolation: pip install scipy") from exc

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise OcculusRasterError(f"xy must be (N, 2), got shape {xy.shape}")
    if z.ndim != 1 or z.shape[0] != xy.shape[0]:
        raise OcculusRasterError(
            f"z must be (N,) matching xy rows, got z.shape={z.shape}, xy.shape={xy.shape}"
        )
    if len(xy) == 0:
        raise OcculusRasterError("Cannot interpolate from zero source points")
    if power <= 0:
        raise OcculusRasterError(f"power must be positive, got {power}")

    k_actual = min(k, len(xy))

    # Build KDTree from source points
    tree = KDTree(xy)

    # Create meshgrid of cell centres
    gx, gy = np.meshgrid(grid_x, grid_y)
    query_pts = np.column_stack((gx.ravel(), gy.ravel()))

    # Query k nearest neighbours
    distances, indices = tree.query(query_pts, k=k_actual, workers=-1)

    # Ensure 2D arrays even when k_actual == 1
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    result = np.full(len(query_pts), nodata, dtype=np.float64)

    # Handle exact matches (distance == 0) — use the coincident point's Z
    exact_mask = distances[:, 0] == 0.0
    if exact_mask.any():
        result[exact_mask] = z[indices[exact_mask, 0]]

    # IDW for non-exact points
    interp_mask = ~exact_mask
    if interp_mask.any():
        d = distances[interp_mask]
        idx = indices[interp_mask]
        z_vals = z[idx]

        # Apply max_dist filter
        if max_dist is not None:
            valid = d <= max_dist
            # If no neighbours within max_dist, cell stays nodata
            has_valid = valid.any(axis=1)
            # For cells with some valid neighbours, zero out invalid ones
            d_masked = np.where(valid, d, np.inf)
        else:
            has_valid = np.ones(d.shape[0], dtype=bool)
            d_masked = d

        # Compute weights: 1 / d^power
        with np.errstate(divide="ignore"):
            weights = 1.0 / np.power(d_masked, power)
        weights = np.where(np.isinf(weights), 0.0, weights)

        w_sum = weights.sum(axis=1)
        safe = (w_sum > 0) & has_valid
        weighted_z = (weights * z_vals).sum(axis=1)

        interp_indices = np.where(interp_mask)[0]
        result[interp_indices[safe]] = weighted_z[safe] / w_sum[safe]

    grid = result.reshape(len(grid_y), len(grid_x))

    n_filled = int((grid != nodata).sum())
    logger.debug(
        "idw_interpolate: %d/%d cells filled (k=%d, power=%.1f)",
        n_filled,
        grid.size,
        k_actual,
        power,
    )
    return grid


def nearest_interpolate(
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    *,
    max_dist: float | None = None,
    nodata: float = -9999.0,
) -> NDArray[np.float64]:
    """Nearest-neighbour interpolation onto a regular grid.

    Each grid cell centre is assigned the Z value of the closest source point.

    Parameters
    ----------
    xy : NDArray[np.float64]
        Source point XY coordinates, shape (N, 2).
    z : NDArray[np.float64]
        Source point Z values, shape (N,).
    grid_x : NDArray[np.float64]
        Grid cell centre X coordinates, shape (nx,).
    grid_y : NDArray[np.float64]
        Grid cell centre Y coordinates, shape (ny,).
    max_dist : float or None, optional
        Maximum distance to accept a neighbour. Cells with no point within
        this radius are set to ``nodata``. ``None`` disables the limit.
    nodata : float, optional
        Value assigned to cells with no data, by default -9999.0.

    Returns
    -------
    NDArray[np.float64]
        Interpolated grid of shape (ny, nx).

    Raises
    ------
    OcculusRasterError
        If inputs are invalid or interpolation fails.
    """
    try:
        from scipy.spatial import KDTree  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for nearest-neighbour interpolation: pip install scipy"
        ) from exc

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise OcculusRasterError(f"xy must be (N, 2), got shape {xy.shape}")
    if z.ndim != 1 or z.shape[0] != xy.shape[0]:
        raise OcculusRasterError(
            f"z must be (N,) matching xy rows, got z.shape={z.shape}, xy.shape={xy.shape}"
        )
    if len(xy) == 0:
        raise OcculusRasterError("Cannot interpolate from zero source points")

    tree = KDTree(xy)

    gx, gy = np.meshgrid(grid_x, grid_y)
    query_pts = np.column_stack((gx.ravel(), gy.ravel()))

    distances, indices = tree.query(query_pts, k=1, workers=-1)

    result = z[indices].copy()

    if max_dist is not None:
        result[distances > max_dist] = nodata

    grid = result.reshape(len(grid_y), len(grid_x))

    n_filled = int((grid != nodata).sum())
    logger.debug(
        "nearest_interpolate: %d/%d cells filled",
        n_filled,
        grid.size,
    )
    return grid
