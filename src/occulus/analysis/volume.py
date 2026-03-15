"""Cut/fill volume computation between two point cloud surfaces.

Rasterizes two surfaces (e.g., pre- and post-construction terrain models) onto
matching grids and computes per-cell elevation differences to derive cut, fill,
and net volumes.

All computations use pure NumPy — no optional dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "VolumeResult",
    "compute_volume",
]


@dataclass
class VolumeResult:
    """Result of a cut/fill volume computation.

    Attributes
    ----------
    cut_volume : float
        Volume of material removed (surface above reference), in cubic units.
    fill_volume : float
        Volume of material added (surface below reference), in cubic units.
    net_volume : float
        Net volume change (cut - fill). Positive means net removal.
    resolution : float
        Grid cell size used for the computation.
    area : float
        Total planimetric area covered by the computation grid.
    cut_area : float
        Planimetric area where the surface is above the reference.
    fill_area : float
        Planimetric area where the surface is below the reference.
    """

    cut_volume: float
    fill_volume: float
    net_volume: float
    resolution: float
    area: float
    cut_area: float
    fill_area: float


def compute_volume(
    surface: PointCloud,
    reference: PointCloud,
    resolution: float = 1.0,
    method: str = "grid",
) -> VolumeResult:
    """Compute cut/fill volumes between two point cloud surfaces.

    Both surfaces are rasterized onto a common 2D grid. For each cell the
    mean elevation of each surface is computed, and the per-cell difference
    ``(surface_z - reference_z)`` is multiplied by the cell area to obtain
    volume.

    Parameters
    ----------
    surface : PointCloud
        The measured (as-built or post-event) surface.
    reference : PointCloud
        The reference (design or pre-event) surface.
    resolution : float, optional
        Grid cell size in coordinate units, by default 1.0.
    method : str, optional
        Volume computation method. Currently only ``"grid"`` is supported.

    Returns
    -------
    VolumeResult
        Dataclass containing cut, fill, and net volumes together with area
        statistics.

    Raises
    ------
    OcculusValidationError
        If ``resolution`` is not positive, either cloud is empty, or an
        unsupported ``method`` is specified.
    """
    if resolution <= 0:
        raise OcculusValidationError(f"resolution must be positive, got {resolution}")
    if surface.n_points == 0:
        raise OcculusValidationError("surface point cloud is empty")
    if reference.n_points == 0:
        raise OcculusValidationError("reference point cloud is empty")
    if method != "grid":
        raise OcculusValidationError(
            f"Unsupported method '{method}'. Only 'grid' is currently supported."
        )

    # Determine common bounding box
    all_xy = np.vstack((surface.xyz[:, :2], reference.xyz[:, :2]))
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    if nx < 1 or ny < 1:
        raise OcculusValidationError(
            "Point clouds are too small to form a grid at the given resolution. "
            f"Extents: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], resolution={resolution}"
        )

    surface_grid = _rasterize_mean_z(surface.xyz, x_min, y_min, resolution, nx, ny)
    reference_grid = _rasterize_mean_z(reference.xyz, x_min, y_min, resolution, nx, ny)

    # Only compute where both surfaces have data
    valid = np.isfinite(surface_grid) & np.isfinite(reference_grid)
    diff = np.full_like(surface_grid, np.nan)
    diff[valid] = surface_grid[valid] - reference_grid[valid]

    cell_area = resolution * resolution

    cut_mask = valid & (diff > 0)
    fill_mask = valid & (diff < 0)

    cut_volume = float(np.nansum(diff[cut_mask]) * cell_area)
    fill_volume = float(np.nansum(np.abs(diff[fill_mask])) * cell_area)
    net_volume = cut_volume - fill_volume

    n_valid = int(valid.sum())
    n_cut = int(cut_mask.sum())
    n_fill = int(fill_mask.sum())

    logger.info(
        "compute_volume: cut=%.2f, fill=%.2f, net=%.2f (grid %dx%d, res=%.2f)",
        cut_volume,
        fill_volume,
        net_volume,
        nx,
        ny,
        resolution,
    )

    return VolumeResult(
        cut_volume=cut_volume,
        fill_volume=fill_volume,
        net_volume=net_volume,
        resolution=resolution,
        area=float(n_valid * cell_area),
        cut_area=float(n_cut * cell_area),
        fill_area=float(n_fill * cell_area),
    )


def _rasterize_mean_z(
    xyz: NDArray[np.float64],
    x_min: float,
    y_min: float,
    resolution: float,
    nx: int,
    ny: int,
) -> NDArray[np.float64]:
    """Rasterize points to a 2D grid using mean Z per cell.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        (N, 3) point array.
    x_min : float
        Minimum X coordinate of the grid origin.
    y_min : float
        Minimum Y coordinate of the grid origin.
    resolution : float
        Grid cell size.
    nx : int
        Number of columns.
    ny : int
        Number of rows.

    Returns
    -------
    NDArray[np.float64]
        (ny, nx) grid of mean Z values. Cells with no points are ``NaN``.
    """
    col = np.clip(((xyz[:, 0] - x_min) / resolution).astype(np.intp), 0, nx - 1)
    row = np.clip(((xyz[:, 1] - y_min) / resolution).astype(np.intp), 0, ny - 1)

    z_sum = np.zeros((ny, nx), dtype=np.float64)
    z_count = np.zeros((ny, nx), dtype=np.int64)

    # Use np.add.at for unbuffered accumulation
    np.add.at(z_sum, (row, col), xyz[:, 2])
    np.add.at(z_count, (row, col), 1)

    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    occupied = z_count > 0
    grid[occupied] = z_sum[occupied] / z_count[occupied]

    return grid
