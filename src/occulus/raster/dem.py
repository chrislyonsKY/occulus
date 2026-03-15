"""DEM / DSM / DTM generation from point clouds.

Creates regular-grid elevation rasters from classified point cloud data:

- :func:`create_dsm` — Digital Surface Model (max-Z of all points)
- :func:`create_dtm` — Digital Terrain Model (ground-only surface)
- :func:`create_dem` — Alias for :func:`create_dtm`

All functions produce a :class:`RasterResult` containing the grid, edge
coordinates, resolution, and CRS metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusRasterError
from occulus.raster.interpolation import idw_interpolate, nearest_interpolate
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "RasterResult",
    "create_dem",
    "create_dsm",
    "create_dtm",
]

_INTERPOLATION_METHODS = {"idw", "nearest"}


@dataclass
class RasterResult:
    """Result container for rasterized elevation models.

    Attributes
    ----------
    data : NDArray[np.float64]
        Elevation grid of shape (ny, nx).
    x_edges : NDArray[np.float64]
        X bin edges of length nx + 1.
    y_edges : NDArray[np.float64]
        Y bin edges of length ny + 1.
    resolution : float
        Grid cell size in coordinate units.
    crs : str
        Coordinate reference system identifier (EPSG code or WKT), or
        empty string if unknown.
    nodata : float
        Value used for cells with no data. Default -9999.0.
    """

    data: NDArray[np.float64]
    x_edges: NDArray[np.float64]
    y_edges: NDArray[np.float64]
    resolution: float
    crs: str
    nodata: float = -9999.0


def _validate_inputs(
    cloud: PointCloud,
    resolution: float,
    method: str,
) -> None:
    """Validate common inputs for DEM creation functions.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    resolution : float
        Grid cell size.
    method : str
        Interpolation method name.

    Raises
    ------
    OcculusRasterError
        If any input is invalid.
    """
    if cloud.n_points == 0:
        raise OcculusRasterError("Cannot rasterize an empty point cloud")
    if resolution <= 0:
        raise OcculusRasterError(f"resolution must be positive, got {resolution}")
    if method not in _INTERPOLATION_METHODS:
        raise OcculusRasterError(
            f"Unknown interpolation method '{method}'. Supported: {sorted(_INTERPOLATION_METHODS)}"
        )


def _build_grid_edges(
    cloud: PointCloud,
    resolution: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute regular grid bin edges from point cloud bounds.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    resolution : float
        Grid cell size.

    Returns
    -------
    x_edges : NDArray[np.float64]
        X bin edges.
    y_edges : NDArray[np.float64]
        Y bin edges.
    """
    bounds = cloud.bounds
    x_min, y_min = bounds[0, 0], bounds[0, 1]
    x_max, y_max = bounds[1, 0], bounds[1, 1]

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)

    return x_edges, y_edges


def _grid_centres(
    edges: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute cell centres from bin edges.

    Parameters
    ----------
    edges : NDArray[np.float64]
        Bin edges of length n + 1.

    Returns
    -------
    NDArray[np.float64]
        Cell centres of length n.
    """
    return (edges[:-1] + edges[1:]) / 2.0


def _bin_max_z(
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    x_edges: NDArray[np.float64],
    y_edges: NDArray[np.float64],
    nodata: float,
) -> NDArray[np.float64]:
    """Compute maximum Z per grid cell using direct binning.

    Parameters
    ----------
    xy : NDArray[np.float64]
        Point XY coordinates, shape (N, 2).
    z : NDArray[np.float64]
        Point Z values, shape (N,).
    x_edges : NDArray[np.float64]
        X bin edges.
    y_edges : NDArray[np.float64]
        Y bin edges.
    nodata : float
        Value for empty cells.

    Returns
    -------
    NDArray[np.float64]
        Grid of shape (ny, nx) with max-Z per cell.
    """
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    x_min = x_edges[0]
    y_min = y_edges[0]
    resolution = float(x_edges[1] - x_edges[0])

    grid = np.full((ny, nx), nodata, dtype=np.float64)

    col = np.clip(((xy[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    row = np.clip(((xy[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)

    # Vectorised max-Z per cell using lexsort
    cell_id = row * nx + col
    order = np.lexsort((z, cell_id))
    cell_sorted = cell_id[order]
    z_sorted = z[order]

    # Find last occurrence of each cell (max Z because sorted ascending)
    breaks = np.where(np.diff(cell_sorted) != 0)[0]
    last_indices = np.append(breaks, len(cell_sorted) - 1)
    unique_cells = cell_sorted[last_indices]

    r_unique = unique_cells // nx
    c_unique = unique_cells % nx
    grid[r_unique, c_unique] = z_sorted[last_indices]

    return grid


def _interpolate_grid(
    xy: NDArray[np.float64],
    z: NDArray[np.float64],
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    method: str,
    nodata: float,
) -> NDArray[np.float64]:
    """Dispatch to the requested interpolation method.

    Parameters
    ----------
    xy : NDArray[np.float64]
        Source XY coordinates, shape (N, 2).
    z : NDArray[np.float64]
        Source Z values, shape (N,).
    grid_x : NDArray[np.float64]
        Grid cell centre X coordinates.
    grid_y : NDArray[np.float64]
        Grid cell centre Y coordinates.
    method : str
        ``"idw"`` or ``"nearest"``.
    nodata : float
        Nodata value for empty cells.

    Returns
    -------
    NDArray[np.float64]
        Interpolated grid of shape (ny, nx).
    """
    if method == "idw":
        return idw_interpolate(xy, z, grid_x, grid_y, nodata=nodata)
    return nearest_interpolate(xy, z, grid_x, grid_y, nodata=nodata)


def create_dsm(
    cloud: PointCloud,
    resolution: float = 1.0,
    *,
    method: str = "idw",
    nodata: float = -9999.0,
) -> RasterResult:
    """Create a Digital Surface Model (max-Z of all points).

    The DSM represents the highest surface at each grid cell, including
    buildings, vegetation, and other above-ground features.

    For cells that contain at least one point, the maximum Z value is used
    directly. Empty cells are filled using the specified interpolation method.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    resolution : float, optional
        Grid cell size in coordinate units, by default 1.0.
    method : str, optional
        Interpolation method for gap-filling: ``"idw"`` or ``"nearest"``,
        by default ``"idw"``.
    nodata : float, optional
        Value assigned to cells with insufficient data, by default -9999.0.

    Returns
    -------
    RasterResult
        DSM elevation grid with metadata.

    Raises
    ------
    OcculusRasterError
        If the cloud is empty, resolution is non-positive, or interpolation
        method is unknown.
    """
    _validate_inputs(cloud, resolution, method)

    x_edges, y_edges = _build_grid_edges(cloud, resolution)
    xy = cloud.xyz[:, :2]
    z = cloud.xyz[:, 2]

    # Direct binning for max-Z
    grid = _bin_max_z(xy, z, x_edges, y_edges, nodata)

    # Interpolate empty cells
    empty_mask = grid == nodata
    if empty_mask.any() and (~empty_mask).any():
        grid_cx = _grid_centres(x_edges)
        grid_cy = _grid_centres(y_edges)
        filled = _interpolate_grid(xy, z, grid_cx, grid_cy, method, nodata)

        # Only fill cells that were empty
        grid[empty_mask] = filled[empty_mask]

    crs = cloud.metadata.coordinate_system

    logger.info(
        "create_dsm: %dx%d grid (res=%.2f), z_range=[%.2f, %.2f]",
        grid.shape[1],
        grid.shape[0],
        resolution,
        float(np.min(grid[grid != nodata])) if (grid != nodata).any() else 0.0,
        float(np.max(grid[grid != nodata])) if (grid != nodata).any() else 0.0,
    )

    return RasterResult(
        data=grid,
        x_edges=x_edges,
        y_edges=y_edges,
        resolution=resolution,
        crs=crs,
        nodata=nodata,
    )


def create_dtm(
    cloud: PointCloud,
    resolution: float = 1.0,
    *,
    method: str = "idw",
    ground_class: int = 2,
    nodata: float = -9999.0,
) -> RasterResult:
    """Create a Digital Terrain Model from ground-classified points.

    The DTM represents the bare-earth surface using only points classified
    as ground (ASPRS class 2 by default). Empty cells are filled using the
    specified interpolation method.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have a classification array with ground
        points matching ``ground_class``.
    resolution : float, optional
        Grid cell size in coordinate units, by default 1.0.
    method : str, optional
        Interpolation method for gap-filling: ``"idw"`` or ``"nearest"``,
        by default ``"idw"``.
    ground_class : int, optional
        ASPRS classification code for ground points, by default 2.
    nodata : float, optional
        Value assigned to cells with no data, by default -9999.0.

    Returns
    -------
    RasterResult
        DTM elevation grid with metadata.

    Raises
    ------
    OcculusRasterError
        If the cloud is empty, has no classification, contains no ground
        points, resolution is non-positive, or method is unknown.
    """
    _validate_inputs(cloud, resolution, method)

    if cloud.classification is None:
        raise OcculusRasterError(
            "DTM requires a classification array. "
            "Run classify_ground_csf() or classify_ground_pmf() first."
        )

    ground_mask = cloud.classification == ground_class
    if not ground_mask.any():
        raise OcculusRasterError(
            f"No ground points found (class {ground_class}). Run ground classification first."
        )

    ground_xyz = cloud.xyz[ground_mask]
    xy = ground_xyz[:, :2]
    z = ground_xyz[:, 2]

    # Build grid over the full cloud extent, not just ground points
    x_edges, y_edges = _build_grid_edges(cloud, resolution)
    grid_cx = _grid_centres(x_edges)
    grid_cy = _grid_centres(y_edges)

    # Interpolate ground surface
    grid = _interpolate_grid(xy, z, grid_cx, grid_cy, method, nodata)

    crs = cloud.metadata.coordinate_system

    logger.info(
        "create_dtm: %dx%d grid (res=%.2f), %d ground pts, z_range=[%.2f, %.2f]",
        grid.shape[1],
        grid.shape[0],
        resolution,
        int(ground_mask.sum()),
        float(np.min(grid[grid != nodata])) if (grid != nodata).any() else 0.0,
        float(np.max(grid[grid != nodata])) if (grid != nodata).any() else 0.0,
    )

    return RasterResult(
        data=grid,
        x_edges=x_edges,
        y_edges=y_edges,
        resolution=resolution,
        crs=crs,
        nodata=nodata,
    )


def create_dem(
    cloud: PointCloud,
    resolution: float = 1.0,
    *,
    method: str = "idw",
    ground_class: int = 2,
    nodata: float = -9999.0,
) -> RasterResult:
    """Create a Digital Elevation Model (alias for :func:`create_dtm`).

    This function is a convenience alias. In common usage, DEM and DTM are
    used interchangeably to mean the bare-earth surface. See
    :func:`create_dtm` for full documentation.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud with ground classification.
    resolution : float, optional
        Grid cell size in coordinate units, by default 1.0.
    method : str, optional
        Interpolation method: ``"idw"`` or ``"nearest"``, by default ``"idw"``.
    ground_class : int, optional
        ASPRS classification code for ground, by default 2.
    nodata : float, optional
        Nodata value, by default -9999.0.

    Returns
    -------
    RasterResult
        DTM elevation grid with metadata.

    Raises
    ------
    OcculusRasterError
        See :func:`create_dtm`.
    """
    return create_dtm(
        cloud,
        resolution,
        method=method,
        ground_class=ground_class,
        nodata=nodata,
    )
