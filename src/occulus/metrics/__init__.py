"""Point cloud metrics, statistics, and raster products.

Available functions
-------------------
- :func:`point_density` — 2D point density raster (pts/m²)
- :func:`canopy_height_model` — max-Z raster above ground (aerial/UAV)
- :func:`coverage_statistics` — summary statistics over the density map
- :func:`compute_cloud_statistics` — per-cloud elevation and intensity statistics

All functions are pure NumPy and do not require optional dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusValidationError, UnsupportedPlatformError
from occulus.types import Platform, PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "CloudStatistics",
    "CoverageStatistics",
    "canopy_height_model",
    "compute_cloud_statistics",
    "coverage_statistics",
    "point_density",
]


@dataclass
class CloudStatistics:
    """Summary statistics for a point cloud.

    Attributes
    ----------
    n_points : int
        Total point count.
    bounds : NDArray[np.float64]
        Bounding box as (2, 3) array ``[[xmin,ymin,zmin],[xmax,ymax,zmax]]``.
    centroid : NDArray[np.float64]
        Centroid as (3,) array.
    z_min : float
        Minimum elevation.
    z_max : float
        Maximum elevation.
    z_mean : float
        Mean elevation.
    z_std : float
        Standard deviation of elevation.
    z_percentiles : dict[int, float]
        Elevation percentiles at 5, 25, 50, 75, 95.
    intensity_mean : float | None
        Mean intensity value, or ``None`` if no intensity data.
    intensity_std : float | None
        Standard deviation of intensity, or ``None`` if no intensity data.
    """

    n_points: int
    bounds: NDArray[np.float64]
    centroid: NDArray[np.float64]
    z_min: float
    z_max: float
    z_mean: float
    z_std: float
    z_percentiles: dict[int, float]
    intensity_mean: float | None = None
    intensity_std: float | None = None


@dataclass
class CoverageStatistics:
    """Coverage statistics derived from a point density map.

    Attributes
    ----------
    mean_density : float
        Mean point density over occupied cells (pts/unit²).
    min_density : float
        Minimum density over all cells.
    max_density : float
        Maximum density over all cells.
    std_density : float
        Standard deviation of density.
    gap_fraction : float
        Fraction of raster cells with zero density (coverage gaps).
    total_area : float
        Total raster area in coordinate units squared.
    covered_area : float
        Area of cells with at least one point.
    """

    mean_density: float
    min_density: float
    max_density: float
    std_density: float
    gap_fraction: float
    total_area: float
    covered_area: float


def compute_cloud_statistics(cloud: PointCloud) -> CloudStatistics:
    """Compute summary statistics for a point cloud.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have at least one point.

    Returns
    -------
    CloudStatistics
        Elevation and intensity summary statistics.

    Raises
    ------
    OcculusValidationError
        If the cloud is empty.
    """
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot compute statistics of an empty cloud")

    z = cloud.xyz[:, 2]
    pct_keys = [5, 25, 50, 75, 95]
    z_percentiles = {p: float(np.percentile(z, p)) for p in pct_keys}

    intensity_mean: float | None = None
    intensity_std: float | None = None
    if cloud.intensity is not None:
        intensity_mean = float(cloud.intensity.mean())
        intensity_std = float(cloud.intensity.std())

    return CloudStatistics(
        n_points=cloud.n_points,
        bounds=cloud.bounds,
        centroid=cloud.centroid,
        z_min=float(z.min()),
        z_max=float(z.max()),
        z_mean=float(z.mean()),
        z_std=float(z.std()),
        z_percentiles=z_percentiles,
        intensity_mean=intensity_mean,
        intensity_std=intensity_std,
    )


def point_density(
    cloud: PointCloud,
    resolution: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute a 2D point density raster.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    resolution : float, optional
        Raster cell size in coordinate units, by default 1.0.

    Returns
    -------
    density : NDArray[np.float64]
        2D density array of shape (ny, nx) in points per cell.
        Divide by ``resolution**2`` to get pts/unit².
    x_edges : NDArray[np.float64]
        X bin edges of length nx+1.
    y_edges : NDArray[np.float64]
        Y bin edges of length ny+1.

    Raises
    ------
    OcculusValidationError
        If ``resolution`` is not positive or the cloud is empty.
    """
    if resolution <= 0:
        raise OcculusValidationError(f"resolution must be positive, got {resolution}")
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot compute density of an empty cloud")

    x = cloud.xyz[:, 0]
    y = cloud.xyz[:, 1]

    x_edges = np.arange(x.min(), x.max() + resolution, resolution)
    y_edges = np.arange(y.min(), y.max() + resolution, resolution)

    density, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges))
    density = density.T  # shape (ny, nx), consistent with (row=y, col=x)

    logger.debug(
        "point_density: raster %dx%d (res=%.3f), %.1f pts/cell avg",
        density.shape[1],
        density.shape[0],
        resolution,
        density[density > 0].mean() if density.any() else 0,
    )
    return density, x_edges, y_edges


def canopy_height_model(
    cloud: PointCloud,
    resolution: float = 1.0,
    *,
    ground_class: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a Canopy Height Model (CHM) raster.

    Separates ground (ASPRS class 2) from non-ground points, interpolates
    a ground surface, and computes height above ground for each raster cell.

    Parameters
    ----------
    cloud : PointCloud
        Input cloud. Should have a classification array with ground class 2.
        Must be aerial or UAV platform.
    resolution : float, optional
        Raster cell size in coordinate units, by default 1.0.
    ground_class : int, optional
        ASPRS classification code for ground points, by default 2.

    Returns
    -------
    chm : NDArray[np.float64]
        CHM raster of shape (ny, nx) — height above ground per cell.
        Cells with no non-ground points are set to 0.
    x_edges : NDArray[np.float64]
        X bin edges.
    y_edges : NDArray[np.float64]
        Y bin edges.

    Raises
    ------
    UnsupportedPlatformError
        If the cloud platform is ``TERRESTRIAL``.
    OcculusValidationError
        If ``resolution`` is not positive, the cloud is empty, or no
        classification array is present.
    """
    if cloud.platform == Platform.TERRESTRIAL:
        raise UnsupportedPlatformError(
            "canopy_height_model requires aerial or UAV data, not terrestrial. "
            "For TLS analysis, use point_density() instead."
        )
    if resolution <= 0:
        raise OcculusValidationError(f"resolution must be positive, got {resolution}")
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot build CHM from an empty cloud")
    if cloud.classification is None:
        raise OcculusValidationError(
            "CHM requires a classification array. "
            "Run classify_ground_csf() or classify_ground_pmf() first."
        )

    xyz = cloud.xyz
    cls = cloud.classification

    ground_mask = cls == ground_class
    if not ground_mask.any():
        raise OcculusValidationError(
            f"No ground points found (class {ground_class}). Run ground classification first."
        )

    ground_xyz = xyz[ground_mask]
    veg_xyz = xyz[~ground_mask]

    x_min = xyz[:, 0].min()
    x_max = xyz[:, 0].max()
    y_min = xyz[:, 1].min()
    y_max = xyz[:, 1].max()

    x_edges = np.arange(x_min, x_max + resolution, resolution)
    y_edges = np.arange(y_min, y_max + resolution, resolution)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # Build ground surface: minimum Z per cell — vectorised via np.minimum.at
    ground_surface = np.full((ny, nx), np.inf)
    g_col = np.clip(((ground_xyz[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    g_row = np.clip(((ground_xyz[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)
    g_flat = g_row * nx + g_col
    np.minimum.at(ground_surface.ravel(), g_flat, ground_xyz[:, 2])
    ground_surface[ground_surface == np.inf] = np.nan

    # Fill NaN ground cells with nearest finite value
    valid_mask = np.isfinite(ground_surface)
    if valid_mask.sum() > 0 and (~valid_mask).sum() > 0:
        from scipy.spatial import KDTree  # type: ignore[import-untyped]

        rows, cols = np.where(valid_mask)
        valid_pts = np.column_stack((rows, cols))
        nv_rows, nv_cols = np.where(~valid_mask)
        nv_pts = np.column_stack((nv_rows, nv_cols))
        fill_tree = KDTree(valid_pts)
        _, fill_nn = fill_tree.query(nv_pts, k=1, workers=-1)
        ground_surface[nv_rows, nv_cols] = ground_surface[rows[fill_nn], cols[fill_nn]]

    # Build vegetation max-Z raster — vectorised via np.maximum.at
    veg_max = np.full((ny, nx), -np.inf)
    if len(veg_xyz) > 0:
        v_col = np.clip(((veg_xyz[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
        v_row = np.clip(((veg_xyz[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)
        v_flat = v_row * nx + v_col
        np.maximum.at(veg_max.ravel(), v_flat, veg_xyz[:, 2])
    veg_max[veg_max == -np.inf] = np.nan

    # CHM = max vegetation Z - ground Z
    chm = np.zeros((ny, nx), dtype=np.float64)
    has_veg = np.isfinite(veg_max)
    chm[has_veg] = np.maximum(0.0, veg_max[has_veg] - ground_surface[has_veg])

    logger.info(
        "canopy_height_model: %dx%d raster, max_height=%.2f, res=%.2f",
        nx,
        ny,
        float(chm.max()),
        resolution,
    )
    return chm, x_edges, y_edges


def coverage_statistics(
    cloud: PointCloud,
    resolution: float = 1.0,
) -> CoverageStatistics:
    """Compute coverage statistics from a point density map.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    resolution : float, optional
        Raster cell size for density computation, by default 1.0.

    Returns
    -------
    CoverageStatistics
        Density statistics and gap fraction.

    Raises
    ------
    OcculusValidationError
        If ``resolution`` is not positive or the cloud is empty.
    """
    density, _x_edges, _y_edges = point_density(cloud, resolution)

    cell_area = resolution**2
    total_cells = density.size
    covered_cells = int((density > 0).sum())
    gap_fraction = 1.0 - covered_cells / total_cells if total_cells > 0 else 1.0

    occupied = density[density > 0]
    return CoverageStatistics(
        mean_density=float(occupied.mean() / cell_area) if len(occupied) > 0 else 0.0,
        min_density=float(occupied.min() / cell_area) if len(occupied) > 0 else 0.0,
        max_density=float(occupied.max() / cell_area) if len(occupied) > 0 else 0.0,
        std_density=float(occupied.std() / cell_area) if len(occupied) > 0 else 0.0,
        gap_fraction=gap_fraction,
        total_area=float(total_cells * cell_area),
        covered_area=float(covered_cells * cell_area),
    )
