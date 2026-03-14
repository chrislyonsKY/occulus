"""Point cloud filtering and downsampling operations.

All filters operate on :class:`~occulus.types.PointCloud` instances and return
new clouds — they never mutate the input.

Available filters
-----------------
- :func:`voxel_downsample` — grid-based spatial downsampling
- :func:`random_downsample` — uniform random point selection
- :func:`statistical_outlier_removal` — remove points with unusual neighbourhood distances
- :func:`radius_outlier_removal` — remove points with too few neighbours within a radius
- :func:`crop` — axis-aligned bounding box crop

All implementations use pure NumPy and SciPy. No optional dependencies required.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusValidationError
from occulus.types import AcquisitionMetadata, PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "crop",
    "radius_outlier_removal",
    "random_downsample",
    "statistical_outlier_removal",
    "voxel_downsample",
]


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Downsample a point cloud by retaining one point per voxel cell.

    Points are grouped into a regular 3D grid of cubes with side length
    ``voxel_size``. For each occupied voxel the first point (after lexicographic
    sort on voxel index) is retained, preserving all per-point attributes.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have at least one point.
    voxel_size : float
        Edge length of each voxel cube, in the same units as the cloud
        coordinates. Must be strictly positive.

    Returns
    -------
    PointCloud
        Downsampled cloud of the same concrete subtype as the input.

    Raises
    ------
    OcculusValidationError
        If ``voxel_size`` is not positive or the cloud is empty.
    """
    if voxel_size <= 0:
        raise OcculusValidationError(f"voxel_size must be positive, got {voxel_size}")
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot downsample an empty point cloud")

    xyz = cloud.xyz
    origin = xyz.min(axis=0)
    voxel_idx = np.floor((xyz - origin) / voxel_size).astype(np.int64)

    # Encode 3D voxel index as a single integer key for grouping
    max_idx = voxel_idx.max(axis=0) + 1
    flat = (
        voxel_idx[:, 0] * (max_idx[1] * max_idx[2]) + voxel_idx[:, 1] * max_idx[2] + voxel_idx[:, 2]
    )

    sort_order = np.argsort(flat, kind="stable")
    flat_sorted = flat[sort_order]
    _, first_occurrences = np.unique(flat_sorted, return_index=True)
    selected = sort_order[first_occurrences]

    logger.debug(
        "voxel_downsample: %d → %d points (voxel_size=%.4f)",
        cloud.n_points,
        len(selected),
        voxel_size,
    )
    return _subset(cloud, selected)


def random_downsample(
    cloud: PointCloud,
    fraction: float,
    *,
    seed: int | None = None,
) -> PointCloud:
    """Randomly downsample a point cloud to a fraction of its points.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    fraction : float
        Fraction of points to retain, in (0.0, 1.0].
    seed : int | None, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    PointCloud
        Downsampled cloud of the same concrete subtype as the input.

    Raises
    ------
    OcculusValidationError
        If ``fraction`` is not in (0.0, 1.0] or the cloud is empty.
    """
    if not (0.0 < fraction <= 1.0):
        raise OcculusValidationError(f"fraction must be in (0.0, 1.0], got {fraction}")
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot downsample an empty point cloud")

    rng = np.random.default_rng(seed)
    k = max(1, int(cloud.n_points * fraction))
    selected = rng.choice(cloud.n_points, size=k, replace=False)
    selected.sort()

    logger.debug("random_downsample: %d → %d points", cloud.n_points, k)
    return _subset(cloud, selected)


def statistical_outlier_removal(
    cloud: PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[PointCloud, NDArray[np.bool_]]:
    """Remove statistical outliers based on nearest-neighbour distances.

    For each point the mean distance to its ``nb_neighbors`` nearest
    neighbours is computed. Points whose mean distance exceeds
    ``global_mean + std_ratio * global_std`` are classified as outliers
    and removed.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    nb_neighbors : int, optional
        Number of nearest neighbours to consider (excluding self), by default 20.
    std_ratio : float, optional
        Standard deviation multiplier for the removal threshold, by default 2.0.
        Lower values remove points more aggressively.

    Returns
    -------
    PointCloud
        Cloud with outliers removed.
    NDArray[np.bool_]
        Boolean inlier mask of length ``n_points`` (``True`` = kept).

    Raises
    ------
    OcculusValidationError
        If ``nb_neighbors`` is non-positive or exceeds the number of points.
    """
    if nb_neighbors <= 0:
        raise OcculusValidationError(f"nb_neighbors must be positive, got {nb_neighbors}")
    if nb_neighbors >= cloud.n_points:
        raise OcculusValidationError(
            f"nb_neighbors ({nb_neighbors}) must be less than n_points ({cloud.n_points})"
        )

    tree = KDTree(cloud.xyz)
    distances, _ = tree.query(cloud.xyz, k=nb_neighbors + 1, workers=-1)
    mean_distances = distances[:, 1:].mean(axis=1)  # exclude self (distance = 0)

    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    inlier_mask: NDArray[np.bool_] = mean_distances <= threshold

    logger.debug(
        "statistical_outlier_removal: removed %d/%d points",
        (~inlier_mask).sum(),
        cloud.n_points,
    )
    return _subset(cloud, np.where(inlier_mask)[0]), inlier_mask


def radius_outlier_removal(
    cloud: PointCloud,
    radius: float,
    min_neighbors: int = 2,
) -> tuple[PointCloud, NDArray[np.bool_]]:
    """Remove points that have fewer than ``min_neighbors`` within ``radius``.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    radius : float
        Search radius in the same units as the cloud coordinates.
        Must be strictly positive.
    min_neighbors : int, optional
        Minimum number of neighbours (excluding self) required to keep a
        point, by default 2.

    Returns
    -------
    PointCloud
        Cloud with isolated points removed.
    NDArray[np.bool_]
        Boolean inlier mask of length ``n_points`` (``True`` = kept).

    Raises
    ------
    OcculusValidationError
        If ``radius`` is not positive or ``min_neighbors`` is non-positive.
    """
    if radius <= 0:
        raise OcculusValidationError(f"radius must be positive, got {radius}")
    if min_neighbors <= 0:
        raise OcculusValidationError(f"min_neighbors must be positive, got {min_neighbors}")

    tree = KDTree(cloud.xyz)
    counts: NDArray[np.intp] = np.asarray(
        tree.query_ball_point(cloud.xyz, r=radius, return_length=True, workers=-1)
    )
    inlier_mask: NDArray[np.bool_] = (counts - 1) >= min_neighbors

    logger.debug(
        "radius_outlier_removal: removed %d/%d points (r=%.4f, min_nb=%d)",
        (~inlier_mask).sum(),
        cloud.n_points,
        radius,
        min_neighbors,
    )
    return _subset(cloud, np.where(inlier_mask)[0]), inlier_mask


def crop(
    cloud: PointCloud,
    bbox: tuple[float, float, float, float, float, float],
) -> PointCloud:
    """Crop a point cloud to an axis-aligned bounding box.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    bbox : tuple[float, float, float, float, float, float]
        Bounding box as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.
        Bounds are inclusive.

    Returns
    -------
    PointCloud
        Points that fall within (inclusive of) the bounding box.

    Raises
    ------
    OcculusValidationError
        If ``bbox`` does not have exactly 6 elements or any min >= max.
    """
    if len(bbox) != 6:
        raise OcculusValidationError(f"bbox must have 6 elements, got {len(bbox)}")

    xmin, ymin, zmin, xmax, ymax, zmax = bbox

    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        raise OcculusValidationError(
            f"bbox min values must be strictly less than max values: {bbox}"
        )

    xyz = cloud.xyz
    mask: NDArray[np.bool_] = (
        (xyz[:, 0] >= xmin)
        & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin)
        & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin)
        & (xyz[:, 2] <= zmax)
    )

    logger.debug(
        "crop: %d → %d points inside bbox %s",
        cloud.n_points,
        int(mask.sum()),
        bbox,
    )
    return _subset(cloud, np.where(mask)[0])


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _subset(cloud: PointCloud, indices: NDArray[np.intp]) -> PointCloud:
    """Return a new PointCloud containing only the points at ``indices``.

    Preserves the concrete subtype and all per-point attribute arrays.

    Parameters
    ----------
    cloud : PointCloud
        Source cloud.
    indices : NDArray[np.intp]
        Integer indices of points to keep.

    Returns
    -------
    PointCloud
        Subset cloud of the same concrete type as the input.
    """

    def _sel(arr: NDArray | None) -> NDArray | None:  # type: ignore[type-arg]
        return arr[indices] if arr is not None else None

    meta = AcquisitionMetadata(
        platform=cloud.metadata.platform,
        scanner_model=cloud.metadata.scanner_model,
        scan_date=cloud.metadata.scan_date,
        coordinate_system=cloud.metadata.coordinate_system,
        point_density_per_sqm=cloud.metadata.point_density_per_sqm,
        scan_positions=cloud.metadata.scan_positions,
        flight_altitude_m=cloud.metadata.flight_altitude_m,
        scan_angle_range=cloud.metadata.scan_angle_range,
    )

    kwargs: dict[str, object] = dict(
        intensity=_sel(cloud.intensity),
        classification=_sel(cloud.classification),
        rgb=_sel(cloud.rgb),
        normals=_sel(cloud.normals),
        return_number=_sel(cloud.return_number),
        number_of_returns=_sel(cloud.number_of_returns),
        metadata=meta,
    )
    return cloud.__class__(cloud.xyz[indices], **kwargs)  # type: ignore[arg-type]
