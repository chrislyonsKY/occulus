"""Object segmentation — extract individual objects from point clouds.

Provides general 3D DBSCAN clustering and forestry-specific tree
segmentation using a Canopy Height Model (CHM) watershed approach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusSegmentationError, UnsupportedPlatformError
from occulus.types import Platform, PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "SegmentationResult",
    "cluster_dbscan",
    "segment_trees",
]


@dataclass
class SegmentationResult:
    """Result of object segmentation.

    Attributes
    ----------
    labels : NDArray[np.int32]
        Per-point segment labels. ``-1`` = noise / unassigned.
    n_segments : int
        Number of unique segments (excluding noise label -1).
    segment_sizes : dict[int, int]
        Mapping of label → point count (noise label excluded).
    """

    labels: NDArray[np.int32]
    n_segments: int
    segment_sizes: dict[int, int] = field(default_factory=dict)


def cluster_dbscan(
    cloud: PointCloud,
    eps: float,
    min_samples: int = 10,
    *,
    use_2d: bool = False,
) -> SegmentationResult:
    """Cluster a point cloud using DBSCAN.

    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    groups points that are closely packed together and marks outliers as
    noise (label -1). No prior knowledge of the number of clusters is needed.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    eps : float
        Neighbourhood radius: maximum distance between two points for them
        to be considered neighbours. In the same units as the cloud coordinates.
    min_samples : int, optional
        Minimum number of points in a neighbourhood for a core point,
        by default 10.
    use_2d : bool, optional
        If ``True``, clustering is performed in the XY plane only (useful
        for separating vertical objects like trees above ground), by default
        ``False``.

    Returns
    -------
    SegmentationResult
        Per-point labels and cluster statistics.

    Raises
    ------
    OcculusSegmentationError
        If eps or min_samples are invalid, or the cloud is empty.
    """
    if eps <= 0:
        raise OcculusSegmentationError(f"eps must be positive, got {eps}")
    if min_samples <= 0:
        raise OcculusSegmentationError(f"min_samples must be positive, got {min_samples}")
    if cloud.n_points == 0:
        raise OcculusSegmentationError("Cannot cluster an empty cloud")

    pts = cloud.xyz[:, :2] if use_2d else cloud.xyz
    labels = _dbscan(pts, eps=eps, min_samples=min_samples)

    unique_labels = set(labels) - {-1}
    n_segments = len(unique_labels)
    segment_sizes = {int(lbl): int((labels == lbl).sum()) for lbl in unique_labels}

    logger.info(
        "cluster_dbscan: %d clusters, %d noise points (eps=%.3f, min_samples=%d)",
        n_segments,
        int((labels == -1).sum()),
        eps,
        min_samples,
    )

    return SegmentationResult(
        labels=labels.astype(np.int32),
        n_segments=n_segments,
        segment_sizes=segment_sizes,
    )


def segment_trees(
    cloud: PointCloud,
    *,
    resolution: float = 1.0,
    min_height: float = 2.0,
    min_crown_area: float = 4.0,
    max_raster_size: int = 5000,
) -> SegmentationResult:
    """Segment individual trees using a CHM-based watershed approach.

    Builds a 2D Canopy Height Model (CHM) raster from the input cloud,
    detects local maxima as tree tops, then performs a marker-controlled
    watershed segmentation to delineate individual tree crowns.

    This function is intended for aerial or UAV clouds. Terrestrial clouds
    will raise an error.

    Parameters
    ----------
    cloud : PointCloud
        Input cloud. Should be non-ground points (vegetation only) for
        best results. Must be aerial or UAV platform.
    resolution : float, optional
        CHM raster resolution in coordinate units, by default 1.0.
    min_height : float, optional
        Minimum tree height above the minimum Z of the cloud, by default 2.0.
    min_crown_area : float, optional
        Minimum crown area in square coordinate units to retain a tree
        segment, by default 4.0.
    max_raster_size : int, optional
        Maximum allowed dimension (width or height) for the CHM raster.
        If the computed raster would exceed this in either dimension, the
        resolution is automatically coarsened to fit. By default 5000.

    Returns
    -------
    SegmentationResult
        Per-point tree labels. Label -1 = not assigned to any tree.

    Raises
    ------
    UnsupportedPlatformError
        If the cloud platform is ``TERRESTRIAL``.
    OcculusSegmentationError
        If no trees are detected.
    """
    if cloud.platform == Platform.TERRESTRIAL:
        raise UnsupportedPlatformError(
            "segment_trees is intended for aerial/UAV clouds, not terrestrial scans. "
            "Use cluster_dbscan() for TLS object segmentation."
        )
    if cloud.n_points == 0:
        raise OcculusSegmentationError("Cannot segment trees from an empty cloud")

    from scipy.ndimage import label as nd_label  # type: ignore[import-untyped]
    from scipy.ndimage import maximum_filter, watershed_ift

    xyz = cloud.xyz
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()
    z_min = xyz[:, 2].min()

    nx = max(2, int(np.ceil((x_max - x_min) / resolution)) + 1)
    ny = max(2, int(np.ceil((y_max - y_min) / resolution)) + 1)

    # Auto-coarsen resolution if raster would exceed max_raster_size
    max_dim = max(nx, ny)
    if max_dim > max_raster_size:
        old_resolution = resolution
        resolution = resolution * (max_dim / max_raster_size)
        nx = max(2, int(np.ceil((x_max - x_min) / resolution)) + 1)
        ny = max(2, int(np.ceil((y_max - y_min) / resolution)) + 1)
        logger.warning(
            "CHM raster %dx%d exceeds max_raster_size=%d, coarsening resolution from %.4f to %.4f",
            max(2, int(np.ceil((x_max - x_min) / old_resolution)) + 1),
            max(2, int(np.ceil((y_max - y_min) / old_resolution)) + 1),
            max_raster_size,
            old_resolution,
            resolution,
        )

    col_idx = np.clip(((xyz[:, 0] - x_min) / resolution).astype(int), 0, nx - 1)
    row_idx = np.clip(((xyz[:, 1] - y_min) / resolution).astype(int), 0, ny - 1)

    # Build max-Z raster (CHM) — vectorised via np.maximum.at
    heights = xyz[:, 2] - z_min
    chm = np.zeros((ny, nx), dtype=np.float64)
    flat_idx = row_idx * nx + col_idx
    np.maximum.at(chm.ravel(), flat_idx, heights)

    # Detect local maxima (tree tops) via morphological maximum filter
    neighborhood = max(3, int(np.ceil(min_crown_area**0.5 / resolution)) | 1)
    local_max = maximum_filter(chm, size=neighborhood)
    treetop_mask = (chm == local_max) & (chm >= min_height)

    # Label connected components of tree tops
    treetop_labels, n_tops = nd_label(treetop_mask)

    if n_tops == 0:
        raise OcculusSegmentationError(
            f"No trees detected (min_height={min_height}). Try lowering min_height or resolution."
        )

    # Marker-controlled watershed via scipy watershed_ift.
    # watershed_ift expects a uint8/uint16 cost surface and int32 markers.
    # We invert the CHM so that high canopy = low cost (seeds expand from peaks).
    chm_max = chm.max()
    if chm_max > 0:
        cost = ((1.0 - chm / chm_max) * 65534).astype(np.uint16)
    else:
        cost = np.zeros_like(chm, dtype=np.uint16)
    labels_grid = watershed_ift(cost, treetop_labels.astype(np.int32))

    # Map each point to its tree label — vectorised
    height_mask = heights >= min_height
    grid_labels = labels_grid[row_idx, col_idx] - 1  # 0 in grid → -1 (unlabelled)
    pt_labels = np.where(height_mask, grid_labels, np.int32(-1)).astype(np.int32)

    # Prune tiny segments
    unique_labels, counts = np.unique(pt_labels[pt_labels >= 0], return_counts=True)
    min_count = min_crown_area / (resolution**2)
    segment_sizes: dict[int, int] = {}
    small_labels = set()
    for lbl, count in zip(unique_labels, counts, strict=False):
        if count >= min_count:
            segment_sizes[int(lbl)] = int(count)
        else:
            small_labels.add(int(lbl))
    if small_labels:
        small_mask = np.isin(pt_labels, list(small_labels))
        pt_labels[small_mask] = -1

    n_segments = len(segment_sizes)
    logger.info(
        "segment_trees: %d trees detected from %d treetops (res=%.2f)",
        n_segments,
        n_tops,
        resolution,
    )

    return SegmentationResult(
        labels=pt_labels,
        n_segments=n_segments,
        segment_sizes=segment_sizes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dbscan(
    pts: NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> NDArray[np.int32]:
    """Pure-NumPy/SciPy DBSCAN implementation.

    Parameters
    ----------
    pts : NDArray[np.float64]
        Input points (N, D).
    eps : float
        Neighbourhood radius.
    min_samples : int
        Minimum neighbourhood size for a core point.

    Returns
    -------
    NDArray[np.int32]
        Labels array; -1 = noise.
    """
    n = len(pts)
    tree = KDTree(pts)
    neighbour_lists = tree.query_ball_point(pts, r=eps, workers=-1)

    labels = np.full(n, -1, dtype=np.int32)
    is_core = np.array([len(nb) >= min_samples for nb in neighbour_lists])
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue

        # BFS to expand cluster
        labels[i] = cluster_id
        queue = list(neighbour_lists[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                if is_core[j]:
                    queue.extend(neighbour_lists[j])
            elif labels[j] == -2:  # border point, reassign
                labels[j] = cluster_id

        cluster_id += 1

    return labels
