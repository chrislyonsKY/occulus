"""Ground classification algorithms for point clouds.

Separates ground from non-ground points using:

- **CSF** — Cloth Simulation Filter (Zhang et al. 2016). Drapes a virtual
  cloth over an inverted copy of the cloud; cloth nodes that settle close
  to cloud points are classified as ground.
- **PMF** — Progressive Morphological Filter (Zhang et al. 2003). Applies
  morphological opening with progressively growing window sizes.

Both functions return a copy of the input cloud with the ``classification``
array set (ASPRS class 2 = ground).

References
----------
- CSF: Zhang et al. (2016) "An Easy-to-Use Airborne LiDAR Data Filtering
  Method Based on Cloth Simulation." *Remote Sensing* 8(6), 501.
- PMF: Zhang et al. (2003) "A Progressive Morphological Filter for Removing
  Nonground Measurements from Airborne LIDAR Data." *IEEE TGRS* 41(4), 872–882.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusSegmentationError
from occulus.types import AcquisitionMetadata, Platform, PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "classify_ground_csf",
    "classify_ground_pmf",
]


def classify_ground_csf(
    cloud: PointCloud,
    *,
    cloth_resolution: float | None = None,
    rigidness: int = 3,
    iterations: int = 500,
    class_threshold: float | None = None,
) -> PointCloud:
    """Classify ground points using the Cloth Simulation Filter (CSF).

    The cloud is inverted along Z, a regular cloth grid is draped over it
    under gravity, and the final cloth height is compared to the cloud to
    assign ground labels (ASPRS class 2).

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have at least 10 points.
    cloth_resolution : float | None, optional
        Spacing between cloth grid nodes in coordinate units. Smaller values
        produce finer results at higher computational cost. Defaults to a
        platform-aware value (2.0 for aerial, 0.5 for terrestrial, 1.5 for UAV).
    rigidness : int, optional
        Cloth rigidness: 1 (mountain), 2 (complex terrain), 3 (flat terrain),
        by default 3. Higher values pull the cloth tighter to the surface.
    iterations : int, optional
        Maximum cloth simulation iterations, by default 500.
    class_threshold : float | None, optional
        Maximum distance between a point and the cloth surface for the point
        to be classified as ground. Defaults to ``cloth_resolution / 2``.

    Returns
    -------
    PointCloud
        Copy of the input cloud with ``classification`` array set.
        Ground points have class 2; non-ground points retain their original
        classification (or 1 = unassigned if none was present).

    Raises
    ------
    OcculusSegmentationError
        If the cloud has fewer than 10 points or the simulation produces no
        ground points.
    """
    if cloud.n_points < 10:
        raise OcculusSegmentationError(f"CSF requires at least 10 points, got {cloud.n_points}")

    # Platform-aware defaults
    if cloth_resolution is None:
        cloth_resolution = _default_cloth_resolution(cloud.platform)
    if class_threshold is None:
        class_threshold = cloth_resolution / 2.0

    logger.debug(
        "classify_ground_csf: n=%d res=%.3f rigidness=%d iters=%d",
        cloud.n_points,
        cloth_resolution,
        rigidness,
        iterations,
    )

    xyz = cloud.xyz
    # --- Step 1: Invert Z (CSF works on upside-down cloud) ---
    z_max = xyz[:, 2].max()
    inv_xyz = xyz.copy()
    inv_xyz[:, 2] = z_max - xyz[:, 2]

    # --- Step 2: Build cloth grid ---
    x_min, y_min = inv_xyz[:, 0].min(), inv_xyz[:, 1].min()
    x_max, y_max = inv_xyz[:, 0].max(), inv_xyz[:, 1].max()

    nx = max(2, int(np.ceil((x_max - x_min) / cloth_resolution)) + 1)
    ny = max(2, int(np.ceil((y_max - y_min) / cloth_resolution)) + 1)

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (ny, nx)
    cloth_z = np.full_like(grid_x, fill_value=np.inf)  # start above all points

    # --- Step 3: Project cloud onto grid (lowest Z per cell) ---
    tree_2d = KDTree(inv_xyz[:, :2])
    cloth_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    _, nn_idx = tree_2d.query(cloth_pts, k=1, workers=-1)
    nn_idx = nn_idx.ravel()
    projected_z = inv_xyz[nn_idx, 2].reshape(ny, nx)
    cloth_z = projected_z.copy()

    # --- Step 4: Cloth simulation ---
    damping = {1: 0.75, 2: 0.5, 3: 0.3}.get(rigidness, 0.5)

    for _ in range(iterations):
        z_prev = cloth_z.copy()

        # Gravity: move cloth downward toward projected terrain
        gravity_force = projected_z - cloth_z
        cloth_z += gravity_force * 0.1

        # Internal spring forces (smooth cloth between neighbours)
        # Average of 4 cardinal neighbours
        z_padded = np.pad(cloth_z, 1, mode="edge")
        z_avg = (
            (
                z_padded[:-2, 1:-1]  # up
                + z_padded[2:, 1:-1]  # down
                + z_padded[1:-1, :-2]  # left
                + z_padded[1:-1, 2:]  # right
            )
            / 4.0
        )
        cloth_z += (z_avg - cloth_z) * (1.0 - damping)

        # Hard constraint: cloth cannot go below terrain
        cloth_z = np.maximum(cloth_z, projected_z)

        # Convergence check
        if np.max(np.abs(cloth_z - z_prev)) < 1e-4:
            break

    # --- Step 5: Classify points ---
    # For each point find nearest cloth node and compare Z distance
    cloth_xy = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    cloth_z_flat = cloth_z.ravel()

    cloth_tree = KDTree(cloth_xy)
    _, cloth_nn = cloth_tree.query(inv_xyz[:, :2], k=1, workers=-1)
    cloth_nn = cloth_nn.ravel()

    # Distance in original (non-inverted) Z
    cloth_heights = cloth_z_flat[cloth_nn]  # cloth Z in inverted space
    # Convert cloth Z back to original space: cloth_Z_orig = z_max - cloth_Z_inv
    cloth_z_original = z_max - cloth_heights
    z_diff = np.abs(xyz[:, 2] - cloth_z_original)
    ground_mask = z_diff <= class_threshold

    n_ground = int(ground_mask.sum())
    if n_ground == 0:
        raise OcculusSegmentationError(
            "CSF produced no ground points. Try increasing cloth_resolution or class_threshold."
        )

    # Build output classification array
    base_class = (
        cloud.classification.copy()
        if cloud.classification is not None
        else np.ones(cloud.n_points, dtype=np.uint8)
    )
    classification = base_class.astype(np.uint8)
    classification[ground_mask] = 2

    logger.info(
        "classify_ground_csf: %d ground / %d non-ground points",
        n_ground,
        cloud.n_points - n_ground,
    )

    return _copy_with_classification(cloud, classification)


def classify_ground_pmf(
    cloud: PointCloud,
    *,
    cell_size: float = 1.0,
    slope: float = 0.3,
    initial_distance: float = 0.15,
    max_distance: float = 2.5,
    max_window_size: float = 20.0,
) -> PointCloud:
    """Classify ground points using the Progressive Morphological Filter (PMF).

    Creates a series of morphological opening operations with progressively
    growing window sizes. Points that are too far above the morphologically
    opened surface are classified as non-ground.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have at least 10 points.
    cell_size : float, optional
        Grid cell size for the terrain surface, by default 1.0.
    slope : float, optional
        Slope tolerance in metres per metre (e.g. 0.3 = 30% grade), by default 0.3.
    initial_distance : float, optional
        Initial maximum distance threshold in metres, by default 0.15.
    max_distance : float, optional
        Maximum allowed height difference above the filtered surface,
        by default 2.5.
    max_window_size : float, optional
        Maximum morphological window radius in metres, by default 20.0.

    Returns
    -------
    PointCloud
        Copy of the input cloud with ``classification`` array set.
        Ground points have class 2.

    Raises
    ------
    OcculusSegmentationError
        If the cloud has fewer than 10 points or no ground points are found.
    """
    if cloud.n_points < 10:
        raise OcculusSegmentationError(f"PMF requires at least 10 points, got {cloud.n_points}")

    xyz = cloud.xyz

    # Build 2D minimum elevation grid
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    nx = max(2, int(np.ceil((x_max - x_min) / cell_size)) + 1)
    ny = max(2, int(np.ceil((y_max - y_min) / cell_size)) + 1)

    # Assign each point to a grid cell; keep minimum Z per cell
    col_idx = np.clip(((xyz[:, 0] - x_min) / cell_size).astype(int), 0, nx - 1)
    row_idx = np.clip(((xyz[:, 1] - y_min) / cell_size).astype(int), 0, ny - 1)
    cell_flat = row_idx * nx + col_idx

    z_min_grid = np.full(ny * nx, np.inf)
    for i in range(cloud.n_points):
        z_min_grid[cell_flat[i]] = min(z_min_grid[cell_flat[i]], xyz[i, 2])

    # Replace inf cells with nearest neighbour Z
    valid = np.isfinite(z_min_grid)
    if valid.sum() < 3:
        raise OcculusSegmentationError("PMF: insufficient valid cells in grid")

    cells = np.column_stack(
        (
            np.arange(ny * nx) % nx,
            np.arange(ny * nx) // nx,
        )
    )
    tree_grid = KDTree(cells[valid])
    _, nn_grid = tree_grid.query(cells[~valid], k=1, workers=-1)
    z_min_grid[~valid] = z_min_grid[valid][nn_grid.ravel()]

    z_surface = z_min_grid.reshape(ny, nx)

    # Progressively apply morphological opening
    ground_mask = np.ones(cloud.n_points, dtype=bool)
    w = 1
    z_surface.copy()

    while w * cell_size <= max_window_size:
        # Morphological erosion then dilation (= opening) with window w
        from scipy.ndimage import maximum_filter, minimum_filter  # type: ignore[import-untyped]

        eroded = minimum_filter(z_surface, size=2 * w + 1)
        opened = maximum_filter(eroded, size=2 * w + 1)

        # Distance threshold grows with window size
        dh = initial_distance + slope * (w * cell_size)
        dh = min(dh, max_distance)

        # A point is non-ground if it exceeds opened surface by more than dh
        z_surface.ravel()[cell_flat]
        opened_pt = opened.ravel()[cell_flat]
        ground_mask &= (xyz[:, 2] - opened_pt) <= dh

        z_surface = opened
        w = min(w * 2, w + 1)  # exponential-ish growth

    n_ground = int(ground_mask.sum())
    if n_ground == 0:
        raise OcculusSegmentationError(
            "PMF produced no ground points. Try adjusting cell_size, slope, or max_distance."
        )

    base_class = (
        cloud.classification.copy()
        if cloud.classification is not None
        else np.ones(cloud.n_points, dtype=np.uint8)
    )
    classification = base_class.astype(np.uint8)
    classification[ground_mask] = 2

    logger.info(
        "classify_ground_pmf: %d ground / %d non-ground points",
        n_ground,
        cloud.n_points - n_ground,
    )

    return _copy_with_classification(cloud, classification)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _default_cloth_resolution(platform: Platform) -> float:
    """Return the default CSF cloth resolution for a given platform.

    Parameters
    ----------
    platform : Platform
        Acquisition platform.

    Returns
    -------
    float
        Recommended cloth resolution in coordinate units.
    """
    return {
        Platform.AERIAL: 2.0,
        Platform.UAV: 1.5,
        Platform.TERRESTRIAL: 0.5,
        Platform.UNKNOWN: 2.0,
    }.get(platform, 2.0)


def _copy_with_classification(
    cloud: PointCloud,
    classification: NDArray[np.uint8],
) -> PointCloud:
    """Return a copy of ``cloud`` with a new classification array.

    Parameters
    ----------
    cloud : PointCloud
        Source cloud.
    classification : NDArray[np.uint8]
        New classification array of length ``n_points``.

    Returns
    -------
    PointCloud
        Copy preserving all attributes and the concrete subtype.
    """
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
        intensity=cloud.intensity,
        classification=classification,
        rgb=cloud.rgb,
        normals=cloud.normals,
        return_number=cloud.return_number,
        number_of_returns=cloud.number_of_returns,
        metadata=meta,
    )
    return cloud.__class__(cloud.xyz, **kwargs)  # type: ignore[arg-type]
