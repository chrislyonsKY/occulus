"""Powerline detection — wire and pylon extraction from point clouds.

Detects power transmission and distribution lines from classified LiDAR
point clouds by combining height-above-ground filtering with geometric
feature analysis (linearity, planarity, verticality).  Wire candidates
exhibit high linearity while pylon candidates exhibit high verticality
and spatial clustering.  Optionally fits catenary curves to wire segments
and checks minimum ground clearance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator  # type: ignore[import-untyped]
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusSegmentationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "PowerlineResult",
    "detect_powerlines",
]


@dataclass
class CatenarySegment:
    """A single fitted catenary curve representing one wire span.

    Attributes
    ----------
    indices : NDArray[np.intp]
        Indices into the original point cloud for points in this segment.
    a : float
        Catenary shape parameter (lower = more sag).
    x0 : float
        Horizontal offset of the catenary vertex.
    z0 : float
        Vertical offset of the catenary vertex.
    rmse : float
        Root-mean-square error of the catenary fit in coordinate units.
    """

    indices: NDArray[np.intp]
    a: float
    x0: float
    z0: float
    rmse: float


@dataclass
class ClearanceViolation:
    """A location where a wire segment is below the minimum clearance.

    Attributes
    ----------
    point_index : int
        Index into the original point cloud.
    height_above_ground : float
        Actual height above ground at this point (coordinate units).
    min_clearance : float
        Required minimum clearance that was violated.
    """

    point_index: int
    height_above_ground: float
    min_clearance: float


@dataclass
class PowerlineResult:
    """Result of powerline detection.

    Attributes
    ----------
    wire_mask : NDArray[np.bool_]
        Per-point boolean mask identifying wire points.
    pylon_mask : NDArray[np.bool_]
        Per-point boolean mask identifying pylon/tower points.
    wire_segments : list[CatenarySegment]
        List of detected wire segments with optional catenary fits.
    pylon_positions : NDArray[np.float64]
        Centroid positions of detected pylons as (M, 3) array.
    clearance_violations : list[ClearanceViolation]
        Locations where wires are below the minimum clearance threshold.
    """

    wire_mask: NDArray[np.bool_]
    pylon_mask: NDArray[np.bool_]
    wire_segments: list[CatenarySegment] = field(default_factory=list)
    pylon_positions: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    clearance_violations: list[ClearanceViolation] = field(default_factory=list)


def detect_powerlines(
    cloud: PointCloud,
    *,
    min_height_above_ground: float = 3.0,
    max_height_above_ground: float = 50.0,
    linearity_threshold: float = 0.7,
    ground_class: int = 2,
    catenary_fit: bool = True,
    min_clearance: float | None = None,
    strict: bool = True,
    min_wire_span: float = 50.0,
    max_pylon_xy_extent: float = 5.0,
    min_pylon_z_extent: float = 8.0,
    max_wire_height_std: float = 3.0,
    pylon_association_radius: float = 10.0,
) -> PowerlineResult:
    """Detect powerline wires and pylons in a classified point cloud.

    The algorithm:

    1. Separate ground (``ground_class``) from non-ground using the
       cloud's ``classification`` attribute.
    2. Interpolate a ground surface and compute height-above-ground (HAG)
       for every non-ground point.
    3. Filter candidates by the HAG height band.
    4. Compute per-point geometric features (linearity, planarity,
       verticality) via PCA on local KDTree neighbourhoods.
    5. Classify wire candidates (high linearity) and pylon candidates
       (high verticality, low linearity, spatially clustered).
    6. Cluster wire candidates with DBSCAN into individual wire segments.
    7. Optionally fit catenary curves to each wire segment.
    8. Optionally flag clearance violations where wires are below
       ``min_clearance``.

    When ``strict=True`` (the default), additional false-positive
    reduction filters are applied:

    - **Wire segment length**: DBSCAN clusters whose 3D extent is
      shorter than ``min_wire_span`` are rejected (building edges,
      fences).
    - **Pylon geometry**: Pylon clusters whose XY bounding box exceeds
      ``max_pylon_xy_extent`` or whose Z extent is less than
      ``min_pylon_z_extent`` are rejected (buildings, tree crowns).
    - **Height band consistency**: Wire clusters whose height standard
      deviation exceeds ``max_wire_height_std`` are rejected (tree
      crowns with scattered points).
    - **Wire–pylon association**: Wire segments whose endpoints are
      not within ``pylon_association_radius`` of a detected pylon are
      downgraded (removed from the confirmed wire mask but retained
      as segments with ``rmse=inf``).

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have a ``classification`` array with at
        least some points labelled as ``ground_class``.
    min_height_above_ground : float, optional
        Minimum height above the interpolated ground surface for a point
        to be considered a powerline candidate, by default 3.0.
    max_height_above_ground : float, optional
        Maximum height above ground, by default 50.0.
    linearity_threshold : float, optional
        Minimum linearity value (0--1) for a point to be classified as a
        wire candidate, by default 0.7.
    ground_class : int, optional
        ASPRS classification code for ground points, by default 2.
    catenary_fit : bool, optional
        If ``True``, fit a catenary curve to each wire segment,
        by default ``True``.
    min_clearance : float | None, optional
        If provided, flag wire points whose HAG is below this value as
        clearance violations, by default ``None`` (no clearance check).
    strict : bool, optional
        If ``True``, enable false-positive reduction filters for wire
        span length, pylon geometry, height consistency, and wire–pylon
        association.  Set to ``False`` to use the original permissive
        behaviour, by default ``True``.
    min_wire_span : float, optional
        Minimum horizontal extent (metres) for a wire DBSCAN cluster to
        be accepted.  Only used when ``strict=True``, by default 50.0.
    max_pylon_xy_extent : float, optional
        Maximum XY bounding box size (metres) for a pylon cluster.
        Clusters wider than this are rejected as buildings.  Only used
        when ``strict=True``, by default 5.0.
    min_pylon_z_extent : float, optional
        Minimum Z extent (metres) for a pylon cluster.  Clusters shorter
        than this are rejected.  Only used when ``strict=True``,
        by default 8.0.
    max_wire_height_std : float, optional
        Maximum standard deviation (metres) of Z values within a wire
        cluster.  Clusters exceeding this are rejected.  Only used when
        ``strict=True``, by default 3.0.
    pylon_association_radius : float, optional
        Maximum distance (metres) from a wire segment endpoint to the
        nearest pylon centroid for the segment to be considered
        connected.  Only used when ``strict=True``, by default 10.0.

    Returns
    -------
    PowerlineResult
        Detection results including wire/pylon masks, segments, pylon
        positions, and optional clearance violations.

    Raises
    ------
    OcculusSegmentationError
        If the cloud has no classification array, too few ground points
        for interpolation, or no powerline candidates are found.
    """
    # --- Input validation ---------------------------------------------------
    if cloud.n_points == 0:
        raise OcculusSegmentationError("Cannot detect powerlines in an empty cloud")

    if cloud.classification is None:
        raise OcculusSegmentationError(
            "Cloud must have a classification array. "
            "Run classify_ground_csf() or classify_ground_pmf() first."
        )

    if min_height_above_ground < 0:
        raise OcculusSegmentationError(
            f"min_height_above_ground must be >= 0, got {min_height_above_ground}"
        )
    if max_height_above_ground <= min_height_above_ground:
        raise OcculusSegmentationError(
            f"max_height_above_ground ({max_height_above_ground}) must be greater "
            f"than min_height_above_ground ({min_height_above_ground})"
        )
    if not 0.0 < linearity_threshold <= 1.0:
        raise OcculusSegmentationError(
            f"linearity_threshold must be in (0, 1], got {linearity_threshold}"
        )

    # --- Step 1: Separate ground / non-ground --------------------------------
    ground_mask = cloud.classification == ground_class
    n_ground = int(ground_mask.sum())

    if n_ground < 3:
        raise OcculusSegmentationError(
            f"Need at least 3 ground points for surface interpolation, "
            f"found {n_ground} with class={ground_class}"
        )

    non_ground_mask = ~ground_mask
    n_non_ground = int(non_ground_mask.sum())

    if n_non_ground == 0:
        raise OcculusSegmentationError(
            "All points are classified as ground; no candidates for powerline detection"
        )

    logger.debug(
        "Ground separation: %d ground, %d non-ground points",
        n_ground,
        n_non_ground,
    )

    # --- Step 2: Compute height above ground ---------------------------------
    ground_xyz = cloud.xyz[ground_mask]
    hag = _compute_height_above_ground(cloud.xyz, ground_xyz)

    # --- Step 3: Filter by height band ---------------------------------------
    non_ground_indices = np.where(non_ground_mask)[0]
    height_band = (hag[non_ground_indices] >= min_height_above_ground) & (
        hag[non_ground_indices] <= max_height_above_ground
    )
    candidate_indices = non_ground_indices[height_band]

    if len(candidate_indices) == 0:
        raise OcculusSegmentationError(
            f"No points found in height band [{min_height_above_ground}, "
            f"{max_height_above_ground}] above ground"
        )

    logger.debug(
        "Height filter: %d candidates in [%.1f, %.1f] m above ground",
        len(candidate_indices),
        min_height_above_ground,
        max_height_above_ground,
    )

    # --- Step 4: Compute geometric features ----------------------------------
    candidate_xyz = cloud.xyz[candidate_indices]
    linearity, planarity, verticality = _compute_geometric_features(
        candidate_xyz, k=min(20, len(candidate_indices))
    )

    # --- Step 5: Wire and pylon classification -------------------------------
    wire_local = linearity >= linearity_threshold
    pylon_local = (verticality >= 0.6) & (linearity < linearity_threshold * 0.5)

    wire_indices = candidate_indices[wire_local]
    pylon_candidate_indices = candidate_indices[pylon_local]

    logger.debug(
        "Feature classification: %d wire candidates, %d pylon candidates",
        len(wire_indices),
        len(pylon_candidate_indices),
    )

    # --- Step 6: Cluster wires with DBSCAN -----------------------------------
    wire_segments: list[CatenarySegment] = []
    wire_mask = np.zeros(cloud.n_points, dtype=bool)

    if len(wire_indices) > 0:
        wire_labels = _dbscan_cluster(cloud.xyz[wire_indices], eps=3.0, min_samples=5)

        unique_labels = set(wire_labels) - {-1}
        n_rejected_span = 0
        n_rejected_height = 0
        for lbl in sorted(unique_labels):
            seg_local = np.where(wire_labels == lbl)[0]
            seg_global = wire_indices[seg_local]
            seg_xyz = cloud.xyz[seg_global]

            # --- Strict filter: minimum wire span length ---
            if strict:
                xy_extent = seg_xyz[:, :2].max(axis=0) - seg_xyz[:, :2].min(axis=0)
                span = float(np.linalg.norm(xy_extent))
                if span < min_wire_span:
                    n_rejected_span += 1
                    continue

            # --- Strict filter: height band consistency ---
            if strict:
                z_std = float(seg_xyz[:, 2].std())
                if z_std > max_wire_height_std:
                    n_rejected_height += 1
                    continue

            wire_mask[seg_global] = True
            seg = CatenarySegment(
                indices=seg_global,
                a=0.0,
                x0=0.0,
                z0=0.0,
                rmse=float("inf"),
            )
            wire_segments.append(seg)

        if strict and (n_rejected_span > 0 or n_rejected_height > 0):
            logger.debug(
                "Strict wire filtering: rejected %d short segments, %d high-variance segments",
                n_rejected_span,
                n_rejected_height,
            )

        if not strict:
            # In permissive mode, mark all wire candidates (original behaviour)
            wire_mask[wire_indices] = True

        logger.info(
            "Wire clustering: %d wire segments from %d wire points",
            len(wire_segments),
            int(wire_mask.sum()),
        )

    # --- Step 6b: Cluster pylons ---------------------------------------------
    pylon_mask = np.zeros(cloud.n_points, dtype=bool)
    pylon_positions = np.empty((0, 3), dtype=np.float64)

    if len(pylon_candidate_indices) > 0:
        pylon_labels = _dbscan_cluster(cloud.xyz[pylon_candidate_indices], eps=5.0, min_samples=3)

        pylon_unique = set(pylon_labels) - {-1}
        if pylon_unique:
            centroids = []
            accepted_indices: list[NDArray[np.intp]] = []
            n_rejected_wide = 0
            n_rejected_short = 0
            for lbl in sorted(pylon_unique):
                seg_local = np.where(pylon_labels == lbl)[0]
                seg_global = pylon_candidate_indices[seg_local]
                seg_xyz = cloud.xyz[seg_global]

                if strict:
                    # Reject clusters with wide XY footprint (buildings)
                    xy_min = seg_xyz[:, :2].min(axis=0)
                    xy_max = seg_xyz[:, :2].max(axis=0)
                    xy_extent = xy_max - xy_min
                    if float(xy_extent.max()) > max_pylon_xy_extent:
                        n_rejected_wide += 1
                        continue

                    # Reject clusters that are not tall enough
                    z_extent = float(seg_xyz[:, 2].max() - seg_xyz[:, 2].min())
                    if z_extent < min_pylon_z_extent:
                        n_rejected_short += 1
                        continue

                centroid = seg_xyz.mean(axis=0)
                centroids.append(centroid)
                accepted_indices.append(seg_global)

            if centroids:
                pylon_positions = np.array(centroids, dtype=np.float64)
                for idx_arr in accepted_indices:
                    pylon_mask[idx_arr] = True

            if strict and (n_rejected_wide > 0 or n_rejected_short > 0):
                logger.debug(
                    "Strict pylon filtering: rejected %d wide clusters, %d short clusters",
                    n_rejected_wide,
                    n_rejected_short,
                )
        else:
            # No pylon clusters found — mark all candidates in permissive mode
            if not strict:
                pylon_mask[pylon_candidate_indices] = True

        if not strict:
            # In permissive mode, mark all pylon candidates (original behaviour)
            pylon_mask[pylon_candidate_indices] = True

        logger.info(
            "Pylon clustering: %d pylons detected",
            len(pylon_positions),
        )

    # --- Step 6c: Wire-pylon association (strict mode) -----------------------
    if strict and len(pylon_positions) >= 2 and wire_segments:
        wire_segments, wire_mask = _filter_orphan_wires(
            cloud.xyz,
            wire_segments,
            wire_mask,
            pylon_positions,
            pylon_association_radius,
        )

    # --- Step 7: Catenary fitting --------------------------------------------
    if catenary_fit and wire_segments:
        wire_segments = _fit_catenaries(cloud.xyz, wire_segments)

    # --- Step 8: Clearance analysis ------------------------------------------
    clearance_violations: list[ClearanceViolation] = []
    confirmed_wire_indices = np.where(wire_mask)[0]
    if min_clearance is not None and len(confirmed_wire_indices) > 0:
        for idx in confirmed_wire_indices:
            h = hag[idx]
            if h < min_clearance:
                clearance_violations.append(
                    ClearanceViolation(
                        point_index=int(idx),
                        height_above_ground=float(h),
                        min_clearance=min_clearance,
                    )
                )
        if clearance_violations:
            logger.warning(
                "Clearance analysis: %d violations (min_clearance=%.1f)",
                len(clearance_violations),
                min_clearance,
            )

    return PowerlineResult(
        wire_mask=wire_mask,
        pylon_mask=pylon_mask,
        wire_segments=wire_segments,
        pylon_positions=pylon_positions,
        clearance_violations=clearance_violations,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_orphan_wires(
    xyz: NDArray[np.float64],
    segments: list[CatenarySegment],
    wire_mask: NDArray[np.bool_],
    pylon_positions: NDArray[np.float64],
    radius: float,
) -> tuple[list[CatenarySegment], NDArray[np.bool_]]:
    """Remove wire segments whose endpoints are not near any pylon.

    A wire segment is considered "connected" if both its start-end
    extremes (the points with minimum and maximum projection along
    the segment's principal horizontal axis) are within ``radius``
    of at least one pylon centroid (XY distance).

    Orphan segments are removed from the confirmed wire mask but kept
    in the segment list with ``rmse`` set to ``inf`` so callers can
    inspect them as candidates.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Full point cloud coordinates (N, 3).
    segments : list[CatenarySegment]
        Detected wire segments.
    wire_mask : NDArray[np.bool_]
        Current wire point mask (modified in-place and returned).
    pylon_positions : NDArray[np.float64]
        Pylon centroid positions (M, 3).
    radius : float
        Maximum XY distance from an endpoint to a pylon centroid.

    Returns
    -------
    tuple[list[CatenarySegment], NDArray[np.bool_]]
        Filtered segment list and updated wire mask.
    """
    pylon_tree = KDTree(pylon_positions[:, :2])
    confirmed: list[CatenarySegment] = []
    n_orphan = 0

    for seg in segments:
        pts = xyz[seg.indices]
        if len(pts) < 2:
            # Cannot determine endpoints — treat as orphan
            wire_mask[seg.indices] = False
            n_orphan += 1
            continue

        # Project onto principal horizontal axis to find endpoints
        xy = pts[:, :2]
        centroid_xy = xy.mean(axis=0)
        xy_centered = xy - centroid_xy
        try:
            cov = np.cov(xy_centered, rowvar=False)
            _, evecs = np.linalg.eigh(cov)
            principal = evecs[:, -1]
        except np.linalg.LinAlgError:
            wire_mask[seg.indices] = False
            n_orphan += 1
            continue

        t = xy_centered @ principal
        start_xy = xy[np.argmin(t)]
        end_xy = xy[np.argmax(t)]

        # Check proximity to pylons (XY only)
        d_start, _ = pylon_tree.query(start_xy)
        d_end, _ = pylon_tree.query(end_xy)

        if d_start <= radius and d_end <= radius:
            confirmed.append(seg)
        else:
            # Downgrade: remove from mask, keep segment as candidate
            wire_mask[seg.indices] = False
            n_orphan += 1

    if n_orphan > 0:
        logger.debug(
            "Wire-pylon association: %d orphan segments removed, %d confirmed",
            n_orphan,
            len(confirmed),
        )

    return confirmed, wire_mask


def _compute_height_above_ground(
    xyz: NDArray[np.float64],
    ground_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute height above ground for every point by interpolating ground surface.

    Uses Delaunay-based linear interpolation of ground point elevations.
    Points outside the convex hull of the ground use nearest-neighbour
    ground elevation as fallback.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        All point positions (N, 3).
    ground_xyz : NDArray[np.float64]
        Ground point positions (M, 3).

    Returns
    -------
    NDArray[np.float64]
        Height above ground for each point (N,).
    """
    try:
        interp = LinearNDInterpolator(ground_xyz[:, :2], ground_xyz[:, 2])
        ground_z = interp(xyz[:, :2])
    except Exception as exc:
        raise OcculusSegmentationError(f"Ground surface interpolation failed: {exc}") from exc

    # Fill NaN (outside convex hull) with nearest-neighbour ground elevation
    nan_mask = np.isnan(ground_z)
    if nan_mask.any():
        ground_tree = KDTree(ground_xyz[:, :2])
        _, nn_idx = ground_tree.query(xyz[nan_mask, :2])
        ground_z[nan_mask] = ground_xyz[nn_idx, 2]

    hag = xyz[:, 2] - ground_z
    return hag.astype(np.float64)


def _compute_geometric_features(
    pts: NDArray[np.float64],
    k: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute linearity, planarity, and verticality via PCA on local neighbourhoods.

    For each point, the *k* nearest neighbours are used to form a
    3x3 covariance matrix.  The eigenvalues (lambda1 >= lambda2 >= lambda3)
    yield:

    - linearity  = (lambda1 - lambda2) / lambda1
    - planarity  = (lambda2 - lambda3) / lambda1
    - verticality = 1 - |e3 . [0,0,1]|  where e3 is the normal eigenvector

    Parameters
    ----------
    pts : NDArray[np.float64]
        Points (N, 3).
    k : int, optional
        Number of nearest neighbours for PCA, by default 20.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        linearity, planarity, verticality — each (N,) array in [0, 1].
    """
    n = len(pts)
    if n < 2:
        return (
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            np.zeros(n, dtype=np.float64),
        )

    k = min(k, n)
    tree = KDTree(pts)
    _, nn_indices = tree.query(pts, k=k)

    # Ensure nn_indices is 2D even for k=1
    if nn_indices.ndim == 1:
        nn_indices = nn_indices[:, np.newaxis]

    linearity = np.zeros(n, dtype=np.float64)
    planarity = np.zeros(n, dtype=np.float64)
    verticality = np.zeros(n, dtype=np.float64)

    up = np.array([0.0, 0.0, 1.0])

    for i in range(n):
        neighbours = pts[nn_indices[i]]
        cov = np.cov(neighbours, rowvar=False)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        l1, l2, l3 = eigenvalues
        if l1 <= 0:
            continue

        linearity[i] = (l1 - l2) / l1
        planarity[i] = (l2 - l3) / l1

        # Verticality: how aligned is the smallest eigenvector with vertical
        e3 = eigenvectors[:, 2]  # normal direction (smallest eigenvalue)
        verticality[i] = 1.0 - abs(float(np.dot(e3, up)))

    return linearity, planarity, verticality


def _dbscan_cluster(
    pts: NDArray[np.float64],
    eps: float,
    min_samples: int,
) -> NDArray[np.int32]:
    """DBSCAN clustering using scipy KDTree.

    Parameters
    ----------
    pts : NDArray[np.float64]
        Points to cluster (N, D).
    eps : float
        Neighbourhood radius.
    min_samples : int
        Minimum neighbours for a core point.

    Returns
    -------
    NDArray[np.int32]
        Cluster labels; -1 = noise.
    """
    n = len(pts)
    if n == 0:
        return np.array([], dtype=np.int32)

    tree = KDTree(pts)
    neighbour_lists = tree.query_ball_point(pts, r=eps, workers=-1)

    labels = np.full(n, -1, dtype=np.int32)
    is_core = np.array([len(nb) >= min_samples for nb in neighbour_lists])
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue

        labels[i] = cluster_id
        queue = list(neighbour_lists[i])
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                if is_core[j]:
                    queue.extend(neighbour_lists[j])

        cluster_id += 1

    return labels


def _catenary(
    x: NDArray[np.float64],
    a: float,
    x0: float,
    z0: float,
) -> NDArray[np.float64]:
    """Catenary curve: z = a * cosh((x - x0) / a) + z0.

    Parameters
    ----------
    x : NDArray[np.float64]
        Horizontal positions.
    a : float
        Shape parameter (sag).
    x0 : float
        Horizontal offset of vertex.
    z0 : float
        Vertical offset of vertex.

    Returns
    -------
    NDArray[np.float64]
        Predicted z values.
    """
    return a * np.cosh((x - x0) / a) + z0


def _fit_catenaries(
    xyz: NDArray[np.float64],
    segments: list[CatenarySegment],
) -> list[CatenarySegment]:
    """Fit catenary curves to wire segments.

    Projects each segment onto its principal axis (PCA) then fits
    z = a * cosh((t - x0) / a) + z0 along that axis.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Full point cloud coordinates (N, 3).
    segments : list[CatenarySegment]
        Wire segments to fit.

    Returns
    -------
    list[CatenarySegment]
        Segments with updated catenary parameters and RMSE.
    """
    fitted: list[CatenarySegment] = []

    for seg in segments:
        if len(seg.indices) < 5:
            fitted.append(seg)
            continue

        pts = xyz[seg.indices]

        # Project onto principal horizontal axis via PCA of XY
        xy_centered = pts[:, :2] - pts[:, :2].mean(axis=0)
        try:
            cov_xy = np.cov(xy_centered, rowvar=False)
            evals, evecs = np.linalg.eigh(cov_xy)
        except np.linalg.LinAlgError:
            fitted.append(seg)
            continue

        principal = evecs[:, np.argmax(evals)]
        t = xy_centered @ principal  # parametric distance along wire

        z = pts[:, 2]
        z.max() - z.min()
        t_range = t.max() - t.min()

        if t_range < 1e-6:
            fitted.append(seg)
            continue

        # Initial guesses
        a0 = max(t_range, 1.0)
        x0_0 = float(np.median(t))
        z0_0 = float(z.min()) - a0

        try:
            popt, _ = curve_fit(
                _catenary,
                t,
                z,
                p0=[a0, x0_0, z0_0],
                maxfev=5000,
            )
            z_pred = _catenary(t, *popt)
            rmse = float(np.sqrt(np.mean((z - z_pred) ** 2)))

            fitted.append(
                CatenarySegment(
                    indices=seg.indices,
                    a=float(popt[0]),
                    x0=float(popt[1]),
                    z0=float(popt[2]),
                    rmse=rmse,
                )
            )
            logger.debug(
                "Catenary fit: a=%.2f, rmse=%.3f for segment with %d points",
                popt[0],
                rmse,
                len(seg.indices),
            )
        except (RuntimeError, ValueError) as exc:
            logger.debug("Catenary fit failed for segment: %s", exc)
            fitted.append(seg)

    return fitted
