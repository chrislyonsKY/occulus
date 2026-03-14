"""Geometric feature extraction from point clouds.

Detects and fits geometric primitives (planes, cylinders) using RANSAC,
and computes eigenvalue-based geometric descriptors for each point.

Available functions
-------------------
- :func:`detect_planes` — sequential RANSAC plane detection
- :func:`detect_cylinders` — RANSAC cylinder fitting
- :func:`compute_geometric_features` — per-point eigenvalue features

All implementations use pure NumPy and SciPy. No optional dependencies required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusFeatureError, OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "PlaneResult",
    "CylinderResult",
    "GeometricFeatures",
    "detect_planes",
    "detect_cylinders",
    "compute_geometric_features",
]


@dataclass
class PlaneResult:
    """A detected planar surface.

    Attributes
    ----------
    equation : NDArray[np.float64]
        Plane equation ``[a, b, c, d]`` where ``ax + by + cz + d = 0``.
        ``[a, b, c]`` is the unit normal.
    normal : NDArray[np.float64]
        Unit normal vector (3,).
    inlier_mask : NDArray[np.bool_]
        Boolean mask of points belonging to this plane (length = original n_points).
    n_inliers : int
        Number of inlier points.
    rmse : float
        RMS distance of inlier points to the fitted plane.
    """

    equation: NDArray[np.float64]
    normal: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    n_inliers: int
    rmse: float


@dataclass
class CylinderResult:
    """A detected cylindrical surface.

    Attributes
    ----------
    axis_point : NDArray[np.float64]
        A point on the cylinder axis (3,).
    axis_direction : NDArray[np.float64]
        Unit direction vector of the cylinder axis (3,).
    radius : float
        Fitted cylinder radius.
    inlier_mask : NDArray[np.bool_]
        Boolean mask of inlier points.
    n_inliers : int
        Number of inlier points.
    rmse : float
        RMS distance of inlier points to the cylinder surface.
    """

    axis_point: NDArray[np.float64]
    axis_direction: NDArray[np.float64]
    radius: float
    inlier_mask: NDArray[np.bool_]
    n_inliers: int
    rmse: float


@dataclass
class GeometricFeatures:
    """Per-point eigenvalue-based geometric features.

    All arrays have length ``n_points``.

    Attributes
    ----------
    linearity : NDArray[np.float64]
        ``(λ1 - λ2) / λ1`` — high for linear structures (edges, poles).
    planarity : NDArray[np.float64]
        ``(λ2 - λ3) / λ1`` — high for planar surfaces.
    sphericity : NDArray[np.float64]
        ``λ3 / λ1`` — high for volumetric / isotropic regions.
    omnivariance : NDArray[np.float64]
        ``(λ1 * λ2 * λ3) ** (1/3)`` — overall dispersion.
    anisotropy : NDArray[np.float64]
        ``(λ1 - λ3) / λ1``.
    eigenentropy : NDArray[np.float64]
        ``-sum(λi * log(λi))`` — disorder measure.
    curvature : NDArray[np.float64]
        ``λ3 / (λ1 + λ2 + λ3)`` — surface curvature estimate.
    """

    linearity: NDArray[np.float64]
    planarity: NDArray[np.float64]
    sphericity: NDArray[np.float64]
    omnivariance: NDArray[np.float64]
    anisotropy: NDArray[np.float64]
    eigenentropy: NDArray[np.float64]
    curvature: NDArray[np.float64]


def detect_planes(
    cloud: PointCloud,
    *,
    distance_threshold: float = 0.02,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    max_planes: int = 10,
    min_points: int = 100,
) -> list[PlaneResult]:
    """Detect planar surfaces using sequential RANSAC.

    Iteratively finds the largest plane, removes its inliers, and repeats
    until fewer than ``min_points`` remain or ``max_planes`` planes are found.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    distance_threshold : float, optional
        Maximum distance from a point to the plane model to count as an
        inlier, by default 0.02.
    ransac_n : int, optional
        Number of points sampled per RANSAC hypothesis, by default 3.
    num_iterations : int, optional
        Number of RANSAC iterations per plane, by default 1000.
    max_planes : int, optional
        Maximum number of planes to extract, by default 10.
    min_points : int, optional
        Stop extracting when fewer than this many points remain,
        by default 100.

    Returns
    -------
    list[PlaneResult]
        Detected planes, ordered by decreasing inlier count.

    Raises
    ------
    OcculusFeatureError
        If the cloud has too few points.
    """
    if cloud.n_points < ransac_n:
        raise OcculusFeatureError(
            f"detect_planes requires at least {ransac_n} points, got {cloud.n_points}"
        )

    xyz = cloud.xyz.copy()
    remaining_mask = np.ones(cloud.n_points, dtype=bool)
    planes: list[PlaneResult] = []

    rng = np.random.default_rng()

    while remaining_mask.sum() >= min_points and len(planes) < max_planes:
        pts = xyz[remaining_mask]
        n_remaining = len(pts)

        best_inliers = 0
        best_eq: NDArray[np.float64] = np.zeros(4)

        for _ in range(num_iterations):
            sample_idx = rng.choice(n_remaining, size=ransac_n, replace=False)
            sample = pts[sample_idx]

            # Fit plane through 3 points
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            d = -np.dot(normal, sample[0])
            eq = np.array([normal[0], normal[1], normal[2], d])

            # Signed distances
            dists = np.abs(pts @ normal + d)
            n_inliers = int((dists <= distance_threshold).sum())

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_eq = eq

        if best_inliers < min_points:
            break

        # Refit on all inliers for accurate equation
        normal_best = best_eq[:3]
        d_best = best_eq[3]
        dists_all = np.abs(pts @ normal_best + d_best)
        inlier_local = dists_all <= distance_threshold

        inlier_pts = pts[inlier_local]
        centroid = inlier_pts.mean(axis=0)
        centred = inlier_pts - centroid
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        normal_fit = Vt[-1]
        if np.dot(normal_fit, normal_best) < 0:
            normal_fit = -normal_fit
        d_fit = -np.dot(normal_fit, centroid)
        final_eq = np.array([*normal_fit, d_fit])

        dists_final = np.abs(pts @ normal_fit + d_fit)
        final_inlier_local = dists_final <= distance_threshold
        rmse = float(np.sqrt((dists_final[final_inlier_local] ** 2).mean()))

        # Map local inlier mask back to full cloud mask
        remaining_indices = np.where(remaining_mask)[0]
        full_inlier_mask = np.zeros(cloud.n_points, dtype=bool)
        full_inlier_mask[remaining_indices[final_inlier_local]] = True

        planes.append(PlaneResult(
            equation=final_eq,
            normal=normal_fit,
            inlier_mask=full_inlier_mask,
            n_inliers=int(final_inlier_local.sum()),
            rmse=rmse,
        ))

        # Remove inliers from remaining points
        remaining_mask[remaining_indices[final_inlier_local]] = False
        logger.debug(
            "detect_planes: found plane %d (%d inliers, rmse=%.4f), %d remaining",
            len(planes), int(final_inlier_local.sum()), rmse, remaining_mask.sum(),
        )

    logger.info("detect_planes: found %d planes", len(planes))
    return planes


def detect_cylinders(
    cloud: PointCloud,
    *,
    distance_threshold: float = 0.02,
    radius_range: tuple[float, float] = (0.01, 2.0),
    num_iterations: int = 5000,
    min_points: int = 100,
) -> list[CylinderResult]:
    """Detect cylindrical surfaces using RANSAC.

    Fits cylinders to point cloud data — useful for extracting pipes,
    poles, and tree trunks from TLS or UAV point clouds. Requires normals
    on the input cloud for best results.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Normals improve axis estimation significantly.
    distance_threshold : float, optional
        Maximum distance from a point to the cylinder surface to be an
        inlier, by default 0.02.
    radius_range : tuple[float, float], optional
        ``(min_radius, max_radius)`` to accept as a valid cylinder,
        by default ``(0.01, 2.0)``.
    num_iterations : int, optional
        Number of RANSAC iterations, by default 5000.
    min_points : int, optional
        Minimum inliers required for a cylinder to be reported, by default 100.

    Returns
    -------
    list[CylinderResult]
        Detected cylinders, ordered by decreasing inlier count.

    Raises
    ------
    OcculusValidationError
        If ``radius_range`` is invalid.
    OcculusFeatureError
        If the cloud has too few points.
    """
    r_min, r_max = radius_range
    if r_min >= r_max or r_min <= 0:
        raise OcculusValidationError(
            f"radius_range must be (min, max) with 0 < min < max, got {radius_range}"
        )
    if cloud.n_points < 6:
        raise OcculusFeatureError(
            f"detect_cylinders requires at least 6 points, got {cloud.n_points}"
        )

    xyz = cloud.xyz
    rng = np.random.default_rng()
    cylinders: list[CylinderResult] = []
    best_inliers = 0
    best_result: CylinderResult | None = None

    for _ in range(num_iterations):
        # Sample 2 points and use normals to estimate axis
        idx = rng.choice(cloud.n_points, size=2, replace=False)
        p1, p2 = xyz[idx[0]], xyz[idx[1]]

        if cloud.has_normals and cloud.normals is not None:
            n1, n2 = cloud.normals[idx[0]], cloud.normals[idx[1]]
            axis = np.cross(n1, n2)
        else:
            axis = p2 - p1

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            continue
        axis = axis / axis_norm

        # Project points onto plane perpendicular to axis
        proj = xyz - np.outer(xyz @ axis, axis)
        p1_proj = p1 - (p1 @ axis) * axis
        radii = np.linalg.norm(proj - p1_proj, axis=1)

        # Estimate radius as median distance from p1_proj
        radius_est = float(np.median(radii))
        if not (r_min <= radius_est <= r_max):
            continue

        dists = np.abs(radii - radius_est)
        n_inliers = int((dists <= distance_threshold).sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            inlier_mask = dists <= distance_threshold
            rmse = float(np.sqrt((dists[inlier_mask] ** 2).mean()))
            best_result = CylinderResult(
                axis_point=p1.copy(),
                axis_direction=axis.copy(),
                radius=radius_est,
                inlier_mask=inlier_mask,
                n_inliers=n_inliers,
                rmse=rmse,
            )

    if best_result is not None and best_result.n_inliers >= min_points:
        cylinders.append(best_result)

    logger.info("detect_cylinders: found %d cylinders", len(cylinders))
    return cylinders


def compute_geometric_features(
    cloud: PointCloud,
    radius: float,
    *,
    max_nn: int = 30,
) -> GeometricFeatures:
    """Compute eigenvalue-based geometric features for each point.

    For each point, PCA is performed on its local neighbourhood within
    ``radius``. The three eigenvalues (λ1 ≥ λ2 ≥ λ3) are used to derive
    seven geometric descriptors.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Must have at least 3 points.
    radius : float
        Neighbourhood search radius.
    max_nn : int, optional
        Maximum number of neighbours per point for PCA, by default 30.

    Returns
    -------
    GeometricFeatures
        Seven feature arrays of length ``n_points``.

    Raises
    ------
    OcculusValidationError
        If radius is not positive.
    OcculusFeatureError
        If the cloud has fewer than 3 points.
    """
    if radius <= 0:
        raise OcculusValidationError(f"radius must be positive, got {radius}")
    if cloud.n_points < 3:
        raise OcculusFeatureError(
            f"compute_geometric_features requires at least 3 points, got {cloud.n_points}"
        )

    xyz = cloud.xyz
    n = len(xyz)
    tree = KDTree(xyz)
    neighbour_lists = tree.query_ball_point(xyz, r=radius, workers=-1)

    linearity = np.zeros(n)
    planarity = np.zeros(n)
    sphericity = np.zeros(n)
    omnivariance = np.zeros(n)
    anisotropy = np.zeros(n)
    eigenentropy = np.zeros(n)
    curvature = np.zeros(n)

    for i, neighbours in enumerate(neighbour_lists):
        if len(neighbours) < 3:
            continue

        idx = np.array(neighbours)
        if len(idx) > max_nn:
            dists = np.linalg.norm(xyz[idx] - xyz[i], axis=1)
            idx = idx[np.argpartition(dists, max_nn)[:max_nn]]

        pts = xyz[idx]
        centred = pts - pts.mean(axis=0)
        cov = centred.T @ centred / len(pts)
        eigenvalues = np.linalg.eigvalsh(cov)  # ascending order

        # Ensure non-negative (numerical noise)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending: λ1 ≥ λ2 ≥ λ3
        l1, l2, l3 = eigenvalues
        lsum = l1 + l2 + l3

        if l1 < 1e-12:
            continue

        linearity[i] = (l1 - l2) / l1
        planarity[i] = (l2 - l3) / l1
        sphericity[i] = l3 / l1
        omnivariance[i] = (l1 * l2 * l3) ** (1.0 / 3.0) if l1 * l2 * l3 > 0 else 0.0
        anisotropy[i] = (l1 - l3) / l1

        # Eigenentropy: normalise eigenvalues before log
        if lsum > 1e-12:
            lnorm = eigenvalues / lsum
            lnorm = lnorm[lnorm > 1e-12]
            eigenentropy[i] = float(-np.sum(lnorm * np.log(lnorm)))

        curvature[i] = l3 / lsum if lsum > 1e-12 else 0.0

    logger.debug("compute_geometric_features: processed %d points (r=%.4f)", n, radius)
    return GeometricFeatures(
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=omnivariance,
        anisotropy=anisotropy,
        eigenentropy=eigenentropy,
        curvature=curvature,
    )
