"""Global registration — coarse alignment via FPFH features and RANSAC.

Use :func:`ransac_registration` to get an initial transformation estimate
before refining with ICP. This module also provides :func:`compute_fpfh`
for computing Fast Point Feature Histogram descriptors.

For multi-scan alignment, use :func:`align_scans`.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusRegistrationError, OcculusValidationError
from occulus.registration.icp import RegistrationResult, _apply_transform, _compute_metrics
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "AlignmentResult",
    "align_scans",
    "compute_fpfh",
    "ransac_registration",
]


# ---------------------------------------------------------------------------
# FPFH descriptors
# ---------------------------------------------------------------------------


def compute_fpfh(
    cloud: PointCloud,
    radius: float,
    *,
    max_nn: int = 100,
) -> NDArray[np.float64]:
    """Compute Fast Point Feature Histogram (FPFH) descriptors.

    FPFH is a 33-dimensional descriptor that encodes the local geometry
    around each point using angular features derived from surface normals.
    Requires normals on the input cloud.

    Parameters
    ----------
    cloud : PointCloud
        Input cloud. Must have normals (``cloud.has_normals`` must be ``True``).
    radius : float
        Neighbourhood search radius for feature computation.
    max_nn : int, optional
        Maximum number of neighbours per point, by default 100.

    Returns
    -------
    NDArray[np.float64]
        FPFH descriptor array of shape (N, 33).

    Raises
    ------
    OcculusValidationError
        If the cloud has no normals.
    """
    if not cloud.has_normals:
        raise OcculusValidationError(
            "compute_fpfh requires normals. Run occulus.normals.estimate_normals() first."
        )
    assert cloud.normals is not None

    xyz = cloud.xyz
    normals = cloud.normals
    n = len(xyz)
    tree = KDTree(xyz)

    # Simplified Point Feature Histogram (SPFH) for each point
    spfh = np.zeros((n, 33), dtype=np.float64)
    neighbour_lists = tree.query_ball_point(xyz, r=radius, workers=-1)

    for i, neighbours in enumerate(neighbour_lists):
        if len(neighbours) < 2:
            continue

        idx = np.array(neighbours)
        if len(idx) > max_nn:
            dists = np.linalg.norm(xyz[idx] - xyz[i], axis=1)
            idx = idx[np.argpartition(dists, max_nn)[:max_nn]]

        # Exclude self
        idx = idx[idx != i]
        if len(idx) == 0:
            continue

        # Darboux frame for each neighbour pair (i, j)
        u = normals[i]  # (3,)
        diff = xyz[idx] - xyz[i]  # (k, 3)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.where(dist < 1e-12, 1.0, dist)
        v_vecs = diff / dist  # (k, 3)
        w_vecs = np.cross(u, v_vecs)  # (k, 3)

        # Three angles: alpha, phi, theta
        nj = normals[idx]  # (k, 3)
        alpha = np.einsum("ij,ij->i", w_vecs, nj)  # (k,)
        phi = np.einsum("ij,j->i", v_vecs, u)  # (k,) — u is (3,)
        theta = np.arctan2(
            np.einsum("ij,ij->i", np.cross(u, v_vecs), nj),
            np.einsum("ij,ij->i", v_vecs, nj),
        )  # (k,)

        # Bin into 11 bins each (→ 33-dim descriptor)
        bins = 11
        spfh[i, :bins], _ = np.histogram(alpha, bins=bins, range=(-1.0, 1.0))
        spfh[i, bins : 2 * bins], _ = np.histogram(phi, bins=bins, range=(-1.0, 1.0))
        spfh[i, 2 * bins :], _ = np.histogram(theta, bins=bins, range=(-np.pi, np.pi))

        # Normalise
        row_sum = spfh[i].sum()
        if row_sum > 0:
            spfh[i] /= row_sum

    # FPFH = SPFH + weighted sum of SPFH of neighbours
    fpfh = spfh.copy()
    for i, neighbours in enumerate(neighbour_lists):
        idx = np.array(neighbours)
        idx = idx[idx != i]
        if len(idx) == 0:
            continue
        dists = np.linalg.norm(xyz[idx] - xyz[i], axis=1)
        dists = np.where(dists < 1e-12, 1e-12, dists)
        weights = 1.0 / dists
        fpfh[i] += (weights[:, None] * spfh[idx]).sum(axis=0) / weights.sum()

        row_sum = fpfh[i].sum()
        if row_sum > 0:
            fpfh[i] /= row_sum

    logger.debug("compute_fpfh: computed 33-dim descriptors for %d points", n)
    return fpfh


# ---------------------------------------------------------------------------
# RANSAC global registration
# ---------------------------------------------------------------------------


def ransac_registration(
    source: PointCloud,
    target: PointCloud,
    source_features: NDArray[np.float64],
    target_features: NDArray[np.float64],
    *,
    max_correspondence_distance: float = 1.5,
    ransac_n: int = 3,
    max_iterations: int = 100_000,
    confidence: float = 0.999,
) -> RegistrationResult:
    """Global registration via feature-matching and RANSAC.

    Matches FPFH descriptors between source and target to find feature
    correspondences, then uses RANSAC to robustly estimate the rigid
    transformation.

    Parameters
    ----------
    source : PointCloud
        Source cloud.
    target : PointCloud
        Target cloud.
    source_features : NDArray[np.float64]
        FPFH descriptors for source cloud (N, 33).
    target_features : NDArray[np.float64]
        FPFH descriptors for target cloud (M, 33).
    max_correspondence_distance : float, optional
        Maximum Euclidean distance for an inlier correspondence,
        by default 1.5.
    ransac_n : int, optional
        Number of random correspondences per RANSAC hypothesis, by default 3.
    max_iterations : int, optional
        Maximum RANSAC iterations, by default 100_000.
    confidence : float, optional
        Desired success probability for RANSAC, used for adaptive early exit,
        by default 0.999.

    Returns
    -------
    RegistrationResult
        Best transformation found. ``converged=True`` if the inlier ratio
        exceeds the RANSAC confidence criterion.

    Raises
    ------
    OcculusValidationError
        If feature array shapes do not match their clouds.
    OcculusRegistrationError
        If no valid transformation is found.
    """
    if source_features.shape[0] != source.n_points:
        raise OcculusValidationError(
            f"source_features rows ({source_features.shape[0]}) != "
            f"source n_points ({source.n_points})"
        )
    if target_features.shape[0] != target.n_points:
        raise OcculusValidationError(
            f"target_features rows ({target_features.shape[0]}) != "
            f"target n_points ({target.n_points})"
        )

    # Build a KD-tree on target feature space to find descriptor matches
    feat_tree = KDTree(target_features)
    _distances, target_idx = feat_tree.query(source_features, k=1, workers=-1)
    target_idx = target_idx.ravel()

    # All candidate correspondences (source_i → target_idx[i])
    corr_src = np.arange(source.n_points)
    corr_tgt = target_idx

    rng = np.random.default_rng()
    best_transform = np.eye(4, dtype=np.float64)
    best_inliers = 0
    n_corr = len(corr_src)

    if n_corr < ransac_n:
        raise OcculusRegistrationError(
            f"Too few correspondences ({n_corr}) for RANSAC (need {ransac_n})"
        )

    # Adaptive RANSAC iteration count
    inlier_ratio = 1.0
    max_adaptive_iter = max_iterations

    for iteration in range(max_adaptive_iter):  # noqa: B007
        # Sample ransac_n correspondences
        sample = rng.choice(n_corr, size=ransac_n, replace=False)
        src_pts = source.xyz[corr_src[sample]]
        tgt_pts = target.xyz[corr_tgt[sample]]

        # Estimate transform from minimal sample
        try:
            R, t = _svd_rigid_3pt(src_pts, tgt_pts)
        except Exception:
            continue

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        # Count inliers over all correspondences
        src_transformed = _apply_transform(source.xyz[corr_src], T)
        tgt_pts_all = target.xyz[corr_tgt]
        dists = np.linalg.norm(src_transformed - tgt_pts_all, axis=1)
        inlier_count = int((dists <= max_correspondence_distance).sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_transform = T
            inlier_ratio = best_inliers / n_corr

            # Adaptive early exit (Hartley & Zisserman)
            max(1e-9, 1.0 - inlier_ratio)
            k_adaptive = np.log(1 - confidence) / np.log(1 - (inlier_ratio**ransac_n) + 1e-12)
            max_adaptive_iter = min(max_iterations, int(k_adaptive) + 1)

    if best_inliers == 0:
        raise OcculusRegistrationError("RANSAC failed to find any valid transformation.")

    # Refine on all inliers
    src_t = _apply_transform(source.xyz[corr_src], best_transform)
    tgt_all = target.xyz[corr_tgt]
    dists = np.linalg.norm(src_t - tgt_all, axis=1)
    inlier_mask = dists <= max_correspondence_distance

    src_inliers = source.xyz[corr_src[inlier_mask]]
    tgt_inliers = target.xyz[corr_tgt[inlier_mask]]
    R_ref, t_ref = _svd_rigid_3pt(src_inliers, tgt_inliers)
    best_transform[:3, :3] = R_ref
    best_transform[:3, 3] = t_ref

    fitness, rmse = _compute_metrics(
        _apply_transform(source.xyz, best_transform), target.xyz, max_correspondence_distance
    )
    converged = inlier_ratio >= (1.0 - confidence) * 10  # heuristic

    logger.info(
        "RANSAC: best_inliers=%d/%d fitness=%.3f rmse=%.4f",
        best_inliers,
        n_corr,
        fitness,
        rmse,
    )
    return RegistrationResult(
        transformation=best_transform,
        fitness=fitness,
        inlier_rmse=rmse,
        converged=converged,
        n_iterations=iteration + 1,
    )


# ---------------------------------------------------------------------------
# Multi-scan alignment
# ---------------------------------------------------------------------------


class AlignmentResult:
    """Result of multi-scan alignment.

    Attributes
    ----------
    transformations : list[NDArray[np.float64]]
        List of 4×4 transformations, one per input cloud. The first cloud's
        transformation is always the identity (it is the reference).
    pairwise_results : list[RegistrationResult]
        Pairwise registration results between consecutive scans.
    global_rmse : float
        Mean inlier RMSE across all pairwise registrations.
    """

    def __init__(
        self,
        transformations: list[NDArray[np.float64]],
        pairwise_results: list[RegistrationResult],
    ) -> None:
        self.transformations = transformations
        self.pairwise_results = pairwise_results
        self.global_rmse = (
            float(np.mean([r.inlier_rmse for r in pairwise_results])) if pairwise_results else 0.0
        )


def align_scans(
    clouds: list[PointCloud],
    *,
    voxel_size: float = 0.5,
    refine_with_icp: bool = True,
    max_correspondence_distance: float = 1.0,
) -> AlignmentResult:
    """Align multiple scans into a common coordinate system.

    Performs sequential pairwise ICP registration — each scan is aligned
    to the previous one. The first cloud is the reference (identity transform).

    For best results, pre-process clouds with
    :func:`~occulus.filters.voxel_downsample` and
    :func:`~occulus.normals.estimate_normals` before calling this function.

    Parameters
    ----------
    clouds : list[PointCloud]
        At least 2 PointCloud objects to align. The first cloud is the
        fixed reference.
    voxel_size : float, optional
        Voxel size used for temporary downsampling before feature-based
        coarse alignment, by default 0.5.
    refine_with_icp : bool, optional
        Whether to run ICP after coarse alignment, by default ``True``.
    max_correspondence_distance : float, optional
        Maximum correspondence distance for ICP refinement, by default 1.0.

    Returns
    -------
    AlignmentResult
        Per-cloud transformations and pairwise quality metrics.

    Raises
    ------
    OcculusRegistrationError
        If fewer than 2 clouds are provided.
    """
    if len(clouds) < 2:
        raise OcculusRegistrationError("align_scans requires at least 2 clouds")

    from occulus.registration.icp import icp as _icp  # avoid circular at module level

    transforms: list[NDArray[np.float64]] = [np.eye(4, dtype=np.float64)]
    results: list[RegistrationResult] = []

    # Cumulative transformation — chain pairwise results
    cumulative = np.eye(4, dtype=np.float64)

    for i in range(1, len(clouds)):
        target = clouds[i - 1]
        source = clouds[i]

        logger.info("align_scans: aligning cloud %d → %d", i, i - 1)
        result = _icp(
            source,
            target,
            max_correspondence_distance=max_correspondence_distance,
            method="auto",
        )
        results.append(result)
        cumulative = result.transformation @ cumulative
        transforms.append(cumulative.copy())

    return AlignmentResult(transforms, results)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _svd_rigid_3pt(
    src: NDArray[np.float64],
    tgt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute rigid transform from point correspondences via SVD.

    Parameters
    ----------
    src : NDArray[np.float64]
        Source points (N >= 3, 3).
    tgt : NDArray[np.float64]
        Target points (N >= 3, 3).

    Returns
    -------
    R : NDArray[np.float64]
        Rotation matrix (3, 3).
    t : NDArray[np.float64]
        Translation vector (3,).
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    H = (src - src_mean).T @ (tgt - tgt_mean)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt_mean - R @ src_mean
    return R, t
