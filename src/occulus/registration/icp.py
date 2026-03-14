"""Iterative Closest Point (ICP) registration.

Aligns a source point cloud to a target using iterative closest-point
matching. Two variants are provided:

- **Point-to-point** — minimises sum of squared Euclidean distances between
  corresponding point pairs. Uses SVD-based optimal rotation.
- **Point-to-plane** — minimises sum of squared distances projected onto target
  normals. Converges faster near the solution but requires normals on the target.

The dispatcher :func:`icp` selects the best variant automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from occulus.exceptions import OcculusRegistrationError, OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of an ICP or global registration.

    Attributes
    ----------
    transformation : NDArray[np.float64]
        4×4 rigid transformation matrix (source → target frame).
    fitness : float
        Fraction of source points with a correspondence within
        ``max_correspondence_distance``. Range [0, 1]; higher is better.
    inlier_rmse : float
        Root-mean-square error of inlier correspondences (in coordinate units).
    converged : bool
        Whether the algorithm met its convergence criterion.
    n_iterations : int
        Number of iterations actually performed.
    """

    transformation: NDArray[np.float64]
    fitness: float
    inlier_rmse: float
    converged: bool
    n_iterations: int = 0


# Keep the old name for backwards-compat inside this package
ICPResult = RegistrationResult


def icp(
    source: PointCloud,
    target: PointCloud,
    *,
    max_correspondence_distance: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    init_transform: NDArray[np.float64] | None = None,
    method: str = "auto",
) -> RegistrationResult:
    """Align source to target using ICP, auto-selecting the best variant.

    If ``method="auto"`` (the default) and the target has normals, point-to-plane
    ICP is used. Otherwise point-to-point ICP is used.

    Parameters
    ----------
    source : PointCloud
        Source cloud to be transformed.
    target : PointCloud
        Target cloud (fixed reference).
    max_correspondence_distance : float, optional
        Maximum Euclidean distance for a point pair to be considered a
        correspondence, by default 1.0.
    max_iterations : int, optional
        Maximum number of ICP iterations, by default 50.
    tolerance : float, optional
        Convergence criterion — iteration stops when the change in the
        transformation matrix (Frobenius norm) falls below this value,
        by default 1e-6.
    init_transform : NDArray[np.float64] | None, optional
        Initial 4×4 transformation applied to source before the first
        iteration. Identity matrix if ``None``.
    method : str, optional
        ``"auto"``, ``"point_to_point"``, or ``"point_to_plane"``,
        by default ``"auto"``.

    Returns
    -------
    RegistrationResult
        Registration result containing the 4×4 transformation and quality
        metrics.

    Raises
    ------
    OcculusValidationError
        If ``method`` is unrecognised.
    OcculusRegistrationError
        If registration fails to produce a valid transformation.
    """
    valid_methods = {"auto", "point_to_point", "point_to_plane"}
    if method not in valid_methods:
        raise OcculusValidationError(f"method must be one of {valid_methods}, got '{method}'")

    use_p2plane = (method == "point_to_plane") or (method == "auto" and target.has_normals)

    if use_p2plane and not target.has_normals:
        raise OcculusValidationError(
            "point_to_plane ICP requires normals on the target cloud. "
            "Run occulus.normals.estimate_normals(target) first."
        )

    if use_p2plane:
        return icp_point_to_plane(
            source,
            target,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            tolerance=tolerance,
            init_transform=init_transform,
        )
    else:
        return icp_point_to_point(
            source,
            target,
            max_correspondence_distance=max_correspondence_distance,
            max_iterations=max_iterations,
            tolerance=tolerance,
            init_transform=init_transform,
        )


def icp_point_to_point(
    source: PointCloud,
    target: PointCloud,
    *,
    max_correspondence_distance: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    init_transform: NDArray[np.float64] | None = None,
) -> RegistrationResult:
    """Point-to-point ICP registration using SVD.

    Minimises the sum of squared Euclidean distances between each source
    point and its closest target point. The optimal rigid transformation is
    solved analytically via Singular Value Decomposition.

    Parameters
    ----------
    source : PointCloud
        Source cloud to be transformed.
    target : PointCloud
        Target cloud (fixed reference).
    max_correspondence_distance : float, optional
        Maximum distance for a pair to be a valid correspondence.
    max_iterations : int, optional
        Maximum ICP iterations.
    tolerance : float, optional
        Convergence tolerance on transformation change.
    init_transform : NDArray[np.float64] | None, optional
        Initial 4×4 transformation guess.

    Returns
    -------
    RegistrationResult
        Registration result.

    Raises
    ------
    OcculusRegistrationError
        If too few correspondences are found to compute a transformation.
    """
    transform = _init_transform(init_transform)
    src_xyz = _apply_transform(source.xyz, transform)
    tree = KDTree(target.xyz)

    prev_transform = transform.copy()
    converged = False

    for iteration in range(max_iterations):
        distances, indices = tree.query(src_xyz, k=1, workers=-1)
        indices = indices.ravel()
        distances = distances.ravel()

        inlier_mask = distances <= max_correspondence_distance
        if inlier_mask.sum() < 3:
            raise OcculusRegistrationError(
                f"Too few correspondences ({inlier_mask.sum()}) at iteration {iteration}. "
                "Increase max_correspondence_distance or improve initial alignment."
            )

        src_inliers = src_xyz[inlier_mask]
        tgt_inliers = target.xyz[indices[inlier_mask]]

        delta_R, delta_t = _svd_rigid(src_inliers, tgt_inliers)

        # Apply incremental transformation
        delta = np.eye(4)
        delta[:3, :3] = delta_R
        delta[:3, 3] = delta_t
        transform = delta @ transform
        src_xyz = _apply_transform(source.xyz, transform)

        # Convergence check
        change = np.linalg.norm(transform - prev_transform, ord="fro")
        if change < tolerance:
            converged = True
            logger.debug("ICP p2p converged at iteration %d (change=%.2e)", iteration, change)
            break
        prev_transform = transform.copy()

    fitness, rmse = _compute_metrics(src_xyz, target.xyz, max_correspondence_distance)
    logger.info(
        "ICP p2p: fitness=%.3f rmse=%.4f converged=%s iter=%d",
        fitness,
        rmse,
        converged,
        iteration + 1,
    )
    return RegistrationResult(
        transformation=transform,
        fitness=fitness,
        inlier_rmse=rmse,
        converged=converged,
        n_iterations=iteration + 1,
    )


def icp_point_to_plane(
    source: PointCloud,
    target: PointCloud,
    *,
    max_correspondence_distance: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    init_transform: NDArray[np.float64] | None = None,
) -> RegistrationResult:
    """Point-to-plane ICP registration.

    Minimises the sum of squared distances projected onto target surface
    normals. Requires normals on the target cloud. Typically converges
    faster and with better accuracy than point-to-point near the solution.

    Parameters
    ----------
    source : PointCloud
        Source cloud to be transformed.
    target : PointCloud
        Target cloud with normals (fixed reference).
    max_correspondence_distance : float, optional
        Maximum distance for a pair to be a valid correspondence.
    max_iterations : int, optional
        Maximum ICP iterations.
    tolerance : float, optional
        Convergence tolerance on transformation change.
    init_transform : NDArray[np.float64] | None, optional
        Initial 4×4 transformation guess.

    Returns
    -------
    RegistrationResult
        Registration result.

    Raises
    ------
    OcculusValidationError
        If target cloud does not have normals.
    OcculusRegistrationError
        If too few correspondences are found.
    """
    if not target.has_normals:
        raise OcculusValidationError(
            "Point-to-plane ICP requires normals on the target cloud. "
            "Use occulus.normals.estimate_normals(target) first."
        )
    assert target.normals is not None  # guaranteed above

    transform = _init_transform(init_transform)
    src_xyz = _apply_transform(source.xyz, transform)
    tree = KDTree(target.xyz)

    prev_transform = transform.copy()
    converged = False

    for iteration in range(max_iterations):
        distances, indices = tree.query(src_xyz, k=1, workers=-1)
        indices = indices.ravel()
        distances = distances.ravel()

        inlier_mask = distances <= max_correspondence_distance
        if inlier_mask.sum() < 6:
            raise OcculusRegistrationError(
                f"Too few correspondences ({inlier_mask.sum()}) at iteration {iteration}."
            )

        src_pts = src_xyz[inlier_mask]
        tgt_pts = target.xyz[indices[inlier_mask]]
        tgt_nrm = target.normals[indices[inlier_mask]]

        # Point-to-plane linear system (Segal et al.)
        # Build 6-unknown system [r1 r2 r3 t1 t2 t3] via least squares
        A, b = _build_p2plane_system(src_pts, tgt_pts, tgt_nrm)
        try:
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise OcculusRegistrationError(
                f"Point-to-plane linear system is singular at iteration {iteration}: {exc}"
            ) from exc

        # Extract rotation (small-angle approximation → exact rotation via Rodrigues)
        rx, ry, rz, tx, ty, tz = x
        delta = np.eye(4)
        delta[:3, :3] = _rodrigues_to_matrix(np.array([rx, ry, rz]))
        delta[:3, 3] = [tx, ty, tz]
        transform = delta @ transform
        src_xyz = _apply_transform(source.xyz, transform)

        change = np.linalg.norm(transform - prev_transform, ord="fro")
        if change < tolerance:
            converged = True
            logger.debug("ICP p2plane converged at iteration %d", iteration)
            break
        prev_transform = transform.copy()

    fitness, rmse = _compute_metrics(src_xyz, target.xyz, max_correspondence_distance)
    logger.info(
        "ICP p2plane: fitness=%.3f rmse=%.4f converged=%s iter=%d",
        fitness,
        rmse,
        converged,
        iteration + 1,
    )
    return RegistrationResult(
        transformation=transform,
        fitness=fitness,
        inlier_rmse=rmse,
        converged=converged,
        n_iterations=iteration + 1,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _init_transform(init: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Return the initial 4×4 transform, defaulting to identity.

    Parameters
    ----------
    init : NDArray[np.float64] | None
        Caller-supplied transform or None.

    Returns
    -------
    NDArray[np.float64]
        4×4 float64 array.

    Raises
    ------
    OcculusValidationError
        If supplied array is not (4, 4).
    """
    if init is None:
        return np.eye(4, dtype=np.float64)
    arr = np.asarray(init, dtype=np.float64)
    if arr.shape != (4, 4):
        raise OcculusValidationError(f"init_transform must be a (4, 4) array, got {arr.shape}")
    return arr.copy()


def _apply_transform(xyz: NDArray[np.float64], T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply a 4×4 rigid transform to an (N, 3) array.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Input points (N, 3).
    T : NDArray[np.float64]
        4×4 transformation matrix.

    Returns
    -------
    NDArray[np.float64]
        Transformed points (N, 3).
    """
    n = xyz.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    homogeneous = np.hstack((xyz, ones))  # (N, 4)
    return (T @ homogeneous.T).T[:, :3]


def _svd_rigid(
    src: NDArray[np.float64],
    tgt: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute optimal rigid rotation and translation via SVD.

    Solves min_R,t ||tgt - (R @ src.T + t)||_F.

    Parameters
    ----------
    src : NDArray[np.float64]
        Source points (N, 3).
    tgt : NDArray[np.float64]
        Target points (N, 3).

    Returns
    -------
    R : NDArray[np.float64]
        Optimal rotation matrix (3, 3).
    t : NDArray[np.float64]
        Optimal translation vector (3,).
    """
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = tgt_mean - R @ src_mean
    return R, t


def _build_p2plane_system(
    src: NDArray[np.float64],
    tgt: NDArray[np.float64],
    nrm: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build the linear system for point-to-plane ICP.

    Parameters
    ----------
    src : NDArray[np.float64]
        Transformed source points (N, 3).
    tgt : NDArray[np.float64]
        Corresponding target points (N, 3).
    nrm : NDArray[np.float64]
        Target normals at corresponding points (N, 3).

    Returns
    -------
    A : NDArray[np.float64]
        System matrix (N, 6).
    b : NDArray[np.float64]
        RHS vector (N,).
    """
    nx, ny, nz = nrm[:, 0], nrm[:, 1], nrm[:, 2]
    sx, sy, sz = src[:, 0], src[:, 1], src[:, 2]

    # Cross products for rotation columns
    A = np.column_stack(
        [
            nz * sy - ny * sz,
            nx * sz - nz * sx,
            ny * sx - nx * sy,
            nx,
            ny,
            nz,
        ]
    )
    b = np.einsum("ij,ij->i", tgt - src, nrm)
    return A, b


def _rodrigues_to_matrix(rvec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a rotation vector to a rotation matrix via Rodrigues' formula.

    Parameters
    ----------
    rvec : NDArray[np.float64]
        Rotation vector (3,). Magnitude = rotation angle in radians.

    Returns
    -------
    NDArray[np.float64]
        Rotation matrix (3, 3).
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return np.eye(3)
    axis = rvec / angle
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _compute_metrics(
    src_xyz: NDArray[np.float64],
    tgt_xyz: NDArray[np.float64],
    max_dist: float,
) -> tuple[float, float]:
    """Compute fitness and RMSE for a registration result.

    Parameters
    ----------
    src_xyz : NDArray[np.float64]
        Transformed source points (N, 3).
    tgt_xyz : NDArray[np.float64]
        Target points (M, 3).
    max_dist : float
        Inlier distance threshold.

    Returns
    -------
    fitness : float
        Fraction of source points that are inliers.
    rmse : float
        RMSE of inlier distances.
    """
    tree = KDTree(tgt_xyz)
    distances, _ = tree.query(src_xyz, k=1, workers=-1)
    distances = distances.ravel()
    inlier_mask = distances <= max_dist
    fitness = float(inlier_mask.sum()) / len(src_xyz)
    rmse = float(np.sqrt((distances[inlier_mask] ** 2).mean())) if inlier_mask.any() else 0.0
    return fitness, rmse
