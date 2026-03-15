"""M3C2 point cloud change detection (Lague et al. 2013).

Implements the Multiscale Model to Model Cloud Comparison algorithm for
computing signed distances between two point cloud epochs along local
surface normals, with per-point uncertainty quantification via Level of
Detection at a configurable confidence level.

Reference
---------
Lague, D., Brodu, N., & Leroux, J. (2013). Accurate 3D comparison of
complex topography with terrestrial laser scanner: Application to the
Rangitikei canyon (N-Z). *ISPRS Journal of Photogrammetry and Remote
Sensing*, 82, 10--26.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]
from scipy.stats import norm as _norm  # type: ignore[import-untyped]

from occulus.exceptions import OcculusChangeDetectionError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = ["M3C2Result", "m3c2"]


@dataclass
class M3C2Result:
    """Container for M3C2 change detection results.

    Attributes
    ----------
    distances : NDArray[np.float64]
        Signed M3C2 distances for each core point.  Shape ``(N,)``.
        Positive values indicate epoch2 surface is farther along the
        normal direction; ``NaN`` where computation was not possible.
    uncertainties : NDArray[np.float64]
        Level of Detection (LoD) at the requested confidence level for
        each core point.  Shape ``(N,)``.  ``NaN`` where computation
        was not possible.
    normals : NDArray[np.float64]
        Unit surface normals at core points.  Shape ``(N, 3)``.
    core_points : NDArray[np.float64]
        Coordinates of the core points used for comparison.  Shape ``(N, 3)``.
    significant_change : NDArray[np.bool_]
        Boolean mask of length ``N`` where ``True`` indicates the
        absolute distance exceeds the LoD (i.e., statistically
        significant change was detected).
    """

    distances: NDArray[np.float64]
    uncertainties: NDArray[np.float64]
    normals: NDArray[np.float64]
    core_points: NDArray[np.float64]
    significant_change: NDArray[np.bool_]


def m3c2(
    epoch1: PointCloud,
    epoch2: PointCloud,
    *,
    core_points: NDArray[np.float64] | None = None,
    normal_scale: float = 1.0,
    projection_scale: float = 1.0,
    max_cylinder_depth: float = 10.0,
    registration_error: float = 0.0,
    confidence: float = 0.95,
) -> M3C2Result:
    """Compute M3C2 signed distances between two point cloud epochs.

    For each core point the algorithm:

    1. Estimates a local surface normal from *epoch1* neighbours within
       ``normal_scale``.
    2. Projects a cylinder of radius ``projection_scale`` along the
       normal into both epochs (capped at ``max_cylinder_depth``).
    3. Computes the mean projected position in each epoch.
    4. Reports the signed distance between the two means along the
       normal, together with an uncertainty estimate (Level of
       Detection at the requested ``confidence``).

    Parameters
    ----------
    epoch1 : PointCloud
        Reference epoch point cloud.
    epoch2 : PointCloud
        Comparison epoch point cloud.
    core_points : NDArray[np.float64] | None, optional
        Explicit (M, 3) array of core point coordinates.  If ``None``,
        the coordinates of *epoch1* are used directly.
    normal_scale : float, optional
        Radius for the neighbourhood used to estimate local surface
        normals at each core point, by default 1.0.
    projection_scale : float, optional
        Radius of the cylinder projected along the normal into each
        epoch, by default 1.0.
    max_cylinder_depth : float, optional
        Maximum half-length of the projection cylinder along the
        normal direction, by default 10.0.
    registration_error : float, optional
        Known registration error between the two epochs (metres),
        folded into the LoD computation, by default 0.0.
    confidence : float, optional
        Confidence level for the Level of Detection, by default 0.95.
        Must be in the open interval (0, 1).

    Returns
    -------
    M3C2Result
        Dataclass containing distances, uncertainties, normals,
        core_points, and significant_change arrays.

    Raises
    ------
    OcculusChangeDetectionError
        If either epoch has fewer than 3 points, ``confidence`` is
        outside (0, 1), ``normal_scale`` or ``projection_scale`` is
        non-positive, or if normal estimation fails for all core
        points.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if epoch1.n_points < 3:
        raise OcculusChangeDetectionError(
            f"epoch1 must have at least 3 points, got {epoch1.n_points}"
        )
    if epoch2.n_points < 3:
        raise OcculusChangeDetectionError(
            f"epoch2 must have at least 3 points, got {epoch2.n_points}"
        )
    if not (0.0 < confidence < 1.0):
        raise OcculusChangeDetectionError(f"confidence must be in (0, 1), got {confidence}")
    if normal_scale <= 0.0:
        raise OcculusChangeDetectionError(f"normal_scale must be positive, got {normal_scale}")
    if projection_scale <= 0.0:
        raise OcculusChangeDetectionError(
            f"projection_scale must be positive, got {projection_scale}"
        )
    if max_cylinder_depth <= 0.0:
        raise OcculusChangeDetectionError(
            f"max_cylinder_depth must be positive, got {max_cylinder_depth}"
        )
    if registration_error < 0.0:
        raise OcculusChangeDetectionError(
            f"registration_error must be non-negative, got {registration_error}"
        )

    # Core points default to epoch1 coordinates
    if core_points is not None:
        if core_points.ndim != 2 or core_points.shape[1] != 3:
            raise OcculusChangeDetectionError(
                f"core_points must be (M, 3) array, got shape {core_points.shape}"
            )
        cp = np.ascontiguousarray(core_points, dtype=np.float64)
    else:
        cp = epoch1.xyz.copy()

    n_core = cp.shape[0]
    z_value = float(_norm.ppf((1.0 + confidence) / 2.0))

    logger.info(
        "M3C2: %d core points, normal_scale=%.3f, projection_scale=%.3f, "
        "max_depth=%.3f, reg_error=%.4f, confidence=%.2f (z=%.4f)",
        n_core,
        normal_scale,
        projection_scale,
        max_cylinder_depth,
        registration_error,
        confidence,
        z_value,
    )

    # ------------------------------------------------------------------
    # Step 1: Build KD-trees
    # ------------------------------------------------------------------
    tree1 = KDTree(epoch1.xyz)
    tree2 = KDTree(epoch2.xyz)

    # ------------------------------------------------------------------
    # Step 2: Estimate surface normals at core points
    # ------------------------------------------------------------------
    normals = _estimate_core_normals(cp, epoch1.xyz, tree1, normal_scale)

    # ------------------------------------------------------------------
    # Step 3–6: Per-core-point cylinder projection and distance
    # ------------------------------------------------------------------
    distances = np.full(n_core, np.nan, dtype=np.float64)
    uncertainties = np.full(n_core, np.nan, dtype=np.float64)

    for i in range(n_core):
        n_hat = normals[i]
        if np.all(np.isnan(n_hat)):
            continue

        # Project cylinder into epoch1
        mean1, std1, count1 = _cylinder_stats(
            cp[i],
            n_hat,
            epoch1.xyz,
            tree1,
            projection_scale,
            max_cylinder_depth,
        )
        # Project cylinder into epoch2
        mean2, std2, count2 = _cylinder_stats(
            cp[i],
            n_hat,
            epoch2.xyz,
            tree2,
            projection_scale,
            max_cylinder_depth,
        )

        if count1 < 1 or count2 < 1:
            continue

        # Signed distance along normal direction
        distances[i] = mean2 - mean1

        # Level of Detection (LoD)
        var_term = std1**2 / count1 + std2**2 / count2
        uncertainties[i] = z_value * np.sqrt(var_term + registration_error**2)

    significant = np.abs(distances) > uncertainties

    n_valid = int(np.isfinite(distances).sum())
    n_sig = int(significant.sum())
    logger.info(
        "M3C2 complete: %d/%d valid distances, %d significant changes",
        n_valid,
        n_core,
        n_sig,
    )

    return M3C2Result(
        distances=distances,
        uncertainties=uncertainties,
        normals=normals,
        core_points=cp,
        significant_change=significant,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _estimate_core_normals(
    core_points: NDArray[np.float64],
    reference_xyz: NDArray[np.float64],
    tree: KDTree,
    radius: float,
) -> NDArray[np.float64]:
    """Estimate surface normals at core points using PCA on reference cloud.

    Parameters
    ----------
    core_points : NDArray[np.float64]
        Core point coordinates, shape ``(M, 3)``.
    reference_xyz : NDArray[np.float64]
        Reference epoch point coordinates, shape ``(N, 3)``.
    tree : KDTree
        KD-tree built from ``reference_xyz``.
    radius : float
        Neighbourhood search radius.

    Returns
    -------
    NDArray[np.float64]
        Unit normals of shape ``(M, 3)``.  Rows are ``NaN`` where
        the local neighbourhood has fewer than 3 points.
    """
    n_core = core_points.shape[0]
    normals = np.full((n_core, 3), np.nan, dtype=np.float64)

    neighbour_lists = tree.query_ball_point(core_points, r=radius, workers=-1)

    for i, neighbours in enumerate(neighbour_lists):
        if len(neighbours) < 3:
            continue

        pts = reference_xyz[neighbours]
        centred = pts - pts.mean(axis=0)
        cov = centred.T @ centred / len(pts)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue corresponds to the surface normal
        normal = eigenvectors[:, 0]
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normals[i] = normal / norm_len

    n_valid = int(np.isfinite(normals[:, 0]).sum())
    logger.debug(
        "_estimate_core_normals: %d/%d core points got valid normals",
        n_valid,
        n_core,
    )
    return normals


def _cylinder_stats(
    origin: NDArray[np.float64],
    normal: NDArray[np.float64],
    cloud_xyz: NDArray[np.float64],
    tree: KDTree,
    radius: float,
    max_depth: float,
) -> tuple[float, float, int]:
    """Compute mean and std of point projections inside a cylinder.

    The cylinder is centred at ``origin``, oriented along ``normal``,
    with the given ``radius`` and capped at ``max_depth`` in each
    direction along the normal.

    Parameters
    ----------
    origin : NDArray[np.float64]
        Centre of the cylinder, shape ``(3,)``.
    normal : NDArray[np.float64]
        Unit normal (cylinder axis), shape ``(3,)``.
    cloud_xyz : NDArray[np.float64]
        Point cloud coordinates, shape ``(N, 3)``.
    tree : KDTree
        KD-tree built from ``cloud_xyz``.
    radius : float
        Cylinder radius perpendicular to the normal.
    max_depth : float
        Maximum distance along the normal from the origin.

    Returns
    -------
    tuple[float, float, int]
        ``(mean_projection, std_projection, count)`` of points inside
        the cylinder.  Returns ``(0.0, 0.0, 0)`` if no points found.
    """
    # Bounding sphere radius for initial candidate retrieval
    search_radius = np.sqrt(radius**2 + max_depth**2)
    candidates = tree.query_ball_point(origin, r=search_radius, workers=1)

    if len(candidates) == 0:
        return 0.0, 0.0, 0

    pts = cloud_xyz[candidates]
    diff = pts - origin  # (K, 3)

    # Project onto normal axis
    along = diff @ normal  # (K,)

    # Perpendicular distance from axis
    perp = diff - np.outer(along, normal)
    perp_dist = np.linalg.norm(perp, axis=1)

    # Filter: inside cylinder
    mask = (np.abs(along) <= max_depth) & (perp_dist <= radius)
    projections = along[mask]

    count = len(projections)
    if count == 0:
        return 0.0, 0.0, 0

    mean_proj = float(projections.mean())
    std_proj = float(projections.std(ddof=0)) if count > 1 else 0.0

    return mean_proj, std_proj, count
