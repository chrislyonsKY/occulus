"""Surface normal estimation and orientation for point clouds.

Normal vectors are required for several downstream operations including
Poisson surface reconstruction and point-to-plane ICP registration.

Available functions
-------------------
- :func:`estimate_normals` — PCA-based normal estimation from local neighbourhoods
- :func:`orient_normals_to_viewpoint` — flip normals to face a known viewpoint

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

__all__ = ["estimate_normals", "orient_normals_to_viewpoint"]


def estimate_normals(
    cloud: PointCloud,
    *,
    radius: float | None = None,
    max_nn: int = 30,
) -> PointCloud:
    """Estimate surface normals via PCA on local neighbourhoods.

    For each point, the ``max_nn`` nearest neighbours within ``radius`` are
    collected and a PCA is performed. The eigenvector corresponding to the
    smallest eigenvalue of the local covariance matrix is taken as the surface
    normal. Normals are not oriented — use :func:`orient_normals_to_viewpoint`
    to flip them toward a known viewpoint.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud. Returned cloud will have the ``normals`` attribute
        populated with unit vectors.
    radius : float | None, optional
        Neighbourhood search radius. If ``None``, a radius equal to twice the
        mean nearest-neighbour distance is used.
    max_nn : int, optional
        Maximum number of neighbours to use per point, by default 30.
        More neighbours → smoother normals.

    Returns
    -------
    PointCloud
        A new cloud of the same concrete type with ``normals`` set.

    Raises
    ------
    OcculusValidationError
        If the cloud has fewer than 3 points (PCA is not meaningful).
    """
    if cloud.n_points < 3:
        raise OcculusValidationError(
            f"estimate_normals requires at least 3 points, got {cloud.n_points}"
        )
    if max_nn < 3:
        raise OcculusValidationError(f"max_nn must be at least 3, got {max_nn}")

    xyz = cloud.xyz
    tree = KDTree(xyz)

    # Auto-radius: 2× the mean of the nearest-neighbour distances
    if radius is None:
        nn_distances, _ = tree.query(xyz, k=2, workers=-1)
        radius = float(nn_distances[:, 1].mean() * 2.0)
        logger.debug("estimate_normals: auto radius=%.6f", radius)

    normals = np.zeros_like(xyz)

    # Query neighbours for all points at once then process individually
    neighbour_lists = tree.query_ball_point(xyz, r=radius, workers=-1)

    for i, neighbours in enumerate(neighbour_lists):
        if len(neighbours) < 3:
            # Degenerate neighbourhood — use z-up as fallback
            normals[i] = [0.0, 0.0, 1.0]
            continue

        # Limit to max_nn closest neighbours
        if len(neighbours) > max_nn:
            idx = np.array(neighbours)
            dists = np.linalg.norm(xyz[idx] - xyz[i], axis=1)
            neighbours = idx[np.argpartition(dists, max_nn)[:max_nn]].tolist()

        pts = xyz[neighbours]
        centred = pts - pts.mean(axis=0)
        cov = centred.T @ centred / len(pts)
        _eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Smallest eigenvalue → most perpendicular direction = normal
        normals[i] = eigenvectors[:, 0]

    # Normalise to unit length (guard against zero-length normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms

    logger.info("estimate_normals: computed normals for %d points", cloud.n_points)

    return _copy_with_normals(cloud, normals)


def orient_normals_to_viewpoint(
    cloud: PointCloud,
    viewpoint: NDArray[np.float64],
) -> PointCloud:
    """Flip normals so that they face a known viewpoint (scanner position).

    A normal is flipped if the dot product with the vector from the point
    to the viewpoint is negative (i.e., the normal points away).

    Parameters
    ----------
    cloud : PointCloud
        Input cloud. Must have normals (``cloud.has_normals`` must be ``True``).
    viewpoint : NDArray[np.float64]
        3D position of the viewpoint (e.g. scanner origin) as a (3,) array.

    Returns
    -------
    PointCloud
        A new cloud with consistently oriented normals.

    Raises
    ------
    OcculusValidationError
        If the cloud has no normals or ``viewpoint`` is not a (3,) array.
    """
    if not cloud.has_normals:
        raise OcculusValidationError("Cloud has no normals. Run estimate_normals() first.")
    viewpoint = np.asarray(viewpoint, dtype=np.float64)
    if viewpoint.shape != (3,):
        raise OcculusValidationError(f"viewpoint must be a (3,) array, got shape {viewpoint.shape}")

    assert cloud.normals is not None  # guarded above
    normals = cloud.normals.copy()
    to_viewpoint = viewpoint - cloud.xyz  # (N, 3) vectors toward viewpoint
    dot = np.einsum("ij,ij->i", normals, to_viewpoint)
    flip_mask = dot < 0
    normals[flip_mask] *= -1

    logger.debug(
        "orient_normals_to_viewpoint: flipped %d/%d normals",
        flip_mask.sum(),
        cloud.n_points,
    )

    return _copy_with_normals(cloud, normals)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _copy_with_normals(cloud: PointCloud, normals: NDArray[np.float64]) -> PointCloud:
    """Return a copy of ``cloud`` with ``normals`` replaced.

    Parameters
    ----------
    cloud : PointCloud
        Source cloud.
    normals : NDArray[np.float64]
        New (N, 3) normal array.

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
        classification=cloud.classification,
        rgb=cloud.rgb,
        normals=normals,
        return_number=cloud.return_number,
        number_of_returns=cloud.number_of_returns,
        metadata=meta,
    )
    return cloud.__class__(cloud.xyz, **kwargs)  # type: ignore[arg-type]
