"""Coordinate reference system transforms for point clouds.

Provides reprojection of :class:`~occulus.types.PointCloud` objects and
low-level coordinate array transforms using ``pyproj``.  ``pyproj`` is
lazy-imported at call time so that the rest of Occulus works without it.

All transforms use ``pyproj.Transformer.from_crs(..., always_xy=True)``
to avoid axis-order surprises.  The deprecated ``pyproj.transform()``
function is never used.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusCRSError

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "reproject",
    "transform_coordinates",
]


def transform_coordinates(
    xyz: NDArray[np.float64],
    source_crs: str,
    target_crs: str,
) -> NDArray[np.float64]:
    """Transform an (N, 3) coordinate array from one CRS to another.

    Uses ``pyproj.Transformer.from_crs`` with ``always_xy=True`` so that
    the first column is always easting/longitude and the second is
    northing/latitude, regardless of the CRS axis order definition.

    The Z (elevation) column is passed through unchanged.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Input coordinates as an (N, 3) float64 array.
    source_crs : str
        Source CRS identifier accepted by ``pyproj`` (e.g. ``"EPSG:4326"``).
    target_crs : str
        Target CRS identifier accepted by ``pyproj``.

    Returns
    -------
    NDArray[np.float64]
        Transformed coordinates as an (N, 3) float64 array.

    Raises
    ------
    OcculusCRSError
        If ``pyproj`` is not installed, the CRS identifiers are invalid,
        or the transform itself fails.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise OcculusCRSError(f"xyz must be an (N, 3) array, got shape {xyz.shape}")

    if source_crs == target_crs:
        logger.debug(
            "transform_coordinates: source and target CRS are identical (%s), returning copy",
            source_crs,
        )
        return xyz.copy()

    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise OcculusCRSError("pyproj is required for CRS transforms: pip install pyproj") from exc

    try:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    except Exception as exc:
        raise OcculusCRSError(
            f"Failed to create transformer from '{source_crs}' to '{target_crs}': {exc}"
        ) from exc

    try:
        x_out, y_out = transformer.transform(xyz[:, 0], xyz[:, 1])
    except Exception as exc:
        raise OcculusCRSError(
            f"Coordinate transform failed ({source_crs} -> {target_crs}): {exc}"
        ) from exc

    result = np.column_stack(
        [
            np.asarray(x_out, dtype=np.float64),
            np.asarray(y_out, dtype=np.float64),
            xyz[:, 2],
        ]
    )

    logger.debug(
        "transform_coordinates: %d points, %s -> %s",
        len(xyz),
        source_crs,
        target_crs,
    )
    return np.ascontiguousarray(result, dtype=np.float64)


def reproject(
    cloud: PointCloud,
    target_crs: str,
    source_crs: str | None = None,
) -> PointCloud:
    """Reproject a point cloud to a new coordinate reference system.

    If ``source_crs`` is not provided it is inferred from
    ``cloud.metadata.coordinate_system``.  When that field is also empty
    an :class:`~occulus.exceptions.OcculusCRSError` is raised.

    The returned cloud is a **new** object — the input is never mutated.
    All per-point attributes (intensity, classification, rgb, normals,
    return_number, number_of_returns) are carried over unchanged.  The
    ``metadata.coordinate_system`` field on the result is set to
    ``target_crs``.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    target_crs : str
        Target CRS identifier (e.g. ``"EPSG:32617"``).
    source_crs : str | None, optional
        Source CRS identifier.  If ``None``, the CRS is read from
        ``cloud.metadata.coordinate_system``.

    Returns
    -------
    PointCloud
        New cloud with reprojected XYZ and updated metadata.

    Raises
    ------
    OcculusCRSError
        If the source CRS cannot be determined, or if the transform fails.
    """
    from occulus.types import AcquisitionMetadata

    resolved_source = source_crs or cloud.metadata.coordinate_system
    if not resolved_source:
        raise OcculusCRSError(
            "Cannot determine source CRS: provide source_crs or set "
            "cloud.metadata.coordinate_system"
        )

    new_xyz = transform_coordinates(cloud.xyz, resolved_source, target_crs)

    # Build updated metadata with the new CRS
    meta = AcquisitionMetadata(
        platform=cloud.metadata.platform,
        scanner_model=cloud.metadata.scanner_model,
        scan_date=cloud.metadata.scan_date,
        coordinate_system=target_crs,
        point_density_per_sqm=cloud.metadata.point_density_per_sqm,
        scan_positions=deepcopy(cloud.metadata.scan_positions),
        flight_altitude_m=cloud.metadata.flight_altitude_m,
        scan_angle_range=cloud.metadata.scan_angle_range,
    )

    def _sel(arr: NDArray | None) -> NDArray | None:  # type: ignore[type-arg]
        """Return a copy of *arr* or ``None``."""
        return arr.copy() if arr is not None else None

    kwargs: dict[str, object] = dict(
        intensity=_sel(cloud.intensity),
        classification=_sel(cloud.classification),
        rgb=_sel(cloud.rgb),
        normals=_sel(cloud.normals),
        return_number=_sel(cloud.return_number),
        number_of_returns=_sel(cloud.number_of_returns),
        metadata=meta,
    )

    logger.debug(
        "reproject: %d points, %s -> %s",
        cloud.n_points,
        resolved_source,
        target_crs,
    )
    return cloud.__class__(new_xyz, **kwargs)  # type: ignore[arg-type]
