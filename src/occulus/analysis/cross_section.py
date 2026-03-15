"""Cross-section and profile extraction along polylines.

Extract elevation profiles from point clouds by slicing along user-defined
polylines with a configurable corridor width. Useful for road design,
embankment analysis, and terrain inspection.

All computations use pure NumPy — no optional dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "CrossSection",
    "extract_cross_section",
    "extract_profiles",
]


@dataclass
class CrossSection:
    """A cross-section profile extracted from a point cloud.

    Attributes
    ----------
    station : NDArray[np.float64]
        Station distances along the profile, shape (M,).
    elevation : NDArray[np.float64]
        Elevation values at each station, shape (M,).
    points : NDArray[np.float64]
        The 3D points contributing to this cross-section, shape (K, 3).
    width : float
        Corridor half-width used to select points.
    polyline : NDArray[np.float64]
        The polyline vertices defining this cross-section, shape (P, 2) or (P, 3).
    """

    station: NDArray[np.float64]
    elevation: NDArray[np.float64]
    points: NDArray[np.float64]
    width: float
    polyline: NDArray[np.float64]


def extract_cross_section(
    cloud: PointCloud,
    polyline: NDArray[np.float64],
    width: float = 1.0,
    resolution: float = 0.1,
) -> CrossSection:
    """Extract a cross-section profile along a polyline.

    Points within ``width`` distance of the polyline are projected onto the
    polyline centreline. The resulting station-elevation profile is binned at
    the specified ``resolution``.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    polyline : NDArray[np.float64]
        Polyline vertices as (P, 2) or (P, 3) array. Only XY coordinates are
        used for the corridor selection; Z is ignored if present.
    width : float, optional
        Half-width of the corridor around the polyline, by default 1.0.
        Points within this distance of the polyline centreline are included.
    resolution : float, optional
        Station spacing for the output profile, by default 0.1.

    Returns
    -------
    CrossSection
        Extracted profile with station distances and elevations.

    Raises
    ------
    OcculusValidationError
        If inputs are invalid (empty cloud, degenerate polyline, non-positive
        width or resolution).
    """
    _validate_inputs(cloud, polyline, width, resolution)

    polyline_2d = np.asarray(polyline[:, :2], dtype=np.float64)
    xy = cloud.xyz[:, :2]
    z = cloud.xyz[:, 2]

    # Select points within the corridor
    stations, distances = _project_onto_polyline(xy, polyline_2d)
    corridor_mask = distances <= width

    if not corridor_mask.any():
        logger.warning(
            "extract_cross_section: no points found within width=%.2f of polyline",
            width,
        )
        return CrossSection(
            station=np.array([], dtype=np.float64),
            elevation=np.array([], dtype=np.float64),
            points=np.zeros((0, 3), dtype=np.float64),
            width=width,
            polyline=polyline,
        )

    sel_stations = stations[corridor_mask]
    sel_z = z[corridor_mask]
    sel_points = cloud.xyz[corridor_mask]

    # Bin into regular station intervals
    total_length = _polyline_length(polyline_2d)
    bin_edges = np.arange(0.0, total_length + resolution, resolution)

    if len(bin_edges) < 2:
        logger.warning(
            "extract_cross_section: polyline too short for resolution=%.3f",
            resolution,
        )
        return CrossSection(
            station=np.array([], dtype=np.float64),
            elevation=np.array([], dtype=np.float64),
            points=sel_points,
            width=width,
            polyline=polyline,
        )

    bin_indices = np.clip(np.digitize(sel_stations, bin_edges) - 1, 0, len(bin_edges) - 2)

    n_bins = len(bin_edges) - 1
    station_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_z = np.full(n_bins, np.nan, dtype=np.float64)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.any():
            mean_z[i] = float(sel_z[mask].mean())

    # Remove empty bins
    valid = np.isfinite(mean_z)
    out_station = station_centres[valid]
    out_elevation = mean_z[valid]

    logger.debug(
        "extract_cross_section: %d bins with data out of %d (width=%.2f, res=%.3f)",
        int(valid.sum()),
        n_bins,
        width,
        resolution,
    )

    return CrossSection(
        station=out_station,
        elevation=out_elevation,
        points=sel_points,
        width=width,
        polyline=polyline,
    )


def extract_profiles(
    cloud: PointCloud,
    polyline: NDArray[np.float64],
    interval: float = 10.0,
    width: float = 5.0,
    resolution: float = 0.1,
) -> list[CrossSection]:
    """Extract perpendicular profiles at regular intervals along a polyline.

    At each station along the centreline, a perpendicular cross-section of
    length ``2 * width`` is extracted and sampled at ``resolution``.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    polyline : NDArray[np.float64]
        Centreline polyline vertices as (P, 2) or (P, 3) array.
    interval : float, optional
        Spacing between profile stations along the centreline, by default 10.0.
    width : float, optional
        Half-width of each perpendicular profile, by default 5.0.
    resolution : float, optional
        Station spacing within each profile, by default 0.1.

    Returns
    -------
    list[CrossSection]
        One ``CrossSection`` per profile station. Empty profiles (no points
        in corridor) are included with zero-length arrays.

    Raises
    ------
    OcculusValidationError
        If inputs are invalid (empty cloud, degenerate polyline, non-positive
        interval, width, or resolution).
    """
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot extract profiles from an empty cloud")
    polyline = np.asarray(polyline, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] < 2:
        raise OcculusValidationError(
            f"polyline must be (P, 2) or (P, 3) with P >= 2, got shape {polyline.shape}"
        )
    if interval <= 0:
        raise OcculusValidationError(f"interval must be positive, got {interval}")
    if width <= 0:
        raise OcculusValidationError(f"width must be positive, got {width}")
    if resolution <= 0:
        raise OcculusValidationError(f"resolution must be positive, got {resolution}")

    polyline_2d = polyline[:, :2]
    total_length = _polyline_length(polyline_2d)

    # Generate station distances along the centreline
    station_distances = np.arange(0.0, total_length + interval * 0.5, interval)

    profiles: list[CrossSection] = []
    cloud.xyz[:, :2]

    for dist in station_distances:
        # Find the point and perpendicular direction at this station
        centre, perp = _point_and_perp_at_station(polyline_2d, dist)

        # Build a 2-point perpendicular polyline
        p1 = centre - perp * width
        p2 = centre + perp * width
        perp_line = np.array([p1, p2], dtype=np.float64)

        profile = extract_cross_section(cloud, perp_line, width=width, resolution=resolution)
        profiles.append(profile)

    logger.info(
        "extract_profiles: %d profiles at interval=%.1f along %.1fm centreline",
        len(profiles),
        interval,
        total_length,
    )

    return profiles


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    cloud: PointCloud,
    polyline: NDArray[np.float64],
    width: float,
    resolution: float,
) -> None:
    """Validate common inputs for cross-section extraction.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    polyline : NDArray[np.float64]
        Polyline vertices.
    width : float
        Corridor half-width.
    resolution : float
        Station spacing.

    Raises
    ------
    OcculusValidationError
        On any invalid input.
    """
    if cloud.n_points == 0:
        raise OcculusValidationError("Cannot extract cross-section from an empty cloud")

    polyline = np.asarray(polyline, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] < 2:
        raise OcculusValidationError(
            f"polyline must be (P, 2) or (P, 3) with P >= 2, got shape {polyline.shape}"
        )

    if width <= 0:
        raise OcculusValidationError(f"width must be positive, got {width}")
    if resolution <= 0:
        raise OcculusValidationError(f"resolution must be positive, got {resolution}")


def _polyline_length(polyline_2d: NDArray[np.float64]) -> float:
    """Compute the total 2D length of a polyline.

    Parameters
    ----------
    polyline_2d : NDArray[np.float64]
        (P, 2) polyline vertices.

    Returns
    -------
    float
        Total length.
    """
    diffs = np.diff(polyline_2d, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    return float(segment_lengths.sum())


def _project_onto_polyline(
    points: NDArray[np.float64],
    polyline_2d: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project 2D points onto a polyline, returning station and distance.

    For each point, finds the closest projection onto any polyline segment
    and returns the cumulative station distance along the polyline and the
    perpendicular distance from the polyline.

    Parameters
    ----------
    points : NDArray[np.float64]
        (N, 2) array of XY coordinates.
    polyline_2d : NDArray[np.float64]
        (P, 2) polyline vertices.

    Returns
    -------
    stations : NDArray[np.float64]
        (N,) cumulative station distance along the polyline for each point.
    distances : NDArray[np.float64]
        (N,) perpendicular distance from the polyline for each point.
    """
    n_pts = points.shape[0]
    n_segs = polyline_2d.shape[0] - 1

    best_station = np.full(n_pts, np.inf, dtype=np.float64)
    best_distance = np.full(n_pts, np.inf, dtype=np.float64)

    cumulative_length = 0.0

    for i in range(n_segs):
        a = polyline_2d[i]
        b = polyline_2d[i + 1]
        ab = b - a
        seg_len = float(np.sqrt(ab @ ab))

        if seg_len < 1e-12:
            continue

        ab_unit = ab / seg_len

        # Vector from segment start to each point
        ap = points - a

        # Parameter t along the segment [0, 1]
        t = np.clip((ap @ ab_unit) / seg_len, 0.0, 1.0)

        # Closest point on segment
        proj = a + np.outer(t, ab)
        dist = np.sqrt(((points - proj) ** 2).sum(axis=1))

        station = cumulative_length + t * seg_len

        closer = dist < best_distance
        best_station[closer] = station[closer]
        best_distance[closer] = dist[closer]

        cumulative_length += seg_len

    return best_station, best_distance


def _point_and_perp_at_station(
    polyline_2d: NDArray[np.float64],
    station: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find the 2D point and perpendicular unit vector at a station distance.

    Parameters
    ----------
    polyline_2d : NDArray[np.float64]
        (P, 2) polyline vertices.
    station : float
        Cumulative distance along the polyline.

    Returns
    -------
    point : NDArray[np.float64]
        (2,) XY coordinates at the station.
    perp : NDArray[np.float64]
        (2,) perpendicular unit vector (rotated 90 degrees CCW from tangent).
    """
    cumulative = 0.0
    n_segs = polyline_2d.shape[0] - 1

    for i in range(n_segs):
        a = polyline_2d[i]
        b = polyline_2d[i + 1]
        ab = b - a
        seg_len = float(np.sqrt(ab @ ab))

        if seg_len < 1e-12:
            continue

        if cumulative + seg_len >= station or i == n_segs - 1:
            t = min((station - cumulative) / seg_len, 1.0)
            point = a + t * ab
            tangent = ab / seg_len
            perp = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            return point, perp

        cumulative += seg_len

    # Fallback: return endpoint with last segment direction
    ab = polyline_2d[-1] - polyline_2d[-2]
    seg_len = float(np.sqrt(ab @ ab))
    if seg_len < 1e-12:
        return polyline_2d[-1].copy(), np.array([0.0, 1.0], dtype=np.float64)
    tangent = ab / seg_len
    perp = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    return polyline_2d[-1].copy(), perp
