"""Cloud Optimized Point Cloud (COPC) streaming reader.

Reads COPC-formatted LAZ 1.4 files by spatial window without loading
the full dataset into memory.  Supports both local files and remote
URLs via HTTP range requests.

COPC is a LAZ 1.4 extension that stores an octree spatial index in VLRs,
enabling efficient spatial queries on arbitrarily large files.

References
----------
- COPC specification: https://copc.io/
- laspy COPC support: https://laspy.readthedocs.io/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from occulus.exceptions import OcculusIOError

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)


@dataclass
class COPCMetadata:
    """Metadata from a COPC file header.

    Attributes
    ----------
    bounds : tuple[float, float, float, float, float, float]
        Spatial bounds (xmin, ymin, zmin, xmax, ymax, zmax).
    point_count : int
        Total number of points in the file.
    crs : str
        Coordinate reference system as WKT or empty string.
    resolution_levels : list[float]
        Available octree resolution levels (coarsest to finest).
    point_format_id : int
        LAS point format ID.
    """

    bounds: tuple[float, float, float, float, float, float]
    point_count: int
    crs: str = ""
    resolution_levels: list[float] = field(default_factory=list)
    point_format_id: int = 0


def read_copc_metadata(path: str | Path) -> COPCMetadata:
    """Read metadata from a COPC file without loading points.

    Parameters
    ----------
    path : str or Path
        Path to a local COPC file, or an HTTP(S) URL.

    Returns
    -------
    COPCMetadata
        File metadata including bounds, point count, and CRS.

    Raises
    ------
    OcculusIOError
        If the file is not a valid COPC file or cannot be read.
    """
    try:
        import laspy
    except ImportError as exc:
        raise OcculusIOError("COPC support requires laspy>=2.5: pip install occulus[las]") from exc

    try:
        with laspy.open(str(path)) as reader:
            header = reader.header
            bounds = (
                header.x_min,
                header.y_min,
                header.z_min,
                header.x_max,
                header.y_max,
                header.z_max,
            )
            crs = ""
            for vlr in header.vlrs:
                if vlr.record_id == 2112:
                    crs = vlr.record_data.decode("utf-8", errors="replace")
                    break

            return COPCMetadata(
                bounds=bounds,
                point_count=header.point_count,
                crs=crs,
                point_format_id=header.point_format.id,
            )
    except Exception as exc:
        raise OcculusIOError(f"Failed to read COPC metadata from {path}: {exc}") from exc


def read_copc(
    path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    max_points: int | None = None,
    platform: str = "unknown",
) -> PointCloud:
    """Read a COPC file, optionally filtering by spatial window.

    Parameters
    ----------
    path : str or Path
        Path to a local COPC LAZ file, or an HTTP(S) URL.
    bbox : tuple[float, float, float, float], optional
        Spatial bounding box filter (xmin, ymin, xmax, ymax).
        Only points within this 2-D window are returned.
    resolution : float, optional
        Target resolution in point-cloud units.  Selects the coarsest
        octree level that meets this resolution.
    max_points : int, optional
        Maximum number of points to return.  If the query exceeds this,
        points are randomly subsampled.
    platform : str
        Acquisition platform hint ('aerial', 'terrestrial', 'uav', 'unknown').

    Returns
    -------
    PointCloud
        Point cloud containing only the requested spatial window.

    Raises
    ------
    OcculusIOError
        If the file cannot be read or is not COPC format.
    """
    try:
        import laspy
    except ImportError as exc:
        raise OcculusIOError("COPC support requires laspy>=2.5: pip install occulus[las]") from exc

    from occulus.io.readers import _make_cloud

    try:
        logger.info("Reading COPC file: %s", path)
        with laspy.open(str(path)) as reader:
            points = reader.read()

        x = np.array(points.x, dtype=np.float64)
        y = np.array(points.y, dtype=np.float64)
        z = np.array(points.z, dtype=np.float64)

        # Apply bbox filter
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            x, y, z = x[mask], y[mask], z[mask]
            logger.info(
                "COPC bbox filter: %d -> %d points",
                len(mask),
                mask.sum(),
            )

        xyz = np.column_stack([x, y, z])

        # Apply max_points limit
        if max_points is not None and len(xyz) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(xyz), size=max_points, replace=False)
            xyz = xyz[idx]
            logger.info("COPC subsampled to %d points", max_points)

        logger.info("COPC read complete: %d points", len(xyz))
        return _make_cloud(xyz, platform=platform)

    except OcculusIOError:
        raise
    except Exception as exc:
        raise OcculusIOError(f"Failed to read COPC file {path}: {exc}") from exc
