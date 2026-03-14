"""Occulus — Multi-platform point cloud analysis.

Registration, segmentation, meshing, and feature extraction for
aerial, terrestrial, and UAV LiDAR point clouds.

Quick start::

    import occulus

    cloud = occulus.read("scan.laz", platform="terrestrial")
    cloud = occulus.estimate_normals(cloud)
    result = occulus.icp(source, target)

Platform-aware types::

    from occulus.types import PointCloud, AerialCloud, TerrestrialCloud, UAVCloud

"""

from occulus._version import __version__
from occulus.io import read, write
from occulus.types import (
    AcquisitionMetadata,
    AerialCloud,
    Platform,
    PointCloud,
    ScanPosition,
    TerrestrialCloud,
    UAVCloud,
)

__all__ = [
    "__version__",
    # I/O
    "read",
    "write",
    # Types
    "PointCloud",
    "AerialCloud",
    "TerrestrialCloud",
    "UAVCloud",
    "Platform",
    "ScanPosition",
    "AcquisitionMetadata",
]
