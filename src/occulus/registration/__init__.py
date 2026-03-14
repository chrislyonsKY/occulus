"""Point cloud registration — alignment of multiple scans.

Primary interface::

    from occulus.registration import icp, ransac_registration, RegistrationResult
    from occulus.registration import align_scans, compute_fpfh

Algorithms
----------
- :func:`icp` — point-to-point or point-to-plane ICP (auto-selects based on normals)
- :func:`icp_point_to_point` — explicit point-to-point ICP
- :func:`icp_point_to_plane` — explicit point-to-plane ICP (requires target normals)
- :func:`compute_fpfh` — 33-dim FPFH feature descriptors for global registration
- :func:`ransac_registration` — feature-matching + RANSAC global alignment
- :func:`align_scans` — sequential multi-scan alignment
"""

from occulus.registration.global_registration import (
    AlignmentResult,
    align_scans,
    compute_fpfh,
    ransac_registration,
)
from occulus.registration.icp import (
    RegistrationResult,
    icp,
    icp_point_to_plane,
    icp_point_to_point,
)

__all__ = [
    "AlignmentResult",
    "RegistrationResult",
    "align_scans",
    "compute_fpfh",
    "icp",
    "icp_point_to_plane",
    "icp_point_to_point",
    "ransac_registration",
]
