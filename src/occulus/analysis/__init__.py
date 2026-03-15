"""Point cloud analysis — volume computation and cross-section extraction.

Available functions
-------------------
- :func:`compute_volume` — cut/fill volume between two surfaces
- :func:`extract_cross_section` — profile extraction along a polyline
- :func:`extract_profiles` — perpendicular profiles at regular intervals

Data classes
------------
- :class:`VolumeResult` — cut/fill/net volume and area statistics
- :class:`CrossSection` — station-elevation profile with source points

All implementations use pure NumPy. No optional dependencies required.
"""

from __future__ import annotations

from occulus.analysis.cross_section import (
    CrossSection,
    extract_cross_section,
    extract_profiles,
)
from occulus.analysis.volume import VolumeResult, compute_volume

__all__ = [
    "CrossSection",
    "VolumeResult",
    "compute_volume",
    "extract_cross_section",
    "extract_profiles",
]
