"""Point cloud segmentation — ground classification and object extraction.

Primary interface::

    from occulus.segmentation import (
        classify_ground_csf,
        classify_ground_pmf,
        cluster_dbscan,
        detect_powerlines,
        segment_trees,
        PowerlineResult,
        SegmentationResult,
    )

Ground classification
---------------------
- :func:`classify_ground_csf` — Cloth Simulation Filter (aerial/UAV preferred)
- :func:`classify_ground_pmf` — Progressive Morphological Filter

Object segmentation
-------------------
- :func:`cluster_dbscan` — density-based clustering (general purpose)
- :func:`segment_trees` — CHM-based individual tree delineation (aerial/UAV)

Infrastructure detection
------------------------
- :func:`detect_powerlines` — wire and pylon extraction from classified clouds
"""

from occulus.segmentation.ground import classify_ground_csf, classify_ground_pmf
from occulus.segmentation.objects import SegmentationResult, cluster_dbscan, segment_trees
from occulus.segmentation.powerlines import PowerlineResult, detect_powerlines

__all__ = [
    "PowerlineResult",
    "SegmentationResult",
    "classify_ground_csf",
    "classify_ground_pmf",
    "cluster_dbscan",
    "detect_powerlines",
    "segment_trees",
]
