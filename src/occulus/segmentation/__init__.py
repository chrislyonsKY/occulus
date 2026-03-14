"""Point cloud segmentation — ground classification and object extraction.

Primary interface::

    from occulus.segmentation import (
        classify_ground_csf,
        classify_ground_pmf,
        cluster_dbscan,
        segment_trees,
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
"""

from occulus.segmentation.ground import classify_ground_csf, classify_ground_pmf
from occulus.segmentation.objects import SegmentationResult, cluster_dbscan, segment_trees

__all__ = [
    "classify_ground_csf",
    "classify_ground_pmf",
    "cluster_dbscan",
    "segment_trees",
    "SegmentationResult",
]
