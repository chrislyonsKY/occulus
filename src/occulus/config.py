"""Configuration and constants for Occulus.

All configurable values live here. Never hardcode paths, thresholds,
or environment-specific values anywhere else in the codebase.

Environment Variables
---------------------
OCCULUS_NUM_THREADS
    Number of threads for parallel operations. Defaults to CPU count.

OCCULUS_LOG_LEVEL
    Logging verbosity. Defaults to WARNING.

"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_NUM_THREADS: int = os.cpu_count() or 4
DEFAULT_LOG_LEVEL: str = "WARNING"

# Registration defaults
DEFAULT_ICP_MAX_ITERATIONS: int = 50
DEFAULT_ICP_TOLERANCE: float = 1e-6
DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE: float = 1.0

# Segmentation defaults
DEFAULT_GROUND_CLOTH_RESOLUTION: float = 2.0  # meters — CSF default for aerial
DEFAULT_GROUND_MAX_ANGLE: float = 15.0  # degrees
DEFAULT_MIN_SEGMENT_POINTS: int = 50

# Mesh defaults
DEFAULT_POISSON_DEPTH: int = 8
DEFAULT_BPA_RADII_FACTOR: float = 2.0

# Filter defaults
DEFAULT_SOR_K_NEIGHBORS: int = 20
DEFAULT_SOR_STD_RATIO: float = 2.0
DEFAULT_VOXEL_SIZE: float = 0.05  # meters

# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

NUM_THREADS: int = int(os.environ.get("OCCULUS_NUM_THREADS", DEFAULT_NUM_THREADS))
LOG_LEVEL: str = os.environ.get("OCCULUS_LOG_LEVEL", DEFAULT_LOG_LEVEL)

if NUM_THREADS != DEFAULT_NUM_THREADS:
    logger.debug("Using custom thread count from environment: %d", NUM_THREADS)
