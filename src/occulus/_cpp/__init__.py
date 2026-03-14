"""C++ backend bindings via pybind11.

This package wraps the compiled C++ core library. The actual C++ source
lives in ``cpp/`` at the repository root and is built via CMake + pybind11.

If the compiled extension is not available (e.g., pure-Python fallback),
functions in this module raise ``OcculusCppError`` with install instructions.

Submodules (when compiled):

    occulus._cpp.kdtree       — k-d tree construction and queries
    occulus._cpp.registration — ICP point-to-point and point-to-plane
    occulus._cpp.segmentation — CSF, PMF ground classification
    occulus._cpp.mesh         — Poisson, BPA surface reconstruction
    occulus._cpp.features     — RANSAC plane/cylinder fitting
    occulus._cpp.normals      — PCA normal estimation
    occulus._cpp.filters      — SOR, voxel grid, radius outlier
"""

from __future__ import annotations

import logging

from occulus.exceptions import OcculusCppError

logger = logging.getLogger(__name__)

_CPP_AVAILABLE = False

try:
    from occulus._cpp._core import (  # type: ignore[import-not-found]
        features,
        filters,
        kdtree,
        mesh,
        normals,
        registration,
        segmentation,
    )

    _CPP_AVAILABLE = True
except ImportError:
    logger.debug("C++ backend not available — using pure-Python fallbacks")


def require_cpp(operation: str) -> None:
    """Raise an error if the C++ backend is not available.

    Parameters
    ----------
    operation : str
        Name of the operation requiring C++.

    Raises
    ------
    OcculusCppError
        Always, if C++ is not compiled.
    """
    if not _CPP_AVAILABLE:
        raise OcculusCppError(
            f"The C++ backend is required for '{operation}' but is not installed. "
            "Rebuild with: pip install occulus --no-binary :all: "
            "or install a prebuilt wheel for your platform."
        )
