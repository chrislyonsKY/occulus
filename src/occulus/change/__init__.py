"""Multi-epoch change detection for point clouds.

Available algorithms
--------------------
- :func:`m3c2` — M3C2 signed distance with uncertainty (Lague et al. 2013)

All implementations use pure NumPy and SciPy.  No optional dependencies required.
"""

from occulus.change.m3c2 import M3C2Result, m3c2

__all__ = ["M3C2Result", "m3c2"]
