"""Point cloud I/O — read and write LAS, LAZ, PLY, PCD, and XYZ formats.

Primary interface:

    from occulus.io import read, write

    cloud = read("scan.laz")
    write(cloud, "output.las")

Delegates to format-specific backends. LAS/LAZ via laspy,
PLY/PCD via Open3D (optional), XYZ via NumPy.
"""

from occulus.io.readers import read
from occulus.io.writers import write

__all__ = ["read", "write"]
