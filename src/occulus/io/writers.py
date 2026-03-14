"""Point cloud file writers.

Supports LAS/LAZ (via laspy, optional), PLY (via Open3D, optional),
and XYZ/CSV/TXT (via NumPy, always available).

The public entry point is :func:`write`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from occulus.exceptions import OcculusIOError, OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".las", ".laz", ".ply", ".xyz", ".txt", ".csv"}


def write(
    cloud: PointCloud,
    path: str | Path,
    *,
    compress: bool | None = None,
) -> Path:
    """Write a point cloud to file.

    Parameters
    ----------
    cloud : PointCloud
        The point cloud to write. Must have a valid ``xyz`` array.
    path : str | Path
        Output file path. Format is inferred from the extension.
    compress : bool | None, optional
        For LAS output, whether to compress to LAZ format. If ``None``,
        compression is inferred from the extension (``.laz`` → compressed).

    Returns
    -------
    Path
        The resolved path of the written file.

    Raises
    ------
    OcculusIOError
        If the file cannot be written.
    OcculusValidationError
        If the format is not supported.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in _SUPPORTED_EXTENSIONS:
        raise OcculusValidationError(
            f"Unsupported output format '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    logger.debug("Writing %d points to %s", cloud.n_points, path)

    if ext in (".las", ".laz"):
        compressed = compress if compress is not None else (ext == ".laz")
        return _write_las(cloud, path, compress=compressed)
    elif ext == ".ply":
        return _write_ply(cloud, path)
    else:  # .xyz, .txt, .csv
        return _write_xyz(cloud, path)


# ---------------------------------------------------------------------------
# Format-specific writers
# ---------------------------------------------------------------------------


def _write_las(cloud: PointCloud, path: Path, *, compress: bool) -> Path:
    """Write a point cloud to LAS or LAZ format via laspy.

    Parameters
    ----------
    cloud : PointCloud
        Source point cloud.
    path : Path
        Output file path.
    compress : bool
        Whether to write LAZ (compressed) instead of LAS.

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    ImportError
        If ``laspy`` is not installed.
    OcculusIOError
        If the file cannot be written.
    """
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "laspy is required for LAS/LAZ output: pip install 'occulus[las]'"
        ) from exc

    # Determine LAS point format: use 2 if RGB present, else 0
    has_rgb = cloud.rgb is not None
    point_format_id = 2 if has_rgb else 0

    try:
        header = laspy.LasHeader(point_format=point_format_id, version="1.4")
        # Set scale/offset to preserve precision for large coordinate values
        xyz_min = cloud.xyz.min(axis=0)
        header.offsets = xyz_min
        header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

        las = laspy.LasData(header=header)
        las.x = cloud.xyz[:, 0]
        las.y = cloud.xyz[:, 1]
        las.z = cloud.xyz[:, 2]

        if cloud.intensity is not None:
            # LAS intensity is uint16 (0–65535); scale float intensity if needed
            intensity = cloud.intensity
            if intensity.max() <= 1.0:
                intensity = intensity * 65535.0
            las.intensity = intensity.clip(0, 65535).astype(np.uint16)

        if cloud.classification is not None:
            las.classification = cloud.classification

        if cloud.return_number is not None:
            las.return_number = cloud.return_number

        if cloud.number_of_returns is not None:
            las.number_of_returns = cloud.number_of_returns

        if has_rgb and cloud.rgb is not None:
            # LAS stores 16-bit color
            las.red = cloud.rgb[:, 0].astype(np.uint16) * 257
            las.green = cloud.rgb[:, 1].astype(np.uint16) * 257
            las.blue = cloud.rgb[:, 2].astype(np.uint16) * 257

        las.write(str(path), do_compress=compress)

    except Exception as exc:
        raise OcculusIOError(f"Failed to write LAS/LAZ file {path}: {exc}") from exc

    logger.info("Wrote %d points to %s (compressed=%s)", cloud.n_points, path, compress)
    return path


def _write_ply(cloud: PointCloud, path: Path) -> Path:
    """Write a point cloud to PLY format via Open3D.

    Parameters
    ----------
    cloud : PointCloud
        Source point cloud.
    path : Path
        Output file path.

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    ImportError
        If ``open3d`` is not installed.
    OcculusIOError
        If the file cannot be written.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for PLY output: pip install 'occulus[viz]'") from exc

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(cloud.xyz))

        if cloud.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(cloud.normals))
        if cloud.rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(cloud.rgb.astype(np.float64) / 255.0)
            )

        o3d.io.write_point_cloud(str(path), pcd)
    except Exception as exc:
        raise OcculusIOError(f"Failed to write PLY file {path}: {exc}") from exc

    logger.info("Wrote %d points to %s", cloud.n_points, path)
    return path


def _write_xyz(cloud: PointCloud, path: Path) -> Path:
    """Write a point cloud to a delimited text file via NumPy.

    Writes X, Y, Z columns, with an optional intensity column if present.
    CSV files use comma delimiters; all others use space delimiters.

    Parameters
    ----------
    cloud : PointCloud
        Source point cloud.
    path : Path
        Output file path.

    Returns
    -------
    Path
        Resolved output path.

    Raises
    ------
    OcculusIOError
        If the file cannot be written.
    """
    delimiter = "," if path.suffix.lower() == ".csv" else " "

    if cloud.intensity is not None:
        data = np.column_stack((cloud.xyz, cloud.intensity))
        header = "x y z intensity" if delimiter == " " else "x,y,z,intensity"
    else:
        data = cloud.xyz
        header = "x y z" if delimiter == " " else "x,y,z"

    try:
        np.savetxt(
            str(path),
            data,
            delimiter=delimiter,
            header=header,
            comments="# ",
            fmt="%.6f",
        )
    except Exception as exc:
        raise OcculusIOError(f"Failed to write text file {path}: {exc}") from exc

    logger.info("Wrote %d points to %s", cloud.n_points, path)
    return path
