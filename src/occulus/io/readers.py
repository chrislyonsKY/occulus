"""Point cloud file readers.

Supports LAS/LAZ (via laspy), PLY, PCD (via Open3D, optional), and
XYZ/CSV/TXT (via NumPy). Format is auto-detected from file extension.

The public entry point is :func:`read`. All private ``_read_*`` helpers
accept the same ``platform`` and ``subsample`` keyword arguments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusIOError, OcculusValidationError
from occulus.types import (
    AcquisitionMetadata,
    AerialCloud,
    Platform,
    PointCloud,
    TerrestrialCloud,
    UAVCloud,
)

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".las", ".laz", ".ply", ".pcd", ".xyz", ".txt", ".csv"}


def read(
    path: str | Path,
    *,
    platform: Platform | str = Platform.UNKNOWN,
    subsample: float | None = None,
) -> PointCloud:
    """Read a point cloud file and return the appropriate PointCloud subtype.

    Parameters
    ----------
    path : str | Path
        Path to the point cloud file. Format is inferred from the extension.
    platform : Platform | str, optional
        Acquisition platform hint. When provided, returns the appropriate
        subtype (AerialCloud, TerrestrialCloud, UAVCloud). Defaults to
        ``Platform.UNKNOWN`` which returns the base PointCloud.
    subsample : float | None, optional
        If provided, randomly subsample to this fraction of points (0.0–1.0)
        after reading. Useful for quickly exploring large files.

    Returns
    -------
    PointCloud
        The loaded point cloud (or platform-specific subtype).

    Raises
    ------
    OcculusIOError
        If the file does not exist or cannot be read.
    OcculusValidationError
        If the file extension is not supported or ``subsample`` is out of range.
    """
    path = Path(path)
    if not path.exists():
        raise OcculusIOError(f"File not found: {path}")

    if subsample is not None and not (0.0 < subsample <= 1.0):
        raise OcculusValidationError(
            f"subsample must be in (0.0, 1.0], got {subsample}"
        )

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise OcculusValidationError(
            f"Unsupported format '{ext}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    logger.debug("Reading %s (platform=%s)", path, platform)

    if ext in (".las", ".laz"):
        return _read_las(path, platform=platform, subsample=subsample)
    elif ext == ".ply":
        return _read_ply(path, platform=platform, subsample=subsample)
    elif ext == ".pcd":
        return _read_pcd(path, platform=platform, subsample=subsample)
    else:  # .xyz, .txt, .csv
        return _read_xyz(path, platform=platform, subsample=subsample)


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------


def _read_las(
    path: Path,
    *,
    platform: Platform | str,
    subsample: float | None,
) -> PointCloud:
    """Read a LAS or LAZ file via laspy.

    Parameters
    ----------
    path : Path
        Path to the LAS/LAZ file.
    platform : Platform | str
        Acquisition platform hint.
    subsample : float | None
        Random subsample fraction.

    Returns
    -------
    PointCloud
        Loaded point cloud with all available LAS attributes populated.

    Raises
    ------
    ImportError
        If ``laspy`` is not installed.
    OcculusIOError
        If the file cannot be parsed.
    """
    try:
        import laspy
    except ImportError as exc:
        raise ImportError(
            "laspy is required for LAS/LAZ support: pip install 'occulus[las]'"
        ) from exc

    try:
        las = laspy.read(str(path))
    except Exception as exc:
        raise OcculusIOError(f"Failed to read LAS/LAZ file {path}: {exc}") from exc

    try:
        xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    except Exception as exc:
        raise OcculusIOError(f"Failed to extract XYZ from {path}: {exc}") from exc

    intensity: NDArray[np.float64] | None = None
    classification: NDArray[np.uint8] | None = None
    return_number: NDArray[np.uint8] | None = None
    number_of_returns: NDArray[np.uint8] | None = None
    rgb: NDArray[np.uint8] | None = None

    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)
    if hasattr(las, "classification"):
        classification = np.asarray(las.classification, dtype=np.uint8)
    if hasattr(las, "return_number"):
        return_number = np.asarray(las.return_number, dtype=np.uint8)
    if hasattr(las, "number_of_returns"):
        number_of_returns = np.asarray(las.number_of_returns, dtype=np.uint8)
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # LAS stores 16-bit color; scale to 8-bit
        r = np.asarray(las.red, dtype=np.uint16)
        g = np.asarray(las.green, dtype=np.uint16)
        b = np.asarray(las.blue, dtype=np.uint16)
        scale = 255.0 / max(r.max(), g.max(), b.max(), 1)
        rgb = np.vstack((r, g, b)).T.astype(np.float32)
        rgb = (rgb * scale).clip(0, 255).astype(np.uint8)

    metadata = _metadata_from_las_header(las.header, platform)

    if subsample is not None:
        mask = _subsample_mask(len(xyz), subsample)
        xyz = xyz[mask]
        if intensity is not None:
            intensity = intensity[mask]
        if classification is not None:
            classification = classification[mask]
        if return_number is not None:
            return_number = return_number[mask]
        if number_of_returns is not None:
            number_of_returns = number_of_returns[mask]
        if rgb is not None:
            rgb = rgb[mask]

    logger.info("Read %d points from %s", len(xyz), path)
    return _make_cloud(
        xyz,
        platform,
        intensity=intensity,
        classification=classification,
        return_number=return_number,
        number_of_returns=number_of_returns,
        rgb=rgb,
        metadata=metadata,
    )


def _read_ply(
    path: Path,
    *,
    platform: Platform | str,
    subsample: float | None,
) -> PointCloud:
    """Read a PLY file via Open3D.

    Parameters
    ----------
    path : Path
        Path to the PLY file.
    platform : Platform | str
        Acquisition platform hint.
    subsample : float | None
        Random subsample fraction.

    Returns
    -------
    PointCloud
        Loaded point cloud. Normals and colors included if present in file.

    Raises
    ------
    ImportError
        If ``open3d`` is not installed.
    OcculusIOError
        If the file cannot be parsed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for PLY support: pip install 'occulus[viz]'"
        ) from exc

    try:
        pcd = o3d.io.read_point_cloud(str(path))
    except Exception as exc:
        raise OcculusIOError(f"Failed to read PLY file {path}: {exc}") from exc

    xyz = np.asarray(pcd.points, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise OcculusIOError(f"PLY file contains no points: {path}")

    normals: NDArray[np.float64] | None = None
    rgb: NDArray[np.uint8] | None = None

    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

    if subsample is not None:
        mask = _subsample_mask(len(xyz), subsample)
        xyz = xyz[mask]
        if normals is not None:
            normals = normals[mask]
        if rgb is not None:
            rgb = rgb[mask]

    logger.info("Read %d points from %s", len(xyz), path)
    return _make_cloud(xyz, platform, normals=normals, rgb=rgb)


def _read_pcd(
    path: Path,
    *,
    platform: Platform | str,
    subsample: float | None,
) -> PointCloud:
    """Read a PCD file via Open3D.

    Parameters
    ----------
    path : Path
        Path to the PCD file.
    platform : Platform | str
        Acquisition platform hint.
    subsample : float | None
        Random subsample fraction.

    Returns
    -------
    PointCloud
        Loaded point cloud.

    Raises
    ------
    ImportError
        If ``open3d`` is not installed.
    OcculusIOError
        If the file cannot be parsed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for PCD support: pip install 'occulus[viz]'"
        ) from exc

    try:
        pcd = o3d.io.read_point_cloud(str(path))
    except Exception as exc:
        raise OcculusIOError(f"Failed to read PCD file {path}: {exc}") from exc

    xyz = np.asarray(pcd.points, dtype=np.float64)
    if xyz.shape[0] == 0:
        raise OcculusIOError(f"PCD file contains no points: {path}")

    normals: NDArray[np.float64] | None = None
    rgb: NDArray[np.uint8] | None = None

    if pcd.has_normals():
        normals = np.asarray(pcd.normals, dtype=np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

    if subsample is not None:
        mask = _subsample_mask(len(xyz), subsample)
        xyz = xyz[mask]
        if normals is not None:
            normals = normals[mask]
        if rgb is not None:
            rgb = rgb[mask]

    logger.info("Read %d points from %s", len(xyz), path)
    return _make_cloud(xyz, platform, normals=normals, rgb=rgb)


def _read_xyz(
    path: Path,
    *,
    platform: Platform | str,
    subsample: float | None,
) -> PointCloud:
    """Read an XYZ, TXT, or CSV file via NumPy.

    The first three columns are interpreted as X, Y, Z. An optional fourth
    column is interpreted as intensity. Lines beginning with ``#`` are treated
    as comments and skipped. Delimiters are auto-detected (whitespace or comma).

    Parameters
    ----------
    path : Path
        Path to the delimited text file.
    platform : Platform | str
        Acquisition platform hint.
    subsample : float | None
        Random subsample fraction.

    Returns
    -------
    PointCloud
        Loaded point cloud.

    Raises
    ------
    OcculusIOError
        If the file cannot be parsed or has fewer than 3 columns.
    """
    # Detect delimiter from extension
    delimiter: str | None = "," if path.suffix.lower() == ".csv" else None

    try:
        data = np.loadtxt(str(path), comments="#", delimiter=delimiter)
    except ValueError:
        # Try comma delimiter as fallback
        try:
            data = np.loadtxt(str(path), comments="#", delimiter=",")
        except Exception as exc:
            raise OcculusIOError(
                f"Failed to parse text point cloud {path}: {exc}"
            ) from exc
    except Exception as exc:
        raise OcculusIOError(
            f"Failed to read text point cloud {path}: {exc}"
        ) from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise OcculusIOError(
            f"Expected at least 3 columns (X Y Z), got {data.shape[1]} in {path}"
        )

    xyz = data[:, :3].astype(np.float64)
    intensity: NDArray[np.float64] | None = None
    if data.shape[1] >= 4:
        intensity = data[:, 3].astype(np.float64)

    if subsample is not None:
        mask = _subsample_mask(len(xyz), subsample)
        xyz = xyz[mask]
        if intensity is not None:
            intensity = intensity[mask]

    logger.info("Read %d points from %s", len(xyz), path)
    return _make_cloud(xyz, platform, intensity=intensity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata_from_las_header(header: object, platform: Platform | str) -> AcquisitionMetadata:
    """Build AcquisitionMetadata from a laspy LasHeader object.

    Parameters
    ----------
    header : object
        A ``laspy.LasHeader`` instance.
    platform : Platform | str
        Acquisition platform hint.

    Returns
    -------
    AcquisitionMetadata
        Populated metadata object.
    """
    if isinstance(platform, str):
        try:
            platform = Platform(platform.lower())
        except ValueError:
            platform = Platform.UNKNOWN

    meta = AcquisitionMetadata(platform=platform)

    # laspy 2.x exposes these as attributes; guard with getattr for safety
    system_id = getattr(header, "system_identifier", "")
    if isinstance(system_id, (bytes, bytearray)):
        system_id = system_id.decode("ascii", errors="ignore").strip("\x00").strip()
    meta.scanner_model = system_id or ""

    crs_wkt = ""
    if hasattr(header, "parse_crs"):
        try:
            crs = header.parse_crs()
            if crs is not None:
                crs_wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else str(crs)
        except Exception:
            pass
    meta.coordinate_system = crs_wkt

    return meta


def _subsample_mask(n: int, fraction: float) -> NDArray[np.bool_]:
    """Return a boolean mask selecting ``fraction`` of ``n`` indices.

    Parameters
    ----------
    n : int
        Total number of points.
    fraction : float
        Fraction to keep, in (0.0, 1.0].

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask of length ``n``.
    """
    rng = np.random.default_rng()
    k = max(1, int(n * fraction))
    idx = rng.choice(n, size=k, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def _make_cloud(
    xyz: NDArray[np.float64],
    platform: Platform | str,
    **kwargs: object,
) -> PointCloud:
    """Construct the appropriate PointCloud subtype based on platform.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Point coordinates as (N, 3) array.
    platform : Platform | str
        Acquisition platform identifier.
    **kwargs
        Additional keyword arguments forwarded to the constructor
        (e.g., ``intensity``, ``classification``, ``rgb``, ``metadata``).

    Returns
    -------
    PointCloud
        The platform-appropriate subtype instance.
    """
    if isinstance(platform, str):
        try:
            platform = Platform(platform.lower())
        except ValueError:
            platform = Platform.UNKNOWN

    if platform == Platform.AERIAL:
        return AerialCloud(xyz, **kwargs)  # type: ignore[arg-type]
    elif platform == Platform.TERRESTRIAL:
        return TerrestrialCloud(xyz, **kwargs)  # type: ignore[arg-type]
    elif platform == Platform.UAV:
        return UAVCloud(xyz, **kwargs)  # type: ignore[arg-type]
    else:
        return PointCloud(xyz, **kwargs)  # type: ignore[arg-type]
