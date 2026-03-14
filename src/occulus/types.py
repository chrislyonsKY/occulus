"""Core point cloud types for Occulus.

Platform-aware class hierarchy::

    PointCloud          — base class, platform-agnostic
    ├── AerialCloud     — aerial LiDAR (nadir perspective, large area, moderate density)
    ├── TerrestrialCloud — TLS (scan positions, high density, occlusion-aware)
    └── UAVCloud        — UAV/drone (oblique angles, SfM or LiDAR, variable density)

Each subtype carries acquisition metadata and provides platform-appropriate
defaults for processing algorithms. The base ``PointCloud`` works for any
data where the acquisition platform is unknown or irrelevant.

All types store points as NumPy arrays internally. Open3D interop is
provided via ``to_open3d()`` and ``from_open3d()`` class methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusValidationError

if TYPE_CHECKING:
    pass  # Open3D has no type stubs; handled via lazy import at runtime

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Acquisition platform identifier."""

    UNKNOWN = "unknown"
    AERIAL = "aerial"
    TERRESTRIAL = "terrestrial"
    UAV = "uav"


@dataclass
class ScanPosition:
    """Scanner position and orientation for a single scan setup.

    Used primarily for terrestrial and mobile scanning where the
    instrument location is a first-class concern for registration
    and occlusion analysis.

    Parameters
    ----------
    x : float
        Easting (or X) of the scanner origin, in the cloud's CRS.
    y : float
        Northing (or Y) of the scanner origin.
    z : float
        Elevation (or Z) of the scanner origin.
    roll : float, optional
        Roll angle in degrees, by default 0.0.
    pitch : float, optional
        Pitch angle in degrees, by default 0.0.
    yaw : float, optional
        Yaw (heading) in degrees, by default 0.0.
    scan_id : str, optional
        Human-readable identifier for this setup, by default "".
    """

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    scan_id: str = ""

    def as_array(self) -> NDArray[np.float64]:
        """Return position as a (3,) numpy array [x, y, z].

        Returns
        -------
        NDArray[np.float64]
            Position vector of shape (3,).
        """
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class AcquisitionMetadata:
    """Metadata about how the point cloud was acquired.

    Populated from LAS header fields where available, or set manually
    for non-LAS sources.

    Parameters
    ----------
    platform : Platform, optional
        Acquisition platform, by default ``Platform.UNKNOWN``.
    scanner_model : str, optional
        Scanner model or system identifier, by default "".
    scan_date : str, optional
        ISO 8601 date string, by default "".
    coordinate_system : str, optional
        EPSG code or WKT string describing the CRS, by default "".
    point_density_per_sqm : float | None, optional
        Nominal point density in points per square metre, by default None.
    scan_positions : list[ScanPosition], optional
        Scanner setup positions (TLS/mobile), by default empty list.
    flight_altitude_m : float | None, optional
        Mean flight altitude above ground in metres (aerial/UAV), by default None.
    scan_angle_range : tuple[float, float] | None, optional
        Minimum and maximum scan angle (degrees), by default None.
    """

    platform: Platform = Platform.UNKNOWN
    scanner_model: str = ""
    scan_date: str = ""
    coordinate_system: str = ""  # EPSG code or WKT
    point_density_per_sqm: float | None = None
    scan_positions: list[ScanPosition] = field(default_factory=list)
    flight_altitude_m: float | None = None  # aerial/UAV only
    scan_angle_range: tuple[float, float] | None = None  # min/max scan angle


class PointCloud:
    """Base point cloud container — platform-agnostic.

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Point coordinates as (N, 3) array.
    intensity : NDArray[np.float64] | None, optional
        Intensity values as (N,) array.
    classification : NDArray[np.uint8] | None, optional
        Classification codes as (N,) array (ASPRS LAS standard).
    rgb : NDArray[np.uint8] | None, optional
        Color values as (N, 3) uint8 array (0–255 per channel).
    normals : NDArray[np.float64] | None, optional
        Unit normal vectors as (N, 3) array.
    return_number : NDArray[np.uint8] | None, optional
        Return number as (N,) array.
    number_of_returns : NDArray[np.uint8] | None, optional
        Total returns per pulse as (N,) array.
    metadata : AcquisitionMetadata | None, optional
        Acquisition metadata; an empty ``AcquisitionMetadata`` is used if not provided.

    Raises
    ------
    OcculusValidationError
        If ``xyz`` is not a valid (N, 3) array, or if any optional array has
        a length that does not match ``n_points``.
    """

    def __init__(
        self,
        xyz: NDArray[np.float64],
        *,
        intensity: NDArray[np.float64] | None = None,
        classification: NDArray[np.uint8] | None = None,
        rgb: NDArray[np.uint8] | None = None,
        normals: NDArray[np.float64] | None = None,
        return_number: NDArray[np.uint8] | None = None,
        number_of_returns: NDArray[np.uint8] | None = None,
        metadata: AcquisitionMetadata | None = None,
    ) -> None:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise OcculusValidationError(
                f"xyz must be (N, 3) array, got shape {xyz.shape}"
            )

        n = xyz.shape[0]
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64)

        # Validate optional per-point arrays
        def _check(arr: NDArray | None, name: str, expected_ndim: int, col: int | None) -> None:
            if arr is None:
                return
            if arr.shape[0] != n:
                raise OcculusValidationError(
                    f"{name} length {arr.shape[0]} does not match n_points={n}"
                )
            if arr.ndim != expected_ndim:
                raise OcculusValidationError(
                    f"{name} must be {expected_ndim}D, got {arr.ndim}D"
                )
            if col is not None and arr.shape[1] != col:
                raise OcculusValidationError(
                    f"{name} must have {col} columns, got {arr.shape[1]}"
                )

        _check(intensity, "intensity", 1, None)
        _check(classification, "classification", 1, None)
        _check(return_number, "return_number", 1, None)
        _check(number_of_returns, "number_of_returns", 1, None)
        _check(normals, "normals", 2, 3)
        _check(rgb, "rgb", 2, 3)

        self.intensity = intensity
        self.classification = classification
        self.rgb = rgb
        self.normals = normals
        self.return_number = return_number
        self.number_of_returns = number_of_returns
        self.metadata = metadata or AcquisitionMetadata()

    @property
    def platform(self) -> Platform:
        """The acquisition platform for this cloud."""
        return self.metadata.platform

    @property
    def n_points(self) -> int:
        """Number of points in the cloud."""
        return self.xyz.shape[0]

    @property
    def bounds(self) -> NDArray[np.float64]:
        """Axis-aligned bounding box as (2, 3) array: [[xmin,ymin,zmin],[xmax,ymax,zmax]]."""
        return np.array([self.xyz.min(axis=0), self.xyz.max(axis=0)])

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Centroid of the point cloud as (3,) array."""
        return self.xyz.mean(axis=0)

    @property
    def has_normals(self) -> bool:
        """Whether unit normal vectors are attached to this cloud."""
        return self.normals is not None

    @property
    def has_color(self) -> bool:
        """Whether RGB color is attached to this cloud."""
        return self.rgb is not None

    def to_open3d(self) -> object:
        """Convert to an Open3D PointCloud object.

        Returns
        -------
        open3d.geometry.PointCloud
            The point cloud in Open3D format, with normals and colors
            transferred if present.

        Raises
        ------
        ImportError
            If Open3D is not installed.
        """
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for to_open3d(): pip install 'occulus[viz]'"
            ) from exc

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(self.xyz))

        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.normals)
            )
        if self.rgb is not None:
            pcd.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.rgb.astype(np.float64) / 255.0)
            )

        logger.debug("Converted %d-point cloud to Open3D", self.n_points)
        return pcd

    @classmethod
    def from_open3d(cls, pcd: object, metadata: AcquisitionMetadata | None = None) -> PointCloud:
        """Create a PointCloud from an Open3D PointCloud object.

        Parameters
        ----------
        pcd : open3d.geometry.PointCloud
            Source Open3D point cloud.
        metadata : AcquisitionMetadata | None, optional
            Optional acquisition metadata to attach, by default None.

        Returns
        -------
        PointCloud
            New Occulus PointCloud with xyz, normals, and colors transferred.

        Raises
        ------
        ImportError
            If Open3D is not installed.
        OcculusValidationError
            If the Open3D object has no points.
        """
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for from_open3d(): pip install 'occulus[viz]'"
            ) from exc

        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise OcculusValidationError(
                f"Expected open3d.geometry.PointCloud, got {type(pcd)}"
            )

        xyz = np.asarray(pcd.points, dtype=np.float64)
        if xyz.shape[0] == 0:
            raise OcculusValidationError("Open3D PointCloud has no points")

        normals: NDArray[np.float64] | None = None
        rgb: NDArray[np.uint8] | None = None

        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float64)
        if pcd.has_colors():
            rgb = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

        logger.debug("Converted Open3D cloud (%d pts) to PointCloud", len(xyz))
        return cls(xyz, normals=normals, rgb=rgb, metadata=metadata)

    def __repr__(self) -> str:
        """Short string representation."""
        parts = [
            f"PointCloud({self.n_points:,} points",
            f"platform={self.platform.value}",
        ]
        if self.has_normals:
            parts.append("normals=True")
        if self.has_color:
            parts.append("rgb=True")
        return ", ".join(parts) + ")"

    def __len__(self) -> int:
        """Return the number of points."""
        return self.n_points


class AerialCloud(PointCloud):
    """Point cloud from aerial LiDAR acquisition (ALS).

    Provides aerial-specific capabilities:

    - Ground classification via ASPRS class 2 mask
    - First-return filtering (canopy top extraction)
    - Assumes nadir perspective with moderate density (10–50 pts/m²)
    - Return number / number_of_returns are meaningful attributes

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Point coordinates as (N, 3) array.
    **kwargs
        Additional keyword arguments passed to :class:`PointCloud`.
    """

    def __init__(self, xyz: NDArray[np.float64], **kwargs: object) -> None:
        metadata = kwargs.pop("metadata", None) or AcquisitionMetadata()  # type: ignore[arg-type]
        metadata.platform = Platform.AERIAL  # type: ignore[union-attr]
        kwargs["metadata"] = metadata
        super().__init__(xyz, **kwargs)  # type: ignore[arg-type]

    def ground_points(self) -> NDArray[np.bool_]:
        """Return a boolean mask of ASPRS class 2 (ground) points.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of length ``n_points`` where ``True`` = ground point.

        Raises
        ------
        OcculusValidationError
            If no classification array is present.
        """
        if self.classification is None:
            raise OcculusValidationError(
                "No classification array present. Run classify_ground_csf() first."
            )
        return self.classification == 2

    def first_returns(self) -> PointCloud:
        """Filter to first-return points only.

        First returns represent the uppermost surface intercepted by the laser
        pulse — useful for canopy height models and above-ground features.

        Returns
        -------
        PointCloud
            New cloud containing only first-return points.

        Raises
        ------
        OcculusValidationError
            If no return_number array is present.
        """
        if self.return_number is None:
            raise OcculusValidationError(
                "No return_number array present. Load from a LAS file with return information."
            )
        mask = self.return_number == 1
        return PointCloud(
            self.xyz[mask],
            intensity=self.intensity[mask] if self.intensity is not None else None,
            classification=self.classification[mask] if self.classification is not None else None,
            rgb=self.rgb[mask] if self.rgb is not None else None,
            normals=self.normals[mask] if self.normals is not None else None,
            return_number=self.return_number[mask],
            number_of_returns=self.number_of_returns[mask] if self.number_of_returns is not None else None,
            metadata=self.metadata,
        )


class TerrestrialCloud(PointCloud):
    """Point cloud from terrestrial laser scanning (TLS).

    Provides TLS-specific capabilities:

    - Scan positions are first-class (required for registration)
    - Occlusion-aware operations account for scan geometry
    - Very high point density (1 000–100 000+ pts/m²)
    - Ground classification uses different defaults than aerial

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Point coordinates as (N, 3) array.
    scan_positions : list[ScanPosition] | None, optional
        Scanner setup positions for this cloud, by default None.
    **kwargs
        Additional keyword arguments passed to :class:`PointCloud`.
    """

    def __init__(
        self,
        xyz: NDArray[np.float64],
        *,
        scan_positions: list[ScanPosition] | None = None,
        **kwargs: object,
    ) -> None:
        metadata = kwargs.pop("metadata", None) or AcquisitionMetadata()  # type: ignore[arg-type]
        metadata.platform = Platform.TERRESTRIAL  # type: ignore[union-attr]
        if scan_positions:
            metadata.scan_positions = scan_positions  # type: ignore[union-attr]
        kwargs["metadata"] = metadata
        super().__init__(xyz, **kwargs)  # type: ignore[arg-type]

    @property
    def scan_positions(self) -> list[ScanPosition]:
        """List of scanner setup positions for this cloud."""
        return self.metadata.scan_positions

    def viewpoint_mask(self, scan_index: int = 0) -> NDArray[np.bool_]:
        """Return a mask of points visible from a specific scan position.

        Uses Open3D's hidden-point removal algorithm (Katz et al. 2007).
        Points occluded from the viewpoint are excluded.

        Parameters
        ----------
        scan_index : int, optional
            Index into ``scan_positions`` list, by default 0.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of length ``n_points`` where ``True`` = visible.

        Raises
        ------
        OcculusValidationError
            If ``scan_index`` is out of range or no scan positions are recorded.
        ImportError
            If Open3D is not installed.
        """
        if not self.metadata.scan_positions:
            raise OcculusValidationError(
                "No scan positions recorded. Provide scan_positions when constructing TerrestrialCloud."
            )
        if scan_index >= len(self.metadata.scan_positions):
            raise OcculusValidationError(
                f"scan_index {scan_index} out of range; "
                f"{len(self.metadata.scan_positions)} positions available."
            )

        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for viewpoint_mask(): pip install 'occulus[viz]'"
            ) from exc

        pos = self.metadata.scan_positions[scan_index]
        viewpoint = pos.as_array()
        radius = np.linalg.norm(self.xyz - viewpoint, axis=1).max() * 100.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(self.xyz))

        _, visible_indices = pcd.hidden_point_removal(viewpoint.tolist(), radius)
        mask = np.zeros(self.n_points, dtype=bool)
        mask[np.asarray(visible_indices)] = True

        logger.debug(
            "viewpoint_mask: %d/%d points visible from scan position %s",
            mask.sum(), self.n_points, pos.scan_id or scan_index,
        )
        return mask


class UAVCloud(PointCloud):
    """Point cloud from UAV/drone acquisition (LiDAR or photogrammetric SfM).

    Provides UAV-specific capabilities:

    - Oblique viewing angles (not purely nadir)
    - Variable point density across the scene
    - May be photogrammetric (SfM) rather than LiDAR — different noise profile

    Parameters
    ----------
    xyz : NDArray[np.float64]
        Point coordinates as (N, 3) array.
    is_photogrammetric : bool, optional
        ``True`` if derived from SfM/photogrammetry rather than LiDAR,
        by default ``False``.
    **kwargs
        Additional keyword arguments passed to :class:`PointCloud`.
    """

    def __init__(
        self,
        xyz: NDArray[np.float64],
        *,
        is_photogrammetric: bool = False,
        **kwargs: object,
    ) -> None:
        metadata = kwargs.pop("metadata", None) or AcquisitionMetadata()  # type: ignore[arg-type]
        metadata.platform = Platform.UAV  # type: ignore[union-attr]
        kwargs["metadata"] = metadata
        super().__init__(xyz, **kwargs)  # type: ignore[arg-type]
        self.is_photogrammetric = is_photogrammetric
