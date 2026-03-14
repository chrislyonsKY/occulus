"""Exception hierarchy for Occulus.

All domain exceptions inherit from ``OcculusError`` so callers can
catch the base class to handle any library error:

    try:
        cloud = occulus.read("scan.laz")
    except occulus.OcculusError as exc:
        logger.error("Occulus operation failed: %s", exc)

"""


class OcculusError(Exception):
    """Base exception for all Occulus errors."""


class OcculusIOError(OcculusError):
    """Raised when reading or writing point cloud files fails."""


class OcculusValidationError(OcculusError):
    """Raised when input parameters fail validation."""


class OcculusRegistrationError(OcculusError):
    """Raised when point cloud registration fails to converge."""


class OcculusSegmentationError(OcculusError):
    """Raised when segmentation produces invalid or empty results."""


class OcculusMeshError(OcculusError):
    """Raised when surface reconstruction fails."""


class OcculusFeatureError(OcculusError):
    """Raised when geometric feature extraction fails."""


class OcculusCppError(OcculusError):
    """Raised when the C++ backend encounters an unrecoverable error."""


class OcculusNetworkError(OcculusError):
    """Raised when a network or remote data-access operation fails."""


class UnsupportedPlatformError(OcculusError):
    """Raised when an operation is not supported for the given acquisition platform."""
