"""Tests for the exception hierarchy."""

import pytest

from occulus.exceptions import (
    OcculusCppError,
    OcculusError,
    OcculusFeatureError,
    OcculusIOError,
    OcculusMeshError,
    OcculusNetworkError,
    OcculusRegistrationError,
    OcculusSegmentationError,
    OcculusValidationError,
    UnsupportedPlatformError,
)


def test_base_error_is_exception() -> None:
    """OcculusError inherits from Exception."""
    assert issubclass(OcculusError, Exception)


def test_network_error_is_base_error() -> None:
    """OcculusNetworkError is a subclass of OcculusError."""
    assert issubclass(OcculusNetworkError, OcculusError)


def test_validation_error_is_base_error() -> None:
    """OcculusValidationError is a subclass of OcculusError."""
    assert issubclass(OcculusValidationError, OcculusError)


def test_io_error_is_base_error() -> None:
    """OcculusIOError is a subclass of OcculusError."""
    assert issubclass(OcculusIOError, OcculusError)


def test_registration_error_is_base_error() -> None:
    """OcculusRegistrationError is a subclass of OcculusError."""
    assert issubclass(OcculusRegistrationError, OcculusError)


def test_segmentation_error_is_base_error() -> None:
    """OcculusSegmentationError is a subclass of OcculusError."""
    assert issubclass(OcculusSegmentationError, OcculusError)


def test_mesh_error_is_base_error() -> None:
    """OcculusMeshError is a subclass of OcculusError."""
    assert issubclass(OcculusMeshError, OcculusError)


def test_feature_error_is_base_error() -> None:
    """OcculusFeatureError is a subclass of OcculusError."""
    assert issubclass(OcculusFeatureError, OcculusError)


def test_cpp_error_is_base_error() -> None:
    """OcculusCppError is a subclass of OcculusError."""
    assert issubclass(OcculusCppError, OcculusError)


def test_unsupported_platform_is_base_error() -> None:
    """UnsupportedPlatformError is a subclass of OcculusError."""
    assert issubclass(UnsupportedPlatformError, OcculusError)


def test_can_raise_and_catch_base() -> None:
    """Any domain error can be caught as OcculusError."""
    with pytest.raises(OcculusError):
        raise OcculusNetworkError("connection refused")


def test_can_raise_and_catch_specific() -> None:
    """Specific errors can be caught by their own type."""
    with pytest.raises(OcculusNetworkError):
        raise OcculusNetworkError("connection refused")


def test_error_message_preserved() -> None:
    """Exception message is preserved."""
    msg = "something went wrong"
    exc = OcculusIOError(msg)
    assert str(exc) == msg


def test_chained_exception() -> None:
    """Exceptions can be chained with 'from exc'."""
    original = ValueError("bad value")
    try:
        raise OcculusValidationError("wrapped") from original
    except OcculusValidationError as exc:
        assert exc.__cause__ is original
