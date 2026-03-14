"""Tests for occulus.types — core point cloud types."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.types import (
    AcquisitionMetadata,
    AerialCloud,
    Platform,
    PointCloud,
    ScanPosition,
    TerrestrialCloud,
    UAVCloud,
)
from occulus.exceptions import OcculusValidationError


@pytest.fixture
def sample_xyz() -> np.ndarray:
    """Generate a small random point cloud."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1000, 3))


@pytest.fixture
def sample_cloud(sample_xyz: np.ndarray) -> PointCloud:
    """Create a basic PointCloud."""
    return PointCloud(sample_xyz)


class TestPointCloud:
    """Tests for the base PointCloud class."""

    def test_creation(self, sample_xyz: np.ndarray) -> None:
        """PointCloud accepts valid (N, 3) array."""
        cloud = PointCloud(sample_xyz)
        assert cloud.n_points == 1000
        assert cloud.platform == Platform.UNKNOWN

    def test_invalid_shape_raises(self) -> None:
        """PointCloud rejects non-(N, 3) arrays."""
        with pytest.raises(OcculusValidationError, match="must be"):
            PointCloud(np.zeros((10, 4)))

    def test_bounds(self, sample_cloud: PointCloud) -> None:
        """Bounds returns (2, 3) array."""
        bounds = sample_cloud.bounds
        assert bounds.shape == (2, 3)
        assert np.all(bounds[0] <= bounds[1])

    def test_centroid(self, sample_cloud: PointCloud) -> None:
        """Centroid returns (3,) array."""
        centroid = sample_cloud.centroid
        assert centroid.shape == (3,)

    def test_len(self, sample_cloud: PointCloud) -> None:
        """len() returns point count."""
        assert len(sample_cloud) == 1000

    def test_has_normals_false(self, sample_cloud: PointCloud) -> None:
        """has_normals is False when no normals set."""
        assert not sample_cloud.has_normals

    def test_has_normals_true(self, sample_xyz: np.ndarray) -> None:
        """has_normals is True when normals provided."""
        normals = np.random.default_rng(0).standard_normal((1000, 3))
        cloud = PointCloud(sample_xyz, normals=normals)
        assert cloud.has_normals

    def test_repr(self, sample_cloud: PointCloud) -> None:
        """repr contains point count and platform."""
        r = repr(sample_cloud)
        assert "1,000" in r
        assert "unknown" in r


class TestAerialCloud:
    """Tests for AerialCloud subtype."""

    def test_platform_is_aerial(self, sample_xyz: np.ndarray) -> None:
        """AerialCloud sets platform to AERIAL."""
        cloud = AerialCloud(sample_xyz)
        assert cloud.platform == Platform.AERIAL

    def test_ground_points_raises_without_classification(self, sample_xyz: np.ndarray) -> None:
        """ground_points raises OcculusValidationError when no classification array present."""
        from occulus.exceptions import OcculusValidationError
        cloud = AerialCloud(sample_xyz)
        with pytest.raises(OcculusValidationError, match="classification"):
            cloud.ground_points()

    def test_ground_points_returns_class_2_mask(self, sample_xyz: np.ndarray) -> None:
        """ground_points returns True for ASPRS class 2 points."""
        classification = np.zeros(len(sample_xyz), dtype=np.uint8)
        classification[:100] = 2
        cloud = AerialCloud(sample_xyz, classification=classification)
        mask = cloud.ground_points()
        assert mask.sum() == 100
        assert mask.dtype == bool


class TestTerrestrialCloud:
    """Tests for TerrestrialCloud subtype."""

    def test_platform_is_terrestrial(self, sample_xyz: np.ndarray) -> None:
        """TerrestrialCloud sets platform to TERRESTRIAL."""
        cloud = TerrestrialCloud(sample_xyz)
        assert cloud.platform == Platform.TERRESTRIAL

    def test_scan_positions(self, sample_xyz: np.ndarray) -> None:
        """TerrestrialCloud stores scan positions."""
        pos = ScanPosition(x=0, y=0, z=1.5, scan_id="setup1")
        cloud = TerrestrialCloud(sample_xyz, scan_positions=[pos])
        assert len(cloud.scan_positions) == 1
        assert cloud.scan_positions[0].scan_id == "setup1"


class TestUAVCloud:
    """Tests for UAVCloud subtype."""

    def test_platform_is_uav(self, sample_xyz: np.ndarray) -> None:
        """UAVCloud sets platform to UAV."""
        cloud = UAVCloud(sample_xyz)
        assert cloud.platform == Platform.UAV

    def test_photogrammetric_flag(self, sample_xyz: np.ndarray) -> None:
        """UAVCloud tracks photogrammetric origin."""
        cloud = UAVCloud(sample_xyz, is_photogrammetric=True)
        assert cloud.is_photogrammetric


class TestScanPosition:
    """Tests for ScanPosition dataclass."""

    def test_as_array(self) -> None:
        """as_array returns (3,) numpy array."""
        pos = ScanPosition(x=1.0, y=2.0, z=3.0)
        arr = pos.as_array()
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])
