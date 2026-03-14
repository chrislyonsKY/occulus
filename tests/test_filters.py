"""Tests for occulus.filters — voxel, SOR, radius, crop, random."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusValidationError
from occulus.filters import (
    crop,
    radius_outlier_removal,
    random_downsample,
    statistical_outlier_removal,
    voxel_downsample,
)
from occulus.types import AerialCloud, PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_cloud() -> PointCloud:
    """10 000-point dense cloud in the unit cube."""
    rng = np.random.default_rng(42)
    xyz = rng.random((10_000, 3)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def cloud_with_outliers() -> PointCloud:
    """Cloud with a cluster of points far from the origin."""
    rng = np.random.default_rng(7)
    normal_pts = rng.standard_normal((500, 3))
    outliers = rng.standard_normal((10, 3)) + 100.0  # far away
    xyz = np.vstack((normal_pts, outliers))
    return PointCloud(xyz)


# ---------------------------------------------------------------------------
# voxel_downsample
# ---------------------------------------------------------------------------


class TestVoxelDownsample:
    """Tests for voxel_downsample."""

    def test_reduces_point_count(self, dense_cloud):
        """Downsampled cloud has fewer points than the input."""
        result = voxel_downsample(dense_cloud, voxel_size=0.1)
        assert result.n_points < dense_cloud.n_points

    def test_larger_voxel_fewer_points(self, dense_cloud):
        """Larger voxel size → fewer output points."""
        coarse = voxel_downsample(dense_cloud, voxel_size=0.2)
        fine = voxel_downsample(dense_cloud, voxel_size=0.05)
        assert coarse.n_points < fine.n_points

    def test_returns_same_type(self, tmp_path):
        """voxel_downsample preserves the concrete subtype."""
        rng = np.random.default_rng(0)
        cloud = AerialCloud(rng.random((1000, 3)))
        result = voxel_downsample(cloud, voxel_size=0.1)
        assert isinstance(result, AerialCloud)

    def test_invalid_voxel_size_raises(self, dense_cloud):
        """voxel_downsample raises OcculusValidationError for non-positive voxel_size."""
        with pytest.raises(OcculusValidationError, match="positive"):
            voxel_downsample(dense_cloud, voxel_size=0.0)

    def test_empty_cloud_raises(self):
        """voxel_downsample raises OcculusValidationError for empty cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusValidationError, match="empty"):
            voxel_downsample(empty, voxel_size=0.1)

    def test_preserves_intensity(self, dense_cloud):
        """voxel_downsample preserves intensity attribute."""
        rng = np.random.default_rng(1)
        cloud = PointCloud(
            dense_cloud.xyz,
            intensity=rng.random(dense_cloud.n_points),
        )
        result = voxel_downsample(cloud, voxel_size=0.1)
        assert result.intensity is not None
        assert result.intensity.shape[0] == result.n_points


# ---------------------------------------------------------------------------
# random_downsample
# ---------------------------------------------------------------------------


class TestRandomDownsample:
    """Tests for random_downsample."""

    def test_reduces_point_count(self, dense_cloud):
        """random_downsample returns fewer points."""
        result = random_downsample(dense_cloud, 0.5)
        assert result.n_points == 5000

    def test_reproducible_with_seed(self, dense_cloud):
        """Same seed → same result."""
        a = random_downsample(dense_cloud, 0.3, seed=42)
        b = random_downsample(dense_cloud, 0.3, seed=42)
        np.testing.assert_array_equal(a.xyz, b.xyz)

    def test_invalid_fraction_raises(self, dense_cloud):
        """random_downsample raises for fraction outside (0, 1]."""
        with pytest.raises(OcculusValidationError):
            random_downsample(dense_cloud, 0.0)
        with pytest.raises(OcculusValidationError):
            random_downsample(dense_cloud, 1.5)


# ---------------------------------------------------------------------------
# statistical_outlier_removal
# ---------------------------------------------------------------------------


class TestStatisticalOutlierRemoval:
    """Tests for statistical_outlier_removal."""

    def test_removes_outliers(self, cloud_with_outliers):
        """SOR removes the cluster of distant outliers."""
        result, _mask = statistical_outlier_removal(cloud_with_outliers, nb_neighbors=10)
        assert result.n_points < cloud_with_outliers.n_points
        assert result.n_points > 400  # normal cluster should survive

    def test_returns_inlier_mask(self, cloud_with_outliers):
        """SOR returns a boolean mask of the correct length."""
        _, mask = statistical_outlier_removal(cloud_with_outliers, nb_neighbors=10)
        assert mask.dtype == bool
        assert len(mask) == cloud_with_outliers.n_points

    def test_invalid_nb_neighbors_raises(self, dense_cloud):
        """SOR raises OcculusValidationError for nb_neighbors <= 0."""
        with pytest.raises(OcculusValidationError):
            statistical_outlier_removal(dense_cloud, nb_neighbors=0)

    def test_nb_neighbors_exceeds_n_points_raises(self):
        """SOR raises when nb_neighbors >= n_points."""
        tiny = PointCloud(np.random.default_rng(0).random((5, 3)))
        with pytest.raises(OcculusValidationError):
            statistical_outlier_removal(tiny, nb_neighbors=10)


# ---------------------------------------------------------------------------
# radius_outlier_removal
# ---------------------------------------------------------------------------


class TestRadiusOutlierRemoval:
    """Tests for radius_outlier_removal."""

    def test_removes_isolated_points(self, cloud_with_outliers):
        """Radius filter removes isolated distant points."""
        result, _mask = radius_outlier_removal(cloud_with_outliers, radius=1.0, min_neighbors=5)
        assert result.n_points < cloud_with_outliers.n_points

    def test_returns_inlier_mask(self, cloud_with_outliers):
        """radius_outlier_removal returns boolean mask of correct length."""
        _, mask = radius_outlier_removal(cloud_with_outliers, radius=1.0)
        assert mask.dtype == bool
        assert len(mask) == cloud_with_outliers.n_points

    def test_invalid_radius_raises(self, dense_cloud):
        """radius_outlier_removal raises for non-positive radius."""
        with pytest.raises(OcculusValidationError, match="positive"):
            radius_outlier_removal(dense_cloud, radius=0.0)

    def test_invalid_min_neighbors_raises(self, dense_cloud):
        """radius_outlier_removal raises for non-positive min_neighbors."""
        with pytest.raises(OcculusValidationError):
            radius_outlier_removal(dense_cloud, radius=0.1, min_neighbors=0)


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------


class TestCrop:
    """Tests for crop."""

    def test_restricts_to_bbox(self, dense_cloud):
        """crop returns only points inside the bbox."""
        result = crop(dense_cloud, bbox=(0.0, 0.0, 0.0, 0.5, 0.5, 0.5))
        assert result.n_points > 0
        assert result.xyz[:, 0].max() <= 0.5

    def test_no_points_outside_bbox(self, dense_cloud):
        """All returned points are within the bounding box."""
        bbox = (0.2, 0.2, 0.2, 0.8, 0.8, 0.8)
        result = crop(dense_cloud, bbox=bbox)
        xmin, _ymin, _zmin, xmax, _ymax, _zmax = bbox
        assert np.all(result.xyz[:, 0] >= xmin)
        assert np.all(result.xyz[:, 0] <= xmax)

    def test_invalid_bbox_length_raises(self, dense_cloud):
        """crop raises OcculusValidationError for wrong-length bbox."""
        with pytest.raises(OcculusValidationError, match="6 elements"):
            crop(dense_cloud, bbox=(0.0, 0.0, 0.0, 1.0))  # type: ignore[arg-type]

    def test_degenerate_bbox_raises(self, dense_cloud):
        """crop raises for bbox where min >= max."""
        with pytest.raises(OcculusValidationError):
            crop(dense_cloud, bbox=(0.5, 0.0, 0.0, 0.5, 1.0, 1.0))

    def test_empty_result_returns_empty_cloud(self, dense_cloud):
        """crop returns a cloud with 0 points if bbox is outside data range."""
        result = crop(dense_cloud, bbox=(10.0, 10.0, 10.0, 11.0, 11.0, 11.0))
        assert result.n_points == 0
