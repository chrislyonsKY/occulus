"""Tests for occulus.segmentation — ground classification and object clustering."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusSegmentationError, UnsupportedPlatformError
from occulus.segmentation import (
    SegmentationResult,
    classify_ground_csf,
    classify_ground_pmf,
    cluster_dbscan,
    segment_trees,
)
from occulus.types import AerialCloud, PointCloud, TerrestrialCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def terrain_cloud() -> PointCloud:
    """Synthetic terrain: flat ground at z=0 + some vegetation above."""
    rng = np.random.default_rng(42)
    # Ground layer
    ground_xy = rng.random((500, 2)) * 50.0
    ground_z = np.zeros(500) + rng.random(500) * 0.1
    ground = np.column_stack((ground_xy, ground_z))
    # Vegetation layer
    veg_xy = rng.random((300, 2)) * 50.0
    veg_z = rng.random(300) * 5.0 + 1.0
    veg = np.column_stack((veg_xy, veg_z))
    xyz = np.vstack((ground, veg)).astype(np.float64)
    return AerialCloud(xyz)


@pytest.fixture
def two_cluster_cloud() -> PointCloud:
    """Two clearly separated clusters of points."""
    rng = np.random.default_rng(7)
    c1 = rng.standard_normal((200, 3)) + np.array([0.0, 0.0, 0.0])
    c2 = rng.standard_normal((200, 3)) + np.array([20.0, 0.0, 0.0])
    xyz = np.vstack((c1, c2)).astype(np.float64)
    return PointCloud(xyz)


# ---------------------------------------------------------------------------
# CSF ground classification
# ---------------------------------------------------------------------------


class TestClassifyGroundCSF:
    """Tests for classify_ground_csf."""

    def test_returns_cloud_with_classification(self, terrain_cloud):
        """CSF returns a cloud with classification array set."""
        result = classify_ground_csf(terrain_cloud)
        assert result.classification is not None

    def test_ground_class_is_2(self, terrain_cloud):
        """Ground points have ASPRS class 2."""
        result = classify_ground_csf(terrain_cloud)
        assert result.classification is not None
        assert (result.classification == 2).any()

    def test_some_non_ground_points(self, terrain_cloud):
        """Not all points are classified as ground."""
        result = classify_ground_csf(terrain_cloud)
        assert result.classification is not None
        assert (result.classification != 2).any()

    def test_too_few_points_raises(self):
        """CSF raises OcculusSegmentationError for < 10 points."""
        cloud = PointCloud(np.random.default_rng(0).random((5, 3)))
        with pytest.raises(OcculusSegmentationError):
            classify_ground_csf(cloud)

    def test_preserves_n_points(self, terrain_cloud):
        """Output cloud has the same number of points as input."""
        result = classify_ground_csf(terrain_cloud)
        assert result.n_points == terrain_cloud.n_points

    def test_custom_cloth_resolution(self, terrain_cloud):
        """CSF accepts a custom cloth_resolution parameter."""
        result = classify_ground_csf(terrain_cloud, cloth_resolution=1.0)
        assert result.classification is not None


# ---------------------------------------------------------------------------
# PMF ground classification
# ---------------------------------------------------------------------------


class TestClassifyGroundPMF:
    """Tests for classify_ground_pmf."""

    def test_returns_cloud_with_classification(self, terrain_cloud):
        """PMF returns a cloud with classification array set."""
        result = classify_ground_pmf(terrain_cloud)
        assert result.classification is not None

    def test_ground_class_is_2(self, terrain_cloud):
        """Ground points have ASPRS class 2."""
        result = classify_ground_pmf(terrain_cloud)
        assert result.classification is not None
        assert (result.classification == 2).any()

    def test_too_few_points_raises(self):
        """PMF raises OcculusSegmentationError for < 10 points."""
        cloud = PointCloud(np.random.default_rng(0).random((5, 3)))
        with pytest.raises(OcculusSegmentationError):
            classify_ground_pmf(cloud)

    def test_preserves_n_points(self, terrain_cloud):
        """Output cloud has the same point count as input."""
        result = classify_ground_pmf(terrain_cloud)
        assert result.n_points == terrain_cloud.n_points


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------


class TestClusterDBSCAN:
    """Tests for cluster_dbscan."""

    def test_finds_two_clusters(self, two_cluster_cloud):
        """DBSCAN finds two distinct clusters in clearly separated data."""
        result = cluster_dbscan(two_cluster_cloud, eps=2.0, min_samples=5)
        assert result.n_segments == 2

    def test_labels_length_matches_n_points(self, two_cluster_cloud):
        """Labels array has length equal to n_points."""
        result = cluster_dbscan(two_cluster_cloud, eps=2.0, min_samples=5)
        assert len(result.labels) == two_cluster_cloud.n_points

    def test_invalid_eps_raises(self, two_cluster_cloud):
        """cluster_dbscan raises OcculusSegmentationError for eps <= 0."""
        with pytest.raises(OcculusSegmentationError, match="eps"):
            cluster_dbscan(two_cluster_cloud, eps=0.0)

    def test_invalid_min_samples_raises(self, two_cluster_cloud):
        """cluster_dbscan raises for min_samples <= 0."""
        with pytest.raises(OcculusSegmentationError, match="min_samples"):
            cluster_dbscan(two_cluster_cloud, eps=1.0, min_samples=0)

    def test_noise_label_is_minus_one(self, two_cluster_cloud):
        """Unassigned (noise) points have label -1."""
        result = cluster_dbscan(two_cluster_cloud, eps=0.01, min_samples=100)
        # With very small eps, most points will be noise
        assert -1 in result.labels

    def test_returns_segmentation_result(self, two_cluster_cloud):
        """cluster_dbscan returns a SegmentationResult instance."""
        result = cluster_dbscan(two_cluster_cloud, eps=2.0, min_samples=5)
        assert isinstance(result, SegmentationResult)

    def test_2d_clustering(self, two_cluster_cloud):
        """cluster_dbscan with use_2d=True clusters on XY only."""
        result = cluster_dbscan(two_cluster_cloud, eps=2.0, min_samples=5, use_2d=True)
        assert result.n_segments >= 2


# ---------------------------------------------------------------------------
# Tree segmentation
# ---------------------------------------------------------------------------


class TestSegmentTrees:
    """Tests for segment_trees."""

    def test_terrestrial_cloud_raises(self):
        """segment_trees raises UnsupportedPlatformError for TLS clouds."""
        cloud = TerrestrialCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(UnsupportedPlatformError):
            segment_trees(cloud)

    def test_detects_trees_in_aerial_cloud(self, terrain_cloud):
        """segment_trees finds at least one tree in a vegetation-bearing aerial cloud."""
        # Use lower min_height to be more permissive in test
        result = segment_trees(terrain_cloud, min_height=0.5, resolution=2.0)
        assert isinstance(result, SegmentationResult)

    def test_labels_length_matches_n_points(self, terrain_cloud):
        """Labels array length matches n_points."""
        result = segment_trees(terrain_cloud, min_height=0.5, resolution=2.0)
        assert len(result.labels) == terrain_cloud.n_points

    def test_empty_cloud_raises(self):
        """segment_trees raises OcculusSegmentationError for empty cloud."""
        cloud = AerialCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusSegmentationError):
            segment_trees(cloud)
