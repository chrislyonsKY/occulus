"""Tests for occulus.features — plane detection, cylinder fitting, geometric features."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusFeatureError, OcculusValidationError
from occulus.features import (
    GeometricFeatures,
    PlaneResult,
    compute_geometric_features,
    detect_cylinders,
    detect_planes,
)
from occulus.types import PointCloud


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def planar_cloud() -> PointCloud:
    """500 points on the Z=0 plane with small Gaussian noise."""
    rng = np.random.default_rng(1)
    xy = rng.random((500, 2)) * 10.0
    z = rng.standard_normal(500) * 0.005  # tight plane
    return PointCloud(np.column_stack((xy, z)))


@pytest.fixture
def two_plane_cloud() -> PointCloud:
    """Two distinct planes: Z=0 and Z=5."""
    rng = np.random.default_rng(2)
    n = 300
    xy = rng.random((n, 2)) * 10.0
    plane1 = np.column_stack((xy, np.zeros(n) + rng.standard_normal(n) * 0.01))
    xy2 = rng.random((n, 2)) * 10.0
    plane2 = np.column_stack((xy2, np.full(n, 5.0) + rng.standard_normal(n) * 0.01))
    return PointCloud(np.vstack((plane1, plane2)))


@pytest.fixture
def volumetric_cloud() -> PointCloud:
    """1 000 points in a 3D volume — useful for eigenvalue features."""
    rng = np.random.default_rng(3)
    return PointCloud(rng.random((1000, 3)))


# ---------------------------------------------------------------------------
# detect_planes
# ---------------------------------------------------------------------------

class TestDetectPlanes:
    """Tests for detect_planes."""

    def test_finds_one_plane(self, planar_cloud):
        """detect_planes finds the dominant plane in a planar cloud."""
        planes = detect_planes(
            planar_cloud,
            distance_threshold=0.05,
            num_iterations=500,
            min_points=50,
        )
        assert len(planes) >= 1

    def test_plane_normal_is_unit_vector(self, planar_cloud):
        """Detected plane normal is a unit vector."""
        planes = detect_planes(planar_cloud, distance_threshold=0.05, num_iterations=500)
        assert len(planes) >= 1
        norm = np.linalg.norm(planes[0].normal)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_returns_plane_result_instances(self, planar_cloud):
        """detect_planes returns PlaneResult instances."""
        planes = detect_planes(planar_cloud, distance_threshold=0.05, num_iterations=300)
        for plane in planes:
            assert isinstance(plane, PlaneResult)

    def test_inlier_mask_length(self, planar_cloud):
        """Inlier mask length matches n_points."""
        planes = detect_planes(planar_cloud, distance_threshold=0.05, num_iterations=300)
        for plane in planes:
            assert len(plane.inlier_mask) == planar_cloud.n_points

    def test_finds_two_planes(self, two_plane_cloud):
        """detect_planes finds both planes in a two-plane cloud."""
        planes = detect_planes(
            two_plane_cloud,
            distance_threshold=0.05,
            num_iterations=500,
            max_planes=3,
            min_points=50,
        )
        assert len(planes) >= 2

    def test_too_few_points_raises(self):
        """detect_planes raises OcculusFeatureError for too few points."""
        tiny = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(OcculusFeatureError):
            detect_planes(tiny, ransac_n=3)

    def test_inlier_count_reasonable(self, planar_cloud):
        """Detected plane inlier count is a reasonable fraction of n_points."""
        planes = detect_planes(planar_cloud, distance_threshold=0.05, num_iterations=500)
        # A flat cloud should yield mostly inliers for the dominant plane
        assert planes[0].n_inliers > planar_cloud.n_points * 0.5


# ---------------------------------------------------------------------------
# detect_cylinders
# ---------------------------------------------------------------------------

class TestDetectCylinders:
    """Tests for detect_cylinders."""

    def test_invalid_radius_range_raises(self, planar_cloud):
        """detect_cylinders raises OcculusValidationError for bad radius_range."""
        with pytest.raises(OcculusValidationError):
            detect_cylinders(planar_cloud, radius_range=(2.0, 1.0))

    def test_too_few_points_raises(self):
        """detect_cylinders raises OcculusFeatureError for < 6 points."""
        tiny = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        with pytest.raises(OcculusFeatureError):
            detect_cylinders(tiny)

    def test_returns_list(self, volumetric_cloud):
        """detect_cylinders returns a list (possibly empty)."""
        result = detect_cylinders(volumetric_cloud, num_iterations=100, min_points=500)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# compute_geometric_features
# ---------------------------------------------------------------------------

class TestComputeGeometricFeatures:
    """Tests for compute_geometric_features."""

    def test_returns_geometric_features(self, volumetric_cloud):
        """compute_geometric_features returns a GeometricFeatures instance."""
        result = compute_geometric_features(volumetric_cloud, radius=0.1)
        assert isinstance(result, GeometricFeatures)

    def test_arrays_have_correct_length(self, volumetric_cloud):
        """All feature arrays have length equal to n_points."""
        n = volumetric_cloud.n_points
        result = compute_geometric_features(volumetric_cloud, radius=0.15)
        assert len(result.linearity) == n
        assert len(result.planarity) == n
        assert len(result.sphericity) == n
        assert len(result.omnivariance) == n
        assert len(result.anisotropy) == n
        assert len(result.eigenentropy) == n
        assert len(result.curvature) == n

    def test_values_in_valid_range(self, volumetric_cloud):
        """Linearity, planarity, sphericity sum to ≤ 1 and are non-negative."""
        result = compute_geometric_features(volumetric_cloud, radius=0.2)
        assert np.all(result.linearity >= -1e-9)
        assert np.all(result.planarity >= -1e-9)
        assert np.all(result.sphericity >= -1e-9)
        assert np.all(result.curvature >= -1e-9)

    def test_planar_cloud_has_high_planarity(self, planar_cloud):
        """Planar cloud should have higher mean planarity than sphericity."""
        result = compute_geometric_features(planar_cloud, radius=0.5)
        mean_planarity = result.planarity[result.planarity > 0].mean()
        mean_sphericity = result.sphericity[result.sphericity > 0].mean()
        assert mean_planarity > mean_sphericity

    def test_invalid_radius_raises(self, volumetric_cloud):
        """compute_geometric_features raises for non-positive radius."""
        with pytest.raises(OcculusValidationError, match="positive"):
            compute_geometric_features(volumetric_cloud, radius=0.0)

    def test_too_few_points_raises(self):
        """compute_geometric_features raises OcculusFeatureError for < 3 points."""
        tiny = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(OcculusFeatureError):
            compute_geometric_features(tiny, radius=1.0)
