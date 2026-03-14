"""Tests for occulus.metrics — density, CHM, coverage, statistics."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusValidationError, UnsupportedPlatformError
from occulus.metrics import (
    CloudStatistics,
    CoverageStatistics,
    canopy_height_model,
    compute_cloud_statistics,
    coverage_statistics,
    point_density,
)
from occulus.types import AerialCloud, PointCloud, TerrestrialCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_cube_cloud() -> PointCloud:
    """5 000 points uniformly distributed in the unit cube."""
    rng = np.random.default_rng(42)
    return PointCloud(rng.random((5000, 3)).astype(np.float64))


@pytest.fixture
def classified_aerial_cloud() -> AerialCloud:
    """Aerial cloud with ground (class 2) and vegetation points."""
    rng = np.random.default_rng(10)
    n_ground = 500
    n_veg = 300
    # Ground near z=0
    ground = np.column_stack(
        (
            rng.random((n_ground, 2)) * 50.0,
            rng.random(n_ground) * 0.1,
        )
    )
    # Vegetation above z=1
    veg = np.column_stack(
        (
            rng.random((n_veg, 2)) * 50.0,
            rng.random(n_veg) * 5.0 + 1.0,
        )
    )
    xyz = np.vstack((ground, veg)).astype(np.float64)
    classification = np.zeros(n_ground + n_veg, dtype=np.uint8)
    classification[:n_ground] = 2  # ground
    cloud = AerialCloud(xyz, classification=classification)
    return cloud


# ---------------------------------------------------------------------------
# compute_cloud_statistics
# ---------------------------------------------------------------------------


class TestComputeCloudStatistics:
    """Tests for compute_cloud_statistics."""

    def test_returns_cloud_statistics(self, unit_cube_cloud):
        """compute_cloud_statistics returns a CloudStatistics instance."""
        stats = compute_cloud_statistics(unit_cube_cloud)
        assert isinstance(stats, CloudStatistics)

    def test_n_points_correct(self, unit_cube_cloud):
        """n_points matches input cloud."""
        stats = compute_cloud_statistics(unit_cube_cloud)
        assert stats.n_points == unit_cube_cloud.n_points

    def test_z_range_correct(self, unit_cube_cloud):
        """z_min and z_max match the actual range."""
        stats = compute_cloud_statistics(unit_cube_cloud)
        assert stats.z_min == pytest.approx(unit_cube_cloud.xyz[:, 2].min(), abs=1e-6)
        assert stats.z_max == pytest.approx(unit_cube_cloud.xyz[:, 2].max(), abs=1e-6)

    def test_centroid_correct(self, unit_cube_cloud):
        """Centroid matches numpy mean of xyz."""
        stats = compute_cloud_statistics(unit_cube_cloud)
        np.testing.assert_allclose(stats.centroid, unit_cube_cloud.xyz.mean(axis=0), atol=1e-6)

    def test_percentiles_present(self, unit_cube_cloud):
        """z_percentiles contains expected keys."""
        stats = compute_cloud_statistics(unit_cube_cloud)
        for p in [5, 25, 50, 75, 95]:
            assert p in stats.z_percentiles

    def test_intensity_stats_when_present(self):
        """intensity_mean and intensity_std are set when intensity is provided."""
        rng = np.random.default_rng(0)
        xyz = rng.random((100, 3))
        intensity = rng.random(100) * 1000.0
        cloud = PointCloud(xyz, intensity=intensity)
        stats = compute_cloud_statistics(cloud)
        assert stats.intensity_mean is not None
        assert stats.intensity_mean == pytest.approx(float(intensity.mean()), rel=1e-5)

    def test_empty_cloud_raises(self):
        """compute_cloud_statistics raises for empty cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusValidationError):
            compute_cloud_statistics(empty)


# ---------------------------------------------------------------------------
# point_density
# ---------------------------------------------------------------------------


class TestPointDensity:
    """Tests for point_density."""

    def test_returns_tuple_of_arrays(self, unit_cube_cloud):
        """point_density returns (density, x_edges, y_edges)."""
        density, x_edges, y_edges = point_density(unit_cube_cloud, resolution=0.1)
        assert density.ndim == 2
        assert x_edges.ndim == 1
        assert y_edges.ndim == 1

    def test_density_non_negative(self, unit_cube_cloud):
        """All density values are >= 0."""
        density, _, _ = point_density(unit_cube_cloud, resolution=0.1)
        assert np.all(density >= 0)

    def test_total_points_conserved(self, unit_cube_cloud):
        """Sum of density array equals total point count."""
        density, _, _ = point_density(unit_cube_cloud, resolution=0.05)
        assert int(density.sum()) == unit_cube_cloud.n_points

    def test_invalid_resolution_raises(self, unit_cube_cloud):
        """point_density raises for non-positive resolution."""
        with pytest.raises(OcculusValidationError, match="positive"):
            point_density(unit_cube_cloud, resolution=0.0)

    def test_empty_cloud_raises(self):
        """point_density raises for empty cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusValidationError):
            point_density(empty)


# ---------------------------------------------------------------------------
# canopy_height_model
# ---------------------------------------------------------------------------


class TestCanopyHeightModel:
    """Tests for canopy_height_model."""

    def test_returns_tuple(self, classified_aerial_cloud):
        """canopy_height_model returns (chm, x_edges, y_edges)."""
        chm, x_edges, y_edges = canopy_height_model(classified_aerial_cloud, resolution=2.0)
        assert chm.ndim == 2
        assert x_edges.ndim == 1
        assert y_edges.ndim == 1

    def test_chm_non_negative(self, classified_aerial_cloud):
        """CHM values are >= 0."""
        chm, _, _ = canopy_height_model(classified_aerial_cloud, resolution=2.0)
        assert np.all(chm >= 0)

    def test_terrestrial_cloud_raises(self):
        """canopy_height_model raises UnsupportedPlatformError for TLS."""
        cloud = TerrestrialCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(UnsupportedPlatformError):
            canopy_height_model(cloud)

    def test_no_classification_raises(self):
        """canopy_height_model raises if classification array is missing."""
        cloud = AerialCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(OcculusValidationError, match="classification"):
            canopy_height_model(cloud)

    def test_no_ground_class_raises(self):
        """canopy_height_model raises if no ground points are present."""
        rng = np.random.default_rng(0)
        cloud = AerialCloud(
            rng.random((100, 3)),
            classification=np.ones(100, dtype=np.uint8),  # all class 1 (unassigned)
        )
        with pytest.raises(OcculusValidationError, match="ground"):
            canopy_height_model(cloud)

    def test_invalid_resolution_raises(self, classified_aerial_cloud):
        """canopy_height_model raises for non-positive resolution."""
        with pytest.raises(OcculusValidationError):
            canopy_height_model(classified_aerial_cloud, resolution=-1.0)


# ---------------------------------------------------------------------------
# coverage_statistics
# ---------------------------------------------------------------------------


class TestCoverageStatistics:
    """Tests for coverage_statistics."""

    def test_returns_coverage_statistics(self, unit_cube_cloud):
        """coverage_statistics returns a CoverageStatistics instance."""
        stats = coverage_statistics(unit_cube_cloud, resolution=0.1)
        assert isinstance(stats, CoverageStatistics)

    def test_gap_fraction_in_range(self, unit_cube_cloud):
        """Gap fraction is in [0, 1]."""
        stats = coverage_statistics(unit_cube_cloud, resolution=0.1)
        assert 0.0 <= stats.gap_fraction <= 1.0

    def test_mean_density_positive(self, unit_cube_cloud):
        """Mean density is positive for a non-empty cloud."""
        stats = coverage_statistics(unit_cube_cloud, resolution=0.1)
        assert stats.mean_density > 0

    def test_covered_area_le_total_area(self, unit_cube_cloud):
        """Covered area cannot exceed total area."""
        stats = coverage_statistics(unit_cube_cloud, resolution=0.1)
        assert stats.covered_area <= stats.total_area
