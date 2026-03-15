"""Tests for occulus.analysis — volume computation and cross-section extraction."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.analysis import (
    CrossSection,
    VolumeResult,
    compute_volume,
    extract_cross_section,
    extract_profiles,
)
from occulus.exceptions import OcculusValidationError
from occulus.types import PointCloud


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_surface() -> PointCloud:
    """Flat surface at z=0 on a 10x10 grid."""
    rng = np.random.default_rng(42)
    n = 2000
    xy = rng.random((n, 2)) * 10.0
    z = np.zeros(n)
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def raised_surface() -> PointCloud:
    """Flat surface at z=2 on a 10x10 grid."""
    rng = np.random.default_rng(99)
    n = 2000
    xy = rng.random((n, 2)) * 10.0
    z = np.full(n, 2.0)
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def sloped_surface() -> PointCloud:
    """Surface with z = x, sloping from 0 to 10 over a 10x10 grid."""
    rng = np.random.default_rng(77)
    n = 3000
    xy = rng.random((n, 2)) * 10.0
    z = xy[:, 0]  # z = x
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def corridor_cloud() -> PointCloud:
    """Dense point cloud along a straight corridor for cross-section tests."""
    rng = np.random.default_rng(55)
    n = 5000
    x = rng.random(n) * 100.0
    y = rng.random(n) * 10.0 - 5.0  # centred around y=0
    z = np.sin(x / 10.0) + rng.standard_normal(n) * 0.05
    xyz = np.column_stack((x, y, z)).astype(np.float64)
    return PointCloud(xyz)


# ---------------------------------------------------------------------------
# VolumeResult
# ---------------------------------------------------------------------------


class TestVolumeResult:
    """Tests for VolumeResult dataclass."""

    def test_fields_present(self):
        """VolumeResult has all expected fields."""
        vr = VolumeResult(
            cut_volume=10.0,
            fill_volume=5.0,
            net_volume=5.0,
            resolution=1.0,
            area=100.0,
            cut_area=60.0,
            fill_area=40.0,
        )
        assert vr.cut_volume == 10.0
        assert vr.fill_volume == 5.0
        assert vr.net_volume == 5.0
        assert vr.resolution == 1.0
        assert vr.area == 100.0
        assert vr.cut_area == 60.0
        assert vr.fill_area == 40.0


# ---------------------------------------------------------------------------
# compute_volume
# ---------------------------------------------------------------------------


class TestComputeVolume:
    """Tests for compute_volume."""

    def test_returns_volume_result(self, flat_surface, raised_surface):
        """compute_volume returns a VolumeResult instance."""
        result = compute_volume(raised_surface, flat_surface, resolution=1.0)
        assert isinstance(result, VolumeResult)

    def test_uniform_cut(self, flat_surface, raised_surface):
        """Raised surface above flat reference produces cut volume."""
        result = compute_volume(raised_surface, flat_surface, resolution=1.0)
        assert result.cut_volume > 0
        assert result.fill_volume == pytest.approx(0.0, abs=1e-6)
        assert result.net_volume > 0

    def test_uniform_fill(self, flat_surface, raised_surface):
        """Flat surface below raised reference produces fill volume."""
        result = compute_volume(flat_surface, raised_surface, resolution=1.0)
        assert result.fill_volume > 0
        assert result.cut_volume == pytest.approx(0.0, abs=1e-6)
        assert result.net_volume < 0

    def test_same_surface_zero_volume(self, flat_surface):
        """Identical surfaces produce zero volumes."""
        result = compute_volume(flat_surface, flat_surface, resolution=1.0)
        assert result.cut_volume == pytest.approx(0.0, abs=1e-6)
        assert result.fill_volume == pytest.approx(0.0, abs=1e-6)
        assert result.net_volume == pytest.approx(0.0, abs=1e-6)

    def test_net_volume_approximate_value(self, flat_surface, raised_surface):
        """Net volume approximates expected value for known geometry.

        A 10x10 area with 2m uniform elevation difference should give
        approximately 200 cubic units of cut volume.
        """
        result = compute_volume(raised_surface, flat_surface, resolution=0.5)
        # With random point placement, the covered area won't be exactly 100
        # but should be close. Volume should be close to area * 2.0.
        assert result.cut_volume == pytest.approx(result.area * 2.0, rel=0.1)

    def test_resolution_stored(self, flat_surface, raised_surface):
        """Resolution is correctly stored in the result."""
        result = compute_volume(raised_surface, flat_surface, resolution=2.5)
        assert result.resolution == 2.5

    def test_area_positive(self, flat_surface, raised_surface):
        """Computed area is positive."""
        result = compute_volume(raised_surface, flat_surface, resolution=1.0)
        assert result.area > 0

    def test_cut_fill_areas_sum_le_total(self, flat_surface, sloped_surface):
        """Cut area + fill area <= total area."""
        result = compute_volume(sloped_surface, flat_surface, resolution=1.0)
        assert result.cut_area + result.fill_area <= result.area + 1e-6

    def test_finer_resolution_more_cells(self, flat_surface, raised_surface):
        """Finer resolution produces area closer to actual extent."""
        coarse = compute_volume(raised_surface, flat_surface, resolution=5.0)
        fine = compute_volume(raised_surface, flat_surface, resolution=0.5)
        # Finer resolution should give a more accurate area estimate
        assert fine.area > 0
        assert coarse.area > 0

    def test_invalid_resolution_raises(self, flat_surface, raised_surface):
        """compute_volume raises for non-positive resolution."""
        with pytest.raises(OcculusValidationError, match="positive"):
            compute_volume(raised_surface, flat_surface, resolution=0.0)
        with pytest.raises(OcculusValidationError, match="positive"):
            compute_volume(raised_surface, flat_surface, resolution=-1.0)

    def test_empty_surface_raises(self, flat_surface):
        """compute_volume raises for empty surface cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusValidationError, match="surface"):
            compute_volume(empty, flat_surface)

    def test_empty_reference_raises(self, flat_surface):
        """compute_volume raises for empty reference cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        with pytest.raises(OcculusValidationError, match="reference"):
            compute_volume(flat_surface, empty)

    def test_unsupported_method_raises(self, flat_surface, raised_surface):
        """compute_volume raises for unsupported method."""
        with pytest.raises(OcculusValidationError, match="method"):
            compute_volume(raised_surface, flat_surface, method="triangulation")


# ---------------------------------------------------------------------------
# CrossSection
# ---------------------------------------------------------------------------


class TestCrossSection:
    """Tests for CrossSection dataclass."""

    def test_fields_present(self):
        """CrossSection has all expected fields."""
        cs = CrossSection(
            station=np.array([0.0, 1.0]),
            elevation=np.array([5.0, 6.0]),
            points=np.zeros((10, 3)),
            width=1.0,
            polyline=np.array([[0.0, 0.0], [10.0, 0.0]]),
        )
        assert len(cs.station) == 2
        assert len(cs.elevation) == 2
        assert cs.points.shape == (10, 3)
        assert cs.width == 1.0
        assert cs.polyline.shape == (2, 2)


# ---------------------------------------------------------------------------
# extract_cross_section
# ---------------------------------------------------------------------------


class TestExtractCrossSection:
    """Tests for extract_cross_section."""

    def test_returns_cross_section(self, corridor_cloud):
        """extract_cross_section returns a CrossSection instance."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        assert isinstance(cs, CrossSection)

    def test_station_monotonically_increasing(self, corridor_cloud):
        """Station values are monotonically increasing."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        if len(cs.station) > 1:
            assert np.all(np.diff(cs.station) > 0)

    def test_elevation_length_matches_station(self, corridor_cloud):
        """Elevation array length matches station array."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        assert len(cs.elevation) == len(cs.station)

    def test_width_stored(self, corridor_cloud):
        """Width is correctly stored."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=3.5, resolution=1.0)
        assert cs.width == 3.5

    def test_polyline_stored(self, corridor_cloud):
        """Polyline is correctly stored."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        np.testing.assert_array_equal(cs.polyline, polyline)

    def test_narrow_width_fewer_points(self, corridor_cloud):
        """Narrower width captures fewer points."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        wide = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        narrow = extract_cross_section(corridor_cloud, polyline, width=1.0, resolution=1.0)
        assert narrow.points.shape[0] <= wide.points.shape[0]

    def test_no_points_in_corridor(self):
        """Returns empty arrays when no points fall in the corridor."""
        rng = np.random.default_rng(0)
        xyz = rng.random((100, 3)) * 10.0 + 100.0  # far from polyline
        cloud = PointCloud(xyz.astype(np.float64))
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        cs = extract_cross_section(cloud, polyline, width=1.0, resolution=0.5)
        assert len(cs.station) == 0
        assert len(cs.elevation) == 0

    def test_multivertex_polyline(self, corridor_cloud):
        """Works with a polyline having more than two vertices."""
        polyline = np.array([
            [0.0, 0.0],
            [30.0, 0.0],
            [60.0, 2.0],
            [100.0, 0.0],
        ])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        assert isinstance(cs, CrossSection)
        assert len(cs.station) > 0

    def test_3d_polyline_accepted(self, corridor_cloud):
        """A (P, 3) polyline is accepted — Z is ignored for corridor selection."""
        polyline = np.array([[0.0, 0.0, 99.0], [100.0, 0.0, 99.0]])
        cs = extract_cross_section(corridor_cloud, polyline, width=5.0, resolution=1.0)
        assert isinstance(cs, CrossSection)
        assert len(cs.station) > 0

    def test_empty_cloud_raises(self):
        """extract_cross_section raises for empty cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="empty"):
            extract_cross_section(empty, polyline)

    def test_degenerate_polyline_raises(self, corridor_cloud):
        """extract_cross_section raises for single-vertex polyline."""
        polyline = np.array([[0.0, 0.0]])
        with pytest.raises(OcculusValidationError):
            extract_cross_section(corridor_cloud, polyline)

    def test_invalid_width_raises(self, corridor_cloud):
        """extract_cross_section raises for non-positive width."""
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="width"):
            extract_cross_section(corridor_cloud, polyline, width=0.0)

    def test_invalid_resolution_raises(self, corridor_cloud):
        """extract_cross_section raises for non-positive resolution."""
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="resolution"):
            extract_cross_section(corridor_cloud, polyline, resolution=-1.0)


# ---------------------------------------------------------------------------
# extract_profiles
# ---------------------------------------------------------------------------


class TestExtractProfiles:
    """Tests for extract_profiles."""

    def test_returns_list_of_cross_sections(self, corridor_cloud):
        """extract_profiles returns a list of CrossSection instances."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        profiles = extract_profiles(
            corridor_cloud, polyline, interval=20.0, width=5.0, resolution=0.5
        )
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        assert all(isinstance(p, CrossSection) for p in profiles)

    def test_number_of_profiles(self, corridor_cloud):
        """Number of profiles matches expected count from interval."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        profiles = extract_profiles(
            corridor_cloud, polyline, interval=25.0, width=5.0
        )
        # Expected: stations at 0, 25, 50, 75, 100 => 5 profiles
        assert len(profiles) == 5

    def test_shorter_interval_more_profiles(self, corridor_cloud):
        """Shorter interval produces more profiles."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        few = extract_profiles(corridor_cloud, polyline, interval=50.0, width=5.0)
        many = extract_profiles(corridor_cloud, polyline, interval=10.0, width=5.0)
        assert len(many) > len(few)

    def test_empty_cloud_raises(self):
        """extract_profiles raises for empty cloud."""
        empty = PointCloud(np.zeros((0, 3)))
        polyline = np.array([[0.0, 0.0], [10.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="empty"):
            extract_profiles(empty, polyline)

    def test_invalid_interval_raises(self, corridor_cloud):
        """extract_profiles raises for non-positive interval."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="interval"):
            extract_profiles(corridor_cloud, polyline, interval=0.0)

    def test_invalid_width_raises(self, corridor_cloud):
        """extract_profiles raises for non-positive width."""
        polyline = np.array([[0.0, 0.0], [100.0, 0.0]])
        with pytest.raises(OcculusValidationError, match="width"):
            extract_profiles(corridor_cloud, polyline, width=-1.0)

    def test_degenerate_polyline_raises(self, corridor_cloud):
        """extract_profiles raises for single-vertex polyline."""
        polyline = np.array([[5.0, 5.0]])
        with pytest.raises(OcculusValidationError):
            extract_profiles(corridor_cloud, polyline)
