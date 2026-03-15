"""Tests for occulus.change — M3C2 change detection."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.change import M3C2Result, m3c2
from occulus.exceptions import OcculusChangeDetectionError
from occulus.types import PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_epoch1() -> PointCloud:
    """1 000 points on the XY plane at z=0."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(0, 10, size=(1000, 2))
    z = np.zeros(1000)
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def flat_epoch2_shifted() -> PointCloud:
    """1 000 points on the XY plane at z=0.5 (uniform upward shift)."""
    rng = np.random.default_rng(43)
    xy = rng.uniform(0, 10, size=(1000, 2))
    z = np.full(1000, 0.5)
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def flat_epoch2_same() -> PointCloud:
    """1 000 points on the XY plane at z=0 (no change from epoch1)."""
    rng = np.random.default_rng(44)
    xy = rng.uniform(0, 10, size=(1000, 2))
    z = np.zeros(1000)
    xyz = np.column_stack((xy, z)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def sparse_core_points() -> np.ndarray:
    """A small set of core points on a 3x3 grid at z=0."""
    xs, ys = np.meshgrid(np.linspace(2, 8, 3), np.linspace(2, 8, 3))
    return np.column_stack((xs.ravel(), ys.ravel(), np.zeros(9))).astype(np.float64)


# ---------------------------------------------------------------------------
# Return type and shape tests
# ---------------------------------------------------------------------------


class TestM3C2ReturnType:
    """Verify that m3c2 returns a well-formed M3C2Result."""

    def test_returns_m3c2_result(self, flat_epoch1, flat_epoch2_shifted):
        """m3c2 returns an M3C2Result dataclass."""
        result = m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=2.0, projection_scale=2.0)
        assert isinstance(result, M3C2Result)

    def test_output_shapes_default_core(self, flat_epoch1, flat_epoch2_shifted):
        """When core_points=None, output arrays match epoch1.n_points."""
        result = m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=2.0, projection_scale=2.0)
        n = flat_epoch1.n_points
        assert result.distances.shape == (n,)
        assert result.uncertainties.shape == (n,)
        assert result.normals.shape == (n, 3)
        assert result.core_points.shape == (n, 3)
        assert result.significant_change.shape == (n,)

    def test_output_shapes_custom_core(self, flat_epoch1, flat_epoch2_shifted, sparse_core_points):
        """When core_points are provided, output arrays match their count."""
        result = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            core_points=sparse_core_points,
            normal_scale=3.0,
            projection_scale=3.0,
        )
        n = sparse_core_points.shape[0]
        assert result.distances.shape == (n,)
        assert result.uncertainties.shape == (n,)
        assert result.normals.shape == (n, 3)
        assert result.core_points.shape == (n, 3)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestM3C2Distances:
    """Verify distance and significance detection accuracy."""

    def test_detects_upward_shift(self, flat_epoch1, flat_epoch2_shifted):
        """A uniform 0.5 m upward shift should produce positive distances ~0.5."""
        result = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            normal_scale=2.0,
            projection_scale=2.0,
        )
        valid = np.isfinite(result.distances)
        assert valid.sum() > 0, "Expected at least some valid distances"
        median_dist = np.median(result.distances[valid])
        # The absolute distance should be close to 0.5
        assert abs(abs(median_dist) - 0.5) < 0.15, (
            f"Expected median |distance| near 0.5, got {median_dist}"
        )

    def test_no_change_distances_near_zero(self, flat_epoch1, flat_epoch2_same):
        """When both epochs are at z=0, distances should be near zero."""
        result = m3c2(
            flat_epoch1,
            flat_epoch2_same,
            normal_scale=2.0,
            projection_scale=2.0,
        )
        valid = np.isfinite(result.distances)
        assert valid.sum() > 0
        assert np.abs(result.distances[valid]).mean() < 0.1

    def test_significant_change_detected(self, flat_epoch1, flat_epoch2_shifted):
        """A 0.5 m shift should be detected as significant for many core points."""
        result = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            normal_scale=2.0,
            projection_scale=2.0,
            registration_error=0.0,
        )
        valid = np.isfinite(result.distances)
        if valid.sum() > 0:
            sig_ratio = result.significant_change[valid].mean()
            assert sig_ratio > 0.5, f"Expected most points significant, got ratio {sig_ratio}"

    def test_no_significant_change_when_identical(self, flat_epoch1, flat_epoch2_same):
        """When epochs are identical, few or no points should be significant."""
        result = m3c2(
            flat_epoch1,
            flat_epoch2_same,
            normal_scale=2.0,
            projection_scale=2.0,
        )
        valid = np.isfinite(result.distances)
        if valid.sum() > 0:
            sig_ratio = result.significant_change[valid].mean()
            assert sig_ratio < 0.5


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


class TestM3C2Normals:
    """Verify normal estimation within M3C2."""

    def test_normals_are_unit_vectors(self, flat_epoch1, flat_epoch2_shifted):
        """Valid normals should be unit length."""
        result = m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=2.0, projection_scale=2.0)
        valid = np.all(np.isfinite(result.normals), axis=1)
        assert valid.sum() > 0
        norms = np.linalg.norm(result.normals[valid], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_flat_surface_normals_align_z(self, flat_epoch1, flat_epoch2_shifted):
        """For flat XY surfaces the normals should align with the Z axis."""
        result = m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=2.0, projection_scale=2.0)
        valid = np.all(np.isfinite(result.normals), axis=1)
        z_component = np.abs(result.normals[valid, 2])
        assert z_component.mean() > 0.8


# ---------------------------------------------------------------------------
# Uncertainty / LoD
# ---------------------------------------------------------------------------


class TestM3C2Uncertainty:
    """Verify Level of Detection behaviour."""

    def test_uncertainties_positive(self, flat_epoch1, flat_epoch2_shifted):
        """Valid LoD values should be positive."""
        result = m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=2.0, projection_scale=2.0)
        valid = np.isfinite(result.uncertainties)
        assert valid.sum() > 0
        assert np.all(result.uncertainties[valid] >= 0)

    def test_registration_error_increases_lod(self, flat_epoch1, flat_epoch2_same):
        """Adding registration error should increase LoD values."""
        result_no_err = m3c2(
            flat_epoch1,
            flat_epoch2_same,
            normal_scale=2.0,
            projection_scale=2.0,
            registration_error=0.0,
        )
        result_with_err = m3c2(
            flat_epoch1,
            flat_epoch2_same,
            normal_scale=2.0,
            projection_scale=2.0,
            registration_error=0.1,
        )
        valid = np.isfinite(result_no_err.uncertainties) & np.isfinite(
            result_with_err.uncertainties
        )
        if valid.sum() > 0:
            assert np.all(
                result_with_err.uncertainties[valid] >= result_no_err.uncertainties[valid]
            )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestM3C2Validation:
    """Verify that invalid inputs raise OcculusChangeDetectionError."""

    def test_epoch1_too_few_points(self, flat_epoch2_shifted):
        """epoch1 with < 3 points raises."""
        tiny = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(OcculusChangeDetectionError, match="epoch1"):
            m3c2(tiny, flat_epoch2_shifted)

    def test_epoch2_too_few_points(self, flat_epoch1):
        """epoch2 with < 3 points raises."""
        tiny = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(OcculusChangeDetectionError, match="epoch2"):
            m3c2(flat_epoch1, tiny)

    def test_invalid_confidence_low(self, flat_epoch1, flat_epoch2_shifted):
        """confidence <= 0 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="confidence"):
            m3c2(flat_epoch1, flat_epoch2_shifted, confidence=0.0)

    def test_invalid_confidence_high(self, flat_epoch1, flat_epoch2_shifted):
        """confidence >= 1 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="confidence"):
            m3c2(flat_epoch1, flat_epoch2_shifted, confidence=1.0)

    def test_invalid_normal_scale(self, flat_epoch1, flat_epoch2_shifted):
        """normal_scale <= 0 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="normal_scale"):
            m3c2(flat_epoch1, flat_epoch2_shifted, normal_scale=-1.0)

    def test_invalid_projection_scale(self, flat_epoch1, flat_epoch2_shifted):
        """projection_scale <= 0 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="projection_scale"):
            m3c2(flat_epoch1, flat_epoch2_shifted, projection_scale=0.0)

    def test_invalid_max_cylinder_depth(self, flat_epoch1, flat_epoch2_shifted):
        """max_cylinder_depth <= 0 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="max_cylinder_depth"):
            m3c2(flat_epoch1, flat_epoch2_shifted, max_cylinder_depth=0.0)

    def test_negative_registration_error(self, flat_epoch1, flat_epoch2_shifted):
        """registration_error < 0 raises."""
        with pytest.raises(OcculusChangeDetectionError, match="registration_error"):
            m3c2(flat_epoch1, flat_epoch2_shifted, registration_error=-0.01)

    def test_invalid_core_points_shape(self, flat_epoch1, flat_epoch2_shifted):
        """core_points with wrong shape raises."""
        bad_cp = np.array([[1.0, 2.0]])  # (1, 2) instead of (M, 3)
        with pytest.raises(OcculusChangeDetectionError, match="core_points"):
            m3c2(flat_epoch1, flat_epoch2_shifted, core_points=bad_cp)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestM3C2EdgeCases:
    """Edge cases and boundary conditions."""

    def test_core_points_outside_cloud_produce_nan(self, flat_epoch1, flat_epoch2_shifted):
        """Core points far from both clouds should yield NaN distances."""
        far_cp = np.array([[999.0, 999.0, 999.0]], dtype=np.float64)
        result = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            core_points=far_cp,
            normal_scale=1.0,
            projection_scale=1.0,
        )
        assert np.isnan(result.distances[0])

    def test_nan_distances_not_significant(self, flat_epoch1, flat_epoch2_shifted):
        """NaN distances should not be marked as significant."""
        far_cp = np.array([[999.0, 999.0, 999.0]], dtype=np.float64)
        result = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            core_points=far_cp,
            normal_scale=1.0,
            projection_scale=1.0,
        )
        assert not result.significant_change[0]

    def test_custom_confidence(self, flat_epoch1, flat_epoch2_shifted):
        """Different confidence levels produce different LoD values."""
        r90 = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            normal_scale=2.0,
            projection_scale=2.0,
            confidence=0.90,
        )
        r99 = m3c2(
            flat_epoch1,
            flat_epoch2_shifted,
            normal_scale=2.0,
            projection_scale=2.0,
            confidence=0.99,
        )
        valid = np.isfinite(r90.uncertainties) & np.isfinite(r99.uncertainties)
        if valid.sum() > 0:
            # Higher confidence → larger LoD
            assert np.mean(r99.uncertainties[valid]) > np.mean(r90.uncertainties[valid])

    def test_minimum_viable_clouds(self):
        """m3c2 works with exactly 3 points in each epoch."""
        e1 = PointCloud(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        e2 = PointCloud(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64))
        result = m3c2(e1, e2, normal_scale=5.0, projection_scale=5.0)
        assert result.distances.shape == (3,)
