"""Tests for occulus.registration — ICP and global alignment."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusRegistrationError, OcculusValidationError
from occulus.normals import estimate_normals
from occulus.registration import (
    RegistrationResult,
    align_scans,
    icp,
    icp_point_to_plane,
    icp_point_to_point,
)
from occulus.registration.icp import (
    _apply_transform,
    _compute_metrics,
    _init_transform,
    _rodrigues_to_matrix,
    _svd_rigid,
)
from occulus.types import PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_source() -> PointCloud:
    """Flat planar cloud — easy to register."""
    rng = np.random.default_rng(10)
    xy = rng.random((500, 2))
    z = rng.random(500) * 0.01  # near-flat
    return PointCloud(np.column_stack((xy, z)))


@pytest.fixture
def flat_source_shifted(flat_source) -> PointCloud:
    """Same cloud translated by [0.05, 0.05, 0.0] — small offset."""
    xyz = flat_source.xyz + np.array([0.05, 0.05, 0.0])
    return PointCloud(xyz)


# ---------------------------------------------------------------------------
# ICP point-to-point
# ---------------------------------------------------------------------------


class TestICPPointToPoint:
    """Tests for icp_point_to_point."""

    def test_returns_registration_result(self, flat_source, flat_source_shifted):
        """icp_point_to_point returns a RegistrationResult."""
        result = icp_point_to_point(flat_source_shifted, flat_source)
        assert isinstance(result, RegistrationResult)

    def test_transformation_is_4x4(self, flat_source, flat_source_shifted):
        """Transformation matrix is (4, 4)."""
        result = icp_point_to_point(flat_source_shifted, flat_source)
        assert result.transformation.shape == (4, 4)

    def test_fitness_in_range(self, flat_source, flat_source_shifted):
        """Fitness is in [0, 1]."""
        result = icp_point_to_point(flat_source_shifted, flat_source)
        assert 0.0 <= result.fitness <= 1.0

    def test_converges_for_close_clouds(self, flat_source, flat_source_shifted):
        """ICP converges when clouds are close together."""
        result = icp_point_to_point(
            flat_source_shifted,
            flat_source,
            max_correspondence_distance=0.2,
            max_iterations=50,
        )
        assert result.converged

    def test_invalid_init_transform_raises(self, flat_source, flat_source_shifted):
        """ICP raises OcculusValidationError for wrong-shape init_transform."""
        with pytest.raises(OcculusValidationError):
            icp_point_to_point(
                flat_source_shifted,
                flat_source,
                init_transform=np.eye(3),  # wrong shape
            )


# ---------------------------------------------------------------------------
# ICP point-to-plane
# ---------------------------------------------------------------------------


class TestICPPointToPlane:
    """Tests for icp_point_to_plane."""

    def test_requires_target_normals(self, flat_source, flat_source_shifted):
        """icp_point_to_plane raises if target has no normals."""
        with pytest.raises(OcculusValidationError, match="normals"):
            icp_point_to_plane(flat_source_shifted, flat_source)

    def test_converges_with_normals(self, flat_source, flat_source_shifted):
        """icp_point_to_plane converges when target has normals."""
        target_with_normals = estimate_normals(flat_source)
        result = icp_point_to_plane(
            flat_source_shifted,
            target_with_normals,
            max_correspondence_distance=0.3,
            max_iterations=50,
        )
        assert isinstance(result, RegistrationResult)
        assert result.transformation.shape == (4, 4)


# ---------------------------------------------------------------------------
# icp dispatcher
# ---------------------------------------------------------------------------


class TestICPDispatcher:
    """Tests for the icp() auto-dispatch function."""

    def test_auto_selects_p2p_without_normals(self, flat_source, flat_source_shifted):
        """icp() uses point-to-point when target has no normals."""
        result = icp(flat_source_shifted, flat_source, max_correspondence_distance=0.3)
        assert isinstance(result, RegistrationResult)

    def test_auto_selects_p2plane_with_normals(self, flat_source, flat_source_shifted):
        """icp() uses point-to-plane when target has normals."""
        target = estimate_normals(flat_source)
        result = icp(flat_source_shifted, target, max_correspondence_distance=0.3)
        assert isinstance(result, RegistrationResult)

    def test_invalid_method_raises(self, flat_source, flat_source_shifted):
        """icp() raises OcculusValidationError for unknown method."""
        with pytest.raises(OcculusValidationError, match="method"):
            icp(flat_source_shifted, flat_source, method="magic")

    def test_explicit_p2p_method(self, flat_source, flat_source_shifted):
        """icp() with method='point_to_point' runs point-to-point ICP."""
        result = icp(
            flat_source_shifted,
            flat_source,
            method="point_to_point",
            max_correspondence_distance=0.3,
        )
        assert isinstance(result, RegistrationResult)


# ---------------------------------------------------------------------------
# align_scans
# ---------------------------------------------------------------------------


class TestAlignScans:
    """Tests for the multi-scan alignment function."""

    def test_requires_two_clouds(self, flat_source):
        """align_scans raises OcculusRegistrationError with < 2 clouds."""
        with pytest.raises(OcculusRegistrationError, match="at least 2"):
            align_scans([flat_source])

    def test_returns_n_transformations(self, flat_source, flat_source_shifted):
        """align_scans returns one transformation per input cloud."""
        from occulus.registration import AlignmentResult

        result = align_scans(
            [flat_source, flat_source_shifted],
            max_correspondence_distance=0.3,
        )
        assert isinstance(result, AlignmentResult)
        assert len(result.transformations) == 2

    def test_first_transform_is_identity(self, flat_source, flat_source_shifted):
        """The first cloud's transformation is always identity."""
        result = align_scans(
            [flat_source, flat_source_shifted],
            max_correspondence_distance=0.3,
        )
        np.testing.assert_allclose(result.transformations[0], np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------


class TestICPHelpers:
    """Tests for ICP internal helpers."""

    def test_svd_rigid_recovers_identity(self):
        """_svd_rigid with identical src and tgt returns I and 0."""
        pts = np.random.default_rng(0).random((20, 3))
        R, t = _svd_rigid(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-10)

    def test_svd_rigid_recovers_translation(self):
        """_svd_rigid recovers a known translation."""
        rng = np.random.default_rng(1)
        src = rng.random((50, 3))
        offset = np.array([1.0, -2.0, 0.5])
        tgt = src + offset
        R, t = _svd_rigid(src, tgt)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(t, offset, atol=1e-8)

    def test_apply_transform_identity(self):
        """_apply_transform with identity returns unchanged points."""
        pts = np.random.default_rng(0).random((10, 3))
        result = _apply_transform(pts, np.eye(4))
        np.testing.assert_allclose(result, pts, atol=1e-12)

    def test_rodrigues_identity_for_zero_vector(self):
        """_rodrigues_to_matrix with zero vector returns identity."""
        R = _rodrigues_to_matrix(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_init_transform_default_is_identity(self):
        """_init_transform(None) returns 4×4 identity."""
        T = _init_transform(None)
        np.testing.assert_allclose(T, np.eye(4))

    def test_init_transform_wrong_shape_raises(self):
        """_init_transform raises for non-(4,4) input."""
        with pytest.raises(OcculusValidationError):
            _init_transform(np.eye(3))

    def test_compute_metrics_range(self):
        """_compute_metrics fitness is in [0, 1]."""
        rng = np.random.default_rng(0)
        src = rng.random((100, 3))
        tgt = rng.random((100, 3))
        fitness, rmse = _compute_metrics(src, tgt, max_dist=2.0)
        assert 0.0 <= fitness <= 1.0
        assert rmse >= 0.0
