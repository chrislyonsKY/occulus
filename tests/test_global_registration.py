"""Tests for occulus.registration.global_registration — FPFH and RANSAC."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusRegistrationError, OcculusValidationError
from occulus.normals import estimate_normals
from occulus.registration.global_registration import (
    AlignmentResult,
    _svd_rigid_3pt,
    compute_fpfh,
    ransac_registration,
)
from occulus.types import PointCloud


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_cloud_a() -> PointCloud:
    """300-point planar cloud for RANSAC tests."""
    rng = np.random.default_rng(20)
    xy = rng.random((300, 2)) * 10.0
    z = rng.random(300) * 0.01
    cloud = PointCloud(np.column_stack((xy, z)).astype(np.float64))
    return estimate_normals(cloud, radius=1.0)


@pytest.fixture
def flat_cloud_b(flat_cloud_a) -> PointCloud:
    """Same cloud shifted by a small translation."""
    xyz = flat_cloud_a.xyz + np.array([0.1, 0.1, 0.0])
    cloud = PointCloud(xyz)
    return estimate_normals(cloud, radius=1.0)


# ---------------------------------------------------------------------------
# FPFH descriptors
# ---------------------------------------------------------------------------

class TestComputeFPFH:
    """Tests for compute_fpfh."""

    def test_requires_normals(self):
        """compute_fpfh raises OcculusValidationError if cloud has no normals."""
        cloud = PointCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(OcculusValidationError, match="normals"):
            compute_fpfh(cloud, radius=0.5)

    def test_returns_33_dim_array(self, flat_cloud_a):
        """compute_fpfh returns (N, 33) array."""
        features = compute_fpfh(flat_cloud_a, radius=1.0)
        assert features.shape == (flat_cloud_a.n_points, 33)

    def test_features_non_negative(self, flat_cloud_a):
        """FPFH descriptor values are non-negative."""
        features = compute_fpfh(flat_cloud_a, radius=1.0)
        assert np.all(features >= 0)

    def test_features_are_normalised(self, flat_cloud_a):
        """Each FPFH descriptor sums to approximately 1."""
        features = compute_fpfh(flat_cloud_a, radius=2.0)
        row_sums = features.sum(axis=1)
        # Points with at least one neighbour should be normalised
        active = row_sums > 0
        np.testing.assert_allclose(row_sums[active], 1.0, atol=0.01)

    def test_max_nn_limit(self, flat_cloud_a):
        """compute_fpfh respects max_nn parameter."""
        features = compute_fpfh(flat_cloud_a, radius=2.0, max_nn=5)
        assert features.shape == (flat_cloud_a.n_points, 33)


# ---------------------------------------------------------------------------
# RANSAC registration
# ---------------------------------------------------------------------------

class TestRansacRegistration:
    """Tests for ransac_registration."""

    def test_feature_shape_mismatch_raises(self, flat_cloud_a, flat_cloud_b):
        """ransac_registration raises if feature rows don't match cloud n_points."""
        feats_a = compute_fpfh(flat_cloud_a, radius=1.0)
        feats_b = compute_fpfh(flat_cloud_b, radius=1.0)
        with pytest.raises(OcculusValidationError):
            ransac_registration(flat_cloud_a, flat_cloud_b, feats_a[:-1], feats_b)

    def test_returns_registration_result(self, flat_cloud_a, flat_cloud_b):
        """ransac_registration returns a RegistrationResult."""
        from occulus.registration.icp import RegistrationResult
        feats_a = compute_fpfh(flat_cloud_a, radius=1.0)
        feats_b = compute_fpfh(flat_cloud_b, radius=1.0)
        result = ransac_registration(
            flat_cloud_a, flat_cloud_b, feats_a, feats_b,
            max_correspondence_distance=0.5,
            max_iterations=1000,
        )
        assert isinstance(result, RegistrationResult)

    def test_transformation_is_4x4(self, flat_cloud_a, flat_cloud_b):
        """Transformation matrix is (4, 4)."""
        feats_a = compute_fpfh(flat_cloud_a, radius=1.0)
        feats_b = compute_fpfh(flat_cloud_b, radius=1.0)
        result = ransac_registration(
            flat_cloud_a, flat_cloud_b, feats_a, feats_b,
            max_correspondence_distance=0.5,
            max_iterations=500,
        )
        assert result.transformation.shape == (4, 4)

    def test_fitness_in_range(self, flat_cloud_a, flat_cloud_b):
        """Fitness is in [0, 1]."""
        feats_a = compute_fpfh(flat_cloud_a, radius=1.0)
        feats_b = compute_fpfh(flat_cloud_b, radius=1.0)
        result = ransac_registration(
            flat_cloud_a, flat_cloud_b, feats_a, feats_b,
            max_correspondence_distance=1.0,
            max_iterations=500,
        )
        assert 0.0 <= result.fitness <= 1.0


# ---------------------------------------------------------------------------
# AlignmentResult
# ---------------------------------------------------------------------------

class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_global_rmse_computed(self):
        """AlignmentResult computes global_rmse from pairwise results."""
        from occulus.registration.icp import RegistrationResult
        r1 = RegistrationResult(
            transformation=np.eye(4), fitness=0.9, inlier_rmse=0.05, converged=True
        )
        r2 = RegistrationResult(
            transformation=np.eye(4), fitness=0.8, inlier_rmse=0.03, converged=True
        )
        result = AlignmentResult(
            transformations=[np.eye(4), np.eye(4), np.eye(4)],
            pairwise_results=[r1, r2],
        )
        assert result.global_rmse == pytest.approx(0.04, abs=1e-6)

    def test_empty_results_zero_rmse(self):
        """AlignmentResult with no pairwise results has global_rmse=0."""
        result = AlignmentResult(transformations=[], pairwise_results=[])
        assert result.global_rmse == 0.0


# ---------------------------------------------------------------------------
# _svd_rigid_3pt helper
# ---------------------------------------------------------------------------

class TestSvdRigid3pt:
    """Tests for the _svd_rigid_3pt internal helper."""

    def test_identity_for_same_points(self):
        """_svd_rigid_3pt returns I and 0 for identical src and tgt."""
        pts = np.random.default_rng(5).random((10, 3))
        R, t = _svd_rigid_3pt(pts, pts)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(t, np.zeros(3), atol=1e-8)

    def test_recovers_translation(self):
        """_svd_rigid_3pt recovers a known translation."""
        rng = np.random.default_rng(6)
        src = rng.random((20, 3))
        offset = np.array([2.0, -1.0, 0.5])
        tgt = src + offset
        R, t = _svd_rigid_3pt(src, tgt)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(t, offset, atol=1e-8)
