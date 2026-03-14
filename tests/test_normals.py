"""Tests for occulus.normals — normal estimation and orientation."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusValidationError
from occulus.normals import estimate_normals, orient_normals_to_viewpoint
from occulus.types import PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_cloud() -> PointCloud:
    """1 000 points on the XY plane (z=0)."""
    rng = np.random.default_rng(1)
    xy = rng.random((1000, 2))
    z = np.zeros(1000)
    xyz = np.column_stack((xy, z))
    return PointCloud(xyz.astype(np.float64))


@pytest.fixture
def sphere_cloud() -> PointCloud:
    """1 000 points distributed on a unit sphere."""
    rng = np.random.default_rng(2)
    pts = rng.standard_normal((1000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return PointCloud(pts.astype(np.float64))


# ---------------------------------------------------------------------------
# estimate_normals
# ---------------------------------------------------------------------------


class TestEstimateNormals:
    """Tests for estimate_normals."""

    def test_returns_cloud_with_normals(self, flat_cloud):
        """estimate_normals returns a cloud with normals set."""
        result = estimate_normals(flat_cloud)
        assert result.has_normals
        assert result.normals is not None

    def test_normals_are_unit_vectors(self, flat_cloud):
        """estimate_normals produces unit-length normal vectors."""
        result = estimate_normals(flat_cloud)
        assert result.normals is not None
        norms = np.linalg.norm(result.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normals_shape_matches_n_points(self, flat_cloud):
        """Normals array has shape (n_points, 3)."""
        result = estimate_normals(flat_cloud)
        assert result.normals is not None
        assert result.normals.shape == (flat_cloud.n_points, 3)

    def test_flat_cloud_normals_aligned_with_z(self, flat_cloud):
        """For a flat XY cloud, normals should be approximately ±Z."""
        result = estimate_normals(flat_cloud)
        assert result.normals is not None
        z_component = np.abs(result.normals[:, 2])
        assert z_component.mean() > 0.8

    def test_too_few_points_raises(self):
        """estimate_normals raises OcculusValidationError for < 3 points."""
        cloud = PointCloud(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        with pytest.raises(OcculusValidationError):
            estimate_normals(cloud)

    def test_auto_radius(self, flat_cloud):
        """estimate_normals without explicit radius auto-computes one."""
        result = estimate_normals(flat_cloud)
        assert result.has_normals

    def test_custom_radius(self, flat_cloud):
        """estimate_normals accepts a custom radius."""
        result = estimate_normals(flat_cloud, radius=0.1)
        assert result.has_normals

    def test_preserves_xyz(self, flat_cloud):
        """estimate_normals does not modify the xyz array."""
        result = estimate_normals(flat_cloud)
        np.testing.assert_array_equal(result.xyz, flat_cloud.xyz)


# ---------------------------------------------------------------------------
# orient_normals_to_viewpoint
# ---------------------------------------------------------------------------


class TestOrientNormalsToViewpoint:
    """Tests for orient_normals_to_viewpoint."""

    def test_requires_normals(self, flat_cloud):
        """orient_normals_to_viewpoint raises if cloud has no normals."""
        with pytest.raises(OcculusValidationError, match="no normals"):
            orient_normals_to_viewpoint(flat_cloud, np.array([0.0, 0.0, 10.0]))

    def test_invalid_viewpoint_shape_raises(self, flat_cloud):
        """orient_normals_to_viewpoint raises for wrong viewpoint shape."""
        cloud_with_normals = estimate_normals(flat_cloud)
        with pytest.raises(OcculusValidationError, match="viewpoint"):
            orient_normals_to_viewpoint(cloud_with_normals, np.array([0.0, 0.0]))

    def test_normals_face_viewpoint(self, flat_cloud):
        """After orientation, normals should generally face toward the viewpoint."""
        cloud_with_normals = estimate_normals(flat_cloud)
        viewpoint = np.array([0.5, 0.5, 10.0])  # above the flat cloud
        result = orient_normals_to_viewpoint(cloud_with_normals, viewpoint)
        assert result.normals is not None
        # Most normals should point upward (positive Z) toward viewpoint
        dot = np.einsum(
            "ij,j->i",
            result.normals,
            (viewpoint - result.xyz).mean(axis=0),
        )
        assert (dot > 0).mean() > 0.9

    def test_returns_new_cloud(self, flat_cloud):
        """orient_normals_to_viewpoint returns a new object (not in-place)."""
        cloud_with_normals = estimate_normals(flat_cloud)
        result = orient_normals_to_viewpoint(cloud_with_normals, np.array([0.0, 0.0, 10.0]))
        assert result is not cloud_with_normals

    def test_normals_remain_unit_length(self, flat_cloud):
        """Oriented normals are still unit vectors."""
        cloud_with_normals = estimate_normals(flat_cloud)
        result = orient_normals_to_viewpoint(cloud_with_normals, np.array([0.0, 0.0, 10.0]))
        assert result.normals is not None
        norms = np.linalg.norm(result.normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)
