"""Tests for Open3D interop — mocked to avoid requiring open3d installed.

These tests verify the interop logic in types.py (to_open3d, from_open3d),
viz/__init__.py, and mesh/__init__.py using unittest.mock to patch open3d.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from occulus.exceptions import OcculusMeshError, OcculusValidationError
from occulus.types import AcquisitionMetadata, PointCloud


# ---------------------------------------------------------------------------
# to_open3d
# ---------------------------------------------------------------------------

class TestToOpen3D:
    """Tests for PointCloud.to_open3d()."""

    def test_raises_import_error_without_open3d(self):
        """to_open3d raises ImportError if open3d is not installed."""
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                cloud.to_open3d()

    def test_converts_xyz(self):
        """to_open3d passes xyz to Vector3dVector."""
        rng = np.random.default_rng(1)
        xyz = rng.random((50, 3)).astype(np.float64)
        cloud = PointCloud(xyz)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud.to_open3d()

        mock_o3d.utility.Vector3dVector.assert_called_once()

    def test_converts_normals_when_present(self):
        """to_open3d passes normals when cloud has them."""
        rng = np.random.default_rng(2)
        xyz = rng.random((20, 3)).astype(np.float64)
        normals = rng.random((20, 3)).astype(np.float64)
        cloud = PointCloud(xyz, normals=normals)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud.to_open3d()

        assert mock_o3d.utility.Vector3dVector.call_count >= 2

    def test_converts_rgb_when_present(self):
        """to_open3d passes colors (normalized to [0,1]) when cloud has rgb."""
        rng = np.random.default_rng(3)
        xyz = rng.random((20, 3)).astype(np.float64)
        rgb = (rng.random((20, 3)) * 255).astype(np.uint8)
        cloud = PointCloud(xyz, rgb=rgb)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud.to_open3d()

        # Vector3dVector called for xyz and colors
        assert mock_o3d.utility.Vector3dVector.call_count >= 2


# ---------------------------------------------------------------------------
# from_open3d
# ---------------------------------------------------------------------------

class TestFromOpen3D:
    """Tests for PointCloud.from_open3d()."""

    def test_raises_import_error_without_open3d(self):
        """from_open3d raises ImportError if open3d is not installed."""
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                PointCloud.from_open3d(MagicMock())

    def test_raises_validation_error_for_wrong_type(self):
        """from_open3d raises OcculusValidationError for non-PointCloud input."""
        mock_o3d = MagicMock()
        mock_o3d.geometry.PointCloud = type("PointCloud", (), {})  # real type check

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            with pytest.raises(OcculusValidationError):
                PointCloud.from_open3d("not a point cloud")

    def test_converts_points(self):
        """from_open3d extracts points from Open3D PointCloud."""
        mock_o3d = MagicMock()
        # Use a real class so isinstance() works
        FakeO3DPointCloud = type("PointCloud", (), {})
        mock_o3d.geometry.PointCloud = FakeO3DPointCloud

        xyz = np.random.default_rng(4).random((30, 3)).astype(np.float64)
        mock_pcd = FakeO3DPointCloud()
        mock_pcd.points = xyz
        mock_pcd.has_normals = lambda: False
        mock_pcd.has_colors = lambda: False

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = PointCloud.from_open3d(mock_pcd)

        np.testing.assert_array_equal(cloud.xyz, np.asarray(xyz))


# ---------------------------------------------------------------------------
# Mesh (mocked open3d)
# ---------------------------------------------------------------------------

class TestMeshResultToOpen3D:
    """Tests for MeshResult.to_open3d()."""

    def test_raises_import_error_without_open3d(self):
        """MeshResult.to_open3d raises ImportError if open3d is not installed."""
        from occulus.mesh import MeshResult
        mesh = MeshResult(
            vertices=np.zeros((3, 3)),
            faces=np.zeros((1, 3), dtype=np.int32),
        )
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                mesh.to_open3d()

    def test_converts_to_triangle_mesh(self):
        """MeshResult.to_open3d returns an Open3D TriangleMesh."""
        from occulus.mesh import MeshResult

        mock_o3d = MagicMock()
        mock_mesh = MagicMock()
        mock_o3d.geometry.TriangleMesh.return_value = mock_mesh

        mesh = MeshResult(
            vertices=np.zeros((3, 3)),
            faces=np.array([[0, 1, 2]], dtype=np.int32),
        )

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            result = mesh.to_open3d()

        mock_o3d.geometry.TriangleMesh.assert_called_once()
        mock_o3d.utility.Vector3dVector.assert_called()
        mock_o3d.utility.Vector3iVector.assert_called()

    def test_n_vertices_and_faces(self):
        """MeshResult properties return correct counts."""
        from occulus.mesh import MeshResult
        mesh = MeshResult(
            vertices=np.zeros((10, 3)),
            faces=np.zeros((5, 3), dtype=np.int32),
        )
        assert mesh.n_vertices == 10
        assert mesh.n_faces == 5

    def test_poisson_raises_without_normals(self):
        """poisson_mesh raises OcculusValidationError when cloud has no normals."""
        from occulus.mesh import poisson_mesh
        cloud = PointCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(OcculusValidationError, match="normals"):
            poisson_mesh(cloud)

    def test_bpa_raises_without_normals(self):
        """ball_pivoting_mesh raises OcculusValidationError when cloud has no normals."""
        from occulus.mesh import ball_pivoting_mesh
        cloud = PointCloud(np.random.default_rng(0).random((100, 3)))
        with pytest.raises(OcculusValidationError, match="normals"):
            ball_pivoting_mesh(cloud)

    def test_poisson_raises_import_error_without_open3d(self):
        """poisson_mesh raises ImportError if open3d not installed."""
        from occulus.mesh import poisson_mesh
        from occulus.normals import estimate_normals
        cloud = estimate_normals(PointCloud(np.random.default_rng(0).random((100, 3))))
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                poisson_mesh(cloud)

    def test_poisson_mesh_with_mocked_open3d(self):
        """poisson_mesh returns MeshResult when open3d is mocked."""
        from occulus.mesh import MeshResult, poisson_mesh
        from occulus.normals import estimate_normals

        cloud = estimate_normals(PointCloud(np.random.default_rng(0).random((50, 3))))

        FakeTM = type("TriangleMesh", (), {})
        mock_mesh = FakeTM()
        mock_mesh.vertices = np.zeros((4, 3))
        mock_mesh.triangles = np.zeros((2, 3), dtype=np.int32)
        mock_mesh.has_vertex_normals = lambda: False
        mock_mesh.has_vertex_colors = lambda: False
        mock_mesh.remove_vertices_by_mask = lambda m: None
        densities = np.ones(4)

        mock_o3d = MagicMock()
        mock_o3d.geometry.TriangleMesh = FakeTM
        FakeTM.create_from_point_cloud_poisson = staticmethod(
            lambda pcd, depth, width, scale, linear_fit: (mock_mesh, densities)
        )

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            with patch("occulus.types.PointCloud.to_open3d", return_value=MagicMock()):
                result = poisson_mesh(cloud, density_threshold_quantile=None)

        assert isinstance(result, MeshResult)
        assert result.n_vertices == 4

    def test_bpa_mesh_with_mocked_open3d(self):
        """ball_pivoting_mesh returns MeshResult when open3d is mocked."""
        from occulus.mesh import MeshResult, ball_pivoting_mesh
        from occulus.normals import estimate_normals

        cloud = estimate_normals(PointCloud(np.random.default_rng(1).random((50, 3))))

        FakeTM = type("TriangleMesh", (), {})
        mock_mesh = FakeTM()
        mock_mesh.vertices = np.zeros((3, 3))
        mock_mesh.triangles = np.zeros((1, 3), dtype=np.int32)
        mock_mesh.has_vertex_normals = lambda: False
        mock_mesh.has_vertex_colors = lambda: False

        mock_o3d = MagicMock()
        mock_o3d.geometry.TriangleMesh = FakeTM
        FakeTM.create_from_point_cloud_ball_pivoting = staticmethod(
            lambda pcd, radii: mock_mesh
        )

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            with patch("occulus.types.PointCloud.to_open3d", return_value=MagicMock()):
                result = ball_pivoting_mesh(cloud, radii=[0.1, 0.2])

        assert isinstance(result, MeshResult)
        assert result.n_vertices == 3


# ---------------------------------------------------------------------------
# Viz (mocked open3d)
# ---------------------------------------------------------------------------

class TestVizFunctions:
    """Tests for occulus.viz functions with mocked open3d."""

    def test_visualize_raises_without_open3d(self):
        """visualize raises ImportError if open3d not installed."""
        from occulus.viz import visualize
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                visualize(cloud)

    def test_visualize_raises_no_clouds(self):
        """visualize raises ValueError when called with no clouds."""
        from occulus.viz import visualize
        with pytest.raises(ValueError, match="at least one"):
            visualize()

    def test_visualize_calls_draw_geometries(self):
        """visualize calls open3d.visualization.draw_geometries."""
        from occulus.viz import visualize

        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd
        mock_o3d.utility.Vector3dVector.return_value = MagicMock()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            with patch.object(cloud, "to_open3d", return_value=mock_pcd):
                visualize(cloud)

        mock_o3d.visualization.draw_geometries.assert_called_once()

    def test_visualize_registration_raises_without_open3d(self):
        """visualize_registration raises ImportError if open3d not installed."""
        from occulus.registration.icp import RegistrationResult
        from occulus.viz import visualize_registration
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        result = RegistrationResult(
            transformation=np.eye(4), fitness=1.0, inlier_rmse=0.0, converged=True
        )
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                visualize_registration(cloud, cloud, result)

    def test_visualize_segments_label_length_mismatch_raises(self):
        """visualize_segments raises ValueError for wrong-length labels."""
        from occulus.viz import visualize_segments
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        with pytest.raises(ValueError, match="labels length"):
            visualize_segments(cloud, np.zeros(5, dtype=np.int32))

    def test_visualize_segments_raises_without_open3d(self):
        """visualize_segments raises ImportError if open3d not installed."""
        from occulus.viz import visualize_segments
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        labels = np.zeros(10, dtype=np.int32)
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                visualize_segments(cloud, labels)

    def test_visualize_segments_calls_draw(self):
        """visualize_segments calls draw_geometries with coloured cloud."""
        from occulus.viz import visualize_segments
        cloud = PointCloud(np.random.default_rng(0).random((30, 3)))
        labels = np.array([0] * 10 + [1] * 10 + [-1] * 10, dtype=np.int32)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            visualize_segments(cloud, labels)

        mock_o3d.visualization.draw_geometries.assert_called_once()
