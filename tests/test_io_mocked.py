"""Mocked tests for occulus.io readers/writers requiring laspy or open3d."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from occulus.types import AerialCloud, Platform, PointCloud


# ---------------------------------------------------------------------------
# Mocked LAS reader
# ---------------------------------------------------------------------------


class TestReadLasMocked:
    """Tests for _read_las with mocked laspy."""

    def _make_mock_las(self, n: int = 20) -> MagicMock:
        rng = np.random.default_rng(0)
        xyz = rng.random((n, 3))
        # Use spec to prevent MagicMock from auto-creating red/green/blue
        las = MagicMock(spec=[
            "x", "y", "z", "intensity", "classification",
            "return_number", "number_of_returns", "header",
        ])
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        las.intensity = (rng.random(n) * 1000).astype(np.float32)
        las.classification = np.zeros(n, dtype=np.uint8)
        las.return_number = np.ones(n, dtype=np.uint8)
        las.number_of_returns = np.ones(n, dtype=np.uint8)
        las.header = MagicMock()
        las.header.point_count = n
        las.header.vlrs = []
        return las

    def test_raises_import_error_without_laspy(self, tmp_path):
        """_read_las raises ImportError if laspy not installed."""
        from occulus.io.readers import _read_las
        with patch.dict("sys.modules", {"laspy": None}):
            with pytest.raises(ImportError, match="laspy"):
                _read_las(tmp_path / "test.las", platform=Platform.UNKNOWN, subsample=None)

    def test_reads_basic_las(self, tmp_path):
        """_read_las returns a PointCloud from a mocked laspy object."""
        from occulus.io.readers import _read_las
        mock_las = self._make_mock_las(30)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            cloud = _read_las(tmp_path / "cloud.las", platform=Platform.UNKNOWN, subsample=None)

        assert cloud.n_points == 30
        assert cloud.intensity is not None
        assert cloud.classification is not None

    def test_reads_las_with_rgb(self, tmp_path):
        """_read_las extracts RGB when present in LAS."""
        from occulus.io.readers import _read_las
        n = 20
        mock_las = self._make_mock_las(n)
        mock_las.red = np.full(n, 30000, dtype=np.uint16)
        mock_las.green = np.full(n, 20000, dtype=np.uint16)
        mock_las.blue = np.full(n, 10000, dtype=np.uint16)

        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            cloud = _read_las(tmp_path / "rgb.las", platform=Platform.UNKNOWN, subsample=None)

        assert cloud.rgb is not None
        assert cloud.rgb.shape == (n, 3)

    def test_reads_las_with_subsample(self, tmp_path):
        """_read_las applies subsampling."""
        from occulus.io.readers import _read_las
        mock_las = self._make_mock_las(100)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            cloud = _read_las(tmp_path / "cloud.las", platform=Platform.AERIAL, subsample=0.5)

        assert cloud.n_points <= 60

    def test_raises_io_error_on_laspy_failure(self, tmp_path):
        """_read_las raises OcculusIOError if laspy.read() fails."""
        from occulus.exceptions import OcculusIOError
        from occulus.io.readers import _read_las
        mock_laspy = MagicMock()
        mock_laspy.read.side_effect = RuntimeError("bad file")

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            with pytest.raises(OcculusIOError, match="Failed to read LAS"):
                _read_las(tmp_path / "bad.las", platform=Platform.UNKNOWN, subsample=None)

    def test_read_dispatcher_calls_las_reader(self, tmp_path):
        """read() dispatcher routes .las extension to _read_las."""
        from occulus.io.readers import read
        mock_las = self._make_mock_las(10)
        mock_laspy = MagicMock()
        mock_laspy.read.return_value = mock_las

        las_path = tmp_path / "test.las"
        las_path.touch()

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            cloud = read(las_path)

        assert cloud.n_points == 10


# ---------------------------------------------------------------------------
# Mocked PLY reader
# ---------------------------------------------------------------------------


class TestReadPlyMocked:
    """Tests for _read_ply with mocked open3d."""

    def test_raises_import_error_without_open3d(self, tmp_path):
        """_read_ply raises ImportError if open3d not installed."""
        from occulus.io.readers import _read_ply
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                _read_ply(tmp_path / "test.ply", platform=Platform.UNKNOWN, subsample=None)

    def test_reads_basic_ply(self, tmp_path):
        """_read_ply returns PointCloud from mocked open3d pcd."""
        from occulus.io.readers import _read_ply
        rng = np.random.default_rng(1)
        xyz = rng.random((25, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        ply_path = tmp_path / "test.ply"
        ply_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = _read_ply(ply_path, platform=Platform.UNKNOWN, subsample=None)

        assert cloud.n_points == 25

    def test_reads_ply_with_normals_and_colors(self, tmp_path):
        """_read_ply transfers normals and colors when present."""
        from occulus.io.readers import _read_ply
        rng = np.random.default_rng(2)
        xyz = rng.random((15, 3)).astype(np.float64)
        normals = rng.random((15, 3)).astype(np.float64)
        colors = rng.random((15, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.normals = normals
        mock_pcd.colors = colors
        mock_pcd.has_normals.return_value = True
        mock_pcd.has_colors.return_value = True
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        ply_path = tmp_path / "test.ply"
        ply_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = _read_ply(ply_path, platform=Platform.UNKNOWN, subsample=None)

        assert cloud.has_normals
        assert cloud.rgb is not None

    def test_raises_io_error_on_empty_ply(self, tmp_path):
        """_read_ply raises OcculusIOError for empty point cloud."""
        from occulus.exceptions import OcculusIOError
        from occulus.io.readers import _read_ply

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = np.zeros((0, 3))
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        ply_path = tmp_path / "empty.ply"
        ply_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            with pytest.raises(OcculusIOError, match="no points"):
                _read_ply(ply_path, platform=Platform.UNKNOWN, subsample=None)

    def test_reads_ply_with_subsample(self, tmp_path):
        """_read_ply applies subsampling."""
        from occulus.io.readers import _read_ply
        rng = np.random.default_rng(3)
        xyz = rng.random((100, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        ply_path = tmp_path / "big.ply"
        ply_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = _read_ply(ply_path, platform=Platform.UNKNOWN, subsample=0.5)

        assert cloud.n_points <= 60

    def test_read_dispatcher_calls_ply_reader(self, tmp_path):
        """read() dispatcher routes .ply extension to _read_ply."""
        from occulus.io.readers import read
        rng = np.random.default_rng(4)
        xyz = rng.random((10, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        ply_path = tmp_path / "test.ply"
        ply_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = read(ply_path)

        assert cloud.n_points == 10


# ---------------------------------------------------------------------------
# Mocked PCD reader
# ---------------------------------------------------------------------------


class TestReadPcdMocked:
    """Tests for _read_pcd with mocked open3d."""

    def test_raises_import_error_without_open3d(self, tmp_path):
        """_read_pcd raises ImportError if open3d not installed."""
        from occulus.io.readers import _read_pcd
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                _read_pcd(tmp_path / "test.pcd", platform=Platform.UNKNOWN, subsample=None)

    def test_reads_basic_pcd(self, tmp_path):
        """_read_pcd returns PointCloud from mocked open3d pcd."""
        from occulus.io.readers import _read_pcd
        rng = np.random.default_rng(5)
        xyz = rng.random((20, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        pcd_path = tmp_path / "test.pcd"
        pcd_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = _read_pcd(pcd_path, platform=Platform.UNKNOWN, subsample=None)

        assert cloud.n_points == 20

    def test_read_dispatcher_calls_pcd_reader(self, tmp_path):
        """read() dispatcher routes .pcd extension to _read_pcd."""
        from occulus.io.readers import read
        rng = np.random.default_rng(6)
        xyz = rng.random((10, 3)).astype(np.float64)

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_pcd.points = xyz
        mock_pcd.has_normals.return_value = False
        mock_pcd.has_colors.return_value = False
        mock_o3d.io.read_point_cloud.return_value = mock_pcd

        pcd_path = tmp_path / "test.pcd"
        pcd_path.touch()

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            cloud = read(pcd_path)

        assert cloud.n_points == 10


# ---------------------------------------------------------------------------
# Mocked LAS writer
# ---------------------------------------------------------------------------


class TestWriteLasMocked:
    """Tests for _write_las with mocked laspy."""

    def test_raises_import_error_without_laspy(self, tmp_path):
        """_write_las raises ImportError if laspy not installed."""
        from occulus.io.writers import _write_las
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        with patch.dict("sys.modules", {"laspy": None}):
            with pytest.raises(ImportError, match="laspy"):
                _write_las(cloud, tmp_path / "out.las", compress=False)

    def test_writes_basic_las(self, tmp_path):
        """_write_las calls laspy.LasData.write()."""
        from occulus.io.writers import _write_las
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))

        mock_laspy = MagicMock()
        mock_header = MagicMock()
        mock_las_data = MagicMock()
        mock_laspy.LasHeader.return_value = mock_header
        mock_laspy.LasData.return_value = mock_las_data

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            result = _write_las(cloud, tmp_path / "out.las", compress=False)

        mock_las_data.write.assert_called_once()
        assert result == tmp_path / "out.las"

    def test_writes_las_with_all_attributes(self, tmp_path):
        """_write_las sets intensity, classification, return_number, rgb."""
        from occulus.io.writers import _write_las
        rng = np.random.default_rng(1)
        n = 15
        cloud = PointCloud(
            rng.random((n, 3)),
            intensity=rng.random(n).astype(np.float64),
            classification=np.zeros(n, dtype=np.uint8),
            return_number=np.ones(n, dtype=np.uint8),
            number_of_returns=np.ones(n, dtype=np.uint8),
            rgb=(rng.random((n, 3)) * 255).astype(np.uint8),
        )

        mock_laspy = MagicMock()
        mock_header = MagicMock()
        mock_las_data = MagicMock()
        mock_laspy.LasHeader.return_value = mock_header
        mock_laspy.LasData.return_value = mock_las_data

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            result = _write_las(cloud, tmp_path / "full.las", compress=False)

        mock_las_data.write.assert_called_once()

    def test_write_dispatcher_routes_las(self, tmp_path):
        """write() dispatcher routes .las extension."""
        from occulus.io.writers import write
        cloud = PointCloud(np.random.default_rng(0).random((5, 3)))

        mock_laspy = MagicMock()
        mock_header = MagicMock()
        mock_las_data = MagicMock()
        mock_laspy.LasHeader.return_value = mock_header
        mock_laspy.LasData.return_value = mock_las_data

        with patch.dict("sys.modules", {"laspy": mock_laspy}):
            result = write(cloud, tmp_path / "out.las")

        assert result == tmp_path / "out.las"


# ---------------------------------------------------------------------------
# Mocked PLY writer
# ---------------------------------------------------------------------------


class TestWritePlyMocked:
    """Tests for _write_ply with mocked open3d."""

    def test_raises_import_error_without_open3d(self, tmp_path):
        """_write_ply raises ImportError if open3d not installed."""
        from occulus.io.writers import _write_ply
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))
        with patch.dict("sys.modules", {"open3d": None}):
            with pytest.raises(ImportError, match="open3d"):
                _write_ply(cloud, tmp_path / "out.ply")

    def test_writes_basic_ply(self, tmp_path):
        """_write_ply calls o3d.io.write_point_cloud()."""
        from occulus.io.writers import _write_ply
        cloud = PointCloud(np.random.default_rng(0).random((10, 3)))

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            result = _write_ply(cloud, tmp_path / "out.ply")

        mock_o3d.io.write_point_cloud.assert_called_once()
        assert result == tmp_path / "out.ply"

    def test_writes_ply_with_normals_and_rgb(self, tmp_path):
        """_write_ply transfers normals and colors to Open3D pcd."""
        from occulus.io.writers import _write_ply
        rng = np.random.default_rng(2)
        n = 10
        cloud = PointCloud(
            rng.random((n, 3)),
            normals=rng.random((n, 3)).astype(np.float64),
            rgb=(rng.random((n, 3)) * 255).astype(np.uint8),
        )

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            _write_ply(cloud, tmp_path / "normals.ply")

        # Vector3dVector called for xyz, normals, colors
        assert mock_o3d.utility.Vector3dVector.call_count >= 3

    def test_write_dispatcher_routes_ply(self, tmp_path):
        """write() dispatcher routes .ply extension."""
        from occulus.io.writers import write
        cloud = PointCloud(np.random.default_rng(0).random((5, 3)))

        mock_o3d = MagicMock()
        mock_pcd = MagicMock()
        mock_o3d.geometry.PointCloud.return_value = mock_pcd

        with patch.dict("sys.modules", {"open3d": mock_o3d}):
            result = write(cloud, tmp_path / "out.ply")

        assert result == tmp_path / "out.ply"
