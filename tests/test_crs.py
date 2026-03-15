"""Unit tests for occulus.crs coordinate reference system transforms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from occulus.crs import reproject, transform_coordinates
from occulus.exceptions import OcculusCRSError
from occulus.types import AcquisitionMetadata, PointCloud


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_xyz() -> np.ndarray:
    """Return a small (5, 3) coordinate array."""
    return np.array(
        [
            [500_000.0, 4_200_000.0, 100.0],
            [500_001.0, 4_200_001.0, 101.0],
            [500_002.0, 4_200_002.0, 102.0],
            [500_003.0, 4_200_003.0, 103.0],
            [500_004.0, 4_200_004.0, 104.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture()
def sample_cloud(sample_xyz: np.ndarray) -> PointCloud:
    """Return a PointCloud with a source CRS in metadata."""
    meta = AcquisitionMetadata(coordinate_system="EPSG:32617")
    return PointCloud(
        sample_xyz,
        intensity=np.ones(5, dtype=np.float64),
        metadata=meta,
    )


def _make_mock_transformer(
    x_offset: float = 1.0, y_offset: float = 2.0
) -> MagicMock:
    """Create a mock pyproj Transformer that applies a simple offset."""
    mock_transformer = MagicMock()
    mock_transformer.transform.side_effect = (
        lambda x, y: (x + x_offset, y + y_offset)
    )
    return mock_transformer


# ---------------------------------------------------------------------------
# transform_coordinates
# ---------------------------------------------------------------------------


class TestTransformCoordinates:
    """Tests for the low-level transform_coordinates function."""

    @patch("occulus.crs.transform.Transformer")
    def test_basic_transform(
        self, mock_transformer_cls: MagicMock, sample_xyz: np.ndarray
    ) -> None:
        """Coordinates are transformed and Z is preserved."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()

        result = transform_coordinates(sample_xyz, "EPSG:32617", "EPSG:4326")

        mock_transformer_cls.from_crs.assert_called_once_with(
            "EPSG:32617", "EPSG:4326", always_xy=True
        )
        np.testing.assert_array_almost_equal(result[:, 0], sample_xyz[:, 0] + 1.0)
        np.testing.assert_array_almost_equal(result[:, 1], sample_xyz[:, 1] + 2.0)
        np.testing.assert_array_equal(result[:, 2], sample_xyz[:, 2])

    @patch("occulus.crs.transform.Transformer")
    def test_same_crs_returns_copy(
        self, mock_transformer_cls: MagicMock, sample_xyz: np.ndarray
    ) -> None:
        """When source == target the array is copied without calling pyproj."""
        result = transform_coordinates(sample_xyz, "EPSG:4326", "EPSG:4326")

        mock_transformer_cls.from_crs.assert_not_called()
        np.testing.assert_array_equal(result, sample_xyz)
        assert result is not sample_xyz  # must be a copy

    def test_invalid_shape_raises(self) -> None:
        """Non-(N,3) input raises OcculusCRSError."""
        bad = np.ones((5, 2), dtype=np.float64)
        with pytest.raises(OcculusCRSError, match="must be an \\(N, 3\\) array"):
            transform_coordinates(bad, "EPSG:4326", "EPSG:32617")

    @patch("occulus.crs.transform.Transformer")
    def test_from_crs_failure_raises(
        self, mock_transformer_cls: MagicMock, sample_xyz: np.ndarray
    ) -> None:
        """Invalid CRS identifiers raise OcculusCRSError."""
        mock_transformer_cls.from_crs.side_effect = RuntimeError("bad CRS")

        with pytest.raises(OcculusCRSError, match="Failed to create transformer"):
            transform_coordinates(sample_xyz, "EPSG:BAD", "EPSG:WORSE")

    @patch("occulus.crs.transform.Transformer")
    def test_transform_failure_raises(
        self, mock_transformer_cls: MagicMock, sample_xyz: np.ndarray
    ) -> None:
        """A transform failure at runtime raises OcculusCRSError."""
        mock_t = MagicMock()
        mock_t.transform.side_effect = RuntimeError("transform boom")
        mock_transformer_cls.from_crs.return_value = mock_t

        with pytest.raises(OcculusCRSError, match="Coordinate transform failed"):
            transform_coordinates(sample_xyz, "EPSG:4326", "EPSG:32617")

    def test_pyproj_missing_raises(self, sample_xyz: np.ndarray) -> None:
        """If pyproj cannot be imported an OcculusCRSError is raised."""
        import builtins

        real_import = builtins.__import__

        def _blocked_import(
            name: str, *args: object, **kwargs: object
        ) -> object:
            if name == "pyproj":
                raise ImportError("No module named 'pyproj'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_blocked_import):
            with pytest.raises(OcculusCRSError, match="pyproj is required"):
                transform_coordinates(sample_xyz, "EPSG:4326", "EPSG:32617")

    @patch("occulus.crs.transform.Transformer")
    def test_output_is_contiguous_float64(
        self, mock_transformer_cls: MagicMock, sample_xyz: np.ndarray
    ) -> None:
        """Result array is contiguous float64."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()
        result = transform_coordinates(sample_xyz, "EPSG:32617", "EPSG:4326")

        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]
        assert result.shape == sample_xyz.shape


# ---------------------------------------------------------------------------
# reproject
# ---------------------------------------------------------------------------


class TestReproject:
    """Tests for the high-level reproject function."""

    @patch("occulus.crs.transform.Transformer")
    def test_basic_reproject(
        self, mock_transformer_cls: MagicMock, sample_cloud: PointCloud
    ) -> None:
        """Cloud is reprojected and metadata.coordinate_system is updated."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()

        result = reproject(sample_cloud, "EPSG:4326")

        assert result.metadata.coordinate_system == "EPSG:4326"
        assert result.n_points == sample_cloud.n_points
        # XYZ should have been shifted by the mock transformer
        np.testing.assert_array_almost_equal(
            result.xyz[:, 0], sample_cloud.xyz[:, 0] + 1.0
        )
        np.testing.assert_array_almost_equal(
            result.xyz[:, 1], sample_cloud.xyz[:, 1] + 2.0
        )
        # Z unchanged
        np.testing.assert_array_equal(result.xyz[:, 2], sample_cloud.xyz[:, 2])

    @patch("occulus.crs.transform.Transformer")
    def test_explicit_source_crs(
        self, mock_transformer_cls: MagicMock, sample_cloud: PointCloud
    ) -> None:
        """An explicit source_crs overrides cloud metadata."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()

        reproject(sample_cloud, "EPSG:4326", source_crs="EPSG:26917")

        mock_transformer_cls.from_crs.assert_called_once_with(
            "EPSG:26917", "EPSG:4326", always_xy=True
        )

    def test_no_source_crs_raises(self) -> None:
        """Missing source CRS (metadata and argument) raises OcculusCRSError."""
        cloud = PointCloud(np.ones((3, 3), dtype=np.float64))
        with pytest.raises(OcculusCRSError, match="Cannot determine source CRS"):
            reproject(cloud, "EPSG:4326")

    @patch("occulus.crs.transform.Transformer")
    def test_preserves_attributes(
        self, mock_transformer_cls: MagicMock
    ) -> None:
        """Per-point attributes are carried through to the result."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()

        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        cls_arr = np.array([2, 6], dtype=np.uint8)
        rgb_arr = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        meta = AcquisitionMetadata(coordinate_system="EPSG:32617")
        cloud = PointCloud(
            xyz, classification=cls_arr, rgb=rgb_arr, metadata=meta
        )

        result = reproject(cloud, "EPSG:4326")

        np.testing.assert_array_equal(result.classification, cls_arr)
        np.testing.assert_array_equal(result.rgb, rgb_arr)

    @patch("occulus.crs.transform.Transformer")
    def test_does_not_mutate_input(
        self, mock_transformer_cls: MagicMock, sample_cloud: PointCloud
    ) -> None:
        """The input cloud is not mutated by reproject."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()
        original_xyz = sample_cloud.xyz.copy()
        original_crs = sample_cloud.metadata.coordinate_system

        reproject(sample_cloud, "EPSG:4326")

        np.testing.assert_array_equal(sample_cloud.xyz, original_xyz)
        assert sample_cloud.metadata.coordinate_system == original_crs

    @patch("occulus.crs.transform.Transformer")
    def test_preserves_concrete_subtype(
        self, mock_transformer_cls: MagicMock
    ) -> None:
        """Reprojecting an AerialCloud returns an AerialCloud."""
        from occulus.types import AerialCloud

        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()
        xyz = np.ones((4, 3), dtype=np.float64)
        meta = AcquisitionMetadata(coordinate_system="EPSG:32617")
        cloud = AerialCloud(xyz, metadata=meta)

        result = reproject(cloud, "EPSG:4326")

        assert isinstance(result, AerialCloud)

    @patch("occulus.crs.transform.Transformer")
    def test_preserves_other_metadata_fields(
        self, mock_transformer_cls: MagicMock
    ) -> None:
        """Non-CRS metadata fields survive reprojection."""
        mock_transformer_cls.from_crs.return_value = _make_mock_transformer()
        meta = AcquisitionMetadata(
            coordinate_system="EPSG:32617",
            scanner_model="VLP-16",
            scan_date="2025-01-15",
            point_density_per_sqm=42.0,
        )
        cloud = PointCloud(np.ones((3, 3), dtype=np.float64), metadata=meta)

        result = reproject(cloud, "EPSG:4326")

        assert result.metadata.scanner_model == "VLP-16"
        assert result.metadata.scan_date == "2025-01-15"
        assert result.metadata.point_density_per_sqm == 42.0
