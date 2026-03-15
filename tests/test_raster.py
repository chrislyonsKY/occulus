"""Unit tests for the occulus.raster module.

Tests cover DEM/DSM/DTM creation, IDW and nearest-neighbour interpolation,
the RasterResult dataclass, and GeoTIFF export (with mocked rasterio).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from occulus.exceptions import OcculusExportError, OcculusRasterError
from occulus.raster import (
    RasterResult,
    create_dem,
    create_dsm,
    create_dtm,
    idw_interpolate,
    nearest_interpolate,
)
from occulus.types import AcquisitionMetadata, PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_cloud() -> PointCloud:
    """A 10-point cloud on a flat plane at z=5.0, no classification."""
    rng = np.random.default_rng(42)
    xy = rng.uniform(0.0, 10.0, size=(10, 2))
    z = np.full(10, 5.0)
    xyz = np.column_stack((xy, z))
    return PointCloud(xyz)


@pytest.fixture()
def classified_cloud() -> PointCloud:
    """A cloud with ground (class 2) and vegetation points."""
    rng = np.random.default_rng(99)
    n_ground = 50
    n_veg = 30

    # Ground points at z ~ 100
    ground_xy = rng.uniform(0.0, 20.0, size=(n_ground, 2))
    ground_z = 100.0 + rng.normal(0.0, 0.5, size=n_ground)
    ground_xyz = np.column_stack((ground_xy, ground_z))

    # Vegetation points at z ~ 115
    veg_xy = rng.uniform(0.0, 20.0, size=(n_veg, 2))
    veg_z = 115.0 + rng.normal(0.0, 1.0, size=n_veg)
    veg_xyz = np.column_stack((veg_xy, veg_z))

    xyz = np.vstack((ground_xyz, veg_xyz))
    classification = np.concatenate(
        [
            np.full(n_ground, 2, dtype=np.uint8),
            np.full(n_veg, 1, dtype=np.uint8),
        ]
    )

    return PointCloud(xyz, classification=classification)


@pytest.fixture()
def source_points() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simple XY + Z arrays for interpolation tests."""
    xy = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )
    z = np.array([1.0, 2.0, 3.0, 4.0, 2.5], dtype=np.float64)
    return xy, z


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------


class TestIDWInterpolate:
    """Tests for idw_interpolate."""

    def test_basic_shape(
        self, source_points: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Output shape matches grid dimensions."""
        xy, z = source_points
        grid_x = np.array([2.5, 7.5])
        grid_y = np.array([2.5, 7.5])
        result = idw_interpolate(xy, z, grid_x, grid_y)
        assert result.shape == (2, 2)

    def test_exact_point_returns_exact_value(self) -> None:
        """Grid cell at a source point should return its exact Z."""
        xy = np.array([[5.0, 5.0]], dtype=np.float64)
        z = np.array([42.0], dtype=np.float64)
        grid_x = np.array([5.0])
        grid_y = np.array([5.0])
        result = idw_interpolate(xy, z, grid_x, grid_y)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 42.0)

    def test_nodata_with_max_dist(self) -> None:
        """Cells beyond max_dist should be nodata."""
        xy = np.array([[0.0, 0.0]], dtype=np.float64)
        z = np.array([10.0], dtype=np.float64)
        grid_x = np.array([0.0, 100.0])
        grid_y = np.array([0.0])
        result = idw_interpolate(xy, z, grid_x, grid_y, max_dist=1.0, nodata=-9999.0)
        np.testing.assert_allclose(result[0, 0], 10.0)
        assert result[0, 1] == -9999.0

    def test_invalid_xy_shape(self) -> None:
        """Wrong xy shape raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="xy must be"):
            idw_interpolate(
                np.ones((5, 3)),
                np.ones(5),
                np.array([0.0]),
                np.array([0.0]),
            )

    def test_mismatched_z_length(self) -> None:
        """Z length not matching xy rows raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="z must be"):
            idw_interpolate(
                np.ones((5, 2)),
                np.ones(3),
                np.array([0.0]),
                np.array([0.0]),
            )

    def test_empty_points_raises(self) -> None:
        """Zero source points raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="zero source"):
            idw_interpolate(
                np.empty((0, 2)),
                np.empty(0),
                np.array([0.0]),
                np.array([0.0]),
            )

    def test_negative_power_raises(self) -> None:
        """Non-positive power raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="power must be positive"):
            idw_interpolate(
                np.ones((2, 2)),
                np.ones(2),
                np.array([0.0]),
                np.array([0.0]),
                power=-1.0,
            )


class TestNearestInterpolate:
    """Tests for nearest_interpolate."""

    def test_basic_shape(
        self, source_points: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        """Output shape matches grid dimensions."""
        xy, z = source_points
        grid_x = np.array([2.5, 7.5])
        grid_y = np.array([2.5, 7.5])
        result = nearest_interpolate(xy, z, grid_x, grid_y)
        assert result.shape == (2, 2)

    def test_assigns_nearest_z(self) -> None:
        """Each cell gets the Z of the nearest source point."""
        xy = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        z = np.array([1.0, 9.0], dtype=np.float64)
        grid_x = np.array([1.0, 9.0])
        grid_y = np.array([1.0, 9.0])
        result = nearest_interpolate(xy, z, grid_x, grid_y)
        # (1,1) is closer to (0,0), (9,9) is closer to (10,10)
        np.testing.assert_allclose(result[0, 0], 1.0)
        np.testing.assert_allclose(result[1, 1], 9.0)

    def test_max_dist_nodata(self) -> None:
        """Cells beyond max_dist are set to nodata."""
        xy = np.array([[0.0, 0.0]], dtype=np.float64)
        z = np.array([5.0], dtype=np.float64)
        grid_x = np.array([0.0, 50.0])
        grid_y = np.array([0.0])
        result = nearest_interpolate(xy, z, grid_x, grid_y, max_dist=1.0)
        np.testing.assert_allclose(result[0, 0], 5.0)
        assert result[0, 1] == -9999.0

    def test_invalid_inputs(self) -> None:
        """Invalid shapes raise OcculusRasterError."""
        with pytest.raises(OcculusRasterError):
            nearest_interpolate(
                np.ones((5, 3)),
                np.ones(5),
                np.array([0.0]),
                np.array([0.0]),
            )


# ---------------------------------------------------------------------------
# DEM / DSM / DTM tests
# ---------------------------------------------------------------------------


class TestCreateDSM:
    """Tests for create_dsm."""

    def test_returns_raster_result(self, simple_cloud: PointCloud) -> None:
        """create_dsm returns a RasterResult."""
        result = create_dsm(simple_cloud, resolution=2.0)
        assert isinstance(result, RasterResult)
        assert result.data.ndim == 2
        assert result.resolution == 2.0

    def test_dsm_captures_max_z(self) -> None:
        """DSM should reflect the maximum Z in each cell."""
        # Two points in the same cell, different Z
        xyz = np.array(
            [
                [1.0, 1.0, 5.0],
                [1.0, 1.0, 10.0],
                [1.0, 1.0, 3.0],
            ],
            dtype=np.float64,
        )
        cloud = PointCloud(xyz)
        result = create_dsm(cloud, resolution=5.0)
        # The cell containing (1,1) should have max Z = 10
        valid = result.data[result.data != result.nodata]
        assert valid.max() >= 10.0

    def test_empty_cloud_raises(self) -> None:
        """Empty cloud raises OcculusRasterError."""
        xyz = np.empty((0, 3), dtype=np.float64)
        with pytest.raises(OcculusRasterError, match="empty"):
            # PointCloud requires at least shape validation — work around
            cloud = PointCloud.__new__(PointCloud)
            cloud.xyz = xyz
            cloud.metadata = AcquisitionMetadata()
            cloud.intensity = None
            cloud.classification = None
            cloud.rgb = None
            cloud.normals = None
            cloud.return_number = None
            cloud.number_of_returns = None
            create_dsm(cloud, resolution=1.0)

    def test_invalid_resolution_raises(self, simple_cloud: PointCloud) -> None:
        """Non-positive resolution raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="resolution"):
            create_dsm(simple_cloud, resolution=-1.0)

    def test_invalid_method_raises(self, simple_cloud: PointCloud) -> None:
        """Unknown interpolation method raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="Unknown interpolation"):
            create_dsm(simple_cloud, resolution=1.0, method="cubic")

    def test_dsm_nearest_method(self, simple_cloud: PointCloud) -> None:
        """DSM with nearest interpolation succeeds."""
        result = create_dsm(simple_cloud, resolution=2.0, method="nearest")
        assert isinstance(result, RasterResult)

    def test_crs_propagated(self) -> None:
        """CRS from cloud metadata is propagated to result."""
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
        meta = AcquisitionMetadata(coordinate_system="EPSG:32617")
        cloud = PointCloud(xyz, metadata=meta)
        result = create_dsm(cloud, resolution=1.0)
        assert result.crs == "EPSG:32617"


class TestCreateDTM:
    """Tests for create_dtm."""

    def test_returns_raster_result(self, classified_cloud: PointCloud) -> None:
        """create_dtm returns a RasterResult from classified data."""
        result = create_dtm(classified_cloud, resolution=2.0)
        assert isinstance(result, RasterResult)
        assert result.data.ndim == 2

    def test_dtm_uses_ground_only(self, classified_cloud: PointCloud) -> None:
        """DTM values should be near the ground level (~100), not vegetation (~115)."""
        result = create_dtm(classified_cloud, resolution=5.0)
        valid = result.data[result.data != result.nodata]
        # Ground is at ~100, vegetation at ~115 — DTM should be close to 100
        assert valid.mean() < 110.0

    def test_no_classification_raises(self, simple_cloud: PointCloud) -> None:
        """Cloud without classification raises OcculusRasterError."""
        with pytest.raises(OcculusRasterError, match="classification"):
            create_dtm(simple_cloud, resolution=1.0)

    def test_no_ground_points_raises(self) -> None:
        """Cloud with classification but no ground class raises OcculusRasterError."""
        xyz = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float64)
        cls = np.array([1, 1], dtype=np.uint8)
        cloud = PointCloud(xyz, classification=cls)
        with pytest.raises(OcculusRasterError, match="No ground points"):
            create_dtm(cloud, resolution=1.0)

    def test_custom_ground_class(self) -> None:
        """DTM accepts a custom ground class code."""
        xyz = np.array(
            [
                [0.0, 0.0, 10.0],
                [5.0, 5.0, 12.0],
                [10.0, 10.0, 50.0],
            ],
            dtype=np.float64,
        )
        cls = np.array([8, 8, 1], dtype=np.uint8)
        cloud = PointCloud(xyz, classification=cls)
        result = create_dtm(cloud, resolution=5.0, ground_class=8)
        assert isinstance(result, RasterResult)

    def test_dtm_nearest_method(self, classified_cloud: PointCloud) -> None:
        """DTM with nearest interpolation succeeds."""
        result = create_dtm(classified_cloud, resolution=5.0, method="nearest")
        assert isinstance(result, RasterResult)


class TestCreateDEM:
    """Tests for create_dem (alias for create_dtm)."""

    def test_dem_is_dtm_alias(self, classified_cloud: PointCloud) -> None:
        """create_dem produces the same result as create_dtm."""
        dem = create_dem(classified_cloud, resolution=5.0)
        dtm = create_dtm(classified_cloud, resolution=5.0)
        np.testing.assert_array_equal(dem.data, dtm.data)
        np.testing.assert_array_equal(dem.x_edges, dtm.x_edges)
        np.testing.assert_array_equal(dem.y_edges, dtm.y_edges)


# ---------------------------------------------------------------------------
# RasterResult tests
# ---------------------------------------------------------------------------


class TestRasterResult:
    """Tests for the RasterResult dataclass."""

    def test_defaults(self) -> None:
        """Default nodata is -9999.0."""
        r = RasterResult(
            data=np.zeros((3, 3)),
            x_edges=np.arange(4, dtype=np.float64),
            y_edges=np.arange(4, dtype=np.float64),
            resolution=1.0,
            crs="",
        )
        assert r.nodata == -9999.0

    def test_custom_nodata(self) -> None:
        """Custom nodata value is preserved."""
        r = RasterResult(
            data=np.zeros((2, 2)),
            x_edges=np.arange(3, dtype=np.float64),
            y_edges=np.arange(3, dtype=np.float64),
            resolution=1.0,
            crs="EPSG:4326",
            nodata=-999.0,
        )
        assert r.nodata == -999.0
        assert r.crs == "EPSG:4326"


# ---------------------------------------------------------------------------
# GeoTIFF export tests
# ---------------------------------------------------------------------------


class TestExportGeotiff:
    """Tests for export_geotiff (rasterio is mocked)."""

    def _make_raster(self) -> RasterResult:
        """Create a small RasterResult for testing."""
        return RasterResult(
            data=np.arange(6, dtype=np.float64).reshape(2, 3),
            x_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            y_edges=np.array([0.0, 1.0, 2.0]),
            resolution=1.0,
            crs="EPSG:32617",
        )

    @patch("occulus.raster.export.rasterio", create=True)
    def test_export_calls_rasterio(self, mock_rasterio: MagicMock, tmp_path: Path) -> None:
        """export_geotiff opens rasterio and writes data."""
        # Patch the lazy import inside the function
        mock_dst = MagicMock()
        mock_rasterio.open.return_value.__enter__ = MagicMock(return_value=mock_dst)
        mock_rasterio.open.return_value.__exit__ = MagicMock(return_value=False)

        mock_from_bounds = MagicMock(return_value="mock_transform")

        raster = self._make_raster()
        out_file = tmp_path / "test.tif"

        with (
            patch.dict("sys.modules", {"rasterio": mock_rasterio}),
            patch.dict(
                "sys.modules",
                {"rasterio.transform": MagicMock(from_bounds=mock_from_bounds)},
            ),
        ):
            # Re-import to pick up the mocked module
            import importlib

            import occulus.raster.export

            importlib.reload(occulus.raster.export)

            result = occulus.raster.export.export_geotiff(raster, out_file)

        assert result == out_file.resolve()

    def test_import_error_without_rasterio(self, tmp_path: Path) -> None:
        """Raises ImportError with install hint when rasterio is missing."""
        raster = self._make_raster()
        out_file = tmp_path / "test.tif"

        with patch.dict("sys.modules", {"rasterio": None}):
            import importlib

            import occulus.raster.export

            importlib.reload(occulus.raster.export)

            with pytest.raises(ImportError, match="rasterio"):
                occulus.raster.export.export_geotiff(raster, out_file)

    def test_invalid_raster_shape(self, tmp_path: Path) -> None:
        """3D raster data raises OcculusRasterError."""
        raster = RasterResult(
            data=np.zeros((2, 3, 4)),  # type: ignore[arg-type]
            x_edges=np.arange(4, dtype=np.float64),
            y_edges=np.arange(3, dtype=np.float64),
            resolution=1.0,
            crs="",
        )
        # Need rasterio importable for the check to happen
        mock_rasterio = MagicMock()
        mock_transform = MagicMock()
        with (
            patch.dict("sys.modules", {"rasterio": mock_rasterio}),
            patch.dict("sys.modules", {"rasterio.transform": mock_transform}),
        ):
            import importlib

            import occulus.raster.export

            importlib.reload(occulus.raster.export)

            with pytest.raises(OcculusRasterError, match="2D"):
                occulus.raster.export.export_geotiff(raster, tmp_path / "bad.tif")

    def test_nonexistent_parent_raises(self, tmp_path: Path) -> None:
        """Writing to a path with nonexistent parent raises OcculusExportError."""
        raster = self._make_raster()
        bad_path = tmp_path / "nonexistent_dir" / "test.tif"

        mock_rasterio = MagicMock()
        mock_transform = MagicMock()
        mock_transform.from_bounds = MagicMock(return_value="mock_transform")
        with (
            patch.dict("sys.modules", {"rasterio": mock_rasterio}),
            patch.dict("sys.modules", {"rasterio.transform": mock_transform}),
        ):
            import importlib

            import occulus.raster.export

            importlib.reload(occulus.raster.export)

            with pytest.raises(OcculusExportError, match="Parent directory"):
                occulus.raster.export.export_geotiff(raster, bad_path)
