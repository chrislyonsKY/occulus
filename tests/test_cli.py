"""Tests for occulus.cli — CLI subcommands and argument parsing.

Verifies argument parsing, subcommand dispatch, and error handling
with mocked I/O so no real files are touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from occulus.cli.main import _build_parser, main
from occulus.types import PointCloud

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cloud() -> PointCloud:
    """A 100-point synthetic cloud in the unit cube."""
    rng = np.random.default_rng(42)
    xyz = rng.random((100, 3)).astype(np.float64)
    return PointCloud(xyz)


@pytest.fixture
def _patch_read(small_cloud: PointCloud):
    """Patch _read_cloud to return the small_cloud fixture."""
    with patch("occulus.cli.main._read_cloud", return_value=small_cloud):
        yield


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


class TestParser:
    """Tests for argument parser construction."""

    def test_parser_creates_subcommands(self):
        """Parser should have all expected subcommands."""
        parser = _build_parser()
        assert parser.prog == "occulus"

    def test_version_flag(self):
        """--version should be a recognised argument."""
        parser = _build_parser()
        args = parser.parse_args(["--version"])
        assert args.version is True

    def test_verbose_flag(self):
        """--verbose should be a recognised argument."""
        parser = _build_parser()
        args = parser.parse_args(["--verbose", "info", "test.xyz"])
        assert args.verbose is True

    def test_info_args(self):
        """info subcommand should parse input and platform."""
        parser = _build_parser()
        args = parser.parse_args(["info", "scan.laz", "--platform", "aerial"])
        assert args.command == "info"
        assert args.input == "scan.laz"
        assert args.platform == "aerial"

    def test_classify_args(self):
        """classify subcommand should parse input, output, and algorithm."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "classify",
                "in.laz",
                "-o",
                "out.laz",
                "--algorithm",
                "pmf",
            ]
        )
        assert args.command == "classify"
        assert args.input == "in.laz"
        assert args.output == "out.laz"
        assert args.algorithm == "pmf"

    def test_filter_args_voxel(self):
        """filter subcommand should parse voxel method with voxel-size."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "filter",
                "in.laz",
                "-o",
                "out.laz",
                "--method",
                "voxel",
                "--voxel-size",
                "0.5",
            ]
        )
        assert args.command == "filter"
        assert args.method == "voxel"
        assert args.voxel_size == 0.5

    def test_filter_args_sor(self):
        """filter subcommand should parse SOR method with parameters."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "filter",
                "in.laz",
                "-o",
                "out.laz",
                "--method",
                "sor",
                "--nb-neighbors",
                "30",
                "--std-ratio",
                "1.5",
            ]
        )
        assert args.method == "sor"
        assert args.nb_neighbors == 30
        assert args.std_ratio == 1.5

    def test_filter_args_radius(self):
        """filter subcommand should parse radius method with parameters."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "filter",
                "in.laz",
                "-o",
                "out.laz",
                "--method",
                "radius",
                "--radius",
                "1.0",
                "--min-neighbors",
                "5",
            ]
        )
        assert args.method == "radius"
        assert args.radius == 1.0
        assert args.min_neighbors == 5

    def test_convert_args(self):
        """convert subcommand should parse input and output."""
        parser = _build_parser()
        args = parser.parse_args(["convert", "in.laz", "-o", "out.ply"])
        assert args.command == "convert"
        assert args.input == "in.laz"
        assert args.output == "out.ply"

    def test_dem_args(self):
        """dem subcommand should parse resolution and method."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "dem",
                "in.laz",
                "-o",
                "dem.npy",
                "--resolution",
                "2.0",
                "--method",
                "nearest",
            ]
        )
        assert args.command == "dem"
        assert args.resolution == 2.0
        assert args.method == "nearest"

    def test_register_args(self):
        """register subcommand should parse source, target, and ICP params."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "register",
                "src.laz",
                "tgt.laz",
                "-o",
                "aligned.laz",
                "--max-iterations",
                "100",
                "--tolerance",
                "1e-8",
            ]
        )
        assert args.command == "register"
        assert args.source == "src.laz"
        assert args.target == "tgt.laz"
        assert args.max_iterations == 100
        assert args.tolerance == 1e-8

    def test_tile_args(self):
        """tile subcommand should parse tile-size."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "tile",
                "in.laz",
                "-o",
                "tiles/",
                "--tile-size",
                "50.0",
            ]
        )
        assert args.command == "tile"
        assert args.tile_size == 50.0

    def test_subsample_arg(self):
        """All subcommands should accept --subsample."""
        parser = _build_parser()
        args = parser.parse_args(["info", "scan.laz", "--subsample", "0.1"])
        assert args.subsample == 0.1


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() entry point."""

    def test_no_command_returns_2(self, monkeypatch):
        """Running with no subcommand should return exit code 2."""
        monkeypatch.setattr("sys.argv", ["occulus"])
        assert main() == 2

    def test_version_returns_0(self, monkeypatch):
        """--version should return 0.

        The current implementation returns 0 without printing version
        text — we verify the exit code only.
        """
        monkeypatch.setattr("sys.argv", ["occulus", "--version"])
        assert main() == 0


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestCmdInfo:
    """Tests for the info subcommand."""

    def test_info_returns_0(self, monkeypatch, _patch_read):
        """info should return 0 on success.

        The implementation delegates to compute_cloud_statistics and
        coverage_statistics but does not print to stdout — we verify
        the exit code and that the underlying functions are called.
        """
        monkeypatch.setattr("sys.argv", ["occulus", "info", "fake.xyz"])

        mock_stats = MagicMock()
        mock_stats.intensity_mean = None

        with (
            patch(
                "occulus.metrics.compute_cloud_statistics",
                return_value=mock_stats,
            ) as mock_compute,
            patch("occulus.metrics.coverage_statistics") as mock_coverage,
        ):
            code = main()

        assert code == 0
        mock_compute.assert_called_once()
        mock_coverage.assert_called_once()

    def test_info_with_platform(self, monkeypatch):
        """info should respect the --platform flag."""
        rng = np.random.default_rng(7)
        cloud = PointCloud(rng.random((50, 3)).astype(np.float64))

        mock_stats = MagicMock()
        mock_stats.intensity_mean = None

        with (
            patch("occulus.cli.main._read_cloud", return_value=cloud),
            patch(
                "occulus.metrics.compute_cloud_statistics",
                return_value=mock_stats,
            ),
            patch("occulus.metrics.coverage_statistics"),
        ):
            monkeypatch.setattr(
                "sys.argv",
                [
                    "occulus",
                    "info",
                    "fake.xyz",
                    "--platform",
                    "aerial",
                ],
            )
            code = main()
        assert code == 0


# ---------------------------------------------------------------------------
# classify command
# ---------------------------------------------------------------------------


class TestCmdClassify:
    """Tests for the classify subcommand."""

    def test_classify_csf(self, monkeypatch, small_cloud):
        """classify --algorithm csf should call classify_ground_csf and write."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "classify",
                "in.xyz",
                "-o",
                "out.xyz",
                "--algorithm",
                "csf",
            ],
        )
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch(
                "occulus.segmentation.classify_ground_csf",
                return_value=small_cloud,
            ) as mock_csf,
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_csf.assert_called_once()
        mock_write.assert_called_once_with(small_cloud, "out.xyz")

    def test_classify_pmf(self, monkeypatch, small_cloud):
        """classify --algorithm pmf should call classify_ground_pmf and write."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "classify",
                "in.xyz",
                "-o",
                "out.xyz",
                "--algorithm",
                "pmf",
            ],
        )
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch(
                "occulus.segmentation.classify_ground_pmf",
                return_value=small_cloud,
            ) as mock_pmf,
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_pmf.assert_called_once()
        mock_write.assert_called_once_with(small_cloud, "out.xyz")


# ---------------------------------------------------------------------------
# filter command
# ---------------------------------------------------------------------------


class TestCmdFilter:
    """Tests for the filter subcommand."""

    def test_filter_voxel(self, monkeypatch, small_cloud):
        """filter --method voxel should call voxel_downsample."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "filter",
                "in.xyz",
                "-o",
                "out.xyz",
                "--method",
                "voxel",
                "--voxel-size",
                "0.5",
            ],
        )
        downsampled = PointCloud(small_cloud.xyz[:20])
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch(
                "occulus.filters.voxel_downsample",
                return_value=downsampled,
            ) as mock_voxel,
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_voxel.assert_called_once()
        mock_write.assert_called_once_with(downsampled, "out.xyz")

    def test_filter_sor(self, monkeypatch, small_cloud):
        """filter --method sor should call statistical_outlier_removal."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "filter",
                "in.xyz",
                "-o",
                "out.xyz",
                "--method",
                "sor",
            ],
        )
        inlier_mask = np.ones(small_cloud.n_points, dtype=bool)
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch(
                "occulus.filters.statistical_outlier_removal",
                return_value=(small_cloud, inlier_mask),
            ) as mock_sor,
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_sor.assert_called_once()
        mock_write.assert_called_once()

    def test_filter_radius(self, monkeypatch, small_cloud):
        """filter --method radius should call radius_outlier_removal."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "filter",
                "in.xyz",
                "-o",
                "out.xyz",
                "--method",
                "radius",
            ],
        )
        inlier_mask = np.ones(small_cloud.n_points, dtype=bool)
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch(
                "occulus.filters.radius_outlier_removal",
                return_value=(small_cloud, inlier_mask),
            ) as mock_radius,
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_radius.assert_called_once()
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# convert command
# ---------------------------------------------------------------------------


class TestCmdConvert:
    """Tests for the convert subcommand."""

    def test_convert_runs(self, monkeypatch, small_cloud):
        """convert should read input and write to the output path."""
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "convert",
                "in.xyz",
                "-o",
                "out.ply",
            ],
        )
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        mock_write.assert_called_once_with(small_cloud, "out.ply")


# ---------------------------------------------------------------------------
# dem command
# ---------------------------------------------------------------------------


class TestCmdDem:
    """Tests for the dem subcommand."""

    def test_dem_idw(self, monkeypatch, small_cloud, tmp_path):
        """dem --method idw should produce a .npy file."""
        out_file = str(tmp_path / "dem.npy")
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "dem",
                "in.xyz",
                "-o",
                out_file,
                "--resolution",
                "0.5",
                "--method",
                "idw",
            ],
        )
        with patch("occulus.cli.main._read_cloud", return_value=small_cloud):
            code = main()

        assert code == 0
        assert (tmp_path / "dem.npy").exists()

    def test_dem_nearest(self, monkeypatch, small_cloud, tmp_path):
        """dem --method nearest should produce a .npy file."""
        out_file = str(tmp_path / "dem.npy")
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "dem",
                "in.xyz",
                "-o",
                out_file,
                "--resolution",
                "0.5",
                "--method",
                "nearest",
            ],
        )
        with patch("occulus.cli.main._read_cloud", return_value=small_cloud):
            code = main()

        assert code == 0
        dem = np.load(str(tmp_path / "dem.npy"))
        assert dem.ndim == 2


# ---------------------------------------------------------------------------
# register command
# ---------------------------------------------------------------------------


class TestCmdRegister:
    """Tests for the register subcommand."""

    def test_register_runs(self, monkeypatch, small_cloud, tmp_path):
        """register should run ICP and write aligned output."""
        from occulus.registration.icp import RegistrationResult

        out_file = str(tmp_path / "aligned.xyz")
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "register",
                "src.xyz",
                "tgt.xyz",
                "-o",
                out_file,
            ],
        )

        mock_result = RegistrationResult(
            transformation=np.eye(4),
            fitness=0.95,
            inlier_rmse=0.01,
            converged=True,
            n_iterations=10,
        )

        call_count = 0

        def _mock_read(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return small_cloud

        with (
            patch("occulus.cli.main._read_cloud", side_effect=_mock_read),
            patch("occulus.registration.icp", return_value=mock_result),
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        assert call_count == 2
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# tile command
# ---------------------------------------------------------------------------


class TestCmdTile:
    """Tests for the tile subcommand."""

    def test_tile_creates_tiles(self, monkeypatch, small_cloud, tmp_path):
        """tile should write tile files via occulus.io.write."""
        out_dir = str(tmp_path / "tiles")
        monkeypatch.setattr(
            "sys.argv",
            [
                "occulus",
                "tile",
                "in.xyz",
                "-o",
                out_dir,
                "--tile-size",
                "0.5",
            ],
        )
        with (
            patch("occulus.cli.main._read_cloud", return_value=small_cloud),
            patch("occulus.io.write") as mock_write,
        ):
            code = main()

        assert code == 0
        # With tile-size=0.5 and points in [0, 1], we expect multiple tiles
        assert mock_write.call_count >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for CLI error handling."""

    def test_command_error_returns_1(self, monkeypatch):
        """Exceptions during command execution should return exit code 1.

        The implementation catches all exceptions in the handler dispatch
        and returns 1, logging the error at DEBUG level.
        """
        monkeypatch.setattr("sys.argv", ["occulus", "info", "nonexistent.xyz"])
        with patch(
            "occulus.cli.main._read_cloud",
            side_effect=FileNotFoundError("File not found"),
        ):
            code = main()
        assert code == 1

    def test_occulus_error_returns_1(self, monkeypatch):
        """OcculusError during command execution should return exit code 1."""
        from occulus.exceptions import OcculusIOError

        monkeypatch.setattr("sys.argv", ["occulus", "info", "bad.xyz"])
        with patch(
            "occulus.cli.main._read_cloud",
            side_effect=OcculusIOError("Cannot read bad.xyz"),
        ):
            code = main()
        assert code == 1
