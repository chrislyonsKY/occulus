"""Tests for occulus.io — readers and writers."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusIOError, OcculusValidationError
from occulus.io.readers import _make_cloud, _subsample_mask, read
from occulus.io.writers import write
from occulus.types import (
    AerialCloud,
    Platform,
    PointCloud,
    TerrestrialCloud,
    UAVCloud,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cloud() -> PointCloud:
    """Small synthetic point cloud for writer tests."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((100, 3)).astype(np.float64)
    intensity = rng.random(100).astype(np.float64)
    return PointCloud(xyz, intensity=intensity)


# ---------------------------------------------------------------------------
# Reader validation
# ---------------------------------------------------------------------------


class TestRead:
    """Tests for the read() dispatcher."""

    def test_file_not_found_raises(self, tmp_path):
        """read() raises OcculusIOError for missing files."""
        with pytest.raises(OcculusIOError, match="File not found"):
            read(tmp_path / "nonexistent.las")

    def test_unsupported_extension_raises(self, tmp_path):
        """read() raises OcculusValidationError for unknown extensions."""
        bad_file = tmp_path / "data.docx"
        bad_file.touch()
        with pytest.raises(OcculusValidationError, match="Unsupported format"):
            read(bad_file)

    def test_invalid_subsample_raises(self, tmp_path):
        """read() raises OcculusValidationError for out-of-range subsample."""
        f = tmp_path / "x.xyz"
        f.write_text("1.0 2.0 3.0\n")
        with pytest.raises(OcculusValidationError, match="subsample"):
            read(f, subsample=0.0)

    def test_subsample_above_one_raises(self, tmp_path):
        """read() raises OcculusValidationError for subsample > 1."""
        f = tmp_path / "x.xyz"
        f.write_text("1.0 2.0 3.0\n")
        with pytest.raises(OcculusValidationError, match="subsample"):
            read(f, subsample=1.5)


# ---------------------------------------------------------------------------
# XYZ reader (pure NumPy — no optional deps)
# ---------------------------------------------------------------------------


class TestReadXYZ:
    """Tests for _read_xyz via the read() dispatcher."""

    def test_reads_whitespace_delimited(self, tmp_path):
        """read() parses whitespace-delimited XYZ files."""
        f = tmp_path / "points.xyz"
        f.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0\n")
        cloud = read(f)
        assert cloud.n_points == 3
        assert cloud.xyz.shape == (3, 3)
        assert cloud.xyz[0, 0] == pytest.approx(1.0)

    def test_reads_csv(self, tmp_path):
        """read() parses comma-delimited CSV files."""
        f = tmp_path / "points.csv"
        f.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
        cloud = read(f)
        assert cloud.n_points == 2

    def test_reads_intensity_column(self, tmp_path):
        """XYZ reader picks up fourth column as intensity."""
        f = tmp_path / "points.xyz"
        f.write_text("1.0 2.0 3.0 100.0\n4.0 5.0 6.0 200.0\n")
        cloud = read(f)
        assert cloud.intensity is not None
        assert cloud.intensity[0] == pytest.approx(100.0)

    def test_skips_comment_lines(self, tmp_path):
        """XYZ reader ignores lines starting with #."""
        f = tmp_path / "points.xyz"
        f.write_text("# header\n1.0 2.0 3.0\n4.0 5.0 6.0\n")
        cloud = read(f)
        assert cloud.n_points == 2

    def test_txt_extension(self, tmp_path):
        """read() handles .txt extension as whitespace-delimited XYZ."""
        f = tmp_path / "points.txt"
        f.write_text("1.0 2.0 3.0\n")
        cloud = read(f)
        assert cloud.n_points == 1

    def test_platform_aerial_returns_aerial_cloud(self, tmp_path):
        """read() with platform=aerial returns AerialCloud."""
        f = tmp_path / "points.xyz"
        f.write_text("1.0 2.0 3.0\n")
        cloud = read(f, platform="aerial")
        assert isinstance(cloud, AerialCloud)
        assert cloud.platform == Platform.AERIAL

    def test_platform_terrestrial_returns_terrestrial_cloud(self, tmp_path):
        """read() with platform=terrestrial returns TerrestrialCloud."""
        f = tmp_path / "points.xyz"
        f.write_text("1.0 2.0 3.0\n")
        cloud = read(f, platform=Platform.TERRESTRIAL)
        assert isinstance(cloud, TerrestrialCloud)

    def test_platform_uav_returns_uav_cloud(self, tmp_path):
        """read() with platform=uav returns UAVCloud."""
        f = tmp_path / "points.xyz"
        f.write_text("1.0 2.0 3.0\n")
        cloud = read(f, platform="uav")
        assert isinstance(cloud, UAVCloud)

    def test_subsample_reduces_points(self, tmp_path):
        """read() with subsample returns fewer points."""
        f = tmp_path / "points.xyz"
        lines = "\n".join(f"{i}.0 {i}.0 {i}.0" for i in range(1000))
        f.write_text(lines + "\n")
        cloud = read(f, subsample=0.1)
        assert cloud.n_points < 1000
        assert cloud.n_points >= 1


# ---------------------------------------------------------------------------
# XYZ writer (pure NumPy — no optional deps)
# ---------------------------------------------------------------------------


class TestWriteXYZ:
    """Tests for _write_xyz via the write() dispatcher."""

    def test_writes_xyz_file(self, tmp_path, sample_cloud):
        """write() produces a file at the given path."""
        out = tmp_path / "out.xyz"
        result = write(sample_cloud, out)
        assert result == out
        assert out.exists()

    def test_round_trip_xyz(self, tmp_path):
        """XYZ write then read preserves point count and coordinates."""
        rng = np.random.default_rng(7)
        xyz = rng.standard_normal((50, 3))
        cloud = PointCloud(xyz)
        out = tmp_path / "round_trip.xyz"
        write(cloud, out)
        loaded = read(out)
        assert loaded.n_points == 50
        np.testing.assert_allclose(loaded.xyz, xyz, atol=1e-5)

    def test_writes_csv_file(self, tmp_path, sample_cloud):
        """write() handles .csv extension with comma delimiter."""
        out = tmp_path / "out.csv"
        write(sample_cloud, out)
        content = out.read_text()
        assert "," in content

    def test_unsupported_extension_raises(self, tmp_path, sample_cloud):
        """write() raises OcculusValidationError for unsupported extension."""
        with pytest.raises(OcculusValidationError, match="Unsupported"):
            write(sample_cloud, tmp_path / "out.docx")

    def test_round_trip_with_intensity(self, tmp_path):
        """Round trip preserves intensity column."""
        rng = np.random.default_rng(11)
        xyz = rng.standard_normal((20, 3))
        intensity = rng.random(20)
        cloud = PointCloud(xyz, intensity=intensity)
        out = tmp_path / "with_intensity.xyz"
        write(cloud, out)
        loaded = read(out)
        assert loaded.intensity is not None
        np.testing.assert_allclose(loaded.intensity, intensity, atol=1e-5)


# ---------------------------------------------------------------------------
# _make_cloud helper
# ---------------------------------------------------------------------------


class TestMakeCloud:
    """Tests for the _make_cloud() internal helper."""

    def test_unknown_platform_returns_base(self):
        """_make_cloud with unknown platform returns base PointCloud."""
        xyz = np.ones((10, 3))
        cloud = _make_cloud(xyz, Platform.UNKNOWN)
        assert type(cloud) is PointCloud

    def test_string_platform_accepted(self):
        """_make_cloud accepts string platform values."""
        xyz = np.ones((10, 3))
        cloud = _make_cloud(xyz, "aerial")
        assert isinstance(cloud, AerialCloud)

    def test_invalid_string_platform_falls_back(self):
        """_make_cloud with unrecognised string falls back to base PointCloud."""
        xyz = np.ones((10, 3))
        cloud = _make_cloud(xyz, "drone_v2")  # not a valid Platform value
        assert type(cloud) is PointCloud


# ---------------------------------------------------------------------------
# _subsample_mask helper
# ---------------------------------------------------------------------------


class TestSubsampleMask:
    """Tests for the _subsample_mask() helper."""

    def test_returns_correct_length(self):
        """_subsample_mask returns a mask of length n."""
        mask = _subsample_mask(100, 0.5)
        assert len(mask) == 100

    def test_fraction_respected(self):
        """_subsample_mask selects approximately fraction*n points."""
        mask = _subsample_mask(1000, 0.2)
        assert 150 <= mask.sum() <= 250  # allow ±25 for randomness

    def test_full_fraction(self):
        """_subsample_mask with fraction=1.0 keeps all points."""
        mask = _subsample_mask(50, 1.0)
        assert mask.sum() == 50

    def test_minimum_one_point(self):
        """_subsample_mask always returns at least one point."""
        mask = _subsample_mask(100, 0.001)
        assert mask.sum() >= 1
