"""Tests for occulus.config — constants and environment variable loading."""

from __future__ import annotations

import os

import pytest


class TestConfigDefaults:
    """Tests for config module default values."""

    def test_default_num_threads_positive(self) -> None:
        """DEFAULT_NUM_THREADS is a positive integer."""
        from occulus.config import DEFAULT_NUM_THREADS
        assert isinstance(DEFAULT_NUM_THREADS, int)
        assert DEFAULT_NUM_THREADS > 0

    def test_default_log_level_is_string(self) -> None:
        """DEFAULT_LOG_LEVEL is a string."""
        from occulus.config import DEFAULT_LOG_LEVEL
        assert isinstance(DEFAULT_LOG_LEVEL, str)

    def test_icp_defaults_positive(self) -> None:
        """ICP defaults are positive numbers."""
        from occulus.config import (
            DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE,
            DEFAULT_ICP_MAX_ITERATIONS,
            DEFAULT_ICP_TOLERANCE,
        )
        assert DEFAULT_ICP_MAX_ITERATIONS > 0
        assert DEFAULT_ICP_TOLERANCE > 0
        assert DEFAULT_ICP_MAX_CORRESPONDENCE_DISTANCE > 0

    def test_segmentation_defaults_positive(self) -> None:
        """Segmentation defaults are positive."""
        from occulus.config import (
            DEFAULT_GROUND_CLOTH_RESOLUTION,
            DEFAULT_GROUND_MAX_ANGLE,
            DEFAULT_MIN_SEGMENT_POINTS,
        )
        assert DEFAULT_GROUND_CLOTH_RESOLUTION > 0
        assert DEFAULT_GROUND_MAX_ANGLE > 0
        assert DEFAULT_MIN_SEGMENT_POINTS > 0

    def test_mesh_defaults_positive(self) -> None:
        """Mesh defaults are positive."""
        from occulus.config import DEFAULT_BPA_RADII_FACTOR, DEFAULT_POISSON_DEPTH
        assert DEFAULT_POISSON_DEPTH > 0
        assert DEFAULT_BPA_RADII_FACTOR > 0

    def test_filter_defaults_positive(self) -> None:
        """Filter defaults are positive."""
        from occulus.config import (
            DEFAULT_SOR_K_NEIGHBORS,
            DEFAULT_SOR_STD_RATIO,
            DEFAULT_VOXEL_SIZE,
        )
        assert DEFAULT_SOR_K_NEIGHBORS > 0
        assert DEFAULT_SOR_STD_RATIO > 0
        assert DEFAULT_VOXEL_SIZE > 0

    def test_num_threads_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NUM_THREADS can be overridden via environment variable."""
        monkeypatch.setenv("OCCULUS_NUM_THREADS", "8")
        import importlib
        import occulus.config as cfg
        importlib.reload(cfg)
        assert cfg.NUM_THREADS == 8

    def test_log_level_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LOG_LEVEL can be overridden via environment variable."""
        monkeypatch.setenv("OCCULUS_LOG_LEVEL", "DEBUG")
        import importlib
        import occulus.config as cfg
        importlib.reload(cfg)
        assert cfg.LOG_LEVEL == "DEBUG"
