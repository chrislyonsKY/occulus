"""Shared pytest fixtures for Occulus tests.

Fixtures defined here are automatically available to all test modules.

Fixture Inventory
-----------------
sample_bbox
    A valid EPSG:4326 bounding box covering eastern Kentucky.
sample_point_cloud
    A synthetic PointCloud with 1000 random points for unit tests.
sample_aerial_cloud
    A synthetic AerialCloud with ground/canopy elevation structure.
tmp_output_dir
    A temporary directory for file output tests.

"""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_bbox() -> tuple[float, float, float, float]:
    """A small EPSG:4326 bounding box covering eastern Kentucky.

    Returns
    -------
    tuple of float
        (west, south, east, north) in decimal degrees.

    """
    return (-84.5, 37.8, -84.3, 38.0)


@pytest.fixture
def sample_point_cloud():
    """A synthetic PointCloud with 1000 random points.

    Returns
    -------
    PointCloud
        Random points in a 100×100×50 m volume.

    """
    from occulus.types import PointCloud

    rng = np.random.default_rng(42)
    xyz = rng.uniform(low=[0, 0, 0], high=[100, 100, 50], size=(1000, 3))
    return PointCloud(xyz)


@pytest.fixture
def sample_aerial_cloud():
    """A synthetic AerialCloud with ground and canopy structure.

    Ground points are at z=100±2 m, canopy points at z=120±10 m.

    Returns
    -------
    AerialCloud
        500 ground + 500 canopy points with classification array.

    """
    from occulus.types import AerialCloud

    rng = np.random.default_rng(42)
    n = 500
    xy = rng.uniform(low=[0, 0], high=[100, 100], size=(2 * n, 2))
    z_ground = 100 + rng.normal(0, 2, n)
    z_canopy = 120 + rng.normal(0, 10, n)
    z = np.concatenate([z_ground, z_canopy])
    xyz = np.column_stack([xy, z])
    classification = np.concatenate([np.full(n, 2), np.full(n, 1)])
    return AerialCloud(xyz, classification=classification)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for file output tests.

    Returns
    -------
    Path
        A temporary directory that is cleaned up after the test.

    """
    out = tmp_path / "output"
    out.mkdir()
    return out
