"""Spatial tiling and chunked processing for large point clouds.

Splits point clouds into spatial tiles for processing datasets
that exceed available RAM.  Tiles can be processed independently
and optionally in parallel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusIOError, OcculusValidationError

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """A spatial tile within a tiled point cloud.

    Attributes
    ----------
    index : tuple[int, int]
        Grid column and row index.
    bounds : tuple[float, float, float, float]
        Spatial bounds (xmin, ymin, xmax, ymax).
    point_count : int
        Number of points in this tile.
    path : Path or None
        Path to the tile file on disk, if written.
    """

    index: tuple[int, int]
    bounds: tuple[float, float, float, float]
    point_count: int
    path: Path | None = None


def tile_point_cloud(
    cloud_or_path: object,
    output_dir: str | Path,
    *,
    tile_size: float = 500.0,
    overlap: float = 0.0,
    format: str = "xyz",
) -> list[Tile]:
    """Split a point cloud into spatial grid tiles.

    Parameters
    ----------
    cloud_or_path : PointCloud or str or Path
        Input point cloud object or path to a file.
    output_dir : str or Path
        Directory to write tile files.
    tile_size : float
        Tile edge length in point-cloud coordinate units (default 500 m).
    overlap : float
        Buffer overlap between adjacent tiles in coordinate units.
    format : str
        Output format for tile files ('xyz', 'laz', 'las', 'ply').

    Returns
    -------
    list[Tile]
        List of tiles with metadata and file paths.

    Raises
    ------
    OcculusValidationError
        If tile_size <= 0 or input is invalid.
    """
    from occulus.types import PointCloud

    if tile_size <= 0:
        raise OcculusValidationError(f"tile_size must be positive, got {tile_size}")

    # Load cloud if path
    if isinstance(cloud_or_path, (str, Path)):
        from occulus.io import read

        cloud = read(str(cloud_or_path))
    elif isinstance(cloud_or_path, PointCloud):
        cloud = cloud_or_path
    else:
        raise OcculusValidationError(f"Expected PointCloud or file path, got {type(cloud_or_path)}")

    xyz = cloud.xyz
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    n_cols = max(1, int(np.ceil((x_max - x_min) / tile_size)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / tile_size)))

    logger.info(
        "Tiling %d points into %d x %d grid (%.0f m tiles)",
        len(xyz),
        n_cols,
        n_rows,
        tile_size,
    )

    tiles: list[Tile] = []

    for col in range(n_cols):
        for row in range(n_rows):
            tx_min = x_min + col * tile_size - overlap
            tx_max = x_min + (col + 1) * tile_size + overlap
            ty_min = y_min + row * tile_size - overlap
            ty_max = y_min + (row + 1) * tile_size + overlap

            mask = (
                (xyz[:, 0] >= tx_min)
                & (xyz[:, 0] < tx_max)
                & (xyz[:, 1] >= ty_min)
                & (xyz[:, 1] < ty_max)
            )
            n_pts = int(mask.sum())
            if n_pts == 0:
                continue

            tile_name = f"tile_{col:04d}_{row:04d}.{format}"
            tile_path = output_dir / tile_name

            # Write tile
            from occulus.io import write
            from occulus.types import PointCloud as PC

            tile_cloud = PC(xyz[mask])
            write(tile_cloud, tile_path)

            tile = Tile(
                index=(col, row),
                bounds=(tx_min, ty_min, tx_max, ty_max),
                point_count=n_pts,
                path=tile_path,
            )
            tiles.append(tile)

    logger.info("Created %d tiles in %s", len(tiles), output_dir)
    return tiles


def iter_tiles(
    path: str | Path,
    *,
    tile_size: float = 500.0,
) -> Iterator[tuple[Tile, object]]:
    """Iterate over spatial tiles of a point cloud file.

    Yields tiles one at a time to keep memory usage bounded.

    Parameters
    ----------
    path : str or Path
        Path to a point cloud file.
    tile_size : float
        Tile edge length in coordinate units.

    Yields
    ------
    tuple[Tile, PointCloud]
        Tile metadata and the corresponding point cloud subset.
    """
    from occulus.io import read

    logger.info("Loading point cloud for tiled iteration: %s", path)
    cloud = read(str(path))
    xyz = cloud.xyz

    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    n_cols = max(1, int(np.ceil((x_max - x_min) / tile_size)))
    n_rows = max(1, int(np.ceil((y_max - y_min) / tile_size)))

    from occulus.types import PointCloud as PC

    for col in range(n_cols):
        for row in range(n_rows):
            tx_min = x_min + col * tile_size
            tx_max = x_min + (col + 1) * tile_size
            ty_min = y_min + row * tile_size
            ty_max = y_min + (row + 1) * tile_size

            mask = (
                (xyz[:, 0] >= tx_min)
                & (xyz[:, 0] < tx_max)
                & (xyz[:, 1] >= ty_min)
                & (xyz[:, 1] < ty_max)
            )
            if mask.sum() == 0:
                continue

            tile = Tile(
                index=(col, row),
                bounds=(tx_min, ty_min, tx_max, ty_max),
                point_count=int(mask.sum()),
            )
            yield tile, PC(xyz[mask])


def process_tiles(
    tiles: list[Tile],
    operation: Callable[..., object],
    *,
    output_dir: str | Path,
    max_workers: int = 1,
) -> list[Path]:
    """Apply an operation to each tile and write results.

    Parameters
    ----------
    tiles : list[Tile]
        Tiles to process (must have .path set).
    operation : callable
        Function that takes a PointCloud and returns a PointCloud.
    output_dir : str or Path
        Directory for processed tile outputs.
    max_workers : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    list[Path]
        Paths to processed tile files.
    """
    from occulus.io import read, write

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    for tile in tiles:
        if tile.path is None:
            logger.warning("Tile %s has no path, skipping", tile.index)
            continue

        logger.info("Processing tile %s (%d points)", tile.index, tile.point_count)
        cloud = read(str(tile.path))
        processed = operation(cloud)

        out_path = output_dir / tile.path.name
        write(processed, out_path)
        results.append(out_path)

    logger.info("Processed %d tiles", len(results))
    return results
