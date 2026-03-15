"""Argparse-based CLI for Occulus point cloud analysis.

Subcommands
-----------
- ``info``      — print cloud statistics (point count, bounds, density)
- ``classify``  — ground classification (CSF or PMF)
- ``filter``    — spatial filtering (voxel, SOR, radius)
- ``convert``   — format conversion (LAS/LAZ/PLY/XYZ)
- ``dem``       — DEM rasterization (IDW or nearest)
- ``register``  — ICP registration of source to target
- ``tile``      — tile into spatial chunks
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="occulus",
        description="Occulus — Multi-platform point cloud analysis CLI",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable verbose (DEBUG) logging",
    )

    sub = parser.add_subparsers(dest="command")

    # -- info --
    p_info = sub.add_parser("info", help="print cloud statistics")
    p_info.add_argument("input", help="input point cloud file")
    p_info.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_info.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0] for large files",
    )

    # -- classify --
    p_classify = sub.add_parser("classify", help="ground classification")
    p_classify.add_argument("input", help="input point cloud file")
    p_classify.add_argument("-o", "--output", required=True, help="output file path")
    p_classify.add_argument(
        "--algorithm",
        choices=["csf", "pmf"],
        default="csf",
        help="classification algorithm (default: csf)",
    )
    p_classify.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_classify.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    # -- filter --
    p_filter = sub.add_parser("filter", help="point cloud filtering")
    p_filter.add_argument("input", help="input point cloud file")
    p_filter.add_argument("-o", "--output", required=True, help="output file path")
    p_filter.add_argument(
        "--method",
        choices=["voxel", "sor", "radius"],
        default="voxel",
        help="filtering method (default: voxel)",
    )
    p_filter.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="voxel edge length for voxel downsampling (default: 0.1)",
    )
    p_filter.add_argument(
        "--nb-neighbors",
        type=int,
        default=20,
        help="number of neighbours for SOR (default: 20)",
    )
    p_filter.add_argument(
        "--std-ratio",
        type=float,
        default=2.0,
        help="standard deviation ratio for SOR (default: 2.0)",
    )
    p_filter.add_argument(
        "--radius",
        type=float,
        default=0.5,
        help="search radius for radius outlier removal (default: 0.5)",
    )
    p_filter.add_argument(
        "--min-neighbors",
        type=int,
        default=2,
        help="minimum neighbours for radius outlier removal (default: 2)",
    )
    p_filter.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_filter.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    # -- convert --
    p_convert = sub.add_parser("convert", help="format conversion")
    p_convert.add_argument("input", help="input point cloud file")
    p_convert.add_argument("-o", "--output", required=True, help="output file path")
    p_convert.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_convert.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    # -- dem --
    p_dem = sub.add_parser("dem", help="DEM rasterization")
    p_dem.add_argument("input", help="input point cloud file")
    p_dem.add_argument("-o", "--output", required=True, help="output .npy raster file")
    p_dem.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="raster cell size in coordinate units (default: 1.0)",
    )
    p_dem.add_argument(
        "--method",
        choices=["idw", "nearest"],
        default="idw",
        help="interpolation method (default: idw)",
    )
    p_dem.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_dem.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    # -- register --
    p_register = sub.add_parser("register", help="ICP registration")
    p_register.add_argument("source", help="source point cloud file")
    p_register.add_argument("target", help="target point cloud file")
    p_register.add_argument("-o", "--output", required=True, help="output file for aligned source")
    p_register.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="maximum ICP iterations (default: 50)",
    )
    p_register.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="convergence tolerance (default: 1e-6)",
    )
    p_register.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_register.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    # -- tile --
    p_tile = sub.add_parser("tile", help="tile into spatial chunks")
    p_tile.add_argument("input", help="input point cloud file")
    p_tile.add_argument("-o", "--output", required=True, help="output directory for tiles")
    p_tile.add_argument(
        "--tile-size",
        type=float,
        default=100.0,
        help="tile edge length in coordinate units (default: 100.0)",
    )
    p_tile.add_argument(
        "--platform",
        choices=["aerial", "terrestrial", "uav", "unknown"],
        default="unknown",
        help="acquisition platform (default: unknown)",
    )
    p_tile.add_argument(
        "--subsample",
        type=float,
        default=None,
        help="random subsample fraction (0.0–1.0]",
    )

    return parser


def _setup_logging(verbose: bool) -> None:
    """Configure root logger based on verbosity flag.

    Parameters
    ----------
    verbose : bool
        If ``True``, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _read_cloud(
    path: str,
    *,
    platform: str = "unknown",
    subsample: float | None = None,
) -> PointCloud:
    """Read a point cloud, lazily importing occulus.io.

    Parameters
    ----------
    path : str
        Path to the input file.
    platform : str
        Acquisition platform string.
    subsample : float | None
        Optional subsample fraction.

    Returns
    -------
    PointCloud
        The loaded point cloud.

    Raises
    ------
    SystemExit
        Re-raised as a CLI error if reading fails.
    """
    from occulus.io import read

    return read(path, platform=platform, subsample=subsample)


def _cmd_info(args: argparse.Namespace) -> int:
    """Print point cloud statistics to stdout.

    Displays point count, bounding box, centroid, elevation statistics,
    and per-square-metre density estimate.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``platform``,
        ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    from occulus.metrics import compute_cloud_statistics, coverage_statistics

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)

    stats = compute_cloud_statistics(cloud)

    if stats.intensity_mean is not None:
        pass

    try:
        coverage_statistics(cloud)  # type: ignore[arg-type]
    except Exception:
        logger.debug("Could not compute density", exc_info=True)

    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    """Run ground classification and write the result.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``output``,
        ``algorithm``, ``platform``, ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    from occulus.io import write
    from occulus.segmentation import classify_ground_csf, classify_ground_pmf

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)

    if args.algorithm == "csf":
        result = classify_ground_csf(cloud)  # type: ignore[arg-type]
    else:
        result = classify_ground_pmf(cloud)  # type: ignore[arg-type]

    write(result, args.output)
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    """Apply a spatial filter and write the result.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``output``,
        ``method``, ``voxel_size``, ``nb_neighbors``, ``std_ratio``,
        ``radius``, ``min_neighbors``, ``platform``, ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    from occulus.filters import (
        radius_outlier_removal,
        statistical_outlier_removal,
        voxel_downsample,
    )
    from occulus.io import write

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)

    if args.method == "voxel":
        result = voxel_downsample(cloud, voxel_size=args.voxel_size)  # type: ignore[arg-type]
    elif args.method == "sor":
        result, _ = statistical_outlier_removal(
            cloud,  # type: ignore[arg-type]
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
        )
    else:
        result, _ = radius_outlier_removal(
            cloud,  # type: ignore[arg-type]
            radius=args.radius,
            min_neighbors=args.min_neighbors,
        )

    write(result, args.output)
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    """Convert a point cloud between formats.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``output``,
        ``platform``, ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    from occulus.io import write

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)
    write(cloud, args.output)  # type: ignore[arg-type]
    return 0


def _cmd_dem(args: argparse.Namespace) -> int:
    """Rasterize a DEM from the point cloud and save as .npy.

    Supports IDW (inverse distance weighting) and nearest-neighbour
    interpolation to fill a regular grid of minimum-Z values.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``output``,
        ``resolution``, ``method``, ``platform``, ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    import numpy as np
    from scipy.spatial import KDTree  # type: ignore[import-untyped]

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)

    xyz = cloud.xyz
    res = args.resolution

    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    cols = int(np.ceil((x_max - x_min) / res)) + 1
    rows = int(np.ceil((y_max - y_min) / res)) + 1

    # Grid cell centres
    gx = np.linspace(x_min, x_min + (cols - 1) * res, cols)
    gy = np.linspace(y_min, y_min + (rows - 1) * res, rows)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    tree = KDTree(xyz[:, :2])

    if args.method == "nearest":
        _, idx = tree.query(grid_pts, k=1, workers=-1)
        dem = xyz[idx, 2].reshape(rows, cols)
    else:
        # IDW with k nearest neighbours
        k = min(12, cloud.n_points)
        dists, idx = tree.query(grid_pts, k=k, workers=-1)
        # Avoid division by zero for coincident points
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists**2
        z_vals = xyz[idx, 2]
        dem = (np.sum(weights * z_vals, axis=1) / np.sum(weights, axis=1)).reshape(rows, cols)

    out_path = Path(args.output)
    np.save(str(out_path), dem)
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    """Align source cloud to target via ICP and write result.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``source``, ``target``,
        ``output``, ``max_iterations``, ``tolerance``, ``platform``,
        ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    import numpy as np

    from occulus.io import write
    from occulus.registration import icp
    from occulus.types import PointCloud

    source = _read_cloud(args.source, platform=args.platform, subsample=args.subsample)
    target = _read_cloud(args.target, platform=args.platform, subsample=args.subsample)

    result = icp(
        source,  # type: ignore[arg-type]
        target,  # type: ignore[arg-type]
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )

    # Apply transformation to source
    src_xyz = source.xyz
    ones = np.ones((src_xyz.shape[0], 1), dtype=np.float64)
    homo = np.hstack((src_xyz, ones))
    transformed = (result.transformation @ homo.T).T[:, :3]

    aligned = PointCloud(transformed)
    write(aligned, args.output)

    return 0


def _cmd_tile(args: argparse.Namespace) -> int:
    """Tile a point cloud into spatial chunks and write each tile.

    Divides the XY bounding box into a regular grid of squares with
    edge length ``--tile-size`` and writes each non-empty tile as a
    separate file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes: ``input``, ``output``,
        ``tile_size``, ``platform``, ``subsample``.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    import numpy as np

    from occulus.io import write
    from occulus.types import PointCloud

    cloud = _read_cloud(args.input, platform=args.platform, subsample=args.subsample)

    xyz = cloud.xyz
    tile_size = args.tile_size

    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()

    # Compute tile indices for each point
    col_idx = np.floor((xyz[:, 0] - x_min) / tile_size).astype(int)
    row_idx = np.floor((xyz[:, 1] - y_min) / tile_size).astype(int)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine output extension from input
    in_ext = Path(args.input).suffix
    if not in_ext:
        in_ext = ".xyz"

    tile_keys = np.unique(np.column_stack((row_idx, col_idx)), axis=0)
    n_tiles = 0

    for r, c in tile_keys:
        mask = (row_idx == r) & (col_idx == c)
        tile_xyz = xyz[mask]
        if tile_xyz.shape[0] == 0:
            continue

        tile_cloud = PointCloud(tile_xyz)
        tile_name = f"tile_r{r:04d}_c{c:04d}{in_ext}"
        write(tile_cloud, out_dir / tile_name)
        n_tiles += 1

    return 0


_COMMANDS: dict[str, object] = {
    "info": _cmd_info,
    "classify": _cmd_classify,
    "filter": _cmd_filter,
    "convert": _cmd_convert,
    "dem": _cmd_dem,
    "register": _cmd_register,
    "tile": _cmd_tile,
}


def main() -> int:
    """Parse arguments and dispatch to the appropriate subcommand.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on handled error, 2 on usage error.
    """
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(getattr(args, "verbose", False))

    if args.version:
        return 0

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    handler = _COMMANDS.get(args.command)
    if handler is None:
        return 2

    try:
        return handler(args)  # type: ignore[operator]
    except Exception:
        logger.debug("Command failed", exc_info=True)
        return 1
