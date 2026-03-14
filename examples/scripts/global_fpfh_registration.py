"""Global FPFH + RANSAC Registration Demo.

Downloads two adjacent USGS 3DEP ALS tiles, computes Fast Point Feature
Histograms (FPFH) descriptors, and runs RANSAC-based global registration to
find a coarse alignment without any initial pose estimate.  This is the first
stage of a complete registration pipeline (coarse → ICP fine).

FPFH encodes the local geometry around each point as a 33-dimensional
histogram, enabling robust descriptor matching across large initial misalignments.

Data source
-----------
USGS 3DEP — IN SWRegion 2017 (public domain).
Tile A: USGS_LPC_IN_SWRegion_2017_e0440n4195.laz
Tile B: USGS_LPC_IN_SWRegion_2017_e0441n4195.laz

Usage
-----
    python examples/scripts/global_fpfh_registration.py
    python examples/scripts/global_fpfh_registration.py --source a.las --target b.las
    python examples/scripts/global_fpfh_registration.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TILE_A_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IN_SWRegion_2017/laz/USGS_LPC_IN_SWRegion_2017_e0440n4195.laz"
)
_TILE_B_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IN_SWRegion_2017/laz/USGS_LPC_IN_SWRegion_2017_e0441n4195.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    """Download a tile if not cached.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Cache directory.

    Returns
    -------
    Path
        Local file path.
    """
    out = dest / Path(url).name
    if out.exists():
        logger.info("Cached: %s", out.name)
        return out
    logger.info("Downloading %s (~3 MB)…", Path(url).name)
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the FPFH + RANSAC global registration demo."""
    parser = argparse.ArgumentParser(description="FPFH + RANSAC global registration demo")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.normals import estimate_normals
    from occulus.registration import compute_fpfh, ransac_registration

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_path = args.source or _fetch(_TILE_A_URL, cache_dir)
    tgt_path = args.target or _fetch(_TILE_B_URL, cache_dir)

    logger.info("Reading source tile…")
    src = read(src_path, platform="aerial", subsample=args.subsample)
    logger.info("Reading target tile…")
    tgt = read(tgt_path, platform="aerial", subsample=args.subsample)

    logger.info("Voxel downsampling…")
    src_ds = voxel_downsample(src, voxel_size=args.voxel_size)
    tgt_ds = voxel_downsample(tgt, voxel_size=args.voxel_size)
    logger.info("  src: %d pts  |  tgt: %d pts", src_ds.n_points, tgt_ds.n_points)

    logger.info("Estimating normals…")
    radius = args.voxel_size * 3.0
    src_n = estimate_normals(src_ds, radius=radius)
    tgt_n = estimate_normals(tgt_ds, radius=radius)

    logger.info("Computing FPFH descriptors (radius=%.2f m)…", radius * 5)
    src_feat = compute_fpfh(src_n, radius=radius * 5)
    tgt_feat = compute_fpfh(tgt_n, radius=radius * 5)

    logger.info("Running RANSAC global registration…")
    result = ransac_registration(
        src_n, tgt_n,
        src_feat, tgt_feat,
        max_correspondence_distance=args.voxel_size * 3,
        max_iterations=100_000,
    )

    print("\n=== FPFH + RANSAC Global Registration ===")
    print(f"  Source points  : {src_ds.n_points:,}")
    print(f"  Target points  : {tgt_ds.n_points:,}")
    print(f"  Fitness        : {result.fitness:.4f}")
    print(f"  Inlier RMSE    : {result.inlier_rmse:.4f} m")
    print(f"  Converged      : {result.converged}")
    print(f"\n  Coarse transform:\n{result.transformation}")

    if not args.no_viz:
        try:
            from occulus.viz import visualize_registration
            logger.info("Opening Open3D viewer…")
            visualize_registration(src_n, tgt_n, result,
                                   window_name="FPFH + RANSAC Global Registration")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
