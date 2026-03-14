"""USGS 3DEP Multi-Scan ICP Registration Example.

Downloads two adjacent LiDAR tiles from USGS 3DEP (Indiana — Statewide 2017,
public domain) and registers them with FPFH-guided RANSAC global registration
followed by ICP fine alignment.

This demonstrates a real terrestrial/aerial multi-scan workflow:
  1. Read two overlapping point cloud tiles
  2. Voxel-downsample for speed
  3. Estimate normals
  4. FPFH global registration → coarse alignment
  5. Point-to-plane ICP → fine alignment
  6. Report fitness / RMSE metrics

Usage
-----
    python examples/scripts/usgs_3dep_icp_registration.py
    python examples/scripts/usgs_3dep_icp_registration.py --no-viz
    python examples/scripts/usgs_3dep_icp_registration.py --source a.las --target b.las
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

# Two adjacent 1/4-tiles from USGS 3DEP Indiana Statewide 2017 (~2–4 MB each).
_TILE_A_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IN_SWRegion_2017/laz/USGS_LPC_IN_SWRegion_2017_e0440n4195.laz"
)
_TILE_B_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_IN_SWRegion_2017/laz/USGS_LPC_IN_SWRegion_2017_e0441n4195.laz"
)


def _fetch(url: str, dest: Path) -> Path:
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
    """Run the USGS 3DEP ICP registration demo."""
    parser = argparse.ArgumentParser(description="Occulus ICP registration demo")
    parser.add_argument("--source", type=Path, default=None, help="Local source tile path")
    parser.add_argument("--target", type=Path, default=None, help="Local target tile path")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size for downsampling (m)")
    parser.add_argument("--subsample", type=float, default=0.2, help="Read-time subsample fraction")
    parser.add_argument("--no-viz", action="store_true", help="Skip Open3D visualization")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.normals import estimate_normals
    from occulus.registration import (
        compute_fpfh,
        icp,
        icp_point_to_plane,
        ransac_registration,
    )

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.source is None:
        args.source = _fetch(_TILE_A_URL, cache_dir)
    if args.target is None:
        args.target = _fetch(_TILE_B_URL, cache_dir)

    # -- Read -----------------------------------------------------------------
    logger.info("Reading source tile…")
    src = read(args.source, platform="aerial", subsample=args.subsample)
    logger.info("  source: %s", src)

    logger.info("Reading target tile…")
    tgt = read(args.target, platform="aerial", subsample=args.subsample)
    logger.info("  target: %s", tgt)

    # -- Downsample -----------------------------------------------------------
    logger.info("Voxel downsampling (voxel_size=%.2f m)…", args.voxel_size)
    src_ds = voxel_downsample(src, voxel_size=args.voxel_size)
    tgt_ds = voxel_downsample(tgt, voxel_size=args.voxel_size)
    logger.info("  downsampled: %d → %d  |  %d → %d", src.n_points, src_ds.n_points, tgt.n_points, tgt_ds.n_points)

    # -- Normals --------------------------------------------------------------
    logger.info("Estimating normals…")
    radius = args.voxel_size * 3.0
    src_n = estimate_normals(src_ds, radius=radius)
    tgt_n = estimate_normals(tgt_ds, radius=radius)

    # -- Global registration (FPFH + RANSAC) ----------------------------------
    logger.info("Computing FPFH descriptors…")
    src_feat = compute_fpfh(src_n, radius=radius * 5)
    tgt_feat = compute_fpfh(tgt_n, radius=radius * 5)

    logger.info("Running RANSAC global registration…")
    coarse = ransac_registration(
        src_n, tgt_n,
        src_feat, tgt_feat,
        max_correspondence_distance=args.voxel_size * 3,
        max_iterations=50_000,
    )
    print("\n=== RANSAC Global Registration ===")
    print(f"  Fitness    : {coarse.fitness:.4f}")
    print(f"  Inlier RMSE: {coarse.inlier_rmse:.4f} m")
    print(f"  Converged  : {coarse.converged}")

    # -- Fine ICP (point-to-plane) --------------------------------------------
    logger.info("Running point-to-plane ICP fine alignment…")
    fine = icp_point_to_plane(
        src_n, tgt_n,
        initial_transform=coarse.transformation,
        max_correspondence_distance=args.voxel_size * 1.5,
        max_iterations=200,
    )
    print("\n=== ICP Fine Alignment ===")
    print(f"  Fitness    : {fine.fitness:.4f}")
    print(f"  Inlier RMSE: {fine.inlier_rmse:.6f} m")
    print(f"  Iterations : {fine.n_iterations}")
    print(f"  Converged  : {fine.converged}")
    print(f"\n  Transformation:\n{fine.transformation}")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_registration
            logger.info("Opening Open3D viewer…")
            visualize_registration(src_n, tgt_n, fine, window_name="3DEP ICP Registration")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
