"""Landslide Monitoring via Pre/Post Survey ICP Registration.

Simulates a repeat ALS survey workflow for landslide monitoring.  A "pre-event"
cloud is derived from a real USGS 3DEP tile.  A "post-event" cloud is created
by applying a localised synthetic displacement field to a sub-region of the
tile (mimicking a slide mass moving downslope).  ICP is used to register the
two surveys, then per-point residuals highlight the displaced region.

Data source
-----------
USGS 3DEP — KY Statewide 2019 (hilly terrain, public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz

Usage
-----
    python examples/scripts/landslide_monitoring.py
    python examples/scripts/landslide_monitoring.py --input path/to/cloud.las
    python examples/scripts/landslide_monitoring.py --no-viz
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

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz"
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
    logger.info("Downloading USGS 3DEP tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the landslide monitoring demo."""
    parser = argparse.ArgumentParser(description="Occulus landslide monitoring demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument(
        "--slide-magnitude",
        type=float,
        default=1.5,
        help="Peak displacement magnitude (m) of synthetic slide",
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.normals import estimate_normals
    from occulus.registration import icp_point_to_plane
    from occulus.types import PointCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading pre-event cloud (subsample=%.0f%%)…", args.subsample * 100)
    pre_cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", pre_cloud)

    pre_ds = voxel_downsample(pre_cloud, voxel_size=args.voxel_size)

    # -- Build post-event cloud -----------------------------------------------
    rng = np.random.default_rng(17)
    xyz = pre_ds.xyz.copy()

    # Define slide region: top-left quadrant of the bounding box
    xmid = float(xyz[:, 0].mean())
    ymid = float(xyz[:, 1].mean())
    slide_mask = (xyz[:, 0] < xmid) & (xyz[:, 1] > ymid)

    logger.info("Applying synthetic landslide to %d points…", int(slide_mask.sum()))

    # Displacement tapers smoothly within the slide region
    slide_xyz = xyz[slide_mask]
    cx = float(slide_xyz[:, 0].mean())
    cy = float(slide_xyz[:, 1].mean())
    dist = np.linalg.norm(slide_xyz[:, :2] - np.array([cx, cy]), axis=1)
    sigma = dist.max() * 0.5
    weight = np.exp(-0.5 * (dist / (sigma + 1e-9)) ** 2)

    # Move downslope (negative Y) and downward (negative Z)
    xyz[slide_mask, 1] -= args.slide_magnitude * weight * 2.0
    xyz[slide_mask, 2] -= args.slide_magnitude * weight

    xyz += rng.normal(0, 0.01, xyz.shape).astype(np.float32)
    post_cloud = PointCloud(xyz=xyz)

    # -- Register with ICP ----------------------------------------------------
    radius = args.voxel_size * 3
    pre_n = estimate_normals(pre_ds, radius=radius)
    post_n = estimate_normals(post_cloud, radius=radius)

    logger.info("Running ICP (post → pre)…")
    result = icp_point_to_plane(
        post_n,
        pre_n,
        max_correspondence_distance=args.voxel_size * 3,
        max_iterations=200,
    )

    print("\n=== ICP Alignment (Post → Pre) ===")
    print(f"  Fitness    : {result.fitness:.4f}")
    print(f"  Inlier RMSE: {result.inlier_rmse:.4f} m")
    print(f"  Converged  : {result.converged}")

    # -- Residual analysis ----------------------------------------------------
    T = result.transformation
    xyz_aligned = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    residuals = np.linalg.norm(xyz_aligned - pre_ds.xyz, axis=1)

    threshold = args.slide_magnitude * 0.5
    displaced = residuals > threshold

    print("\n=== Displacement Analysis ===")
    print(f"  Slide region (ground truth) : {int(slide_mask.sum()):,} pts")
    print(f"  Detected displaced (>{threshold:.2f} m): {int(displaced.sum()):,} pts")
    print(f"  Mean residual (all)  : {residuals.mean():.4f} m")
    print(f"  Max residual         : {residuals.max():.4f} m")
    print(f"  Mean residual (slide): {residuals[slide_mask].mean():.4f} m")
    print(f"  Mean residual (stable): {residuals[~slide_mask].mean():.4f} m")

    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments

            labels = displaced.astype("int32")
            logger.info("Opening Open3D viewer (red = displaced)…")
            visualize_segments(pre_ds, labels, window_name="Landslide Monitoring")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
