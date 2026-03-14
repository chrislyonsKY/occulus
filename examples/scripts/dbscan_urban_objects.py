"""DBSCAN Urban Object Clustering from USGS 3DEP ALS.

Loads a USGS 3DEP urban ALS tile, classifies and removes ground points, then
clusters the remaining above-ground returns (buildings, trees, vehicles) using
DBSCAN.  Reports cluster count, size distribution, and dominant object types
inferred from height statistics.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019 (public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/dbscan_urban_objects.py
    python examples/scripts/dbscan_urban_objects.py --input path/to/cloud.las
    python examples/scripts/dbscan_urban_objects.py --no-viz
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
    "USGS_LPC_KY_Metro_Louisville_B2_2019/laz/"
    "USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    """Download tile if not cached.

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
    """Run the DBSCAN urban object clustering demo."""
    parser = argparse.ArgumentParser(description="DBSCAN urban object clustering")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1.5,
                        help="DBSCAN neighbourhood radius (m, default 1.5)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="DBSCAN minimum points per cluster (default 10)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.segmentation import classify_ground_csf, cluster_dbscan
    from occulus.types import AerialCloud, PointCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Ground classification and removal ------------------------------------
    logger.info("Classifying ground…")
    classified = classify_ground_csf(cloud)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        above_mask = classified.classification != 2
        above_xyz = classified.xyz[above_mask]
    else:
        # Heuristic: points more than 0.5 m above the P5 elevation
        z_thresh = float(np.percentile(classified.xyz[:, 2], 5)) + 0.5
        above_mask = classified.xyz[:, 2] > z_thresh
        above_xyz = classified.xyz[above_mask]

    logger.info("Above-ground points: %d", len(above_xyz))

    # Downsample for speed
    above_cloud = PointCloud(xyz=above_xyz.astype(np.float32))
    above_ds = voxel_downsample(above_cloud, voxel_size=0.3)
    logger.info("After downsample: %d points", above_ds.n_points)

    # -- DBSCAN clustering ----------------------------------------------------
    logger.info("Running DBSCAN (eps=%.2f, min_samples=%d)…", args.eps, args.min_samples)
    seg = cluster_dbscan(above_ds, eps=args.eps, min_samples=args.min_samples)

    unique_labels = np.unique(seg.labels)
    n_clusters = int((unique_labels >= 0).sum())
    n_noise = int((seg.labels == -1).sum())

    print("\n=== DBSCAN Urban Object Clustering ===")
    print(f"  Above-ground points: {above_ds.n_points:,}")
    print(f"  Clusters found     : {n_clusters}")
    print(f"  Noise points       : {n_noise:,}  ({n_noise / above_ds.n_points * 100:.1f}%)")

    # Cluster size statistics
    sizes = [int((seg.labels == lbl).sum()) for lbl in unique_labels if lbl >= 0]
    if sizes:
        sizes_arr = np.array(sizes)
        print(f"\n  Cluster size stats:")
        print(f"    Min  : {sizes_arr.min():,} pts")
        print(f"    Max  : {sizes_arr.max():,} pts")
        print(f"    Mean : {sizes_arr.mean():.1f} pts")
        print(f"    Median: {float(np.median(sizes_arr)):.1f} pts")

        # Height-based heuristic: large clusters = buildings, small = trees/vehicles
        n_large = int((sizes_arr > 500).sum())
        n_small = int((sizes_arr <= 500).sum())
        print(f"\n  Large clusters (>500 pts, likely buildings) : {n_large}")
        print(f"  Small clusters (≤500 pts, likely trees/objects): {n_small}")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments
            logger.info("Opening Open3D viewer…")
            visualize_segments(above_ds, seg.labels,
                               window_name=f"DBSCAN Urban Objects — {n_clusters} clusters")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
