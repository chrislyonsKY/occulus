"""Flood Terrain Analysis — Low-Elevation Risk Zone Identification.

Loads a USGS 3DEP ALS tile, classifies ground returns, computes the bare-earth
elevation distribution, and identifies low-elevation zones below a configurable
percentile threshold as a proxy for flood-risk areas.  This is a simplified
first-pass screening — real flood modelling requires hydraulic simulation.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019 (Ohio River floodplain, public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/flood_terrain_analysis.py
    python examples/scripts/flood_terrain_analysis.py --input path/to/cloud.las
    python examples/scripts/flood_terrain_analysis.py --threshold 10 --no-viz
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
    """Run the flood terrain risk analysis demo."""
    parser = argparse.ArgumentParser(description="Occulus flood terrain analysis demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Elevation percentile below which points are flagged as flood risk "
                             "(default 10 = 10th percentile)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="CHM resolution in metres (default 1.0)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Cloud Statistics ===")
    print(f"  Points  : {cloud.n_points:,}")
    print(f"  Z range : {stats.z_min:.2f} – {stats.z_max:.2f} m")

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud)

    # -- Extract ground Z values ----------------------------------------------
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        ground_mask = classified.classification == 2
        ground_xyz = classified.xyz[ground_mask]
    else:
        ground_xyz = classified.xyz

    ground_z = ground_xyz[:, 2]
    n_ground = len(ground_z)

    # -- Flood risk threshold -------------------------------------------------
    flood_threshold = float(np.percentile(ground_z, args.threshold))
    flood_mask = ground_z <= flood_threshold
    n_flood = int(flood_mask.sum())

    percentiles = np.percentile(ground_z, [10, 25, 50, 75, 90])

    print("\n=== Ground Elevation Distribution ===")
    print(f"  Ground points : {n_ground:,}")
    print(f"  Z min         : {ground_z.min():.2f} m")
    print(f"  Z max         : {ground_z.max():.2f} m")
    print(f"  P10 / P50 / P90: {percentiles[0]:.2f} / {percentiles[2]:.2f} / {percentiles[4]:.2f} m")

    print(f"\n=== Flood Risk Zones (below {args.threshold:.0f}th percentile = {flood_threshold:.2f} m) ===")
    print(f"  At-risk points : {n_flood:,}  ({n_flood / n_ground * 100:.1f}% of ground)")
    print(f"  Safe points    : {n_ground - n_flood:,}")
    print(f"  Risk threshold : {flood_threshold:.2f} m AMSL")

    # -- CHM for context ------------------------------------------------------
    chm, x_edges, y_edges = canopy_height_model(classified, resolution=args.resolution)
    valid = ~np.isnan(chm)
    if valid.any():
        low_cells = int((chm[valid] <= flood_threshold).sum())
        print(f"\n=== Grid-Level Flood Risk (resolution={args.resolution} m) ===")
        print(f"  Total valid cells    : {int(valid.sum())}")
        print(f"  At-risk grid cells   : {low_cells}  ({low_cells / valid.sum() * 100:.1f}%)")
        area_m2 = low_cells * args.resolution ** 2
        print(f"  Estimated risk area  : {area_m2:.0f} m²  ({area_m2 / 1e4:.2f} ha)")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments
            all_z = classified.xyz[:, 2]
            per_point_threshold = float(np.percentile(all_z, args.threshold))
            labels = np.where(all_z <= per_point_threshold, 1, 0).astype("int32")
            logger.info("Opening Open3D viewer (red = flood risk)…")
            visualize_segments(classified, labels,
                               window_name=f"Flood Risk — below {args.threshold:.0f}th pct")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
