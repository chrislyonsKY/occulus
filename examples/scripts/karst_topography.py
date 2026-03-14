"""Karst Topography Analysis — Sinkhole Detection via Local Minima.

Eastern Kentucky lies over carboniferous limestone, creating karst features
including sinkholes (dolines).  This script downloads a USGS 3DEP ALS tile
from eastern KY, classifies ground returns, builds a CHM (used as a terrain
surface proxy), and identifies local minima in the surface as candidate
sinkhole locations using scipy.ndimage.

Data source
-----------
USGS 3DEP — KY Statewide 2019, public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz

Usage
-----
    python examples/scripts/karst_topography.py
    python examples/scripts/karst_topography.py --input path/to/cloud.las
    python examples/scripts/karst_topography.py --no-viz
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
    """Run the karst sinkhole detection demo."""
    parser = argparse.ArgumentParser(description="Occulus karst topography demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Grid resolution in metres (default 1.0)"
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.5,
        help="Minimum depth below surroundings to flag as sinkhole (m)",
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf

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

    # -- CHM (terrain surface proxy) -----------------------------------------
    logger.info("Building terrain surface (resolution=%.1f m)…", args.resolution)
    chm, _x_edges, _y_edges = canopy_height_model(classified, resolution=args.resolution)

    valid = ~np.isnan(chm)
    filled = chm.copy()
    if valid.any():
        filled[~valid] = float(np.nanmean(chm))

    # -- Sinkhole detection via local minima ----------------------------------
    try:
        from scipy.ndimage import label, minimum_filter

        logger.info("Detecting local minima (candidate sinkholes)…")
        # A cell is a local minimum if it equals the minimum in a 5x5 window
        min_filtered = minimum_filter(filled, size=5, mode="reflect")
        local_min_mask = (filled == min_filtered) & valid

        # Filter by depth: local min must be >= min_depth below mean of 5x5 neighbourhood
        from scipy.ndimage import uniform_filter

        local_mean = uniform_filter(filled, size=5, mode="reflect")
        depth = local_mean - filled
        sinkhole_mask = local_min_mask & (depth >= args.min_depth)

        labeled, n_candidates = label(sinkhole_mask)
        depths = [float(depth[labeled == i + 1].max()) for i in range(n_candidates)]

        print("\n=== Karst Sinkhole Candidates ===")
        print(f"  Grid size         : {chm.shape[0]} × {chm.shape[1]} cells")
        print(f"  Total local minima: {int(local_min_mask.sum())}")
        print(f"  Candidate sinkholes (depth >= {args.min_depth} m): {n_candidates}")
        if depths:
            print(f"  Max candidate depth : {max(depths):.2f} m")
            print(f"  Mean candidate depth: {sum(depths) / len(depths):.2f} m")
    except ImportError:
        logger.warning("scipy not installed — skipping local minima detection.")
        print("\nInstall scipy to enable sinkhole detection: pip install scipy")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize

            logger.info("Opening Open3D viewer…")
            visualize(classified, window_name="Karst Topography — Eastern KY")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
