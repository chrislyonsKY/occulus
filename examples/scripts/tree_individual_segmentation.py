"""Individual Tree Segmentation from Eastern KY Forested ALS Tile.

Downloads a USGS 3DEP ALS tile from the Daniel Boone National Forest region
of eastern Kentucky.  Classifies ground returns, builds a canopy height model,
runs CHM-watershed individual tree segmentation, and reports crown counts and
per-tree area/height statistics.

Data source
-----------
USGS 3DEP — KY Statewide 2019 (forested tile), public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz

Usage
-----
    python examples/scripts/tree_individual_segmentation.py
    python examples/scripts/tree_individual_segmentation.py --input path/to/cloud.las
    python examples/scripts/tree_individual_segmentation.py --no-viz
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
    """Run the individual tree segmentation demo."""
    parser = argparse.ArgumentParser(description="Individual tree segmentation demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--chm-resolution", type=float, default=0.5)
    parser.add_argument("--min-height", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import canopy_height_model, coverage_statistics
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading forest tile (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_ground = int((classified.classification == 2).sum())
        print(f"\n  Ground points : {n_ground:,}  ({n_ground / cloud.n_points * 100:.1f}%)")

    # -- CHM ------------------------------------------------------------------
    chm, _x_edges, _y_edges = canopy_height_model(classified, resolution=args.chm_resolution)
    valid = ~np.isnan(chm)
    print("\n=== Canopy Height Model ===")
    print(f"  Shape         : {chm.shape[0]} × {chm.shape[1]} cells")
    if valid.any():
        print(f"  Height range  : {float(np.nanmin(chm)):.1f} – {float(np.nanmax(chm)):.1f} m")
        print(f"  Mean height   : {float(np.nanmean(chm)):.1f} m")

    # -- Coverage statistics --------------------------------------------------
    cov = coverage_statistics(classified, resolution=args.chm_resolution)
    print("\n=== Coverage ===")
    print(f"  Gap fraction  : {cov.gap_fraction:.2%}")
    print(f"  Covered area  : {cov.covered_area:.1f} m²")

    # -- Tree segmentation ----------------------------------------------------
    logger.info("Running tree segmentation (min_height=%.1f m)…", args.min_height)
    seg = segment_trees(classified, resolution=args.chm_resolution, min_height=args.min_height)

    print("\n=== Individual Tree Segmentation ===")
    print(f"  Trees detected : {seg.n_segments}")
    print(f"  Noise points   : {int((seg.labels == -1).sum()):,}")

    # Per-tree statistics
    tree_sizes = []
    tree_heights = []
    for tree_id in range(seg.n_segments):
        mask = seg.labels == tree_id
        pts = classified.xyz[mask]
        if len(pts) > 0:
            tree_sizes.append(len(pts))
            tree_heights.append(float(pts[:, 2].max() - pts[:, 2].min()))

    if tree_sizes:
        sizes_arr = np.array(tree_sizes)
        heights_arr = np.array(tree_heights)
        print("\n  Point count per tree:")
        print(
            f"    Min / Max / Mean : {sizes_arr.min()} / {sizes_arr.max()} / {sizes_arr.mean():.1f}"
        )
        print("\n  Crown height range (m):")
        print(
            f"    Min / Max / Mean : {heights_arr.min():.1f} / {heights_arr.max():.1f} / {heights_arr.mean():.1f}"
        )

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments

            logger.info("Opening Open3D viewer (%d trees)…", seg.n_segments)
            visualize_segments(
                classified, seg.labels, window_name=f"Eastern KY Forest — {seg.n_segments} trees"
            )
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
