"""Pacific Northwest Old-Growth Forest Inventory — Oregon/Washington.

Downloads a USGS 3DEP tile over the Willamette National Forest (Oregon) and
runs a full forest inventory pipeline:
  - Ground classification (CSF)
  - Canopy height model
  - Individual tree detection (CHM-watershed)
  - Crown area and height distribution statistics

Old-growth Douglas-fir in this region exceeds 70 m — some of the tallest
forested terrain in USGS 3DEP coverage.

Data source: USGS 3DEP — OR Willamette National Forest 2018 (public domain).

Usage
-----
    python examples/scripts/pacific_northwest_forest.py
    python examples/scripts/pacific_northwest_forest.py --input path/to/forest.las
    python examples/scripts/pacific_northwest_forest.py --no-viz
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

OUTPUTS = Path(__file__).parent.parent / "outputs"

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "legacy/OR_COAST_CENTRAL_OLC_2011_2012/LAZ/"
    "USGS_LPC_OR_COAST_CENTRAL_OLC_2011_2012_OR_Coast_Central_OLC_2011-2012_001690.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    out = dest / Path(url).name
    if out.exists():
        return out
    logger.info("Downloading USGS 3DEP Oregon Willamette tile…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the Pacific Northwest forest inventory pipeline."""
    parser = argparse.ArgumentParser(description="PNW forest inventory — Oregon/WA")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--chm-resolution", type=float, default=0.5)
    parser.add_argument("--min-height", type=float, default=5.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)
    path = args.input or _fetch(_DEMO_URL, cache)

    logger.info("Reading PNW forest tile (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("  loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Cloud Statistics ===")
    print(f"  Points  : {cloud.n_points:,}")
    print(f"  Z range : {stats.z_min:.1f} – {stats.z_max:.1f} m")
    print(f"  Z std   : {stats.z_std:.1f} m")

    logger.info("CSF ground classification…")
    classified = classify_ground_csf(cloud)

    logger.info("Building CHM (resolution=%.1f m)…", args.chm_resolution)
    import numpy as np

    chm = xe = ye = None
    cov = None
    if isinstance(classified, AerialCloud):
        try:
            chm, xe, ye = canopy_height_model(classified, resolution=args.chm_resolution)
            cov = coverage_statistics(classified, resolution=args.chm_resolution)
            valid = chm[~np.isnan(chm)]
            print("\n=== Canopy Height Model ===")
            print(f"  CHM cells : {chm.size:,}")
            print(f"  Max height: {valid.max():.1f} m" if valid.size else "  (empty)")
            print(f"  Mean ht   : {valid[valid > 2].mean():.1f} m (trees > 2 m)")
        except Exception as exc:
            logger.warning("CHM failed: %s", exc)

    if cov:
        print(f"  Gap fraction  : {cov.gap_fraction:.2%}")
        print(f"  Mean density  : {cov.mean_density:.1f} pts/m²")

    logger.info("Individual tree segmentation (min_height=%.1f m)…", args.min_height)
    seg = None
    if isinstance(classified, AerialCloud):
        try:
            seg = segment_trees(
                classified, resolution=args.chm_resolution, min_height=args.min_height
            )
            print("\n=== Tree Inventory ===")
            print(f"  Trees detected : {seg.n_segments}")
        except Exception as exc:
            logger.warning("Tree seg skipped: %s", exc)

    if chm is not None:
        try:
            import matplotlib.pyplot as plt
            from _plot_style import CMAP_CANOPY, apply_report_style, save_figure

            apply_report_style()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                chm, origin="lower", cmap=CMAP_CANOPY, extent=[xe[0], xe[-1], ye[0], ye[-1]]
            )
            plt.colorbar(im, ax=ax, label="Height (m)")
            ax.set_title(
                "Pacific Northwest Old-Growth CHM — Oregon Coast\n"
                f"Trees: {seg.n_segments if seg else 'N/A'}  |  "
                f"Max height: {chm[~np.isnan(chm)].max():.1f} m"
            )
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            OUTPUTS.mkdir(parents=True, exist_ok=True)
            out = OUTPUTS / "pacific_northwest_chm.png"
            save_figure(
                fig,
                out,
                alt_text=(
                    "Canopy height model of Pacific Northwest old-growth forest showing "
                    "Douglas-fir and Sitka spruce heights along the Oregon coast."
                ),
            )
            logger.info("CHM image → %s", out)
            plt.close()
        except ImportError:
            pass

    if not args.no_viz and seg is not None:
        try:
            from occulus.viz import visualize_segments

            visualize_segments(classified, seg.labels, window_name="PNW Forest — Individual Trees")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
