"""Canopy Height Model and Tree Segmentation Example.

Downloads a USGS 3DEP tile from a forested area in eastern Kentucky (Daniel
Boone National Forest region), classifies ground returns, builds a canopy
height model (CHM), and runs the CHM-watershed tree segmentation algorithm.

Data source: USGS 3DEP — KY Statewide 2019, public domain.

Usage
-----
    python examples/scripts/canopy_height_model.py
    python examples/scripts/canopy_height_model.py --input path/to/forest.las
    python examples/scripts/canopy_height_model.py --no-viz
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

# USGS 3DEP KY Statewide 2019 — forested tile, eastern KY (~3 MB)
_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def _fetch(url: str, dest: Path) -> Path:
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
    """Run the CHM and tree segmentation demo."""
    parser = argparse.ArgumentParser(description="Occulus CHM and tree segmentation demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--chm-resolution", type=float, default=0.5, help="CHM cell size (m)")
    parser.add_argument("--min-height", type=float, default=2.0, help="Min tree height (m)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import canopy_height_model, coverage_statistics
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading LiDAR tile (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("  loaded: %s", cloud)

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud)

    # -- Canopy height model --------------------------------------------------
    logger.info("Building canopy height model (resolution=%.1f m)…", args.chm_resolution)
    if isinstance(classified, AerialCloud):
        chm, x_edges, y_edges = canopy_height_model(classified, resolution=args.chm_resolution)
        print("\n=== Canopy Height Model ===")
        import numpy as np

        valid = chm[~np.isnan(chm)]
        if valid.size > 0:
            print(f"  CHM shape     : {chm.shape[0]} × {chm.shape[1]} cells")
            print(f"  Height range  : {valid.min():.1f} – {valid.max():.1f} m")
            print(f"  Mean height   : {valid.mean():.1f} m")
        else:
            print("  (no valid CHM cells)")

        # -- Coverage statistics ----------------------------------------------
        cov = coverage_statistics(classified, resolution=args.chm_resolution)
        print("\n=== Coverage Statistics ===")
        print(f"  Gap fraction  : {cov.gap_fraction:.2%}")
        print(f"  Covered area  : {cov.covered_area:.1f} m²")
        print(f"  Mean density  : {cov.mean_density:.1f} pts/m²")

    # -- Tree segmentation ----------------------------------------------------
    logger.info("Running CHM-watershed tree segmentation (min_height=%.1f m)…", args.min_height)
    try:
        seg = segment_trees(
            classified,
            resolution=args.chm_resolution,
            min_height=args.min_height,
        )
        print("\n=== Tree Segmentation ===")
        print(f"  Trees found   : {seg.n_segments}")
        print(f"  Noise points  : {(seg.labels == -1).sum():,}")
    except Exception as exc:
        logger.warning("Tree segmentation skipped: %s", exc)
        seg = None

    # -- Visualization --------------------------------------------------------
    if not args.no_viz and seg is not None:
        try:
            from occulus.viz import visualize_segments

            logger.info("Opening Open3D viewer (%d trees)…", seg.n_segments)
            visualize_segments(
                classified, seg.labels, window_name=f"Kentucky Forest — {seg.n_segments} trees"
            )
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")

    # -- CHM plot (matplotlib) ------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_CANOPY, apply_report_style, save_figure

        apply_report_style()
        if isinstance(classified, AerialCloud):
            chm, x_edges, y_edges = canopy_height_model(classified, resolution=args.chm_resolution)
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                chm,
                origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                cmap=CMAP_CANOPY,
            )
            plt.colorbar(im, ax=ax, label="Height (m)")
            ax.set_title(
                "Canopy Height Model — Eastern Kentucky\n"
                f"Resolution: {args.chm_resolution} m  |  "
                f"Max height: {valid.max():.1f} m"
                if valid.size
                else ""
            )
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            _out_dir = Path(__file__).parent.parent / "outputs"
            _out_dir.mkdir(parents=True, exist_ok=True)
            out_png = _out_dir / "canopy_height_model.png"
            save_figure(
                fig,
                out_png,
                alt_text=(
                    "Canopy height model raster for eastern Kentucky showing tree heights "
                    "up to 30+ meters in the Daniel Boone National Forest region."
                ),
            )
            logger.info("CHM saved to %s", out_png)
            plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping CHM plot.")


if __name__ == "__main__":
    main()
