"""Kentucky ALS Ground Classification Example.

Downloads a small LiDAR tile from the USGS 3DEP program over Louisville, KY,
then classifies ground returns using the Cloth Simulation Filter (CSF) and
computes basic cloud statistics.

Data source
-----------
USGS 3D Elevation Program (3DEP) — public domain, freely available.
Tile: USGS LPC KY Metro Louisville B2 2019, 1/4-tile excerpt
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/USGS_LPC_KY_Metro_Louisville_B2_2019_
      e1275n4200.laz  (small excerpt fetched at runtime)

Alternatively, any LAS/LAZ file with aerial returns works — point it at a local
file with the --input flag.

Usage
-----
    python examples/scripts/kentucky_ground_classification.py
    python examples/scripts/kentucky_ground_classification.py --input path/to/cloud.las
    python examples/scripts/kentucky_ground_classification.py --no-viz
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

# Public 3DEP excerpt — a ~3 MB 1/4-tile over Jefferson County, KY.
# This is a small enough download to run quickly in demo contexts.
_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/"
    "USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def download_demo_tile(dest: Path) -> Path:
    """Download a small 3DEP LiDAR tile for the demo.

    Parameters
    ----------
    dest : Path
        Directory where the file will be saved.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    out = dest / "ky_louisville_demo.laz"
    if out.exists():
        logger.info("Demo tile already cached at %s", out)
        return out

    logger.info("Downloading USGS 3DEP tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(_DEMO_URL, str(out))
    except Exception as exc:
        logger.error(
            "Download failed: %s\n"
            "Provide a local file with --input, or check your network connection.",
            exc,
        )
        sys.exit(1)

    logger.info("Saved to %s", out)
    return out


def main() -> None:
    """Run the Kentucky ground classification demo."""
    parser = argparse.ArgumentParser(description="Occulus ground classification demo")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to a local LAS/LAZ file (skips download)",
    )
    parser.add_argument(
        "--subsample", type=float, default=0.25,
        help="Fraction of points to use (default 0.25 for speed)",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip Open3D visualization (headless environments)",
    )
    args = parser.parse_args()

    # -- I/O ------------------------------------------------------------------
    from occulus.io import read

    if args.input is not None:
        las_path = args.input
        logger.info("Using local file: %s", las_path)
    else:
        cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
        cache_dir.mkdir(parents=True, exist_ok=True)
        las_path = download_demo_tile(cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Statistics -----------------------------------------------------------
    from occulus.metrics import compute_cloud_statistics

    stats = compute_cloud_statistics(cloud)
    print("\n=== Cloud Statistics ===")
    print(f"  Points       : {cloud.n_points:,}")
    print(f"  Z range      : {stats.z_min:.2f} – {stats.z_max:.2f} m")
    print(f"  Z mean       : {stats.z_mean:.2f} m")
    print(f"  Z std        : {stats.z_std:.2f} m")
    if stats.intensity_mean is not None:
        print(f"  Intensity    : {stats.intensity_mean:.1f} (mean)")

    # -- Ground classification ------------------------------------------------
    from occulus.segmentation import classify_ground_csf

    logger.info("Running CSF ground classification…")
    classified = classify_ground_csf(cloud)

    from occulus.types import AerialCloud
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_ground = int((classified.classification == 2).sum())
        pct = n_ground / cloud.n_points * 100
        print(f"\n=== CSF Ground Classification ===")
        print(f"  Ground points: {n_ground:,} ({pct:.1f}%)")
        print(f"  Other points : {cloud.n_points - n_ground:,}")
    else:
        print("\nClassification complete.")

    # -- Output image ---------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from _plot_style import apply_report_style, save_figure, add_cross_section_line
        apply_report_style()

        xyz = cloud.xyz
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        from occulus.types import AerialCloud
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            colors = np.where(classified.classification == 2, 0, 1).astype(float)
            sc = ax_plan.scatter(xyz[:, 0], xyz[:, 1], c=colors,
                                 cmap="RdYlGn_r", s=0.3, alpha=0.6, rasterized=True)
            ax_plan.set_title("Ground Classification (green=ground, red=above)")
        else:
            ax_plan.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
                            cmap="terrain", s=0.3, alpha=0.6, rasterized=True)
            ax_plan.set_title("Louisville, KY — ALS Point Cloud")
        ax_plan.set_xlabel("Easting (m)"); ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(ax_plan, ax_prof, xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A\u2013A\u2032")

        ax_hist.hist(xyz[:, 2], bins=60, color="#4682B4", alpha=0.75, edgecolor="white")
        ax_hist.set_xlabel("Elevation (m NAVD88)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution — Kentucky Urban/Forest Terrain")

        fig.suptitle(
            "USGS 3DEP LiDAR — Eastern Kentucky (CSF Ground Classification)\n"
            f"Points: {cloud.n_points:,}  |  Z range: {stats.z_min:.1f}\u2013{stats.z_max:.1f} m",
            fontsize=12, fontweight="bold",
        )
        _out_dir = Path(__file__).parent.parent / "outputs"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out = _out_dir / "kentucky_ground_classification.png"
        save_figure(fig, out, alt_text=(
            "Four-panel figure showing Kentucky LiDAR ground classification: "
            "plan view with CSF ground/above-ground coloring, east-west elevation "
            "cross-section, and elevation histogram."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments
            import numpy as np

            if isinstance(classified, AerialCloud) and classified.classification is not None:
                # Label ground=0, vegetation/other=1
                labels = np.where(classified.classification == 2, 0, 1).astype("int32")
            else:
                labels = None

            if labels is not None:
                logger.info("Opening Open3D viewer…")
                visualize_segments(classified, labels, window_name="Kentucky ALS — Ground Classification")
            else:
                from occulus.viz import visualize
                visualize(classified, window_name="Kentucky ALS")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization. Install with: pip install occulus[viz]")


if __name__ == "__main__":
    main()
