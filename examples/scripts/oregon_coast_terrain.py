"""Oregon Coast Terrain Analysis — Pacific Cliffs and Beach Profiles.

Downloads a USGS 3DEP LiDAR tile from the Oregon coastline — where sea cliffs
up to 90 m drop directly to narrow sandy beaches, backed by Sitka spruce and
western red cedar rainforest.

Demonstrates:
  - Ground classification on mixed beach-cliff-forest terrain
  - Canopy height model over coastal forest
  - Cliff-face occlusion in point density maps

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/oregon_coast_terrain.py
    python examples/scripts/oregon_coast_terrain.py --no-viz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# Oregon Coast — Heceta Head / Florence area
_BBOX = "-124.2,44.1,-124.0,44.3"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for an Oregon coast tile."""
    logger.info("Querying USGS National Map for Oregon coastal LiDAR tile…")
    try:
        with urllib.request.urlopen(_TNM_URL, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.error("TNM API query failed: %s", exc)
        sys.exit(1)
    items = data.get("items", [])
    if not items:
        logger.error("No tiles found for Oregon coast bbox %s", _BBOX)
        sys.exit(1)
    url = items[0]["downloadURL"]
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, cache: Path) -> Path:
    dest = cache / url.split("/")[-1]
    if dest.exists():
        logger.info("Using cached tile: %s", dest.name)
        return dest
    logger.info("Downloading Oregon coast tile…")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def main() -> None:
    """Run Oregon coast terrain analysis."""
    parser = argparse.ArgumentParser(description="Oregon coast terrain analysis")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.25)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    logger.info("Reading Oregon coast point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    import numpy as np

    from occulus.types import AerialCloud

    stats = compute_cloud_statistics(cloud)
    print("\n=== Oregon Coast Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.1f} – {stats.z_max:.1f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.0f} m")

    logger.info("CSF ground classification…")
    classified = classify_ground_csf(cloud)

    chm = xe = ye = None
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print("\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")

        try:
            chm, xe, ye = canopy_height_model(classified, resolution=args.resolution)
            cov = coverage_statistics(classified, resolution=args.resolution)
            valid = chm[~np.isnan(chm)]
            print("\n=== Coastal Forest Canopy ===")
            if valid.size:
                print(f"  Max canopy height : {valid.max():.1f} m")
            print(f"  Gap fraction      : {cov.gap_fraction:.2%}")
        except Exception as exc:
            logger.warning("CHM skipped: %s", exc)

    try:
        import matplotlib.pyplot as plt
        from _plot_style import (
            CMAP_CANOPY,
            CMAP_ELEVATION,
            add_cross_section_line,
            apply_report_style,
            save_figure,
        )

        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_chm = fig.add_subplot(gs[1, :])

        sc = ax_plan.scatter(
            xyz[:, 0],
            xyz[:, 1],
            c=xyz[:, 2],
            cmap=CMAP_ELEVATION,
            s=0.3,
            alpha=0.5,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m)")
        ax_plan.set_title("Oregon Coast — Cliff, Beach, and Forest")
        ax_plan.set_xlabel("Easting (m)")
        ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(
            ax_plan, ax_prof, xyz, y_frac=0.5, band_frac=0.03, label="Cross Section A\u2013A\u2032"
        )

        if chm is not None and xe is not None:
            im = ax_chm.imshow(
                chm,
                origin="lower",
                cmap=CMAP_CANOPY,
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
            )
            plt.colorbar(im, ax=ax_chm, label="Canopy height (m)")
            ax_chm.set_title("Coastal Rainforest Canopy Height Model")
            ax_chm.set_xlabel("Easting (m)")
            ax_chm.set_ylabel("Northing (m)")
        else:
            ax_chm.hist(xyz[:, 2], bins=60, color="#008080", alpha=0.75, edgecolor="white")
            ax_chm.set_xlabel("Elevation (m)")
            ax_chm.set_ylabel("Count")
            ax_chm.set_title("Elevation Distribution")

        fig.suptitle(
            "USGS 3DEP LiDAR — Oregon Pacific Coastline (Cliff-Beach-Forest)\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m",
            fontsize=12,
            fontweight="bold",
        )
        out = OUTPUTS / "oregon_coast_terrain.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Four-panel figure showing Oregon coast LiDAR analysis: plan view colored "
                "by elevation showing sea cliffs and forest, east-west cross-section profile, "
                "and coastal rainforest canopy height model."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(cloud, window_name="Oregon Coast Terrain")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
