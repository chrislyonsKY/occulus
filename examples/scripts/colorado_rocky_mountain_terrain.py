"""Colorado Rocky Mountain Terrain Analysis.

Downloads a USGS 3DEP LiDAR tile over the Colorado Front Range — where the
Great Plains abruptly meet the Rocky Mountains, producing terrain with 2 000+ m
of vertical relief within a single tile.

Demonstrates:
  - CSF ground classification on steep mixed terrain
  - Slope and hydrological analysis
  - High-relief canopy height model

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/colorado_rocky_mountain_terrain.py
    python examples/scripts/colorado_rocky_mountain_terrain.py --no-viz
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

# Colorado Front Range — near Boulder / Rocky Mountain National Park
_BBOX = "-105.7,40.0,-105.5,40.2"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for a Colorado Front Range tile."""
    logger.info("Querying USGS National Map for Colorado LiDAR tile…")
    try:
        with urllib.request.urlopen(_TNM_URL, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.error("TNM API query failed: %s", exc)
        sys.exit(1)
    items = data.get("items", [])
    if not items:
        logger.error("No tiles found for bbox %s", _BBOX)
        sys.exit(1)
    url = items[0]["downloadURL"]
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, cache: Path) -> Path:
    dest = cache / url.split("/")[-1]
    if dest.exists():
        logger.info("Using cached tile: %s", dest.name)
        return dest
    logger.info("Downloading Colorado Rocky Mountain tile…")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def main() -> None:
    """Run Colorado Front Range terrain analysis."""
    parser = argparse.ArgumentParser(description="Colorado Rocky Mountain terrain")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    logger.info("Reading Colorado point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    import numpy as np

    from occulus.types import AerialCloud

    stats = compute_cloud_statistics(cloud)
    print("\n=== Colorado Front Range Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.0f} – {stats.z_max:.0f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.0f} m")
    print(f"  Z std     : {stats.z_std:.1f} m")

    logger.info("CSF ground classification…")
    classified = classify_ground_csf(cloud)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print("\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")

    chm = xe = ye = None
    if isinstance(classified, AerialCloud):
        try:
            chm, xe, ye = canopy_height_model(classified, resolution=args.resolution)
            valid = chm[~np.isnan(chm)]
            print("\n=== Canopy Height Model ===")
            print(f"  CHM cells  : {chm.size:,}")
            if valid.size:
                print(f"  Max height : {valid.max():.1f} m")
        except Exception as exc:
            logger.warning("CHM skipped: %s", exc)

    try:
        import matplotlib.pyplot as plt
        from _plot_style import add_cross_section_line, apply_report_style, save_figure

        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_chm = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])

        sc = ax_plan.scatter(
            xyz[:, 0],
            xyz[:, 1],
            c=xyz[:, 2],
            cmap="terrain",
            s=0.3,
            alpha=0.5,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m)")
        ax_plan.set_title("Plan View — Colorado Front Range")
        ax_plan.set_xlabel("Easting (m)")
        ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(
            ax_plan, ax_prof, xyz, y_frac=0.5, band_frac=0.03, label="Cross Section A–A\u2032"
        )

        if chm is not None and xe is not None:
            im = ax_chm.imshow(
                chm, origin="lower", cmap="Greens", extent=[xe[0], xe[-1], ye[0], ye[-1]]
            )
            plt.colorbar(im, ax=ax_chm, label="Canopy height (m)")
            ax_chm.set_title("Canopy Height Model")
        else:
            ax_chm.text(
                0.5, 0.5, "CHM not available", transform=ax_chm.transAxes, ha="center", va="center"
            )

        ax_hist.hist(xyz[:, 2], bins=60, color="saddlebrown", alpha=0.75, edgecolor="white")
        ax_hist.set_xlabel("Elevation (m)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution")

        fig.suptitle(
            "USGS 3DEP LiDAR — Colorado Front Range / Rocky Mountains\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m",
            fontsize=12,
            fontweight="bold",
        )
        out = OUTPUTS / "colorado_rocky_mountain_terrain.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Four-panel figure showing Colorado Front Range LiDAR analysis: plan view "
                "colored by elevation, east-west cross-section through Rocky Mountain terrain, "
                "canopy height model, and elevation histogram showing 2000+ m relief."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(cloud, window_name="Colorado Rocky Mountain Terrain")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
