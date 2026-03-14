"""Louisiana Coastal Wetlands — Mississippi River Delta Terrain Analysis.

Downloads a USGS 3DEP LiDAR tile over the Mississippi River delta / Atchafalaya
Basin in southern Louisiana — one of the most dynamic deltaic landscapes in the
world, with extensive marsh, swamp, and open water at near-zero elevation.

Demonstrates:
  - Ground classification on extremely flat, low-relief terrain
  - Identifying water vs marsh vs forested wetland returns
  - Density analysis for canopy occlusion effects

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/louisiana_wetlands_delta.py
    python examples/scripts/louisiana_wetlands_delta.py --no-viz
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

# Louisiana coastal wetlands bounding box (west of New Orleans)
_BBOX = "-91.2,29.5,-91.0,29.7"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM API for a LiDAR tile URL in Louisiana wetlands."""
    logger.info("Querying USGS National Map for Louisiana LiDAR tile…")
    try:
        with urllib.request.urlopen(_TNM_URL, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.error("TNM API query failed: %s", exc)
        sys.exit(1)
    items = data.get("items", [])
    if not items:
        logger.error("No LiDAR tiles found for Louisiana bbox %s", _BBOX)
        sys.exit(1)
    url = items[0]["downloadURL"]
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, cache: Path) -> Path:
    """Download tile with caching."""
    dest = cache / url.split("/")[-1]
    if dest.exists():
        logger.info("Using cached tile: %s", dest.name)
        return dest
    logger.info("Downloading Louisiana wetlands tile…")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def main() -> None:
    """Run Louisiana wetlands terrain analysis."""
    parser = argparse.ArgumentParser(description="Louisiana wetlands delta analysis")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.25)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    if args.input:
        path = args.input
    else:
        url = _find_tile()
        path = _fetch(url, cache)

    logger.info("Reading point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Louisiana Wetlands Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.2f} – {stats.z_max:.2f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.1f} m (very low — deltaic terrain)")
    print(f"  Z std     : {stats.z_std:.2f} m")

    logger.info("CSF ground classification (flat terrain settings)…")
    classified = classify_ground_csf(cloud, cloth_resolution=1.0, class_threshold=0.5)

    import numpy as np
    from occulus.types import AerialCloud

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")

    try:
        cov = coverage_statistics(classified, resolution=args.resolution)
        print(f"\n=== Coverage Statistics ===")
        print(f"  Gap fraction  : {cov.gap_fraction:.2%}")
        print(f"  Mean density  : {cov.mean_density:.1f} pts/m²")
    except Exception as exc:
        logger.warning("Coverage stats skipped: %s", exc)

    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_CANOPY, apply_report_style, save_figure, add_cross_section_line
        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        sc = ax_plan.scatter(
            xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
            cmap=CMAP_CANOPY, s=0.3, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m NAVD88)")
        ax_plan.set_title("Louisiana Wetlands — Plan View Elevation")
        ax_plan.set_xlabel("Easting (m)"); ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(ax_plan, ax_prof, xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A\u2013A\u2032")

        ax_hist.hist(xyz[:, 2], bins=60, color="#228B22", alpha=0.75, edgecolor="white")
        ax_hist.axvline(0, color="#D32F2F", linestyle="--", lw=1.5, label="Sea level")
        ax_hist.set_xlabel("Elevation (m NAVD88)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution — Near-Zero Deltaic Terrain")
        ax_hist.legend(fontsize=9)

        fig.suptitle(
            "USGS 3DEP LiDAR — Mississippi River Delta, Louisiana\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.1f} m",
            fontsize=12, fontweight="bold",
        )
        out = OUTPUTS / "louisiana_wetlands_delta.png"
        save_figure(fig, out, alt_text=(
            "Four-panel figure showing Louisiana wetlands LiDAR analysis: plan view "
            "colored by elevation, east-west cross-section through deltaic terrain, "
            "and elevation histogram with sea level reference line."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name="Louisiana Wetlands Delta")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
