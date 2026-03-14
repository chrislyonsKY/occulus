"""Arizona Desert Terrain — Sonoran Desert Basin and Range.

Downloads a USGS 3DEP LiDAR tile over the Sonoran Desert near Phoenix/Tucson,
capturing the basin-and-range topography: flat alluvial bajadas cut by desert
arroyos, surrounded by rugged volcanic mountain ranges.

Demonstrates:
  - PMF ground classification on arid, sparsely vegetated terrain
  - Arroyo (ephemeral stream) detection via ground surface
  - Low canopy fraction from desert scrub

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/arizona_desert_terrain.py
    python examples/scripts/arizona_desert_terrain.py --no-viz
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

# Sonoran Desert — southwest of Phoenix, near South Mountain Park
_BBOX = "-112.2,33.2,-112.0,33.4"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for an Arizona desert tile."""
    logger.info("Querying USGS National Map for Arizona LiDAR tile…")
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
    logger.info("Downloading Arizona desert tile…")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def main() -> None:
    """Run Arizona Sonoran Desert terrain analysis."""
    parser = argparse.ArgumentParser(description="Arizona desert terrain analysis")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.25)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf, classify_ground_pmf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    logger.info("Reading Arizona desert point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    import numpy as np
    from occulus.types import AerialCloud

    stats = compute_cloud_statistics(cloud)
    print("\n=== Arizona Desert Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.0f} – {stats.z_max:.0f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.0f} m")

    logger.info("PMF ground classification (good for open desert terrain)…")
    classified_pmf = classify_ground_pmf(cloud)
    logger.info("CSF ground classification…")
    classified_csf = classify_ground_csf(cloud)

    for label, classified in [("PMF", classified_pmf), ("CSF", classified_csf)]:
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            n_g = int((classified.classification == 2).sum())
            print(f"  {label} Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")

    try:
        cov = coverage_statistics(classified_csf, resolution=args.resolution)
        print(f"\n=== Vegetation Coverage ===")
        print(f"  Gap fraction  : {cov.gap_fraction:.2%} (high = sparse desert scrub)")
        print(f"  Mean density  : {cov.mean_density:.1f} pts/m²")
    except Exception as exc:
        logger.warning("Coverage skipped: %s", exc)

    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_HEAT, apply_report_style, save_figure, add_cross_section_line
        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        sc = ax_plan.scatter(
            xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
            cmap=CMAP_HEAT, s=0.3, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m)")
        ax_plan.set_title("Sonoran Desert — Basin and Range Topography")
        ax_plan.set_xlabel("Easting (m)"); ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(ax_plan, ax_prof, xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A\u2013A\u2032")

        ax_hist.hist(xyz[:, 2], bins=60, color="#B8860B", alpha=0.75, edgecolor="white")
        ax_hist.set_xlabel("Elevation (m)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution — Desert Bajada")

        fig.suptitle(
            "USGS 3DEP LiDAR — Sonoran Desert, Arizona (Basin and Range)\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m",
            fontsize=12, fontweight="bold",
        )
        out = OUTPUTS / "arizona_desert_terrain.png"
        save_figure(fig, out, alt_text=(
            "Four-panel figure showing Sonoran Desert LiDAR analysis: plan view colored "
            "by elevation, east-west cross-section profile, and elevation histogram."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name="Arizona Desert — Sonoran Basin")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
