"""Netherlands AHN4 Polder Terrain Analysis.

Downloads a tile from the Dutch AHN4 (Actueel Hoogtebestand Nederland) LiDAR
survey — the most detailed national elevation model in the world at ~10 pts/m².
The tile covers low-lying polder land near Delft (below sea level).

AHN4 data is freely available at https://www.ahn.nl/ under CC0 (public domain).
Tiles are served from the TU Delft GeoTiles service.

Usage
-----
    python examples/scripts/netherlands_ahn4_polder.py
    python examples/scripts/netherlands_ahn4_polder.py --no-viz
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

# AHN4 tile 37ez1 — Delft/Pijnacker area, South Holland (~2–3 m below MSL)
_AHN4_URL = "https://geotiles.citg.tudelft.nl/AHN4_t/37ez1.LAZ"
_TILE_NAME = "37ez1.LAZ"


def _fetch(url: str, dest: Path) -> Path:
    """Download an AHN4 LAZ tile with caching."""
    out = dest / Path(url).name
    if out.exists():
        logger.info("Using cached AHN4 tile: %s", out.name)
        return out
    logger.info("Downloading AHN4 tile from TU Delft GeoTiles…")
    try:
        urllib.request.urlretrieve(url, str(out))
        logger.info("Downloaded → %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local LAZ file.", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run Netherlands AHN4 polder terrain analysis."""
    parser = argparse.ArgumentParser(description="Netherlands AHN4 polder terrain")
    parser.add_argument("--input", type=Path, default=None, help="Local LAZ/LAS file")
    parser.add_argument("--subsample", type=float, default=0.15,
                        help="Point fraction to keep (AHN4 is very dense)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Grid resolution in metres")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_AHN4_URL, cache)

    logger.info("Reading AHN4 tile (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== AHN4 Cloud Statistics ===")
    print(f"  Points      : {cloud.n_points:,}")
    print(f"  Z range     : {stats.z_min:.2f} – {stats.z_max:.2f} m (NAP)")
    print(f"  Z mean      : {stats.z_mean:.2f} m")
    print("  Note: Negative Z values indicate land below sea level (polder)")

    logger.info("CSF ground classification…")
    classified = classify_ground_csf(cloud, cloth_resolution=0.5, class_threshold=0.5)

    import numpy as np
    from occulus.types import AerialCloud

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")

        try:
            chm, xe, ye = canopy_height_model(classified, resolution=args.resolution)
            cov = coverage_statistics(classified, resolution=args.resolution)
            valid = chm[~np.isnan(chm)]
            print(f"\n=== Canopy / Structure ===")
            print(f"  CHM cells    : {chm.size:,}")
            if valid.size:
                print(f"  Max height   : {valid.max():.1f} m")
            print(f"  Gap fraction : {cov.gap_fraction:.2%}")
        except Exception as exc:
            logger.warning("CHM skipped: %s", exc)
            chm = xe = ye = None

    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_DIVERGING, apply_report_style, save_figure, add_cross_section_line
        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        sc = ax_plan.scatter(
            xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
            cmap=CMAP_DIVERGING, s=0.2, alpha=0.6, rasterized=True,
            vmin=np.percentile(xyz[:, 2], 2), vmax=np.percentile(xyz[:, 2], 98),
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m NAP)")
        ax_plan.set_title("Netherlands AHN4 — Elevation Map (Delft Polder)")
        ax_plan.set_xlabel("Easting (m RD)"); ax_plan.set_ylabel("Northing (m RD)")

        add_cross_section_line(ax_plan, ax_prof, xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A\u2013A\u2032")

        below = (xyz[:, 2] < 0).sum()
        ax_hist.hist(xyz[:, 2], bins=80, color="#4682B4", alpha=0.75, edgecolor="white")
        ax_hist.axvline(0, color="#D32F2F", linestyle="--", lw=1.5, label="Sea level (0 m NAP)")
        ax_hist.set_xlabel("Elevation (m NAP)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title(f"Elevation Distribution — {below / cloud.n_points:.1%} of points below sea level")
        ax_hist.legend(fontsize=9)

        fig.suptitle(
            "Netherlands AHN4 LiDAR — Delft Polder (CC0, ahn.nl)\n"
            f"Points: {cloud.n_points:,}  |  Z range: {stats.z_min:.2f}\u2013{stats.z_max:.2f} m NAP",
            fontsize=12, fontweight="bold",
        )
        out = OUTPUTS / "netherlands_ahn4_polder.png"
        save_figure(fig, out, alt_text=(
            "Four-panel figure showing Netherlands AHN4 polder LiDAR analysis: "
            "plan view colored by elevation showing below-sea-level land, east-west "
            "cross-section through polder terrain, and elevation histogram with sea level line."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name="Netherlands AHN4 — Delft Polder")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
