"""Iran Fault Geomorphology — Sabzevar Region, Northeast Iran.

Downloads a subset of the Sabzevar fault photogrammetric point cloud from
OpenTopography (OTDS.102023.32640.2) — 70 km of active fault scarps,
pressure ridges, and offset alluvial fans across the Kopet Dag fold-and-
thrust belt.  The dataset was derived from SPOT-6 satellite imagery and
released under CC BY 4.0.

Citation
--------
Zinke R., Hollingsworth J., Dolan J.F. (2023). "Point clouds derived from
satellite imagery, Sabzevar, Iran, 2013-2014". OpenTopography.
https://doi.org/10.5069/G9KS6PS0

API key
-------
Set OPENTOPO_API_KEY in your environment or in a .env file at the project
root.  Free keys: https://opentopography.org/

Usage
-----
    python examples/scripts/iran_fault_geomorphology.py
    python examples/scripts/iran_fault_geomorphology.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# Bounding box: fault scarp segment near Sabzevar city (small area for demo)
_WEST  = 57.10
_SOUTH = 36.22
_EAST  = 57.30
_NORTH = 36.35
_DATASET = "OTDS.102023.32640.2"

_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


def _load_api_key() -> str:
    """Load OpenTopography API key from environment or .env file."""
    key = os.environ.get("OPENTOPO_API_KEY", "")
    if not key and _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            if line.startswith("OPENTOPO_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError(
            "OPENTOPO_API_KEY not set. Get a free key at https://opentopography.org/ "
            "and add it to the .env file or export OPENTOPO_API_KEY=<key>"
        )
    return key


def _download_tile(cache: Path) -> Path:
    """Download the smaller Sabzevar LAZ file from OpenTopography S3."""
    dest = cache / "sabzevar_1_utm40n.laz"
    if dest.exists():
        logger.info("Using cached tile: %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
        return dest

    url = (
        "https://opentopography.s3.sdsc.edu/dataspace/"
        "OTDS.102023.32640.2/pointcloud/sabzevar_1_utm40n.laz"
    )
    logger.info("Downloading Sabzevar fault tile from OpenTopography S3 (~891 MB)…")
    logger.info("This is a one-time download; future runs use the cached copy.")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        raise
    return dest


def main() -> None:
    """Run Iran fault geomorphology analysis."""
    parser = argparse.ArgumentParser(description="Iran fault geomorphology — Sabzevar")
    parser.add_argument("--input", type=Path, default=None, help="Local LAZ/LAS file")
    parser.add_argument("--subsample", type=float, default=0.25)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    if args.input:
        path = args.input
    else:
        path = _download_tile(cache)

    logger.info("Reading point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Sabzevar Fault Cloud Statistics ===")
    print(f"  Points     : {cloud.n_points:,}")
    print(f"  Z range    : {stats.z_min:.1f} – {stats.z_max:.1f} m")
    print(f"  Relief     : {stats.z_max - stats.z_min:.0f} m")
    print(f"  Z std      : {stats.z_std:.1f} m")

    logger.info("Classifying ground returns with CSF…")
    classified = classify_ground_csf(cloud)

    from occulus.types import AerialCloud
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")
        print(f"  Above  : {cloud.n_points - n_g:,}")

    # --- Visualisation -------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_ELEVATION, apply_report_style, save_figure, add_cross_section_line
        apply_report_style()
        xyz = cloud.xyz

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, :])

        sc = ax_plan.scatter(
            xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
            cmap=CMAP_ELEVATION, s=0.3, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m)")
        ax_plan.set_title("Plan View — Sabzevar Fault Zone, Iran")
        ax_plan.set_xlabel("Easting (m)"); ax_plan.set_ylabel("Northing (m)")

        add_cross_section_line(ax_plan, ax_prof, xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A\u2013A\u2032")

        ax_hist.hist(xyz[:, 2], bins=60, color="#8B4513", alpha=0.75, edgecolor="white")
        ax_hist.set_xlabel("Elevation (m)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution — Fault Scarp Relief")

        fig.suptitle(
            "Iran Fault Geomorphology — Sabzevar, NE Iran (OpenTopography CC BY 4.0)\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m",
            fontsize=12, fontweight="bold",
        )
        out = OUTPUTS / "iran_fault_geomorphology.png"
        save_figure(fig, out, alt_text=(
            "Four-panel figure showing Iran Sabzevar fault geomorphology: plan view "
            "colored by elevation showing fault scarps, east-west cross-section "
            "through the fault zone, and elevation histogram."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name="Iran Fault Geomorphology — Sabzevar")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
