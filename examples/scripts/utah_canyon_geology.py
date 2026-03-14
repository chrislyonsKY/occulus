"""Utah Canyon Geology — Canyonlands and Colorado Plateau.

Downloads a USGS 3DEP LiDAR tile from the Colorado Plateau in Utah — one of the
most dramatic landscapes in North America, with canyon walls dropping 300–500 m,
mesa tops, and entrenched river meanders.

Demonstrates:
  - Ground classification on vertical cliff terrain
  - High-relief CHM and surface roughness
  - Density analysis highlighting shadowed canyon interiors

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/utah_canyon_geology.py
    python examples/scripts/utah_canyon_geology.py --no-viz
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

# Canyonlands / Moab area, Utah
_BBOX = "-109.7,38.5,-109.5,38.7"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for a Utah canyon tile."""
    logger.info("Querying USGS National Map for Utah canyonlands tile…")
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
    logger.info("Downloading Utah canyon tile…")
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    return dest


def main() -> None:
    """Run Utah canyon geology analysis."""
    parser = argparse.ArgumentParser(description="Utah canyon geology analysis")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    logger.info("Reading Utah canyon point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    import numpy as np
    from occulus.types import AerialCloud

    stats = compute_cloud_statistics(cloud)
    print("\n=== Utah Canyonlands Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.0f} – {stats.z_max:.0f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.0f} m (canyon depth)")
    print(f"  Z std     : {stats.z_std:.1f} m")

    logger.info("CSF ground classification on canyon terrain…")
    classified = classify_ground_csf(cloud, cloth_resolution=2.0)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"\n=== Ground Classification ===")
        print(f"  Ground : {n_g:,} ({n_g / cloud.n_points:.1%})")
        print(f"  Above  : {cloud.n_points - n_g:,}")

    try:
        import matplotlib.pyplot as plt
        from _plot_style import apply_report_style, save_figure, add_cross_section_line
        apply_report_style()
        xyz = cloud.xyz

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sc = axes[0].scatter(
            xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
            cmap="copper", s=0.3, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc, ax=axes[0], label="Elevation (m)")
        axes[0].set_title("Plan View — Utah Canyonlands")
        axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")

        add_cross_section_line(axes[0], axes[1], xyz, y_frac=0.5,
                               band_frac=0.03, label="Cross Section A–A\u2032")

        fig.suptitle(
            "USGS 3DEP LiDAR — Utah Canyonlands, Colorado Plateau\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m  |  "
            f"Ground: {n_g:,} ({n_g / cloud.n_points:.1%})",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        out = OUTPUTS / "utah_canyon_geology.png"
        save_figure(fig, out, alt_text=(
            "Two-panel figure showing Utah Canyonlands LiDAR analysis: plan view "
            "colored by elevation showing canyon walls, and east-west cross-section "
            "through canyon terrain with 300-500 m relief."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name="Utah Canyonlands — Colorado Plateau")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
