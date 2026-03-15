"""Urban Building Detection — Point Density & Planarity Analysis.

Downloads a USGS 3DEP LiDAR tile over a dense US city and demonstrates
building detection using geometric feature extraction. Produces a two-panel
figure inspired by urban 3-D mapping workflows:

  1. Point density map (plan view) — buildings appear as high-density clusters
  2. Planarity side view — building facades show high planarity values

Data: USGS 3DEP — public domain, freely available.

Usage
-----
    python examples/scripts/urban_building_detection.py
    python examples/scripts/urban_building_detection.py --no-viz
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

# Chicago Loop / downtown — dense high-rise core
_BBOX = "-87.64,41.875,-87.62,41.895"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for a Chicago downtown tile."""
    logger.info("Querying USGS National Map for Chicago urban LiDAR tile…")
    req = urllib.request.Request(_TNM_URL, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.error("TNM API query failed: %s", exc)
        sys.exit(1)
    items = data.get("items", [])
    if not items:
        logger.error("No tiles found for Chicago bbox %s", _BBOX)
        sys.exit(1)
    url = items[0]["downloadURL"]
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, cache: Path) -> Path:
    """Download a LAZ tile with caching."""
    dest = cache / url.split("/")[-1]
    if dest.exists():
        logger.info("Using cached tile: %s", dest.name)
        return dest
    logger.info("Downloading Chicago urban tile…")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    logger.info("Downloaded → %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def main() -> None:
    """Run urban building detection with point density and planarity."""
    parser = argparse.ArgumentParser(description="Urban building detection via geometric features")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.features import compute_geometric_features
    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics, point_density

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    logger.info("Reading urban point cloud (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Urban Cloud Statistics ===")
    print(f"  Points  : {cloud.n_points:,}")
    print(f"  Z range : {stats.z_min:.1f} – {stats.z_max:.1f} m")

    # Compute point density (returns tuple: density_grid, x_edges, y_edges)
    logger.info("Computing point density (resolution=1.0 m)…")
    density_result = point_density(cloud, resolution=1.0)
    if isinstance(density_result, tuple):
        density_grid = density_result[0]
    else:
        density_grid = density_result

    # Compute geometric features for planarity
    logger.info("Computing geometric features (planarity for building facades)…")
    feats = compute_geometric_features(cloud, radius=3.0)
    planarity = feats.planarity

    print(f"\n=== Geometric Features ===")
    print(f"  Planarity: mean={planarity.mean():.3f}, max={planarity.max():.3f}")
    print(f"  (High planarity = flat surfaces like building walls and roofs)")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from _plot_style import CMAP_HEAT, CMAP_PROFILE, apply_report_style, save_figure

        apply_report_style()
        xyz = cloud.xyz

        fig, (ax_density, ax_side) = plt.subplots(1, 2, figsize=(16, 7))

        # Panel 1: Point density plan view
        if density_grid.ndim == 2:
            im = ax_density.imshow(
                density_grid.T,
                origin="lower",
                cmap=CMAP_HEAT,
                aspect="equal",
                extent=[xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max()],
                interpolation="nearest",
            )
        else:
            # Fallback: scatter plot colored by local density
            im = ax_density.scatter(
                xyz[:, 0], xyz[:, 1], c=xyz[:, 2], cmap=CMAP_HEAT, s=0.3, alpha=0.5, rasterized=True
            )
        plt.colorbar(im, ax=ax_density, label="Points / m²")
        ax_density.set_title("Point Density — Plan View")
        ax_density.set_xlabel("Easting (m)")
        ax_density.set_ylabel("Northing (m)")
        ax_density.set_aspect("equal")

        # Panel 2: Planarity side view (east-west cross section colored by planarity)
        # Sort by easting for clean side view
        order = np.argsort(xyz[:, 0])
        sc = ax_side.scatter(
            xyz[order, 0],
            xyz[order, 2],
            c=planarity[order],
            cmap=CMAP_PROFILE,
            s=0.3,
            alpha=0.6,
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax_side, label="Planarity")
        ax_side.set_title("Planarity (side view — facades = high)")
        ax_side.set_xlabel("Easting (m)")
        ax_side.set_ylabel("Height (m)")

        fig.suptitle(
            "USGS 3DEP LiDAR — Urban Building Detection\n"
            f"Points: {cloud.n_points:,}  |  Z range: {stats.z_min:.1f}\u2013{stats.z_max:.1f} m",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = OUTPUTS / "urban_building_detection.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Two-panel figure showing urban building detection from USGS 3DEP LiDAR. "
                "Left: point density plan view with buildings as high-density clusters on a dark "
                "background using the inferno colormap. Right: planarity side view showing building "
                "facades as high-planarity vertical structures using the viridis colormap."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping figure output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(cloud, window_name="Urban Building Detection")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
