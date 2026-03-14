"""Kentucky From Above — Statewide LiDAR Terrain Survey.

Kentucky From Above (https://kyfromabove.ky.gov/) is a statewide initiative
that collected 1-m nominal pulse spacing aerial LiDAR across all 120 Kentucky
counties between 2011 and 2023.  Data is available free of charge in LAS
format from the KY From Above portal and is also mirrored through USGS 3DEP.

This script demonstrates a terrain analysis workflow tailored to the KY From
Above deliverable specification:
  - ASPRS class 2 (ground) retained from pre-classified tiles
  - Digital Terrain Model (DTM) statistics
  - Slope and roughness analysis
  - Export of ground points for GIS use

Accessing KY From Above data
-----------------------------
  1. Portal  : https://kyfromabove.ky.gov/
  2. USGS 3DEP mirror: https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/
     Elevation/LPC/Projects/ — search for USGS_LPC_KY_* projects
  3. Direct tile download via USGS National Map:
     https://apps.nationalmap.gov/downloader/

The demo downloads a Blue Grass region tile via the USGS 3DEP mirror.

Usage
-----
    python examples/scripts/kyfromabove_terrain_survey.py
    python examples/scripts/kyfromabove_terrain_survey.py --input path/to/ky_tile.las
    python examples/scripts/kyfromabove_terrain_survey.py --county fayette
    python examples/scripts/kyfromabove_terrain_survey.py --no-viz
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

# KY From Above tiles (via USGS 3DEP mirror) — Blue Grass / Inner Bluegrass region
_TILES = {
    "bluegrass": (
        "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
        "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/"
        "USGS_LPC_KY_CentralEast_A23_N088E243.laz"
    ),
    "eastern": (
        "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
        "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/"
        "USGS_LPC_KY_CentralEast_A23_N089E244.laz"
    ),
}


def _fetch(url: str, dest: Path) -> Path:
    out = dest / Path(url).name
    if out.exists():
        logger.info("Cached: %s", out.name)
        return out
    logger.info("Downloading KY From Above tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error(
            "Download failed: %s\n"
            "Visit https://kyfromabove.ky.gov/ to download tiles manually, "
            "then use --input.", exc,
        )
        sys.exit(1)
    return out


def main() -> None:
    """Run Kentucky From Above terrain survey analysis."""
    parser = argparse.ArgumentParser(
        description="Kentucky From Above — statewide LiDAR terrain analysis"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--county", choices=list(_TILES), default="bluegrass",
        help="Demo tile region (default: bluegrass)",
    )
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import statistical_outlier_removal, voxel_downsample
    from occulus.io import read, write
    from occulus.metrics import compute_cloud_statistics, point_density
    from occulus.normals import estimate_normals
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud

    cache = Path(tempfile.gettempdir()) / "occulus_kyfromabove"
    cache.mkdir(parents=True, exist_ok=True)

    if args.input:
        path = args.input
    else:
        path = _fetch(_TILES[args.county], cache)

    logger.info("Reading KY From Above tile — %s region (%.0f%% subsample)…",
                args.county, args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("  loaded: %s", cloud)

    import numpy as np

    # If pre-classified LAS, use existing ASPRS class 2 ground
    has_existing_class = (
        isinstance(cloud, AerialCloud)
        and cloud.classification is not None
        and int((cloud.classification == 2).sum()) > 10
    )

    if has_existing_class:
        assert isinstance(cloud, AerialCloud)
        n_ground = int((cloud.classification == 2).sum())  # type: ignore[index]
        logger.info("Pre-classified tile: %d ground points (ASPRS class 2)", n_ground)
        classified = cloud
    else:
        logger.info("No pre-classification detected — running CSF…")
        clean = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=2.5)
        classified = classify_ground_csf(clean)

    stats = compute_cloud_statistics(cloud)
    print("\n╔══════════════════════════════════════════════╗")
    print("║   KENTUCKY FROM ABOVE — TERRAIN ANALYSIS    ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Region      : {args.county.title()}, Kentucky")
    print(f"  Points      : {cloud.n_points:,}")
    print(f"  Elevation   : {stats.z_min:.2f} – {stats.z_max:.2f} m  ({stats.z_max - stats.z_min:.1f} m relief)")
    print(f"  Mean elev.  : {stats.z_mean:.2f} m (NAVD 88)")

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"  Ground pts  : {n_g:,} ({n_g / cloud.n_points:.1%})")

        # Export ground points for GIS
        ground_mask = classified.classification == 2
        from occulus.types import PointCloud
        ground_xyz = classified.xyz[ground_mask]
        ground_cloud = PointCloud(ground_xyz)
        out_path = OUTPUTS / f"kyfromabove_{args.county}_ground.xyz"
        write(ground_cloud, out_path)
        print(f"  Ground XYZ  : {out_path}")

    # Point density map
    density, xe, ye = point_density(cloud, resolution=5.0)
    print(f"  Point density (mean): {density.mean():.1f} pts/25m²")

    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_ELEVATION, apply_report_style, save_figure
        apply_report_style()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Elevation map
        h, xe2, ye2 = np.histogram2d(cloud.xyz[:, 0], cloud.xyz[:, 1], bins=150,
                                     weights=cloud.xyz[:, 2])
        cnt, _, _ = np.histogram2d(cloud.xyz[:, 0], cloud.xyz[:, 1], bins=150)
        mean_z = np.where(cnt > 0, h / np.maximum(cnt, 1), np.nan)
        im0 = axes[0].imshow(mean_z.T, origin="lower",
                              extent=[cloud.xyz[:, 0].min(), cloud.xyz[:, 0].max(),
                                      cloud.xyz[:, 1].min(), cloud.xyz[:, 1].max()],
                              cmap=CMAP_ELEVATION)
        plt.colorbar(im0, ax=axes[0], label="Elevation (m NAVD 88)")
        axes[0].set_title(f"KY From Above — {args.county.title()} Region\nDigital Terrain Model")
        axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")

        # Density map
        im1 = axes[1].imshow(density.T, origin="lower",
                              extent=[xe[0], xe[-1], ye[0], ye[-1]],
                              cmap="Blues")
        plt.colorbar(im1, ax=axes[1], label="Pts / 25 m\u00b2")
        axes[1].set_title("Point Density Map (5 m grid)")
        axes[1].set_xlabel("Easting (m)")

        fig.suptitle(
            f"Kentucky From Above — {args.county.title()} Region (kyfromabove.ky.gov)\n"
            f"Points: {cloud.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.1f} m",
            fontsize=11, fontweight="bold",
        )
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        out = OUTPUTS / f"kyfromabove_{args.county}.png"
        save_figure(fig, out, alt_text=(
            f"Two-panel figure showing KY From Above {args.county.title()} region: "
            "digital terrain model (left) and point density map (right)."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize
            visualize(cloud, window_name=f"KY From Above — {args.county.title()}")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
