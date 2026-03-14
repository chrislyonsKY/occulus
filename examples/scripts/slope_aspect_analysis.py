"""Slope and Aspect Analysis from ALS Terrain Data.

Loads a USGS 3DEP ALS tile, classifies ground returns, builds a canopy height
model (used here as a proxy DTM surface), then applies numpy gradient operators
to derive slope magnitude and aspect.  Slope and aspect are fundamental terrain
derivatives used in hydrology, ecology, and geomorphology.

Data source
-----------
USGS 3DEP — KY Statewide 2019 (forested/hilly tile), public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz

Usage
-----
    python examples/scripts/slope_aspect_analysis.py
    python examples/scripts/slope_aspect_analysis.py --input path/to/cloud.las
    python examples/scripts/slope_aspect_analysis.py --no-viz
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

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    """Download a tile to *dest* if not already cached.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Cache directory.

    Returns
    -------
    Path
        Local file path.
    """
    out = dest / Path(url).name
    if out.exists():
        logger.info("Cached: %s", out.name)
        return out
    logger.info("Downloading USGS 3DEP tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the slope and aspect analysis demo."""
    parser = argparse.ArgumentParser(description="Occulus slope/aspect analysis demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="DTM grid resolution in metres (default 1.0)"
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, point_density
    from occulus.segmentation import classify_ground_csf

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Cloud statistics -----------------------------------------------------
    stats = compute_cloud_statistics(cloud)
    print("\n=== Cloud Statistics ===")
    print(f"  Points  : {cloud.n_points:,}")
    print(f"  Z range : {stats.z_min:.2f} – {stats.z_max:.2f} m")

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud)

    # -- Point density --------------------------------------------------------
    density, _xe, _ye = point_density(classified, resolution=args.resolution)
    print(f"\n=== Point Density (resolution={args.resolution} m) ===")
    valid_d = density[density > 0]
    print(f"  Mean density : {valid_d.mean():.1f} pts/m²")
    print(f"  Max density  : {valid_d.max():.1f} pts/m²")

    # -- CHM / DTM surface ----------------------------------------------------
    logger.info("Building CHM surface…")
    chm, x_edges, y_edges = canopy_height_model(classified, resolution=args.resolution)

    # Fill NaN cells with local mean for gradient computation
    valid_mask = ~np.isnan(chm)
    filled = chm.copy()
    if valid_mask.any():
        filled[~valid_mask] = float(np.nanmean(chm))

    # -- Slope and aspect (numpy gradient) ------------------------------------
    dy, dx = np.gradient(filled, args.resolution, args.resolution)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_deg = np.degrees(np.arctan2(-dy, dx)) % 360.0

    # Mask cells that had no data
    slope_deg[~valid_mask] = np.nan
    aspect_deg[~valid_mask] = np.nan

    print("\n=== Slope Statistics ===")
    valid_slope = slope_deg[valid_mask]
    print(f"  Min slope  : {valid_slope.min():.2f}°")
    print(f"  Max slope  : {valid_slope.max():.2f}°")
    print(f"  Mean slope : {valid_slope.mean():.2f}°")
    print(
        f"  Steep (>30°): {(valid_slope > 30).sum():,} cells "
        f"({(valid_slope > 30).mean() * 100:.1f}%)"
    )

    print("\n=== Aspect Statistics ===")
    valid_aspect = aspect_deg[valid_mask]
    print(
        f"  North-facing (<45° or >315°): "
        f"{((valid_aspect < 45) | (valid_aspect > 315)).sum():,} cells"
    )
    print(
        f"  South-facing (135°–225°)    : "
        f"{((valid_aspect > 135) & (valid_aspect < 225)).sum():,} cells"
    )

    # -- Optional plot --------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_HEAT, apply_report_style, save_figure

        apply_report_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        im0 = axes[0].imshow(slope_deg, origin="lower", extent=extent, cmap=CMAP_HEAT)
        plt.colorbar(im0, ax=axes[0], label="Slope (\u00b0)")
        axes[0].set_title("Slope Magnitude")
        axes[0].set_xlabel("Easting (m)")
        axes[0].set_ylabel("Northing (m)")
        im1 = axes[1].imshow(aspect_deg, origin="lower", extent=extent, cmap="hsv")
        plt.colorbar(im1, ax=axes[1], label="Aspect (\u00b0)")
        axes[1].set_title("Aspect (0\u00b0=East, 90\u00b0=North)")
        axes[1].set_xlabel("Easting (m)")
        axes[1].set_ylabel("Northing (m)")
        fig.suptitle(
            "USGS 3DEP LiDAR — Slope and Aspect Analysis, Eastern Kentucky\n"
            f"Resolution: {args.resolution} m  |  Mean slope: {valid_slope.mean():.1f}\u00b0  |  "
            f"Max slope: {valid_slope.max():.1f}\u00b0",
            fontsize=11,
            fontweight="bold",
        )
        _out_dir = Path(__file__).parent.parent / "outputs"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out_png = _out_dir / "slope_aspect_analysis.png"
        save_figure(
            fig,
            out_png,
            alt_text=(
                "Two-panel figure showing slope magnitude (left, inferno colormap) and "
                "aspect direction (right, HSV colormap) derived from Kentucky LiDAR terrain."
            ),
        )
        logger.info("Plot saved to %s", out_png)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
