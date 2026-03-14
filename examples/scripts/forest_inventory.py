"""Full Forest Inventory Pipeline from ALS Data.

Runs a complete forest inventory workflow on a USGS 3DEP forested ALS tile:
  1. Read and optionally subsample
  2. CSF ground classification
  3. Canopy height model (CHM)
  4. Individual tree segmentation
  5. Coverage statistics
  6. Per-tree stem density, basal area proxy, and canopy cover

This is the kind of workflow used in forest management, carbon stock
estimation, and biodiversity surveys.

Data source
-----------
USGS 3DEP — KY Statewide 2019, public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz

Usage
-----
    python examples/scripts/forest_inventory.py
    python examples/scripts/forest_inventory.py --input path/to/cloud.las
    python examples/scripts/forest_inventory.py --no-viz
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
    """Download tile if not cached.

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
    """Run the full forest inventory pipeline."""
    parser = argparse.ArgumentParser(description="Occulus forest inventory pipeline")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--chm-resolution", type=float, default=0.5)
    parser.add_argument("--min-height", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, coverage_statistics
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading LiDAR tile (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== 1. Cloud Overview ===")
    print(f"  Total points : {cloud.n_points:,}")
    print(f"  Z range      : {stats.z_min:.2f} – {stats.z_max:.2f} m")

    # -- Ground classification ------------------------------------------------
    logger.info("Step 2: CSF ground classification…")
    classified = classify_ground_csf(cloud)
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_ground = int((classified.classification == 2).sum())
        print(f"\n=== 2. Ground Classification ===")
        print(f"  Ground points : {n_ground:,}  ({n_ground/cloud.n_points*100:.1f}%)")

    # -- CHM ------------------------------------------------------------------
    logger.info("Step 3: Building CHM…")
    chm, x_edges, y_edges = canopy_height_model(classified, resolution=args.chm_resolution)
    valid_chm = chm[~np.isnan(chm)]
    tile_area_m2 = (
        (x_edges[-1] - x_edges[0]) * (y_edges[-1] - y_edges[0])
        if len(x_edges) > 1 and len(y_edges) > 1 else 0.0
    )
    print(f"\n=== 3. Canopy Height Model ===")
    print(f"  Grid size     : {chm.shape[0]} × {chm.shape[1]} cells")
    print(f"  Tile area     : {tile_area_m2:.0f} m²  ({tile_area_m2/1e4:.2f} ha)")
    if valid_chm.size > 0:
        print(f"  Max canopy ht : {valid_chm.max():.1f} m")
        print(f"  Mean canopy ht: {valid_chm.mean():.1f} m")

    # -- Coverage statistics --------------------------------------------------
    logger.info("Step 4: Coverage statistics…")
    cov = coverage_statistics(classified, resolution=args.chm_resolution)
    print(f"\n=== 4. Coverage Statistics ===")
    print(f"  Gap fraction   : {cov.gap_fraction:.2%}")
    print(f"  Covered area   : {cov.covered_area:.1f} m²")
    print(f"  Mean density   : {cov.mean_density:.1f} pts/m²")

    # -- Tree segmentation ----------------------------------------------------
    logger.info("Step 5: Individual tree segmentation…")
    seg = segment_trees(classified, resolution=args.chm_resolution, min_height=args.min_height)
    print(f"\n=== 5. Tree Inventory ===")
    print(f"  Trees detected : {seg.n_segments}")

    if seg.n_segments > 0 and tile_area_m2 > 0:
        stems_per_ha = seg.n_segments / (tile_area_m2 / 1e4)
        print(f"  Stem density   : {stems_per_ha:.0f} stems/ha")

        # Per-tree metrics
        heights = []
        for tid in range(seg.n_segments):
            mask = seg.labels == tid
            pts = classified.xyz[mask]
            if len(pts) > 0:
                heights.append(float(pts[:, 2].max() - pts[:, 2].min()))

        if heights:
            h = np.array(heights)
            print(f"  Crown height (m): min={h.min():.1f}  max={h.max():.1f}  mean={h.mean():.1f}")
            # Lorey's mean height (weighted by count, proxy)
            loreys = float(np.mean(h))
            print(f"  Lorey's height proxy: {loreys:.1f} m")

    # -- Output image ---------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_CANOPY, apply_report_style, save_figure
        apply_report_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # CHM
        im = axes[0].imshow(chm, origin="lower", cmap=CMAP_CANOPY,
                            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        plt.colorbar(im, ax=axes[0], label="Canopy height (m)")
        axes[0].set_title("Canopy Height Model — Eastern Kentucky")
        axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")

        # Tree height histogram
        if seg.n_segments > 0:
            heights_arr = []
            for tid in range(seg.n_segments):
                mask = seg.labels == tid
                pts = classified.xyz[mask]
                if len(pts) > 0:
                    heights_arr.append(float(pts[:, 2].max() - pts[:, 2].min()))
            if heights_arr:
                axes[1].hist(heights_arr, bins=30, color="#228B22", alpha=0.75, edgecolor="white")
                axes[1].set_xlabel("Tree height (m)")
                axes[1].set_ylabel("Count")
                axes[1].set_title(f"Tree Height Distribution ({seg.n_segments} trees)")
        else:
            valid_chm = chm[~np.isnan(chm)]
            axes[1].hist(valid_chm, bins=40, color="#228B22", alpha=0.75, edgecolor="white")
            axes[1].set_xlabel("Canopy height (m)"); axes[1].set_ylabel("Cell count")
            axes[1].set_title("CHM Value Distribution")

        fig.suptitle(
            "USGS 3DEP LiDAR — Full Forest Inventory Pipeline, Kentucky\n"
            f"Trees: {seg.n_segments}  |  Gap fraction: {cov.gap_fraction:.2%}",
            fontsize=12, fontweight="bold",
        )
        _out_dir = Path(__file__).parent.parent / "outputs"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out = _out_dir / "forest_inventory.png"
        save_figure(fig, out, alt_text=(
            "Two-panel figure showing forest inventory results: canopy height model "
            "raster (left) and tree height distribution histogram (right) for an "
            "eastern Kentucky forested tile."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments
            logger.info("Opening Open3D viewer (%d trees)…", seg.n_segments)
            visualize_segments(classified, seg.labels,
                               window_name=f"Forest Inventory — {seg.n_segments} trees")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
