"""End-to-End Aerial LiDAR (ALS) Analysis Workflow.

Demonstrates the complete occulus processing chain for aerial LiDAR data:

  1. Read LAS/LAZ (USGS 3DEP or user-supplied)
  2. Statistical outlier removal
  3. Voxel downsampling
  4. CSF ground classification
  5. Normal estimation
  6. Canopy height model (CHM)
  7. Individual tree segmentation
  8. Coverage statistics (gap fraction)
  9. Eigenvalue geometric features (vegetation vs ground discrimination)
  10. Export classified ground to XYZ

Results are written to examples/outputs/.

Data source: USGS 3DEP — KY Statewide 2019 (public domain).

Usage
-----
    python examples/scripts/full_als_workflow.py
    python examples/scripts/full_als_workflow.py --input path/to/cloud.las
    python examples/scripts/full_als_workflow.py --no-viz --output-dir ./results
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

# Use a different tile than canopy_height_model.py (which uses KY CentralEast)
# North Carolina Piedmont — mixed forest, suburban, varied terrain
_DEMO_BBOX = "-79.1,35.9,-78.9,36.1"
_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_DEMO_BBOX}"
    "&max=1&prodFormats=LAZ"
)


def _find_tile() -> str:
    """Query USGS TNM for a North Carolina Piedmont tile."""
    import json

    logger.info("Querying USGS National Map for NC Piedmont LiDAR tile…")
    req = urllib.request.Request(
        _TNM_URL, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"}
    )
    try:
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        logger.error("TNM API query failed: %s", exc)
        sys.exit(1)
    items = data.get("items", [])
    if not items:
        logger.error("No tiles found for NC bbox %s", _DEMO_BBOX)
        sys.exit(1)
    url = items[0]["downloadURL"]
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, dest: Path) -> Path:
    """Download a LAZ tile with caching."""
    out = dest / Path(url).name
    if out.exists():
        return out
    logger.info("Downloading demo tile…")
    try:
        import ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            out.write_bytes(resp.read())
    except Exception as exc:
        logger.error("Download failed: %s — use --input with a local file.", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the full ALS processing workflow."""
    parser = argparse.ArgumentParser(description="End-to-end ALS workflow")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.25)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from occulus.features import compute_geometric_features
    from occulus.filters import statistical_outlier_removal, voxel_downsample
    from occulus.io import read, write
    from occulus.metrics import canopy_height_model, compute_cloud_statistics, coverage_statistics
    from occulus.normals import estimate_normals
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    # Step 1: Read
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)
    path = args.input or _fetch(_find_tile(), cache)

    logger.info("[1/9] Reading point cloud…")
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("  loaded: %s", cloud)

    # Step 2: Outlier removal
    logger.info("[2/9] Statistical outlier removal…")
    clean, _mask = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=2.5)
    removed = cloud.n_points - clean.n_points
    logger.info("  removed %d outliers (%.1f%%)", removed, removed / cloud.n_points * 100)

    # Step 3: Voxel downsample
    logger.info("[3/9] Voxel downsampling (%.2f m)…", args.voxel_size)
    ds = voxel_downsample(clean, voxel_size=args.voxel_size)
    logger.info("  downsampled: %d → %d", clean.n_points, ds.n_points)

    # Step 4: Ground classification
    logger.info("[4/9] CSF ground classification…")
    classified = classify_ground_csf(ds)

    # Step 5: Normals
    logger.info("[5/9] Estimating normals…")
    ds_n = estimate_normals(ds, radius=args.voxel_size * 4)

    # Step 6: CHM
    logger.info("[6/9] Building canopy height model…")
    chm = cov = None
    if isinstance(classified, AerialCloud):
        try:
            chm, xe, ye = canopy_height_model(classified, resolution=3.0)
            cov = coverage_statistics(classified, resolution=3.0)
        except Exception as exc:
            logger.warning("CHM skipped: %s", exc)

    # Step 7: Tree segmentation
    logger.info("[7/9] Individual tree segmentation…")
    seg = None
    if isinstance(classified, AerialCloud):
        try:
            seg = segment_trees(classified, resolution=2.0, min_height=2.0)
        except Exception as exc:
            logger.warning("Tree segmentation skipped: %s", exc)

    # Step 8: Geometric features
    logger.info("[8/9] Computing eigenvalue geometric features…")
    feats = compute_geometric_features(ds_n, radius=args.voxel_size * 5)

    # Step 9: Export ground points
    logger.info("[9/9] Exporting classified cloud…")
    out_path = args.output_dir / "als_classified_ground.xyz"
    write(classified, out_path)
    logger.info("  ground cloud → %s", out_path)

    # --- Report ---
    stats = compute_cloud_statistics(cloud)
    print("\n╔═══════════════════════════════════════════╗")
    print("║   FULL ALS WORKFLOW — RESULTS SUMMARY    ║")
    print("╚═══════════════════════════════════════════╝")
    print(f"  Input points     : {cloud.n_points:,}")
    print(f"  After SOR        : {clean.n_points:,}")
    print(f"  After voxel ds   : {ds.n_points:,}")
    print(f"  Z range          : {stats.z_min:.1f} – {stats.z_max:.1f} m")

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_g = int((classified.classification == 2).sum())
        print(f"  Ground points    : {n_g:,} ({n_g / ds.n_points:.1%})")

    if chm is not None:
        import numpy as np

        valid = chm[~np.isnan(chm)]
        print(f"  CHM shape        : {chm.shape[0]} × {chm.shape[1]}")
        print(f"  Max canopy ht    : {valid.max():.1f} m" if valid.size else "")
    if cov is not None:
        print(f"  Gap fraction     : {cov.gap_fraction:.2%}")
    if seg is not None:
        print(f"  Trees detected   : {seg.n_segments}")
    print(f"  Mean planarity   : {feats.planarity.mean():.3f}")
    print(f"  Output dir       : {args.output_dir}")

    # Output image — 4-panel pipeline summary
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from _plot_style import (
            CMAP_CANOPY,
            CMAP_ELEVATION,
            CMAP_PROFILE,
            apply_report_style,
            save_figure,
        )

        apply_report_style()
        xyz = ds_n.xyz
        x_o, y_o = xyz[:, 0].min(), xyz[:, 1].min()

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        ax_elev, ax_class, ax_chm, ax_feat = axes.flat

        # (a) Raw elevation
        sc = ax_elev.scatter(
            xyz[:, 0] - x_o, xyz[:, 1] - y_o, c=xyz[:, 2], cmap=CMAP_ELEVATION,
            s=0.3, alpha=0.5, rasterized=True,
        )
        plt.colorbar(sc, ax=ax_elev, label="Elevation (m)")
        ax_elev.set_title("(a) Input Point Cloud — Elevation")
        ax_elev.set_xlabel("Easting (m)")
        ax_elev.set_ylabel("Northing (m)")
        ax_elev.set_aspect("equal")

        # (b) Ground classification
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            cls_xyz = classified.xyz
            cls = classified.classification
            colors = np.where(cls == 2, 0.0, 1.0)
            sc2 = ax_class.scatter(
                cls_xyz[:, 0] - x_o, cls_xyz[:, 1] - y_o, c=colors, cmap="RdYlGn_r",
                s=0.3, alpha=0.5, vmin=0, vmax=1, rasterized=True,
            )
            from matplotlib.lines import Line2D
            legend_h = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a9641", ms=6, label="Ground"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#d7191c", ms=6, label="Non-ground"),
            ]
            ax_class.legend(handles=legend_h, loc="lower right", fontsize=8, framealpha=0.85)
        ax_class.set_title("(b) CSF Ground Classification")
        ax_class.set_xlabel("Easting (m)")
        ax_class.set_ylabel("Northing (m)")
        ax_class.set_aspect("equal")

        # (c) CHM
        if chm is not None:
            im = ax_chm.imshow(
                chm, origin="lower", cmap=CMAP_CANOPY,
                extent=[xe[0] - x_o, xe[-1] - x_o, ye[0] - y_o, ye[-1] - y_o],
            )
            plt.colorbar(im, ax=ax_chm, label="Canopy height (m)")
            tree_count = seg.n_segments if seg else "N/A"
            ax_chm.set_title(f"(c) Canopy Height Model — {tree_count} trees")
        else:
            ax_chm.text(0.5, 0.5, "CHM unavailable", transform=ax_chm.transAxes, ha="center")
            ax_chm.set_title("(c) Canopy Height Model")
        ax_chm.set_xlabel("Easting (m)")
        ax_chm.set_ylabel("Northing (m)")

        # (d) Geometric features — planarity
        order = np.argsort(xyz[:, 0])
        sc3 = ax_feat.scatter(
            xyz[order, 0] - x_o, xyz[order, 2],
            c=feats.planarity[order], cmap=CMAP_PROFILE,
            s=0.3, alpha=0.5, vmin=0, vmax=1, rasterized=True,
        )
        plt.colorbar(sc3, ax=ax_feat, label="Planarity")
        ax_feat.set_title("(d) Eigenvalue Planarity — Side View")
        ax_feat.set_xlabel("Easting (m)")
        ax_feat.set_ylabel("Elevation (m)")

        fig.suptitle(
            "Full ALS Processing Pipeline \u2014 USGS 3DEP\n"
            f"n = {cloud.n_points:,} raw \u2192 {ds.n_points:,} downsampled  |  "
            f"{n_g:,} ground  |  {seg.n_segments if seg else 'N/A'} trees",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        img_out = args.output_dir / "full_als_workflow.png"
        save_figure(
            fig, img_out,
            alt_text=(
                "Four-panel figure showing the full ALS processing pipeline: "
                "(a) raw elevation point cloud, (b) CSF ground classification with "
                "ground (green) and non-ground (red), (c) canopy height model raster "
                "with tree count, (d) eigenvalue planarity side view showing vegetation "
                "and terrain structure."
            ),
        )
        logger.info("Pipeline figure → %s", img_out)
        plt.close()
    except ImportError:
        pass

    if not args.no_viz and seg is not None:
        try:
            from occulus.viz import visualize_segments

            visualize_segments(classified, seg.labels, window_name="Full ALS Workflow")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
