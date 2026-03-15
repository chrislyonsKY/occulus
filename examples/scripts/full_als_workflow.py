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

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    out = dest / Path(url).name
    if out.exists():
        return out
    logger.info("Downloading demo tile…")
    try:
        urllib.request.urlretrieve(url, str(out))
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
    path = args.input or _fetch(_DEMO_URL, cache)

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
            chm, xe, ye = canopy_height_model(classified, resolution=1.0)
            cov = coverage_statistics(classified, resolution=1.0)
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

    # Output image
    if chm is not None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from _plot_style import CMAP_CANOPY, apply_report_style, save_figure

            apply_report_style()
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(
                chm, origin="lower", cmap=CMAP_CANOPY, extent=[xe[0], xe[-1], ye[0], ye[-1]]
            )
            plt.colorbar(im, ax=ax, label="Canopy height (m)")
            ax.set_title(
                "Full ALS Workflow — Canopy Height Model\n"
                f"Points: {cloud.n_points:,}  |  Trees: {seg.n_segments if seg else 'N/A'}"
            )
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
            img_out = args.output_dir / "full_als_chm.png"
            save_figure(
                fig,
                img_out,
                alt_text=(
                    "Canopy height model raster from the full ALS workflow, showing tree "
                    "canopy heights across an eastern Kentucky forested tile."
                ),
            )
            logger.info("CHM image → %s", img_out)
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
