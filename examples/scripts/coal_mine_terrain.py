"""Coal Mine Terrain Analysis — Full Occulus Toolkit Demonstration.

Eastern Kentucky has extensive surface coal mining across the Appalachian
Plateau. This script demonstrates EVERY module in the Occulus library on
Kentucky From Above LiDAR data from the coal-producing region:

  1.  I/O         — read LAS/LAZ, write XYZ
  2.  Filters     — voxel downsample, statistical outlier removal
  3.  Normals     — PCA normal estimation + viewpoint orientation
  4.  Segmentation — CSF ground classification, CHM-watershed trees
  5.  Registration — ICP alignment (synthetic displaced copy)
  6.  Features    — eigenvalue geometric descriptors, RANSAC planes
  7.  Metrics     — point density, canopy height model, coverage stats
  8.  Mesh        — Poisson surface reconstruction (ground subset)
  9.  Viz         — 3D visualization (if Open3D available)

Data source: Kentucky From Above / USGS 3DEP — public domain.
https://kyfromabove.ky.gov/

Usage
-----
    python examples/scripts/coal_mine_terrain.py
    python examples/scripts/coal_mine_terrain.py --input path/to/cloud.las
    python examples/scripts/coal_mine_terrain.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# KY From Above — Eastern KY coal region (via USGS 3DEP mirror)
_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/"
    "USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    """Download a tile if not cached.

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
    logger.info("Downloading KY From Above tile…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the full Occulus toolkit demonstration on coal mine terrain."""
    parser = argparse.ArgumentParser(description="Coal mine terrain — full Occulus toolkit demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.15)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    t_start = time.perf_counter()

    # ── 1. I/O — Read ────────────────────────────────────────────────────────
    from occulus.io import read, write

    logger.info("[1/9] Reading point cloud…")
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("  Loaded: %s", cloud)

    # ── 2. Filters — Outlier removal + Voxel downsample ──────────────────────
    from occulus.filters import statistical_outlier_removal, voxel_downsample

    logger.info("[2/9] Filtering — SOR + voxel downsample…")
    clean, _sor_mask = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=2.5)
    n_removed = cloud.n_points - clean.n_points
    ds = voxel_downsample(clean, voxel_size=0.5)
    logger.info(
        "  SOR: %d outliers removed | Voxel: %d → %d pts", n_removed, clean.n_points, ds.n_points
    )

    # ── 3. Normals — PCA estimation ──────────────────────────────────────────
    from occulus.normals import estimate_normals

    logger.info("[3/9] Estimating surface normals…")
    cloud_n = estimate_normals(ds, radius=2.0)
    logger.info("  Normals computed for %d points", cloud_n.n_points)

    # ── 4. Segmentation — Ground classification + Trees ──────────────────────
    from occulus.segmentation import classify_ground_csf, segment_trees
    from occulus.types import AerialCloud

    logger.info("[4/9] CSF ground classification…")
    classified = classify_ground_csf(ds)

    n_ground = 0
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_ground = int((classified.classification == 2).sum())

    seg = None
    try:
        logger.info("[4/9] CHM-watershed tree segmentation…")
        seg = segment_trees(classified, resolution=args.resolution, min_height=3.0)
        logger.info("  Trees detected: %d", seg.n_segments)
    except Exception as exc:
        logger.warning("  Tree segmentation skipped: %s", exc)

    # ── 5. Registration — ICP on synthetic displaced copy ────────────────────
    from occulus.registration import icp

    logger.info("[5/9] ICP registration (synthetic displacement test)…")
    rng = np.random.default_rng(42)
    # Take a small subset for speed
    subset_n = min(5000, ds.n_points)
    idx = rng.choice(ds.n_points, subset_n, replace=False)
    from occulus.types import PointCloud

    source = PointCloud(ds.xyz[idx])
    # Apply known rigid translation
    displaced_xyz = source.xyz.copy()
    displaced_xyz[:, 0] += 1.5  # shift 1.5 m east
    displaced_xyz[:, 1] += 0.8  # shift 0.8 m north
    displaced_xyz[:, 2] += 0.3  # shift 0.3 m up
    target = PointCloud(displaced_xyz)
    icp_result = icp(
        source, target, max_iterations=50, tolerance=1e-6, max_correspondence_distance=5.0
    )
    logger.info(
        "  ICP converged=%s, fitness=%.3f, RMSE=%.4f, iters=%d",
        icp_result.converged,
        icp_result.fitness,
        icp_result.inlier_rmse,
        icp_result.n_iterations,
    )

    # ── 6. Features — Geometric descriptors + RANSAC planes ──────────────────
    from occulus.features import compute_geometric_features, detect_planes

    logger.info("[6/9] Computing geometric features…")
    feats = compute_geometric_features(cloud_n, radius=3.0)
    logger.info(
        "  Planarity: mean=%.3f | Linearity: mean=%.3f | Sphericity: mean=%.3f",
        feats.planarity.mean(),
        feats.linearity.mean(),
        feats.sphericity.mean(),
    )

    logger.info("[6/9] RANSAC plane detection…")
    planes = detect_planes(cloud_n, distance_threshold=0.3, num_iterations=1000, max_planes=3)
    for i, plane in enumerate(planes):
        logger.info(
            "  Plane %d: %d inliers (%.1f%%)",
            i,
            plane.n_inliers,
            plane.n_inliers / cloud_n.n_points * 100,
        )

    # ── 7. Metrics — Density, CHM, Coverage ──────────────────────────────────
    from occulus.metrics import (
        canopy_height_model,
        compute_cloud_statistics,
        coverage_statistics,
        point_density,
    )

    logger.info("[7/9] Computing metrics…")
    stats = compute_cloud_statistics(cloud)
    density_grid, d_xe, d_ye = point_density(ds, resolution=5.0)

    chm = xe = ye = None
    cov = None
    if isinstance(classified, AerialCloud):
        try:
            chm, xe, ye = canopy_height_model(classified, resolution=args.resolution)
            cov = coverage_statistics(classified, resolution=args.resolution)
        except Exception as exc:
            logger.warning("  CHM/coverage skipped: %s", exc)

    # Slope from CHM
    slope_deg = None
    if chm is not None:
        filled = chm.copy()
        valid_mask = ~np.isnan(chm)
        if valid_mask.any():
            filled[~valid_mask] = float(np.nanmean(chm))
        dy, dx = np.gradient(filled, args.resolution, args.resolution)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        slope_deg[~valid_mask] = np.nan

    # ── 8. Mesh — Poisson reconstruction (ground subset) ─────────────────────
    mesh_result = None
    try:
        from occulus.mesh import poisson

        logger.info("[8/9] Poisson surface reconstruction (ground subset)…")
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            ground_mask = classified.classification == 2
            ground_xyz = classified.xyz[ground_mask]
            # Take a manageable subset for meshing
            if len(ground_xyz) > 20000:
                sub_idx = rng.choice(len(ground_xyz), 20000, replace=False)
                ground_xyz = ground_xyz[sub_idx]
            ground_cloud = PointCloud(ground_xyz)
            ground_n = estimate_normals(ground_cloud, radius=3.0)
            mesh_result = poisson(ground_n, depth=8)
            logger.info(
                "  Mesh: %d vertices, %d triangles",
                len(mesh_result.vertices),
                len(mesh_result.triangles),
            )
    except ImportError:
        logger.info("[8/9] open3d not installed — skipping mesh reconstruction.")
    except Exception as exc:
        logger.warning("[8/9] Mesh skipped: %s", exc)

    # ── 9. I/O — Write ground points ─────────────────────────────────────────
    logger.info("[9/9] Exporting classified ground…")
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        ground_mask = classified.classification == 2
        ground_cloud = PointCloud(classified.xyz[ground_mask])
        out_xyz = OUTPUTS / "coal_mine_ground.xyz"
        write(ground_cloud, out_xyz)
        logger.info("  Ground XYZ → %s", out_xyz)

    elapsed = time.perf_counter() - t_start

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   COAL MINE TERRAIN — FULL TOOLKIT RESULTS      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Input points     : {cloud.n_points:,}")
    print(f"  After SOR        : {clean.n_points:,} ({n_removed:,} outliers)")
    print(f"  After voxel ds   : {ds.n_points:,}")
    print(
        f"  Elevation        : {stats.z_min:.1f} – {stats.z_max:.1f} m "
        f"({stats.z_max - stats.z_min:.0f} m relief)"
    )
    print(f"  Ground points    : {n_ground:,} ({n_ground / ds.n_points:.1%})")
    if seg:
        print(f"  Trees detected   : {seg.n_segments}")
    print(f"  ICP fitness      : {icp_result.fitness:.3f} (RMSE: {icp_result.inlier_rmse:.4f})")
    print(f"  Mean planarity   : {feats.planarity.mean():.3f}")
    print(f"  Planes found     : {len(planes)}")
    if cov:
        print(f"  Gap fraction     : {cov.gap_fraction:.2%}")
        print(f"  Mean density     : {cov.mean_density:.1f} pts/m²")
    if slope_deg is not None:
        valid_s = slope_deg[~np.isnan(slope_deg)]
        print(f"  Mean slope       : {valid_s.mean():.1f}°")
        print(f"  Steep (>30°)     : {(valid_s > 30).mean():.1%}")
    if mesh_result is not None:
        print(
            f"  Mesh             : {len(mesh_result.vertices):,} verts, "
            f"{len(mesh_result.triangles):,} tris"
        )
    print(f"  Elapsed          : {elapsed:.1f} s")

    # ── Output image ──────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from _plot_style import (
            CMAP_ELEVATION,
            CMAP_HEAT,
            add_cross_section_line,
            apply_report_style,
            save_figure,
        )

        apply_report_style()
        xyz = ds.xyz

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        ax_plan = fig.add_subplot(gs[0, 0])
        ax_prof = fig.add_subplot(gs[0, 1])
        ax_feat = fig.add_subplot(gs[0, 2])
        ax_slope = fig.add_subplot(gs[1, 0])
        ax_density = fig.add_subplot(gs[1, 1])
        ax_hist = fig.add_subplot(gs[1, 2])

        # Plan view with cross section
        sc = ax_plan.scatter(
            xyz[:, 0],
            xyz[:, 1],
            c=xyz[:, 2],
            cmap=CMAP_ELEVATION,
            s=0.3,
            alpha=0.5,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax_plan, label="Elevation (m)")
        ax_plan.set_title("Terrain Elevation")
        ax_plan.set_xlabel("Easting (m)")
        ax_plan.set_ylabel("Northing (m)")
        add_cross_section_line(
            ax_plan, ax_prof, xyz, y_frac=0.5, band_frac=0.03, label="Cross Section A\u2013A\u2032"
        )

        # Geometric features (planarity)
        sc2 = ax_feat.scatter(
            xyz[:, 0],
            xyz[:, 1],
            c=feats.planarity,
            cmap="RdYlBu_r",
            s=0.3,
            alpha=0.5,
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(sc2, ax=ax_feat, label="Planarity")
        ax_feat.set_title("Geometric Features (Planarity)")
        ax_feat.set_xlabel("Easting (m)")
        ax_feat.set_ylabel("Northing (m)")

        # Slope map
        if slope_deg is not None:
            im = ax_slope.imshow(
                slope_deg,
                origin="lower",
                cmap=CMAP_HEAT,
                extent=[xe[0], xe[-1], ye[0], ye[-1]],
                vmin=0,
                vmax=45,
            )
            plt.colorbar(im, ax=ax_slope, label="Slope (\u00b0)")
            ax_slope.set_title("Slope (OSMRE limit: 20\u00b0)")
        else:
            ax_slope.text(
                0.5, 0.5, "Slope N/A", transform=ax_slope.transAxes, ha="center", va="center"
            )
        ax_slope.set_xlabel("Easting (m)")
        ax_slope.set_ylabel("Northing (m)")

        # Point density
        im2 = ax_density.imshow(
            density_grid.T,
            origin="lower",
            extent=[d_xe[0], d_xe[-1], d_ye[0], d_ye[-1]],
            cmap="Blues",
        )
        plt.colorbar(im2, ax=ax_density, label="Pts / 25 m\u00b2")
        ax_density.set_title("Point Density (5 m grid)")
        ax_density.set_xlabel("Easting (m)")
        ax_density.set_ylabel("Northing (m)")

        # Elevation histogram
        ax_hist.hist(xyz[:, 2], bins=60, color="#8B4513", alpha=0.75, edgecolor="white")
        ax_hist.set_xlabel("Elevation (m)")
        ax_hist.set_ylabel("Point count")
        ax_hist.set_title("Elevation Distribution")

        fig.suptitle(
            "Coal Mine Terrain — Full Occulus Toolkit Demo (KY From Above)\n"
            f"Points: {ds.n_points:,}  |  Relief: {stats.z_max - stats.z_min:.0f} m  |  "
            f"Ground: {n_ground:,}  |  Trees: {seg.n_segments if seg else 'N/A'}  |  "
            f"ICP RMSE: {icp_result.inlier_rmse:.4f}",
            fontsize=10,
            fontweight="bold",
        )
        out = OUTPUTS / "coal_mine_terrain.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Six-panel figure demonstrating every Occulus module on Appalachian coal mine "
                "terrain: elevation plan view with cross-section, elevation profile, geometric "
                "planarity features, slope map for reclamation compliance, point density raster, "
                "and elevation histogram. Data from Kentucky From Above."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(classified, window_name="Coal Mine Terrain — KY From Above")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
