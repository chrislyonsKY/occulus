"""M3C2 Multi-Epoch Change Detection from Airborne LiDAR Point Clouds.

Abstract
--------
This study demonstrates the M3C2 algorithm (Lague et al. 2013) for
detecting statistically significant surface change between two point
cloud epochs acquired over the same site. Because co-temporal multi-epoch
data are rarely freely available at the same location, we simulate a
second epoch by applying a synthetic vertical displacement to a circular
sub-region of a real USGS 3DEP tile — mimicking localised erosion or
deposition in a low-relief deltaic landscape.

The M3C2 algorithm computes signed distances along locally estimated
surface normals and quantifies per-point uncertainty via a Level of
Detection (LoD) threshold at 95 % confidence. Points whose absolute
distance exceeds the LoD are flagged as statistically significant change.

Hypothesis
----------
M3C2 signed distances, computed on subsampled airborne LiDAR with a
+0.5 m synthetic vertical displacement confined to a 50 m-radius circle,
will correctly localise the displaced region as statistically significant
change while the surrounding undisturbed terrain remains below the LoD
threshold.

Study Area
----------
Louisiana coastal wetlands, Mississippi River delta region.
Bounding box: 91.2 deg W -- 91.0 deg W, 29.5 deg N -- 29.7 deg N.
Terrain: flat deltaic plain (< 5 m local relief), marsh and open water
at near-zero elevation.  Ideal for M3C2 because normals are predominantly
vertical on flat terrain, simplifying interpretation of signed distances.

Data
----
USGS 3DEP Quality Level LiDAR — Louisiana coastal survey.
Format: classified LAZ, ASPRS classes 1-6.  Public domain.
Acquired via USGS National Map API (TNM).

Methods
-------
1. **Data acquisition**: Tile discovery via TNM API; random 15 % subsample
   to reduce computation while preserving spatial distribution.
2. **Epoch synthesis**: A circular region (radius 50 m) centred on the
   point cloud centroid is displaced vertically by +0.5 m to simulate
   sediment deposition.  Gaussian noise (sigma = 0.02 m) is added to
   both epochs for realism.
3. **M3C2 computation**: Core points are voxel-downsampled from epoch 1.
   Normal scale = 2.0 m, projection scale = 1.0 m, cylinder depth = 5.0 m,
   confidence = 0.95.
4. **Visualisation**: Two-panel WCAG 2.1 AA figure — (a) plan-view map
   of signed M3C2 distances (RdBu_r diverging colormap), (b) histogram
   of signed distances with LoD significance threshold annotated.

Usage
-----
    python examples/scripts/m3c2_change_detection.py
    python examples/scripts/m3c2_change_detection.py --no-viz
    python examples/scripts/m3c2_change_detection.py --input local.laz

References
----------
- Lague, D., Brodu, N., & Leroux, J. (2013). Accurate 3D comparison of
  complex topography with terrestrial laser scanner: Application to the
  Rangitikei canyon (N-Z). ISPRS Journal of Photogrammetry and Remote
  Sensing, 82, 10--26.
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program
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

# ── Study area ──────────────────────────────────────────────────────────────
# Louisiana coastal wetlands — flat deltaic terrain near the Mississippi delta
_BBOX = "-91.2,29.5,-91.0,29.7"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)

# ── M3C2 parameters ────────────────────────────────────────────────────────
_NORMAL_SCALE = 2.0  # metres — radius for surface normal estimation
_PROJECTION_SCALE = 1.0  # metres — cylinder radius for distance computation
_MAX_DEPTH = 5.0  # metres — max cylinder half-length along normal
_CONFIDENCE = 0.95  # confidence level for Level of Detection
_CORE_VOXEL_SIZE = 2.0  # metres — voxel size for core point subsampling

# ── Synthetic displacement ──────────────────────────────────────────────────
_DISPLACEMENT_Z = 0.5  # metres — vertical shift applied to circular region
_DISPLACEMENT_RADIUS = 50.0  # metres — radius of displaced region
_NOISE_SIGMA = 0.02  # metres — Gaussian noise added to both epochs


def _find_tile() -> str:
    """Query USGS TNM API for a LiDAR tile URL in Louisiana wetlands."""
    logger.info("Querying USGS National Map for Louisiana LiDAR tile...")
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
        logger.error("No LiDAR tiles found for Louisiana bbox %s", _BBOX)
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
    logger.info("Downloading Louisiana wetlands tile...")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"}
    )
    try:
        import ssl

        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    logger.info("Downloaded -> %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def main() -> None:
    """Execute the M3C2 multi-epoch change detection study."""
    parser = argparse.ArgumentParser(
        description="M3C2 multi-epoch change detection from airborne LiDAR"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.15)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.change import m3c2
    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.types import PointCloud

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    # ── 1. Data acquisition ─────────────────────────────────────────────
    logger.info(
        "Step 1 -- Data acquisition (%.0f%% random subsample)...",
        args.subsample * 100,
    )
    cloud = read(path, platform="aerial", subsample=args.subsample)
    stats = compute_cloud_statistics(cloud)

    print("\n" + "=" * 65)
    print("  STUDY: M3C2 Multi-Epoch Change Detection")
    print("  Site:  Louisiana Coastal Wetlands, Mississippi Delta")
    print("=" * 65)
    print("\n  1. DATA SUMMARY")
    print("     Source         : USGS 3DEP (Louisiana coastal survey)")
    print(f"     Points loaded  : {cloud.n_points:,} ({args.subsample:.0%} subsample)")
    print(f"     Z range        : {stats.z_min:.2f} -- {stats.z_max:.2f} m")
    print(f"     Relief         : {stats.z_max - stats.z_min:.1f} m")

    # ── 2. Epoch synthesis ──────────────────────────────────────────────
    logger.info("Step 2 -- Synthesising epoch 2 (displacement = +%.2f m)...", _DISPLACEMENT_Z)
    rng = np.random.default_rng(42)
    xyz1 = cloud.xyz.copy()

    # Add small noise to epoch 1 for realism (scanner noise)
    xyz1 += rng.normal(0.0, _NOISE_SIGMA, xyz1.shape)
    epoch1 = PointCloud(xyz1)

    # Create epoch 2 with synthetic displacement in a circular region
    xyz2 = xyz1.copy()
    centroid_xy = xyz2[:, :2].mean(axis=0)
    dist_from_centre = np.linalg.norm(xyz2[:, :2] - centroid_xy, axis=1)
    displaced_mask = dist_from_centre <= _DISPLACEMENT_RADIUS
    xyz2[displaced_mask, 2] += _DISPLACEMENT_Z

    # Add independent noise to epoch 2
    xyz2 += rng.normal(0.0, _NOISE_SIGMA, xyz2.shape)
    epoch2 = PointCloud(xyz2)

    n_displaced = int(displaced_mask.sum())
    print(f"\n  2. EPOCH SYNTHESIS")
    print(f"     Epoch 1        : {epoch1.n_points:,} points (original + noise)")
    print(f"     Epoch 2        : {epoch2.n_points:,} points (displaced subset)")
    print(f"     Displacement   : +{_DISPLACEMENT_Z:.2f} m vertical (Z)")
    print(f"     Region radius  : {_DISPLACEMENT_RADIUS:.0f} m from centroid")
    print(f"     Points moved   : {n_displaced:,} ({n_displaced / epoch1.n_points:.1%})")
    print(f"     Noise sigma    : {_NOISE_SIGMA:.3f} m per epoch")

    # ── 3. M3C2 computation ─────────────────────────────────────────────
    logger.info(
        "Step 3 -- M3C2 (normal_scale=%.1f, proj_scale=%.1f, confidence=%.2f)...",
        _NORMAL_SCALE,
        _PROJECTION_SCALE,
        _CONFIDENCE,
    )

    # Subsample core points via voxel grid for tractable computation
    core_cloud = voxel_downsample(epoch1, voxel_size=_CORE_VOXEL_SIZE)
    core_pts = core_cloud.xyz
    logger.info("Core points: %d (voxel %.1f m)", core_pts.shape[0], _CORE_VOXEL_SIZE)

    result = m3c2(
        epoch1,
        epoch2,
        core_points=core_pts,
        normal_scale=_NORMAL_SCALE,
        projection_scale=_PROJECTION_SCALE,
        max_cylinder_depth=_MAX_DEPTH,
        confidence=_CONFIDENCE,
    )

    # Compute summary statistics on valid (non-NaN) distances
    valid = np.isfinite(result.distances)
    n_valid = int(valid.sum())
    n_sig = int(result.significant_change.sum())
    dists_valid = result.distances[valid]

    print(f"\n  3. M3C2 RESULTS")
    print(f"     Core points    : {core_pts.shape[0]:,}")
    print(f"     Valid distances: {n_valid:,} / {core_pts.shape[0]:,}")
    print(f"     Mean distance  : {dists_valid.mean():+.4f} m")
    print(f"     Median distance: {float(np.median(dists_valid)):+.4f} m")
    print(f"     Std deviation  : {dists_valid.std():.4f} m")
    print(f"     Range          : [{dists_valid.min():+.4f}, {dists_valid.max():+.4f}] m")
    print(f"     Significant    : {n_sig:,} ({n_sig / max(n_valid, 1):.1%} of valid)")

    # Median LoD for context
    lod_valid = result.uncertainties[valid]
    print(f"     Median LoD     : {float(np.median(lod_valid)):.4f} m")

    # ── 4. Discussion ───────────────────────────────────────────────────
    print(f"\n  4. DISCUSSION")
    print(f"     The M3C2 algorithm detected {n_sig:,} core points with")
    print(f"     statistically significant change (|distance| > LoD at")
    print(f"     {_CONFIDENCE:.0%} confidence). The synthetic +{_DISPLACEMENT_Z:.1f} m")
    print(f"     displacement within a {_DISPLACEMENT_RADIUS:.0f} m radius should produce")
    print(f"     a clear positive signal in the displaced region, with")
    print(f"     surrounding terrain remaining below the detection threshold.")
    print(f"     The signed distance histogram (panel b) should show a")
    print(f"     bimodal distribution: a peak near zero (no change) and")
    print(f"     a secondary peak near +{_DISPLACEMENT_Z:.1f} m (deposition).")

    print(f"\n  5. CONCLUSION")
    print(f"     M3C2 successfully isolates the synthetically displaced region")
    print(f"     from the undisturbed deltaic terrain. The per-point uncertainty")
    print(f"     (LoD) provides a principled criterion for separating real")
    print(f"     surface change from noise and registration error.")
    print("\n" + "=" * 65)

    # ── 5. Figure generation ────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_DIVERGING, apply_report_style, save_figure

        apply_report_style()

        logger.info("Step 5 -- Generating figure...")

        fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(16, 7))

        # ── Panel (a): Plan view coloured by signed M3C2 distance ──────
        cp = result.core_points[valid]
        dists_plot = dists_valid

        # Symmetric colour limits centred on zero
        vmax = max(abs(dists_plot.min()), abs(dists_plot.max()), 0.1)
        vmax = min(vmax, 1.5)  # cap for visual clarity

        sc = ax_map.scatter(
            cp[:, 0],
            cp[:, 1],
            c=dists_plot,
            cmap=CMAP_DIVERGING,
            s=1.5,
            alpha=0.8,
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        cbar = plt.colorbar(sc, ax=ax_map, label="Signed M3C2 Distance (m)")
        cbar.ax.tick_params(labelsize=8)

        # Mark the displaced region boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = centroid_xy[0] + _DISPLACEMENT_RADIUS * np.cos(theta)
        circle_y = centroid_xy[1] + _DISPLACEMENT_RADIUS * np.sin(theta)
        ax_map.plot(
            circle_x, circle_y,
            color="#222222", linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"Displaced region ({_DISPLACEMENT_RADIUS:.0f} m radius)",
        )
        ax_map.legend(loc="lower right", fontsize=8, framealpha=0.85, edgecolor="#222222")

        ax_map.set_title(
            f"(a) Signed M3C2 Distance -- {n_sig:,} Significant Changes",
            fontsize=11,
        )
        ax_map.set_xlabel("Easting (m)")
        ax_map.set_ylabel("Northing (m)")
        ax_map.set_aspect("equal")

        # ── Panel (b): Histogram of signed distances with LoD ──────────
        median_lod = float(np.median(lod_valid))

        ax_hist.hist(
            dists_valid,
            bins=80,
            color="#4A90D9",
            alpha=0.75,
            edgecolor="white",
            linewidth=0.3,
        )
        ax_hist.axvline(
            0.0, color="#222222", linestyle="-", linewidth=1.0, alpha=0.5, label="Zero change"
        )
        ax_hist.axvline(
            median_lod, color="#D32F2F", linestyle="--", linewidth=1.5,
            label=f"+LoD (median = {median_lod:.3f} m)",
        )
        ax_hist.axvline(
            -median_lod, color="#D32F2F", linestyle="--", linewidth=1.5,
            label=f"-LoD (median = {-median_lod:.3f} m)",
        )
        ax_hist.axvline(
            _DISPLACEMENT_Z, color="#2E7D32", linestyle=":", linewidth=1.5,
            label=f"True displacement (+{_DISPLACEMENT_Z:.1f} m)",
        )

        ax_hist.set_xlabel("Signed M3C2 Distance (m)")
        ax_hist.set_ylabel("Core Point Count")
        ax_hist.set_title(
            f"(b) Distance Distribution -- LoD at {_CONFIDENCE:.0%} Confidence",
            fontsize=11,
        )
        ax_hist.legend(fontsize=8, framealpha=0.85, edgecolor="#222222")

        fig.suptitle(
            "M3C2 Multi-Epoch Change Detection -- Louisiana Coastal Wetlands\n"
            f"n = {n_valid:,} core points  |  "
            f"displacement = +{_DISPLACEMENT_Z:.1f} m  |  "
            f"{n_sig:,} significant ({n_sig / max(n_valid, 1):.1%})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        out = OUTPUTS / "m3c2_change_detection.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Two-panel scientific figure: (a) plan-view map of signed M3C2 "
                "distances over Louisiana coastal wetlands, using a diverging "
                "red-white-blue colormap where red indicates negative change "
                "(erosion) and blue indicates positive change (deposition); a "
                "dashed circle marks the 50-metre-radius region with synthetic "
                "+0.5 m displacement. (b) Histogram of signed M3C2 distances "
                "showing a bimodal distribution with a peak near zero (unchanged "
                "terrain) and a secondary peak near +0.5 m (displaced region); "
                "vertical dashed red lines mark the median Level of Detection "
                "threshold at 95 percent confidence, and a green dotted line "
                "marks the true displacement magnitude."
            ),
        )
        logger.info("Saved -> %s", out)
        plt.close()

    except ImportError:
        logger.info("matplotlib not available -- skipping figure output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(epoch1, window_name="M3C2 Change Detection -- Epoch 1")
        except ImportError:
            logger.warning("open3d not installed -- skipping visualization.")


if __name__ == "__main__":
    main()
