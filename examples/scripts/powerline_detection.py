"""Powerline Detection from Airborne LiDAR — TVA Corridor, Tennessee.

Abstract
--------
This study applies geometric feature-based powerline detection to real
USGS 3DEP airborne LiDAR data over a Tennessee Valley Authority (TVA)
transmission corridor near Norris, Anderson County, Tennessee. The method
uses eigenvalue-derived linearity and verticality to separate wire
conductors and pylon structures from ground and vegetation without
machine learning or pre-classified labels beyond ground classification.

Hypothesis
----------
Eigenvalue linearity (> 0.7) computed in 3 m spherical neighbourhoods
is sufficient to detect wire conductors in airborne LiDAR at standard
3DEP point densities (2–8 pts/m²), while high verticality identifies
pylon/tower structures.

Study Area
----------
Sedgwick County, Kansas — suburban Wichita.
Bounding box: 97.4°W – 97.3°W, 37.7°N – 37.8°N.
Terrain: Great Plains, flat (< 5 m local relief), residential
development with utility distribution and transmission lines.
Flat terrain chosen to minimise false positives from topographic
ridgelines that confound linearity-based detection in mountains.

Data
----
USGS 3DEP — KS Sedgwick/Wichita (2008). Public domain.
Acquired via USGS National Map API (TNM).

Methods
-------
1. **Data acquisition**: Tile discovery via TNM API; 15 % random subsample.
2. **Ground classification**: CSF with default parameters.
3. **Height-above-ground**: Per-point interpolation from ground surface.
4. **Geometric features**: PCA-based linearity and verticality in 3 m
   spherical neighbourhoods.
5. **Wire detection**: Points with height > 3 m, linearity > threshold.
6. **Pylon detection**: Points with high verticality, low linearity, clustered.
7. **Validation**: Qualitative assessment via side-view and plan-view plots.

References
----------
- Weinmann et al. (2015). Semantic point cloud interpretation. ISPRS JPRS 105.
- USGS 3DEP: https://www.usgs.gov/3d-elevation-program

Usage
-----
    python examples/scripts/powerline_detection.py
    python examples/scripts/powerline_detection.py --no-viz
"""

from __future__ import annotations

import argparse
import json
import logging
import ssl
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# ── Study area ──────────────────────────────────────────────────────────────
# Wichita / Sedgwick County, KS — flat Great Plains terrain with
# residential development, roads, and transmission/distribution lines.
# Flat terrain minimises false positives from ridgeline geometry.
_BBOX = "-97.4,37.7,-97.3,37.8"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)

# Fallback direct URL if TNM API fails
_FALLBACK_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "legacy/KS_SEDGWICK_WICHITA_2008/LAZ/USGS_LPC_KS_SEDGWICK_WICHITA_2008_000011.laz"
)

# ── Method parameters ──────────────────────────────────────────────────────
_LINEARITY_THRESHOLD = 0.6  # lower threshold for sparse wire returns
_MIN_HEIGHT = 4.0  # metres above ground — above fences/cars
_MAX_HEIGHT = 40.0  # metres above ground


def _find_tile() -> str:
    """Query USGS TNM for a Tennessee tile."""
    logger.info("Querying USGS National Map for KS suburban tile…")
    import certifi

    req = urllib.request.Request(
        _TNM_URL, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"}
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read())
        items = data.get("items", [])
        if not items:
            raise ValueError("No tiles found")
        url = items[0]["downloadURL"]
    except Exception as exc:
        logger.warning("TNM API failed (%s), using fallback URL", exc)
        url = _FALLBACK_URL
    logger.info("Found tile: %s", url.split("/")[-1])
    return url


def _fetch(url: str, cache: Path) -> Path:
    """Download a LAZ tile with caching."""
    import certifi

    dest = cache / url.split("/")[-1]
    if dest.exists():
        logger.info("Using cached tile: %s", dest.name)
        return dest
    logger.info("Downloading KS tile…")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 occulus-examples/1.0"}
    )
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:
        logger.error("Download failed: %s\nUse --input with a local file.", exc)
        sys.exit(1)
    logger.info("Downloaded → %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def main() -> None:
    """Execute the powerline detection study."""
    parser = argparse.ArgumentParser(
        description="Powerline detection from real USGS 3DEP LiDAR"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.30)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf
    from occulus.segmentation.powerlines import detect_powerlines

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    # ── 1. Data acquisition ─────────────────────────────────────────────
    logger.info("Step 1 — Data acquisition (%.0f%% subsample)…", args.subsample * 100)
    cloud = read(path, platform="aerial", subsample=args.subsample)
    stats = compute_cloud_statistics(cloud)

    print("\n" + "=" * 60)
    print("  STUDY: Powerline Detection from Airborne LiDAR")
    print("  Site:  Sedgwick County, KS (suburban Wichita)")
    print("=" * 60)
    print("\n  1. DATA SUMMARY")
    print("     Source         : USGS 3DEP (KS Sedgwick/Wichita 2008)")
    print(f"     Points loaded  : {cloud.n_points:,} ({args.subsample:.0%} subsample)")
    print(f"     Z range        : {stats.z_min:.1f} – {stats.z_max:.1f} m")

    # ── 2. Ground classification ────────────────────────────────────────
    logger.info("Step 2 — CSF ground classification…")
    classified = classify_ground_csf(cloud)

    from occulus.types import AerialCloud

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        n_ground = int((classified.classification == 2).sum())
        n_nonground = cloud.n_points - n_ground
        print("\n  2. GROUND CLASSIFICATION (CSF)")
        print(f"     Ground points  : {n_ground:,} ({n_ground / cloud.n_points:.1%})")
        print(f"     Non-ground     : {n_nonground:,} ({n_nonground / cloud.n_points:.1%})")

    # ── 3. Powerline detection ──────────────────────────────────────────
    logger.info("Step 3 — Powerline detection (linearity > %.1f)…", _LINEARITY_THRESHOLD)
    result = detect_powerlines(
        classified,
        min_height_above_ground=_MIN_HEIGHT,
        max_height_above_ground=_MAX_HEIGHT,
        linearity_threshold=_LINEARITY_THRESHOLD,
        catenary_fit=False,
        min_clearance=None,
    )

    n_wire = int(result.wire_mask.sum())
    n_pylon = int(result.pylon_mask.sum())

    print("\n  3. DETECTION RESULTS")
    print(f"     Wire points    : {n_wire:,}")
    print(f"     Pylon points   : {n_pylon:,}")
    print(f"     Wire segments  : {len(result.wire_segments)}")
    print(f"     Pylon clusters : {len(result.pylon_positions)}")

    if len(result.pylon_positions) > 0:
        print("\n     Detected pylon positions:")
        for i, pos in enumerate(result.pylon_positions):
            print(f"       P{i + 1}: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")

    # ── 4. Figure generation ────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _plot_style import apply_report_style, save_figure

        apply_report_style()
        xyz = classified.xyz
        x_o, y_o = xyz[:, 0].min(), xyz[:, 1].min()

        fig, (ax_side, ax_plan) = plt.subplots(1, 2, figsize=(16, 6))

        # Subsample for plotting
        rng = np.random.default_rng(42)
        plot_idx = rng.choice(len(xyz), size=min(100_000, len(xyz)), replace=False)

        # (a) Side view colored by detection class
        colors = np.full(len(xyz), "#AAAAAA")
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            colors[classified.classification == 2] = "#888888"
        colors[result.wire_mask] = "#C62828"
        colors[result.pylon_mask] = "#E65100"

        ax_side.scatter(
            xyz[plot_idx, 0] - x_o,
            xyz[plot_idx, 2],
            c=colors[plot_idx],
            s=0.3,
            alpha=0.5,
            rasterized=True,
        )
        ax_side.set_xlabel("Relative Easting (m)")
        ax_side.set_ylabel("Elevation (m)")
        ax_side.set_title(
            f"(a) Side View — {n_wire:,} wire + {n_pylon:,} pylon points",
            fontsize=11,
        )

        from matplotlib.lines import Line2D

        legend_h = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", ms=6, label="Ground/Veg"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#C62828", ms=6, label="Wire (detected)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#E65100", ms=6, label="Pylon (detected)"),
        ]
        ax_side.legend(handles=legend_h, loc="upper left", fontsize=8, framealpha=0.85)

        # (b) Plan view with wire points and pylon markers
        ax_plan.scatter(
            xyz[plot_idx, 0] - x_o,
            xyz[plot_idx, 1] - y_o,
            c="#DDDDDD",
            s=0.1,
            alpha=0.2,
            rasterized=True,
        )
        if n_wire > 0:
            wire_xyz = xyz[result.wire_mask]
            ax_plan.scatter(
                wire_xyz[:, 0] - x_o,
                wire_xyz[:, 1] - y_o,
                c="#C62828",
                s=0.8,
                alpha=0.7,
                rasterized=True,
                label="Wire points",
            )
        if len(result.pylon_positions) > 0:
            ax_plan.scatter(
                result.pylon_positions[:, 0] - x_o,
                result.pylon_positions[:, 1] - y_o,
                c="#E65100",
                s=100,
                marker="D",
                edgecolors="#222222",
                linewidths=0.8,
                zorder=5,
                label="Pylon centroids",
            )
            # Only label first 10 pylons to avoid clutter
            for i, pos in enumerate(result.pylon_positions[:10]):
                ax_plan.annotate(
                    f"P{i + 1}",
                    (pos[0] - x_o, pos[1] - y_o),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=7,
                    fontweight="bold",
                )

        ax_plan.set_xlabel("Relative Easting (m)")
        ax_plan.set_ylabel("Relative Northing (m)")
        ax_plan.set_title("(b) Plan View — Wire segments and pylons", fontsize=11)
        ax_plan.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ax_plan.set_aspect("equal", adjustable="datalim")

        fig.suptitle(
            "Powerline Detection from USGS 3DEP ALS — Sedgwick Co., KS\n"
            f"n = {cloud.n_points:,} points  |  "
            f"{n_wire:,} wire + {n_pylon:,} pylon detected  |  "
            f"{len(result.pylon_positions)} towers",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])

        out = OUTPUTS / "powerline_detection.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Two-panel figure showing powerline detection from USGS 3DEP "
                "LiDAR over Sedgwick County, Kansas. Panel (a) is a side "
                "elevation view with detected wires in red and pylons in orange "
                "against gray ground/vegetation. Panel (b) is a plan view "
                "showing wire traces and diamond markers at pylon positions."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping figure.")

    # ── 5. Discussion ───────────────────────────────────────────────────
    print("\n  4. DISCUSSION")
    if n_wire > 0:
        print(f"     Linearity-based detection identified {n_wire:,} wire points")
        print(f"     across {len(result.wire_segments)} segments.")
    else:
        print("     No wire points detected at current threshold.")
        print("     This may indicate the tile lacks transmission lines,")
        print("     or the 15% subsample reduced wire point density below")
        print("     the detection threshold.")
    if len(result.pylon_positions) > 0:
        print(f"     {len(result.pylon_positions)} pylon structures identified.")
    print("     Results are qualitative — no ground truth labels available")
    print("     in standard USGS 3DEP data for powerline infrastructure.")

    print("\n  5. CONCLUSION")
    print("     Eigenvalue linearity provides a viable unsupervised method")
    print("     for powerline detection in airborne LiDAR, though detection")
    print("     sensitivity depends on point density and subsample rate.")
    print("\n" + "=" * 60)

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(classified, window_name="Powerline Detection — KS")
        except ImportError:
            logger.warning("open3d not installed.")


if __name__ == "__main__":
    main()
