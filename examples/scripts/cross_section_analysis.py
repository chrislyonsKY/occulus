"""Cross-Section Extraction and Terrain Profile Analysis — Moab, Utah.

Downloads a USGS 3DEP LiDAR tile from the Moab / Canyonlands area of Utah
and demonstrates cross-section extraction along a user-defined polyline.

The analysis:
  1. Fetches a real LiDAR tile via the USGS TNM API (Moab / Colorado Plateau)
  2. Subsamples to ~20 % for tractable processing
  3. Defines an east-west polyline cutting across the terrain
  4. Extracts a single cross-section profile with ``extract_cross_section()``
  5. Extracts multiple perpendicular profiles at regular intervals with
     ``extract_profiles()``
  6. Generates a 2-panel WCAG 2.1 AA figure:
       (a) Plan view with cross-section polyline (red dashed)
       (b) Elevation profile along the cross-section (station vs elevation)

Data source
-----------
USGS 3DEP — public domain, freely available via the National Map.
Bounding box: -109.7, 38.5, -109.5, 38.7 (Moab / Canyonlands, Utah).

Usage
-----
    python examples/scripts/cross_section_analysis.py
    python examples/scripts/cross_section_analysis.py --input path/to/cloud.laz
    python examples/scripts/cross_section_analysis.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"

# Moab / Canyonlands area, Utah — dramatic canyon terrain
_BBOX = "-109.7,38.5,-109.5,38.7"


def _find_and_fetch(cache: Path, user_input: Path | None) -> Path:
    """Locate a LiDAR tile — from user path, cache, or USGS download.

    Parameters
    ----------
    cache : Path
        Directory for cached downloads.
    user_input : Path or None
        User-supplied local file, if any.

    Returns
    -------
    Path
        Path to the LiDAR file on disk.
    """
    if user_input is not None:
        if not user_input.exists():
            logger.error("Input file not found: %s", user_input)
            sys.exit(1)
        return user_input

    # Import shared download helpers (certifi-backed)
    sys.path.insert(0, str(Path(__file__).parent))
    from _download import download, find_usgs_tile

    logger.info("Querying USGS National Map for Moab-area LiDAR tile…")
    try:
        url = find_usgs_tile(_BBOX)
    except RuntimeError as exc:
        logger.error("TNM query failed: %s", exc)
        sys.exit(1)

    filename = url.rsplit("/", 1)[-1]
    dest = cache / filename
    return download(url, dest, label=f"Moab LiDAR tile ({filename})")


def main() -> None:
    """Run cross-section extraction analysis on Utah canyon terrain."""
    parser = argparse.ArgumentParser(
        description="Cross-section extraction from USGS 3DEP LiDAR — Moab, Utah"
    )
    parser.add_argument("--input", type=Path, default=None, help="Local LAS/LAZ file")
    parser.add_argument(
        "--subsample", type=float, default=0.2, help="Subsample fraction (default: 0.2)"
    )
    parser.add_argument("--width", type=float, default=50.0, help="Corridor half-width (m)")
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Profile station spacing (m)"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip Open3D visualization")
    args = parser.parse_args()

    from occulus.analysis import extract_cross_section, extract_profiles
    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    # ── 1. Acquire data ──────────────────────────────────────────────────
    path = _find_and_fetch(cache, args.input)

    logger.info(
        "Reading point cloud (%.0f%% subsample)…", args.subsample * 100
    )
    cloud = read(path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    stats = compute_cloud_statistics(cloud)
    print("\n=== Moab Canyon Cloud Statistics ===")
    print(f"  Points    : {cloud.n_points:,}")
    print(f"  Z range   : {stats.z_min:.0f} – {stats.z_max:.0f} m")
    print(f"  Relief    : {stats.z_max - stats.z_min:.0f} m")
    print(f"  Z std     : {stats.z_std:.1f} m")

    # ── 2. Define east-west polyline through middle of extent ─────────
    xyz = cloud.xyz
    x_min, x_max = float(xyz[:, 0].min()), float(xyz[:, 0].max())
    y_min, y_max = float(xyz[:, 1].min()), float(xyz[:, 1].max())
    y_mid = (y_min + y_max) / 2.0

    # Simple east-west line through the centre
    polyline = np.array(
        [[x_min, y_mid], [x_max, y_mid]],
        dtype=np.float64,
    )
    logger.info(
        "Cross-section polyline: (%.0f, %.0f) → (%.0f, %.0f), length %.0f m",
        polyline[0, 0], polyline[0, 1],
        polyline[1, 0], polyline[1, 1],
        np.sqrt(((polyline[1] - polyline[0]) ** 2).sum()),
    )

    # ── 3. Extract single cross-section ──────────────────────────────
    logger.info(
        "Extracting cross-section (width=%.0f m, resolution=%.1f m)…",
        args.width, args.resolution,
    )
    section = extract_cross_section(
        cloud, polyline, width=args.width, resolution=args.resolution
    )

    print("\n=== Cross-Section A–A\u2032 ===")
    print(f"  Corridor width  : ±{args.width:.0f} m")
    print(f"  Resolution      : {args.resolution:.1f} m")
    print(f"  Profile points  : {len(section.station):,}")
    if len(section.station) > 0:
        print(f"  Station range   : {section.station[0]:.0f} – {section.station[-1]:.0f} m")
        print(
            f"  Elevation range : {section.elevation.min():.0f}"
            f" – {section.elevation.max():.0f} m"
        )

    # ── 4. Extract multiple perpendicular profiles ───────────────────
    profile_interval = max(50.0, (x_max - x_min) / 10.0)
    logger.info(
        "Extracting perpendicular profiles every %.0f m…", profile_interval
    )
    profiles = extract_profiles(
        cloud,
        polyline,
        interval=profile_interval,
        width=args.width,
        resolution=args.resolution,
    )

    non_empty = sum(1 for p in profiles if len(p.station) > 0)
    print(f"\n=== Perpendicular Profiles ===")
    print(f"  Interval        : {profile_interval:.0f} m")
    print(f"  Total profiles  : {len(profiles)}")
    print(f"  Non-empty       : {non_empty}")

    # ── 5. Plot ──────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(Path(__file__).parent))
        from _plot_style import (
            CMAP_ELEVATION,
            apply_report_style,
            save_figure,
        )

        apply_report_style()

        fig, (ax_plan, ax_profile) = plt.subplots(1, 2, figsize=(14, 6))

        # Panel (a): Plan view with cross-section polyline
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

        # Draw cross-section polyline (red dashed)
        ax_plan.plot(
            polyline[:, 0],
            polyline[:, 1],
            color="#D32F2F",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            label="Cross Section A\u2013A\u2032",
        )

        # Annotate endpoints
        ax_plan.text(
            polyline[0, 0], polyline[0, 1], "  A  ",
            fontsize=9, color="white", fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#D32F2F", ec="#D32F2F", alpha=0.9),
        )
        ax_plan.text(
            polyline[1, 0], polyline[1, 1], "  A\u2032  ",
            fontsize=9, color="white", fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#D32F2F", ec="#D32F2F", alpha=0.9),
        )

        ax_plan.set_title("(a)  Plan View — Moab Canyon Terrain")
        ax_plan.set_xlabel("Easting (m)")
        ax_plan.set_ylabel("Northing (m)")
        ax_plan.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_plan.set_aspect("equal")

        # Panel (b): Elevation profile
        if len(section.station) > 0:
            ax_profile.scatter(
                section.station,
                section.elevation,
                s=1.0,
                c=section.elevation,
                cmap=CMAP_ELEVATION,
                alpha=0.7,
                rasterized=True,
            )
            ax_profile.plot(
                section.station,
                section.elevation,
                color="#222222",
                linewidth=0.5,
                alpha=0.5,
            )
            ax_profile.fill_between(
                section.station,
                section.elevation,
                section.elevation.min(),
                alpha=0.15,
                color="#4A6FA5",
            )
        else:
            ax_profile.text(
                0.5, 0.5, "No data in corridor",
                transform=ax_profile.transAxes, ha="center", fontsize=12,
            )

        ax_profile.set_title("(b)  Elevation Profile — Cross Section A\u2013A\u2032")
        ax_profile.set_xlabel("Station (m)")
        ax_profile.set_ylabel("Elevation (m)")

        # Suptitle
        relief = stats.z_max - stats.z_min
        fig.suptitle(
            "USGS 3DEP LiDAR — Moab / Canyonlands, Utah\n"
            f"Points: {cloud.n_points:,}  |  Relief: {relief:.0f} m  |  "
            f"Profiles: {non_empty}/{len(profiles)}",
            fontsize=11,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out = OUTPUTS / "cross_section_analysis.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Two-panel figure showing cross-section analysis of Moab, Utah "
                "LiDAR terrain. Panel (a) is a plan view colored by elevation with "
                "a red dashed east-west cross-section line through the middle of "
                "the point cloud. Panel (b) shows the elevation profile along the "
                "cross-section with station distance on the x-axis and elevation "
                "on the y-axis, revealing canyon and mesa topography."
            ),
        )
        logger.info("Saved figure → %s", out)
        plt.close()

    except ImportError:
        logger.info("matplotlib not available — skipping figure generation.")

    # ── 6. Optional Open3D visualization ─────────────────────────────
    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(cloud, window_name="Cross-Section Analysis — Moab, Utah")
        except ImportError:
            logger.warning("open3d not installed — skipping 3D visualization.")


if __name__ == "__main__":
    main()
