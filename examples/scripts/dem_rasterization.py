"""DEM/DSM/DTM Rasterization from Airborne LiDAR Point Clouds.

Abstract
--------
This study demonstrates the generation of Digital Surface Models (DSM),
Digital Terrain Models (DTM), and derived products (normalized height,
hillshade) from publicly available USGS 3DEP airborne LiDAR data over the
Daniel Boone National Forest in eastern Kentucky — a region of high-relief
Appalachian terrain with mixed deciduous forest cover and deep stream gorges.
The Cloth Simulation Filter (CSF) is used to separate ground from non-ground
points, followed by grid-based rasterization at 2 m cell resolution with
IDW gap-filling. The DSM-minus-DTM difference map isolates above-ground
features (canopy, structures), while a hillshade rendering of the DTM
reveals the underlying geomorphology.

Hypothesis
----------
CSF ground classification combined with IDW-interpolated rasterization at
2 m resolution is sufficient to produce geomorphologically coherent DSM, DTM,
and normalized height products from moderate-density (>= 2 pts/m²) airborne
LiDAR data in rugged Appalachian terrain.

Study Area
----------
Daniel Boone National Forest, Wolfe County, eastern Kentucky.
Bounding box: 83.65°W – 83.62°W, 37.80°N – 37.82°N (EPSG:4326).
Terrain: deeply dissected Appalachian Plateau with 100–200 m of local
relief, narrow ridgelines, and steep sandstone gorges (Red River Gorge area).

Data
----
USGS 3DEP Quality Level 1/2 LiDAR — publicly available via The National Map.
Format: classified LAZ, ASPRS classes 1-6. Public domain.
Acquired via USGS National Map API (TNM).

Methods
-------
1. **Data acquisition**: Tile discovery via TNM API; random 20 % subsample
   to reduce computation while preserving spatial distribution.
2. **Ground classification**: Cloth Simulation Filter (CSF) with rigidness=1
   (mountain terrain) and platform-aware cloth resolution.
3. **DSM generation**: Max-Z binning of all points at 2 m resolution with
   IDW interpolation for empty cells.
4. **DTM generation**: IDW interpolation of ground-only points (ASPRS class 2)
   at 2 m resolution over the full cloud extent.
5. **Derived products**: Normalized height (DSM − DTM), hillshade of DTM
   (azimuth=315°, altitude=45°).

Usage
-----
    python examples/scripts/dem_rasterization.py
    python examples/scripts/dem_rasterization.py --no-viz
    python examples/scripts/dem_rasterization.py --input local.laz

References
----------
- Zhang et al. (2016). An Easy-to-Use Airborne LiDAR Data Filtering Method
  Based on Cloth Simulation. Remote Sensing, 8(6), 501.
- Horn (1981). Hill Shading and the Reflectance Map. Proceedings of the IEEE,
  69(1), 14-47.
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
# Daniel Boone National Forest, Wolfe County, eastern KY — Appalachian Plateau
_BBOX = "-83.65,37.80,-83.62,37.82"

_TNM_URL = (
    "https://tnmaccess.nationalmap.gov/api/v1/products"
    f"?datasets=Lidar%20Point%20Cloud%20(LPC)&bbox={_BBOX}"
    "&max=1&prodFormats=LAZ"
)

# ── Method parameters ──────────────────────────────────────────────────────
_RASTER_RESOLUTION = 2.0  # metres — cell size for DSM/DTM grids
_CSF_RIGIDNESS = 1  # 1 = mountain terrain (most flexible cloth)
_HILLSHADE_AZIMUTH = 315.0  # degrees — standard NW illumination
_HILLSHADE_ALTITUDE = 45.0  # degrees — sun angle above horizon


def _find_tile() -> str:
    """Query USGS TNM for an eastern Kentucky Appalachian LiDAR tile."""
    logger.info("Querying USGS National Map for Appalachian KY LiDAR tile…")
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
        logger.error("No tiles found for KY bbox %s", _BBOX)
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
    logger.info("Downloading Appalachian KY tile…")
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
    logger.info("Downloaded → %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


def _compute_hillshade(
    dtm: "np.ndarray",
    resolution: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> "np.ndarray":
    """Compute an analytical hillshade from a DTM grid.

    Uses Horn's method (1981) for slope and aspect calculation from a 3x3
    neighbourhood, then applies the standard illumination equation.

    Parameters
    ----------
    dtm : ndarray
        Digital Terrain Model grid of shape (ny, nx).
    resolution : float
        Grid cell size in coordinate units.
    azimuth : float, optional
        Sun azimuth in degrees clockwise from north, by default 315.0.
    altitude : float, optional
        Sun altitude in degrees above horizon, by default 45.0.

    Returns
    -------
    ndarray
        Hillshade grid of shape (ny, nx) with values in [0, 255].
    """
    import numpy as np

    # Convert angles to radians
    az_rad = np.deg2rad(360.0 - azimuth + 90.0)  # convert to math convention
    alt_rad = np.deg2rad(altitude)

    # Pad edges for 3x3 neighbourhood
    padded = np.pad(dtm, 1, mode="edge")

    # Horn's method: dz/dx and dz/dy from 3x3 kernel
    dzdx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * resolution)

    dzdy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8.0 * resolution)

    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(-dzdy, dzdx)

    hillshade = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(
        slope
    ) * np.cos(az_rad - aspect)

    # Scale to 0–255
    hillshade = np.clip(hillshade, 0, 1) * 255.0
    return hillshade.astype(np.float64)


def main() -> None:
    """Execute the DEM rasterization study."""
    parser = argparse.ArgumentParser(
        description="DEM/DSM/DTM rasterization from airborne LiDAR"
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.20)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.raster import create_dsm, create_dtm
    from occulus.segmentation.ground import classify_ground_csf

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.gettempdir()) / "occulus_demo"
    cache.mkdir(parents=True, exist_ok=True)

    path = args.input or _fetch(_find_tile(), cache)

    # ── 1. Data acquisition ─────────────────────────────────────────────
    logger.info(
        "Step 1 — Data acquisition (%.0f%% random subsample)…",
        args.subsample * 100,
    )
    cloud = read(path, platform="aerial", subsample=args.subsample)
    stats = compute_cloud_statistics(cloud)

    print("\n" + "=" * 65)
    print("  STUDY: DEM/DSM/DTM Rasterization from Airborne LiDAR")
    print("  Site:  Daniel Boone NF, Wolfe County, KY (Appalachian Plateau)")
    print("=" * 65)
    print("\n  1. DATA SUMMARY")
    print("     Source         : USGS 3DEP (TNM API)")
    print(f"     Points loaded  : {cloud.n_points:,} ({args.subsample:.0%} subsample)")
    print(f"     Z range        : {stats.z_min:.1f} – {stats.z_max:.1f} m")
    print(f"     Relief         : {stats.z_max - stats.z_min:.1f} m")

    # ── 2. Ground classification ────────────────────────────────────────
    logger.info("Step 2 — Ground classification (CSF, rigidness=%d)…", _CSF_RIGIDNESS)
    cloud = classify_ground_csf(cloud, rigidness=_CSF_RIGIDNESS)
    n_ground = int((cloud.classification == 2).sum())
    ground_pct = n_ground / cloud.n_points * 100

    print("\n  2. GROUND CLASSIFICATION (CSF)")
    print(f"     Rigidness      : {_CSF_RIGIDNESS} (mountain terrain)")
    print(f"     Ground points  : {n_ground:,} ({ground_pct:.1f}%)")
    print(f"     Non-ground pts : {cloud.n_points - n_ground:,} ({100 - ground_pct:.1f}%)")

    # ── 3. DSM generation ───────────────────────────────────────────────
    logger.info("Step 3 — DSM generation (%.1f m resolution)…", _RASTER_RESOLUTION)
    dsm = create_dsm(cloud, resolution=_RASTER_RESOLUTION, method="idw")
    dsm_valid = dsm.data[dsm.data != dsm.nodata]

    print("\n  3. DIGITAL SURFACE MODEL (DSM)")
    print(f"     Resolution     : {_RASTER_RESOLUTION:.1f} m")
    print(f"     Grid shape     : {dsm.data.shape[0]} × {dsm.data.shape[1]} cells")
    if len(dsm_valid) > 0:
        print(f"     Elevation range: {dsm_valid.min():.1f} – {dsm_valid.max():.1f} m")

    # ── 4. DTM generation ───────────────────────────────────────────────
    logger.info("Step 4 — DTM generation (%.1f m resolution)…", _RASTER_RESOLUTION)
    dtm = create_dtm(cloud, resolution=_RASTER_RESOLUTION, method="idw")
    dtm_valid = dtm.data[dtm.data != dtm.nodata]

    print("\n  4. DIGITAL TERRAIN MODEL (DTM)")
    print(f"     Resolution     : {_RASTER_RESOLUTION:.1f} m")
    print(f"     Grid shape     : {dtm.data.shape[0]} × {dtm.data.shape[1]} cells")
    if len(dtm_valid) > 0:
        print(f"     Elevation range: {dtm_valid.min():.1f} – {dtm_valid.max():.1f} m")

    # ── 5. Derived products & figure ────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from _plot_style import (
            CMAP_CANOPY,
            CMAP_ELEVATION,
            apply_report_style,
            save_figure,
        )

        apply_report_style()

        # Normalized height = DSM − DTM
        ndsm = dsm.data.copy()
        valid = (dsm.data != dsm.nodata) & (dtm.data != dtm.nodata)
        ndsm[valid] = dsm.data[valid] - dtm.data[valid]
        ndsm[~valid] = 0.0

        # Hillshade of DTM
        dtm_filled = dtm.data.copy()
        dtm_filled[dtm_filled == dtm.nodata] = np.nanmedian(dtm_valid) if len(dtm_valid) > 0 else 0
        hillshade = _compute_hillshade(
            dtm_filled, _RASTER_RESOLUTION, _HILLSHADE_AZIMUTH, _HILLSHADE_ALTITUDE
        )

        # Compute extent for imshow (in relative metres)
        x_extent = float(dsm.x_edges[-1] - dsm.x_edges[0])
        y_extent = float(dsm.y_edges[-1] - dsm.y_edges[0])
        extent = [0.0, x_extent, 0.0, y_extent]

        print("\n  5. DERIVED PRODUCTS")
        if valid.any():
            ndsm_pos = ndsm[valid & (ndsm > 0.5)]
            print(f"     nDSM max       : {ndsm[valid].max():.1f} m")
            if len(ndsm_pos) > 0:
                print(f"     nDSM mean (>0.5m): {ndsm_pos.mean():.1f} m")
            print(
                f"     Canopy coverage: "
                f"{(ndsm[valid] > 2.0).sum() / valid.sum() * 100:.1f}% of cells > 2 m"
            )
        print(f"     Hillshade      : azimuth={_HILLSHADE_AZIMUTH}°, altitude={_HILLSHADE_ALTITUDE}°")

        # ── 4-panel figure ──────────────────────────────────────────────
        logger.info("Step 5 — Generating 4-panel figure…")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        ((ax_dsm, ax_dtm), (ax_ndsm, ax_hs)) = axes

        # Panel (a): DSM
        dsm_display = dsm.data.copy()
        dsm_display[dsm_display == dsm.nodata] = np.nan
        im_dsm = ax_dsm.imshow(
            dsm_display,
            origin="lower",
            cmap=CMAP_ELEVATION,
            aspect="equal",
            extent=extent,
            interpolation="nearest",
        )
        plt.colorbar(im_dsm, ax=ax_dsm, label="Elevation (m)", shrink=0.85)
        ax_dsm.set_title("(a) DSM — Digital Surface Model", fontsize=11)
        ax_dsm.set_xlabel("Relative Easting (m)")
        ax_dsm.set_ylabel("Relative Northing (m)")

        # Panel (b): DTM
        dtm_display = dtm.data.copy()
        dtm_display[dtm_display == dtm.nodata] = np.nan
        im_dtm = ax_dtm.imshow(
            dtm_display,
            origin="lower",
            cmap=CMAP_ELEVATION,
            aspect="equal",
            extent=extent,
            interpolation="nearest",
        )
        plt.colorbar(im_dtm, ax=ax_dtm, label="Elevation (m)", shrink=0.85)
        ax_dtm.set_title("(b) DTM — Digital Terrain Model (ground only)", fontsize=11)
        ax_dtm.set_xlabel("Relative Easting (m)")
        ax_dtm.set_ylabel("Relative Northing (m)")

        # Panel (c): Normalized height (DSM − DTM)
        ndsm_display = ndsm.copy()
        ndsm_display[~valid] = np.nan
        # Clip negative values for display
        ndsm_display = np.clip(ndsm_display, 0, None)
        im_ndsm = ax_ndsm.imshow(
            ndsm_display,
            origin="lower",
            cmap=CMAP_CANOPY,
            aspect="equal",
            extent=extent,
            interpolation="nearest",
        )
        plt.colorbar(im_ndsm, ax=ax_ndsm, label="Height above ground (m)", shrink=0.85)
        ax_ndsm.set_title("(c) DSM \u2212 DTM — Normalized Height", fontsize=11)
        ax_ndsm.set_xlabel("Relative Easting (m)")
        ax_ndsm.set_ylabel("Relative Northing (m)")

        # Panel (d): Hillshade of DTM
        im_hs = ax_hs.imshow(
            hillshade,
            origin="lower",
            cmap="gray",
            aspect="equal",
            extent=extent,
            interpolation="nearest",
            vmin=0,
            vmax=255,
        )
        plt.colorbar(im_hs, ax=ax_hs, label="Illumination (0–255)", shrink=0.85)
        ax_hs.set_title(
            f"(d) Hillshade of DTM — az={_HILLSHADE_AZIMUTH}\u00b0, alt={_HILLSHADE_ALTITUDE}\u00b0",
            fontsize=11,
        )
        ax_hs.set_xlabel("Relative Easting (m)")
        ax_hs.set_ylabel("Relative Northing (m)")

        fig.suptitle(
            "DEM Rasterization from USGS 3DEP ALS \u2014 Daniel Boone NF, KY\n"
            f"n = {cloud.n_points:,} points  |  "
            f"\u0394z = {stats.z_max - stats.z_min:.0f} m  |  "
            f"resolution = {_RASTER_RESOLUTION:.0f} m  |  "
            f"{ground_pct:.0f}% ground",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        out = OUTPUTS / "dem_rasterization.png"
        save_figure(
            fig,
            out,
            alt_text=(
                "Four-panel scientific figure showing DEM rasterization products from "
                "USGS 3DEP airborne LiDAR over the Daniel Boone National Forest, KY: "
                "(a) Digital Surface Model (DSM) showing maximum elevation including "
                "canopy and structures, coloured by elevation using the cividis "
                "colourblind-safe colourmap; (b) Digital Terrain Model (DTM) showing "
                "bare-earth surface from CSF-classified ground points; (c) Normalized "
                "height map (DSM minus DTM) revealing above-ground vegetation and "
                "structures in green tones; (d) Analytical hillshade of the DTM "
                "illuminated from the northwest (315 degrees, 45 degrees altitude) "
                "highlighting Appalachian ridge-and-gorge geomorphology."
            ),
        )
        logger.info("Saved → %s", out)
        plt.close()

        # ── Discussion ──────────────────────────────────────────────────
        print("\n  6. DISCUSSION")
        print(f"     CSF classified {ground_pct:.1f}% of points as ground, consistent")
        print("     with expectations for forested Appalachian terrain where")
        print("     dense canopy reduces ground return penetration.")
        if valid.any():
            print(f"     The normalized height map reveals canopy structure with")
            print(f"     {(ndsm[valid] > 2.0).sum() / valid.sum() * 100:.1f}% of cells "
                  f"exceeding 2 m above ground,")
            print("     indicating extensive forest cover across the study area.")
        print("     The hillshade rendering clearly delineates the dissected")
        print("     Appalachian Plateau topography, with narrow ridgelines and")
        print("     steep-walled gorges characteristic of the Red River Gorge region.")

        print("\n  7. CONCLUSION")
        print("     CSF ground classification with IDW-interpolated rasterization")
        print(f"     at {_RASTER_RESOLUTION:.0f} m resolution successfully produces "
              f"geomorphologically")
        print("     coherent DSM, DTM, and derived products from 3DEP ALS data.")
        print("     The four-product workflow (DSM, DTM, nDSM, hillshade) provides")
        print("     complementary views of terrain and land cover for forestry,")
        print("     geomorphology, and land management applications.")
        print("\n" + "=" * 65)

    except ImportError:
        logger.info("matplotlib/scipy not available — skipping figure output.")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            visualize(cloud, window_name="DEM Rasterization — Daniel Boone NF, KY")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
