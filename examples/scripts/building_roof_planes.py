"""Building Roof Plane Detection from Urban ALS.

Loads a USGS 3DEP urban ALS tile, removes ground points, and runs plane
detection on the remaining above-ground returns to extract roof segments.
Reports the number of planes detected, their areas, and normal vectors —
useful for solar potential assessment, building reconstruction, and change
detection.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019 (public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/building_roof_planes.py
    python examples/scripts/building_roof_planes.py --input path/to/cloud.las
    python examples/scripts/building_roof_planes.py --no-viz
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
    "USGS_LPC_KY_Metro_Louisville_B2_2019/laz/"
    "USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz"
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
    """Run the building roof plane detection demo."""
    parser = argparse.ArgumentParser(description="Building roof plane detection demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--voxel-size", type=float, default=0.3)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.features import detect_planes
    from occulus.filters import voxel_downsample
    from occulus.io import read
    from occulus.normals import estimate_normals
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud, PointCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading urban tile (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Ground removal -------------------------------------------------------
    logger.info("Classifying ground…")
    classified = classify_ground_csf(cloud)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        above_mask = classified.classification != 2
        above_xyz = classified.xyz[above_mask].astype(np.float32)
    else:
        z_thresh = float(np.percentile(classified.xyz[:, 2], 5)) + 0.5
        above_xyz = classified.xyz[classified.xyz[:, 2] > z_thresh].astype(np.float32)

    logger.info("Above-ground points: %d", len(above_xyz))

    above_cloud = PointCloud(xyz=above_xyz)
    above_ds = voxel_downsample(above_cloud, voxel_size=args.voxel_size)
    logger.info("After downsample: %d points", above_ds.n_points)

    # -- Estimate normals for plane detection ---------------------------------
    above_n = estimate_normals(above_ds, radius=args.voxel_size * 4)

    # -- Plane detection ------------------------------------------------------
    logger.info("Running plane detection…")
    planes = detect_planes(above_n)

    print("\n=== Building Roof Plane Detection ===")
    print(f"  Above-ground points : {above_ds.n_points:,}")
    print(f"  Planes detected     : {len(planes)}")

    if planes:
        areas = [p.area for p in planes if hasattr(p, "area")]
        if areas:
            areas_arr = np.array(areas)
            print("\n  Plane area (m²):")
            print(
                f"    Min / Max / Mean : {areas_arr.min():.1f} / {areas_arr.max():.1f} / {areas_arr.mean():.1f}"
            )

        print("\n  Top 5 planes by inlier count:")
        sorted_planes = sorted(planes, key=lambda p: p.n_inliers, reverse=True)
        for i, plane in enumerate(sorted_planes[:5]):
            n = plane.normal
            tilt = float(np.degrees(np.arccos(np.clip(abs(n[2]), 0, 1))))
            print(
                f"    Plane {i + 1}: {plane.n_inliers:,} pts  "
                f"normal=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f})  tilt={tilt:.1f}°"
            )

        # Identify flat vs pitched roofs (tilt threshold 10°)
        flat = sum(
            1 for p in planes if float(np.degrees(np.arccos(np.clip(abs(p.normal[2]), 0, 1)))) < 10
        )
        print(f"\n  Flat roofs (tilt < 10°) : {flat}")
        print(f"  Pitched roofs           : {len(planes) - flat}")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize

            logger.info("Opening Open3D viewer…")
            visualize(above_n, window_name="Building Roof Planes")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
