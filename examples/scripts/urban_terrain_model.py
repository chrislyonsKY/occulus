"""Urban Bare-Earth Digital Terrain Model from USGS 3DEP ALS.

Downloads a USGS 3DEP ALS tile covering a suburban area, classifies ground
returns with CSF, builds a digital terrain model (DTM) using the canopy height
model function on the ground-only cloud, and reports density and elevation
statistics useful for urban planning and flood-risk assessment.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019 (public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/urban_terrain_model.py
    python examples/scripts/urban_terrain_model.py --input path/to/cloud.las
    python examples/scripts/urban_terrain_model.py --resolution 1.0 --no-viz
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
        Local path.
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
    """Run the urban DTM demo."""
    parser = argparse.ArgumentParser(description="Occulus urban terrain model demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument(
        "--resolution", type=float, default=0.5, help="DTM grid resolution in metres (default 0.5)"
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.filters import statistical_outlier_removal
    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics, point_density
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Remove outliers before terrain modelling ----------------------------
    logger.info("Applying statistical outlier removal…")
    cloud_clean, _mask = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=2.0)
    logger.info("  %d → %d points after SOR", cloud.n_points, cloud_clean.n_points)

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud_clean)

    # -- Point density --------------------------------------------------------
    density_grid = point_density(classified, resolution=args.resolution)
    print("\n=== Point Density ===")
    print(f"  Grid shape  : {density_grid.shape}")
    print(f"  Mean density: {float(density_grid[density_grid > 0].mean()):.1f} pts/m²")
    print(f"  Max density : {float(density_grid.max()):.1f} pts/m²")

    # -- DTM (ground-only CHM) ------------------------------------------------
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        ground_mask = classified.classification == 2
        n_ground = int(ground_mask.sum())
        logger.info("Building DTM from %d ground points…", n_ground)

        # Build a ground-only cloud for DTM
        compute_cloud_statistics(classified)
        ground_z = classified.xyz[ground_mask, 2]

        print("\n=== Bare-Earth DTM Statistics ===")
        print(f"  Ground points  : {n_ground:,}")
        print(f"  Elevation min  : {ground_z.min():.2f} m")
        print(f"  Elevation max  : {ground_z.max():.2f} m")
        print(f"  Elevation mean : {ground_z.mean():.2f} m")
        print(f"  Elevation std  : {ground_z.std():.2f} m")
        print(f"  Relief (range) : {ground_z.max() - ground_z.min():.2f} m")

        # Compute percentiles
        p10, p50, p90 = np.percentile(ground_z, [10, 50, 90])
        print(f"  P10 / P50 / P90: {p10:.2f} / {p50:.2f} / {p90:.2f} m")
    else:
        print("\n  (classification not available)")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            import numpy as np

            from occulus.viz import visualize_segments

            if isinstance(classified, AerialCloud) and classified.classification is not None:
                labels = np.where(classified.classification == 2, 0, 1).astype("int32")
                logger.info("Opening Open3D viewer…")
                visualize_segments(classified, labels, window_name="Urban DTM — Ground vs Above")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
