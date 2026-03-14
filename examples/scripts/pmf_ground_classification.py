"""PMF Ground Classification on USGS 3DEP Urban Tile (Nashville, TN).

Demonstrates the Progressive Morphological Filter (PMF) approach to ground
classification on a dense urban ALS tile.  PMF iteratively applies morphological
opening at increasing window sizes and is often better suited to flat urban areas
than CSF, which is optimised for complex terrain.

Data source
-----------
USGS 3D Elevation Program (3DEP) — public domain, freely available.
Tile: USGS LPC KY Metro Louisville B2 2019 (used as urban ALS proxy)
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/pmf_ground_classification.py
    python examples/scripts/pmf_ground_classification.py --input path/to/cloud.las
    python examples/scripts/pmf_ground_classification.py --no-viz
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
    """Download a remote LiDAR tile to *dest*, returning the local path.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    dest : Path
        Directory where the file will be written.

    Returns
    -------
    Path
        Local path of the downloaded file.
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
    """Run the PMF ground classification demo."""
    parser = argparse.ArgumentParser(description="Occulus PMF ground classification demo")
    parser.add_argument(
        "--input", type=Path, default=None, help="Path to a local LAS/LAZ file (skips download)"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.3, help="Read-time subsample fraction (default 0.3)"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip Open3D visualization")
    args = parser.parse_args()

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_pmf
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- Cloud statistics -----------------------------------------------------
    stats = compute_cloud_statistics(cloud)
    print("\n=== Cloud Statistics ===")
    print(f"  Points   : {cloud.n_points:,}")
    print(f"  Z range  : {stats.z_min:.2f} – {stats.z_max:.2f} m")
    print(f"  Z mean   : {stats.z_mean:.2f} m")
    print(f"  Z std    : {stats.z_std:.2f} m")

    # -- PMF ground classification --------------------------------------------
    logger.info("Running PMF ground classification…")
    classified = classify_ground_pmf(cloud)

    if isinstance(classified, AerialCloud) and classified.classification is not None:
        import numpy as np

        n_ground = int((classified.classification == 2).sum())
        n_other = cloud.n_points - n_ground
        pct = n_ground / cloud.n_points * 100
        print("\n=== PMF Ground Classification ===")
        print(f"  Ground points : {n_ground:,}  ({pct:.1f}%)")
        print(f"  Other points  : {n_other:,}  ({100 - pct:.1f}%)")
        ground_z = classified.xyz[classified.classification == 2, 2]
        print(f"  Ground Z range: {ground_z.min():.2f} – {ground_z.max():.2f} m")
    else:
        print("\nPMF classification complete.")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            import numpy as np

            from occulus.viz import visualize_segments

            labels = (
                np.where(classified.classification == 2, 0, 1).astype("int32")
                if isinstance(classified, AerialCloud) and classified.classification is not None
                else None
            )
            if labels is not None:
                logger.info("Opening Open3D viewer…")
                visualize_segments(classified, labels, window_name="PMF Ground Classification")
            else:
                from occulus.viz import visualize

                visualize(classified, window_name="PMF Ground Classification")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
