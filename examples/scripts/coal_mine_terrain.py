"""Coal Mine Terrain Analysis — Reclamation Monitoring Concept.

Eastern Kentucky (Pike / Letcher counties) has extensive surface coal mining.
ALS surveys are used before and after reclamation to track topographic recovery.
This script demonstrates the concept: loads a USGS 3DEP tile from eastern KY,
classifies ground, computes elevation percentiles as a baseline, then synthesises
a "post-reclamation" cloud by adding a Gaussian displacement field and computes
the change statistics.

Data source
-----------
USGS 3DEP — KY Statewide 2019 (Pike/Letcher county region), public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz

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
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "USGS_LPC_KY_Statewide_2019/laz/USGS_LPC_KY_Statewide_2019_e1380n4170.laz"
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
    logger.info("Downloading USGS 3DEP tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the coal mine terrain reclamation monitoring demo."""
    parser = argparse.ArgumentParser(description="Occulus coal mine terrain demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.3)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading 'before' point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud_before = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud_before)

    # -- Ground classification ------------------------------------------------
    logger.info("Classifying ground with CSF…")
    classified = classify_ground_csf(cloud_before)

    # -- Before statistics ----------------------------------------------------
    stats_before = compute_cloud_statistics(classified)
    ground_z = classified.xyz[:, 2]
    if isinstance(classified, AerialCloud) and classified.classification is not None:
        ground_mask = classified.classification == 2
        ground_z = classified.xyz[ground_mask, 2]

    percentiles = np.percentile(ground_z, [10, 25, 50, 75, 90])
    print("\n=== Before-Reclamation Ground Elevation ===")
    print(f"  Points (ground)  : {len(ground_z):,}")
    print(f"  Z min / max      : {ground_z.min():.2f} / {ground_z.max():.2f} m")
    print(f"  P10 / P50 / P90  : {percentiles[0]:.2f} / {percentiles[2]:.2f} / {percentiles[4]:.2f} m")
    print(f"  Relief           : {ground_z.max() - ground_z.min():.2f} m")

    # -- Synthesise post-reclamation cloud (Gaussian fill / cut simulation) ---
    logger.info("Synthesising post-reclamation cloud…")
    rng = np.random.default_rng(42)
    xyz_after = classified.xyz.copy()
    # Simulate cut-and-fill: raise low areas, lower spoil piles
    # Simple Gaussian perturbation biased upward in low zones
    z_norm = (xyz_after[:, 2] - ground_z.min()) / max(ground_z.ptp(), 1.0)
    displacement = rng.normal(loc=0.5 * (1 - z_norm), scale=0.5, size=len(xyz_after))
    xyz_after[:, 2] += displacement

    after_z = xyz_after[:, 2]
    percentiles_after = np.percentile(after_z, [10, 25, 50, 75, 90])

    print("\n=== After-Reclamation Ground Elevation (Synthetic) ===")
    print(f"  Z min / max      : {after_z.min():.2f} / {after_z.max():.2f} m")
    print(f"  P10 / P50 / P90  : {percentiles_after[0]:.2f} / {percentiles_after[2]:.2f} / {percentiles_after[4]:.2f} m")
    print(f"  Relief           : {after_z.max() - after_z.min():.2f} m")

    # -- Change statistics ----------------------------------------------------
    delta_z = after_z - classified.xyz[:, 2]
    print("\n=== Elevation Change (After − Before) ===")
    print(f"  Mean change  : {delta_z.mean():+.3f} m")
    print(f"  Std change   : {delta_z.std():.3f} m")
    print(f"  Max fill     : {delta_z.max():+.3f} m")
    print(f"  Max cut      : {delta_z.min():+.3f} m")
    print(f"  Net volume   : {delta_z.sum():.1f} m³ (1 m² / pt assumed)")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize
            logger.info("Opening Open3D viewer…")
            visualize(classified, window_name="Coal Mine — Before Reclamation")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
