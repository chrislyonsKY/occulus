"""Change Detection via ICP Residual Analysis.

Downloads a USGS 3DEP tile as the "before" survey, applies a synthetic
displacement field to simulate terrain change (e.g. erosion, construction),
then aligns the two clouds with ICP and computes per-point residuals as a
change proxy.  Prints an RMSE summary and a simple histogram of residuals.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019, public domain.
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      USGS_LPC_KY_Metro_Louisville_B2_2019/laz/
      USGS_LPC_KY_Metro_Louisville_B2_2019_e1275n4200.laz

Usage
-----
    python examples/scripts/change_detection.py
    python examples/scripts/change_detection.py --input path/to/cloud.las
    python examples/scripts/change_detection.py --no-viz
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
    """Run the change detection demo."""
    parser = argparse.ArgumentParser(description="Occulus change detection demo")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument(
        "--displacement", type=float, default=0.3, help="Max synthetic displacement magnitude (m)"
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    import numpy as np

    from occulus.filters import voxel_downsample
    from occulus.normals import estimate_normals
    from occulus.registration import icp_point_to_plane
    from occulus.types import PointCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from occulus.io import read

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading 'before' cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud_before = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud_before)

    # Downsample for speed
    before_ds = voxel_downsample(cloud_before, voxel_size=args.voxel_size)

    # -- Synthesise "after" with localised displacement ----------------------
    rng = np.random.default_rng(99)
    xyz_after = before_ds.xyz.copy()
    # Apply Gaussian-weighted displacement to a central patch
    cx, cy = before_ds.xyz[:, :2].mean(axis=0)
    r = np.linalg.norm(before_ds.xyz[:, :2] - np.array([cx, cy]), axis=1)
    sigma = r.max() * 0.3
    weight = np.exp(-0.5 * (r / sigma) ** 2)
    dz = args.displacement * weight * rng.choice([-1, 1], size=len(weight))
    xyz_after[:, 2] += dz
    xyz_after += rng.normal(0, 0.01, xyz_after.shape).astype(np.float32)

    after_cloud = PointCloud(xyz=xyz_after)
    logger.info("'After' cloud: %d points", after_cloud.n_points)

    # -- Estimate normals for point-to-plane ICP -----------------------------
    radius = args.voxel_size * 3
    before_n = estimate_normals(before_ds, radius=radius)
    after_n = estimate_normals(after_cloud, radius=radius)

    # -- ICP alignment --------------------------------------------------------
    logger.info("Aligning before/after with ICP…")
    result = icp_point_to_plane(
        after_n,
        before_n,
        max_correspondence_distance=args.voxel_size * 2,
        max_iterations=150,
    )

    print("\n=== ICP Alignment (After → Before) ===")
    print(f"  Fitness    : {result.fitness:.4f}")
    print(f"  Inlier RMSE: {result.inlier_rmse:.4f} m")
    print(f"  Converged  : {result.converged}")

    # -- Residual analysis as change proxy -----------------------------------
    # Apply recovered transform to after cloud and compute nearest-point distances
    T = result.transformation
    np.ones((len(xyz_after), 1), dtype=np.float32)
    xyz_aligned = (T[:3, :3] @ xyz_after.T).T + T[:3, 3]
    residuals = np.linalg.norm(xyz_aligned - before_ds.xyz, axis=1)

    print("\n=== Change Residuals (point-wise distances post-alignment) ===")
    print(f"  Mean residual  : {residuals.mean():.4f} m")
    print(f"  Median residual: {float(np.median(residuals)):.4f} m")
    print(f"  Max residual   : {residuals.max():.4f} m")
    print(f"  RMSE           : {float(np.sqrt(np.mean(residuals**2))):.4f} m")
    print(
        f"  Points > 0.1 m : {(residuals > 0.1).sum():,}  ({(residuals > 0.1).mean() * 100:.1f}%)"
    )

    # Histogram
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0, float("inf")]
    labels = ["0–5 cm", "5–10 cm", "10–20 cm", "20–50 cm", "50–100 cm", ">100 cm"]
    print("\n  Residual distribution:")
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels, strict=False):
        count = int(((residuals >= lo) & (residuals < hi)).sum())
        bar = "#" * min(count // max(len(residuals) // 40, 1), 40)
        print(f"    {lab:>12}: {count:6d}  {bar}")

    if not args.no_viz:
        try:
            from occulus.viz import visualize

            logger.info("Opening Open3D viewer…")
            visualize(before_ds, window_name="Change Detection — Before Cloud")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
