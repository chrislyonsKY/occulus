"""Building Facade Registration Demo (Synthetic Data).

Generates two synthetic point clouds representing a building facade scanned
from slightly different viewpoints (a common scenario in mobile mapping or
repeated TLS surveys).  Registers them using ICP and reports alignment quality.

The facade is modelled as a flat wall with windows (rectangular cutouts) to
provide distinctive geometric features for registration.

Data source
-----------
Fully synthetic — no download required.

Usage
-----
    python examples/scripts/building_facade_registration.py
    python examples/scripts/building_facade_registration.py --n-points 8000 --no-viz
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_facade(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic building facade with window cutouts.

    Parameters
    ----------
    n : int
        Approximate number of points.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        (N, 3) XYZ array.
    """
    pts: list[np.ndarray] = []
    # Main wall: 10 m wide × 8 m tall at x=0
    while len(pts) == 0 or sum(len(p) for p in pts) < n:
        y = rng.uniform(-5.0, 5.0, n)
        z = rng.uniform(0.0, 8.0, n)
        x = rng.normal(0.0, 0.01, n)
        # Remove points inside window regions
        win_mask = (
            ((np.abs(y - (-2.5)) < 0.8) & (z > 2.0) & (z < 4.0))
            | ((np.abs(y - (2.5)) < 0.8) & (z > 2.0) & (z < 4.0))
            | ((np.abs(y - (-2.5)) < 0.8) & (z > 5.5) & (z < 7.5))
            | ((np.abs(y - (2.5)) < 0.8) & (z > 5.5) & (z < 7.5))
        )
        wall = np.column_stack([x[~win_mask], y[~win_mask], z[~win_mask]])
        pts.append(wall)
        if sum(len(p) for p in pts) >= n:
            break
    return np.vstack(pts)[:n].astype(np.float32)


def main() -> None:
    """Run the building facade registration demo."""
    parser = argparse.ArgumentParser(description="Building facade registration demo")
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.normals import estimate_normals
    from occulus.registration import icp_point_to_plane
    from occulus.types import PointCloud

    rng = np.random.default_rng(3)

    facade = _make_facade(args.n_points, rng)
    logger.info("Facade cloud: %d points", len(facade))

    # Viewpoint 2: small rotation about Y + slight lateral offset
    angle = np.radians(5.0)
    R = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    t = np.array([0.0, 0.2, 0.0])
    facade2 = (R @ facade.T).T + t
    facade2 += rng.normal(0, args.noise, facade2.shape).astype(np.float32)

    src = voxel_downsample(PointCloud(xyz=facade), voxel_size=0.05)
    tgt = voxel_downsample(PointCloud(xyz=facade2), voxel_size=0.05)
    logger.info("Downsampled: src=%d pts, tgt=%d pts", src.n_points, tgt.n_points)

    radius = 0.2
    src_n = estimate_normals(src, radius=radius)
    tgt_n = estimate_normals(tgt, radius=radius)

    logger.info("Running ICP point-to-plane…")
    result = icp_point_to_plane(
        src_n,
        tgt_n,
        max_correspondence_distance=0.3,
        max_iterations=200,
    )

    # Ground-truth transform
    T_gt = np.eye(4)
    T_gt[:3, :3] = R
    T_gt[:3, 3] = t
    t_err = np.linalg.norm(result.transformation - T_gt)

    print("\n=== Building Facade Registration ===")
    print(f"  Fitness           : {result.fitness:.4f}")
    print(f"  Inlier RMSE       : {result.inlier_rmse:.6f} m")
    print(f"  Iterations        : {result.n_iterations}")
    print(f"  Converged         : {result.converged}")
    print(f"  Transform error   : {t_err:.6f} (Frobenius norm vs ground truth)")
    print(f"\n  Recovered transform:\n{result.transformation}")

    if not args.no_viz:
        try:
            from occulus.viz import visualize_registration

            logger.info("Opening Open3D viewer…")
            visualize_registration(src_n, tgt_n, result, window_name="Building Facade Registration")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
