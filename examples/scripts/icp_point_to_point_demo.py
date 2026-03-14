"""ICP Point-to-Point Registration Demo (Synthetic Data).

Generates two synthetic point clouds with a known rigid transform (rotation +
translation), runs icp_point_to_point to recover the transform, and verifies
the result by comparing the recovered transformation matrix against the ground
truth.  Reports fitness, RMSE, and transform error metrics.

Data source
-----------
Fully synthetic — no download required.  Ground-truth transform is known so
registration accuracy can be quantitatively verified.

Usage
-----
    python examples/scripts/icp_point_to_point_demo.py
    python examples/scripts/icp_point_to_point_demo.py --n-points 8000 --no-viz
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_bunny(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic 'bunny-like' point cloud from superimposed shapes.

    Parameters
    ----------
    n : int
        Number of points.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        (N, 3) XYZ array.
    """
    # Body: ellipsoid
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = 1.5 * np.sin(theta) * np.cos(phi)
    y = 1.0 * np.sin(theta) * np.sin(phi)
    z = 1.2 * np.cos(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def _apply_transform(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation R and translation t to xyz.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) input points.
    R : np.ndarray
        (3, 3) rotation matrix.
    t : np.ndarray
        (3,) translation vector.

    Returns
    -------
    np.ndarray
        (N, 3) transformed points.
    """
    return (R @ xyz.T).T + t


def main() -> None:
    """Run the ICP point-to-point demo."""
    parser = argparse.ArgumentParser(description="ICP point-to-point demo")
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--noise", type=float, default=0.005, help="Gaussian noise std dev (m)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.registration import icp_point_to_point
    from occulus.types import PointCloud

    rng = np.random.default_rng(42)

    # Ground-truth transform
    angle = np.radians(15.0)
    R_gt = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_gt = np.array([0.3, -0.2, 0.1])

    source_xyz = _make_bunny(args.n_points, rng)
    target_xyz = _apply_transform(source_xyz, R_gt, t_gt)
    target_xyz += rng.normal(0, args.noise, target_xyz.shape).astype(np.float32)

    src = voxel_downsample(PointCloud(xyz=source_xyz), voxel_size=0.05)
    tgt = voxel_downsample(PointCloud(xyz=target_xyz), voxel_size=0.05)
    logger.info("Source: %d pts | Target: %d pts", src.n_points, tgt.n_points)

    # -- ICP point-to-point ---------------------------------------------------
    logger.info("Running ICP point-to-point…")
    result = icp_point_to_point(
        src,
        tgt,
        max_correspondence_distance=0.2,
        max_iterations=200,
    )

    print("\n=== ICP Point-to-Point Result ===")
    print(f"  Fitness      : {result.fitness:.4f}")
    print(f"  Inlier RMSE  : {result.inlier_rmse:.6f} m")
    print(f"  Iterations   : {result.n_iterations}")
    print(f"  Converged    : {result.converged}")
    print(f"\n  Recovered transform:\n{result.transformation}")

    # -- Verify against ground truth -----------------------------------------
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt
    T_err = result.transformation - T_gt
    print("\n=== Ground-Truth Verification ===")
    print(f"  Transform error (Frobenius): {np.linalg.norm(T_err):.6f}")
    print(f"  Translation error          : {np.linalg.norm(T_err[:3, 3]):.6f} m")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_registration

            logger.info("Opening Open3D viewer…")
            visualize_registration(src, tgt, result, window_name="ICP Point-to-Point")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
