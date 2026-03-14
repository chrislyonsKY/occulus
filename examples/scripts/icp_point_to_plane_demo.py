"""ICP Point-to-Plane Registration Demo (Synthetic Data).

Generates two synthetic point clouds with a known rigid transform, runs
icp_point_to_plane (which uses surface normals and typically converges faster
than point-to-point ICP), and compares convergence speed and final accuracy
against the ground truth.

Point-to-plane ICP minimises the distance from each source point to the
tangent plane of its nearest target point, which typically converges in fewer
iterations than point-to-point ICP on smooth surfaces.

Data source
-----------
Fully synthetic — no download required.

Usage
-----
    python examples/scripts/icp_point_to_plane_demo.py
    python examples/scripts/icp_point_to_plane_demo.py --n-points 8000 --no-viz
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_surface(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth sinusoidal surface point cloud.

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
    xy = rng.uniform(-3.0, 3.0, (n, 2))
    z = 0.5 * np.sin(xy[:, 0]) * np.cos(xy[:, 1])
    z += rng.normal(0, 0.005, n)
    return np.column_stack([xy[:, 0], xy[:, 1], z]).astype(np.float32)


def main() -> None:
    """Run the ICP point-to-plane demo."""
    parser = argparse.ArgumentParser(description="ICP point-to-plane demo")
    parser.add_argument("--n-points", type=int, default=6000)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.normals import estimate_normals
    from occulus.registration import icp_point_to_plane, icp_point_to_point
    from occulus.types import PointCloud

    rng = np.random.default_rng(7)

    # Ground-truth transform
    angle = np.radians(12.0)
    R_gt = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ])
    t_gt = np.array([0.25, -0.15, 0.05])

    source_xyz = _make_surface(args.n_points, rng)
    target_xyz = (R_gt @ source_xyz.T).T + t_gt
    target_xyz += rng.normal(0, args.noise, target_xyz.shape).astype(np.float32)

    src_raw = voxel_downsample(PointCloud(xyz=source_xyz), voxel_size=0.05)
    tgt_raw = voxel_downsample(PointCloud(xyz=target_xyz), voxel_size=0.05)
    logger.info("Source: %d pts | Target: %d pts", src_raw.n_points, tgt_raw.n_points)

    radius = 0.15

    # Estimate normals for point-to-plane
    src_n = estimate_normals(src_raw, radius=radius)
    tgt_n = estimate_normals(tgt_raw, radius=radius)

    # -- Point-to-point (for comparison) -------------------------------------
    logger.info("Running ICP point-to-point…")
    t0 = time.perf_counter()
    res_p2p = icp_point_to_point(
        src_raw, tgt_raw,
        max_correspondence_distance=0.2,
        max_iterations=200,
    )
    t_p2p = time.perf_counter() - t0

    # -- Point-to-plane -------------------------------------------------------
    logger.info("Running ICP point-to-plane…")
    t0 = time.perf_counter()
    res_p2pl = icp_point_to_plane(
        src_n, tgt_n,
        max_correspondence_distance=0.2,
        max_iterations=200,
    )
    t_p2pl = time.perf_counter() - t0

    # -- Ground-truth error ---------------------------------------------------
    T_gt = np.eye(4)
    T_gt[:3, :3] = R_gt
    T_gt[:3, 3] = t_gt

    def _t_err(res):
        return np.linalg.norm(res.transformation - T_gt)

    print("\n=== ICP Convergence Comparison ===")
    print(f"{'Metric':<30} {'Point-to-Point':>18} {'Point-to-Plane':>18}")
    print("-" * 68)
    print(f"{'Fitness':<30} {res_p2p.fitness:>18.4f} {res_p2pl.fitness:>18.4f}")
    print(f"{'Inlier RMSE (m)':<30} {res_p2p.inlier_rmse:>18.6f} {res_p2pl.inlier_rmse:>18.6f}")
    print(f"{'Iterations':<30} {res_p2p.n_iterations:>18d} {res_p2pl.n_iterations:>18d}")
    print(f"{'Wall time (s)':<30} {t_p2p:>18.4f} {t_p2pl:>18.4f}")
    print(f"{'Transform error (Frobenius)':<30} {_t_err(res_p2p):>18.6f} {_t_err(res_p2pl):>18.6f}")
    print(f"{'Converged':<30} {str(res_p2p.converged):>18} {str(res_p2pl.converged):>18}")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_registration
            logger.info("Opening Open3D viewer (point-to-plane result)…")
            visualize_registration(src_n, tgt_n, res_p2pl,
                                   window_name="ICP Point-to-Plane")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
