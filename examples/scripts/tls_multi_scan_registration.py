"""TLS Multi-Scan Registration Demo (Synthetic Data).

Simulates three overlapping terrestrial laser scanner (TLS) scans of a simple
scene (a sphere sitting on a plane) captured from different positions.  Each
scan is generated synthetically using numpy, then all three are globally aligned
using occulus.registration.align_scans().  The script reports the global RMSE
and per-pair fitness metrics from the AlignmentResult.

Data source
-----------
Fully synthetic — no download required.  Three point clouds are generated from
known rigid transforms so that registration accuracy can be verified.

Usage
-----
    python examples/scripts/tls_multi_scan_registration.py
    python examples/scripts/tls_multi_scan_registration.py --n-points 5000 --no-viz
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _make_scene(n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic scene: flat plane + sphere.

    Parameters
    ----------
    n_pts : int
        Approximate total number of points.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        (N, 3) XYZ array.
    """
    half = n_pts // 2
    # Flat plane at z=0
    plane_xy = rng.uniform(-5.0, 5.0, (half, 2))
    plane_z = rng.normal(0.0, 0.01, (half, 1))
    plane = np.hstack([plane_xy, plane_z])
    # Sphere of radius 1 centred at (0, 0, 1.5)
    theta = rng.uniform(0, np.pi, (half,))
    phi = rng.uniform(0, 2 * np.pi, (half,))
    r = 1.0 + rng.normal(0, 0.01, (half,))
    sx = r * np.sin(theta) * np.cos(phi)
    sy = r * np.sin(theta) * np.sin(phi)
    sz = r * np.cos(theta) + 1.5
    sphere = np.column_stack([sx, sy, sz])
    return np.vstack([plane, sphere])


def _rigid_transform(xyz: np.ndarray, angle_deg: float, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid Z-axis rotation + translation.

    Parameters
    ----------
    xyz : np.ndarray
        (N, 3) input points.
    angle_deg : float
        Rotation about the Z axis in degrees.
    translation : np.ndarray
        (3,) translation vector.

    Returns
    -------
    np.ndarray
        (N, 3) transformed points.
    """
    a = np.radians(angle_deg)
    R = np.array(
        [
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return (R @ xyz.T).T + translation


def main() -> None:
    """Run the TLS multi-scan registration demo."""
    parser = argparse.ArgumentParser(description="TLS multi-scan registration demo")
    parser.add_argument("--n-points", type=int, default=4000, help="Points per scan (default 4000)")
    parser.add_argument(
        "--voxel-size", type=float, default=0.1, help="Voxel size for downsampling (m)"
    )
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    from occulus.filters import voxel_downsample
    from occulus.normals import estimate_normals
    from occulus.registration import align_scans
    from occulus.types import PointCloud

    rng = np.random.default_rng(0)
    scene = _make_scene(args.n_points, rng)

    # Three scans at different orientations / offsets
    configs = [
        (0.0, np.array([0.0, 0.0, 0.0])),
        (15.0, np.array([0.5, -0.3, 0.0])),
        (30.0, np.array([1.0, 0.5, 0.0])),
    ]

    clouds: list[PointCloud] = []
    for i, (angle, t) in enumerate(configs):
        xyz = _rigid_transform(scene, angle, t)
        # Add noise
        xyz += rng.normal(0, 0.005, xyz.shape)
        cloud = PointCloud(xyz=xyz.astype(np.float32))
        ds = voxel_downsample(cloud, voxel_size=args.voxel_size)
        ds_n = estimate_normals(ds, radius=args.voxel_size * 3)
        clouds.append(ds_n)
        logger.info("Scan %d: %d points (after downsample)", i, ds_n.n_points)

    # -- Multi-scan alignment -------------------------------------------------
    logger.info("Running align_scans on %d TLS scans…", len(clouds))
    result = align_scans(clouds, voxel_size=args.voxel_size)

    print("\n=== Multi-Scan Alignment Result ===")
    print(f"  Global RMSE    : {result.global_rmse:.6f} m")
    print(f"  Scan pairs     : {len(result.pairwise_results)}")
    for idx, pr in enumerate(result.pairwise_results):
        print(f"  Pair {idx}: fitness={pr.fitness:.4f}  inlier_rmse={pr.inlier_rmse:.4f} m")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize

            logger.info("Opening Open3D viewer…")
            visualize(*clouds, window_name="TLS Multi-Scan Registration")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
