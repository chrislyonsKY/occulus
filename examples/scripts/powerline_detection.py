"""Powerline Detection from Synthetic Transmission Corridor LiDAR.

Generates a realistic synthetic point cloud representing a 1 km power
transmission corridor and runs the Occulus powerline detection pipeline.
The synthetic scene contains:

  - Ground plane with gentle slope (1000 m x 200 m)
  - Vegetation clusters (heights 5--20 m, randomised placement)
  - Three high-voltage transmission pylons at x = 0, 500, 1000 m
  - Three catenary wire conductors strung between pylons (15--20 m,
    ~2 m sag)
  - Gaussian measurement noise

The detection algorithm separates ground, computes height-above-ground
and geometric features (linearity / verticality), then classifies wire
and pylon points.  A 2-panel figure shows:

  (a) Side view (XZ) coloured by detection class
  (b) Plan view (XY) with wire segments and pylon positions

Data source
-----------
Fully synthetic -- no external downloads required.

Usage
-----
    python examples/scripts/powerline_detection.py
    python examples/scripts/powerline_detection.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic scene parameters
# ---------------------------------------------------------------------------
_RNG_SEED = 42
_CORRIDOR_LENGTH = 1000.0  # metres (X axis)
_CORRIDOR_WIDTH = 200.0  # metres (Y axis)
_GROUND_SLOPE = 0.005  # 0.5 % grade along X
_GROUND_DENSITY = 0.5  # pts/m^2
_VEG_CLUSTERS = 40  # number of vegetation clusters
_VEG_PTS_PER_CLUSTER = (80, 250)  # point count range per cluster
_VEG_HEIGHT_RANGE = (5.0, 20.0)
_VEG_RADIUS = 8.0  # horizontal spread of each cluster
_PYLON_X = [0.0, 500.0, 1000.0]
_PYLON_HEIGHT = 20.0  # metres
_PYLON_PTS = 200  # points per pylon
_WIRE_ATTACH_HEIGHT = 18.0  # height at attachment points
_WIRE_SAG = 2.0  # metres of sag at midspan
_WIRE_Y_OFFSETS = [-3.0, 0.0, 3.0]  # three conductors
_WIRE_PTS_PER_SPAN = 400  # per conductor per span
_NOISE_SIGMA = 0.05  # metres (instrument noise)


def _generate_ground(rng: np.random.Generator) -> np.ndarray:
    """Generate a ground plane with gentle slope and random sampling.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Ground points as (N, 3) float64 array.
    """
    n_ground = int(_CORRIDOR_LENGTH * _CORRIDOR_WIDTH * _GROUND_DENSITY)
    x = rng.uniform(0, _CORRIDOR_LENGTH, n_ground)
    y = rng.uniform(0, _CORRIDOR_WIDTH, n_ground)
    z = _GROUND_SLOPE * x + rng.normal(0, 0.02, n_ground)
    return np.column_stack([x, y, z])


def _generate_vegetation(rng: np.random.Generator) -> np.ndarray:
    """Generate random vegetation clusters.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Vegetation points as (N, 3) float64 array.
    """
    all_pts: list[np.ndarray] = []
    for _ in range(_VEG_CLUSTERS):
        cx = rng.uniform(50, _CORRIDOR_LENGTH - 50)
        cy = rng.uniform(10, _CORRIDOR_WIDTH - 10)
        n_pts = rng.integers(_VEG_PTS_PER_CLUSTER[0], _VEG_PTS_PER_CLUSTER[1])
        dx = rng.normal(0, _VEG_RADIUS / 2, n_pts)
        dy = rng.normal(0, _VEG_RADIUS / 2, n_pts)
        heights = rng.uniform(_VEG_HEIGHT_RANGE[0], _VEG_HEIGHT_RANGE[1], n_pts)
        # Scale heights by distance from cluster centre (taper)
        dist = np.sqrt(dx**2 + dy**2)
        taper = np.clip(1.0 - dist / _VEG_RADIUS, 0.2, 1.0)
        z_ground = _GROUND_SLOPE * (cx + dx)
        z = z_ground + heights * taper
        pts = np.column_stack([cx + dx, cy + dy, z])
        all_pts.append(pts)
    return np.vstack(all_pts) if all_pts else np.empty((0, 3))


def _catenary_z(x: np.ndarray, x_start: float, x_end: float, h_attach: float, sag: float) -> np.ndarray:
    """Compute catenary Z values between two attachment points.

    Parameters
    ----------
    x : np.ndarray
        Horizontal positions along span.
    x_start : float
        X coordinate of the first pylon.
    x_end : float
        X coordinate of the second pylon.
    h_attach : float
        Height at attachment points.
    sag : float
        Vertical sag at midspan.

    Returns
    -------
    np.ndarray
        Z coordinates following a parabolic sag approximation.
    """
    span = x_end - x_start
    t = (x - x_start) / span  # normalised 0..1
    # Parabolic sag: z = h_attach - 4*sag*t*(1-t)
    return h_attach - 4.0 * sag * t * (1.0 - t)


def _generate_wires(rng: np.random.Generator) -> np.ndarray:
    """Generate three conductors between each pair of pylons.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Wire points as (N, 3) float64 array.
    """
    all_pts: list[np.ndarray] = []
    for i in range(len(_PYLON_X) - 1):
        x_start, x_end = _PYLON_X[i], _PYLON_X[i + 1]
        for y_off in _WIRE_Y_OFFSETS:
            x = np.linspace(x_start + 5, x_end - 5, _WIRE_PTS_PER_SPAN)
            y_centre = _CORRIDOR_WIDTH / 2 + y_off
            y = y_centre + rng.normal(0, 0.1, _WIRE_PTS_PER_SPAN)
            z_ground = _GROUND_SLOPE * x
            z = z_ground + _catenary_z(x, x_start, x_end, _WIRE_ATTACH_HEIGHT, _WIRE_SAG)
            all_pts.append(np.column_stack([x, y, z]))
    return np.vstack(all_pts) if all_pts else np.empty((0, 3))


def _generate_pylons(rng: np.random.Generator) -> np.ndarray:
    """Generate vertical pylon structures.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Pylon points as (N, 3) float64 array.
    """
    all_pts: list[np.ndarray] = []
    for px in _PYLON_X:
        py = _CORRIDOR_WIDTH / 2
        z_base = _GROUND_SLOPE * px
        z = np.linspace(z_base, z_base + _PYLON_HEIGHT, _PYLON_PTS)
        x = px + rng.normal(0, 0.15, _PYLON_PTS)
        y = py + rng.normal(0, 0.15, _PYLON_PTS)
        all_pts.append(np.column_stack([x, y, z]))
    return np.vstack(all_pts) if all_pts else np.empty((0, 3))


def _build_scene() -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Build the full synthetic scene.

    Returns
    -------
    tuple
        (xyz, classification, n_ground, n_veg, n_wire, n_pylon) where xyz
        is (N, 3) float64 and classification is (N,) uint8 with ground=2,
        vegetation=3, wire=14, pylon=15.
    """
    rng = np.random.default_rng(_RNG_SEED)

    ground = _generate_ground(rng)
    veg = _generate_vegetation(rng)
    wires = _generate_wires(rng)
    pylons = _generate_pylons(rng)

    n_ground = len(ground)
    n_veg = len(veg)
    n_wire = len(wires)
    n_pylon = len(pylons)

    xyz = np.vstack([ground, veg, wires, pylons]).astype(np.float64)

    # Add instrument noise
    xyz += rng.normal(0, _NOISE_SIGMA, xyz.shape)

    # Build classification array (ASPRS codes)
    classification = np.zeros(len(xyz), dtype=np.uint8)
    offset = 0
    classification[offset : offset + n_ground] = 2  # ground
    offset += n_ground
    classification[offset : offset + n_veg] = 3  # low/med/high vegetation
    offset += n_veg
    classification[offset : offset + n_wire] = 14  # wire conductor
    offset += n_wire
    classification[offset : offset + n_pylon] = 15  # tower

    return xyz, classification, n_ground, n_veg, n_wire, n_pylon


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# WCAG 2.1 AA palette — all colours have >= 4.5:1 contrast on white
_CLR_GROUND = "#888888"  # gray
_CLR_VEG = "#2E7D32"  # dark green
_CLR_WIRE = "#C62828"  # dark red
_CLR_PYLON = "#E65100"  # dark orange
_CLR_UNCLASSIFIED = "#424242"  # dark gray


def _plot_results(
    xyz: np.ndarray,
    truth_class: np.ndarray,
    result: object,
    output_path: Path,
) -> None:
    """Generate a 2-panel figure showing detection results.

    Parameters
    ----------
    xyz : np.ndarray
        Full point cloud (N, 3).
    truth_class : np.ndarray
        Ground-truth classification (N,) uint8.
    result : PowerlineResult
        Detection result from ``detect_powerlines()``.
    output_path : Path
        Output PNG path.
    """
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _plot_style import apply_report_style, save_figure

    apply_report_style()

    fig, (ax_side, ax_plan) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Panel (a): Side view (XZ) coloured by detection -----------------------
    # Assign colours based on detection result
    colors = np.full(len(xyz), _CLR_UNCLASSIFIED, dtype=object)
    colors[truth_class == 2] = _CLR_GROUND
    colors[truth_class == 3] = _CLR_VEG

    # Override with detection results
    colors[result.wire_mask] = _CLR_WIRE
    colors[result.pylon_mask] = _CLR_PYLON

    # Downsample ground for cleaner plot
    ground_idx = np.where(truth_class == 2)[0]
    keep_ground = ground_idx[::10]
    non_ground_idx = np.where(truth_class != 2)[0]
    plot_idx = np.concatenate([keep_ground, non_ground_idx])
    plot_idx.sort()

    ax_side.scatter(
        xyz[plot_idx, 0],
        xyz[plot_idx, 2],
        c=colors[plot_idx],
        s=0.3,
        alpha=0.6,
        rasterized=True,
    )
    ax_side.set_xlabel("Along-corridor distance (m)")
    ax_side.set_ylabel("Elevation (m)")
    ax_side.set_title("(a) Side view (XZ) -- Detection classes", fontsize=11)

    # Legend with explicit markers (information not conveyed by colour alone)
    legend_entries = [
        ("Ground", _CLR_GROUND, "s"),
        ("Vegetation", _CLR_VEG, "^"),
        ("Wires (detected)", _CLR_WIRE, "-"),
        ("Pylons (detected)", _CLR_PYLON, "D"),
    ]
    for label, clr, marker in legend_entries:
        if marker == "-":
            ax_side.plot([], [], color=clr, linewidth=2, label=label)
        else:
            ax_side.scatter([], [], c=clr, marker=marker, s=30, label=label)
    ax_side.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # -- Panel (b): Plan view (XY) with wire segments and pylon positions ------
    # Light background of all points
    ax_plan.scatter(
        xyz[plot_idx, 0],
        xyz[plot_idx, 1],
        c="#CCCCCC",
        s=0.1,
        alpha=0.3,
        rasterized=True,
    )

    # Overlay detected wires
    if result.wire_mask.any():
        wire_pts = xyz[result.wire_mask]
        ax_plan.scatter(
            wire_pts[:, 0],
            wire_pts[:, 1],
            c=_CLR_WIRE,
            s=0.8,
            alpha=0.7,
            rasterized=True,
            label="Wire points",
        )

    # Mark pylon positions
    if len(result.pylon_positions) > 0:
        ax_plan.scatter(
            result.pylon_positions[:, 0],
            result.pylon_positions[:, 1],
            c=_CLR_PYLON,
            s=120,
            marker="D",
            edgecolors="#222222",
            linewidths=0.8,
            zorder=5,
            label="Pylon centroids",
        )
        for i, pos in enumerate(result.pylon_positions):
            ax_plan.annotate(
                f"P{i + 1}",
                (pos[0], pos[1]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                fontweight="bold",
                color="#222222",
            )

    ax_plan.set_xlabel("Along-corridor distance (m)")
    ax_plan.set_ylabel("Across-corridor distance (m)")
    ax_plan.set_title("(b) Plan view (XY) -- Wire segments and pylons", fontsize=11)
    ax_plan.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_plan.set_aspect("equal", adjustable="datalim")

    fig.tight_layout(pad=1.5)

    alt_text = (
        "Two-panel figure showing powerline detection results on synthetic "
        "LiDAR data. Panel (a) is a side view (XZ) with ground in gray, "
        "vegetation in green, detected wires in red, and pylons in orange. "
        "Panel (b) is a plan view (XY) showing wire segment traces and "
        "diamond markers at detected pylon centroids across a 1 km "
        "transmission corridor."
    )
    save_figure(fig, output_path, alt_text=alt_text)
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the powerline detection demo on synthetic data."""
    parser = argparse.ArgumentParser(
        description="Powerline detection on synthetic transmission corridor LiDAR"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip interactive Open3D visualisation",
    )
    parser.add_argument(
        "--linearity-threshold",
        type=float,
        default=0.7,
        help="Linearity threshold for wire classification (default: 0.7)",
    )
    args = parser.parse_args()

    from occulus.segmentation.powerlines import detect_powerlines
    from occulus.types import PointCloud

    # -- Build synthetic scene ------------------------------------------------
    logger.info("Generating synthetic transmission corridor scene...")
    xyz, classification, n_ground, n_veg, n_wire, n_pylon = _build_scene()

    logger.info(
        "Scene: %d total points  (ground=%d, vegetation=%d, wires=%d, pylons=%d)",
        len(xyz),
        n_ground,
        n_veg,
        n_wire,
        n_pylon,
    )

    # Create PointCloud with ground classification only (class 2)
    # The detector should find wires/pylons from the non-ground points
    detection_class = np.zeros(len(xyz), dtype=np.uint8)
    detection_class[classification == 2] = 2  # only ground is pre-classified
    cloud = PointCloud(xyz=xyz, classification=detection_class)
    logger.info("PointCloud created: %s", cloud)

    # -- Run detection --------------------------------------------------------
    logger.info("Running detect_powerlines()...")
    result = detect_powerlines(
        cloud,
        min_height_above_ground=3.0,
        max_height_above_ground=50.0,
        linearity_threshold=args.linearity_threshold,
        catenary_fit=True,
        min_clearance=7.0,
    )

    # -- Report results -------------------------------------------------------
    n_wire_det = int(result.wire_mask.sum())
    n_pylon_det = int(result.pylon_mask.sum())

    print("\n=== Powerline Detection Results ===")
    print(f"  Total points          : {cloud.n_points:,}")
    print(f"  Wire points detected  : {n_wire_det:,}")
    print(f"  Pylon points detected : {n_pylon_det:,}")
    print(f"  Wire segments         : {len(result.wire_segments)}")
    print(f"  Pylon clusters        : {len(result.pylon_positions)}")
    print(f"  Clearance violations  : {len(result.clearance_violations)}")

    if result.wire_segments:
        fitted = [s for s in result.wire_segments if s.rmse < float("inf")]
        print(f"\n  Catenary fits: {len(fitted)} / {len(result.wire_segments)}")
        if fitted:
            rmses = np.array([s.rmse for s in fitted])
            print(f"    RMSE range: {rmses.min():.3f} -- {rmses.max():.3f} m")

    if len(result.pylon_positions) > 0:
        print("\n  Detected pylon positions:")
        for i, pos in enumerate(result.pylon_positions):
            print(f"    Pylon {i + 1}: X={pos[0]:.1f}  Y={pos[1]:.1f}  Z={pos[2]:.1f}")

    # Truth comparison (since we know the labels)
    true_wire = classification == 14
    true_pylon = classification == 15
    wire_tp = int((result.wire_mask & true_wire).sum())
    wire_fp = int((result.wire_mask & ~true_wire).sum())
    pylon_tp = int((result.pylon_mask & true_pylon).sum())
    pylon_fp = int((result.pylon_mask & ~true_pylon).sum())
    wire_recall = wire_tp / max(int(true_wire.sum()), 1) * 100
    pylon_recall = pylon_tp / max(int(true_pylon.sum()), 1) * 100

    print("\n  Detection accuracy (vs. synthetic truth):")
    print(f"    Wire recall   : {wire_recall:.1f}%  ({wire_tp} TP, {wire_fp} FP)")
    print(f"    Pylon recall  : {pylon_recall:.1f}%  ({pylon_tp} TP, {pylon_fp} FP)")

    # -- Figure ---------------------------------------------------------------
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "powerline_detection.png"

    logger.info("Generating figure...")
    _plot_results(xyz, classification, result, output_path)

    # -- Interactive visualisation --------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize

            logger.info("Opening Open3D viewer...")
            visualize(cloud, window_name="Powerline Detection")
        except ImportError:
            logger.warning("open3d not installed -- skipping interactive visualisation.")


if __name__ == "__main__":
    main()
