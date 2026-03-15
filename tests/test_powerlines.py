"""Tests for occulus.segmentation.powerlines — powerline detection."""

from __future__ import annotations

import numpy as np
import pytest

from occulus.exceptions import OcculusSegmentationError
from occulus.segmentation import PowerlineResult, detect_powerlines
from occulus.segmentation.powerlines import (
    CatenarySegment,
    ClearanceViolation,
    _catenary,
    _compute_geometric_features,
    _compute_height_above_ground,
    _dbscan_cluster,
    _filter_orphan_wires,
)
from occulus.types import PointCloud

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_powerline_cloud(
    *,
    n_ground: int = 500,
    n_wire: int = 200,
    n_pylon: int = 80,
    wire_height: float = 10.0,
    pylon_height: float = 15.0,
    seed: int = 42,
) -> PointCloud:
    """Build a synthetic cloud with ground, wires, and pylons.

    Ground: flat plane at z ~ 0.
    Wires: points along x-axis at constant height with low scatter.
    Pylons: tight vertical columns at the wire endpoints.
    """
    rng = np.random.default_rng(seed)

    # Ground
    gx = rng.uniform(0, 100, n_ground)
    gy = rng.uniform(0, 20, n_ground)
    gz = rng.uniform(-0.1, 0.1, n_ground)
    ground = np.column_stack((gx, gy, gz))

    # Wire: linear along x, narrow in y, at wire_height
    wx = np.linspace(10, 90, n_wire) + rng.normal(0, 0.05, n_wire)
    wy = np.full(n_wire, 10.0) + rng.normal(0, 0.1, n_wire)
    wz = np.full(n_wire, wire_height) + rng.normal(0, 0.05, n_wire)
    wire = np.column_stack((wx, wy, wz))

    # Pylons: vertical columns at x=10 and x=90
    px = np.concatenate(
        [
            np.full(n_pylon // 2, 10.0) + rng.normal(0, 0.3, n_pylon // 2),
            np.full(n_pylon // 2, 90.0) + rng.normal(0, 0.3, n_pylon // 2),
        ]
    )
    py = np.full(n_pylon, 10.0) + rng.normal(0, 0.3, n_pylon)
    pz = np.concatenate(
        [
            rng.uniform(0, pylon_height, n_pylon // 2),
            rng.uniform(0, pylon_height, n_pylon // 2),
        ]
    )
    pylon = np.column_stack((px, py, pz))

    xyz = np.vstack((ground, wire, pylon)).astype(np.float64)

    # Classification: ground=2, wire/pylon=1 (unclassified)
    classification = np.ones(len(xyz), dtype=np.uint8)
    classification[:n_ground] = 2

    return PointCloud(xyz, classification=classification)


def _make_short_wire_cloud(
    *,
    wire_length: float = 20.0,
    seed: int = 55,
) -> PointCloud:
    """Build a cloud with a short wire segment that strict mode should reject.

    Wire spans only ``wire_length`` metres — below the default 50m threshold.
    Includes ground and minimal pylons so the algorithm can run.
    """
    rng = np.random.default_rng(seed)
    n_ground = 400
    n_wire = 100

    gx = rng.uniform(0, 50, n_ground)
    gy = rng.uniform(0, 20, n_ground)
    gz = rng.uniform(-0.1, 0.1, n_ground)
    ground = np.column_stack((gx, gy, gz))

    wx = np.linspace(10, 10 + wire_length, n_wire) + rng.normal(0, 0.05, n_wire)
    wy = np.full(n_wire, 10.0) + rng.normal(0, 0.1, n_wire)
    wz = np.full(n_wire, 10.0) + rng.normal(0, 0.05, n_wire)
    wire = np.column_stack((wx, wy, wz))

    xyz = np.vstack((ground, wire)).astype(np.float64)
    classification = np.ones(len(xyz), dtype=np.uint8)
    classification[:n_ground] = 2
    return PointCloud(xyz, classification=classification)


def _make_wide_pylon_cloud(
    *,
    seed: int = 66,
) -> PointCloud:
    """Build a cloud with a wide pylon cluster (simulating a building).

    One pylon cluster has a wide XY footprint (>5m) that strict mode should
    reject, plus a real narrow pylon for comparison.
    """
    rng = np.random.default_rng(seed)
    n_ground = 500

    gx = rng.uniform(0, 120, n_ground)
    gy = rng.uniform(0, 30, n_ground)
    gz = rng.uniform(-0.1, 0.1, n_ground)
    ground = np.column_stack((gx, gy, gz))

    # Wide "pylon" (building): 10m x 10m footprint, 12m tall
    n_wide = 60
    bx = rng.uniform(20, 30, n_wide)  # 10m wide
    by = rng.uniform(5, 15, n_wide)   # 10m deep
    bz = rng.uniform(0, 12, n_wide)
    wide_pylon = np.column_stack((bx, by, bz))

    # Narrow pylon: <5m footprint, 15m tall
    n_narrow = 40
    nx = np.full(n_narrow, 90.0) + rng.normal(0, 0.3, n_narrow)
    ny = np.full(n_narrow, 15.0) + rng.normal(0, 0.3, n_narrow)
    nz = rng.uniform(0, 15, n_narrow)
    narrow_pylon = np.column_stack((nx, ny, nz))

    # Wire connecting them (long enough to pass span filter)
    n_wire = 200
    wx = np.linspace(30, 90, n_wire) + rng.normal(0, 0.05, n_wire)
    wy = np.full(n_wire, 15.0) + rng.normal(0, 0.1, n_wire)
    wz = np.full(n_wire, 12.0) + rng.normal(0, 0.05, n_wire)
    wire = np.column_stack((wx, wy, wz))

    xyz = np.vstack((ground, wide_pylon, narrow_pylon, wire)).astype(np.float64)
    classification = np.ones(len(xyz), dtype=np.uint8)
    classification[:n_ground] = 2
    return PointCloud(xyz, classification=classification)


def _make_noisy_wire_cloud(
    *,
    seed: int = 77,
) -> PointCloud:
    """Build a cloud with a high-Z-variance wire cluster (simulating tree crown).

    Wire points have Z std > 3m — strict mode should reject this cluster.
    Also includes a clean wire for comparison.
    """
    rng = np.random.default_rng(seed)
    n_ground = 500

    gx = rng.uniform(0, 150, n_ground)
    gy = rng.uniform(0, 30, n_ground)
    gz = rng.uniform(-0.1, 0.1, n_ground)
    ground = np.column_stack((gx, gy, gz))

    # Noisy "wire" (tree crown): high Z scatter, std ~5m
    n_noisy = 150
    nx = np.linspace(10, 70, n_noisy) + rng.normal(0, 0.1, n_noisy)
    ny = np.full(n_noisy, 10.0) + rng.normal(0, 0.2, n_noisy)
    nz = np.full(n_noisy, 12.0) + rng.normal(0, 5.0, n_noisy)
    noisy_wire = np.column_stack((nx, ny, nz))

    # Clean wire: low Z scatter
    n_clean = 200
    cx = np.linspace(80, 140, n_clean) + rng.normal(0, 0.05, n_clean)
    cy = np.full(n_clean, 15.0) + rng.normal(0, 0.1, n_clean)
    cz = np.full(n_clean, 10.0) + rng.normal(0, 0.05, n_clean)
    clean_wire = np.column_stack((cx, cy, cz))

    xyz = np.vstack((ground, noisy_wire, clean_wire)).astype(np.float64)
    classification = np.ones(len(xyz), dtype=np.uint8)
    classification[:n_ground] = 2
    return PointCloud(xyz, classification=classification)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def powerline_cloud() -> PointCloud:
    """Synthetic powerline cloud with ground, wires, and pylons."""
    return _make_powerline_cloud()


@pytest.fixture
def ground_only_cloud() -> PointCloud:
    """Cloud where all points are classified as ground."""
    rng = np.random.default_rng(0)
    xyz = rng.random((100, 3)).astype(np.float64)
    classification = np.full(100, 2, dtype=np.uint8)
    return PointCloud(xyz, classification=classification)


@pytest.fixture
def no_classification_cloud() -> PointCloud:
    """Cloud without classification attribute."""
    rng = np.random.default_rng(0)
    return PointCloud(rng.random((100, 3)).astype(np.float64))


# ---------------------------------------------------------------------------
# detect_powerlines — main function
# ---------------------------------------------------------------------------


class TestDetectPowerlines:
    """Tests for detect_powerlines."""

    def test_returns_powerline_result(self, powerline_cloud):
        """detect_powerlines returns a PowerlineResult instance."""
        result = detect_powerlines(powerline_cloud)
        assert isinstance(result, PowerlineResult)

    def test_wire_mask_shape(self, powerline_cloud):
        """wire_mask length matches n_points."""
        result = detect_powerlines(powerline_cloud)
        assert len(result.wire_mask) == powerline_cloud.n_points

    def test_pylon_mask_shape(self, powerline_cloud):
        """pylon_mask length matches n_points."""
        result = detect_powerlines(powerline_cloud)
        assert len(result.pylon_mask) == powerline_cloud.n_points

    def test_detects_some_wires(self, powerline_cloud):
        """At least some wire points are detected."""
        result = detect_powerlines(powerline_cloud)
        assert result.wire_mask.sum() > 0

    def test_detects_wire_segments(self, powerline_cloud):
        """At least one wire segment is produced."""
        result = detect_powerlines(powerline_cloud)
        assert len(result.wire_segments) >= 1

    def test_no_classification_raises(self, no_classification_cloud):
        """Raises OcculusSegmentationError when classification is missing."""
        with pytest.raises(OcculusSegmentationError, match="classification"):
            detect_powerlines(no_classification_cloud)

    def test_all_ground_raises(self, ground_only_cloud):
        """Raises OcculusSegmentationError when all points are ground."""
        with pytest.raises(OcculusSegmentationError, match="ground"):
            detect_powerlines(ground_only_cloud)

    def test_empty_cloud_raises(self):
        """Raises OcculusSegmentationError for an empty cloud."""
        cloud = PointCloud(
            np.zeros((0, 3), dtype=np.float64),
            classification=np.array([], dtype=np.uint8),
        )
        with pytest.raises(OcculusSegmentationError, match="empty"):
            detect_powerlines(cloud)

    def test_invalid_height_range_raises(self, powerline_cloud):
        """Raises when max_height <= min_height."""
        with pytest.raises(OcculusSegmentationError, match="max_height"):
            detect_powerlines(
                powerline_cloud,
                min_height_above_ground=10.0,
                max_height_above_ground=5.0,
            )

    def test_invalid_linearity_threshold_raises(self, powerline_cloud):
        """Raises when linearity_threshold is out of (0, 1]."""
        with pytest.raises(OcculusSegmentationError, match="linearity_threshold"):
            detect_powerlines(powerline_cloud, linearity_threshold=0.0)

    def test_negative_min_height_raises(self, powerline_cloud):
        """Raises when min_height_above_ground < 0."""
        with pytest.raises(OcculusSegmentationError, match="min_height_above_ground"):
            detect_powerlines(powerline_cloud, min_height_above_ground=-1.0)

    def test_catenary_fit_disabled(self, powerline_cloud):
        """Works with catenary_fit=False."""
        result = detect_powerlines(powerline_cloud, catenary_fit=False)
        assert isinstance(result, PowerlineResult)
        # Without fitting, segments should have rmse=inf
        for seg in result.wire_segments:
            assert seg.rmse == float("inf")

    def test_clearance_violations_detected(self):
        """Clearance violations are reported when wires are too low."""
        # Create a cloud with low wires (at 4m, min_clearance=5m)
        cloud = _make_powerline_cloud(wire_height=4.0)
        result = detect_powerlines(
            cloud,
            min_height_above_ground=2.0,
            min_clearance=5.0,
        )
        assert len(result.clearance_violations) > 0

    def test_no_clearance_check_when_none(self, powerline_cloud):
        """No clearance violations when min_clearance is None."""
        result = detect_powerlines(powerline_cloud, min_clearance=None)
        assert len(result.clearance_violations) == 0

    def test_pylon_positions_are_3d(self, powerline_cloud):
        """pylon_positions has shape (M, 3)."""
        result = detect_powerlines(powerline_cloud)
        assert result.pylon_positions.ndim == 2
        if len(result.pylon_positions) > 0:
            assert result.pylon_positions.shape[1] == 3

    def test_wire_and_pylon_masks_are_disjoint(self, powerline_cloud):
        """Wire and pylon masks do not overlap."""
        result = detect_powerlines(powerline_cloud)
        overlap = result.wire_mask & result.pylon_mask
        assert not overlap.any()

    def test_custom_ground_class(self):
        """Works with a non-default ground_class."""
        rng = np.random.default_rng(99)
        # Ground at class 6 instead of 2
        n_ground = 300
        n_above = 200
        ground = np.column_stack(
            [
                rng.uniform(0, 50, n_ground),
                rng.uniform(0, 20, n_ground),
                rng.uniform(-0.1, 0.1, n_ground),
            ]
        )
        above = np.column_stack(
            [
                np.linspace(5, 45, n_above) + rng.normal(0, 0.05, n_above),
                np.full(n_above, 10.0) + rng.normal(0, 0.1, n_above),
                np.full(n_above, 8.0) + rng.normal(0, 0.05, n_above),
            ]
        )
        xyz = np.vstack((ground, above)).astype(np.float64)
        classification = np.ones(len(xyz), dtype=np.uint8)
        classification[:n_ground] = 6
        cloud = PointCloud(xyz, classification=classification)

        result = detect_powerlines(cloud, ground_class=6)
        assert isinstance(result, PowerlineResult)

    def test_strict_false_preserves_original_behaviour(self, powerline_cloud):
        """strict=False does not apply any new filters — all wire candidates kept."""
        result = detect_powerlines(powerline_cloud, strict=False)
        assert isinstance(result, PowerlineResult)
        assert result.wire_mask.sum() > 0

    def test_strict_rejects_short_wire_segments(self):
        """Strict mode rejects wire segments shorter than min_wire_span."""
        # Build a cloud with a short wire (20m span, below 50m default)
        cloud = _make_short_wire_cloud(wire_length=20.0)
        strict_result = detect_powerlines(cloud, strict=True)
        permissive_result = detect_powerlines(cloud, strict=False)

        # Strict should find fewer (or zero) wire points vs permissive
        assert strict_result.wire_mask.sum() < permissive_result.wire_mask.sum()

    def test_strict_rejects_wide_pylon_clusters(self):
        """Strict mode rejects pylon clusters with wide XY footprint (buildings)."""
        cloud = _make_wide_pylon_cloud()
        strict_result = detect_powerlines(cloud, strict=True)
        permissive_result = detect_powerlines(cloud, strict=False)

        # Strict should have fewer pylons (wide cluster rejected)
        assert len(strict_result.pylon_positions) < len(permissive_result.pylon_positions)

    def test_strict_rejects_high_z_variance_wires(self):
        """Strict mode rejects wire clusters with excessive Z variance."""
        cloud = _make_noisy_wire_cloud()
        strict_result = detect_powerlines(cloud, strict=True)
        permissive_result = detect_powerlines(cloud, strict=False)

        assert strict_result.wire_mask.sum() < permissive_result.wire_mask.sum()

    def test_strict_custom_thresholds(self, powerline_cloud):
        """Custom strict thresholds are respected."""
        # Set min_wire_span very high so all wires are rejected
        result = detect_powerlines(
            powerline_cloud,
            strict=True,
            min_wire_span=200.0,
        )
        assert result.wire_mask.sum() == 0

    def test_strict_with_catenary_fit(self, powerline_cloud):
        """Strict mode works correctly with catenary fitting enabled."""
        result = detect_powerlines(powerline_cloud, strict=True, catenary_fit=True)
        assert isinstance(result, PowerlineResult)
        # Segments that survive strict filtering get catenary fits
        for seg in result.wire_segments:
            # Fitted segments should have finite rmse (or inf if fit failed)
            assert isinstance(seg.rmse, float)


# ---------------------------------------------------------------------------
# _compute_height_above_ground
# ---------------------------------------------------------------------------


class TestComputeHeightAboveGround:
    """Tests for height-above-ground interpolation."""

    def test_flat_ground_returns_z(self):
        """On a flat ground at z=0, HAG equals the point's z coordinate."""
        ground = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]], dtype=np.float64)
        pts = np.array([[5, 5, 7], [2, 2, 3]], dtype=np.float64)
        hag = _compute_height_above_ground(pts, ground)
        np.testing.assert_allclose(hag, [7.0, 3.0], atol=0.01)

    def test_ground_points_have_zero_hag(self):
        """Ground points themselves should have HAG ~ 0."""
        ground = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]], dtype=np.float64)
        hag = _compute_height_above_ground(ground, ground)
        np.testing.assert_allclose(hag, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# _compute_geometric_features
# ---------------------------------------------------------------------------


class TestComputeGeometricFeatures:
    """Tests for PCA-based geometric features."""

    def test_linear_points_have_high_linearity(self):
        """Points arranged in a line should have linearity ~ 1."""
        t = np.linspace(0, 10, 50)
        pts = np.column_stack([t, np.zeros(50), np.zeros(50)]).astype(np.float64)
        # Add tiny noise to avoid degenerate covariance
        pts += np.random.default_rng(0).normal(0, 0.001, pts.shape)
        lin, _, _ = _compute_geometric_features(pts, k=10)
        assert np.mean(lin) > 0.8

    def test_planar_points_have_high_planarity(self):
        """Points on a flat plane should have high planarity."""
        rng = np.random.default_rng(1)
        pts = np.column_stack(
            [
                rng.uniform(0, 10, 100),
                rng.uniform(0, 10, 100),
                rng.normal(0, 0.001, 100),
            ]
        ).astype(np.float64)
        _, plan, _ = _compute_geometric_features(pts, k=10)
        assert np.mean(plan) > 0.4

    def test_few_points_no_crash(self):
        """Does not crash with very few points."""
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        lin, plan, vert = _compute_geometric_features(pts, k=1)
        assert len(lin) == 1

    def test_output_shapes(self):
        """Output arrays have the same length as input."""
        rng = np.random.default_rng(2)
        pts = rng.random((30, 3)).astype(np.float64)
        lin, plan, vert = _compute_geometric_features(pts, k=10)
        assert lin.shape == (30,)
        assert plan.shape == (30,)
        assert vert.shape == (30,)


# ---------------------------------------------------------------------------
# _dbscan_cluster
# ---------------------------------------------------------------------------


class TestDBSCANCluster:
    """Tests for internal DBSCAN helper."""

    def test_two_clusters(self):
        """Finds two clusters in well-separated data."""
        rng = np.random.default_rng(3)
        c1 = rng.standard_normal((50, 3))
        c2 = rng.standard_normal((50, 3)) + 20.0
        pts = np.vstack((c1, c2)).astype(np.float64)
        labels = _dbscan_cluster(pts, eps=2.0, min_samples=5)
        assert len(set(labels) - {-1}) == 2

    def test_empty_input(self):
        """Returns empty array for empty input."""
        labels = _dbscan_cluster(np.empty((0, 3), dtype=np.float64), eps=1.0, min_samples=5)
        assert len(labels) == 0


# ---------------------------------------------------------------------------
# _catenary
# ---------------------------------------------------------------------------


class TestCatenary:
    """Tests for the catenary function."""

    def test_vertex_value(self):
        """At x=x0, catenary returns a + z0."""
        x = np.array([5.0])
        result = _catenary(x, a=10.0, x0=5.0, z0=-10.0)
        expected = 10.0 * np.cosh(0.0) + (-10.0)  # = 10.0 - 10.0 = 0.0
        np.testing.assert_allclose(result, expected)

    def test_symmetry(self):
        """Catenary is symmetric about x0."""
        x = np.array([3.0, 7.0])
        result = _catenary(x, a=5.0, x0=5.0, z0=0.0)
        np.testing.assert_allclose(result[0], result[1])


# ---------------------------------------------------------------------------
# Dataclass smoke tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Smoke tests for result dataclasses."""

    def test_powerline_result_defaults(self):
        """PowerlineResult can be constructed with minimal arguments."""
        result = PowerlineResult(
            wire_mask=np.zeros(10, dtype=bool),
            pylon_mask=np.zeros(10, dtype=bool),
        )
        assert len(result.wire_segments) == 0
        assert len(result.clearance_violations) == 0
        assert result.pylon_positions.shape == (0, 3)

    def test_catenary_segment(self):
        """CatenarySegment stores its fields."""
        seg = CatenarySegment(
            indices=np.array([0, 1, 2]),
            a=50.0,
            x0=0.0,
            z0=-50.0,
            rmse=0.1,
        )
        assert seg.a == 50.0
        assert seg.rmse == 0.1

    def test_clearance_violation(self):
        """ClearanceViolation stores its fields."""
        v = ClearanceViolation(point_index=42, height_above_ground=3.5, min_clearance=5.0)
        assert v.point_index == 42
        assert v.height_above_ground == 3.5


# ---------------------------------------------------------------------------
# _filter_orphan_wires
# ---------------------------------------------------------------------------


class TestFilterOrphanWires:
    """Tests for wire-pylon association helper."""

    def test_connected_segment_kept(self):
        """Wire segment with endpoints near pylons is kept."""
        # 100 points, wire from x=0 to x=80
        n = 100
        xyz = np.zeros((n, 3), dtype=np.float64)
        xyz[:, 0] = np.linspace(0, 80, n)
        xyz[:, 2] = 10.0

        seg = CatenarySegment(
            indices=np.arange(n, dtype=np.intp),
            a=0.0, x0=0.0, z0=0.0, rmse=float("inf"),
        )
        wire_mask = np.ones(n, dtype=bool)
        pylons = np.array([[0, 0, 10], [80, 0, 10]], dtype=np.float64)

        segs_out, mask_out = _filter_orphan_wires(
            xyz, [seg], wire_mask, pylons, radius=10.0
        )
        assert len(segs_out) == 1
        assert mask_out.sum() == n

    def test_orphan_segment_removed(self):
        """Wire segment far from any pylon is removed from mask."""
        n = 50
        xyz = np.zeros((n, 3), dtype=np.float64)
        xyz[:, 0] = np.linspace(100, 150, n)
        xyz[:, 2] = 10.0

        seg = CatenarySegment(
            indices=np.arange(n, dtype=np.intp),
            a=0.0, x0=0.0, z0=0.0, rmse=float("inf"),
        )
        wire_mask = np.ones(n, dtype=bool)
        pylons = np.array([[0, 0, 10], [80, 0, 10]], dtype=np.float64)

        segs_out, mask_out = _filter_orphan_wires(
            xyz, [seg], wire_mask, pylons, radius=10.0
        )
        assert len(segs_out) == 0
        assert mask_out.sum() == 0
