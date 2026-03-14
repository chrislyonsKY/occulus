"""Visualization helpers using Open3D.

Optional module — requires ``pip install occulus[viz]``.

Available functions
-------------------
- :func:`visualize` — display one or more point clouds in an Open3D window
- :func:`visualize_registration` — overlay source and target with transform applied
- :func:`visualize_segments` — colorize a cloud by segment labels

All functions raise ``ImportError`` if Open3D is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.types import PointCloud

if TYPE_CHECKING:
    from occulus.registration.icp import RegistrationResult
    from occulus.segmentation.objects import SegmentationResult

logger = logging.getLogger(__name__)

__all__ = [
    "visualize",
    "visualize_registration",
    "visualize_segments",
]


def visualize(
    *clouds: PointCloud,
    point_size: float = 1.0,
    background: tuple[float, float, float] = (0.1, 0.1, 0.1),
    window_name: str = "Occulus Viewer",
    show_normals: bool = False,
) -> None:
    """Display one or more point clouds in an Open3D interactive viewer.

    Parameters
    ----------
    *clouds : PointCloud
        One or more point clouds to display simultaneously.
    point_size : float, optional
        Rendered point size in pixels, by default 1.0.
    background : tuple[float, float, float], optional
        Background colour as (R, G, B) floats in [0, 1], by default (0.1, 0.1, 0.1).
    window_name : str, optional
        Title of the viewer window, by default ``"Occulus Viewer"``.
    show_normals : bool, optional
        Whether to display normal vectors, by default ``False``.

    Raises
    ------
    ImportError
        If Open3D is not installed.
    ValueError
        If no clouds are provided.
    """
    if not clouds:
        raise ValueError("visualize() requires at least one PointCloud argument")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for visualization: pip install 'occulus[viz]'"
        ) from exc

    geometries = [cloud.to_open3d() for cloud in clouds]

    logger.debug("visualize: displaying %d cloud(s)", len(geometries))
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        point_show_normal=show_normals,
    )


def visualize_registration(
    source: PointCloud,
    target: PointCloud,
    result: RegistrationResult,
    *,
    source_color: tuple[float, float, float] = (1.0, 0.2, 0.2),
    target_color: tuple[float, float, float] = (0.2, 0.2, 1.0),
    window_name: str = "Registration Result",
) -> None:
    """Visualize a registration result with source and target overlaid.

    The source cloud is rendered in ``source_color`` after applying the
    registration transformation. The target remains in ``target_color``.

    Parameters
    ----------
    source : PointCloud
        Source cloud (will be transformed for display).
    target : PointCloud
        Target cloud (fixed reference).
    result : RegistrationResult
        Registration result containing the 4×4 transformation matrix.
    source_color : tuple[float, float, float], optional
        RGB colour for transformed source points, by default red.
    target_color : tuple[float, float, float], optional
        RGB colour for target points, by default blue.
    window_name : str, optional
        Viewer window title, by default ``"Registration Result"``.

    Raises
    ------
    ImportError
        If Open3D is not installed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for visualization: pip install 'occulus[viz]'"
        ) from exc

    src_pcd = source.to_open3d()
    tgt_pcd = target.to_open3d()

    # Apply transformation to source
    src_pcd.transform(result.transformation)  # type: ignore[attr-defined]

    # Paint uniform colours
    src_pcd.paint_uniform_color(list(source_color))  # type: ignore[attr-defined]
    tgt_pcd.paint_uniform_color(list(target_color))  # type: ignore[attr-defined]

    logger.debug(
        "visualize_registration: fitness=%.3f rmse=%.4f converged=%s",
        result.fitness,
        result.inlier_rmse,
        result.converged,
    )
    o3d.visualization.draw_geometries(
        [src_pcd, tgt_pcd],
        window_name=window_name,
    )


def visualize_segments(
    cloud: PointCloud,
    labels: NDArray[np.int32],
    *,
    noise_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    window_name: str = "Segmentation Result",
) -> None:
    """Visualize a segmented point cloud with each segment in a distinct colour.

    Points with label ``-1`` (noise) are rendered in ``noise_color``.

    Parameters
    ----------
    cloud : PointCloud
        Original point cloud.
    labels : NDArray[np.int32]
        Per-point segment labels of length ``n_points``. Label -1 = noise.
    noise_color : tuple[float, float, float], optional
        RGB colour for unassigned (noise) points, by default grey.
    window_name : str, optional
        Viewer window title, by default ``"Segmentation Result"``.

    Raises
    ------
    ImportError
        If Open3D is not installed.
    ValueError
        If ``labels`` length does not match ``cloud.n_points``.
    """
    if len(labels) != cloud.n_points:
        raise ValueError(
            f"labels length ({len(labels)}) does not match n_points ({cloud.n_points})"
        )

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for visualization: pip install 'occulus[viz]'"
        ) from exc

    unique_labels = sorted(set(labels.tolist()) - {-1})
    n_labels = len(unique_labels)

    # Build a colour palette using HSV
    colors = np.zeros((cloud.n_points, 3), dtype=np.float64)
    colors[labels == -1] = noise_color

    if n_labels > 0:
        hues = np.linspace(0, 1, n_labels, endpoint=False)
        label_to_color: dict[int, NDArray[np.float64]] = {}
        for i, lbl in enumerate(unique_labels):
            h = hues[i]
            # Convert HSV (h, 0.8, 0.9) to RGB
            label_to_color[lbl] = _hsv_to_rgb(h, 0.8, 0.9)

        for lbl, color in label_to_color.items():
            colors[labels == lbl] = color

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(cloud.xyz))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    logger.debug(
        "visualize_segments: %d segments + %d noise points",
        n_labels,
        int((labels == -1).sum()),
    )
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _hsv_to_rgb(h: float, s: float, v: float) -> NDArray[np.float64]:
    """Convert HSV colour to RGB.

    Parameters
    ----------
    h : float
        Hue in [0, 1).
    s : float
        Saturation in [0, 1].
    v : float
        Value in [0, 1].

    Returns
    -------
    NDArray[np.float64]
        RGB colour as (3,) array with values in [0, 1].
    """
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    segment = i % 6
    if segment == 0:
        return np.array([v, t, p])
    elif segment == 1:
        return np.array([q, v, p])
    elif segment == 2:
        return np.array([p, v, t])
    elif segment == 3:
        return np.array([p, q, v])
    elif segment == 4:
        return np.array([t, p, v])
    else:
        return np.array([v, p, q])
