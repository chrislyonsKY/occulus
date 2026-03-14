"""Surface reconstruction from point clouds.

All reconstruction functions delegate to Open3D (optional dependency).
Import this module only when ``open3d`` is installed (``pip install occulus[viz]``).

Available functions
-------------------
- :func:`poisson_mesh` — Screened Poisson reconstruction (watertight, requires normals)
- :func:`ball_pivoting_mesh` — Ball Pivoting Algorithm (respects boundaries, requires normals)
- :func:`alpha_shape_mesh` — Alpha shape (concave, normals optional)

Each function returns a :class:`MeshResult` wrapping the Open3D TriangleMesh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusMeshError, OcculusValidationError
from occulus.types import PointCloud

logger = logging.getLogger(__name__)

__all__ = [
    "MeshResult",
    "poisson_mesh",
    "ball_pivoting_mesh",
    "alpha_shape_mesh",
]


@dataclass
class MeshResult:
    """Result of surface reconstruction.

    Attributes
    ----------
    vertices : NDArray[np.float64]
        Mesh vertices as (V, 3) array.
    faces : NDArray[np.int32]
        Triangle face indices as (F, 3) array.
    vertex_normals : NDArray[np.float64] | None
        Per-vertex normals as (V, 3) array, or ``None``.
    vertex_colors : NDArray[np.float64] | None
        Per-vertex colors as (V, 3) float64 array in [0, 1], or ``None``.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int32]
    vertex_normals: NDArray[np.float64] | None = None
    vertex_colors: NDArray[np.float64] | None = None

    @property
    def n_vertices(self) -> int:
        """Number of mesh vertices."""
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        """Number of triangle faces."""
        return self.faces.shape[0]

    def to_open3d(self) -> object:
        """Convert to an Open3D TriangleMesh.

        Returns
        -------
        open3d.geometry.TriangleMesh
            The mesh in Open3D format.

        Raises
        ------
        ImportError
            If Open3D is not installed.
        """
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for to_open3d(): pip install 'occulus[viz]'"
            ) from exc

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(self.vertices))
        mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(self.faces))
        if self.vertex_normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.vertex_normals)
            )
        if self.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(self.vertex_colors)
            )
        return mesh


def poisson_mesh(
    cloud: PointCloud,
    *,
    depth: int = 8,
    width: float = 0.0,
    scale: float = 1.1,
    linear_fit: bool = False,
    density_threshold_quantile: float | None = 0.01,
) -> MeshResult:
    """Screened Poisson surface reconstruction.

    Produces a watertight mesh suitable for volume computation and rendering.
    Normals are **required** on the input cloud.

    Parameters
    ----------
    cloud : PointCloud
        Input cloud with normals. Run :func:`~occulus.normals.estimate_normals`
        and :func:`~occulus.normals.orient_normals_to_viewpoint` first.
    depth : int, optional
        Octree depth controlling mesh resolution, by default 8.
        Higher values produce finer meshes at higher cost.
    width : float, optional
        Target octree cell width (overrides ``depth`` when > 0), by default 0.0.
    scale : float, optional
        Ratio of the solving domain diameter to the bounding-box diameter,
        by default 1.1.
    linear_fit : bool, optional
        Use linear interpolation for iso-surface extraction, by default ``False``.
    density_threshold_quantile : float | None, optional
        Fraction (0–1) of low-density vertices to remove after reconstruction.
        Set to ``None`` to keep all vertices, by default 0.01.

    Returns
    -------
    MeshResult
        Reconstructed mesh.

    Raises
    ------
    OcculusValidationError
        If the cloud has no normals.
    OcculusMeshError
        If reconstruction fails or produces an empty mesh.
    ImportError
        If Open3D is not installed.
    """
    if not cloud.has_normals:
        raise OcculusValidationError(
            "Poisson reconstruction requires normals. "
            "Run occulus.normals.estimate_normals() first."
        )

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for Poisson reconstruction: pip install 'occulus[viz]'"
        ) from exc

    pcd = cloud.to_open3d()

    try:
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit,
        )
    except Exception as exc:
        raise OcculusMeshError(f"Poisson reconstruction failed: {exc}") from exc

    if density_threshold_quantile is not None:
        densities_np = np.asarray(densities)
        threshold = np.quantile(densities_np, density_threshold_quantile)
        vertices_to_remove = densities_np < threshold
        mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

    return _o3d_mesh_to_result(mesh_o3d)


def ball_pivoting_mesh(
    cloud: PointCloud,
    *,
    radii: list[float] | None = None,
    radii_factor: float = 3.0,
) -> MeshResult:
    """Ball Pivoting Algorithm (BPA) surface reconstruction.

    Rolls a virtual ball of the specified radius across the point cloud.
    Normals are **required**. Unlike Poisson, BPA does not fill holes and
    handles boundary regions more naturally.

    Parameters
    ----------
    cloud : PointCloud
        Input cloud with normals.
    radii : list[float] | None, optional
        Ball radii to use. Multiple radii help fill gaps at different scales.
        If ``None``, auto-computed from the average nearest-neighbour distance.
    radii_factor : float, optional
        When auto-computing radii, multiply the mean NN distance by this factor
        to produce three candidate radii, by default 3.0.

    Returns
    -------
    MeshResult
        Reconstructed mesh.

    Raises
    ------
    OcculusValidationError
        If the cloud has no normals.
    OcculusMeshError
        If reconstruction fails or produces an empty mesh.
    ImportError
        If Open3D is not installed.
    """
    if not cloud.has_normals:
        raise OcculusValidationError(
            "BPA reconstruction requires normals. "
            "Run occulus.normals.estimate_normals() first."
        )

    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for BPA reconstruction: pip install 'occulus[viz]'"
        ) from exc

    if radii is None:
        from scipy.spatial import KDTree  # type: ignore[import-untyped]
        tree = KDTree(cloud.xyz)
        distances, _ = tree.query(cloud.xyz, k=2, workers=-1)
        mean_nn = float(distances[:, 1].mean())
        radii = [mean_nn * radii_factor * f for f in (1.0, 2.0, 4.0)]

    pcd = cloud.to_open3d()

    try:
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii),
        )
    except Exception as exc:
        raise OcculusMeshError(f"BPA reconstruction failed: {exc}") from exc

    return _o3d_mesh_to_result(mesh_o3d)


def alpha_shape_mesh(
    cloud: PointCloud,
    *,
    alpha: float = 1.0,
) -> MeshResult:
    """Alpha shape surface reconstruction.

    Computes the alpha-shape (generalised convex hull) of the point cloud.
    Produces non-watertight meshes that respect concavities. Normals are
    not required but are computed on the result.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    alpha : float, optional
        Alpha parameter controlling concavity. Smaller values produce tighter,
        more concave meshes, by default 1.0.

    Returns
    -------
    MeshResult
        Reconstructed mesh.

    Raises
    ------
    OcculusMeshError
        If reconstruction fails or produces an empty mesh.
    ImportError
        If Open3D is not installed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for alpha shape reconstruction: pip install 'occulus[viz]'"
        ) from exc

    pcd = cloud.to_open3d()

    try:
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
        mesh_o3d.compute_vertex_normals()
    except Exception as exc:
        raise OcculusMeshError(f"Alpha shape reconstruction failed: {exc}") from exc

    return _o3d_mesh_to_result(mesh_o3d)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _o3d_mesh_to_result(mesh: object) -> MeshResult:
    """Convert an Open3D TriangleMesh to a :class:`MeshResult`.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Source mesh.

    Returns
    -------
    MeshResult
        Converted mesh result.

    Raises
    ------
    OcculusMeshError
        If the mesh has no vertices or faces.
    """
    import open3d as o3d  # already imported by caller; safe to re-import

    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        raise OcculusMeshError(f"Expected TriangleMesh, got {type(mesh)}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    if vertices.shape[0] == 0:
        raise OcculusMeshError("Reconstruction produced an empty mesh (no vertices).")
    if faces.shape[0] == 0:
        raise OcculusMeshError("Reconstruction produced an empty mesh (no faces).")

    vertex_normals: NDArray[np.float64] | None = None
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    vertex_colors: NDArray[np.float64] | None = None
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)

    logger.info(
        "Mesh: %d vertices, %d faces", vertices.shape[0], faces.shape[0]
    )
    return MeshResult(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
    )
