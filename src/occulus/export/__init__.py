"""3D Tiles and Potree export for web visualization.

Converts point clouds to formats suitable for browser-based 3-D viewers:

- **3D Tiles** (Cesium) — ``tileset.json`` + ``.pnts`` binary tiles
- **Potree** — octree hierarchy with ``metadata.json``
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from occulus.exceptions import OcculusExportError

if TYPE_CHECKING:
    from occulus.types import PointCloud

logger = logging.getLogger(__name__)


def export_3dtiles(
    cloud: PointCloud,
    output_dir: str | Path,
    *,
    max_points_per_tile: int = 50_000,
    geometric_error: float = 10.0,
) -> Path:
    """Export a point cloud as a 3D Tiles tileset.

    Generates a ``tileset.json`` and ``.pnts`` binary files suitable
    for display in CesiumJS or other 3D Tiles viewers.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    output_dir : str or Path
        Output directory for the tileset.
    max_points_per_tile : int
        Maximum points per ``.pnts`` file.
    geometric_error : float
        Root tile geometric error in metres.

    Returns
    -------
    Path
        Path to the generated ``tileset.json``.

    Raises
    ------
    OcculusExportError
        If the export fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = cloud.xyz
    n = len(xyz)

    if n == 0:
        raise OcculusExportError("Cannot export empty point cloud to 3D Tiles")

    # Compute bounding volume (region or box)
    center = xyz.mean(axis=0)
    half_extent = (xyz.max(axis=0) - xyz.min(axis=0)) / 2

    # Split into chunks
    chunks = []
    for i in range(0, n, max_points_per_tile):
        chunk = xyz[i : i + max_points_per_tile]
        chunks.append(chunk)

    logger.info("Exporting %d points as %d 3D Tiles chunks", n, len(chunks))

    # Write .pnts files
    children = []
    for idx, chunk in enumerate(chunks):
        pnts_name = f"tile_{idx:04d}.pnts"
        pnts_path = output_dir / pnts_name
        _write_pnts(chunk, center, pnts_path)

        chunk_center = chunk.mean(axis=0)
        chunk_half = (chunk.max(axis=0) - chunk.min(axis=0)) / 2

        children.append(
            {
                "boundingVolume": {
                    "box": [
                        *chunk_center.tolist(),
                        chunk_half[0],
                        0,
                        0,
                        0,
                        chunk_half[1],
                        0,
                        0,
                        0,
                        chunk_half[2],
                    ]
                },
                "geometricError": 0.0,
                "content": {"uri": pnts_name},
            }
        )

    # Write tileset.json
    tileset = {
        "asset": {"version": "1.0", "generator": "occulus"},
        "geometricError": geometric_error,
        "root": {
            "boundingVolume": {
                "box": [
                    *center.tolist(),
                    half_extent[0],
                    0,
                    0,
                    0,
                    half_extent[1],
                    0,
                    0,
                    0,
                    half_extent[2],
                ]
            },
            "geometricError": geometric_error,
            "refine": "ADD",
            "children": children,
        },
    }

    tileset_path = output_dir / "tileset.json"
    tileset_path.write_text(json.dumps(tileset, indent=2))
    logger.info("3D Tiles tileset → %s (%d tiles)", tileset_path, len(chunks))
    return tileset_path


def _write_pnts(
    xyz: NDArray[np.float64],
    center: NDArray[np.float64],
    path: Path,
) -> None:
    """Write a .pnts binary file (3D Tiles Point Cloud format).

    Parameters
    ----------
    xyz : NDArray
        (N, 3) point positions.
    center : NDArray
        (3,) center offset for RTC (Relative-To-Center).
    path : Path
        Output file path.
    """
    n = len(xyz)
    # Positions relative to center (float32)
    positions = (xyz - center).astype(np.float32)

    # Feature table JSON
    ft_json = json.dumps(
        {
            "POINTS_LENGTH": n,
            "RTC_CENTER": center.tolist(),
            "POSITION": {"byteOffset": 0},
        }
    )
    # Pad to 8-byte boundary
    while len(ft_json) % 8 != 0:
        ft_json += " "
    ft_json_bytes = ft_json.encode("utf-8")

    ft_binary = positions.tobytes()

    # Header: magic, version, byteLength, ftJSONByteLength, ftBinaryByteLength,
    #          btJSONByteLength, btBinaryByteLength
    header_size = 28
    total_size = header_size + len(ft_json_bytes) + len(ft_binary)

    with open(path, "wb") as f:
        f.write(b"pnts")  # magic
        f.write(struct.pack("<I", 1))  # version
        f.write(struct.pack("<I", total_size))
        f.write(struct.pack("<I", len(ft_json_bytes)))
        f.write(struct.pack("<I", len(ft_binary)))
        f.write(struct.pack("<I", 0))  # batch table JSON
        f.write(struct.pack("<I", 0))  # batch table binary
        f.write(ft_json_bytes)
        f.write(ft_binary)


def export_potree(
    cloud: PointCloud,
    output_dir: str | Path,
    *,
    max_depth: int = 10,
    max_points_per_node: int = 50_000,
) -> Path:
    """Export a point cloud in Potree 2.0 format.

    Generates an octree hierarchy with ``metadata.json`` and binary
    ``.bin`` chunks for use with the Potree web viewer.

    Parameters
    ----------
    cloud : PointCloud
        Input point cloud.
    output_dir : str or Path
        Output directory.
    max_depth : int
        Maximum octree depth.
    max_points_per_node : int
        Maximum points before splitting a node.

    Returns
    -------
    Path
        Path to the generated ``metadata.json``.

    Raises
    ------
    OcculusExportError
        If the export fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = cloud.xyz
    n = len(xyz)

    if n == 0:
        raise OcculusExportError("Cannot export empty point cloud to Potree")

    bb_min = xyz.min(axis=0)
    bb_max = xyz.max(axis=0)
    bb_size = bb_max - bb_min
    cube_size = float(bb_size.max())

    logger.info("Exporting %d points as Potree octree (max depth %d)", n, max_depth)

    # Build simple octree
    nodes = _build_octree(xyz, bb_min, cube_size, max_depth, max_points_per_node)

    # Write node binary files
    octree_dir = output_dir / "octree"
    octree_dir.mkdir(exist_ok=True)

    hierarchy = {}
    for node_key, node_xyz in nodes.items():
        bin_path = octree_dir / f"{node_key}.bin"
        node_xyz.astype(np.float32).tofile(bin_path)
        hierarchy[node_key] = len(node_xyz)

    # Write metadata
    metadata = {
        "version": "2.0",
        "name": "occulus_export",
        "points": n,
        "boundingBox": {
            "min": bb_min.tolist(),
            "max": bb_max.tolist(),
        },
        "encoding": "DEFAULT",
        "scale": [0.001, 0.001, 0.001],
        "offset": bb_min.tolist(),
        "hierarchy": hierarchy,
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Potree export → %s (%d nodes)", meta_path, len(nodes))
    return meta_path


def _build_octree(
    xyz: NDArray[np.float64],
    origin: NDArray[np.float64],
    size: float,
    max_depth: int,
    max_points: int,
    key: str = "r",
    depth: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Recursively build an octree, returning a dict of node_key -> points."""
    nodes: dict[str, NDArray[np.float64]] = {}

    if len(xyz) == 0:
        return nodes

    if len(xyz) <= max_points or depth >= max_depth:
        nodes[key] = xyz
        return nodes

    half = size / 2

    for octant in range(8):
        ox = origin[0] + (half if octant & 1 else 0)
        oy = origin[1] + (half if octant & 2 else 0)
        oz = origin[2] + (half if octant & 4 else 0)

        mask = (
            (xyz[:, 0] >= ox)
            & (xyz[:, 0] < ox + half)
            & (xyz[:, 1] >= oy)
            & (xyz[:, 1] < oy + half)
            & (xyz[:, 2] >= oz)
            & (xyz[:, 2] < oz + half)
        )

        child_xyz = xyz[mask]
        if len(child_xyz) > 0:
            child_key = f"{key}{octant}"
            child_nodes = _build_octree(
                child_xyz,
                np.array([ox, oy, oz]),
                half,
                max_depth,
                max_points,
                child_key,
                depth + 1,
            )
            nodes.update(child_nodes)

    return nodes
