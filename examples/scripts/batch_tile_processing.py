"""Batch LiDAR Tile Processing Pipeline.

Processes a directory of LAS/LAZ files in batch, applying a standardized
workflow to each tile:
  1. Read + subsample
  2. Statistical outlier removal
  3. Voxel downsample
  4. CSF ground classification
  5. Point density computation
  6. Write per-tile statistics to CSV

Designed for processing USGS 3DEP tile archives, NOAA coastal LiDAR
deliveries, or any collection of classified/unclassified LAS tiles.

Usage
-----
    python examples/scripts/batch_tile_processing.py --input-dir ./tiles/
    python examples/scripts/batch_tile_processing.py --input-dir ./tiles/ --pattern "*.laz"
    python examples/scripts/batch_tile_processing.py --input-dir ./tiles/ --workers 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUTS = Path(__file__).parent.parent / "outputs"


def process_tile(path: Path, voxel_size: float, subsample: float) -> dict:
    """Process a single tile and return a statistics dictionary.

    Parameters
    ----------
    path : Path
        Input LAS/LAZ file.
    voxel_size : float
        Voxel size for downsampling (metres).
    subsample : float
        Read-time subsample fraction.

    Returns
    -------
    dict
        Per-tile statistics.
    """
    from occulus.filters import statistical_outlier_removal, voxel_downsample
    from occulus.io import read
    from occulus.metrics import compute_cloud_statistics
    from occulus.segmentation import classify_ground_csf
    from occulus.types import AerialCloud

    t0 = time.perf_counter()
    row: dict = {"file": path.name, "error": ""}

    try:
        cloud = read(path, platform="aerial", subsample=subsample)
        row["n_raw"] = cloud.n_points

        clean, _mask = statistical_outlier_removal(cloud, nb_neighbors=16, std_ratio=2.5)
        ds = voxel_downsample(clean, voxel_size=voxel_size)
        row["n_downsampled"] = ds.n_points

        classified = classify_ground_csf(ds)
        if isinstance(classified, AerialCloud) and classified.classification is not None:
            n_g = int((classified.classification == 2).sum())
            row["n_ground"] = n_g
            row["pct_ground"] = round(n_g / ds.n_points * 100, 1)

        stats = compute_cloud_statistics(cloud)
        row["z_min"] = round(stats.z_min, 3)
        row["z_max"] = round(stats.z_max, 3)
        row["z_mean"] = round(stats.z_mean, 3)
        row["z_std"] = round(stats.z_std, 3)
        if stats.intensity_mean is not None:
            row["intensity_mean"] = round(stats.intensity_mean, 2)

        row["elapsed_s"] = round(time.perf_counter() - t0, 2)
        logger.info("  ✓ %s — %d pts, %.2f s", path.name, cloud.n_points, row["elapsed_s"])

    except Exception as exc:
        row["error"] = str(exc)
        logger.error("  ✗ %s — %s", path.name, exc)

    return row


def main() -> None:
    """Run batch tile processing."""
    parser = argparse.ArgumentParser(description="Batch LiDAR tile processing")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of LAS/LAZ files")
    parser.add_argument("--pattern", default="*.las", help="Glob pattern (default: *.las)")
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--subsample", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = parser.parse_args()

    tiles = sorted(args.input_dir.glob(args.pattern))
    if not tiles:
        logger.error("No files matching %s in %s", args.pattern, args.input_dir)
        return

    logger.info("Found %d tiles to process.", len(tiles))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, tile in enumerate(tiles, 1):
        logger.info("[%d/%d] %s", i, len(tiles), tile.name)
        rows.append(process_tile(tile, args.voxel_size, args.subsample))

    # Write CSV
    csv_path = args.output_dir / "batch_statistics.csv"
    if rows:
        fieldnames = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Statistics → %s", csv_path)

    success = sum(1 for r in rows if not r.get("error"))
    print(f"\nProcessed {success}/{len(tiles)} tiles successfully.")
    print(f"CSV report → {csv_path}")

    # Summary plot
    if success > 1:
        try:
            import matplotlib.pyplot as plt

            names = [r["file"][:20] for r in rows if not r.get("error")]
            counts = [r.get("n_raw", 0) for r in rows if not r.get("error")]
            _fig, ax = plt.subplots(figsize=(max(8, len(names)), 4))
            ax.bar(range(len(names)), counts, color="steelblue")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Point count (raw)")
            ax.set_title(f"Batch Processing — {len(names)} Tiles")
            plt.tight_layout()
            out = args.output_dir / "batch_point_counts.png"
            plt.savefig(out, dpi=150)
            logger.info("Chart → %s", out)
            plt.close()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
