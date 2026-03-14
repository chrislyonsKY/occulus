"""LAS/LAZ to XYZ/CSV Format Converter.

Converts one or more LAS or LAZ point cloud files to space-delimited XYZ or
comma-delimited CSV format.  Useful for importing LiDAR data into GIS software,
spreadsheets, or analysis pipelines that do not support LAS directly.

Attributes written (when present): X Y Z Intensity Classification ReturnNumber

Usage
-----
    python examples/scripts/las_to_xyz_converter.py input.las
    python examples/scripts/las_to_xyz_converter.py *.las --format csv --output-dir ./converted
    python examples/scripts/las_to_xyz_converter.py cloud.laz --subsample 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def convert_file(src: Path, dest_dir: Path, fmt: str, subsample: float | None) -> Path:
    """Convert a single LAS/LAZ file to XYZ or CSV.

    Parameters
    ----------
    src : Path
        Input LAS/LAZ file.
    dest_dir : Path
        Directory for output file.
    fmt : str
        ``"xyz"`` or ``"csv"``.
    subsample : float | None
        Optional subsample fraction.

    Returns
    -------
    Path
        Output file path.
    """
    from occulus.io import read, write

    ext = ".csv" if fmt == "csv" else ".xyz"
    dest = dest_dir / (src.stem + ext)

    logger.info("Reading %s…", src.name)
    cloud = read(src, subsample=subsample)
    logger.info("  %s — %d points", src.name, cloud.n_points)

    write(cloud, dest)
    logger.info("  → %s", dest)
    return dest


def main() -> None:
    """Run the LAS→XYZ/CSV converter."""
    parser = argparse.ArgumentParser(description="Convert LAS/LAZ to XYZ or CSV")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input LAS/LAZ files")
    parser.add_argument(
        "--format", choices=["xyz", "csv"], default="xyz",
        help="Output format (default: xyz)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument("--subsample", type=float, default=None,
                        help="Subsample fraction (0–1)")
    args = parser.parse_args()

    converted = []
    for src in args.inputs:
        if not src.exists():
            logger.error("Not found: %s", src)
            continue
        dest_dir = args.output_dir or src.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            out = convert_file(src, dest_dir, args.format, args.subsample)
            converted.append(out)
        except Exception as exc:
            logger.error("Failed %s: %s", src.name, exc)

    print(f"\nConverted {len(converted)} file(s).")
    for p in converted:
        print(f"  {p}")


if __name__ == "__main__":
    main()
