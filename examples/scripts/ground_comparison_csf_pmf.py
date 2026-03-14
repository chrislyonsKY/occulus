"""CSF vs PMF Ground Classification Comparison.

Loads a single USGS 3DEP ALS tile and runs both the Cloth Simulation Filter
(CSF) and Progressive Morphological Filter (PMF) ground classification
algorithms.  The script reports point counts for each method and computes the
agreement rate — the fraction of points both classifiers agree on.

Data source
-----------
USGS 3DEP — KY Metro Louisville B2 2019 (public domain).
URL : https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
      
      KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz

Usage
-----
    python examples/scripts/ground_comparison_csf_pmf.py
    python examples/scripts/ground_comparison_csf_pmf.py --input path/to/cloud.las
    python examples/scripts/ground_comparison_csf_pmf.py --no-viz
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEMO_URL = (
    "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
    "KY_CentralEast_A23/KY_CentralEast_1_A23/LAZ/USGS_LPC_KY_CentralEast_A23_N088E243.laz"
)


def _fetch(url: str, dest: Path) -> Path:
    """Download a remote tile, returning the cached local path.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Cache directory.

    Returns
    -------
    Path
        Local file path.
    """
    out = dest / Path(url).name
    if out.exists():
        logger.info("Cached: %s", out.name)
        return out
    logger.info("Downloading USGS 3DEP tile (~3 MB)…")
    try:
        urllib.request.urlretrieve(url, str(out))
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)
    return out


def main() -> None:
    """Run the CSF vs PMF comparison demo."""
    parser = argparse.ArgumentParser(description="CSF vs PMF ground classification comparison")
    parser.add_argument("--input", type=Path, default=None,
                        help="Local LAS/LAZ file (skips download)")
    parser.add_argument("--subsample", type=float, default=0.3,
                        help="Read-time subsample fraction (default 0.3)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip Open3D visualization")
    args = parser.parse_args()

    import numpy as np

    from occulus.io import read
    from occulus.segmentation import classify_ground_csf, classify_ground_pmf
    from occulus.types import AerialCloud

    cache_dir = Path(tempfile.gettempdir()) / "occulus_demo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    las_path = args.input or _fetch(_DEMO_URL, cache_dir)

    logger.info("Reading point cloud (subsample=%.0f%%)…", args.subsample * 100)
    cloud = read(las_path, platform="aerial", subsample=args.subsample)
    logger.info("Loaded: %s", cloud)

    # -- CSF ------------------------------------------------------------------
    logger.info("Running CSF ground classification…")
    csf_result = classify_ground_csf(cloud)

    # -- PMF ------------------------------------------------------------------
    logger.info("Running PMF ground classification…")
    pmf_result = classify_ground_pmf(cloud)

    # -- Comparison -----------------------------------------------------------
    print("\n=== Ground Classification Comparison ===")
    print(f"  Total points : {cloud.n_points:,}")

    if (
        isinstance(csf_result, AerialCloud) and csf_result.classification is not None
        and isinstance(pmf_result, AerialCloud) and pmf_result.classification is not None
    ):
        csf_ground = csf_result.classification == 2
        pmf_ground = pmf_result.classification == 2

        n_csf = int(csf_ground.sum())
        n_pmf = int(pmf_ground.sum())
        both_ground = int((csf_ground & pmf_ground).sum())
        either_ground = int((csf_ground | pmf_ground).sum())
        agreement = int((csf_ground == pmf_ground).sum())
        agreement_pct = agreement / cloud.n_points * 100

        print(f"\n  CSF ground points  : {n_csf:,}  ({n_csf / cloud.n_points * 100:.1f}%)")
        print(f"  PMF ground points  : {n_pmf:,}  ({n_pmf / cloud.n_points * 100:.1f}%)")
        print(f"\n  Both classify ground  : {both_ground:,}")
        print(f"  Either classifies ground: {either_ground:,}")
        print(f"  Agreement rate        : {agreement_pct:.2f}%")
        print(f"  Disagreement points   : {cloud.n_points - agreement:,}")

        # Jaccard similarity
        jaccard = both_ground / either_ground if either_ground > 0 else float("nan")
        print(f"  Jaccard similarity    : {jaccard:.4f}")
    else:
        print("  (classification arrays not available for comparison)")

    # -- Output image ---------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from _plot_style import CMAP_CATEGORY, apply_report_style, save_figure
        apply_report_style()
        xyz = cloud.xyz

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if (
            isinstance(csf_result, AerialCloud) and csf_result.classification is not None
            and isinstance(pmf_result, AerialCloud) and pmf_result.classification is not None
        ):
            csf_g = csf_result.classification == 2
            pmf_g = pmf_result.classification == 2
            cat = np.zeros(len(xyz), dtype=float)
            cat[csf_g & ~pmf_g] = 1
            cat[~csf_g & pmf_g] = 2
            cat[~csf_g & ~pmf_g] = 3
            sc = axes[0].scatter(xyz[:, 0], xyz[:, 1], c=cat,
                                 cmap=CMAP_CATEGORY, s=0.3, alpha=0.6, rasterized=True, vmin=0, vmax=3)
            axes[0].set_title("CSF vs PMF: both=blue, CSF only=orange, PMF only=green, neither=red")
        else:
            axes[0].scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
                            cmap="terrain", s=0.3, rasterized=True)
            axes[0].set_title("Eastern Kentucky — ALS Point Cloud")
        axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")

        axes[1].hist(xyz[:, 2], bins=60, color="#4169E1", alpha=0.75, edgecolor="white")
        axes[1].set_xlabel("Elevation (m)"); axes[1].set_ylabel("Point count")
        axes[1].set_title("Elevation Distribution")

        fig.suptitle(
            "USGS 3DEP LiDAR — CSF vs PMF Ground Classification, Eastern KY\n"
            f"Agreement: {agreement_pct:.1f}%  |  Jaccard: {jaccard:.3f}",
            fontsize=12, fontweight="bold",
        )
        _out_dir = Path(__file__).parent.parent / "outputs"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out = _out_dir / "ground_comparison_csf_pmf.png"
        save_figure(fig, out, alt_text=(
            "Two-panel figure comparing CSF and PMF ground classification: plan view "
            "colored by agreement category (both, CSF only, PMF only, neither) and "
            "elevation histogram."
        ))
        logger.info("Saved → %s", out)
        plt.close()
    except ImportError:
        logger.info("matplotlib not available — skipping image output.")

    # -- Visualization --------------------------------------------------------
    if not args.no_viz:
        try:
            from occulus.viz import visualize_segments
            if (
                isinstance(csf_result, AerialCloud) and csf_result.classification is not None
                and isinstance(pmf_result, AerialCloud) and pmf_result.classification is not None
            ):
                # Encode: 0=both ground, 1=CSF only, 2=PMF only, 3=neither
                labels = np.zeros(cloud.n_points, dtype="int32")
                csf_g = csf_result.classification == 2
                pmf_g = pmf_result.classification == 2
                labels[csf_g & ~pmf_g] = 1
                labels[~csf_g & pmf_g] = 2
                labels[~csf_g & ~pmf_g] = 3
                logger.info("Opening Open3D viewer…")
                visualize_segments(cloud, labels,
                                   window_name="CSF vs PMF (0=both, 1=CSF, 2=PMF, 3=neither)")
        except ImportError:
            logger.warning("open3d not installed — skipping visualization.")


if __name__ == "__main__":
    main()
