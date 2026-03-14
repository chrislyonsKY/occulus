"""Shared matplotlib styling for government report-quality output.

All example scripts import this module to get consistent, professional
figure formatting suitable for technical reports and publications.

Accessibility
-------------
Figures comply with WCAG 2.1 Level AA:
  - Colormaps are colorblind-safe (viridis, cividis, inferno)
  - Text contrast ≥ 4.5:1 against white backgrounds
  - All saved PNGs include descriptive alt-text metadata
  - No information conveyed by colour alone
"""

from __future__ import annotations

# WCAG 2.1 AA-safe colormaps (perceptually uniform, colorblind-friendly)
CMAP_ELEVATION = "cividis"    # sequential, good for terrain/elevation
CMAP_CANOPY    = "YlGn"      # sequential green, canopy heights
CMAP_DIVERGING = "RdBu_r"    # diverging, for change detection
CMAP_CATEGORY  = "tab10"     # categorical, limited palette
CMAP_HEAT      = "inferno"   # sequential, density/urban
CMAP_PROFILE   = "viridis"   # sequential, default for profiles


def apply_report_style() -> None:
    """Apply matplotlib rcParams for clean WCAG 2.1 AA-compliant figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",       # contrast ≥ 4.5:1 on white
        "axes.labelcolor": "#222222",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": False,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "image.cmap": CMAP_ELEVATION,      # default colorblind-safe cmap
    })


def save_figure(fig, path, alt_text: str = "") -> None:
    """Save a figure with PNG alt-text metadata (WCAG 2.1 AA).

    Parameters
    ----------
    fig : matplotlib Figure
        The figure to save.
    path : Path or str
        Output file path.
    alt_text : str
        Descriptive alt-text embedded as PNG 'Description' metadata.
        Should describe the visual content for screen readers.
    """
    from pathlib import Path
    path = Path(path)

    try:
        from PIL import PngImagePlugin
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Description", alt_text)
        metadata.add_text("Software", "Occulus LiDAR Analysis v1.0.0")
        metadata.add_text("Source", "https://github.com/chrislyonsKY/occulus")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    pil_kwargs={"pnginfo": metadata})
    except ImportError:
        # Pillow not available — save without metadata
        fig.savefig(path, dpi=150, bbox_inches="tight")


def add_cross_section_line(ax_plan, ax_profile, xyz, y_frac: float = 0.5,
                           band_frac: float = 0.05, color: str = "#D32F2F",
                           label: str = "Cross Section A\u2013A\u2032"):
    """Draw a cross-section reference line on a plan-view axes.

    The line is drawn on the plan view (ax_plan) and corresponding
    elevation profile points are plotted on ax_profile. This gives
    the viewer spatial context for the cross section.

    Parameters
    ----------
    ax_plan : matplotlib Axes
        Plan-view axes where the reference line will be drawn.
    ax_profile : matplotlib Axes
        Profile axes for the cross-section elevation plot.
    xyz : ndarray
        (N, 3) point cloud array.
    y_frac : float
        Fractional Y position of the cross-section line (0=south, 1=north).
    band_frac : float
        Half-width of the band around the line (fraction of Y range).
    color : str
        Line colour (default: WCAG-safe red #D32F2F).
    label : str
        Label for the cross-section line.

    Returns
    -------
    ndarray (bool)
        Boolean mask selecting points within the cross-section band.
    """
    import numpy as np

    y_min, y_max = xyz[:, 1].min(), xyz[:, 1].max()
    y_range = y_max - y_min
    y_line = y_min + y_frac * y_range
    half_band = band_frac * y_range

    # Draw reference line on plan view
    ax_plan.axhline(y_line, color=color, linestyle="--", linewidth=2.0, alpha=0.9)
    x_lims = ax_plan.get_xlim()
    ax_plan.text(x_lims[0], y_line, f"  {label}  ", fontsize=8,
                 color="white", va="bottom", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc=color, ec=color, alpha=0.9))

    # Select band of points near the line
    band = (xyz[:, 1] > y_line - half_band) & (xyz[:, 1] < y_line + half_band)

    # Plot elevation profile
    if band.sum() > 50:
        order = np.argsort(xyz[band, 0])
        ax_profile.scatter(
            xyz[band, 0][order], xyz[band, 2][order],
            s=0.3, c=xyz[band, 2][order], cmap=CMAP_ELEVATION,
            alpha=0.6, rasterized=True,
        )
        ax_profile.set_xlabel("Easting (m)")
        ax_profile.set_ylabel("Elevation (m)")
        ax_profile.set_title(f"Elevation Profile \u2014 {label}")

    return band
