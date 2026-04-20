"""
Plot overlays (cage boundary, stim circle, etc.)
=================================================

Shared drawing helpers that multiple plot modules use to overlay
cage boundaries and stimulus markers on their axes.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter


def _sig1_formatter(x, pos):
    """Format a tick value with 1 significant digit."""
    if x == 0:
        return "0"
    import math
    digits = -int(math.floor(math.log10(abs(x))))
    return f"{x:.{max(digits, 0)}f}"


_SIG1 = FuncFormatter(_sig1_formatter)


def make_colorbar(mappable, ax, *, title="", shrink=0.6):
    """Create a colorbar with horizontal title and 1-sig-digit ticks."""
    import matplotlib.pyplot as plt
    cb = plt.colorbar(mappable, ax=ax, shrink=shrink)
    if title:
        cb.ax.set_title(title, fontsize=7)
    cb.ax.yaxis.set_major_formatter(_SIG1)
    return cb


def draw_cage_squares(ax, scale: float = 1.0, **kwargs) -> None:
    """Draw two concentric cage-boundary squares centered at (0, 0).

    The inner square has side length 10 (half-side 5) and the outer
    square has side length 20 (half-side 10), both multiplied by
    ``scale``. For heatmap plots in bin-index space, use ``scale=1``.
    For pixel-space plots, use ``scale=bin_size`` (typically 20).

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    scale:
        Multiplier applied to the square dimensions. Default 1.0.
    """
    defaults = dict(
        linewidth=1.0,
        edgecolor="#888888",
        facecolor="none",
        linestyle="--",
        zorder=1,
    )
    defaults.update(kwargs)
    for half_side in (10.0, 20.0):
        hs = half_side * scale
        rect = Rectangle(
            (-hs, -hs), 2 * hs, 2 * hs,
            **{**defaults},  # fresh copy so patches don't share state
        )
        ax.add_patch(rect)


def draw_stim_circle(
    ax,
    stim_xy: np.ndarray,
    radius_px: float,
    **kwargs,
) -> None:
    """Draw a dotted circle showing the stimulus proximity radius.

    Does nothing if ``stim_xy`` contains NaN (no stim annotated).
    """
    if not np.all(np.isfinite(stim_xy)):
        return
    defaults = dict(
        linewidth=1.0,
        edgecolor="black",
        facecolor="none",
        linestyle=":",
        zorder=2,
    )
    defaults.update(kwargs)
    circ = Circle(
        (float(stim_xy[0]), float(stim_xy[1])),
        float(radius_px),
        **defaults,
    )
    ax.add_patch(circ)
