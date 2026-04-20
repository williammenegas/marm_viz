"""
2D density heatmap with log compression and custom colormaps
=============================================================

Port of ``vis_colorful_v4_grid.m``. Builds a 2D histogram of
``(u, v)`` coordinates, smooths it with a Gaussian, normalizes to
``[0, 1]``, tone-maps with a log compression curve to reduce the
visual dominance of the peak bin, and plots the result with a
distinctive white → mid-colour → black colormap plus contour lines
at equal levels in the tone-mapped space.

The two available colormaps match the lab's reference figures:

* ``'purple'`` — white → lavender → purple → deep purple → black
* ``'green'``  — white → mint → medium green → deep teal → black

Usage::

    from marm_viz import plot_density_2d
    plot_density_2d(u, v, mid_color="purple")
    plot_density_2d(rec, mid_color="green", use_body_xy=True)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ..types import Recording, RecordingSet
from .sources import attach_sources
from ._overlays import make_colorbar


# -----------------------------------------------------------------------
# Custom colormaps (white → mid → black)
# -----------------------------------------------------------------------

_PURPLE_STOPS = np.array([0.00, 0.30, 0.60, 0.85, 1.00])
_PURPLE_RGB = np.array([
    [1.00, 1.00, 1.00],  # white
    [0.93, 0.88, 1.00],  # lavender
    [0.60, 0.40, 0.80],  # purple
    [0.30, 0.10, 0.40],  # deep purple
    [0.00, 0.00, 0.00],  # black
])

_GREEN_STOPS = np.array([0.00, 0.30, 0.60, 0.85, 1.00])
_GREEN_RGB = np.array([
    [1.00, 1.00, 1.00],  # white
    [0.90, 1.00, 0.90],  # mint
    [0.40, 0.80, 0.55],  # medium green
    [0.10, 0.60, 0.30],  # deep teal
    [0.00, 0.00, 0.00],  # black
])


def _white_mid_black(mid: str, n: int = 256, gamma: float = 1.0) -> LinearSegmentedColormap:
    """Build a white → mid → black colormap with optional gamma
    shaping. ``mid`` must be ``"purple"`` or ``"green"``.
    """
    mid = mid.lower().strip()
    if mid == "purple":
        stops, rgb = _PURPLE_STOPS, _PURPLE_RGB
    elif mid == "green":
        stops, rgb = _GREEN_STOPS, _GREEN_RGB
    else:
        raise ValueError(f"mid must be 'purple' or 'green', got {mid!r}")

    t = np.linspace(0, 1, n)
    tg = t ** gamma
    colors = np.empty((n, 3), dtype=float)
    for ch in range(3):
        colors[:, ch] = np.interp(tg, stops, rgb[:, ch])
    colors = np.clip(colors, 0.0, 1.0)
    return LinearSegmentedColormap.from_list(f"white_{mid}_black", colors)


# -----------------------------------------------------------------------
# Tone mapping
# -----------------------------------------------------------------------

def _log_tone_map(X: np.ndarray, alpha: float) -> np.ndarray:
    """Map X in ``[0, 1]`` to Y in ``[0, 1]`` with log compression.

    * ``alpha = 0`` → linear (Y = X)
    * ``alpha > 0`` → ``log1p(alpha*X) / log1p(alpha)``

    Higher alpha gives more aggressive compression so low-count
    bins become more visible next to the peak.
    """
    Xc = np.clip(X, 0.0, 1.0)
    if alpha <= 0:
        return Xc
    return np.log1p(alpha * Xc) / np.log1p(alpha)


# -----------------------------------------------------------------------
# Binning + smoothing
# -----------------------------------------------------------------------

def _binned_smooth_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    sigma_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """2D histogram + Gaussian smooth. Returns ``(raw, smoothed)``
    both with shape ``(len(x_edges) - 1, len(y_edges) - 1)``.
    """
    raw, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    smoothed = gaussian_filter(raw, sigma=max(0.5, sigma_px))
    return raw, smoothed


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def plot_density_2d(
    u: "Union[np.ndarray, Recording, RecordingSet]",
    v: "Optional[np.ndarray]" = None,
    *,
    use_body_xy: bool = False,
    x_max: float = 0.8,
    bin_width: float = 0.01,
    smooth_sigma: float = 0.02,
    n_levels: int = 7,
    mid_color: str = "purple",
    cmap_gamma: float = 1.0,
    log_alpha: float = 10.0,
    normalize_recording_xy: bool = True,
    title: "Optional[str]" = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Plot a 2D density heatmap with log compression and contours.

    Parameters
    ----------
    u, v:
        Either two 1D numpy arrays of coordinates to bin, OR a
        single :class:`Recording` / :class:`RecordingSet` (in which
        case ``v`` must be None and the coordinates are pulled from
        ``head_xy`` or ``body_xy``).
    use_body_xy:
        When ``u`` is a Recording/RecordingSet, use ``body_xy``
        instead of ``head_xy``. Ignored when arrays are passed
        directly.
    x_max:
        Upper limit of the plot/histogram domain in both x and y.
        The MATLAB default is 0.8 (assumes input is already
        normalized into ``[0, x_max]``).
    bin_width:
        Grid bin size in data units. Default 0.01 (80 × 80 bins for
        ``x_max = 0.8``).
    smooth_sigma:
        Gaussian smoothing width in data units. Converted to bin
        units internally.
    n_levels:
        Number of contour levels (equally spaced in tone-mapped
        space).
    mid_color:
        ``"purple"`` or ``"green"``. Picks the colormap family.
    cmap_gamma:
        Gamma shaping applied to the colormap (>1 darkens midtones).
    log_alpha:
        Log tone-map aggressiveness. 0 = linear; 2..10 = mild to
        strong compression. Default 10 (strong).
    normalize_recording_xy:
        When ``u`` is a Recording/RecordingSet, rescale the raw
        cage-centered pixel coordinates into ``[0, x_max]`` before
        binning. This mirrors the ``(Red + 400) / 1000`` transform
        in the MATLAB reference script, which was tuned for an
        ``[-400, +400]`` pixel range. When passing arrays directly
        the normalization is skipped.
    title:
        Optional figure title.
    figsize:
        Figure size in inches.
    """
    # --- extract coordinates ---
    if isinstance(u, (Recording, RecordingSet)):
        if v is not None:
            raise ValueError(
                "when passing a Recording/RecordingSet, v must be None"
            )
        if isinstance(u, RecordingSet):
            rec = u.concatenate()
        else:
            rec = u
        xy = rec.body_xy if use_body_xy else rec.head_xy
        if xy.size == 0:
            raise ValueError("recording has no xy data to plot")

        if normalize_recording_xy:
            # MATLAB: (Red + 400) / 1000  → maps [-400, 600] to [0, 1]
            # Use the same convention: add half-range then divide.
            half = 400.0
            full = 1000.0
            xs = (xy[:, 0] + half) / full
            ys = (xy[:, 1] + half) / full
        else:
            xs = xy[:, 0]
            ys = xy[:, 1]
    else:
        xs = np.asarray(u, dtype=float).ravel()
        if v is None:
            raise ValueError("when u is an array, v must also be an array")
        ys = np.asarray(v, dtype=float).ravel()
        if xs.shape != ys.shape:
            raise ValueError(
                f"u and v must have the same shape, got {xs.shape} vs {ys.shape}"
            )

    # Drop non-finite + out-of-range
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[finite]
    ys = ys[finite]

    # --- build grid ---
    edges = np.arange(0, x_max + bin_width / 2, bin_width)
    sigma_px = max(0.5, smooth_sigma / bin_width)
    _, smoothed = _binned_smooth_2d(xs, ys, edges, edges, sigma_px)

    # --- normalize + tone-map ---
    peak = max(smoothed.max(), np.finfo(float).eps)
    normed = smoothed / peak
    toned = _log_tone_map(normed, log_alpha)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # --- plot ---
    cmap = _white_mid_black(mid_color, n=256, gamma=cmap_gamma)
    rel_levels = np.linspace(0, 1, n_levels)

    fig, ax = plt.subplots(figsize=figsize)
    # np.histogram2d indexes as [x, y] so the array is already in
    # "row = x, col = y" order. For imshow we transpose so rows are
    # y and use origin='lower' so y increases upward.
    im = ax.imshow(
        toned.T,
        origin="lower",
        extent=(0, x_max, 0, x_max),
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    # Contours in the same tone-mapped space.
    cs = ax.contour(
        centers, centers, toned.T,
        levels=rel_levels,
        colors="black",
        linewidths=0.9,
        alpha=0.35,
    )

    ax.set_xlim(0, x_max); ax.set_ylim(0, x_max)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Density")
    ax.grid(True, alpha=0.15)
    ax.tick_params(direction="out")
    if title:
        ax.set_title(title, fontweight="bold")
    make_colorbar(im, ax, title="Density")

    # Sources: long-form grid (one row per bin) — x_center, y_center,
    # and both the raw (normed) density and the tone-mapped density
    # that's actually plotted. Users who want to re-apply a different
    # tone map can do so from the `density_normalized` column.
    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    sources: "dict[str, dict[str, np.ndarray]]" = {
        "panel_1_density": {
            "x_center": xx.ravel(),
            "y_center": yy.ravel(),
            "density_normalized": normed.ravel(),
            "density_tone_mapped": toned.ravel(),
        }
    }

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig
