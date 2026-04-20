"""
Category fraction + occupancy + stay/move heatmaps
===================================================

Port of ``plot_video_data_2.m``. Three separate figures:

1. :func:`plot_category_fraction_heatmaps` — 5 panels, one per
   broad behaviour category. Each bin shows the fraction of total
   frames spent in that bin in that category. All panels share a
   common colour scale so category usage is comparable at a glance.
2. :func:`plot_xy_occupancy_heatmap` — 1 panel, fraction of total
   frames in each bin (the marginal over all categories).
3. :func:`plot_stay_move_heatmaps` — 2 panels, P(stay | in bin)
   and Probability of moving. A frame is "staying" if the next frame
   lands in the same bin; "moving" otherwise. Useful for spotting
   preferred rest locations vs transit corridors.

All three use the same binning convention (``round(xy / bin_size)``)
so bins line up across figures.
"""

from __future__ import annotations

from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..types import Recording, RecordingSet
from ..collect.state_remap import CATEGORY_COLORS, CATEGORY_NAMES
from .sources import attach_sources
from ._overlays import draw_cage_squares, draw_stim_circle, make_colorbar


def _to_recording(data: Union[Recording, RecordingSet]) -> Recording:
    if isinstance(data, RecordingSet):
        return data.concatenate()
    return data


def _bin_xy(xy: np.ndarray, bin_size: float) -> np.ndarray:
    """Divide by ``bin_size`` and round, preserving NaNs."""
    out = xy.astype(float).copy()
    m = np.isfinite(out)
    out[m] = np.round(out[m] / bin_size)
    return out


def _ramp_to_color(rgb: Tuple[float, float, float], n: int = 256) -> LinearSegmentedColormap:
    """Linear white → ``rgb`` colormap. Used by the category panels."""
    white = np.ones(3)
    stops = np.linspace(0, 1, n)[:, None]
    colors = (1 - stops) * white + stops * np.asarray(rgb)
    return LinearSegmentedColormap.from_list("ramp", colors)


def _prep_binned(
    rec: Recording,
    bin_size: float,
    use_head_xy: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """Bin positions and gather valid (x_bin, y_bin, category) samples.
    Returns ``(xb, yb, cat, x_min, y_min, nx, ny)``.
    """
    xy = rec.head_xy if use_head_xy else rec.body_xy
    cat = rec.behav_cat

    xy_b = _bin_xy(xy, bin_size)
    xb_all = xy_b[:, 0]
    yb_all = xy_b[:, 1]

    good = (
        np.isfinite(xb_all)
        & np.isfinite(yb_all)
        & np.isfinite(cat)
        & (cat >= 1)
        & (cat <= 5)
    )
    xb = xb_all[good].astype(int)
    yb = yb_all[good].astype(int)
    cat_g = cat[good].astype(int)

    if xb.size == 0:
        return xb, yb, cat_g, 0, 0, 0, 0

    x_min = int(xb.min()); x_max = int(xb.max())
    y_min = int(yb.min()); y_max = int(yb.max())
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    return xb, yb, cat_g, x_min, y_min, nx, ny


def plot_category_fraction_heatmaps(
    data: Union[Recording, RecordingSet],
    *,
    bin_size: float = 20.0,
    use_head_xy: bool = True,
    stim_radius_px: float = 120.0,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """6-panel heatmap: one panel per broad behaviour category (5)
    plus a 6th "near stimulus" panel.

    The first 5 panels colour each bin by ``fraction of total frames
    in that bin in that category`` and share a common colour scale
    so category magnitudes are directly comparable.

    The 6th panel (bottom-right) colours each bin by the fraction
    of frames in that bin where the head is within
    ``stim_radius_px`` of the stimulus, regardless of behaviour
    category. Rendered with a black colour ramp to visually
    distinguish it from the behaviour-category panels above.

    Zero-count bins are rendered transparent (NaN mask). The
    "near-stim" panel degrades to empty if ``dist_to_stim`` is all
    NaN for the input (which happens when the stimulus position
    was never annotated for any of the recordings being pooled).
    """
    rec = _to_recording(data)
    xb, yb, cat, x_min_cat, y_min_cat, nx_cat, ny_cat = _prep_binned(
        rec, bin_size, use_head_xy
    )

    # Also bin every frame (regardless of category) for the near-stim
    # panel. This panel cares about position + dist_to_stim, not about
    # the cluster label, so it has a different valid-frame mask than
    # the category panels.
    xy_all = rec.head_xy if use_head_xy else rec.body_xy
    ds_all = rec.dist_to_stim
    xy_all_b = _bin_xy(xy_all, bin_size)
    good_stim = (
        np.isfinite(xy_all_b[:, 0])
        & np.isfinite(xy_all_b[:, 1])
        & np.isfinite(ds_all)
    )
    xb_stim = xy_all_b[good_stim, 0].astype(int)
    yb_stim = xy_all_b[good_stim, 1].astype(int)
    ds_stim = ds_all[good_stim]

    # Unified bin extent across category + near-stim samples so all
    # 6 panels share the same spatial axes.
    if xb.size and xb_stim.size:
        x_min = int(min(x_min_cat, xb_stim.min()))
        x_max = int(max(x_min_cat + nx_cat - 1, xb_stim.max()))
        y_min = int(min(y_min_cat, yb_stim.min()))
        y_max = int(max(y_min_cat + ny_cat - 1, yb_stim.max()))
    elif xb.size:
        x_min = x_min_cat
        x_max = x_min_cat + nx_cat - 1
        y_min = y_min_cat
        y_max = y_min_cat + ny_cat - 1
    elif xb_stim.size:
        x_min = int(xb_stim.min()); x_max = int(xb_stim.max())
        y_min = int(yb_stim.min()); y_max = int(yb_stim.max())
    else:
        x_min = y_min = 0; x_max = y_max = 0
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    xy_label = "head" if use_head_xy else "body"
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    if cat.size == 0 and xb_stim.size == 0:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(
            0.5, 0.5, "No finite samples",
            ha="center", va="center", transform=axes[0, 0].transAxes,
        )
        return fig

    # Build the 5 category fraction grids sharing a colour scale.
    total_frames = cat.size
    counts = np.zeros((5, ny, nx), dtype=float)
    if cat.size > 0:
        for k_idx, k in enumerate(range(1, 6)):
            m = cat == k
            if not m.any():
                continue
            xi = xb[m] - x_min
            yi = yb[m] - y_min
            np.add.at(counts[k_idx], (yi, xi), 1.0)
    frac = counts / max(total_frames, 1)
    global_max = frac.max() if frac.size > 0 else 1.0
    if not np.isfinite(global_max) or global_max <= 0:
        global_max = 1.0

    extent = (
        x_min - 0.5, x_min + nx - 0.5,
        y_min - 0.5, y_min + ny - 0.5,
    )

    for k_idx, k in enumerate(range(1, 6)):
        ax = axes.flat[k_idx]
        grid = frac[k_idx].copy()
        grid[grid == 0] = np.nan  # transparent where zero
        im = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=_ramp_to_color(CATEGORY_COLORS[k]),
            vmin=0,
            vmax=0.002,
            interpolation="nearest",
        )
        ax.grid(True, alpha=0.15)
        draw_cage_squares(ax, scale=1.0)
        ax.set_xlim(-22, 22); ax.set_ylim(-22, 22)
        ax.tick_params(pad=5)
        ax.set_xlabel("X bin")
        ax.set_ylabel("Y bin", rotation=0, ha="right")
        ax.set_title(f"{CATEGORY_NAMES[k].title()}", fontsize=10)
        make_colorbar(im, ax, title="Fraction")

    # --- Panel 6: near stimulus ---
    # Show total position distribution as a white→black occupancy
    # heatmap, with the stimulus proximity boundary overlaid as a
    # dotted circle.
    ax6 = axes.flat[5]
    # Build an occupancy grid from ALL valid-xy frames (regardless of
    # category label), so the distribution includes every frame.
    xy_all = rec.head_xy if use_head_xy else rec.body_xy
    xy_all_b = _bin_xy(xy_all, bin_size)
    good_all = np.isfinite(xy_all_b[:, 0]) & np.isfinite(xy_all_b[:, 1])
    xb_all_valid = xy_all_b[good_all, 0].astype(int)
    yb_all_valid = xy_all_b[good_all, 1].astype(int)

    occ_grid = np.zeros((ny, nx), dtype=float)
    if xb_all_valid.size > 0:
        np.add.at(occ_grid, (yb_all_valid - y_min, xb_all_valid - x_min), 1.0)
    total_occ = max(occ_grid.sum(), 1)
    occ_frac = occ_grid / total_occ
    occ_frac[occ_frac == 0] = np.nan

    if np.isfinite(occ_frac).any():
        im6 = ax6.imshow(
            occ_frac,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=_ramp_to_color((0.0, 0.0, 0.0)),
            interpolation="nearest",
        )
        make_colorbar(im6, ax6, title="Fraction")
    # Overlay stim radius circle in bin-index space.
    stim_xy = rec.stim_xy_centered
    if np.all(np.isfinite(stim_xy)):
        stim_bx = stim_xy[0] / bin_size
        stim_by = stim_xy[1] / bin_size
        radius_bins = stim_radius_px / bin_size
        draw_stim_circle(
            ax6,
            np.array([stim_bx, stim_by]),
            radius_bins,
            linewidth=2.0,
        )
    ax6.grid(True, alpha=0.15)
    draw_cage_squares(ax6, scale=1.0)
    ax6.set_xlim(-22, 22); ax6.set_ylim(-22, 22)
    ax6.tick_params(pad=5)
    ax6.set_xlabel("X bin")
    ax6.set_ylabel("Y bin", rotation=0, ha="right")
    ax6.set_title(f"near stimulus (<{stim_radius_px:g} px)", fontsize=10)

    # Build sources: one panel per behaviour category plus near-stim,
    # long-format (x_bin_center, y_bin_center, value) for non-zero /
    # finite bins.
    sources: "dict[str, dict[str, np.ndarray]]" = {}
    for k_idx, k in enumerate(range(1, 6)):
        grid = frac[k_idx]
        nz_y, nz_x = np.nonzero(grid)
        if nz_y.size == 0:
            sources[f"panel_{k}_{CATEGORY_NAMES[k].title()}"] = {
                "x_bin": np.zeros(0),
                "y_bin": np.zeros(0),
                "fraction": np.zeros(0),
            }
            continue
        sources[f"panel_{k}_{CATEGORY_NAMES[k].title()}"] = {
            "x_bin": ((nz_x + x_min) * bin_size).astype(float),
            "y_bin": ((nz_y + y_min) * bin_size).astype(float),
            "fraction": grid[nz_y, nz_x],
        }

    nz_y, nz_x = np.nonzero(np.isfinite(occ_frac))
    sources["panel_6_near_stimulus"] = {
        "x_bin": ((nz_x + x_min) * bin_size).astype(float),
        "y_bin": ((nz_y + y_min) * bin_size).astype(float),
        "occupancy_fraction": occ_frac[nz_y, nz_x],
    }

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig


def plot_xy_occupancy_heatmap(
    data: Union[Recording, RecordingSet],
    *,
    bin_size: float = 20.0,
    use_head_xy: bool = True,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Single-panel occupancy heatmap: fraction of total frames in
    each xy bin, regardless of behaviour category. Uses matplotlib's
    ``plasma`` colormap.
    """
    rec = _to_recording(data)
    xb, yb, cat, x_min, y_min, nx, ny = _prep_binned(rec, bin_size, use_head_xy)

    xy_label = "head" if use_head_xy else "body"
    fig, ax = plt.subplots(figsize=figsize)

    if xb.size == 0:
        ax.text(0.5, 0.5, "No finite xy samples", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    grid = np.zeros((ny, nx), dtype=float)
    np.add.at(grid, (yb - y_min, xb - x_min), 1.0)
    total = grid.sum()
    grid = grid / max(total, 1)

    extent = (
        x_min - 0.5, x_min + nx - 0.5,
        y_min - 0.5, y_min + ny - 0.5,
    )
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="plasma",
        vmax=0.008,
        interpolation="nearest",
    )
    ax.grid(True, alpha=0.15)
    draw_cage_squares(ax, scale=1.0)
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
    ax.tick_params(pad=5)
    ax.set_xlabel(f"X bin")
    ax.set_ylabel("Y bin", rotation=0, ha="right")
    ax.set_title(f"Occupancy ")
    make_colorbar(im, ax, title="Fraction")

    # Sources: long-form for non-zero bins.
    nz_y, nz_x = np.nonzero(grid > 0)
    sources: "dict[str, dict[str, np.ndarray]]" = {
        "panel_1_occupancy": {
            "x_bin": ((nz_x + x_min) * bin_size).astype(float),
            "y_bin": ((nz_y + y_min) * bin_size).astype(float),
            "fraction": grid[nz_y, nz_x],
        }
    }

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig


def plot_stay_move_heatmaps(
    data: Union[Recording, RecordingSet],
    *,
    bin_size: float = 20.0,
    use_head_xy: bool = True,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Single-panel ``Probability of moving`` heatmap.

    A frame counts as "staying" if the *next* frame's binned xy
    lands in the same bin; "moving" otherwise. The plotted value
    is ``Probability of moving`` = 1 - P(stay | in bin), coloured with
    plasma on the ``[0, 1]`` probability scale. Bins with no valid
    transitions are rendered transparent.
    """
    rec = _to_recording(data)
    xy = rec.head_xy if use_head_xy else rec.body_xy

    fig, ax = plt.subplots(figsize=figsize)

    if xy.size == 0:
        ax.text(0.5, 0.5, "Missing xy", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    xy_b = _bin_xy(xy, bin_size)
    xb = xy_b[:, 0]; yb = xy_b[:, 1]

    finite = np.isfinite(xb) & np.isfinite(yb)
    if not finite.any():
        ax.text(0.5, 0.5, "No valid transitions", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Pairs: both current and next frame must be finite
    good_now = finite.copy()
    good_next = np.concatenate([finite[1:], [False]])
    good_pair = good_now & good_next
    good_pair[-1] = False

    if not good_pair.any():
        ax.text(0.5, 0.5, "No valid transitions", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    xb0 = xb[good_pair].astype(int)
    yb0 = yb[good_pair].astype(int)
    next_idx = np.where(good_pair)[0] + 1
    xb1 = xb[next_idx].astype(int)
    yb1 = yb[next_idx].astype(int)

    # Grid covers all finite bins (so axes match the occupancy figure)
    xb_all = xb[finite].astype(int)
    yb_all = yb[finite].astype(int)
    x_min = int(xb_all.min()); x_max = int(xb_all.max())
    y_min = int(yb_all.min()); y_max = int(yb_all.max())
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1

    denom = np.zeros((ny, nx), dtype=float)
    stay = np.zeros((ny, nx), dtype=float)
    is_stay = (xb1 == xb0) & (yb1 == yb0)
    np.add.at(denom, (yb0 - y_min, xb0 - x_min), 1.0)
    np.add.at(stay, (yb0 - y_min, xb0 - x_min), is_stay.astype(float))

    with np.errstate(invalid="ignore", divide="ignore"):
        p_stay = np.where(denom > 0, stay / denom, np.nan)
    p_move = np.where(np.isfinite(p_stay), 1.0 - p_stay, 0.0)
    # Mask bins with fewer than 100 observations.
    p_move[denom < 100] = 0.0

    extent = (
        x_min - 0.5, x_min + nx - 0.5,
        y_min - 0.5, y_min + ny - 0.5,
    )
    im = ax.imshow(
        p_move,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="plasma",
        vmin=0,
        vmax=0.3,
        interpolation="nearest",
    )
    ax.grid(True, alpha=0.15)
    draw_cage_squares(ax, scale=1.0)
    ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
    ax.tick_params(pad=5)
    ax.set_xlabel("X bin")
    ax.set_ylabel("Y bin", rotation=0, ha="right")
    ax.set_title("Probability of moving")
    make_colorbar(im, ax, title="Probability")

    # Sources: long-form (x_bin_center, y_bin_center, probability).
    nz_y, nz_x = np.nonzero(~np.isnan(p_move))
    sources: "dict[str, dict[str, np.ndarray]]" = {
        "panel_1_p_move": {
            "x_bin": ((nz_x + x_min) * bin_size).astype(float),
            "y_bin": ((nz_y + y_min) * bin_size).astype(float),
            "probability": p_move[nz_y, nz_x],
        }
    }

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig
