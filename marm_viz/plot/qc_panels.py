"""
QC panel figures
================

8-panel "at a glance" QC figure for a recording: head position
coloured by depth / distance-to-stim / angle-to-stim, behaviour
category scatter, and histograms of those three quantities. Port of
``plot_video_data_1.m``'s ``qc_plots_for_recording_depths_randomsample``.

Usage::

    from marm_viz import plot_qc_panels
    plot_qc_panels(rec)                   # single recording
    plot_qc_panels(rec_set)               # pooled across a RecordingSet
    plot_qc_panels(rec, n_frames=20_000)  # random-sample for speed
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..types import Recording, RecordingSet
from ..collect.state_remap import CATEGORY_COLORS, CATEGORY_NAMES
from .sources import attach_sources
from ._overlays import draw_cage_squares, make_colorbar


def _white_purple_black(n: int = 256) -> LinearSegmentedColormap:
    """White → purple → black colormap for continuous QC scatters."""
    colors = [
        (1.0, 1.0, 1.0),       # white
        (0.5, 0.0, 0.5),       # purple
        (0.0, 0.0, 0.0),       # black
    ]
    return LinearSegmentedColormap.from_list("wpb", colors, N=n)


def _to_recording(data: Union[Recording, RecordingSet]) -> Recording:
    """Collapse a RecordingSet into a single augmented+concatenated
    Recording. Single Recordings are returned unchanged.
    """
    if isinstance(data, RecordingSet):
        return data.concatenate()
    return data


def _random_frame_indices(n: int, n_plot: Optional[int]) -> np.ndarray:
    """Return a sorted random subset of frame indices. If
    ``n_plot`` is None or >= ``n``, returns all indices.
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n_plot is None or n_plot >= n:
        return np.arange(n)
    rng = np.random.default_rng()
    return np.sort(rng.choice(n, size=n_plot, replace=False))


def plot_qc_panels(
    data: Union[Recording, RecordingSet],
    *,
    n_frames: Optional[int] = 50_000,
    depth_clim: tuple = (0.0, 20.0),
    point_size: float = 6.0,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Create the 8-panel QC figure for a recording or recording set.

    Panels (row-major, 2x4 grid):

    1. Head XY coloured by depth spread (Q75-Q25).
    2. Head XY coloured by distance to stimulus.
    3. Head XY coloured by head-direction angle to stimulus.
    4. Head XY coloured by broad behaviour category.
    5. Histogram of depth spread.
    6. Histogram of distance to stimulus.
    7. Histogram of angle to stimulus.
    8. Bar chart of behaviour-category usage (fraction of frames).

    The four scatter panels have no colorbars or legends — the
    distributions (panels 5-8 directly below them) tell the same
    story more quantitatively, so redundant legends just crowd the
    plot.

    Source data for every panel is attached to the returned figure
    via :func:`marm_viz.plot.sources.attach_sources`. The CLI writes
    it as ``*_source_panel_*.csv`` alongside the PNG; library users
    can retrieve it via :func:`marm_viz.plot.sources.get_sources` or
    :func:`marm_viz.plot.sources.save_sources_to_csv`.

    Parameters
    ----------
    data:
        A single :class:`~marm_viz.types.Recording` or a
        :class:`~marm_viz.types.RecordingSet`. Recording sets are
        flipped-augmented and concatenated before plotting.
    n_frames:
        Random-sample this many frames before plotting. Use None
        to plot everything (slow for big concatenations). Defaults
        to 50,000.
    depth_clim:
        ``(vmin, vmax)`` colour limits for the depth panel. Defaults
        to the ``[0, 20]`` range used by the MATLAB reference plot.
    point_size:
        Scatter marker area.
    figsize:
        Figure size in inches.
    """
    rec = _to_recording(data)
    idx = _random_frame_indices(rec.n_frames, n_frames)

    head = rec.head_xy[idx]
    dm = rec.depth_metric[idx]
    ds = rec.dist_to_stim[idx]
    ang = rec.angle_to_stim[idx]
    cat = rec.behav_cat[idx]

    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # We collect per-panel source arrays as we go so the caller can
    # dump them to CSVs via sources.save_sources_to_csv().
    sources: "dict[str, dict[str, np.ndarray]]" = {}

    # --- row 1: scatter panels with colorbars ---
    ax = axes[0, 0]
    m = np.all(np.isfinite(head), axis=1) & np.isfinite(dm)
    sc = ax.scatter(
        head[m, 0], head[m, 1], s=point_size, c=dm[m],
        cmap=_white_purple_black(), vmin=depth_clim[0], vmax=depth_clim[1],
    )
    make_colorbar(sc, ax, title="Depth")
    draw_cage_squares(ax, scale=20.0)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.set_xlim(-420, 420); ax.set_ylim(-420, 420)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Depth Q75-Q25")
    sources["panel_1_xy_depth"] = {
        "x": head[m, 0], "y": head[m, 1], "depth_q75_minus_q25": dm[m],
    }

    ax = axes[0, 1]
    m = np.all(np.isfinite(head), axis=1) & np.isfinite(ds)
    if m.any():
        sc = ax.scatter(head[m, 0], head[m, 1], s=point_size, c=ds[m],
                   cmap=_white_purple_black().reversed())
        make_colorbar(sc, ax, title="Distance (px)")
    draw_cage_squares(ax, scale=20.0)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.set_xlim(-420, 420); ax.set_ylim(-420, 420)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Dist to stim")
    sources["panel_2_xy_dist_to_stim"] = {
        "x": head[m, 0], "y": head[m, 1], "dist_to_stim": ds[m],
    }

    ax = axes[0, 2]
    m = np.all(np.isfinite(head), axis=1) & np.isfinite(ang)
    if m.any():
        sc = ax.scatter(head[m, 0], head[m, 1], s=point_size, c=ang[m],
                   cmap=_white_purple_black())
        make_colorbar(sc, ax, title="Angle (deg)")
    draw_cage_squares(ax, scale=20.0)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.set_xlim(-420, 420); ax.set_ylim(-420, 420)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Head→stim angle")
    sources["panel_3_xy_angle_to_stim"] = {
        "x": head[m, 0], "y": head[m, 1], "angle_to_stim_deg": ang[m],
    }

    ax = axes[0, 3]
    good = np.all(np.isfinite(head), axis=1) & np.isfinite(cat)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cat_cmap = ListedColormap([CATEGORY_COLORS[k] for k in range(1, 6)])
    cat_bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    cat_norm = BoundaryNorm(cat_bounds, cat_cmap.N)
    if good.any():
        sc_cat = ax.scatter(
            head[good, 0], head[good, 1], s=point_size,
            c=cat[good], cmap=cat_cmap, norm=cat_norm, edgecolor="none",
        )
        cb = plt.colorbar(sc_cat, ax=ax, shrink=0.6, ticks=[1, 2, 3, 4, 5])
        cb.ax.set_yticklabels(
            ["climb", "active", "alone", "near-act", "near-rest"],
            fontsize=6,
        )
    draw_cage_squares(ax, scale=20.0)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.set_xlim(-420, 420); ax.set_ylim(-420, 420)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Behavioral State")
    sources["panel_4_xy_behav_category"] = {
        "x": head[good, 0],
        "y": head[good, 1],
        "behav_cat": cat[good],
    }

    # --- row 2: histograms + bar chart ---
    ax = axes[1, 0]
    dm_finite = dm[np.isfinite(dm)]
    if dm_finite.size > 0:
        counts, edges, _ = ax.hist(dm_finite, bins=50, color="#555555")
        sources["panel_5_hist_depth"] = {
            "bin_left_edge": edges[:-1],
            "bin_right_edge": edges[1:],
            "count": counts,
        }
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Depth Q75-Q25"); ax.set_ylabel("Count")
    ax.set_title("Hist: depth")

    ax = axes[1, 1]
    ds_finite = ds[np.isfinite(ds)]
    if ds_finite.size > 0:
        counts, edges, _ = ax.hist(ds_finite, bins=50, color="#555555")
        sources["panel_6_hist_dist_to_stim"] = {
            "bin_left_edge": edges[:-1],
            "bin_right_edge": edges[1:],
            "count": counts,
        }
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Dist to stim"); ax.set_ylabel("Count")
    ax.set_title("Hist: dist-to-stim")

    ax = axes[1, 2]
    ang_finite = ang[np.isfinite(ang)]
    if ang_finite.size > 0:
        counts, edges, _ = ax.hist(ang_finite, bins=50, color="#555555")
        sources["panel_7_hist_angle_to_stim"] = {
            "bin_left_edge": edges[:-1],
            "bin_right_edge": edges[1:],
            "count": counts,
        }
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlabel("Head->stim angle (deg)"); ax.set_ylabel("Count")
    ax.set_title("Hist: angle-to-stim")

    ax = axes[1, 3]
    finite_cat = cat[np.isfinite(cat)]
    counts = np.array([(finite_cat == k).sum() for k in range(1, 6)])
    total = max(counts.sum(), 1)
    frac = counts / total
    bar_colors = [CATEGORY_COLORS[k] for k in range(1, 6)]
    ax.bar(range(1, 6), frac, color=bar_colors)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(
        ["climb", "active", "alone", "near-act", "near-rest"],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_ylabel("Fraction of frames")
    ax.set_title("Behaviour usage")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, axis="y")
    # Note: category_name column has object dtype (strings) which
    # save_sources_to_csv converts via float() — which would fail.
    # So we emit a numeric-only source here; the mapping
    # 1=climbing, 2=active, 3=alone, 4=near-active, 5=near-resting
    # is in the CATEGORY_NAMES constant.
    sources["panel_8_behav_usage"] = {
        "category_id": np.arange(1, 6),
        "count": counts,
        "fraction": frac,
    }

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig
