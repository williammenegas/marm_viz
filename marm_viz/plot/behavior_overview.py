"""
Behavior overview: 80-state usage + t-SNE landscape
====================================================

Two separate figures:

1. :func:`plot_state_usage` — bar chart of all 80 states grouped
   and coloured by broad category.
2. :func:`plot_tsne_centroids` — one bubble per state in t-SNE
   space, sized and coloured by usage (white→purple→black).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..types import Recording, RecordingSet
from ..collect.state_remap import (
    CATEGORY_COLORS,
    CATEGORY_NAMES,
    _INDS_CLIMBING,
    _INDS_ACTIVE_LOW,
    _INDS_ACTIVE_MID,
    _INDS_ACTIVE_HIGH,
    _INDS_RED_RESTING,
    _INDS_NEAR_OTHER_ACTIVE,
    _INDS_NEAR_OTHER_RESTING,
)
from .sources import attach_sources
from ._overlays import make_colorbar


def _to_recording(data: Union[Recording, RecordingSet]) -> Recording:
    if isinstance(data, RecordingSet):
        return data.concatenate()
    return data


def _compute_state_fracs(
    states: np.ndarray,
    n_states: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (counts, fracs) arrays of length n_states."""
    valid = states[np.isfinite(states)]
    counts = np.zeros(n_states, dtype=float)
    for s in range(1, n_states + 1):
        counts[s - 1] = np.sum(valid == s)
    total = max(valid.size, 1)
    return counts, counts / total


def plot_state_usage(
    data: Union[Recording, RecordingSet],
    *,
    n_states: int = 80,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Bar chart of all 80 states grouped and coloured by broad
    category, with gaps between groups.
    """
    rec = _to_recording(data)
    counts, fracs = _compute_state_fracs(rec.states_remapped, n_states)

    cat_lists = [
        (1, sorted(_INDS_CLIMBING)),
        (2, sorted(_INDS_ACTIVE_LOW + _INDS_ACTIVE_MID + _INDS_ACTIVE_HIGH)),
        (3, sorted(_INDS_RED_RESTING)),
        (4, sorted(_INDS_NEAR_OTHER_ACTIVE)),
        (5, sorted(_INDS_NEAR_OTHER_RESTING)),
    ]
    gap = 1.5
    x_pos = []
    bar_fracs = []
    bar_colors = []
    tick_positions = []
    tick_labels = []
    x = 0.0
    for cat_id, state_ids in cat_lists:
        group_start = x
        for s in state_ids:
            x_pos.append(x)
            bar_fracs.append(fracs[s - 1])
            bar_colors.append(CATEGORY_COLORS[cat_id])
            x += 1.0
        tick_positions.append((group_start + x - 1.0) / 2.0)
        tick_labels.append(CATEGORY_NAMES[cat_id])
        x += gap

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_pos, bar_fracs, color=bar_colors, width=0.9, edgecolor="none")
    ax.set_xlabel("Behaviour state (grouped by category)")
    ax.set_ylabel("Fraction of frames")
    ax.set_title("Behavioral state usage")
    ax.set_ylim(0, 0.08)
    ax.set_xlim(-1, x)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CATEGORY_COLORS[k], label=CATEGORY_NAMES[k])
        for k in range(1, 6)
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              framealpha=0.8, ncol=1)

    fig.tight_layout()
    sources = {
        "panel_1_state_usage": {
            "state_id": np.arange(1, n_states + 1, dtype=float),
            "count": counts,
            "fraction": fracs,
        }
    }
    attach_sources(fig, sources)
    return fig


def plot_tsne_centroids(
    data: Union[Recording, RecordingSet],
    *,
    n_states: int = 80,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """One bubble per state in t-SNE space. Size and colour encode
    usage frequency (white→purple→black).
    """
    rec = _to_recording(data)
    states = rec.states_remapped
    tsne = rec.tsne_xy
    _, fracs = _compute_state_fracs(states, n_states)

    fig, ax = plt.subplots(figsize=figsize)

    valid_tsne = np.all(np.isfinite(tsne), axis=1) & np.isfinite(states)
    if valid_tsne.any():
        cx, cy, cu, cs_ids = [], [], [], []
        for s in range(1, n_states + 1):
            mask = valid_tsne & (states == s)
            if mask.sum() == 0:
                continue
            cx.append(np.nanmean(tsne[mask, 0]))
            cy.append(np.nanmean(tsne[mask, 1]))
            cu.append(fracs[s - 1])
            cs_ids.append(s)
        cx = np.array(cx)
        cy = np.array(cy)
        cu = np.array(cu)
        cs_ids = np.array(cs_ids)

        min_size, max_size = 30.0, 600.0
        if cu.max() > 0:
            sizes = min_size + (cu / cu.max()) * (max_size - min_size)
        else:
            sizes = np.full_like(cu, min_size)

        wpb = LinearSegmentedColormap.from_list(
            "wpb", [(1, 1, 1), (0.5, 0, 0.5), (0, 0, 0)], N=256,
        )
        sc = ax.scatter(
            cx, cy, s=sizes, c=cu,
            cmap=wpb, edgecolor="gray", linewidth=0.5, alpha=0.9,
        )
        make_colorbar(sc, ax, title="State usage")
    else:
        cx = cy = cu = cs_ids = np.zeros(0)
        ax.text(0.5, 0.5, "No t-SNE data\n(hcoord files not found?)",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE centroids")
    ax.set_aspect("equal")
    ax.set_xlim(-100, 100); ax.set_ylim(-100, 100)
    # no grid

    fig.tight_layout()
    sources = {}
    if hasattr(cs_ids, '__len__') and len(cs_ids) > 0:
        sources["panel_1_tsne_centroids"] = {
            "state_id": cs_ids.astype(float),
            "tsne_x": cx,
            "tsne_y": cy,
            "usage_fraction": cu,
        }
    attach_sources(fig, sources)
    return fig
