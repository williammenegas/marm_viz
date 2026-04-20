"""
Behavioural detail plots
========================

Three standalone figures:

1. :func:`plot_transition_matrix` — 5×5 heatmap of
   P(next category | current category).
2. :func:`plot_bout_durations` — mean bout duration for each of the
   80 remapped states, grouped and coloured by broad category.
3. :func:`plot_head_direction_polar` — two polar histograms of
   head-direction angle relative to stimulus: one for frames near
   the stimulus, one for frames far away.
"""

from __future__ import annotations

from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np

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


def _iter_recordings(data: Union[Recording, RecordingSet]) -> List[Recording]:
    if isinstance(data, Recording):
        return [data]
    return list(data.recs)


def _to_recording(data: Union[Recording, RecordingSet]) -> Recording:
    if isinstance(data, RecordingSet):
        return data.concatenate()
    return data


# =====================================================================
# 1. Transition matrix
# =====================================================================

def plot_transition_matrix(
    data: Union[Recording, RecordingSet],
    *,
    n_states: int = 80,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """80×80 transition probability matrix: P(next state | current).

    States are ordered by broad category, then by state ID within
    each category. Diagonal is NaN'd out.
    """
    recs = _iter_recordings(data)

    # Build category-grouped state order.
    cat_lists = [
        (1, sorted(_INDS_CLIMBING)),
        (2, sorted(_INDS_ACTIVE_LOW + _INDS_ACTIVE_MID + _INDS_ACTIVE_HIGH)),
        (3, sorted(_INDS_RED_RESTING)),
        (4, sorted(_INDS_NEAR_OTHER_ACTIVE)),
        (5, sorted(_INDS_NEAR_OTHER_RESTING)),
    ]
    ordered_states = []
    group_boundaries = []  # cumulative index where each group ends
    for _, ids in cat_lists:
        ordered_states.extend(ids)
        group_boundaries.append(len(ordered_states))

    # Map state_id → index in the ordered list.
    state_to_idx = {s: i for i, s in enumerate(ordered_states)}
    n = len(ordered_states)

    counts = np.zeros((n, n), dtype=float)
    for r in recs:
        sr = r.states_remapped
        valid = np.isfinite(sr)
        for t in range(len(sr) - 1):
            if valid[t] and valid[t + 1]:
                s_now = int(sr[t])
                s_next = int(sr[t + 1])
                i = state_to_idx.get(s_now)
                j = state_to_idx.get(s_next)
                if i is not None and j is not None:
                    counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)

    probs_display = probs.copy()
    np.fill_diagonal(probs_display, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(probs_display, cmap="plasma", vmin=0, vmax=0.0001,
                   interpolation="nearest")

    # Draw category group boundaries as white lines.
    for b in group_boundaries[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=0.8)

    # Category labels at group centers.
    cat_names = [CATEGORY_NAMES[k].title() for k, _ in cat_lists]
    centers = []
    prev = 0
    for b in group_boundaries:
        centers.append((prev + b - 1) / 2.0)
        prev = b
    ax.set_xticks(centers)
    ax.set_xticklabels(cat_names, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(cat_names, fontsize=7)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("Transition matrix  P(next | current)")
    make_colorbar(im, ax, title="Probability")
    fig.tight_layout()

    sources = {
        "panel_1_transitions": {
            "ordered_state_ids": np.array(ordered_states, dtype=float),
        }
    }
    attach_sources(fig, sources)
    return fig


# =====================================================================
# 2. Bout duration distribution
# =====================================================================

def _state_to_category(state_id: int) -> int:
    """Return the broad-category id (1-5) for a remapped state."""
    if state_id in _INDS_CLIMBING:
        return 1
    if state_id in (
        _INDS_ACTIVE_LOW + _INDS_ACTIVE_MID + _INDS_ACTIVE_HIGH
    ):
        return 2
    if state_id in _INDS_RED_RESTING:
        return 3
    if state_id in _INDS_NEAR_OTHER_ACTIVE:
        return 4
    if state_id in _INDS_NEAR_OTHER_RESTING:
        return 5
    return 3  # fallback (shouldn't happen now)


def _bout_lengths(arr: np.ndarray, state: int) -> List[int]:
    """Return a list of consecutive-run lengths for ``state`` in
    ``arr``. NaN frames break runs.
    """
    bouts = []
    current_len = 0
    for v in arr:
        if np.isfinite(v) and int(v) == state:
            current_len += 1
        else:
            if current_len > 0:
                bouts.append(current_len)
            current_len = 0
    if current_len > 0:
        bouts.append(current_len)
    return bouts


def plot_bout_durations(
    data: Union[Recording, RecordingSet],
    *,
    fps: float = 60.0,
    n_states: int = 80,
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Bar plot of mean bout duration for each of the 80 states.

    A "bout" is a consecutive run of frames assigned to the same
    remapped state. Bars are grouped by broad category with gaps
    between groups (same layout as the behaviour overview bar chart).

    Parameters
    ----------
    data:
        A :class:`Recording` or :class:`RecordingSet`.
    fps:
        Frames per second, used to convert frame counts to seconds.
    n_states:
        Number of remapped states (default 80).
    figsize:
        Figure size in inches.
    """
    recs = _iter_recordings(data)

    # Collect bout lengths per state across all recordings.
    all_bouts: "dict[int, list[int]]" = {s: [] for s in range(1, n_states + 1)}
    for r in recs:
        sr = r.states_remapped
        for s in range(1, n_states + 1):
            all_bouts[s].extend(_bout_lengths(sr, s))

    # Mean bout duration per state (in seconds).
    mean_dur = np.zeros(n_states, dtype=float)
    for s in range(1, n_states + 1):
        bouts = all_bouts[s]
        if bouts:
            mean_dur[s - 1] = np.mean(bouts) / fps
        else:
            mean_dur[s - 1] = 0.0

    # Group by category (same layout as behavior_overview).
    cat_lists = [
        (1, sorted(_INDS_CLIMBING)),
        (2, sorted(_INDS_ACTIVE_LOW + _INDS_ACTIVE_MID + _INDS_ACTIVE_HIGH)),
        (3, sorted(_INDS_RED_RESTING)),
        (4, sorted(_INDS_NEAR_OTHER_ACTIVE)),
        (5, sorted(_INDS_NEAR_OTHER_RESTING)),
    ]
    gap = 1.5
    x_pos = []
    bar_vals = []
    bar_colors = []
    tick_positions = []
    tick_labels = []
    state_ids_ordered = []
    x = 0.0
    for cat_id, state_ids in cat_lists:
        group_start = x
        for s in state_ids:
            x_pos.append(x)
            bar_vals.append(mean_dur[s - 1])
            bar_colors.append(CATEGORY_COLORS[cat_id])
            state_ids_ordered.append(s)
            x += 1.0
        group_center = (group_start + x - 1.0) / 2.0
        tick_positions.append(group_center)
        tick_labels.append(CATEGORY_NAMES[cat_id])
        x += gap

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_pos, bar_vals, color=bar_colors, width=0.9, edgecolor="none")
    ax.set_xlabel("Behaviour state (grouped by category)")
    ax.set_ylabel("Mean bout duration (seconds)")
    ax.set_title("Bout durations (80 states)")
    ax.set_xlim(-1, x)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=20, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=CATEGORY_COLORS[k], label=CATEGORY_NAMES[k])
        for k in range(1, 6)
    ]
    ax.legend(
        handles=legend_items, loc="upper right", fontsize=7,
        framealpha=0.8, ncol=2,
    )

    fig.tight_layout()

    sources = {
        "panel_1_bout_durations": {
            "state_id": np.array(state_ids_ordered, dtype=float),
            "mean_bout_seconds": np.array(bar_vals),
        }
    }
    attach_sources(fig, sources)
    return fig


# =====================================================================
# 3. Head direction polar plot
# =====================================================================

def plot_head_direction_polar(
    data: Union[Recording, RecordingSet],
    *,
    radius_px: float = 120.0,
    n_bins: int = 36,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Two polar histograms of head-direction angle relative to
    stimulus: near (left) vs far (right).

    "Near" = ``dist_to_stim < radius_px``.
    "Far" = ``dist_to_stim >= radius_px``.

    Parameters
    ----------
    data:
        A :class:`Recording` or :class:`RecordingSet`.
    radius_px:
        Distance threshold separating near from far.
    n_bins:
        Number of angular bins (default 36 = 10° each).
    figsize:
        Figure size in inches.
    """
    rec = _to_recording(data)

    ang = rec.angle_to_stim  # degrees, 0-180
    dist = rec.dist_to_stim

    valid = np.isfinite(ang) & np.isfinite(dist)
    ang_v = ang[valid]
    dist_v = dist[valid]

    near = ang_v[dist_v < radius_px]
    far = ang_v[dist_v >= radius_px]

    # Convert degrees to radians for polar plot.
    # angle_to_stim is 0-180 (unsigned angle). Mirror it to get
    # a full 0-360 polar view: place the original at 0-180 and
    # mirror at 180-360.
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_edges_rad = np.deg2rad(bin_edges)
    bin_centers_rad = np.deg2rad(0.5 * (bin_edges[:-1] + bin_edges[1:]))
    bar_width = np.deg2rad(360 / n_bins) * 0.9

    def _mirror_and_hist(angles_deg: np.ndarray) -> np.ndarray:
        """Mirror unsigned angles (0-180) to 0-360 and histogram."""
        mirrored = np.concatenate([angles_deg, 360 - angles_deg])
        counts, _ = np.histogram(mirrored, bins=bin_edges)
        total = max(counts.sum(), 1)
        return counts / total  # normalised density

    fig, (ax_near, ax_far) = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw={"projection": "polar"},
    )
    fig.suptitle(
        f"Head direction relative to stimulus (near/far threshold = "
        f"{radius_px:g} px)",
        fontsize=11,
    )

    sources: "dict[str, dict[str, np.ndarray]]" = {}

    # Compute both densities first so we can set a shared radius max.
    densities = {}
    for key, angles in (("near", near), ("far", far)):
        if angles.size > 0:
            densities[key] = _mirror_and_hist(angles)
        else:
            densities[key] = None
    r_max = 0.0
    for d in densities.values():
        if d is not None:
            r_max = max(r_max, d.max())
    r_max = max(r_max, 0.01)  # avoid zero
    # Round up to a clean number for the axis.
    r_max = np.ceil(r_max * 100) / 100

    for ax, key, label, panel_name in (
        (ax_near, "near", f"Near", "panel_1_near"),
        (ax_far, "far", f"Far", "panel_2_far"),
    ):
        density = densities[key]
        if density is not None:
            ax.bar(
                bin_centers_rad, density, width=bar_width,
                color="#555555", edgecolor="none", zorder=3,
            )
            sources[panel_name] = {
                "bin_center_deg": np.rad2deg(bin_centers_rad),
                "density": density,
            }
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes,
            )
        ax.set_ylim(0, r_max)
        ax.set_yticks([r_max])
        ax.set_yticklabels([f"{r_max:.2f}"], fontsize=7)
        # Orange background wedge for "toward stimulus" (315°–45°).
        theta_wedge = np.linspace(np.deg2rad(315), np.deg2rad(405), 100)
        ax.fill_between(theta_wedge, 0, r_max, color="orange",
                        alpha=0.12, zorder=0)
        ax.set_title(label, fontsize=9, pad=15)
        ax.set_theta_zero_location("N")  # 0° at top = toward stim
        ax.set_theta_direction(-1)       # clockwise

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    attach_sources(fig, sources)
    return fig
