"""
Head position track (per-animal grid)
=====================================

2×2 or 2×3 grid: one panel per animal (colour-coded) plus a merged
panel showing all animals together. Shows the first ``max_minutes``
of data.
"""

from __future__ import annotations

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

from ..types import Recording, RecordingSet
from .sources import attach_sources
from ._overlays import draw_cage_squares


_COLOR_ORDER = ["Red", "White", "Blue", "Yellow"]
_ANIMAL_COLORS = {
    "Red": "#d62728",
    "White": "#7f7f7f",
    "Blue": "#1f77b4",
    "Yellow": "#e6b422",
}


def _get_track(recs: List[Recording], max_frames: int):
    """Concatenate head_xy from recordings, truncated to max_frames."""
    parts_x, parts_y = [], []
    total = 0
    for r in recs:
        remaining = max_frames - total
        if remaining <= 0:
            break
        n = min(r.n_frames, remaining)
        parts_x.append(r.head_xy[:n, 0])
        parts_y.append(r.head_xy[:n, 1])
        total += n
    if not parts_x:
        return np.array([]), np.array([])
    return np.concatenate(parts_x), np.concatenate(parts_y)


def plot_head_track(
    data: Union[Recording, RecordingSet],
    *,
    fps: float = 60.0,
    max_minutes: float = 30.0,
    linewidth: float = 0.5,
    alpha: float = 0.6,
    figsize_per_panel: tuple = (5, 5),
) -> "tuple[plt.Figure, plt.Figure]":
    """Returns (fig_individual, fig_merged): a 2×2 grid of per-animal
    tracks and a single merged plot."""
    if isinstance(data, Recording):
        recs_by_color: Dict[str, List[Recording]] = {data.color: [data]}
    else:
        recs_by_color = {}
        for r in data.recs:
            recs_by_color.setdefault(r.color, []).append(r)

    # Only animals with actual data.
    animals = [c for c in _COLOR_ORDER if c in recs_by_color]
    max_frames = int(max_minutes * fps * 60)

    # Get tracks per animal.
    tracks: Dict[str, tuple] = {}
    for color in animals:
        x, y = _get_track(recs_by_color[color], max_frames)
        if x.size > 0 and np.isfinite(x).any():
            tracks[color] = (x, y)
    present = [c for c in animals if c in tracks]
    mins_shown = min(max_minutes, max_frames / (fps * 60))

    sources: "dict[str, dict[str, np.ndarray]]" = {}

    # --- Figure 1: 2×2 per-animal grid ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(figsize_per_panel[0] * 2,
                                               figsize_per_panel[1] * 2),
                               squeeze=False)
    for i, color in enumerate(present):
        ax = axes1.flat[i]
        x, y = tracks[color]
        ax.plot(x, y, color=_ANIMAL_COLORS.get(color, "gray"),
                linewidth=linewidth, alpha=alpha)
        draw_cage_squares(ax, scale=20.0)
        ax.set_aspect("equal")
        ax.set_xlim(-420, 420); ax.set_ylim(-420, 420)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
        ax.set_title(color, fontsize=11)
        sources[f"panel_{color}"] = {"x": x, "y": y}
    for j in range(len(present), 4):
        axes1.flat[j].axis("off")
    fig1.tight_layout()
    attach_sources(fig1, sources)

    # --- Figure 2: single merged panel ---
    fig2, ax2 = plt.subplots(figsize=figsize_per_panel)
    for color in present:
        x, y = tracks[color]
        ax2.plot(x, y, color=_ANIMAL_COLORS.get(color, "gray"),
                 linewidth=linewidth, alpha=alpha)
    draw_cage_squares(ax2, scale=20.0)
    ax2.set_aspect("equal")
    ax2.set_xlim(-420, 420); ax2.set_ylim(-420, 420)
    ax2.grid(True, alpha=0.15)
    ax2.set_xlabel("X (px)"); ax2.set_ylabel("Y (px)")
    ax2.set_title(f"Head tracks ({mins_shown:g} min)", fontsize=11)
    fig2.tight_layout()
    attach_sources(fig2, {})

    return fig1, fig2
