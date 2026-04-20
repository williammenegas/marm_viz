"""
Stacked ethogram raster (per-category rows)
============================================

For each animal, one row per broad behaviour category (5 rows).
Each row shows binary presence (coloured = in category, white =
not). Animals are separated by white space.
"""

from __future__ import annotations

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np

from ..types import Recording, RecordingSet
from ..collect.state_remap import CATEGORY_COLORS, CATEGORY_NAMES
from .sources import attach_sources


_COLOR_ORDER = ["Red", "White", "Blue", "Yellow"]
_CAT_ORDER = [1, 2, 3, 4, 5]


def plot_ethogram(
    data: Union[Recording, RecordingSet],
    *,
    fps: float = 60.0,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Stacked ethogram with one row per category per animal."""
    # Group recordings by animal colour.
    if isinstance(data, Recording):
        recs_by_color: Dict[str, List[Recording]] = {data.color: [data]}
    else:
        recs_by_color = {}
        for r in data.recs:
            recs_by_color.setdefault(r.color, []).append(r)

    # Build per-animal concatenated category array. Skip animals with
    # no valid data.
    cat_per_animal: Dict[str, np.ndarray] = {}
    for color in _COLOR_ORDER:
        if color not in recs_by_color:
            continue
        parts = [r.behav_cat for r in recs_by_color[color]]
        cat = np.concatenate(parts) if parts else np.array([])
        if np.isfinite(cat).any():
            cat_per_animal[color] = cat

    animals = [c for c in _COLOR_ORDER if c in cat_per_animal]
    if not animals:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No behavioural data", ha="center",
                va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    n_cats = len(_CAT_ORDER)
    row_height = 0.8
    cat_gap = 0.1       # gap between category rows within an animal
    animal_gap = 0.6     # gap between animal groups

    # Compute total height and y positions.
    # Each animal block: n_cats * row_height + (n_cats-1) * cat_gap
    block_h = n_cats * row_height + (n_cats - 1) * cat_gap
    total_h = len(animals) * block_h + (len(animals) - 1) * animal_gap

    fig, ax = plt.subplots(figsize=figsize)
    sources: "dict[str, dict[str, np.ndarray]]" = {}

    y_animal_ticks = []  # (y_center, label) for animal names
    y_cat_ticks = []     # (y_center, label) for category names

    for a_idx, color in enumerate(animals):
        cat = cat_per_animal[color]
        n = len(cat)
        t_max = n / fps / 60.0  # minutes

        # Bottom of this animal's block.
        block_bottom = (len(animals) - 1 - a_idx) * (block_h + animal_gap)
        y_animal_ticks.append(
            (block_bottom + block_h / 2, color)
        )

        for k_idx, k in enumerate(_CAT_ORDER):
            y_bot = block_bottom + (n_cats - 1 - k_idx) * (row_height + cat_gap)

            # Binary mask: 1 where cat == k, 0 otherwise.
            mask = np.where(np.isfinite(cat) & (cat == k), 1.0, 0.0)
            img = mask.reshape(1, -1)

            # Use a 2-color cmap: 0 = white, 1 = category colour.
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([(1, 1, 1), CATEGORY_COLORS[k]])

            ax.imshow(
                img,
                aspect="auto",
                cmap=cmap,
                vmin=0, vmax=1,
                interpolation="nearest",
                extent=(0, t_max, y_bot, y_bot + row_height),
            )

            # Category label for every animal block.
            y_cat_ticks.append(
                (y_bot + row_height / 2,
                 CATEGORY_NAMES[k].title())
            )

        sources[f"panel_{color}"] = {
            "frame": np.arange(n, dtype=float),
            "behav_cat": cat,
        }

    ax.set_xlabel("Time (minutes)")
    ax.set_title("Ethogram")

    # Animal labels on the left (primary y-axis).
    ax.set_yticks([y for y, _ in y_animal_ticks])
    ax.set_yticklabels([lbl for _, lbl in y_animal_ticks], fontsize=10)
    ax.set_ylim(-animal_gap / 2, total_h + animal_gap / 2)

    # Category labels on the right (secondary y-axis).
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([y for y, _ in y_cat_ticks])
    ax2.set_yticklabels([lbl for _, lbl in y_cat_ticks], fontsize=7)

    fig.tight_layout()
    attach_sources(fig, sources)
    return fig
