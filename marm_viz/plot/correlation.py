"""
Cagemate behavioural correlation
================================

Port of ``collect_hlabel.m``'s pairwise correlation analysis.

For each time chunk, compute each animal's 80-state usage vector
(fraction of frames in each remapped state). Then compute the
Spearman rank correlation between each pair of animals' usage
vectors within that chunk. Produces two panels:

1. **4×4 pairwise correlation matrix** — mean Spearman ρ across
   all chunks. Self-correlations NaN'd (diagonal blank).
2. **Per-chunk time series** — one line per animal pair, showing
   how behavioral similarity evolves over the session.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ..types import Recording, RecordingSet
from .sources import attach_sources


_COLOR_ORDER = ["Red", "White", "Blue", "Yellow"]
_ANIMAL_COLORS = {
    "Red": "#d62728",
    "White": "#7f7f7f",
    "Blue": "#1f77b4",
    "Yellow": "#bcbd22",
}


def _compute_usage_vectors(
    recs_by_color: Dict[str, List[Recording]],
    chunk_frames: int,
    n_states: int = 80,
) -> Dict[str, np.ndarray]:
    """For each animal colour, concatenate all recordings' remapped
    state labels, chunk them, and compute the 80-state usage vector
    per chunk.

    Returns ``{color: (n_chunks, 80) ndarray}``. If an animal has
    no data, its matrix is ``(n_chunks, 80)`` filled with NaN (so
    pairwise correlations with that animal are NaN).
    """
    # Find the global frame count (longest animal determines n_chunks).
    max_frames = 0
    labels_by_color: Dict[str, np.ndarray] = {}
    for color in _COLOR_ORDER:
        if color not in recs_by_color:
            labels_by_color[color] = np.array([])
            continue
        parts = [r.states_remapped for r in recs_by_color[color]]
        cat = np.concatenate(parts) if parts else np.array([])
        labels_by_color[color] = cat
        if cat.size > max_frames:
            max_frames = cat.size

    n_chunks = max(max_frames // chunk_frames, 1)

    usage: Dict[str, np.ndarray] = {}
    for color in _COLOR_ORDER:
        cat = labels_by_color.get(color, np.array([]))
        mat = np.full((n_chunks, n_states), np.nan)
        for ck in range(n_chunks):
            a = ck * chunk_frames
            b = (ck + 1) * chunk_frames
            if a >= cat.size:
                break
            chunk = cat[a:min(b, cat.size)]
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                continue
            counts = np.zeros(n_states)
            for s in range(1, n_states + 1):
                counts[s - 1] = np.sum(valid == s)
            total = max(counts.sum(), 1)
            mat[ck] = counts / total
        usage[color] = mat

    return usage


def plot_cagemate_correlation(
    data: Union[Recording, RecordingSet],
    *,
    chunk_minutes: float = 30.0,
    fps: float = 60.0,
    n_states: int = 80,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Two-panel cagemate correlation figure.

    Parameters
    ----------
    data:
        A :class:`Recording` or :class:`RecordingSet`. Per-animal
        identity is preserved (no concatenation across animals).
    chunk_minutes:
        Chunk length in minutes for computing usage vectors.
    fps:
        Frames per second.
    n_states:
        Number of remapped states (default 80).
    figsize:
        Figure size in inches.
    """
    # Group recordings by animal colour.
    if isinstance(data, Recording):
        recs_by_color: Dict[str, List[Recording]] = {data.color: [data]}
    else:
        recs_by_color = {}
        for r in data.recs:
            recs_by_color.setdefault(r.color, []).append(r)

    chunk_frames = int(round(fps * chunk_minutes * 60))
    usage = _compute_usage_vectors(recs_by_color, chunk_frames, n_states)

    # Animals present with valid behavioral data.
    animals = [
        c for c in _COLOR_ORDER
        if c in recs_by_color
        and np.isfinite(usage[c]).any()
    ]
    n_animals = len(animals)
    if n_animals == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid behavioral data",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        attach_sources(fig, {})
        return fig
    n_chunks = max(usage[c].shape[0] for c in animals)

    # Compute per-chunk pairwise Spearman correlations.
    # pairs[i] = (color_a, color_b)
    pairs: List[Tuple[str, str]] = []
    for i in range(n_animals):
        for j in range(i + 1, n_animals):
            pairs.append((animals[i], animals[j]))

    chunk_corrs: Dict[Tuple[str, str], np.ndarray] = {}
    for a, b in pairs:
        corrs = np.full(n_chunks, np.nan)
        for ck in range(n_chunks):
            va = usage[a][ck] if ck < usage[a].shape[0] else None
            vb = usage[b][ck] if ck < usage[b].shape[0] else None
            if (va is not None and vb is not None
                    and np.isfinite(va).any() and np.isfinite(vb).any()):
                # Replace any NaN in individual states with 0 for the
                # correlation (NaN states = unused = 0 usage).
                va_clean = np.nan_to_num(va, nan=0.0)
                vb_clean = np.nan_to_num(vb, nan=0.0)
                rho, _ = spearmanr(va_clean, vb_clean)
                corrs[ck] = rho
        chunk_corrs[(a, b)] = corrs

    # Build the 4×4 mean-correlation matrix.
    corr_matrix = np.full((n_animals, n_animals), np.nan)
    for (a, b), corrs in chunk_corrs.items():
        i = animals.index(a)
        j = animals.index(b)
        mean_rho = np.nanmean(corrs)
        corr_matrix[i, j] = mean_rho
        corr_matrix[j, i] = mean_rho  # symmetric

    # ========================= Plot ==================================
    _PAIR_COLORS = [
        "#d94030", "#1f7fe0", "#1a8c4e",
        "#8e44ad", "#16a3b8", "#e67e22",
    ]
    fig, (ax_mat, ax_ts) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        "Cagemate correlation (80-state Spearman)",
        fontsize=11,
    )

    # --- Left: correlation matrix ---
    im = ax_mat.imshow(corr_matrix, cmap="plasma", vmin=-1, vmax=1)
    for i in range(n_animals):
        for j in range(n_animals):
            if i == j:
                continue
            val = corr_matrix[i, j]
            if np.isfinite(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax_mat.text(j, i, f"{val:.2f}", ha="center",
                            va="center", fontsize=10, color=color)
    ax_mat.set_xticks(range(n_animals))
    ax_mat.set_xticklabels(animals, fontsize=9)
    ax_mat.set_yticks(range(n_animals))
    ax_mat.set_yticklabels(animals, fontsize=9)
    ax_mat.set_title("Mean Spearman ρ")

    # --- Right: per-chunk time series ---
    chunk_x = np.arange(1, n_chunks + 1)
    for pi, ((a, b), corrs) in enumerate(chunk_corrs.items()):
        label = f"{a}–{b}"
        ax_ts.plot(
            chunk_x, corrs,
            marker="o", markersize=4, linewidth=1.2,
            color=_PAIR_COLORS[pi % len(_PAIR_COLORS)],
            alpha=0.8, label=label,
        )
    ax_ts.set_xlabel(f"Chunk # ({chunk_minutes:g} min each)")
    ax_ts.set_ylabel("Spearman ρ")
    ax_ts.set_title("Per-chunk pairwise correlation")
    ax_ts.set_ylim(-1, 1)
    ax_ts.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_ts.grid(True, alpha=0.15)
    if pairs:
        ax_ts.legend(loc="best", fontsize=7, framealpha=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # ========================= Sources ===============================
    sources: "dict[str, dict[str, np.ndarray]]" = {}

    # Correlation matrix (long-form).
    mat_a, mat_b, mat_rho = [], [], []
    for i in range(n_animals):
        for j in range(n_animals):
            if i != j and np.isfinite(corr_matrix[i, j]):
                mat_a.append(i + 1)
                mat_b.append(j + 1)
                mat_rho.append(corr_matrix[i, j])
    if mat_a:
        sources["panel_1_corr_matrix"] = {
            "animal_a_idx": np.array(mat_a, dtype=float),
            "animal_b_idx": np.array(mat_b, dtype=float),
            "mean_spearman_rho": np.array(mat_rho),
        }

    # Per-chunk time series (one source per pair).
    for (a, b), corrs in chunk_corrs.items():
        safe_key = f"panel_2_{a}_{b}"
        sources[safe_key] = {
            "chunk": chunk_x.astype(float),
            "spearman_rho": corrs,
        }

    attach_sources(fig, sources)
    return fig
