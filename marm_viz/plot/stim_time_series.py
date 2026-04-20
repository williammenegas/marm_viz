"""
Chunked stimulus-proximity time series
=======================================

Port of ``plot_video_data_ts_1.m``. For each chunk of time (default:
5-minute chunks at 60 fps = 18,000 frames per chunk), compute the
fraction of frames the animal's head is within ``radius_px`` of the
stimulus. Produces three figures:

1. One dot per (recording, chunk) pair showing the in-radius
   fraction, plus a black "mean across recordings" dot per chunk.
2. Per-chunk pooled XY scatter subplots.
3. Pooled XY scatter coloured by inside-radius vs outside-radius.

Usage::

    from marm_viz import plot_stim_proximity_time_series
    figs = plot_stim_proximity_time_series(rec_set,
                                           radius_px=120,
                                           chunk_minutes=5,
                                           fps=60)
    # figs is a (fig1, fig2, fig3) tuple
"""

from __future__ import annotations

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..types import Recording, RecordingSet
from .sources import attach_sources
from ._overlays import draw_cage_squares


def _iter_recordings(data: Union[Recording, RecordingSet]) -> List[Recording]:
    """Turn whatever the user passes into a flat list of recordings
    (no augmentation / concatenation — this plot wants per-recording
    granularity).
    """
    if isinstance(data, Recording):
        return [data]
    return list(data.recs)


def _sample_mask(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean sampling mask. ``0 < rate < 1`` = Bernoulli keep
    probability; ``rate >= 1`` = keep every Nth (step = round(rate)).
    """
    if n <= 0:
        return np.zeros(0, dtype=bool)
    if np.isfinite(rate) and 0 < rate < 1:
        m = rng.random(n) < rate
        if not m.any():
            m[rng.integers(0, n)] = True
        return m
    if np.isfinite(rate) and rate >= 1:
        step = max(1, int(round(rate)))
        m = np.zeros(n, dtype=bool)
        m[::step] = True
        return m
    return np.ones(n, dtype=bool)


def plot_stim_proximity_time_series(
    data: Union[Recording, RecordingSet],
    *,
    radius_px: float = 120.0,
    chunk_minutes: float = 5.0,
    fps: float = 60.0,
    use_full_chunks_only: bool = True,
    scatter_sample_rate: float = 0.05,
    common_xy_limits_px: Tuple[float, float, float, float] = (-600, 600, -600, 600),
    seed: "int | None" = None,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Build the three stim-proximity figures. Returns a tuple
    ``(fig_fraction, fig_chunk_scatters, fig_inside_outside)``.

    Parameters
    ----------
    data:
        A :class:`Recording` or :class:`RecordingSet`. Recording
        sets are not augmented/concatenated — each recording stays
        separate so the per-recording scatter in figure 1 is
        meaningful.
    radius_px:
        Radius (in cage-centered pixels) that defines "close to the
        stimulus". Defaults to 120.
    chunk_minutes:
        Chunk length in minutes. Default 5.
    fps:
        Frames per second used to convert ``chunk_minutes`` into a
        frame count. Default 60.
    use_full_chunks_only:
        If True (default), discard the trailing partial chunk in
        each recording so all plotted chunks are the same length.
    scatter_sample_rate:
        Fraction of points to keep for the two scatter figures.
        ``0.01`` = 1%. Set to ``1.0`` to keep everything (slow).
    common_xy_limits_px:
        ``(xmin, xmax, ymin, ymax)`` limits for the scatter axes.
        Using fixed limits makes it easier to compare chunks.
    seed:
        Optional RNG seed for reproducible sampling.
    """
    rng = np.random.default_rng(seed)
    frames_per_chunk = int(round(fps * chunk_minutes * 60))

    recs = _iter_recordings(data)

    # per-chunk list of per-recording fractions
    chunk_fractions: List[List[float]] = []
    # per-chunk list of pooled xy arrays
    chunk_xys: List[List[np.ndarray]] = []

    # For figure 3: pooled xy + inside-mask across all recordings/chunks
    pooled_xy: List[np.ndarray] = []
    pooled_inside: List[np.ndarray] = []

    def _ensure_len(lst: list, idx: int):
        while len(lst) <= idx:
            lst.append([])

    for r in recs:
        if r.head_xy.size == 0:
            continue
        N = r.n_frames
        if use_full_chunks_only:
            n_chunks = N // frames_per_chunk
        else:
            n_chunks = int(np.ceil(N / frames_per_chunk))
        if n_chunks < 1:
            continue

        # Try dist_to_stim first; fall back to head_xy ↔ stim_xy_centered.
        dist_full = r.dist_to_stim if r.dist_to_stim.size == N else None
        stim_xy = r.stim_xy_centered
        stim_ok = np.all(np.isfinite(stim_xy))

        for ck in range(n_chunks):
            a = ck * frames_per_chunk
            b = min(N, (ck + 1) * frames_per_chunk)
            if use_full_chunks_only and (b - a) < frames_per_chunk:
                continue

            xy = r.head_xy[a:b]
            valid_xy = np.all(np.isfinite(xy), axis=1)
            if r.bad_xy.size == N:
                valid_xy &= ~r.bad_xy[a:b]
            if not valid_xy.any():
                continue

            d = dist_full[a:b] if dist_full is not None else np.full(b - a, np.nan)
            if not np.isfinite(d).any() and stim_ok:
                dx = xy[:, 0] - stim_xy[0]
                dy = xy[:, 1] - stim_xy[1]
                d = np.sqrt(dx * dx + dy * dy)

            valid = valid_xy & np.isfinite(d)
            frac_in = float(np.mean(d[valid] < radius_px)) if valid.any() else np.nan

            _ensure_len(chunk_fractions, ck)
            _ensure_len(chunk_xys, ck)
            chunk_fractions[ck].append(frac_in)
            chunk_xys[ck].append(xy[valid_xy])

            if valid.any():
                pooled_xy.append(xy[valid])
                pooled_inside.append(d[valid] < radius_px)

    # Trim trailing empty slots
    while chunk_fractions and not chunk_fractions[-1] and not chunk_xys[-1]:
        chunk_fractions.pop()
        chunk_xys.pop()

    # ================= FIGURE 1: fraction per chunk ====================
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    mean_xs, mean_ys = [], []
    for ck, vals_list in enumerate(chunk_fractions, start=1):
        vals = np.asarray(vals_list, dtype=float)
        jitter = (rng.random(len(vals)) - 0.5) * 0.12
        ax1.scatter(
            ck + jitter, vals, s=30,
            facecolor="#b0b0b0", edgecolor="#b0b0b0", alpha=0.7,
            label="_nolegend_",
        )
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size > 0:
            mn = finite_vals.mean()
            ax1.scatter(
                ck, mn, s=70,
                facecolor="black", edgecolor="black", alpha=0.9,
                label="_nolegend_",
            )
            mean_xs.append(ck)
            mean_ys.append(mn)
    # Thick black line connecting the means.
    if len(mean_xs) > 1:
        ax1.plot(mean_xs, mean_ys, color="black", linewidth=2.0,
                 zorder=3, alpha=0.9)
    ax1.set_xlabel("Chunk # (5 min bins)")
    ax1.set_ylabel("Fraction of time near stimulus")
    ax1.set_title("Stimulus proximity across time")
    if chunk_fractions:
        ax1.set_xlim(0.5, len(chunk_fractions) + 0.5)
    ax1.set_ylim(0, 1)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.15)

    # ================= FIGURE 2: per-chunk heatmaps ====================
    max_scatter_chunks = 6
    n_chunks = min(len(chunk_xys), max_scatter_chunks)
    n_cols = max(n_chunks, 1)
    n_rows = 1
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, 3.8),
        squeeze=False,
    )
    bin_size = 20
    for ck in range(n_chunks):
        xy_chunks = chunk_xys[ck]
        ax = axes2.flat[ck]
        if not xy_chunks:
            ax.text(0.5, 0.5, f"Chunk {ck + 1}\n(no data)", ha="center",
                    va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        pooled = np.vstack(xy_chunks)
        valid = np.all(np.isfinite(pooled), axis=1)
        px = pooled[valid, 0]
        py = pooled[valid, 1]
        # Bin into occupancy heatmap (same as xy_occupancy).
        xb = np.round(px / bin_size).astype(int)
        yb = np.round(py / bin_size).astype(int)
        x_min, x_max = -20, 20
        y_min, y_max = -20, 20
        clip_m = (xb >= x_min) & (xb <= x_max) & (yb >= y_min) & (yb <= y_max)
        xb, yb = xb[clip_m], yb[clip_m]
        nx = x_max - x_min + 1
        ny = y_max - y_min + 1
        grid = np.zeros((ny, nx), dtype=float)
        np.add.at(grid, (yb - y_min, xb - x_min), 1.0)
        total = max(grid.sum(), 1)
        grid = grid / total
        extent = (x_min - 0.5, x_min + nx - 0.5,
                  y_min - 0.5, y_min + ny - 0.5)
        ax.imshow(grid, origin="lower", extent=extent, aspect="equal",
                  cmap="plasma", vmax=0.008, interpolation="nearest")
        t_start = ck * chunk_minutes; t_end = (ck + 1) * chunk_minutes; ax.set_title(f"{t_start:g}-{t_end:g} min", fontsize=9)
        ax.set_xlabel("X bin"); ax.set_ylabel("Y bin")
        ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
        draw_cage_squares(ax, scale=1.0)
    # Hide any unused subplots
    for extra in range(n_chunks, n_rows * n_cols):
        axes2.flat[extra].axis("off")
    fig2.tight_layout()

    # ================= FIGURE 3: inside/outside pooled =================
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    if pooled_xy:
        all_xy = np.vstack(pooled_xy)
        m = _sample_mask(all_xy.shape[0], scatter_sample_rate, rng)
        xy_s = all_xy[m]
        ax3.scatter(
            xy_s[:, 0], xy_s[:, 1], s=6,
            color="#888888", alpha=0.15, edgecolor="none",
        )
        # Near-stimulus radius circle.
        circle = plt.Circle((0, 0), radius_px, fill=False,
                            edgecolor="black", linewidth=1.5,
                            linestyle="--", zorder=5)
        ax3.add_patch(circle)
        # Cage boundary rectangles (edge 400 and 800) centered at 0,0.
        from matplotlib.patches import Rectangle
        for edge in (400, 800):
            rect = Rectangle((-edge / 2, -edge / 2), edge, edge,
                             fill=False, edgecolor="black",
                             linewidth=1.0, linestyle="--", zorder=4)
            ax3.add_patch(rect)
    else:
        ax3.text(0.5, 0.5, "No pooled points with valid dist-to-stim",
                 ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("X"); ax3.set_ylabel("Y")
    ax3.set_title("Randomly sampled XY coords")
    ax3.set_aspect("equal")
    ax3.set_xlim(-400, 400); ax3.set_ylim(-400, 400)
    ax3.grid(True, alpha=0.15)

    # ================= Source data attachment =========================
    # Figure 1: per-chunk per-recording fractions, long-format (chunk,
    # recording_idx_within_chunk, fraction). One row per dot.
    chunk_ids_f1 = []
    rec_ids_f1 = []
    frac_f1 = []
    for ck, vals_list in enumerate(chunk_fractions, start=1):
        for ri, v in enumerate(vals_list):
            chunk_ids_f1.append(ck)
            rec_ids_f1.append(ri)
            frac_f1.append(v)
    f1_sources: "dict[str, dict[str, np.ndarray]]" = {
        "panel_1_per_chunk_fractions": {
            "chunk": np.asarray(chunk_ids_f1, dtype=float),
            "recording_index": np.asarray(rec_ids_f1, dtype=float),
            "fraction_within_radius": np.asarray(frac_f1, dtype=float),
        }
    }
    attach_sources(fig1, f1_sources)

    # Figure 2: per-chunk pooled XY, one CSV per chunk panel.
    f2_sources: "dict[str, dict[str, np.ndarray]]" = {}
    for ck, xy_chunks in enumerate(chunk_xys, start=1):
        if not xy_chunks:
            f2_sources[f"panel_{ck}_chunk_xy"] = {
                "x": np.zeros(0), "y": np.zeros(0),
            }
            continue
        pooled = np.vstack(xy_chunks)
        f2_sources[f"panel_{ck}_chunk_xy"] = {
            "x": pooled[:, 0].astype(float),
            "y": pooled[:, 1].astype(float),
        }
    attach_sources(fig2, f2_sources)

    # Figure 3: pooled xy + inside-mask (all recordings/chunks).
    if pooled_xy:
        all_xy_full = np.vstack(pooled_xy)
        all_inside_full = np.concatenate(pooled_inside).astype(bool)
        f3_sources: "dict[str, dict[str, np.ndarray]]" = {
            "panel_1_inside_outside": {
                "x": all_xy_full[:, 0].astype(float),
                "y": all_xy_full[:, 1].astype(float),
                "inside_radius": all_inside_full.astype(float),
            }
        }
    else:
        f3_sources = {
            "panel_1_inside_outside": {
                "x": np.zeros(0), "y": np.zeros(0),
                "inside_radius": np.zeros(0),
            }
        }
    attach_sources(fig3, f3_sources)

    return fig1, fig2, fig3
