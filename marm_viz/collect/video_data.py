"""
Build :class:`~marm_viz.types.Recording` objects from marm_behavior output folders
==================================================================================

This is the main entry point for turning a folder of marm_behavior
artifacts into plot-ready :class:`~marm_viz.types.Recording` (per
video × colour) or :class:`~marm_viz.types.RecordingSet` (whole folder
or experiment) structures. Mirrors the per-video main loop of the
original MATLAB ``collect_video_data_3.m``, minus the metadata
filtering logic, which is out of scope for the visualization package.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)
from typing import Iterable, List, Optional

import numpy as np

from ..io.marm_behavior_loader import (
    discover_video_stems,
    load_cage_boundaries,
    load_depths,
    load_edges,
    load_filtered,
    load_hcoord,
    load_hlabel,
    load_stim_center,
)
from ..types import Recording, RecordingSet
from .geometry import CageGeometry, compute_per_frame_fields
from .state_remap import (
    apply_state_remap,
    broad_category_for,
    load_state_remap,
)

#: Short → long colour name mapping used in metadata.
_COLOR_LONG = {"r": "Red", "w": "White", "b": "Blue", "y": "Yellow"}


def _apply_trim(arr: np.ndarray, trim_first: int, trim_last: int) -> np.ndarray:
    """Drop ``trim_first`` rows from the start and ``trim_last`` from
    the end. Returns an empty array (not an error) if the cut would
    leave nothing.
    """
    if arr.size == 0:
        return arr
    stop = arr.shape[0] - trim_last if trim_last > 0 else arr.shape[0]
    if stop <= trim_first:
        return arr[:0]
    return arr[trim_first:stop]


def build_recording(
    folder: "str | Path",
    video_stem: str,
    color_key: str,
    *,
    trim_first_n: int = 1200,
    trim_last_n: int = 1,
    max_frames: int = 500_000,
    xy_absmax: float = 500.0,
    head_forward_ref: "np.ndarray | None" = None,
    state_remap: "str | Path | np.ndarray | None" = None,
    cage_override: "CageGeometry | None" = None,
    stim_xy_pixel_override: "np.ndarray | None" = None,
    use_filtered: bool = False,
) -> Optional[Recording]:
    """Build a :class:`Recording` for one (video × colour) pair from
    marm_behavior outputs.

    Parameters
    ----------
    folder:
        Directory containing the marm_behavior artifacts for this
        video (``edges_*.mat``, ``depths_*.mat``, etc.).
    video_stem:
        Video filename without extension, as used in marm_behavior
        output filenames (e.g. ``"test_4"`` for ``edges_test_4.mat``).
    color_key:
        One of ``"r"``, ``"w"``, ``"b"``, ``"y"``. Case-insensitive;
        also accepts ``"Red"``, ``"White"``, etc. — only the first
        character is used.
    trim_first_n, trim_last_n:
        How many frames to drop from the start/end of every edge,
        depth, and label array. Defaults match the marm_behavior
        pipeline defaults (1200 DLC burn-in frames + 1 trailing).
    max_frames:
        Hard cap on the number of frames kept. Defaults to 500k.
    xy_absmax:
        Frames whose absolute centred position exceeds this in any
        axis are rejected and set to NaN (see
        :func:`~marm_viz.collect.geometry.compute_per_frame_fields`).
    head_forward_ref:
        Optional ``(2,)`` reference vector for ``head_angle = 0``.
        Defaults to ``[0, -1]``.
    state_remap:
        Optional reorder lookup for 80-state cluster IDs. Can be a
        path to a text file (one int per line), a ``(80,)`` numpy
        array, or None (identity).
    cage_override:
        If provided, use this :class:`CageGeometry` instead of
        trying to load ``<video_stem>_cage_boundaries.mat``. Useful
        for testing or for videos where the cage annotation is
        stored elsewhere.
    stim_xy_pixel_override:
        If provided, use this stimulus centre instead of loading
        ``<video_stem>_center.txt``.
    use_filtered:
        If True, try to load filtered positions from
        ``filtered_<video_stem>.mat`` (produced by marm_filt).
        Falls back to edges data if not available.

    Returns
    -------
    Recording or None
        Returns None if any of the required inputs can't be loaded
        (missing edges file, unreadable boundaries, etc.) so the
        caller can just iterate over many videos and skip the bad
        ones. Raises only for programming errors (invalid colour
        key, bad state remap shape, etc.).
    """
    folder = Path(folder)
    color_key = color_key.lower()[:1]
    if color_key not in _COLOR_LONG:
        raise ValueError(f"unknown colour key: {color_key!r}")

    # --- load position data ---
    X_full, Y_full = None, None
    if use_filtered:
        result = load_filtered(folder, video_stem, color_key)
        if result is not None:
            X_full, Y_full = result
            log.debug(f"[build_recording] using filtered data")

    if X_full is None:
        try:
            X_full, Y_full = load_edges(folder, video_stem, color_key)
        except (FileNotFoundError, KeyError, ValueError):
            return None
        except OSError as e:
            log.warning(f"skipped (corrupt/unreadable mat file: {e})")
            return None

    # --- load depths (optional) ---
    depths_full = load_depths(folder, video_stem, color_key)

    # --- trim ---
    X = _apply_trim(X_full, trim_first_n, trim_last_n)
    Y = _apply_trim(Y_full, trim_first_n, trim_last_n)
    if X.shape[0] == 0:
        return None

    # Cap at max_frames.
    n = min(X.shape[0], max_frames)
    X = X[:n]
    Y = Y[:n]

    if depths_full is not None:
        D = _apply_trim(depths_full, trim_first_n, trim_last_n)
        # Align frame counts to min of edges/depths after trim.
        n_aligned = min(X.shape[0], D.shape[0])
        X = X[:n_aligned]
        Y = Y[:n_aligned]
        D = D[:n_aligned]
    else:
        D = None

    # --- cage geometry ---
    if cage_override is not None:
        cage = cage_override
    else:
        b = load_cage_boundaries(folder, video_stem)
        if b is not None:
            try:
                cage = CageGeometry.from_boundaries_matrix(b)
            except ValueError:
                cage = CageGeometry.default()
        else:
            cage = CageGeometry.default()

    # --- stim centre ---
    if stim_xy_pixel_override is not None:
        stim_xy = np.asarray(stim_xy_pixel_override, dtype=float)
        log.debug(f"[build_recording] stim source: override = [{stim_xy[0]:.1f}, {stim_xy[1]:.1f}]")
    else:
        stim_xy = load_stim_center(folder, video_stem)
        if stim_xy is not None:
            log.debug(f"[build_recording] stim source: file = [{stim_xy[0]:.1f}, {stim_xy[1]:.1f}]")
        else:
            log.debug(f"[build_recording] stim source: not found")

    # --- Edge data stats (first few frames, for diagnostics) ---
    _n_show = min(5, X.shape[0])
    log.debug(f"[build_recording] edge X[:{_n_show}, 0]: {X[:_n_show, 0]}")
    log.debug(f"[build_recording] edge Y[:{_n_show}, 0]: {Y[:_n_show, 0]}")

    # --- run the per-frame computation ---
    try:
        fields = compute_per_frame_fields(
            X, Y, D,
            cage=cage,
            stim_xy_pixel=stim_xy,
            head_forward_ref=head_forward_ref,
            xy_absmax=xy_absmax,
        )
    except ValueError:
        return None

    # --- Diagnostic: final output ---
    log.debug(f"[build_recording] FINAL stim_xy_centered = "
          f"[{fields['stim_xy_centered'][0]:.1f}, {fields['stim_xy_centered'][1]:.1f}]")
    log.debug(f"[build_recording] FINAL door_xy          = "
          f"[{fields['door_xy'][0]:.1f}, {fields['door_xy'][1]:.1f}]")

    n_frames = fields["n_frames"]

    # --- labels (optional) ---
    raw_labels = load_hlabel(folder, video_stem, color_key)
    states_remapped = np.full(n_frames, np.nan)
    behav_cat = np.full(n_frames, np.nan)
    if raw_labels is not None:
        remap_lookup: "np.ndarray | None"
        if isinstance(state_remap, np.ndarray):
            remap_lookup = state_remap
        else:
            remap_lookup = load_state_remap(state_remap)
        remapped = apply_state_remap(raw_labels, remap_lookup)
        remapped = _apply_trim(remapped, trim_first_n, trim_last_n)
        # Align to n_frames
        if remapped.shape[0] >= n_frames:
            states_remapped = remapped[:n_frames]
        else:
            states_remapped[: remapped.shape[0]] = remapped
        # NaN out frames rejected by the xy_absmax cut
        states_remapped[fields["bad_xy"]] = np.nan
        behav_cat = broad_category_for(states_remapped)

    # --- t-SNE coordinates (optional) ---
    raw_tsne = load_hcoord(folder, video_stem, color_key)
    tsne_xy = np.full((n_frames, 2), np.nan)
    if raw_tsne is not None:
        trimmed_tsne = _apply_trim(raw_tsne, trim_first_n, trim_last_n)
        n_tsne = min(trimmed_tsne.shape[0], n_frames)
        tsne_xy[:n_tsne] = trimmed_tsne[:n_tsne]
        tsne_xy[fields["bad_xy"]] = np.nan

    return Recording(
        n_frames=n_frames,
        head_xy=fields["head_xy"],
        body_xy=fields["body_xy"],
        head_front_xy=fields["head_front_xy"],
        head_center_xy=fields["head_center_xy"],
        head_angle=fields["head_angle"],
        angle_to_stim=fields["angle_to_stim"],
        dist_to_stim=fields["dist_to_stim"],
        depth_metric=fields["depth_metric"],
        behav_cat=behav_cat,
        states_remapped=states_remapped,
        tsne_xy=tsne_xy,
        stim_xy_centered=fields["stim_xy_centered"],
        door_xy=fields["door_xy"],
        bad_xy=fields["bad_xy"],
        metadata={
            "folder": str(folder),
            "fname": video_stem,
            "color": _COLOR_LONG[color_key],
            "cage_half_w": cage.floor_half_w,
            "cage_half_h": cage.floor_half_h,
        },
    )


def build_recording_set(
    folder: "str | Path",
    *,
    colors: "Iterable[str] | None" = None,
    video_stems: "Iterable[str] | None" = None,
    **kwargs,
) -> RecordingSet:
    """Build a :class:`RecordingSet` by iterating over every
    ``(video_stem, colour)`` combination in a folder.

    Parameters
    ----------
    folder:
        Directory containing marm_behavior artifacts.
    colors:
        Iterable of colour keys to include (default: all four
        ``r, w, b, y``). Anything not in ``{r, w, b, y}`` is ignored.
    video_stems:
        Iterable of video stems to include. If None, all stems
        found via :func:`discover_video_stems` are used.
    **kwargs:
        Forwarded to :func:`build_recording`.

    Returns
    -------
    RecordingSet
        May be empty if no valid recordings were built (e.g. the
        folder has no edges files, or all of them had unloadable
        colours).
    """
    folder = Path(folder)
    if video_stems is None:
        video_stems = discover_video_stems(folder)
    if colors is None:
        colors = ("r", "w", "b", "y")

    stems_list = list(video_stems)
    colors_list = list(colors)
    print(f"[marm_viz] Scanning {folder}")
    print(f"[marm_viz] Found {len(stems_list)} video(s): "
          f"{', '.join(stems_list[:5])}"
          f"{'...' if len(stems_list) > 5 else ''}")

    rs = RecordingSet()
    for si, stem in enumerate(stems_list, 1):
        for c in colors_list:
            color_name = _COLOR_LONG.get(c.lower()[:1], c)
            print(f"[marm_viz]   [{si}/{len(stems_list)}] "
                  f"Loading {color_name} / {stem} ...", end="", flush=True)
            rec = build_recording(folder, stem, c, **kwargs)
            if rec is not None and rec.n_frames > 0:
                rs.append(rec)
                print(f" {rec.n_frames:,} frames")
            else:
                print(" skipped (no data)")
    print(f"[marm_viz] Built {len(rs)} recording(s) total")
    return rs
