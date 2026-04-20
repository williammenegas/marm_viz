"""
Interactive stimulus-centre annotation tool
==============================================

Python port of the lab's ``find_center_of_stimulus.m``. For one
video, displays frame 1 and asks the user to click once to mark the
stimulus centre. Saves ``<video_stem>_center.txt`` with the ``x`` and
``y`` pixel coordinates on one whitespace-separated line — the same
format the MATLAB reference writes via
``save(..., 'x_center', 'y_center', '-ascii')`` and the marm_viz
loader reads via :func:`marm_viz.io.marm_behavior_loader.load_stim_center`.

Requires ``opencv-python`` to read video frames. Install with the
``annotate`` extra::

    pip install 'marm_viz[annotate]'
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .cage_boundaries import _read_frame  # reuse the cv2 reader


def _save_center(path: Path, x: float, y: float) -> None:
    """Write ``x y`` to ``path`` as whitespace-separated floats,
    one per line. Matches the format MATLAB writes with
    ``save('-ascii')`` for two scalars (and is round-trippable via
    :func:`load_stim_center`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{float(x)}\n{float(y)}\n")


def _load_center(path: Path) -> Optional[Tuple[float, float]]:
    if not path.exists():
        return None
    try:
        v = np.loadtxt(str(path)).ravel()
    except Exception:
        return None
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    return float(v[0]), float(v[1])


def _collect_click(
    frame: np.ndarray,
    title: str = "",
    crop_left_half: bool = True,
) -> Tuple[float, float]:
    """Show the frame and collect one click from the user. Returns
    ``(x, y)`` in pixel coordinates relative to the displayed image.
    """
    import matplotlib.pyplot as plt

    display = frame[:, :1280, :] if (crop_left_half and frame.shape[1] > 1280) else frame

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(display)
    ax.set_title(
        f"{title}\nClick once on the stimulus centre "
        f"(right-click / middle-click to abort)",
        fontsize=11,
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    try:
        pts = fig.ginput(n=1, timeout=0, show_clicks=True)
    finally:
        plt.close(fig)

    if not pts:
        raise RuntimeError("stim centre annotation cancelled (no click)")
    x, y = pts[0]
    return float(x), float(y)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def annotate_stim_center(
    video_path: "str | Path",
    *,
    output_path: "str | Path | None" = None,
    skip_if_exists: bool = True,
    frame_index: int = 0,
    crop_left_half: bool = True,
    override_xy: "Optional[Tuple[float, float]]" = None,
) -> Path:
    """Annotate the stimulus centre for a single video.

    Opens a matplotlib window showing frame ``frame_index`` and asks
    the user to click once. Writes ``x`` and ``y`` to
    ``<video_stem>_center.txt`` next to the video (or
    ``output_path`` if given).

    Parameters
    ----------
    video_path:
        Path to the .avi file.
    output_path:
        Where to write the centre file. Defaults to
        ``<video_parent>/<video_stem>_center.txt``.
    skip_if_exists:
        If True (default) and the centre file already exists,
        return its path without re-annotating. Matches the MATLAB
        try/catch behaviour.
    frame_index:
        Which frame to display (0-indexed). Default 0 = first
        frame, matching the MATLAB reference.
    crop_left_half:
        If True (default), display only the leftmost 1280 pixels
        of the frame (stereo RGB half).
    override_xy:
        If given, skip the interactive prompt and write this
        ``(x, y)`` directly. Used by tests and scripted callers.

    Returns
    -------
    Path
        Path to the written (or pre-existing) centre file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video does not exist: {video_path}")

    stem = video_path.stem
    if output_path is None:
        output_path = video_path.parent / f"{stem}_center.txt"
    output_path = Path(output_path)

    if skip_if_exists and output_path.exists():
        print(f"[stim] {output_path.name} already exists, skipping")
        return output_path

    if override_xy is not None:
        x, y = override_xy
    else:
        print(f"[stim] annotating {video_path.name}")
        frame = _read_frame(video_path, frame_index=frame_index)
        x, y = _collect_click(
            frame,
            title=video_path.name,
            crop_left_half=crop_left_half,
        )

    _save_center(output_path, x, y)
    print(f"[stim] wrote {output_path} (x={x:.1f}, y={y:.1f})")
    return output_path


def annotate_folder_centers(
    folder: "str | Path",
    *,
    avi_pattern: str = "*.avi",
    skip_if_exists: bool = True,
    frame_index: int = 0,
    crop_left_half: bool = True,
) -> List[Path]:
    """Annotate every ``.avi`` in a folder. Videos whose centre
    file already exists are skipped unless ``skip_if_exists=False``.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    videos = sorted(folder.glob(avi_pattern))
    if not videos:
        videos = sorted(folder.glob(avi_pattern.upper()))
    if not videos:
        print(f"[stim] no videos matching {avi_pattern!r} in {folder}")
        return []
    out: List[Path] = []
    for i, v in enumerate(videos, 1):
        if "error" in v.name.lower():
            # Matches MATLAB `if size(strfind(file_name_avi,'error'),1)==0`
            print(f"[stim] [{i}/{len(videos)}] {v.name} — skipped (name contains 'error')")
            continue
        print(f"\n[stim] [{i}/{len(videos)}] {v.name}")
        try:
            p = annotate_stim_center(
                v,
                skip_if_exists=skip_if_exists,
                frame_index=frame_index,
                crop_left_half=crop_left_half,
            )
            out.append(p)
        except RuntimeError as e:
            print(f"[stim] skipped {v.name}: {e}")
    return out
