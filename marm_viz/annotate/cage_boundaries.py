"""
Interactive cage-boundaries annotation tool
=============================================

Python port of the lab's ``find_cage_boundaries.m``. For one video,
displays frame ~100 and asks the user to draw three rectangles with
click-and-drag:

1. **Floor** — the cage floor rectangle (used as the cage-centre
   reference by the collect stage).
2. **Door** — the door rectangle (used to decide whether to
   180-rotate the coordinate system so the door is always "above").
3. **Edge left/right** — a rectangle whose x-range defines the
   valid-x cut. Frames where the animal's head x is outside this
   range are rejected.

The result is saved as ``<video_stem>_cage_boundaries.mat``
containing a ``boundaries`` variable: a 3×4 matrix where each row is
``[x, y, w, h]`` in the same ``(row, col) = (top, left)`` pixel
convention MATLAB's ``getrect`` uses.

Requires ``opencv-python`` to read video frames. Install with the
``annotate`` extra::

    pip install 'marm_viz[annotate]'
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Matplotlib and scipy.io are imported at call-time so the metadata
# module can be used on headless machines without pulling them in.


def _read_frame(video_path: Path, frame_index: int = 99) -> np.ndarray:
    """Read frame ``frame_index`` (0-indexed) from an .avi via
    OpenCV and return it as an RGB ``(H, W, 3) uint8`` array. The
    MATLAB reference reads 100 frames in a loop (``for g=1:100 f =
    readFrame(v); end``), which lands on frame 100 (1-indexed) =
    index 99 here.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "reading video frames requires opencv-python. Install "
            "with `pip install opencv-python` or "
            "`pip install 'marm_viz[annotate]'`."
        ) from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = min(frame_index, max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            # Fall back to the very first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            raise RuntimeError(f"failed to read any frame from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _save_boundaries(path: Path, boundaries: np.ndarray) -> None:
    """Save a 3×4 boundaries matrix to a .mat file. Matches the
    MATLAB ``save(... 'boundaries')`` variable-name convention so
    the existing loader picks it up.
    """
    try:
        from scipy.io import savemat
    except ImportError as e:
        raise RuntimeError(
            "saving boundaries requires scipy. Install with "
            "`pip install scipy`."
        ) from e
    savemat(str(path), {"boundaries": np.asarray(boundaries, dtype=float)})


def _load_boundaries(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        from scipy.io import loadmat
        mat = loadmat(str(path))
    except Exception:
        return None
    b = mat.get("boundaries")
    if b is None:
        return None
    return np.asarray(b, dtype=float)


# ---------------------------------------------------------------------------
# Interactive drawing
# ---------------------------------------------------------------------------

def _collect_three_rects(
    frame: np.ndarray,
    title_suffix: str = "",
    crop_left_half: bool = True,
) -> List[Tuple[float, float, float, float]]:
    """Show the frame in a matplotlib figure and collect three
    click-and-drag rectangles, one at a time. Returns a list of
    ``(x, y, w, h)`` tuples in pixel coordinates.

    The three prompts mirror the MATLAB reference order:

    1. "rect around floor"
    2. "rect around door"
    3. "boundary left/right"
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    # Match MATLAB: only display the left-half of a stereo frame
    display = frame[:, :1280, :] if (crop_left_half and frame.shape[1] > 1280) else frame

    prompts = [
        "Rect around FLOOR (drag then press Enter; close window to abort)",
        "Rect around DOOR (drag then press Enter; close window to abort)",
        "Rect defining EDGE left/right x-range (drag then press Enter)",
    ]

    rects: List[Tuple[float, float, float, float]] = []
    for prompt in prompts:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(display)
        full_title = f"{title_suffix}\n{prompt}".strip()
        ax.set_title(full_title, fontsize=11)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")

        # Draw any rectangles we've already collected so the user
        # sees context.
        for prev in rects:
            x, y, w, h = prev
            rect_patch = plt.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect_patch)

        # State shared with the event handlers
        state = {"rect": None, "done": False}

        def on_select(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if None in (x0, y0, x1, y1):
                return
            x = float(min(x0, x1))
            y = float(min(y0, y1))
            w = float(abs(x1 - x0))
            h = float(abs(y1 - y0))
            state["rect"] = (x, y, w, h)
            ax.set_title(
                f"{full_title}\nselected: ({x:.0f}, {y:.0f}) "
                f"{w:.0f}×{h:.0f}  —  press Enter to accept",
                fontsize=10,
            )
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key in ("enter", "return"):
                if state["rect"] is not None:
                    state["done"] = True
                    plt.close(fig)
            elif event.key == "escape":
                plt.close(fig)

        # rectprops for matplotlib<3.5 vs props for >=3.5
        try:
            selector = RectangleSelector(
                ax,
                on_select,
                useblit=False,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
                props=dict(edgecolor="red", facecolor="none", linewidth=2),
            )
        except TypeError:
            selector = RectangleSelector(
                ax,
                on_select,
                drawtype="box",
                useblit=False,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
                rectprops=dict(edgecolor="red", facecolor="none", linewidth=2),
            )

        # Keep a reference so the selector isn't garbage-collected
        # before plt.show() returns.
        fig._selector = selector

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

        if not state["done"] or state["rect"] is None:
            raise RuntimeError(
                f"cage annotation cancelled at step {len(rects) + 1}/3 "
                f"(no rectangle was accepted)"
            )
        rects.append(state["rect"])

    return rects


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def annotate_cage_boundaries(
    video_path: "str | Path",
    *,
    output_path: "str | Path | None" = None,
    skip_if_exists: bool = True,
    frame_index: int = 99,
    crop_left_half: bool = True,
    override_rects: "Optional[List[Tuple[float, float, float, float]]]" = None,
) -> Path:
    """Annotate cage boundaries for a single video.

    Opens a matplotlib window showing frame ``frame_index`` of the
    video and asks the user to draw three rectangles: floor, door,
    and edge x-range. Writes the 3×4 ``boundaries`` matrix to
    ``<video_stem>_cage_boundaries.mat`` in the same folder as the
    video (or ``output_path`` if given).

    Parameters
    ----------
    video_path:
        Path to the ``.avi`` file.
    output_path:
        Where to write the .mat file. Defaults to
        ``<video_parent>/<video_stem>_cage_boundaries.mat``.
    skip_if_exists:
        If True (default) and the output file already exists,
        return its path without re-annotating. Pass ``False`` to
        force re-annotation.
    frame_index:
        Which frame to display (0-indexed). Default is 99 to match
        the MATLAB reference (reads 100 frames, lands on frame 100).
    crop_left_half:
        If True (default), display only the leftmost 1280 pixels
        of the frame. This matches the lab's stereo 2560×720 layout
        where the RGB half is on the left and depth on the right.
    override_rects:
        If given, skip the interactive prompt entirely and use
        these three ``(x, y, w, h)`` rectangles. Used by tests and
        by callers that already know the coordinates.

    Returns
    -------
    Path
        Path to the written (or pre-existing) boundaries .mat file.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video does not exist: {video_path}")

    stem = video_path.stem
    if output_path is None:
        output_path = video_path.parent / f"{stem}_cage_boundaries.mat"
    output_path = Path(output_path)

    if skip_if_exists and output_path.exists():
        print(f"[cage] {output_path.name} already exists, skipping")
        return output_path

    if override_rects is not None:
        if len(override_rects) != 3:
            raise ValueError(
                f"override_rects must have exactly 3 rects, got {len(override_rects)}"
            )
        rects = [tuple(map(float, r)) for r in override_rects]
    else:
        print(f"[cage] annotating {video_path.name}")
        frame = _read_frame(video_path, frame_index=frame_index)
        rects = _collect_three_rects(
            frame,
            title_suffix=video_path.name,
            crop_left_half=crop_left_half,
        )

    boundaries = np.array(rects, dtype=float)
    if boundaries.shape != (3, 4):
        raise ValueError(
            f"expected a 3x4 boundaries matrix, got {boundaries.shape}"
        )
    _save_boundaries(output_path, boundaries)
    print(f"[cage] wrote {output_path}")
    return output_path


def annotate_folder_cages(
    folder: "str | Path",
    *,
    avi_pattern: str = "*.avi",
    skip_if_exists: bool = True,
    frame_index: int = 99,
    crop_left_half: bool = True,
) -> List[Path]:
    """Annotate every ``.avi`` in a folder. Videos whose boundaries
    file already exists are skipped unless ``skip_if_exists=False``.

    Returns the list of boundaries files produced (or re-used).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"not a directory: {folder}")
    videos = sorted(folder.glob(avi_pattern))
    if not videos:
        # Also try uppercase, which matches how some labs record
        videos = sorted(folder.glob(avi_pattern.upper()))
    if not videos:
        print(f"[cage] no videos matching {avi_pattern!r} in {folder}")
        return []
    out: List[Path] = []
    for i, v in enumerate(videos, 1):
        print(f"\n[cage] [{i}/{len(videos)}] {v.name}")
        try:
            p = annotate_cage_boundaries(
                v,
                skip_if_exists=skip_if_exists,
                frame_index=frame_index,
                crop_left_half=crop_left_half,
            )
            out.append(p)
        except RuntimeError as e:
            print(f"[cage] skipped {v.name}: {e}")
    return out
