"""
marm_behavior output loader
===========================

Reads the artifacts written by the marm_behavior pipeline for a given
video and returns raw numpy arrays ready for the geometry/collect
modules. Handles both v7 (scipy.io.loadmat) and v7.3 (h5py) ``.mat``
files transparently.

The file layout expected is the one marm_behavior writes:

* ``edges_<video_stem>.mat``: contains per-colour edge X/Y matrices
  under the variable names ``F_3RE/F_4RE`` (Red), ``F_3E/F_4E``
  (White), ``F_3BE/F_4BE`` (Blue), ``F_3YE/F_4YE`` (Yellow). Each
  matrix is ``(n_frames, n_edge_cols)`` with at least 125 columns;
  this loader slices columns 105..125 (inclusive, 1-indexed) —
  matching the MATLAB ``collect_video_data_3.m`` convention.
* ``depths_<video_stem>.mat``: per-colour depth matrices
  ``depthRed``, ``depthWhite``, ``depthBlue``, ``depthYellow``. Each
  is ``(n_frames, >=375)``; columns 1..375 are kept.
* ``hlabel_<Color>_<video_stem>.csv``: per-colour 80-state cluster
  IDs, one integer per frame.
* ``<video_stem>_cage_boundaries.mat`` (optional): ``boundaries``
  variable, 3×4 matrix. See :class:`~marm_viz.collect.geometry.CageGeometry`.
* ``<video_stem>_center.txt`` (optional): stimulus centre in raw
  pixel coordinates, one x and one y on separate lines or
  whitespace-separated.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)
from typing import Dict, Optional, Tuple

import numpy as np


# Mapping from colour → variable names in edges/depths .mat files.
_COLOR_TO_EDGE_VARS: Dict[str, Tuple[str, str]] = {
    "r": ("F_3RE", "F_4RE"),
    "w": ("F_3E", "F_4E"),
    "b": ("F_3BE", "F_4BE"),
    "y": ("F_3YE", "F_4YE"),
}

_COLOR_TO_DEPTH_VAR: Dict[str, str] = {
    "r": "depthRed",
    "w": "depthWhite",
    "b": "depthBlue",
    "y": "depthYellow",
}

_COLOR_LONG: Dict[str, str] = {
    "r": "Red",
    "w": "White",
    "b": "Blue",
    "y": "Yellow",
}


def _load_mat_any(path: Path) -> Dict[str, np.ndarray]:
    """Load a .mat file as a dict of ``name -> ndarray``. Works for
    both v7 (scipy) and v7.3 (HDF5 via h5py) formats.
    """
    try:
        from scipy.io import loadmat  # type: ignore
        mat = loadmat(str(path), squeeze_me=False)
        return {k: np.asarray(v) for k, v in mat.items() if not k.startswith("__")}
    except NotImplementedError:
        pass  # v7.3 file — fall through to h5py
    except Exception:
        pass  # try h5py as last resort

    try:
        import h5py  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"cannot load {path}: scipy.io.loadmat failed and h5py is not "
            "installed. Install h5py for v7.3 .mat support."
        ) from e

    out: Dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as f:
        for k in f.keys():
            if k.startswith("#"):
                continue
            # h5py stores MATLAB matrices transposed relative to scipy
            out[k] = np.asarray(f[k]).T
    return out


def load_edges(
    folder: "str | Path",
    video_stem: str,
    color_key: str,
    *,
    edge_col_start: int = 104,  # MATLAB 105:125 → Python 104:125 (inclusive)
    edge_col_end: int = 125,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the (X, Y) edge matrices for one colour from a
    ``edges_<video_stem>.mat`` file.

    Returns ``(X, Y)`` both sliced to the head-and-immediate-body
    columns the geometry code expects. Raises ``FileNotFoundError``
    if the file doesn't exist and ``KeyError`` if the colour-specific
    variables are missing from the file.
    """
    folder = Path(folder)
    path = folder / f"edges_{video_stem}.mat"
    if not path.exists():
        raise FileNotFoundError(f"edges file not found: {path}")

    color_key = color_key.lower()[:1]
    if color_key not in _COLOR_TO_EDGE_VARS:
        raise ValueError(f"unknown colour key: {color_key!r}")
    x_var, y_var = _COLOR_TO_EDGE_VARS[color_key]

    mat = _load_mat_any(path)
    if x_var not in mat or y_var not in mat:
        raise KeyError(
            f"edges file {path.name} is missing required variables "
            f"{x_var!r}/{y_var!r} for colour {color_key!r}. Found: "
            f"{sorted(mat.keys())}"
        )

    X = np.asarray(mat[x_var], dtype=float)
    Y = np.asarray(mat[y_var], dtype=float)
    if X.shape != Y.shape:
        raise ValueError(
            f"X/Y shape mismatch in {path.name}: {X.shape} vs {Y.shape}"
        )
    if X.shape[1] < edge_col_end:
        raise ValueError(
            f"{path.name}: expected at least {edge_col_end} columns, "
            f"got {X.shape[1]}"
        )

    return X[:, edge_col_start:edge_col_end], Y[:, edge_col_start:edge_col_end]


def load_depths(
    folder: "str | Path",
    video_stem: str,
    color_key: str,
    *,
    max_depth_cols: int = 375,
) -> Optional[np.ndarray]:
    """Load the depth matrix for one colour from a
    ``depths_<video_stem>.mat`` file.

    Returns ``None`` if the depths file is missing or the colour's
    variable isn't found or has too few columns (the geometry code
    treats depth as optional and will fill ``depth_metric`` with
    NaN in that case).
    """
    folder = Path(folder)
    path = folder / f"depths_{video_stem}.mat"
    if not path.exists():
        return None

    color_key = color_key.lower()[:1]
    depth_var = _COLOR_TO_DEPTH_VAR.get(color_key)
    if depth_var is None:
        return None

    try:
        mat = _load_mat_any(path)
    except Exception:
        return None

    D = mat.get(depth_var)
    if D is None or D.size == 0:
        return None

    D = np.asarray(D, dtype=float)
    if D.shape[1] < max_depth_cols:
        return None
    return D[:, :max_depth_cols]


def load_hlabel(
    folder: "str | Path",
    video_stem: str,
    color_key: str,
) -> Optional[np.ndarray]:
    """Load the 80-state cluster label CSV for one colour, or
    ``None`` if missing.
    """
    folder = Path(folder)
    color_name = _COLOR_LONG[color_key.lower()[:1]]
    path = folder / f"hlabel_{color_name}_{video_stem}.csv"
    if not path.exists():
        return None
    return np.loadtxt(str(path), delimiter=",").ravel()


def load_hcoord(
    folder: "str | Path",
    video_stem: str,
    color_key: str,
) -> Optional[np.ndarray]:
    """Load the 2D t-SNE coordinates CSV for one colour, or
    ``None`` if missing. Returns an ``(N, 2)`` array.
    """
    folder = Path(folder)
    color_name = _COLOR_LONG[color_key.lower()[:1]]
    path = folder / f"hcoord_{color_name}_{video_stem}.csv"
    if not path.exists():
        return None
    arr = np.loadtxt(str(path), delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr


def load_cage_boundaries(
    folder: "str | Path",
    video_stem: str,
) -> Optional[np.ndarray]:
    """Load the 3x4 cage boundaries matrix, or ``None`` if the file
    doesn't exist (the caller should fall back to
    :meth:`CageGeometry.default`).
    """
    folder = Path(folder)
    path = folder / f"{video_stem}_cage_boundaries.mat"
    if not path.exists():
        log.debug(f"[load_cage_boundaries] file not found: {path.name}")
        return None
    try:
        mat = _load_mat_any(path)
    except Exception:
        return None
    b = mat.get("boundaries")
    if b is None:
        return None
    b = np.asarray(b, dtype=float)
    log.debug(f"[load_cage_boundaries] file: {path.name}")
    log.debug(f"[load_cage_boundaries]   row 0 (cage floor):  x={b[0,0]:.1f}  y={b[0,1]:.1f}  w={b[0,2]:.1f}  h={b[0,3]:.1f}")
    log.debug(f"[load_cage_boundaries]   row 1 (door):        x={b[1,0]:.1f}  y={b[1,1]:.1f}  w={b[1,2]:.1f}  h={b[1,3]:.1f}")
    log.debug(f"[load_cage_boundaries]   row 2 (edge range):  x={b[2,0]:.1f}  y={b[2,1]:.1f}  w={b[2,2]:.1f}  h={b[2,3]:.1f}")
    return b


def load_stim_center(
    folder: "str | Path",
    video_stem: str,
) -> Optional[np.ndarray]:
    """Load the stimulus centre ``(x, y)`` from a ``<stem>_center.txt``
    file, or ``None`` if missing or unreadable.
    """
    folder = Path(folder)
    for candidate in (
        folder / f"{video_stem}_center.txt",
        folder / f"{video_stem}_center",
    ):
        if not candidate.exists():
            continue
        try:
            v = np.loadtxt(str(candidate)).ravel()
        except Exception:
            continue
        v = v[np.isfinite(v)]
        if v.size >= 2:
            result = np.array([float(v[0]), float(v[1])])
            log.debug(f"[load_stim_center] file: {candidate.name}")
            log.debug(f"[load_stim_center]   raw values in file: {v[:4]}")
            log.debug(f"[load_stim_center]   returning as [x, y] = [{result[0]:.1f}, {result[1]:.1f}]")
            return result
    log.debug(f"[load_stim_center] no center file found for {video_stem}")
    return None


def discover_video_stems(folder: "str | Path") -> list:
    """Return the sorted list of video stems present in a folder,
    as inferred from ``edges_<stem>.mat`` files. Used by
    :func:`~marm_viz.collect.video_data.build_recording_set` to
    iterate over every video in an experiment folder without having
    to list colours or auxiliary files.
    """
    folder = Path(folder)
    stems = []
    for p in sorted(folder.glob("edges_*.mat")):
        stem = p.stem[len("edges_"):]
        stems.append(stem)
    return stems
