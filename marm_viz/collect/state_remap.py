"""
Behavior state reordering and broad-category mapping
====================================================

The marm_behavior nn stage produces 80-state cluster IDs (1..80).
For visualization purposes the lab applies two transformations:

1. **Reordering**: a fixed permutation that puts semantically related
   clusters next to each other. The permutation is provided via a
   user-supplied list of 80 integers (typically loaded from
   ``ts_db_list.mat``'s ``list_5`` variable in the original MATLAB
   pipeline). This package ships an identity remap by default;
   override via :func:`set_state_remap` or by passing a ``state_remap``
   argument to :func:`marm_viz.collect.video_data.build_recording`.

2. **Broad categorization**: the 80 reordered states are grouped into
   5 broad behavioral categories for the colour-coded scatter plots:

   ============  =====================  =================
   Category id   Label                  Colour
   ============  =====================  =================
   1             climbing               blue
   2             active                 purple
   3             alone                  red
   4             near other active      orange
   5             near other resting     yellow
   ============  =====================  =================

   The category groupings are baked in (matching the original
   MATLAB ``map_states_to_broad_categories``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Broad category groupings (from the lab's reference MATLAB code).
# ---------------------------------------------------------------------------

# These are 1-indexed state IDs in the *reordered* (post-remap) space.
_INDS_RED_RESTING = [
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
]
_INDS_ACTIVE_HIGH = [30, 31, 32, 33, 35, 40, 41, 42]
_INDS_NEAR_OTHER_ACTIVE = [
    36, 37, 38, 39, 44, 45, 46, 47, 48, 49, 50, 51, 52, 58, 59,
]
_INDS_ACTIVE_MID = [15, 19, 21, 22, 23, 24, 27, 29, 34]
_INDS_ACTIVE_LOW = [6, 7, 8, 9, 10, 11, 12, 28]
_INDS_CLIMBING = [
    1, 2, 3, 4, 5, 13, 14, 16, 17, 18, 20, 25, 26,
]
_INDS_NEAR_OTHER_RESTING = [43, 53, 54, 55, 56, 57]


#: Colours used by the broad-category scatter plots, as 0..1 RGB tuples.
CATEGORY_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (102 / 255, 98 / 255, 215 / 255),   # climbing — blue
    2: (168 / 255, 95 / 255, 186 / 255),   # active — purple
    3: (240 / 255, 92 / 255, 155 / 255),   # alone — red
    4: (251 / 255, 141 / 255, 123 / 255),  # near other active — orange
    5: (254 / 255, 193 / 255, 93 / 255),   # near other resting — yellow
}

#: Human-readable names for the five broad categories.
CATEGORY_NAMES: Dict[int, str] = {
    1: "climbing",
    2: "active",
    3: "alone",
    4: "near other active",
    5: "near other resting",
}


def broad_category_for(state_ids: np.ndarray) -> np.ndarray:
    """Map an array of reordered 80-state cluster IDs to broad
    category labels in ``{1, 2, 3, 4, 5}``.

    Parameters
    ----------
    state_ids:
        ``(N,)`` integer array of cluster IDs (post-remap, in
        ``[1, 80]``). NaN entries (frames without a label) are
        passed through as NaN.

    Returns
    -------
    np.ndarray
        ``(N,)`` float array of category IDs. Float (not int) so
        NaN can be represented for frames without labels.
    """
    state_ids = np.asarray(state_ids, dtype=float).ravel()
    out = np.full_like(state_ids, np.nan)

    # The five categories. Order matters because some clusters appear
    # in multiple groupings (matching the MATLAB precedence).
    is_blue = np.isin(state_ids, _INDS_CLIMBING)
    is_purple = np.isin(
        state_ids, _INDS_ACTIVE_LOW + _INDS_ACTIVE_MID + _INDS_ACTIVE_HIGH
    )
    is_red = np.isin(state_ids, _INDS_RED_RESTING)
    is_orange = np.isin(state_ids, _INDS_NEAR_OTHER_ACTIVE)
    is_yellow = np.isin(state_ids, _INDS_NEAR_OTHER_RESTING)

    out[is_blue] = 1
    out[is_purple] = 2
    out[is_red] = 3
    out[is_orange] = 4
    out[is_yellow] = 5
    return out


# ---------------------------------------------------------------------------
# Reordering lookup
# ---------------------------------------------------------------------------

def _identity_remap(n_states: int = 80) -> np.ndarray:
    """Return the identity remap: state ``i`` → state ``i``."""
    return np.arange(1, n_states + 1, dtype=int)


def _bundled_remap_path() -> Path:
    """Return the path to the bundled ``default_state_remap.txt``
    file, which ships the lab's ``list_5`` permutation of the 80
    raw cluster IDs. This is what every plot uses by default so
    cluster ordering (and therefore broad-category grouping) is
    consistent with the rest of the lab's figures.
    """
    return Path(__file__).resolve().parent.parent / "data" / "default_state_remap.txt"


def load_state_remap(path: "str | Path | None" = None) -> np.ndarray:
    """Load a state-reorder lookup from a text file (one integer per
    line).

    Resolution order:

    1. If ``path`` is given and exists, load it.
    2. Otherwise fall back to the bundled ``default_state_remap.txt``
       (the lab's ``list_5`` permutation of 1..80, extracted from
       ``ts_db_list.mat``).
    3. As a last resort (bundled file missing somehow), return the
       identity remap so the caller still gets a usable array.

    The returned array ``L`` is used as ``L[raw_label - 1]`` to
    translate raw cluster IDs (as written by the marm_behavior nn
    stage) into the reordered space the broad-category mapping
    expects.
    """
    if path is not None:
        p = Path(path)
        if p.exists():
            return np.loadtxt(str(p), dtype=int).ravel()

    bundled = _bundled_remap_path()
    if bundled.exists():
        return np.loadtxt(str(bundled), dtype=int).ravel()

    return _identity_remap()


def apply_state_remap(
    raw_labels: np.ndarray,
    remap: "np.ndarray | None" = None,
) -> np.ndarray:
    """Reorder raw cluster IDs into the canonical visualization order.

    Parameters
    ----------
    raw_labels:
        ``(N,)`` integer array of raw 1-indexed cluster IDs (the
        contents of an ``hlabel_<Color>_<video>.csv`` file).
    remap:
        ``(80,)`` integer reorder lookup. If None, identity is used
        and the returned array equals ``raw_labels`` (modulo the
        float promotion needed to hold NaNs).

    Returns
    -------
    np.ndarray
        ``(N,)`` float array. NaNs in the input or out-of-range raw
        IDs are passed through as NaN.
    """
    raw = np.asarray(raw_labels, dtype=float).ravel()
    if remap is None:
        remap = _identity_remap()
    out = np.full_like(raw, np.nan)
    valid = np.isfinite(raw) & (raw >= 1) & (raw <= len(remap))
    idx = (raw[valid].astype(int) - 1)
    out[valid] = remap[idx]
    return out
