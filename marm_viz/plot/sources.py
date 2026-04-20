"""
Source-data export helpers
==========================

Each plot function in :mod:`marm_viz.plot` optionally attaches a dict
of per-panel source arrays to the ``matplotlib.figure.Figure`` it
returns via :func:`attach_sources`. The CLI (and any library caller)
can then pull those arrays off with :func:`save_sources_to_csv`,
writing one CSV file per panel alongside the main PNG.

The source dicts use short panel names as keys (e.g. ``"panel_1_xy_depth"``)
and ``dict[str, np.ndarray]`` values — each inner dict becomes one
CSV, one column per inner key, aligned row-wise. The attachment
mechanism uses a private figure attribute, so library users who
don't care about source CSVs see no API change at all.

Design
------
* Plot functions build a dict ``{panel_name: {col: array, ...}, ...}``
  as they plot and call :func:`attach_sources` once at the end.
* :func:`save_sources_to_csv` writes one ``<prefix>_source_<panel>.csv``
  per attached panel, using ``np.savetxt`` so we don't require pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

# Matplotlib's Figure is only used as an anonymous attribute carrier;
# we don't import it at module load time to keep imports light.


#: Attribute name used to stash the sources dict on a Figure.
#: Underscore-prefixed so it's clearly private and doesn't collide
#: with any future matplotlib attribute.
_SOURCES_ATTR = "_marm_viz_sources"


def attach_sources(
    fig,
    sources: "Mapping[str, Mapping[str, np.ndarray]]",
) -> None:
    """Attach a per-panel source-data dict to a Figure.

    Stores the dict under ``fig._marm_viz_sources`` so the CLI (and
    any library caller) can retrieve it later via :func:`get_sources`
    and write it out as CSVs via :func:`save_sources_to_csv`.

    Calling this repeatedly merges new panels into any existing dict
    rather than overwriting — useful for the stim-time-series plot
    which returns three figures but only builds one unified source
    dict. (In practice each plot function calls this once, though.)

    Parameters
    ----------
    fig:
        The matplotlib Figure to tag.
    sources:
        ``{panel_name: {column_name: 1d_ndarray, ...}, ...}``.
        All arrays under one panel must have the same length.
    """
    existing = getattr(fig, _SOURCES_ATTR, None) or {}
    merged = dict(existing)
    merged.update(sources)
    setattr(fig, _SOURCES_ATTR, merged)


def get_sources(fig) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Return the sources dict stashed on ``fig``, or None if the
    plot function didn't attach one.
    """
    return getattr(fig, _SOURCES_ATTR, None)


def save_sources_to_csv(
    fig,
    out_prefix: "str | Path",
) -> "list[Path]":
    """Write one CSV per panel attached to ``fig``.

    File naming follows the pattern::

        <out_prefix>_source_<panel_name>.csv

    For example, with ``out_prefix='/tmp/qc_all'`` and a panel named
    ``'panel_1_xy_depth'``, the output file is
    ``/tmp/qc_all_source_panel_1_xy_depth.csv``.

    Each CSV has a header row with the column names and one row per
    data point. NaNs are written as the string ``nan`` (numpy
    default).

    Parameters
    ----------
    fig:
        Figure previously tagged with :func:`attach_sources`. Figures
        without sources are a no-op — the function returns an empty
        list.
    out_prefix:
        Path stem (without extension). A common pattern is to use the
        PNG path with its ``.png`` suffix stripped::

            save_sources_to_csv(fig, png_path.with_suffix(''))

    Returns
    -------
    list[Path]
        Paths of the CSV files written. Empty if the figure has no
        attached sources.
    """
    sources = get_sources(fig)
    if not sources:
        return []

    out_prefix = Path(out_prefix)
    written: "list[Path]" = []

    for panel_name, columns in sources.items():
        if not columns:
            continue
        # Sanitise panel name for filesystem use — just replace
        # anything other than alnum/underscore/hyphen with '_'.
        safe = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in panel_name
        )
        out_path = out_prefix.parent / (
            f"{out_prefix.name}_source_{safe}.csv"
        )

        col_names = list(columns.keys())
        arrs = [np.asarray(columns[k]).ravel() for k in col_names]

        # All columns must have the same length. If not, pad with NaN
        # rather than raising — plots sometimes want to export
        # different-length columns side-by-side (e.g. a histogram's
        # bin edges and counts are N+1 vs N).
        n = max((a.size for a in arrs), default=0)
        if n == 0:
            continue
        padded = []
        for a in arrs:
            if a.size == n:
                padded.append(a.astype(float, copy=False))
            else:
                out = np.full(n, np.nan, dtype=float)
                out[: a.size] = a.astype(float, copy=False)
                padded.append(out)
        stacked = np.column_stack(padded)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(col_names)
        np.savetxt(
            str(out_path),
            stacked,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6g",
        )
        written.append(out_path)

    return written
