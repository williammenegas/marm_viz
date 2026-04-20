"""
Cage geometry and per-frame derived fields
==========================================

Functions for converting raw edge-matrix coordinates (in pixel space)
into the cage-centered, optionally-rotated coordinate system that the
visualization uses.

The pipeline mirrors ``compute_head_body_angles_and_dist_centeronly_with_depths``
from the MATLAB ``collect_video_data_3.m`` script, but split into
small testable pieces.
"""

from __future__ import annotations

import logging
import warnings

log = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CageGeometry:
    """Cage geometry annotation.

    Attributes
    ----------
    cage_center:
        ``(2,)`` pixel coordinates of the cage centre.
    door_center:
        ``(2,)`` pixel coordinates of the cage door centre. Used to
        decide whether the recording needs a 180-degree rotation
        (so the door always ends up on the same side after
        normalization).
    edge_x_range:
        ``(x1, x2)`` valid pixel-x range. Frames where the head's
        x falls outside this range are NaN'd out.
    """

    cage_center: np.ndarray
    door_center: np.ndarray
    edge_x_range: Tuple[float, float]
    floor_half_w: float = 0.0
    floor_half_h: float = 0.0

    @classmethod
    def from_boundaries_matrix(cls, boundaries: np.ndarray) -> "CageGeometry":
        """Build a :class:`CageGeometry` from a 3x4 boundaries matrix
        of the same format as ``<video>_cage_boundaries.mat``.

        The MATLAB convention (which this follows) is:

        * Row 0: ``[x, y, w, h]`` of the cage rectangle
        * Row 1: ``[x, y, w, h]`` of the door rectangle
        * Row 2: ``[x, y, w, h]`` of a region whose ``x`` and
          ``x + w`` are used as the valid edge range.
        """
        b = np.asarray(boundaries, dtype=float)
        if b.shape != (3, 4):
            raise ValueError(
                f"boundaries must be a 3x4 matrix, got shape {b.shape}"
            )
        cage_center = np.array(
            [b[0, 0] + 0.5 * b[0, 2], b[0, 1] + 0.5 * b[0, 3]]
        )
        door_center = np.array(
            [b[1, 0] + 0.5 * b[1, 2], b[1, 1] + 0.5 * b[1, 3]]
        )
        edge_x_range = (float(b[2, 0]), float(b[2, 0] + b[2, 2]))
        log.debug(f"[CageGeometry] cage_center (px) = [{cage_center[0]:.1f}, {cage_center[1]:.1f}]")
        log.debug(f"[CageGeometry] door_center (px) = [{door_center[0]:.1f}, {door_center[1]:.1f}]")
        log.debug(f"[CageGeometry] edge_x_range     = [{edge_x_range[0]:.1f}, {edge_x_range[1]:.1f}]")
        return cls(
            cage_center=cage_center,
            door_center=door_center,
            edge_x_range=edge_x_range,
            floor_half_w=float(b[0, 2]) / 2.0,
            floor_half_h=float(b[0, 3]) / 2.0,
        )

    @classmethod
    def default(cls, frame_width: int = 1280, frame_height: int = 720) -> "CageGeometry":
        """Trivial fallback when no boundaries file is available:
        place the cage centre at the frame centre, the door at
        ``(centre_x, 0)``, and use the full x range.
        """
        return cls(
            cage_center=np.array([frame_width / 2.0, frame_height / 2.0]),
            door_center=np.array([frame_width / 2.0, 0.0]),
            edge_x_range=(0.0, float(frame_width)),
        )


def _rotmat_deg(deg: float) -> np.ndarray:
    """2x2 rotation matrix in degrees."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s], [s, c]])


def _angle_between_rows(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle in degrees between corresponding rows of two ``(N, 2)``
    arrays. NaN rows propagate to NaN output.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n = v1.shape[0]
    out = np.full(n, np.nan)
    good = np.all(np.isfinite(v1), axis=1) & np.all(np.isfinite(v2), axis=1)
    if not np.any(good):
        return out

    a = v1[good]
    b = v2[good]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    nz = (na > 0) & (nb > 0)
    if not np.any(nz):
        return out

    a = a[nz] / na[nz, None]
    b = b[nz] / nb[nz, None]
    dp = np.sum(a * b, axis=1)
    dp = np.clip(dp, -1.0, 1.0)
    angles = np.degrees(np.arccos(dp))

    idx = np.where(good)[0][nz]
    out[idx] = angles
    return out


def _safe_nanmedian_2d(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    """``np.nanmedian`` with the all-NaN warning suppressed."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        with np.errstate(invalid="ignore"):
            return np.nanmedian(arr, axis=axis)


def compute_per_frame_fields(
    X: np.ndarray,
    Y: np.ndarray,
    depths: "np.ndarray | None",
    cage: CageGeometry,
    stim_xy_pixel: "np.ndarray | None" = None,
    *,
    head_forward_ref: "np.ndarray | None" = None,
    xy_absmax: float = 500.0,
) -> dict:
    """Compute the per-frame derived fields used by the plotting code.

    This is the Python equivalent of
    ``compute_head_body_angles_and_dist_centeronly_with_depths`` from
    the MATLAB ``collect_video_data_3.m`` script.

    Parameters
    ----------
    X, Y:
        ``(N, n_edge_cols)`` float matrices of edge x and y
        coordinates from a marm_behavior ``edges_*.mat`` file. The
        first 4 columns are the head body parts; the remaining
        columns are body body-parts.
    depths:
        Optional ``(N, 375)`` float matrix of per-edge depth
        samples from ``depths_*.mat``. Used to compute the
        per-frame depth-spread metric. If None, the depth metric
        is all-NaN.
    cage:
        Cage geometry annotation (see :class:`CageGeometry`).
    stim_xy_pixel:
        Optional ``(2,)`` stimulus position in raw pixel
        coordinates. If None, the dist/angle-to-stim fields are
        all-NaN.
    head_forward_ref:
        Optional ``(2,)`` reference vector that defines the
        ``head_angle = 0`` direction. Defaults to ``[0, -1]``
        (pointing in the negative-y direction).
    xy_absmax:
        Frames whose absolute centred-position exceeds this in
        any coordinate are flagged ``bad_xy = True`` and have all
        their position-derived fields NaN'd out.

    Returns
    -------
    dict
        A dict with all the per-frame arrays the
        :class:`marm_viz.types.Recording` constructor needs.
    """
    if head_forward_ref is None:
        head_forward_ref = np.array([0.0, -1.0])
    head_forward_ref = np.asarray(head_forward_ref, dtype=float).ravel()
    head_forward_ref = head_forward_ref / max(np.linalg.norm(head_forward_ref), 1e-12)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError(
            f"X and Y must have the same shape; got {X.shape} vs {Y.shape}"
        )
    n = X.shape[0]
    if X.shape[1] < 5:
        raise ValueError(
            f"need at least 5 edge columns (4 head + 1+ body); got {X.shape[1]}"
        )

    head_cols = slice(0, 4)
    body_cols = slice(4, X.shape[1])

    head_x_raw = _safe_nanmedian_2d(X[:, head_cols], axis=1)
    head_y_raw = _safe_nanmedian_2d(Y[:, head_cols], axis=1)
    body_x_raw = _safe_nanmedian_2d(X[:, body_cols], axis=1)
    body_y_raw = _safe_nanmedian_2d(Y[:, body_cols], axis=1)

    head_front_raw = np.column_stack([X[:, 0], Y[:, 0]])
    head_cent_raw = np.column_stack([X[:, 3], Y[:, 3]])

    # Depth spread metric: Q75 - Q25 across the 375 columns per frame.
    depth_metric = np.full(n, np.nan)
    if depths is not None and depths.size > 0:
        D = np.asarray(depths, dtype=float)
        if D.shape[0] == n:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="All-NaN slice encountered"
                )
                with np.errstate(invalid="ignore"):
                    q = np.nanpercentile(D, [25, 75], axis=1)
            depth_metric = q[1] - q[0]

    cx, cy = cage.cage_center
    head = np.column_stack([head_x_raw - cx, head_y_raw - cy])
    body = np.column_stack([body_x_raw - cx, body_y_raw - cy])
    head_front = head_front_raw - cage.cage_center
    head_cent = head_cent_raw - cage.cage_center

    door = cage.door_center - cage.cage_center
    if stim_xy_pixel is not None and np.all(np.isfinite(stim_xy_pixel)):
        stim_raw = np.asarray(stim_xy_pixel, dtype=float).copy()
        # Both the stim annotation and the edge tracking data use
        # the same y-down pixel convention — no axis flip needed.
        stim_centered = stim_raw - cage.cage_center
    else:
        stim_centered = np.array([np.nan, np.nan])

    # --- Diagnostic: positions after centering, before any rotation ---
    _valid_head = np.all(np.isfinite(head), axis=1)
    log.debug(f"[compute] === AFTER CENTERING (before rotation) ===")
    log.debug(f"[compute]   cage_center (px)      = [{cx:.1f}, {cy:.1f}]")
    log.debug(f"[compute]   stim_xy_pixel (raw)   = {stim_xy_pixel}")
    log.debug(f"[compute]   stim_centered         = [{stim_centered[0]:.1f}, {stim_centered[1]:.1f}]")
    log.debug(f"[compute]   door_centered         = [{door[0]:.1f}, {door[1]:.1f}]")
    if _valid_head.any():
        _hx = np.nanmedian(head[_valid_head, 0])
        _hy = np.nanmedian(head[_valid_head, 1])
        log.debug(f"[compute]   head median (centered)= [{_hx:.1f}, {_hy:.1f}]  ({_valid_head.sum()} valid frames)")

    # NaN out frames whose x is outside the valid edge x range.
    edge_x1, edge_x2 = cage.edge_x_range
    in_head = (
        np.isfinite(head_x_raw)
        & (head_x_raw >= edge_x1)
        & (head_x_raw <= edge_x2)
    )
    in_body = (
        np.isfinite(body_x_raw)
        & (body_x_raw >= edge_x1)
        & (body_x_raw <= edge_x2)
    )
    head[~in_head] = np.nan
    head_front[~in_head] = np.nan
    head_cent[~in_head] = np.nan
    body[~in_body] = np.nan
    depth_metric[~in_head] = np.nan

    # If the door is above the cage centre in raw pixels, rotate the
    # whole frame 180 degrees so the door is always BELOW the cage
    # centre in the normalised output. This normalises the orientation
    # across recordings filmed from cameras mounted on opposite sides
    # of the rig.
    _do_rotate = np.isfinite(door[1]) and door[1] < 0
    log.debug(f"[compute]   door[1] = {door[1]:.1f}  →  180° rotation = {'YES' if _do_rotate else 'no'}")
    if _do_rotate:
        R = _rotmat_deg(180)
        head = head @ R.T
        body = body @ R.T
        head_front = head_front @ R.T
        head_cent = head_cent @ R.T
        door = R @ door
        if np.all(np.isfinite(stim_centered)):
            stim_centered = R @ stim_centered

    # --- Diagnostic: positions after rotation ---
    _valid_head2 = np.all(np.isfinite(head), axis=1)
    log.debug(f"[compute] === AFTER ROTATION ===")
    log.debug(f"[compute]   stim_centered (final) = [{stim_centered[0]:.1f}, {stim_centered[1]:.1f}]")
    log.debug(f"[compute]   door_centered (final) = [{door[0]:.1f}, {door[1]:.1f}]")
    if _valid_head2.any():
        _hx2 = np.nanmedian(head[_valid_head2, 0])
        _hy2 = np.nanmedian(head[_valid_head2, 1])
        log.debug(f"[compute]   head median (final)   = [{_hx2:.1f}, {_hy2:.1f}]  ({_valid_head2.sum()} valid frames)")
    log.debug(f"[compute] ==========================")

    # Hard reject frames where any centred xy exceeds the absolute cap.
    def exceeds(arr: np.ndarray) -> np.ndarray:
        return np.any(np.isfinite(arr) & (np.abs(arr) > xy_absmax), axis=1)

    bad_xy = exceeds(head) | exceeds(head_cent) | exceeds(head_front) | exceeds(body)
    head[bad_xy] = np.nan
    body[bad_xy] = np.nan
    head_front[bad_xy] = np.nan
    head_cent[bad_xy] = np.nan
    depth_metric[bad_xy] = np.nan

    # Head direction vector and angle.
    v_head_dir = head_front - head_cent
    head_angle = _angle_between_rows(
        v_head_dir, np.tile(head_forward_ref, (n, 1))
    )

    angle_to_stim = np.full(n, np.nan)
    dist_to_stim = np.full(n, np.nan)
    if np.all(np.isfinite(stim_centered)):
        v_stim = np.tile(stim_centered, (n, 1)) - head_cent
        angle_to_stim = _angle_between_rows(v_head_dir, v_stim)
        dist_to_stim = np.sqrt(
            (head_cent[:, 0] - stim_centered[0]) ** 2
            + (head_cent[:, 1] - stim_centered[1]) ** 2
        )
        dist_to_stim[~np.all(np.isfinite(head_cent), axis=1)] = np.nan

    head_angle[bad_xy] = np.nan
    angle_to_stim[bad_xy] = np.nan
    dist_to_stim[bad_xy] = np.nan

    # Convert from y-down pixel-centered to y-up cage-centered
    # coordinates. The edge data, cage boundaries, and stim annotation
    # are all in y-down image convention, so after centering the y axis
    # still points downward. The plotting code renders with
    # origin="lower" (y-up), and the Recording docstring defines
    # bottom-left as (-w/2, -h/2), so we negate y here so that
    # "below cage centre" → negative y, "above" → positive y.
    # Scalar quantities (head_angle, angle_to_stim, dist_to_stim)
    # are invariant under y-negation, so they are not affected.
    head[:, 1] *= -1
    body[:, 1] *= -1
    head_front[:, 1] *= -1
    head_cent[:, 1] *= -1
    stim_centered[1] *= -1
    door[1] *= -1

    return dict(
        n_frames=n,
        head_xy=head,
        body_xy=body,
        head_front_xy=head_front,
        head_center_xy=head_cent,
        head_angle=head_angle,
        angle_to_stim=angle_to_stim,
        dist_to_stim=dist_to_stim,
        depth_metric=depth_metric,
        stim_xy_centered=stim_centered,
        door_xy=door,
        bad_xy=bad_xy,
    )
