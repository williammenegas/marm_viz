"""
Core data structures
====================

A single :class:`Recording` represents one (video × animal) pair after
all the per-frame derived fields have been computed: head/body
positions in cage-centered coordinates, head direction, distance and
angle to the optional stimulus, depth metric, and behavior cluster
labels. Multiple recordings are bundled together in a
:class:`RecordingSet` for batch plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Recording:
    """All per-frame data for a single (video × animal) pair.

    The arrays are aligned: every ``(N,)`` or ``(N, k)`` field has
    the same number of rows ``N``, where ``N`` is the number of
    valid frames after trimming and clipping. NaN entries are used
    for frames where a quantity could not be computed (e.g., the
    animal was outside the valid arena for that frame).

    Attributes
    ----------
    n_frames:
        Number of frames in this recording (``= len(head_xy)``).
    head_xy:
        ``(N, 2)`` head position in cage-centered pixel coordinates.
        ``head_xy[:, 0]`` is x, ``head_xy[:, 1]`` is y. Bottom-left
        of the cage is ``(-cage_w/2, -cage_h/2)``.
    body_xy:
        ``(N, 2)`` body centroid position (median of body parts
        beyond the head triangle), same coordinate system.
    head_front_xy:
        ``(N, 2)`` position of the front body part of the head
        triangle (used to compute head direction).
    head_center_xy:
        ``(N, 2)`` position of the centre body part of the head
        triangle (anchor for head direction and distance to stim).
    head_angle:
        ``(N,)`` head direction in degrees, measured from the
        ``head_forward_ref`` direction (default: ``[0, -1]``, i.e.
        pointing in the negative-y direction).
    angle_to_stim:
        ``(N,)`` angle in degrees between the head direction and
        the vector from the head centre to the stimulus. NaN if the
        stimulus position is unknown for this recording.
    dist_to_stim:
        ``(N,)`` Euclidean distance from the head centre to the
        stimulus, in pixels. NaN if the stimulus is unknown.
    depth_metric:
        ``(N,)`` per-frame depth-spread metric (Q75 - Q25 of the
        375 depth-lookup samples for this animal). High values
        indicate the animal occupies a wide depth range that
        frame; low values mean it's on a single depth plane.
    behav_cat:
        ``(N,)`` integer broad-category label in ``{1, 2, 3, 4, 5}``
        or NaN. Mapped from the 80-state cluster IDs via the
        :func:`marm_viz.collect.state_remap.broad_category_for`
        function. The five categories are: 1=climbing,
        2=active, 3=alone, 4=near-other-active,
        5=near-other-resting.
    states_remapped:
        ``(N,)`` integer 80-state cluster label after applying
        the user-supplied state-reorder lookup, or NaN if no
        labels were available.
    stim_xy_centered:
        ``(2,)`` stimulus position in cage-centered coordinates,
        or ``[nan, nan]`` if unknown.
    door_xy:
        ``(2,)`` door centre position in cage-centered coordinates,
        or ``[nan, nan]`` if unknown.
    bad_xy:
        ``(N,)`` boolean mask: True for frames that were rejected
        because at least one position field exceeded ``xy_absmax``
        (the cage absolute-position cutoff). All frame data on
        those frames is set to NaN.
    metadata:
        Free-form dict of source/identification info: video
        filename, colour, folder, animal id, genotype, etc.
    """

    n_frames: int

    head_xy: np.ndarray
    body_xy: np.ndarray
    head_front_xy: np.ndarray
    head_center_xy: np.ndarray

    head_angle: np.ndarray
    angle_to_stim: np.ndarray
    dist_to_stim: np.ndarray
    depth_metric: np.ndarray

    behav_cat: np.ndarray
    states_remapped: np.ndarray
    tsne_xy: np.ndarray  # shape (N, 2) — t-SNE coords from hcoord, or NaN

    stim_xy_centered: np.ndarray  # shape (2,)
    door_xy: np.ndarray  # shape (2,)

    bad_xy: np.ndarray  # shape (N,) bool

    metadata: dict = field(default_factory=dict)

    @property
    def color(self) -> str:
        """Animal colour: 'Red', 'White', 'Blue', or 'Yellow'."""
        return self.metadata.get("color", "Unknown")

    @property
    def fname(self) -> str:
        """Source video filename stem (no extension)."""
        return self.metadata.get("fname", "unknown")


@dataclass
class RecordingSet:
    """A collection of :class:`Recording` objects, plus utilities for
    flipping (data augmentation) and concatenation.

    The visualization functions in :mod:`marm_viz.plot` accept either
    a single :class:`Recording` or a :class:`RecordingSet`. When
    given a :class:`RecordingSet`, they pool data across all member
    recordings before plotting.
    """

    recs: List[Recording] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.recs)

    def __iter__(self):
        return iter(self.recs)

    def append(self, rec: Recording) -> None:
        self.recs.append(rec)

    def augment_flip_x(self) -> "RecordingSet":
        """Return a new :class:`RecordingSet` with each recording
        plus an x-flipped copy. Used by the QC plots so left/right
        bias in the lab arena doesn't bleed into the heatmaps.
        """
        out = RecordingSet()
        for rec in self.recs:
            out.append(rec)
            out.append(_flip_recording_x(rec))
        return out

    def concatenate(self) -> Recording:
        """Concatenate every recording in this set into a single
        :class:`Recording` whose arrays are vertical stacks of the
        member arrays. Metadata is replaced with a synthetic
        "ALL/CONCAT" stub, but cage dimensions and stimulus position
        are carried through from the first recording that has them.
        """
        if not self.recs:
            raise ValueError("cannot concatenate empty RecordingSet")

        def stack(field_name: str) -> np.ndarray:
            return np.concatenate(
                [getattr(r, field_name) for r in self.recs], axis=0
            )

        # Carry through cage dims from first recording that has them.
        cage_half_w = 0.0
        cage_half_h = 0.0
        for r in self.recs:
            hw = r.metadata.get("cage_half_w", 0)
            hh = r.metadata.get("cage_half_h", 0)
            if hw > 0 and hh > 0:
                cage_half_w = hw
                cage_half_h = hh
                break

        # Carry through the first valid stim position.
        stim_xy = np.array([np.nan, np.nan])
        for r in self.recs:
            if np.all(np.isfinite(r.stim_xy_centered)):
                stim_xy = r.stim_xy_centered.copy()
                break

        return Recording(
            n_frames=sum(r.n_frames for r in self.recs),
            head_xy=stack("head_xy"),
            body_xy=stack("body_xy"),
            head_front_xy=stack("head_front_xy"),
            head_center_xy=stack("head_center_xy"),
            head_angle=stack("head_angle"),
            angle_to_stim=stack("angle_to_stim"),
            dist_to_stim=stack("dist_to_stim"),
            depth_metric=stack("depth_metric"),
            behav_cat=stack("behav_cat"),
            states_remapped=stack("states_remapped"),
            tsne_xy=stack("tsne_xy"),
            stim_xy_centered=stim_xy,
            door_xy=np.array([np.nan, np.nan]),
            bad_xy=stack("bad_xy"),
            metadata={
                "fname": "CONCAT",
                "color": "ALL",
                "family": "ALL",
                "cage_half_w": cage_half_w,
                "cage_half_h": cage_half_h,
            },
        )


def _flip_recording_x(rec: Recording) -> Recording:
    """Internal helper: return a copy of ``rec`` with all x
    coordinates negated. Used by :meth:`RecordingSet.augment_flip_x`.
    """
    def flip_xy(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        if out.size > 0:
            out[..., 0] = -out[..., 0]
        return out

    return Recording(
        n_frames=rec.n_frames,
        head_xy=flip_xy(rec.head_xy),
        body_xy=flip_xy(rec.body_xy),
        head_front_xy=flip_xy(rec.head_front_xy),
        head_center_xy=flip_xy(rec.head_center_xy),
        head_angle=rec.head_angle.copy(),
        angle_to_stim=rec.angle_to_stim.copy(),
        dist_to_stim=rec.dist_to_stim.copy(),
        depth_metric=rec.depth_metric.copy(),
        behav_cat=rec.behav_cat.copy(),
        states_remapped=rec.states_remapped.copy(),
        tsne_xy=rec.tsne_xy.copy(),  # t-SNE space is arbitrary; don't flip
        stim_xy_centered=flip_xy(rec.stim_xy_centered.reshape(1, 2)).reshape(2,),
        door_xy=flip_xy(rec.door_xy.reshape(1, 2)).reshape(2,),
        bad_xy=rec.bad_xy.copy(),
        metadata={**rec.metadata, "fname": rec.metadata.get("fname", "") + "_FLIPX"},
    )
