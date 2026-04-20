"""
marm_viz.annotate
=================

Interactive tools for creating the auxiliary files the collect stage
needs but which marm_behavior doesn't produce:

* :mod:`marm_viz.annotate.metadata` — console prompts for
  ``animals_present.txt``, ``animal_ID.txt``, ``animals_genotype.txt``.
* :mod:`marm_viz.annotate.cage_boundaries` — click-and-drag interactive
  cage annotation; writes ``<stem>_cage_boundaries.mat``. Port of
  the lab's ``find_cage_boundaries.m``.
* :mod:`marm_viz.annotate.stim_center` — click-once stimulus centre
  annotation; writes ``<stem>_center.txt``. Port of
  ``find_center_of_stimulus.m``.

The visual tools (cage boundaries, stim centre) require
``opencv-python`` for .avi reading. Install with the optional extra::

    pip install 'marm_viz[annotate]'

The metadata prompt module is pure stdlib and works without any
extra dependencies.
"""

from .metadata import (
    prompt_metadata,
    COLORS,
    GENOTYPE_CODES,
)
from .cage_boundaries import annotate_cage_boundaries, annotate_folder_cages
from .stim_center import annotate_stim_center, annotate_folder_centers

__all__ = [
    "COLORS",
    "GENOTYPE_CODES",
    "prompt_metadata",
    "annotate_cage_boundaries",
    "annotate_folder_cages",
    "annotate_stim_center",
    "annotate_folder_centers",
]
