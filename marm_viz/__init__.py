"""
marm_viz
========

Visualization package for marm_behavior outputs. Loads the artifacts
written by the marm_behavior pipeline (edges_*.mat, depths_*.mat,
hlabel_*.csv) into a per-(video x animal) data structure and provides
a set of plotting routines for spatial, behavioural, and density
visualizations.

Quick-start (library use)::

    from marm_viz import build_recording_set, plot_qc_panels
    rs = build_recording_set('path/to/marm_behavior/folder')
    fig = plot_qc_panels(rs)
    fig.savefig('qc.png', dpi=150)

Quick-start (CLI)::

    python -m marm_viz path/to/folder --plot qc
    python -m marm_viz path/to/folder --plot heatmaps
    python -m marm_viz path/to/folder --plot timeseries
    python -m marm_viz path/to/folder --plot density --mid-color green
"""

from .types import Recording, RecordingSet
from .collect.video_data import build_recording, build_recording_set
from .plot.qc_panels import plot_qc_panels
from .plot.category_heatmaps import (
    plot_category_fraction_heatmaps,
    plot_xy_occupancy_heatmap,
    plot_stay_move_heatmaps,
)
from .plot.stim_time_series import plot_stim_proximity_time_series
from .plot.density_2d import plot_density_2d
from .plot.behavior_overview import plot_state_usage, plot_tsne_centroids
from .plot.ethogram import plot_ethogram
from .plot.behavior_detail import (
    plot_transition_matrix,
    plot_head_direction_polar,
)
from .plot.correlation import plot_cagemate_correlation

__all__ = [
    "Recording",
    "RecordingSet",
    "build_recording",
    "build_recording_set",
    "plot_qc_panels",
    "plot_category_fraction_heatmaps",
    "plot_xy_occupancy_heatmap",
    "plot_stay_move_heatmaps",
    "plot_stim_proximity_time_series",
    "plot_density_2d",
    "plot_state_usage",
    "plot_tsne_centroids",
    "plot_ethogram",
    "plot_transition_matrix",
    "plot_head_direction_polar",
    "plot_cagemate_correlation",
]

__version__ = "0.1.0"
