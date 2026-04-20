# marm_viz

Visualization toolkit for marmoset homecage behavioral data. Designed to work with outputs from [`marm_behavior`](https://github.com/williammenegas/marm_behavior).

## Installation

```bash
pip install marm-viz
```

## Quick Start

Point `marm_viz` at a folder containing `marm_behavior` outputs (hlabel, hcoord, edges, depths files):

```bash
# Generate all 10 plot types (default: JPG output)
python -m marm_viz /path/to/recording_folder

# Single plot type
python -m marm_viz /path/to/recording_folder --plot ethogram

# Vector output for Illustrator
python -m marm_viz /path/to/recording_folder --format svg
```

## Plot Types

| Flag | Output | Description |
|------|--------|-------------|
| `qc` | QC panels | 8-panel overview: head XY colored by depth/distance/angle/behavior + histograms |
| `heatmaps` | Category heatmaps | Per-category spatial occupancy + near-stimulus + overall occupancy + P(move) |
| `timeseries` | Stim proximity | Fraction near stimulus across time chunks + per-chunk heatmaps + pooled XY |
| `density` | 2D density | Smoothed occupancy density with contours |
| `behavior` | State usage + t-SNE | 80-state bar chart grouped by category + t-SNE centroid bubble plot |
| `ethogram` | Ethogram | Per-category binary raster for each animal across time |
| `transitions` | Transition matrix | 5x5 category transition probabilities (diagonal NaN'd) |
| `polar` | Head direction | Polar histogram of head angle relative to stimulus (near vs far) |
| `track` | Head tracks | Per-animal + merged XY trajectory line plots |
| `correlation` | Cagemate correlation | Pairwise Spearman correlation of 80-state usage vectors across time chunks |

## CLI Options

```
--plot TYPE        Which plot to generate (default: all)
--format FMT       Output format: jpg (default), svg, png, pdf
--output-dir DIR   Output directory (default: <input>/marm_viz_output/)
--video STEM       Only process this video stem
--type-to-use STR  Only process videos whose filename contains STR (case-insensitive)
--verbose          Print detailed diagnostic information
--fps N            Frames per second (default: 60)
--chunk-minutes N  Chunk size in minutes for time series (default: 5)
--radius-px N      Near-stimulus radius in pixels (default: 120)
--augment          Enable left/right flip augmentation
```

## Python API

```python
from marm_viz import build_recording_set, plot_ethogram

data = build_recording_set("/path/to/folder")
fig = plot_ethogram(data, fps=60)
fig.savefig("ethogram.svg")
```

## License

MIT
