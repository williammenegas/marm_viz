# marm_viz

Visualization toolkit for marmoset homecage behavioral data. Designed to work with outputs from [`marm_behavior`](https://github.com/williammenegas/marm_behavior).

## Installation

```bash
pip install marm-viz
```

For the interactive annotation tools (cage boundaries, stimulus centre), install with the `annotate` extra:

```bash
pip install 'marm-viz[annotate]'
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

## CLI Reference

### Mode Selection

If neither `--plot` nor `--annotate` is given, all 10 plot types are generated.

| Flag | Description |
|------|-------------|
| `--plot TYPE` | Generate a single plot type (see table above) |
| `--annotate MODE` | Interactive annotation: `metadata`, `boundaries`, `center`, or `all` |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR`, `-o` | `<folder>/marm_viz_output/` | Output directory (created if missing) |
| `--format FMT` | `jpg` | Output format: `jpg`, `svg`, `png`, or `pdf` |

### Recording Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--video STEM` | all | Process only this video stem (e.g. `test_4`) |
| `--color {r,w,b,y}` | all | Process only this animal colour |
| `--animal-to-use COLOR` | all | Same as `--color` but also accepts full names (`Red`, `White`, `Blue`, `Yellow`) |
| `--type-to-use STR` | all | Only process videos whose filename contains STR (case-insensitive) |
| `--time-to-use MINUTES` | all | Limit each recording to this many minutes of data |
| `--state-remap FILE` | bundled | Path to a custom state-reorder file (80 ints, one per line) |

### Plotting Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--fps N` | `60` | Frames per second |
| `--bin-size N` | `20` | Bin size in pixels for heatmap plots |
| `--radius-px N` | `120` | Near-stimulus radius in pixels |
| `--chunk-minutes N` | `5` | Chunk length in minutes for time series |
| `--n-frames N` | `50000` | Max frames for QC scatter plots (0 = use all) |
| `--scatter-sample-rate F` | `0.01` | Fraction of points for scatter subplots (1%) |
| `--mid-color {purple,green}` | `purple` | Colormap middle colour for density plots |
| `--log-alpha N` | `10` | Log tone-map aggressiveness for density (0 = linear) |
| `--use-body-xy` | off | Use body centroid instead of head position |
| `--augment` | off | Left/right flip augmentation (doubles data, removes spatial bias) |

### Annotation Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--force` | off | Re-annotate files that already exist |
| `--id-parent-dir DIR` | parent of `folder` | Where `animal_ID.txt` should live |

### General

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Print detailed diagnostic information |
| `-h`, `--help` | Show help and exit |

## Python API

```python
from marm_viz import build_recording_set, plot_ethogram

data = build_recording_set("/path/to/folder")
fig = plot_ethogram(data, fps=60)
fig.savefig("ethogram.svg")
```

For detailed documentation on each plot function and the `Recording` data structure, see the [User Guide](USER_GUIDE.md).

## License

MIT
