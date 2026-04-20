# marm_viz — User Guide

This guide explains each plot type, what it shows, how to call it
from Python and from the CLI, and what every flag does. For install
and quickstart see [README.md](README.md).

---

## Concepts

### The `Recording` data structure

Every plot function takes a `Recording` or `RecordingSet`. A
`Recording` is one (video × animal-colour) pair after the per-frame
derived fields have been computed. Its relevant arrays are all
length-`N` and aligned:

| Field | Shape | Meaning |
|---|---|---|
| `head_xy` | `(N, 2)` | Head position, cage-centered px |
| `body_xy` | `(N, 2)` | Body centroid, cage-centered px |
| `head_front_xy`, `head_center_xy` | `(N, 2)` | Head front / anchor for direction |
| `head_angle` | `(N,)` | Head direction in degrees |
| `dist_to_stim`, `angle_to_stim` | `(N,)` | Relative to the stimulus |
| `depth_metric` | `(N,)` | Q75 − Q25 of depth samples (proxy for 3D occupancy) |
| `behav_cat` | `(N,)` | Broad category 1..5 or NaN |
| `states_remapped` | `(N,)` | 80-state cluster ID (reordered) or NaN |
| `stim_xy_centered`, `door_xy` | `(2,)` | Per-video, cage-centered |
| `bad_xy` | `(N,)` bool | Frames rejected by the xy_absmax cap |

NaN propagates through all the plotting functions — frames where a
quantity couldn't be computed are just skipped. This means you can
have a recording with no depth file (depth panel goes NaN, every
other panel still works) or no stimulus centre (dist/angle panels go
NaN).

A `RecordingSet` is a list of `Recording` objects plus utilities:
`.augment_flip_x()` returns a new set with each recording plus an
x-flipped copy, and `.concatenate()` stacks every member into a
single pooled `Recording`. Every plot function that accepts a set
does these two steps internally before plotting, so you always get
a left/right-symmetric pooled view.

### The 5 broad behaviour categories

The 80 raw cluster IDs from the marm_behavior `nn` stage are grouped
into 5 categories for the coloured scatter plots and heatmaps:

| ID | Name | Colour |
|---|---|---|
| 1 | climbing | blue (`#6662D7`) |
| 2 | active | purple (`#A85FBA`) |
| 3 | alone | red (`#F05C9B`) |
| 4 | near other active | orange (`#FB8D7B`) |
| 5 | near other resting | yellow (`#FEC15D`) |

These colours are used consistently across every plot that shows
behaviour categories. Set via `marm_viz.collect.state_remap.CATEGORY_COLORS`.

### State reordering

The marm_behavior `nn` stage writes raw 80-state cluster IDs. The
lab's reference MATLAB pipeline applies a fixed permutation
(`list_5` from `ts_db_list.mat`) to reorder them into a canonical
ordering before the broad-category grouping is applied, so that
semantically related clusters end up in the same broad category.

marm_viz **ships this permutation bundled** as
`marm_viz/data/default_state_remap.txt` (80 lines, one integer per
line). It's loaded automatically by
`marm_viz.collect.state_remap.load_state_remap()` whenever you
don't pass an explicit path. The upshot is: the broad-category
colours, category-assignment logic, and everything downstream that
depends on state ordering is consistent with your other figures
out of the box, no configuration needed.

If you have a different reorder lookup (e.g., for a different
number of clusters), put it in a text file and pass it via
`--state-remap path/to/file.txt` on the CLI or
`state_remap=path` to `build_recording()` / `build_recording_set()`.
The loader resolves in this order:

1. If you pass a path and it exists, use it.
2. Otherwise, load the bundled `default_state_remap.txt`.
3. As a last resort (bundled file somehow missing), fall back to
   the identity remap 1..80.

---

## Plot modes

### `qc` — 8-panel QC figure

A single 2×4 figure giving you an at-a-glance overview of one or many
recordings. Ported from `plot_video_data_1.m`'s
`qc_plots_for_recording_depths_randomsample`.

**Panels:**

1. Head XY coloured by depth Q75−Q25 (clim = `[0, 20]`)
2. Head XY coloured by distance to stimulus
3. Head XY coloured by head-direction angle to stimulus
4. Head XY coloured by broad behaviour category (5 colours)
5. Histogram of depth Q75−Q25
6. Histogram of distance to stimulus
7. Histogram of angle to stimulus
8. Bar chart of behaviour-category usage fractions

**CLI:**

```bash
python -m marm_viz path/to/folder --plot qc
python -m marm_viz path/to/folder --plot qc --video test_4 --color w
python -m marm_viz path/to/folder --plot qc --n-frames 0      # plot everything
python -m marm_viz path/to/folder --plot qc --n-frames 20000  # random-sample 20k
```

**Library:**

```python
from marm_viz import plot_qc_panels, build_recording_set
rs = build_recording_set("folder")
fig = plot_qc_panels(rs, n_frames=50_000, depth_clim=(0, 20))
fig.savefig("qc.png", dpi=150, bbox_inches="tight")
```

Random sampling is important for big concatenations — plotting every
frame of a 4-animal × 2-hour recording is ~2M points per scatter and
matplotlib gets slow. The default 50k is enough to see every structural
feature.

---

### `heatmaps` — 5-panel + occupancy + stay/move

Three figures, all using the same `round(xy / bin_size)` binning so
bins line up across panels. Ported from `plot_video_data_2.m`.

**Figure 1 — category fraction heatmaps (5 panels):** one panel per
broad category. Each bin is coloured by the fraction of total frames
the animal spent in that bin in that category. All 5 panels share
the same colour scale (set to the global max across panels) so
category magnitudes are directly comparable. Each panel uses a
white → category-colour ramp. Zero-count bins are transparent.

**Figure 2 — xy occupancy heatmap (1 panel):** fraction of total
frames in each bin regardless of category. Uses viridis.

**Figure 3 — stay/move heatmaps (2 panels):** P(stay | in bin) and
P(move | in bin), both on `[0, 1]`. A frame counts as "staying" if
the next frame lands in the same bin. Useful for distinguishing
preferred rest spots from transit corridors.

**CLI:**

```bash
python -m marm_viz folder --plot heatmaps                  # default bin_size=20
python -m marm_viz folder --plot heatmaps --bin-size 10    # finer grid
python -m marm_viz folder --plot heatmaps --use-body-xy    # use body_xy instead of head_xy
```

**Library:**

```python
from marm_viz import (plot_category_fraction_heatmaps, plot_xy_occupancy_heatmap,
                      plot_stay_move_heatmaps)
fig_cat  = plot_category_fraction_heatmaps(rs, bin_size=20, use_head_xy=True)
fig_occ  = plot_xy_occupancy_heatmap(rs, bin_size=20, use_head_xy=True)
fig_stay = plot_stay_move_heatmaps(rs, bin_size=20, use_head_xy=True)
```

---

### `timeseries` — chunked stim proximity

Three figures looking at how often the animal's head is within a
user-set radius of the stimulus, broken down by time chunks. Ported
from `plot_video_data_ts_1.m`. Input is **not** flip-augmented or
concatenated — per-recording granularity matters here.

**Figure 1 — per-chunk fraction dotplot:** x-axis = chunk index,
y-axis = fraction of time in radius. Grey jittered dots = one per
(recording, chunk); black dot = mean across recordings for that chunk.

**Figure 2 — per-chunk xy scatter grid:** one subplot per chunk index,
showing a random sample of `head_xy` points pooled across recordings.
All subplots share the same axis limits.

**Figure 3 — inside/outside pooled scatter:** one panel with every
sampled point from every recording, coloured by whether it was
within `radius_px` of the stim (black) or not (grey).

**CLI:**

```bash
# Defaults: 5-minute chunks at 60 fps, radius 120 px, 1% scatter sampling
python -m marm_viz folder --plot timeseries

# Tighter radius, shorter chunks
python -m marm_viz folder --plot timeseries --radius-px 80 --chunk-minutes 2

# 30 fps recordings
python -m marm_viz folder --plot timeseries --fps 30

# Dense scatter (5% of points instead of 1%)
python -m marm_viz folder --plot timeseries --scatter-sample-rate 0.05
```

**Library:**

```python
from marm_viz import plot_stim_proximity_time_series
f1, f2, f3 = plot_stim_proximity_time_series(
    rs, radius_px=120, chunk_minutes=5, fps=60, scatter_sample_rate=0.01,
)
```

---

### `density` — log-compressed 2D density

A single figure: 2D histogram of coordinates, Gaussian-smoothed,
normalized to `[0, 1]`, log-tone-mapped to reduce peak-bin dominance,
and drawn with a distinctive white → mid → black colormap plus contour
lines at equal levels in the tone-mapped space. Ported from
`vis_colorful_v4_grid.m`.

**Why log compression?** Raw 2D histograms of animal trajectories
are extremely peaked — the animal has a few favourite spots and the
rest of the cage is visited rarely. On a linear colour scale, the
peak bin dominates and everything else is washed out. Log compression
(`Y = log1p(alpha*X) / log1p(alpha)`) makes low-count bins visible
without completely flattening the distribution.

**CLI:**

```bash
python -m marm_viz folder --plot density                     # default: purple, alpha=10
python -m marm_viz folder --plot density --mid-color green   # green ramp
python -m marm_viz folder --plot density --log-alpha 0       # linear (no compression)
python -m marm_viz folder --plot density --log-alpha 3       # mild compression
python -m marm_viz folder --plot density --use-body-xy       # use body instead of head
```

**Library:**

Two call signatures — pass arrays directly, or pass a recording:

```python
from marm_viz import plot_density_2d

# Direct arrays (normalized into [0, x_max] first)
u = ((rec.head_xy[:, 0] + 400) / 1000)
v = ((rec.head_xy[:, 1] + 400) / 1000)
fig = plot_density_2d(u, v, mid_color="purple")

# Or pass a Recording/RecordingSet and let the function do the
# MATLAB-matching (raw_px + 400) / 1000 normalization:
fig = plot_density_2d(rec, mid_color="green", log_alpha=5)
```

Arguments:

- `x_max` (default 0.8) — plot / histogram domain upper limit
- `bin_width` (default 0.01) — grid bin size in data units
- `smooth_sigma` (default 0.05) — Gaussian smoothing width in data units
- `n_levels` (default 7) — number of contour lines
- `mid_color` — `"purple"` or `"green"`
- `log_alpha` — 0 = linear, 2..10 = mild to strong compression

---

## Annotation modes

marm_viz needs three kinds of auxiliary files alongside the
marm_behavior outputs, and none of them are produced by the
marm_behavior pipeline itself:

| File | Scope | What's in it |
|---|---|---|
| `animals_present.txt` | one per **recording folder** | 4 ints (one per colour): 1=present, 0=absent |
| `animals_genotype.txt` | one per **recording folder** | 4 ints: genotype code per colour (lab convention: 0=WT, 1=Shank3 het, 2=Shank3 hom) |
| `animal_ID.txt` | one per **parent folder** (shared across sessions) | 4 ints: numeric animal ID per colour (-1 = unknown) |
| `<stem>_cage_boundaries.mat` | one per **video** | 3×4 `boundaries` matrix: row 0 = cage rect, row 1 = door rect, row 2 = edge x-range |
| `<stem>_center.txt` | one per **video** | 2 floats: stimulus centre ``x`` and ``y`` in pixel coordinates |

The `--annotate` CLI modes let you create all five without leaving
Python.

**NOTE:** `gene_therapy.txt` is intentionally not prompted by marm_viz.
If you want one in the folder, create it manually — marm_viz doesn't
use it.

### `metadata` — text prompts for animals_present / animal_ID / animals_genotype

Runs three sequential console prompts, one file at a time, with
four per-colour sub-prompts each. For every value, the tool shows
the current default in brackets (from an existing file if there is
one, or from a sensible fallback if there isn't) and accepts an
empty line to keep that default.

**CLI:**

```bash
# Prompt for any missing files. Existing files get a confirm dialog
# ("edit? [y/N]") before being touched.
python -m marm_viz /data/session1 --annotate metadata

# Re-edit every file regardless of whether it already exists.
python -m marm_viz /data/session1 --annotate metadata --force

# Write animal_ID.txt into a custom location (default is the parent
# of --folder, matching the MATLAB convention of one ID file per cage).
python -m marm_viz /data/session1 --annotate metadata \
    --id-parent-dir /data
```

**Library:**

```python
from marm_viz.annotate import prompt_metadata

# Interactive: launches console prompts
prompt_metadata("/data/session1")

# Non-interactive: write values directly, no prompts
prompt_metadata("/data/session1", non_interactive={
    "animals_present":  [1, 0, 1, 1],       # Red, White, Blue, Yellow
    "animal_ID":        [247, -1, 115, 238],
    "animals_genotype": [0, 0, 1, 2],
})
```

The `metadata` tool is **pure stdlib** — it works over SSH with no
display and needs none of the optional dependencies.

### `boundaries` — interactive cage annotation

Python port of `find_cage_boundaries.m`. For each .avi in the
folder, opens a matplotlib window showing frame 99 (the MATLAB
reference reads 100 frames in a loop, ending on frame 100) of the
video, cropped to the leftmost 1280 pixels (stereo-RGB half). You
draw three rectangles one at a time by click-dragging, then press
**Enter** to accept each one:

1. Rect around **floor** — the cage floor rectangle. The centre of
   this rectangle becomes the cage-centre reference for all
   coordinate transforms.
2. Rect around **door** — the door rectangle. If the door centre
   ends up below the cage centre after centering, every subsequent
   frame's coordinates get a 180° rotation so "door above" is a
   cross-recording invariant.
3. Rect defining **edge left/right** — only the rectangle's x-range
   matters; it defines the valid x-range for head positions.
   Frames where the head's x falls outside this range are rejected.

The result is written as `<video_stem>_cage_boundaries.mat` with a
`boundaries` variable — a 3×4 matrix, same format as the MATLAB
script's output so both pipelines read it interchangeably.

**CLI:**

```bash
# Annotate every video in the folder. Already-annotated videos
# are skipped.
python -m marm_viz /data/session1 --annotate boundaries

# Re-annotate every video, overwriting existing files.
python -m marm_viz /data/session1 --annotate boundaries --force
```

**Library:**

```python
from marm_viz.annotate import annotate_cage_boundaries, annotate_folder_cages

# One video, interactive
annotate_cage_boundaries("/data/session1/video1.avi")

# One video, scripted (skips the interactive prompt)
annotate_cage_boundaries(
    "/data/session1/video1.avi",
    override_rects=[
        (200, 100, 880, 520),   # floor:       [x, y, w, h]
        (600,  50,  80,  30),   # door:        [x, y, w, h]
        (100,   0, 1080, 720),  # edge x-range [x, y, w, h]
    ],
)

# Whole folder
annotate_folder_cages("/data/session1")
```

Keyboard shortcut inside the matplotlib window: **Esc** aborts the
current rectangle without saving.

**Requires** `opencv-python` for .avi reading. Install with
`pip install 'marm_viz[annotate]'` or `pip install opencv-python`.

### `center` — interactive stimulus-centre annotation

Python port of `find_center_of_stimulus.m`. For each .avi, opens a
matplotlib window showing the first frame and asks you to **click
once** on the stimulus centre. Saves `<video_stem>_center.txt` with
the ``x`` and ``y`` coordinates on separate lines (same format the
MATLAB script writes, readable by both pipelines).

Videos whose filename contains the string `"error"` are skipped
automatically, matching the MATLAB reference behaviour.

**CLI:**

```bash
python -m marm_viz /data/session1 --annotate center
python -m marm_viz /data/session1 --annotate center --force  # re-annotate
```

**Library:**

```python
from marm_viz.annotate import annotate_stim_center, annotate_folder_centers

# One video, interactive
annotate_stim_center("/data/session1/video1.avi")

# One video, scripted
annotate_stim_center(
    "/data/session1/video1.avi",
    override_xy=(640.0, 200.0),
)

# Whole folder
annotate_folder_centers("/data/session1")
```

**Requires** `opencv-python` (same as the boundaries tool).

### `all` — run metadata + boundaries + center in sequence

```bash
python -m marm_viz /data/session1 --annotate all
```

Equivalent to running the three individual modes back-to-back in
the order `metadata → boundaries → center`. Each sub-tool still
respects `--force` and `skip_if_exists` individually — if a file
already exists and you don't pass `--force`, that sub-tool prints
a "skipping" message and moves on.

### Typical workflow for a new set of recordings

After marm_behavior has finished processing your videos and before
you run any `marm_viz --plot` commands:

```bash
cd /data/new_session
python -m marm_viz . --annotate all
```

This walks you through:
1. Metadata prompts (console)
2. Cage boundaries drawer for every .avi (matplotlib windows)
3. Stim-centre click tool for every .avi (matplotlib windows)

Once it's done, every subsequent `--plot` command will find all
the auxiliary files and produce plots with proper cage-centering
and stim-relative fields. If you miss a file, the plot commands
still run — they just fall back to the "no auxiliary file"
behaviour documented in the plot mode sections (frame-centre cage,
NaN dist/angle-to-stim).

---

### Mode selection

| Flag | Default | What it does |
|---|---|---|
| `folder` (positional) | required | Directory with marm_behavior artifacts. The CLI auto-discovers every video via `edges_*.mat` files. |
| `--plot {qc,heatmaps,timeseries,density}` | one of --plot/--annotate required | Which figure type to produce. Mutually exclusive with `--annotate`. |
| `--annotate {metadata,boundaries,center,all}` | one of --plot/--annotate required | Which annotation tool to run. Mutually exclusive with `--plot`. |

### Input / output

| Flag | Default | What it does |
|---|---|---|
| `-o`, `--output-dir PATH` | `./marm_viz_output` | Where PNGs are written. Only used by `--plot`. |
| `--video STEM` | all | Restrict to one video stem (e.g. `--video test_4`). |
| `--type-to-use STR` | all | Only process videos whose stem contains STR (case-insensitive). E.g. `--type-to-use day1` processes only videos with "day1" in their filename. |
| `--color {r,w,b,y}` | all | Restrict to one colour. |
| `--state-remap PATH` | bundled `list_5` | Optional 80-line text file with a custom state reorder lookup. Default loads `marm_viz/data/default_state_remap.txt`. |

### Shared plotting flags

| Flag | Default | What it does |
|---|---|---|
| `--bin-size FLOAT` | `20` | Bin size (px) for the heatmap figures. |
| `--use-body-xy` | off | Use `body_xy` instead of `head_xy` in heatmap / density plots. |
| `--n-frames INT` | `50000` | Random-sample this many frames for the QC figure (0 = plot everything). |

### `timeseries`-specific flags

| Flag | Default | What it does |
|---|---|---|
| `--radius-px FLOAT` | `120` | In-radius cutoff for the proximity fraction. |
| `--chunk-minutes FLOAT` | `5` | Chunk length in minutes. |
| `--fps FLOAT` | `60` | Frames per second (for chunk → frames conversion). |
| `--scatter-sample-rate FLOAT` | `0.01` | Fraction of points to sample for the xy scatter figures. |

### `density`-specific flags

| Flag | Default | What it does |
|---|---|---|
| `--mid-color {purple,green}` | `purple` | Which white → mid → black colormap family. |
| `--log-alpha FLOAT` | `10` | Log tone-map strength. 0 = linear, higher = more compression. |

### Annotate-mode flags

| Flag | Default | What it does |
|---|---|---|
| `--force` | off | Re-annotate files that already exist instead of skipping them. Applies to all three annotate sub-tools. |
| `--id-parent-dir PATH` | parent of `--folder` | Where `animal_ID.txt` should be written by `--annotate metadata`. Default matches the MATLAB convention (one ID file per cage, shared across recording sessions). |

---

## Worked examples

### End-to-end on a typical folder

Given a `/data/exp7/` directory with several videos that have gone
through the marm_behavior pipeline (so each video has `edges_*.mat`,
`depths_*.mat`, `hlabel_*.csv` for each present colour, plus the
`*_cage_boundaries.mat` and `*_center.txt` annotations), produce
every plot type:

```bash
cd /data/exp7
python -m marm_viz . --plot qc         -o ./viz
python -m marm_viz . --plot heatmaps   -o ./viz
python -m marm_viz . --plot timeseries -o ./viz --radius-px 100
python -m marm_viz . --plot density    -o ./viz --mid-color green
```

All outputs land in `/data/exp7/viz/`.

### Per-animal density plots

Run a loop to get one density plot per animal colour:

```bash
for c in r w b y; do
    python -m marm_viz /data/exp7 --plot density --color $c \
        -o /data/exp7/viz --mid-color purple
done
```

Output filenames are `density_2d_all_purple.png`, but the data is
filtered to the single colour via `--color`.

### Folder without cage annotations

If your folder has `edges_*.mat` and `depths_*.mat` but no
`*_cage_boundaries.mat` or `*_center.txt` files, everything still
runs — the loader silently falls back to a frame-centre cage and
skips the dist/angle-to-stim derived fields. The QC figure's panels
2, 3, 6, 7 will be empty (scatter with no points; histogram with
nothing to bin); all the other panels and plot modes work.

If you'd rather inject the cage/stim info programmatically instead
of dropping those files alongside every video, use the library API:

```python
from marm_viz import build_recording
from marm_viz.collect.geometry import CageGeometry
import numpy as np

rec = build_recording(
    "folder", "my_video", "w",
    cage_override=CageGeometry.from_boundaries_matrix(
        np.array([[200, 100, 880, 520],
                  [600, 50, 80, 30],
                  [100, 0, 1080, 720]], dtype=float)
    ),
    stim_xy_pixel_override=np.array([640.0, 200.0]),
)
```

### Regenerating plots after editing source code

Because `pip install -e .` creates an editable install, any changes
to the Python sources are picked up on the next invocation — no
reinstall required. Good workflow during development:

```bash
# Terminal 1: edit files
# Terminal 2: re-run CLI
python -m marm_viz /data/exp7 --plot qc
# ... tweak plot code ...
python -m marm_viz /data/exp7 --plot qc
```

---

## Troubleshooting

### "no recordings built from <folder>"

The folder has no files matching `edges_*.mat`, or all of them failed
to load. Sanity check:

```bash
ls path/to/folder/edges_*.mat
python -c "from scipy.io import loadmat; m = loadmat('path/to/folder/edges_YOURVIDEO.mat'); print(sorted(m.keys()))"
```

You should see variables named `F_3E`/`F_4E`, `F_3RE`/`F_4RE`, etc.
depending on which colours are present.

### All behaviour-category panels are empty

The `hlabel_*.csv` files aren't being found or aren't valid. Check:

- Filenames must match exactly: `hlabel_White_<stem>.csv`,
  `hlabel_Red_<stem>.csv`, etc. (Color capitalised).
- File format: one integer per line, 1..80.
- The cluster IDs need to fall within the remap range (identity
  remap: 1..80; custom remap: whatever length the file is).

### Density plot is mostly white

Your coordinates probably don't fall within `[0, x_max]` (default 0.8)
after the normalization. Two likely causes:

1. You're passing raw cage-centered px coordinates to the function
   without normalization. Solution: let the function do it by
   passing a `Recording` instead of arrays, or scale your arrays
   into `[0, 0.8]` first.
2. Your cage is larger than the ±400 pixel range the default
   normalization assumes. Solution: either widen `x_max` and
   adjust `bin_width` proportionally, or do your own normalization
   and pass arrays.

### QC figure runs out of memory

Reduce `--n-frames` — the default of 50k is conservative but big
concatenations (lots of videos × 4 colours) can still push it. Try
`--n-frames 10000` for a quick look.

### "cannot load edges_*.mat: scipy.io.loadmat failed and h5py is not installed"

Your `.mat` files are saved as v7.3 (HDF5 under the hood), and h5py
isn't installed in your environment. Install it:

```bash
pip install h5py
```

Then re-run. The loader will pick it up automatically.
