"""
Command-line entry point
========================

Example usage::

    # Plot modes (write PNGs to --output-dir)
    python -m marm_viz /path/to/folder --plot qc
    python -m marm_viz /path/to/folder --plot heatmaps
    python -m marm_viz /path/to/folder --plot timeseries --radius-px 120
    python -m marm_viz /path/to/folder --plot density --mid-color green

    # Restrict to a single video or colour
    python -m marm_viz /path/to/folder --plot qc --video test_4 --color w

    # Annotation modes (interactive; create the auxiliary files the
    # collect stage needs but marm_behavior doesn't produce)
    python -m marm_viz /path/to/folder --annotate metadata
    python -m marm_viz /path/to/folder --annotate boundaries
    python -m marm_viz /path/to/folder --annotate center
    python -m marm_viz /path/to/folder --annotate all    # all three in sequence

The default output directory is ``./marm_viz_output`` next to the CWD;
override with ``--output-dir``. Plot modes always save to PNG files;
annotate modes open interactive matplotlib windows (``boundaries`` /
``center``) or console prompts (``metadata``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# NOTE: we delay importing matplotlib and switching its backend
# until we know whether the user picked a plot mode (needs Agg, no
# display) or an annotate mode (needs an interactive backend).

from .collect.video_data import build_recording, build_recording_set
from .types import RecordingSet


_PLOT_CHOICES = (
    "qc", "heatmaps", "timeseries", "density",
    "behavior", "ethogram", "transitions", "polar", "track",
    "correlation",
)
_ANNOTATE_CHOICES = ("metadata", "boundaries", "center", "all")


def _save_and_close(fig, path: Path, dpi: int = 200, fmt: str = "jpg") -> Path:
    """Save ``fig``, emit source CSVs, print, close. Returns actual path."""
    import matplotlib.pyplot as plt
    from .plot.sources import save_sources_to_csv

    # Override extension with the user's chosen format.
    path = path.with_suffix(f".{fmt}")
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    print(f"[marm_viz] wrote {path}")
    written = save_sources_to_csv(fig, path.with_suffix(""))
    for p in written:
        print(f"[marm_viz]   source -> {p.name}")
    plt.close(fig)
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m marm_viz",
        description=(
            "Generate visualizations from a folder of marm_behavior "
            "outputs."
        ),
    )
    p.add_argument(
        "folder",
        type=Path,
        help="Directory containing marm_behavior artifacts "
        "(edges_*.mat, depths_*.mat, hlabel_*.csv, etc.), or a "
        "specific .avi file (in which case only that video's data "
        "is used).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed diagnostic information during processing.",
    )

    # Mode selection: --plot / --annotate are optional. If neither is
    # given, defaults to generating all plot types.
    mode_group = p.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--plot",
        choices=_PLOT_CHOICES,
        default=None,
        help="Which plot type to produce (writes PNGs to --output-dir). "
        "If neither --plot nor --annotate is given, all plot types "
        "are generated.",
    )
    mode_group.add_argument(
        "--annotate",
        choices=_ANNOTATE_CHOICES,
        help="Interactively create auxiliary metadata files. "
        "'metadata' runs console prompts for animals_present / "
        "animal_ID / animals_genotype; 'boundaries' opens a matplotlib "
        "window per video for cage-rectangle annotation; 'center' "
        "opens a window per video for one-click stim-centre "
        "annotation; 'all' runs metadata + boundaries + center.",
    )

    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for saved files. Default: "
        "<folder>/marm_viz_output (inside the input folder). "
        "Created if missing. Only used by --plot.",
    )
    p.add_argument(
        "--format",
        choices=("jpg", "svg", "png", "pdf"),
        default="jpg",
        help="Output file format. 'jpg' (default) produces raster images. 'svg' "
        "graphics editable in Illustrator. 'pdf' is also vector. "
        "'png' is raster.",
    )

    # Recording selection
    p.add_argument(
        "--video",
        default=None,
        help="Process only this video stem (e.g. 'test_4'). Default: "
        "all stems found in the folder.",
    )
    p.add_argument(
        "--color",
        choices=("r", "w", "b", "y"),
        default=None,
        help="Process only this colour. Default: all four.",
    )
    p.add_argument(
        "--state-remap",
        type=Path,
        default=None,
        help="Optional path to a state-reorder text file (one int "
        "per line, 80 lines). Default: the bundled list_5 "
        "permutation from ts_db_list.mat.",
    )
    p.add_argument(
        "--time-to-use", "--time_to_use",
        type=float,
        default=None,
        help="Limit each recording to this many minutes of data. "
        "Default: use all available data.",
    )
    p.add_argument(
        "--animal-to-use", "--animal_to_use",
        choices=("Red", "White", "Blue", "Yellow",
                 "red", "white", "blue", "yellow",
                 "r", "w", "b", "y"),
        default=None,
        help="Only include data from this animal colour. "
        "Default: all animals.",
    )
    p.add_argument(
        "--type-to-use", "--type_to_use",
        default=None,
        help="Only include videos whose stem contains this string. "
        "E.g. --type-to-use day1 will only process videos "
        "with 'day1' in their filename.",
    )

    # Shared plotting flags
    p.add_argument(
        "--bin-size",
        type=float,
        default=20.0,
        help="Bin size (pixels) for heatmap plots. Default: 20.",
    )
    p.add_argument(
        "--use-body-xy",
        action="store_true",
        help="Use body_xy instead of head_xy in heatmap / density "
        "plots.",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Apply left/right flip augmentation before plotting. "
        "Doubles the data by adding an x-mirrored copy of every "
        "recording, which removes left/right spatial bias from "
        "heatmaps and density plots. Off by default.",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=50_000,
        help="Random-sample this many frames for the QC figure "
        "(use 0 to plot everything). Default: 50000.",
    )

    # Stim-proximity flags (used by --plot timeseries and also by the
    # 6th "near-stimulus" panel in --plot heatmaps).
    p.add_argument(
        "--radius-px",
        type=float,
        default=120.0,
        help="Radius (px) for stim-proximity calculations (time "
        "series + near-stim panel in heatmaps). Default: 120.",
    )
    p.add_argument(
        "--chunk-minutes",
        type=float,
        default=5.0,
        help="Chunk length in minutes for the time series. Default: 5.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Frames per second for the time series. Default: 60.",
    )
    p.add_argument(
        "--scatter-sample-rate",
        type=float,
        default=0.01,
        help="Fraction of points to sample for the scatter subplots. "
        "Default: 0.01 (1%%).",
    )

    # Density flags
    p.add_argument(
        "--mid-color",
        choices=("purple", "green"),
        default="purple",
        help="Colormap middle colour for density plots. Default: purple.",
    )
    p.add_argument(
        "--log-alpha",
        type=float,
        default=10.0,
        help="Log tone-map aggressiveness for density plots. 0=linear. "
        "Default: 10.",
    )

    # Annotate-mode flags
    p.add_argument(
        "--force",
        action="store_true",
        help="For --annotate modes: re-annotate files that already "
        "exist instead of skipping them.",
    )
    p.add_argument(
        "--id-parent-dir",
        type=Path,
        default=None,
        help="For --annotate metadata: directory where animal_ID.txt "
        "should live. Defaults to the parent of --folder (matching "
        "the MATLAB convention).",
    )

    return p


def _run_plot_mode(args, plot_types: "list[str]") -> int:
    """Handle --plot modes. Imports matplotlib with the Agg backend
    so this runs on headless machines.
    """
    # Headless backend: we only save PNGs.
    import matplotlib
    matplotlib.use("Agg")

    from .plot.category_heatmaps import (
        plot_category_fraction_heatmaps,
        plot_stay_move_heatmaps,
        plot_xy_occupancy_heatmap,
    )
    from .plot.density_2d import plot_density_2d
    from .plot.qc_panels import plot_qc_panels
    from .plot.stim_time_series import plot_stim_proximity_time_series
    from .plot.behavior_overview import plot_state_usage, plot_tsne_centroids
    from .plot.ethogram import plot_ethogram
    from .plot.behavior_detail import (
        plot_transition_matrix,
        plot_head_direction_polar,
    )
    from .plot.head_track import plot_head_track
    from .plot.correlation import plot_cagemate_correlation

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = None if args.n_frames <= 0 else args.n_frames

    # Resolve --time-to-use to max_frames.
    build_kw: dict = {"state_remap": args.state_remap}
    if args.time_to_use is not None:
        build_kw["max_frames"] = int(round(args.time_to_use * args.fps * 60))
        print(f"[marm_viz] Limiting each recording to {args.time_to_use:g} min "
              f"({build_kw['max_frames']:,} frames)")

    # Resolve --animal-to-use to a single-colour filter.
    _NAME_TO_KEY = {
        "red": "r", "white": "w", "blue": "b", "yellow": "y",
        "r": "r", "w": "w", "b": "b", "y": "y",
    }
    animal_color = None
    if args.animal_to_use is not None:
        animal_color = _NAME_TO_KEY[args.animal_to_use.lower()]
        print(f"[marm_viz] Filtering to animal: "
              f"{args.animal_to_use.title()}")

    if args.video is not None and args.color is not None:
        rec = build_recording(
            args.folder, args.video, args.color,
            **build_kw,
        )
        if rec is None:
            print(
                f"error: could not build recording for "
                f"{args.video} / {args.color}",
                file=sys.stderr,
            )
            return 3
        data = rec
        stem = f"{args.video}_{args.color}"
    else:
        # Determine colour filter: --animal-to-use overrides --color.
        if animal_color is not None:
            colors = (animal_color,)
        elif args.color is not None:
            colors = (args.color,)
        else:
            colors = None
        stems = (args.video,) if args.video else None

        # Filter stems by --type-to-use if specified.
        if args.type_to_use is not None and stems is None:
            from .io.marm_behavior_loader import discover_video_stems
            all_stems = discover_video_stems(args.folder)
            stems = [s for s in all_stems if args.type_to_use.lower() in s.lower()]
            if not stems:
                print(f"error: no videos matching '{args.type_to_use}' "
                      f"in {args.folder}", file=sys.stderr)
                return 3
            print(f"[marm_viz] Filtered to {len(stems)} video(s) "
                  f"matching '{args.type_to_use}'")

        rs = build_recording_set(
            args.folder,
            colors=colors,
            video_stems=stems,
            **build_kw,
        )
        if len(rs) == 0:
            print(f"error: no recordings built from {args.folder}", file=sys.stderr)
            return 3
        data = rs
        stem = args.video or "all"

    # Optionally flip-augment the data (doubles every recording with
    # an x-mirrored copy). Off by default; enable with --augment.
    if args.augment and isinstance(data, RecordingSet):
        data = data.augment_flip_x()
        print("[marm_viz] applied left/right flip augmentation")

    n_recs = 1 if (args.video and args.color) else len(data)
    print(f"[marm_viz] loaded {n_recs} recording(s) from {args.folder}")
    print(f"[marm_viz] plot types = {plot_types}")
    print(f"[marm_viz] output dir = {args.output_dir}")

    for plot_type in plot_types:
        print(f"[marm_viz] Generating: {plot_type} ...")
        if plot_type == "qc":
            fig = plot_qc_panels(data, n_frames=n_frames)
            out = args.output_dir / f"qc_panels_{stem}.png"
            _save_and_close(fig, out, fmt=args.format)

        elif plot_type == "heatmaps":
            fig1 = plot_category_fraction_heatmaps(
                data,
                bin_size=args.bin_size,
                use_head_xy=not args.use_body_xy,
                stim_radius_px=args.radius_px,
            )
            out1 = args.output_dir / f"category_heatmaps_{stem}.png"
            _save_and_close(fig1, out1, fmt=args.format)

            fig2 = plot_xy_occupancy_heatmap(
                data, bin_size=args.bin_size, use_head_xy=not args.use_body_xy,
            )
            out2 = args.output_dir / f"xy_occupancy_{stem}.png"
            _save_and_close(fig2, out2, fmt=args.format)

            fig3 = plot_stay_move_heatmaps(
                data, bin_size=args.bin_size, use_head_xy=not args.use_body_xy,
            )
            out3 = args.output_dir / f"stay_move_{stem}.png"
            _save_and_close(fig3, out3, fmt=args.format)

        elif plot_type == "timeseries":
            f1, f2, f3 = plot_stim_proximity_time_series(
                data,
                radius_px=args.radius_px,
                chunk_minutes=args.chunk_minutes,
                fps=args.fps,
                scatter_sample_rate=args.scatter_sample_rate,
            )
            out1 = args.output_dir / f"stim_fraction_{stem}.png"
            out2 = args.output_dir / f"stim_chunk_scatters_{stem}.png"
            out3 = args.output_dir / f"stim_inside_outside_{stem}.png"
            _save_and_close(f1, out1, fmt=args.format)
            _save_and_close(f2, out2, fmt=args.format)
            _save_and_close(f3, out3, fmt=args.format)

        elif plot_type == "density":
            fig = plot_density_2d(
                data,
                use_body_xy=args.use_body_xy,
                mid_color=args.mid_color,
                log_alpha=args.log_alpha,
            )
            out = args.output_dir / f"density_2d_{stem}_{args.mid_color}.png"
            _save_and_close(fig, out, fmt=args.format)

        elif plot_type == "behavior":
            fig1 = plot_state_usage(data)
            out1 = args.output_dir / f"state_usage_{stem}.png"
            _save_and_close(fig1, out1, fmt=args.format)

            fig2 = plot_tsne_centroids(data)
            out2 = args.output_dir / f"tsne_centroids_{stem}.png"
            _save_and_close(fig2, out2, fmt=args.format)

        elif plot_type == "ethogram":
            fig = plot_ethogram(data, fps=args.fps)
            out = args.output_dir / f"ethogram_{stem}.png"
            _save_and_close(fig, out, fmt=args.format)

        elif plot_type == "transitions":
            fig = plot_transition_matrix(data)
            out = args.output_dir / f"transition_matrix_{stem}.png"
            _save_and_close(fig, out, fmt=args.format)

        elif plot_type == "polar":
            fig = plot_head_direction_polar(
                data, radius_px=args.radius_px,
            )
            out = args.output_dir / f"head_direction_polar_{stem}.png"
            _save_and_close(fig, out, fmt=args.format)

        elif plot_type == "track":
            fig_ind, fig_merged = plot_head_track(data, fps=args.fps)
            out1 = args.output_dir / f"head_track_individual_{stem}.png"
            _save_and_close(fig_ind, out1, fmt=args.format)
            out2 = args.output_dir / f"head_track_merged_{stem}.png"
            _save_and_close(fig_merged, out2, fmt=args.format)

        elif plot_type == "correlation":
            fig = plot_cagemate_correlation(
                data,
                chunk_minutes=args.chunk_minutes,
                fps=args.fps,
            )
            out = args.output_dir / f"correlation_{stem}.png"
            _save_and_close(fig, out, fmt=args.format)

    return 0


def _run_annotate_mode(args) -> int:
    """Handle --annotate modes. Does NOT force a matplotlib backend
    so the default interactive backend is used.
    """
    from .annotate.metadata import prompt_metadata
    from .annotate.cage_boundaries import annotate_folder_cages
    from .annotate.stim_center import annotate_folder_centers

    print(f"[marm_viz] annotate mode = {args.annotate!r}")
    print(f"[marm_viz] folder = {args.folder}")

    want_metadata = args.annotate in ("metadata", "all")
    want_bounds = args.annotate in ("boundaries", "all")
    want_center = args.annotate in ("center", "all")

    if want_metadata:
        prompt_metadata(
            args.folder,
            id_parent_dir=args.id_parent_dir,
            force=args.force,
        )

    if want_bounds:
        annotate_folder_cages(
            args.folder,
            skip_if_exists=not args.force,
        )

    if want_center:
        annotate_folder_centers(
            args.folder,
            skip_if_exists=not args.force,
        )

    return 0


def _check_missing_annotations(folder: Path) -> None:
    """Scan for AVIs missing cage boundary or stimulus centre files.
    Warn the user and offer to annotate interactively.
    """
    avis = sorted(folder.glob("*.avi"))
    if not avis:
        avis = sorted(folder.glob("*.AVI"))
    if not avis:
        return  # no videos to check

    missing_bounds = []
    missing_center = []
    for avi in avis:
        stem = avi.stem
        if not (folder / f"{stem}_cage_boundaries.mat").exists():
            missing_bounds.append(avi)
        if not (folder / f"{stem}_center.txt").exists():
            missing_center.append(avi)

    if not missing_bounds and not missing_center:
        return  # all files present

    print()
    if missing_bounds:
        print(f"[marm_viz] WARNING: {len(missing_bounds)} video(s) "
              f"missing cage boundary annotations:")
        for v in missing_bounds[:5]:
            print(f"           - {v.name}")
        if len(missing_bounds) > 5:
            print(f"           ... and {len(missing_bounds) - 5} more")

    if missing_center:
        print(f"[marm_viz] WARNING: {len(missing_center)} video(s) "
              f"missing stimulus centre annotations:")
        for v in missing_center[:5]:
            print(f"           - {v.name}")
        if len(missing_center) > 5:
            print(f"           ... and {len(missing_center) - 5} more")

    print()
    print("[marm_viz] Without these files, spatial plots (heatmaps, "
          "stim proximity, polar) may be inaccurate or empty.")
    print("[marm_viz] Options:")
    print("           [a] Annotate missing files now (interactive)")
    print("           [b] Annotate only boundaries now")
    print("           [c] Annotate only stimulus centres now")
    print("           [s] Skip and continue plotting anyway")
    print()

    try:
        choice = input("[marm_viz] Your choice [a/b/c/s]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "s"

    if choice in ("a", "b"):
        from .annotate.cage_boundaries import annotate_folder_cages
        print()
        annotate_folder_cages(folder, skip_if_exists=True)

    if choice in ("a", "c"):
        from .annotate.stim_center import annotate_folder_centers
        print()
        annotate_folder_centers(folder, skip_if_exists=True)

    if choice == "s" or choice not in ("a", "b", "c"):
        print("[marm_viz] Continuing without annotations...")
    print()


def main(argv: "list[str] | None" = None) -> int:
    import logging
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # If the user pointed at a specific video file rather than a
    # folder, use the file's parent as the data folder and its stem
    # as the --video filter so only that video's data is plotted.
    if args.folder.is_file():
        args.video = args.video or args.folder.stem
        args.folder = args.folder.parent

    if not args.folder.is_dir():
        print(f"error: path does not exist: {args.folder}", file=sys.stderr)
        return 2

    # Resolve output dir: default is <folder>/marm_viz_output.
    if args.output_dir is None:
        args.output_dir = args.folder / "marm_viz_output"

    if args.annotate is not None:
        return _run_annotate_mode(args)

    # Check for missing annotation files before plotting.
    _check_missing_annotations(args.folder)

    # Determine which plot types to run. If --plot was given, run
    # just that one; if neither --plot nor --annotate was given,
    # run all plot types.
    if args.plot is not None:
        plot_types = [args.plot]
    else:
        plot_types = list(_PLOT_CHOICES)

    return _run_plot_mode(args, plot_types)


if __name__ == "__main__":
    raise SystemExit(main())
