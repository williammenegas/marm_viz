"""
Metadata text-prompt tool
=========================

Interactively create (or update) the three auxiliary metadata files
the marm_behavior folder layout expects but which aren't produced by
any pipeline stage:

* ``animals_present.txt`` — 4 integers (0/1 per colour), written in
  the **recording folder**.
* ``animals_genotype.txt`` — 4 integers (genotype code per colour),
  written in the **recording folder**.
* ``animal_ID.txt`` — 4 integers (numeric animal IDs per colour),
  written in the **parent** folder of the recording folder (so a
  single ID list can cover multiple recording sessions from the
  same cage).

The file format is one integer per line. Missing-value convention:
``-1`` for ``animal_ID.txt`` (so the line is still a parseable int);
``0`` means "not present" in ``animals_present.txt``; genotype codes
are free but the lab convention is documented in
:data:`GENOTYPE_CODES`.

The module is pure stdlib — no matplotlib, no opencv — so it can
run on a headless server over SSH. For the visual cage/stim tools,
see :mod:`marm_viz.annotate.cage_boundaries` and
:mod:`marm_viz.annotate.stim_center`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Colour order used throughout marm_behavior (matches the
#: ap4 / animal_ID / animals_genotype row order).
COLORS: Tuple[str, ...] = ("Red", "White", "Blue", "Yellow")

#: Lab-convention genotype codes for ``animals_genotype.txt``.
#: You can use other numbers; these are just hints in the prompt.
GENOTYPE_CODES: Dict[int, str] = {
    0: "WT (wild type)",
    1: "Shank3 heterozygous",
    2: "Shank3 homozygous",
}

#: Per-file defaults used when the user just hits Enter at a prompt.
_DEFAULT_PRESENT = [1, 1, 1, 1]
_DEFAULT_ID = [-1, -1, -1, -1]
_DEFAULT_GENOTYPE = [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _read_four_ints(path: Path) -> Optional[List[int]]:
    """Read 4 whitespace-separated integers from ``path``, or return
    None if the file doesn't exist or can't be parsed.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text()
        tokens = text.replace(",", " ").split()
        values = [int(float(t)) for t in tokens[:4]]
    except Exception:
        return None
    if len(values) < 4:
        return None
    return values[:4]


def _write_four_ints(path: Path, values: List[int]) -> None:
    """Write 4 integers to ``path``, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(v)) for v in values) + "\n")


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

def _prompt_four_ints(
    label: str,
    defaults: List[int],
    *,
    valid_range: "Tuple[int, int] | None" = None,
    hint: "str | None" = None,
) -> List[int]:
    """Prompt the user for 4 integers with per-colour sub-prompts.

    Shows the current default for each colour and accepts an empty
    line to keep that default. If ``valid_range`` is given, values
    outside the range are re-prompted.
    """
    print(f"\n{label}")
    if hint:
        print(f"  ({hint})")
    out: List[int] = list(defaults)
    for i, color in enumerate(COLORS):
        while True:
            prompt = f"  {color:<7} [{out[i]}]: "
            try:
                raw = input(prompt).strip()
            except EOFError:
                raw = ""
            if raw == "":
                break
            try:
                v = int(raw)
            except ValueError:
                print(f"    not an integer: {raw!r}; try again")
                continue
            if valid_range is not None:
                lo, hi = valid_range
                if not (lo <= v <= hi):
                    print(f"    out of range [{lo}, {hi}]: {v}; try again")
                    continue
            out[i] = v
            break
    return out


def _confirm_overwrite(path: Path, existing: List[int]) -> bool:
    """If ``path`` exists, show the existing values and ask whether
    to overwrite. Returns True if the user wants to edit, False if
    they want to keep the file as-is.
    """
    values_str = ", ".join(f"{c}={v}" for c, v in zip(COLORS, existing))
    print(f"\n  {path.name} already exists: {values_str}")
    try:
        raw = input("  edit? [y/N]: ").strip().lower()
    except EOFError:
        raw = ""
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prompt_metadata(
    rec_dir: "str | Path",
    *,
    id_parent_dir: "str | Path | None" = None,
    force: bool = False,
    non_interactive: "Dict[str, List[int]] | None" = None,
) -> Dict[str, Path]:
    """Create or update the three metadata files for one recording folder.

    Parameters
    ----------
    rec_dir:
        Recording folder (where ``edges_*.mat`` etc. live).
        ``animals_present.txt`` and ``animals_genotype.txt`` are
        written here.
    id_parent_dir:
        Where ``animal_ID.txt`` is written. Defaults to ``rec_dir``'s
        parent (matching the MATLAB convention). Pass ``rec_dir``
        explicitly if you want the ID file alongside the others.
    force:
        If True, always re-prompt even for files that already exist.
        If False (default), ask the user whether to edit existing
        files.
    non_interactive:
        If given, skip every prompt and write the provided values
        directly. Keys: ``"animals_present"``, ``"animal_ID"``,
        ``"animals_genotype"``. Each value must be a list of 4 ints.
        Used by unit tests and by callers that already know the
        metadata.

    Returns
    -------
    dict
        ``{"animals_present": Path, "animal_ID": Path,
        "animals_genotype": Path}`` pointing at the (possibly newly
        created) files.
    """
    rec_dir = Path(rec_dir)
    if not rec_dir.exists():
        raise FileNotFoundError(f"recording folder does not exist: {rec_dir}")

    if id_parent_dir is None:
        id_parent_dir = rec_dir.parent
    else:
        id_parent_dir = Path(id_parent_dir)

    paths = {
        "animals_present": rec_dir / "animals_present.txt",
        "animal_ID": Path(id_parent_dir) / "animal_ID.txt",
        "animals_genotype": rec_dir / "animals_genotype.txt",
    }

    if non_interactive is not None:
        for key, values in non_interactive.items():
            if key not in paths:
                raise ValueError(f"unknown metadata key: {key!r}")
            if len(values) != 4:
                raise ValueError(
                    f"{key}: need 4 values (one per colour), got {len(values)}"
                )
            _write_four_ints(paths[key], list(values))
        return paths

    print(f"\n[marm_viz.annotate.metadata] recording folder: {rec_dir}")
    print(f"[marm_viz.annotate.metadata] ID parent folder:  {id_parent_dir}")
    print(f"[marm_viz.annotate.metadata] colour order:      {', '.join(COLORS)}")

    # --- animals_present ---
    existing = _read_four_ints(paths["animals_present"])
    if existing is not None and not force:
        if _confirm_overwrite(paths["animals_present"], existing):
            values = _prompt_four_ints(
                "animals_present.txt  (1 = present, 0 = not in this video)",
                existing,
                valid_range=(0, 1),
            )
            _write_four_ints(paths["animals_present"], values)
            print(f"  wrote {paths['animals_present']}")
        else:
            print("  kept existing file")
    else:
        defaults = existing if existing is not None else _DEFAULT_PRESENT
        values = _prompt_four_ints(
            "animals_present.txt  (1 = present, 0 = not in this video)",
            defaults,
            valid_range=(0, 1),
        )
        _write_four_ints(paths["animals_present"], values)
        print(f"  wrote {paths['animals_present']}")

    # --- animal_ID (in parent dir) ---
    existing = _read_four_ints(paths["animal_ID"])
    if existing is not None and not force:
        if _confirm_overwrite(paths["animal_ID"], existing):
            values = _prompt_four_ints(
                "animal_ID.txt  (integer ID per colour; -1 for unknown)",
                existing,
                hint="shared across recording sessions for the same cage",
            )
            _write_four_ints(paths["animal_ID"], values)
            print(f"  wrote {paths['animal_ID']}")
        else:
            print("  kept existing file")
    else:
        defaults = existing if existing is not None else _DEFAULT_ID
        values = _prompt_four_ints(
            "animal_ID.txt  (integer ID per colour; -1 for unknown)",
            defaults,
            hint="shared across recording sessions for the same cage",
        )
        _write_four_ints(paths["animal_ID"], values)
        print(f"  wrote {paths['animal_ID']}")

    # --- animals_genotype ---
    existing = _read_four_ints(paths["animals_genotype"])
    hint_parts = [f"{k}={v}" for k, v in sorted(GENOTYPE_CODES.items())]
    hint = "lab convention: " + "; ".join(hint_parts)
    if existing is not None and not force:
        if _confirm_overwrite(paths["animals_genotype"], existing):
            values = _prompt_four_ints(
                "animals_genotype.txt  (integer genotype code per colour)",
                existing,
                hint=hint,
            )
            _write_four_ints(paths["animals_genotype"], values)
            print(f"  wrote {paths['animals_genotype']}")
        else:
            print("  kept existing file")
    else:
        defaults = existing if existing is not None else _DEFAULT_GENOTYPE
        values = _prompt_four_ints(
            "animals_genotype.txt  (integer genotype code per colour)",
            defaults,
            hint=hint,
        )
        _write_four_ints(paths["animals_genotype"], values)
        print(f"  wrote {paths['animals_genotype']}")

    print("\n[marm_viz.annotate.metadata] done.")
    return paths
