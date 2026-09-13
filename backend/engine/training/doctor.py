"""Check the setup and say what to run next.

    python -m engine.training.doctor
    python -m engine.training.doctor --pig-dir path/to/PianoFingeringDataset_v1.2

Verifies the install, sanity-checks the engine against fingerings every piano
method agrees on, inspects the PIG folder if one is given, estimates how long
the benchmark will take, and prints the exact next command.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

OK = "  ok   "
BAD = " FAIL  "
WARN = " note  "

REQUIRED = [("numpy", "numpy"), ("music21", "music21")]
OPTIONAL = [("pytest", "pytest")]


def _line(status: str, text: str) -> None:
    print(f"[{status}] {text}")


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    good = (major, minor) >= (3, 9)
    _line(OK if good else BAD, f"Python {major}.{minor}" + ("" if good else ", need 3.9 or newer"))
    return good


def check_packages() -> bool:
    everything_ok = True
    for label, module in REQUIRED:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            _line(OK, f"{label} {version}")
        except ImportError:
            _line(BAD, f"{label} is missing. Run: pip install -r requirements.txt")
            everything_ok = False
    for label, module in OPTIONAL:
        try:
            importlib.import_module(module)
            _line(OK, f"{label} (optional)")
        except ImportError:
            _line(WARN, f"{label} not installed, so you cannot run the test suite")
    return everything_ok


def check_engine() -> bool:
    try:
        from ..fingering import fingers_for_pitches
    except Exception as exc:  # noqa: BLE001
        _line(BAD, f"cannot import the engine: {exc}")
        _line(WARN, "run this from the backend/ directory")
        return False

    cases: List[Tuple[str, list, str, list]] = [
        ("C major scale, right hand", [60, 62, 64, 65, 67, 69, 71, 72], "right", [1, 2, 3, 1, 2, 3, 4, 5]),
        ("C major scale, left hand", [60, 62, 64, 65, 67, 69, 71, 72], "left", [5, 4, 3, 2, 1, 3, 2, 1]),
        ("C major arpeggio, right hand", [60, 64, 67, 72], "right", [1, 2, 3, 5]),
    ]
    everything_ok = True
    for label, pitches, hand, expected in cases:
        try:
            got = fingers_for_pitches(pitches, hand)
        except Exception as exc:  # noqa: BLE001
            _line(BAD, f"{label}: {exc}")
            everything_ok = False
            continue
        if got == expected:
            _line(OK, f"{label}: {''.join(map(str, got))}")
        else:
            _line(BAD, f"{label}: got {''.join(map(str, got))}, expected {''.join(map(str, expected))}")
            everything_ok = False
    return everything_ok


def check_pig(pig_dir: Path) -> Optional[Path]:
    if not pig_dir.exists():
        _line(BAD, f"{pig_dir} does not exist")
        return None

    try:
        from .pig import find_fingering_files, is_macos_junk, load_pig_dir
    except Exception as exc:  # noqa: BLE001
        _line(BAD, f"cannot import the PIG loader: {exc}")
        return None

    all_files = sorted(pig_dir.rglob("*_fingering.txt"))
    files = [p for p in all_files if not is_macos_junk(p)]
    skipped = len(all_files) - len(files)
    if skipped:
        _line(WARN, f"ignored {skipped} macOS metadata files (the __MACOSX folder inside the zip)")
    if not files:
        _line(BAD, f"no usable *_fingering.txt files anywhere under {pig_dir}")
        _line(WARN, "point --pig-dir at the folder you unzipped, usually PianoFingeringDataset_v1.2")
        return None

    # A release may hold several folders of fingering files. Use the fullest.
    by_folder: dict = {}
    for path in files:
        by_folder.setdefault(path.parent, []).append(path)
    folder = max(by_folder, key=lambda key: len(by_folder[key]))
    if len(by_folder) > 1:
        _line(WARN, f"{len(by_folder)} folders contain fingering files; using the fullest")
    _line(OK, f"{len(by_folder[folder])} fingering files found in {folder}")

    started = time.time()
    sequences = load_pig_dir(folder)
    elapsed = time.time() - started
    if not sequences:
        _line(BAD, "the files were found but none of them parsed")
        _line(WARN, f"check the first few lines of {files[0].name} against the format in pig.py")
        return None

    pieces = {s.piece for s in sequences}
    annotators = {s.annotator for s in sequences}
    notes = sum(len(s.hand_data) for s in sequences)
    right = sum(1 for s in sequences if s.hand == "right")
    left = sum(1 for s in sequences if s.hand == "left")
    _line(OK, f"parsed in {elapsed:.1f}s: {len(pieces)} pieces, {len(annotators)} annotators, "
              f"{len(sequences)} hand sequences ({right} right, {left} left), {notes} notes")

    fingers = {f for s in sequences for f in s.fingers}
    if not fingers <= {1, 2, 3, 4, 5}:
        _line(BAD, f"unexpected finger numbers: {sorted(fingers)}")
    else:
        _line(OK, "all finger numbers are in 1 to 5")

    if len(pieces) < 50:
        _line(WARN, f"only {len(pieces)} pieces. The full PIG release has 150, "
                    "so you may have pointed at a subfolder")

    # Rough runtime estimate. Feature preparation runs at roughly 400 notes
    # per second per worker on a typical laptop and is cached afterwards.
    seconds = notes / 400.0
    _line(WARN, f"first benchmark run will spend about {seconds/60:.0f} min preparing features "
                f"with 1 worker, or {seconds/60/4:.0f} min with --workers 4. "
                "It is cached, so later runs skip it.")
    return folder


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", type=Path, default=None,
                        help="Folder you unzipped PIG into (the check searches it recursively)")
    args = parser.parse_args(argv)

    print("FingerFlow setup check\n")
    print("Install")
    healthy = check_python()
    healthy = check_packages() and healthy
    print("\nEngine")
    healthy = check_engine() and healthy

    folder = None
    if args.pig_dir:
        print("\nPIG dataset")
        folder = check_pig(args.pig_dir)
    else:
        print("\nPIG dataset")
        _line(WARN, "not checked. Re-run with --pig-dir once you have downloaded it")

    print("\n" + "=" * 62)
    if not healthy:
        print("Fix the failures above before going further.")
        return 1
    if folder is None:
        print("The engine works. Next: register and download PIG from")
        print("  https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/register.php")
        print("unzip it, then run:")
        print("  python -m engine.training.doctor --pig-dir path/to/PianoFingeringDataset_v1.2")
        return 0

    # One line, no continuation character: a backslash continues a command in
    # bash but is a syntax error in PowerShell, where the equivalent is a
    # backtick. A single line is correct in every shell.
    print("Everything is in place. Run the benchmark (all on one line):\n")
    print(f"  python -m engine.training.benchmark --pig-dir \"{folder}\""
          " --cache .feature_cache --workers 4 --save-weights weights_learned.json")
    print("\nIt prints your match rates next to the published ones, and runs the timing ablation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
