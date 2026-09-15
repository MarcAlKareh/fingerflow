"""Rank whole fingerings for a passage, not one note at a time.

``explain_passage.py`` answers "what else could finger 10 be?". That is the
wrong question when a *shape* feels wrong, because changing one note of an
awkward group usually just moves the awkwardness. This answers "what else
could the whole group be?" by constraining a span of notes to a fingering and
re-optimising everything around it, so each candidate is a genuine complete
solution and the costs are comparable.

    # every fingering of notes 9 to 13, ranked
    python scripts/alternatives.py --musicxml beethoven.mxl --span-cm 23 \\
        --segment 9-13 --top 8 --weights backend/weights_learned.json

    # price one fingering you already have in your fingers
    python scripts/alternatives.py --notes "C6 D6 C6 A#5 G#5" --span-cm 23 \\
        --force 1:3,2:4,3:3,4:2,5:1

A forced finger that no assignment can supply (a held note already owns it)
is reported as impossible rather than silently ignored.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engine.passage import (  # noqa: E402
    CHORD, PassageSpec, format_fingering, parse_segment,
)
from engine.weights import Weights  # noqa: E402


def parse_force(text: str, n: int) -> Dict[int, int]:
    fixed: Dict[int, int] = {}
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        index, finger = item.split(":")
        k = int(index) - 1
        if not (0 <= k < n):
            raise SystemExit(f"--force index {index} is outside 1-{n}")
        fixed[k] = int(finger)
    return fixed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--notes")
    source.add_argument("--musicxml", type=Path)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--bpm", type=float, default=60.0)
    parser.add_argument("--note-value", type=int, default=8)
    parser.add_argument("--goal", default=None, choices=[None, "expression", "speed"])
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--segment", default=None, help="Note range to re-finger, e.g. 9-13")
    parser.add_argument("--force", default=None, help="Fingering to price, e.g. 9:3,10:4,11:3")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args(argv)
    raw = argv if argv is not None else sys.argv[1:]
    bpm_given = any(a.startswith("--bpm") for a in raw)

    try:
        if args.musicxml:
            spec = PassageSpec.from_musicxml(args.musicxml, hand=args.hand, span_cm=args.span_cm,
                                             tempo_bpm=args.bpm if bpm_given else None)
        else:
            spec = PassageSpec.from_note_names(args.notes, hand=args.hand, span_cm=args.span_cm,
                                               bpm=args.bpm, note_value=args.note_value)
    except ValueError as exc:
        raise SystemExit(str(exc))

    passage = spec.build()
    weights = (Weights.load(args.weights).with_preset(args.goal) if args.weights
               else Weights.default(args.goal))

    best_path, base_cost = passage.best(weights)
    base_fingers = passage.fingers(best_path)
    print(f"{len(passage)} events, {args.hand} hand, span {args.span_cm:g} cm, "
          f"weights: {args.weights or 'built-in defaults'}")
    print(f"engine's choice  {' '.join(str(f) for f in base_fingers)}   cost {base_cost:.2f}\n")

    if args.force:
        fixed = parse_force(args.force, len(passage))
        got = passage.constrained(weights, fixed)
        shown = " ".join(f"{k+1}:{f}" for k, f in sorted(fixed.items()))
        if got is None:
            print(f"forcing {shown} is impossible here.")
        else:
            path, cost = got
            print(f"forcing {shown}")
            print(f"  {' '.join(str(f) for f in passage.fingers(path))}   "
                  f"cost {cost:.2f}  ({cost - base_cost:+.2f} against the engine's choice)")
        if not args.segment:
            return 0
        print()

    if not args.segment:
        print("Give --segment a-b to rank every fingering of a group of notes.")
        return 0

    try:
        lo, hi = parse_segment(args.segment, len(passage))
    except ValueError as exc:
        raise SystemExit(str(exc))
    span = list(range(lo, hi + 1))
    options = [passage.options(k) for k in span]
    names = passage.names
    print(f"every fingering of notes {lo+1}-{hi+1} "
          f"({' '.join(names[lo:hi+1])}), with the rest re-optimised")
    print(f"{'fingering':<16s}{'cost':>8s}{'extra':>8s}   whole passage")
    print("-" * 78)

    found: List[Tuple[float, Tuple[int, ...], List[int]]] = []
    seen = set()
    for combo in itertools.product(*options):
        got = passage.constrained(weights, dict(zip(span, combo)))
        if got is None:
            continue
        path, cost = got
        whole = tuple(passage.fingers(path))
        if CHORD in combo or whole in seen:
            continue
        seen.add(whole)
        found.append((cost, combo, list(whole)))

    found.sort(key=lambda row: row[0])
    for cost, combo, whole in found[: args.top]:
        mark = "  <- engine" if list(combo) == base_fingers[lo:hi + 1] else ""
        print(f"{format_fingering(combo):<16s}{cost:>8.2f}{cost - base_cost:>+8.2f}   "
              f"{' '.join(str(f) for f in whole)}{mark}")
    if not found:
        print("(no feasible fingering of this segment)")
        return 0

    print(f"\n{len(found)} distinct solutions. A gap under about 1.5 is within the "
          f"model's own noise:\nany of those is worth trying at the piano.")
    if len(found) > 1:
        runner_up = found[1][1]
        print(f"\nIf you disagree with the top row, record it:\n"
              f"  python scripts/prefer.py record "
              f"{'--musicxml ' + str(args.musicxml) if args.musicxml else '--notes ' + repr(args.notes)} "
              f"--hand {args.hand} --span-cm {args.span_cm:g} --segment {lo+1}-{hi+1} "
              f"--prefer {format_fingering(runner_up)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
