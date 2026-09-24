"""Teach the engine your hand, one comparison at a time.

Three commands.

``record`` stores a judgement you made at the piano. Give it the passage,
the group of notes you are judging, and the two fingerings:

    python scripts/prefer.py record --musicxml beethoven.mxl --hand right \\
        --span-cm 23 --segment 9-13 --prefer 3-4-3-2-1 --over 3-5-4-3-2 \\
        --why "2 to 4 up a tone is uncomfortable"

The whole passage is copied into the record, not a path to the file, so the
judgement survives the file being moved or re-exported. Leave ``--over`` out
and the engine's own current choice is used, which is the usual case: you
disagreed with what it gave you.

``fit`` learns from everything recorded so far:

    python scripts/prefer.py fit --prefs my_prefs.jsonl \\
        --base backend/weights_learned.json --out weights_kevin.json

``check`` scores weights against the preferences without changing anything:

    python scripts/prefer.py check --prefs my_prefs.jsonl --weights weights_kevin.json

Read the leave-one-out figure, not the fitted one. Fitting twenty
preferences into ninety-nine weights can satisfy all twenty and have learned
nothing, and the difference between the two numbers is how you tell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engine.passage import PassageSpec, format_fingering, parse_fingering, parse_segment  # noqa: E402
from engine.training.preferences import (  # noqa: E402
    DEFAULT_EPOCHS, DEFAULT_L2, DEFAULT_LR, DEFAULT_MARGIN, Preference,
    append_preference, drift, fit, leave_one_out, load_preferences, resolve_all, satisfied,
)
from engine.weights import Weights  # noqa: E402


def load_weights(path: Path | None, goal: str | None = None) -> Weights:
    return Weights.load(path).with_preset(goal) if path else Weights.default(goal)


def build_spec(args) -> PassageSpec:
    if args.musicxml:
        return PassageSpec.from_musicxml(
            args.musicxml, hand=args.hand, span_cm=args.span_cm,
            tempo_bpm=args.bpm if args.bpm_given else None, label=args.label or "")
    return PassageSpec.from_note_names(
        args.notes, hand=args.hand, span_cm=args.span_cm, bpm=args.bpm,
        note_value=args.note_value, label=args.label or "")


def cmd_record(args) -> int:
    spec = build_spec(args)
    passage = spec.build()
    weights = load_weights(args.weights, args.goal)
    lo, hi = parse_segment(args.segment, len(passage))
    span = list(range(lo, hi + 1))

    prefer = parse_fingering(args.prefer)
    if args.over:
        over = parse_fingering(args.over)
    else:
        path, _ = passage.best(weights)
        over = passage.fingers(path)[lo:hi + 1]
        print(f"using the engine's own choice as the rejected side: {format_fingering(over)}")
    if prefer == over:
        print("The two fingerings are identical, so there is nothing to learn from.", file=sys.stderr)
        return 1
    for name, fingers in (("--prefer", prefer), ("--over", over)):
        if len(fingers) != len(span):
            print(f"{name} has {len(fingers)} fingers for a {len(span)}-note segment "
                  f"(notes {lo+1}-{hi+1})", file=sys.stderr)
            return 1

    pref = Preference(source=spec, segment=(lo + 1, hi + 1), prefer=prefer, over=over,
                      why=args.why or "", weight=args.strength)
    # Resolving now means an unplayable fingering is caught here rather than
    # at fitting time, when the context of the mistake is long gone.
    from engine.training.preferences import resolve

    try:
        comp = resolve(pref, weights)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    gap = comp.cost_gap(weights)
    append_preference(args.prefs, pref)
    notes = " ".join(passage.names[lo:hi + 1])
    print(f"recorded: notes {lo+1}-{hi+1} ({notes})")
    print(f"  you prefer {format_fingering(prefer)} over {format_fingering(over)}")
    print(f"  the current weights disagree by {gap:+.2f}"
          if gap >= 0 else
          f"  the current weights already agree, by {-gap:.2f}")
    print(f"  {sum(1 for _ in load_preferences(args.prefs))} preferences on file in {args.prefs}")
    return 0


def report(comparisons, base, weights, label: str) -> None:
    flags = satisfied(comparisons, weights)
    print(f"\n{label}: {sum(flags)}/{len(flags)} preferences satisfied")
    print(f"  {'passage':<34s}{'prefer':>10s}{'over':>10s}{'gap':>9s}")
    for comp, ok in zip(comparisons, flags):
        gap = comp.cost_gap(weights)
        name = (comp.pref.source.label or "passage")[:24]
        lo, hi = comp.pref.segment
        print(f"  {name + f' {lo}-{hi}':<34s}"
              f"{format_fingering(comp.pref.prefer):>10s}"
              f"{format_fingering(comp.pref.over):>10s}"
              f"{gap:>+9.2f}  {'ok' if ok else 'no'}")
    print("  gap is the preferred fingering's cost minus the rejected one's; "
          "negative means the model agrees with you.")


def cmd_fit(args) -> int:
    prefs = load_preferences(args.prefs)
    if not prefs:
        print(f"No preferences in {args.prefs}. Record some first.", file=sys.stderr)
        return 1
    base = load_weights(args.base, args.goal)
    print(f"{len(prefs)} preferences, base weights {args.base or 'built-in defaults'}", file=sys.stderr)
    comparisons = resolve_all(prefs, base)

    report(comparisons, base, base, "Before adaptation")

    adapted = fit(comparisons, base, epochs=args.epochs, lr=args.lr,
                  margin=args.margin, l2_to_prior=args.l2_to_prior)
    report(comparisons, base, adapted, "After adaptation (fitted, optimistic)")

    held = leave_one_out(comparisons, base, epochs=args.epochs, lr=args.lr,
                         margin=args.margin, l2_to_prior=args.l2_to_prior)
    print(f"\nLeave-one-out: {sum(held)}/{len(held)} preferences predicted from the others.")
    print("  This is the number that means something. If it is close to the fitted")
    print("  figure, the preferences share structure the model has generalised. If it")
    print("  is near chance, you have taught it these passages and nothing more.")

    distance, movers = drift(base, adapted, top=args.show_weights)
    print(f"\nThe weights moved {distance:.3f} from the prior. Largest changes:")
    print(f"  {'feature':<26s}{'prior':>9s}{'yours':>9s}{'change':>9s}")
    for name, before, after in movers:
        print(f"  {name:<26s}{before:>9.3f}{after:>9.3f}{after - before:>+9.3f}")

    if args.out:
        adapted.save(args.out)
        print(f"\nWritten to {args.out}. Use it with --weights on any of the other tools.")
        print("Keep the population weights too: this file is your hand, not the model.")
    else:
        print("\nNothing written. Pass --out to save.")
    return 0


def cmd_check(args) -> int:
    prefs = load_preferences(args.prefs)
    if not prefs:
        print(f"No preferences in {args.prefs}.", file=sys.stderr)
        return 1
    base = load_weights(args.base, args.goal)
    weights = load_weights(args.weights, args.goal)
    comparisons = resolve_all(prefs, base)
    report(comparisons, base, weights, f"{args.weights or 'built-in defaults'}")
    return 0


def add_passage_args(parser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--notes", help="Space-separated note names, e.g. 'C6 D6 C6 A#5 G#5'")
    source.add_argument("--musicxml", type=Path)
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--bpm", type=float, default=60.0)
    parser.add_argument("--note-value", type=int, default=8)
    parser.add_argument("--label", default=None, help="How this passage should appear in reports")


def add_fit_args(parser) -> None:
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Largest single passive-aggressive step")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="How much cheaper the preferred fingering must become")
    parser.add_argument("--l2-to-prior", type=float, default=DEFAULT_L2,
                        help="Pull back toward the population weights; raise it if "
                             "leave-one-out is much worse than fitted")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)

    rec = subs.add_parser("record", help="Store one judgement")
    add_passage_args(rec)
    rec.add_argument("--segment", required=True, help="Note range you are judging, e.g. 9-13")
    rec.add_argument("--prefer", required=True, help="The fingering you would keep, e.g. 3-4-3-2-1")
    rec.add_argument("--over", default=None, help="The one you reject (default: the engine's choice)")
    rec.add_argument("--why", default=None, help="A note to your future self")
    rec.add_argument("--strength", type=float, default=1.0,
                     help="1.0 for an ordinary preference, higher if you feel strongly")
    rec.add_argument("--weights", type=Path, default=None, help="Weights the engine is currently using")
    rec.add_argument("--goal", default=None, choices=[None, "beginner", "expression", "speed"],
                     help="Preset the engine is currently using, so --over reflects what you were shown")
    rec.add_argument("--prefs", type=Path, default=Path("my_prefs.jsonl"))
    rec.set_defaults(func=cmd_record)

    fit_p = subs.add_parser("fit", help="Learn from every recorded judgement")
    fit_p.add_argument("--prefs", type=Path, default=Path("my_prefs.jsonl"))
    fit_p.add_argument("--base", type=Path, default=None, help="Population weights to start from")
    fit_p.add_argument("--out", type=Path, default=None)
    fit_p.add_argument("--show-weights", type=int, default=8)
    fit_p.add_argument("--goal", default=None, choices=[None, "beginner", "expression", "speed"])
    add_fit_args(fit_p)
    fit_p.set_defaults(func=cmd_fit)

    chk = subs.add_parser("check", help="Score weights against the preferences")
    chk.add_argument("--prefs", type=Path, default=Path("my_prefs.jsonl"))
    chk.add_argument("--base", type=Path, default=None)
    chk.add_argument("--weights", type=Path, default=None)
    chk.add_argument("--goal", default=None, choices=[None, "beginner", "expression", "speed"])
    chk.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    raw = argv if argv is not None else sys.argv[1:]
    args.bpm_given = any(a.startswith("--bpm") for a in raw)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
