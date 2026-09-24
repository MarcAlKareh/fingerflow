"""Finger one passage and show the reasoning, including the near-misses.

For a passage you are actually struggling with, the chosen fingering is only
half the answer. The other half is how close the alternatives were: a finger
the engine rejected by a hair is worth trying at the piano, while one it
rejected by a mile is not.

    python scripts/explain_passage.py --hand right --span-cm 23 --bpm 54 \
        --notes "Eb6 C6 Bb5 Ab5 G5 F5 Eb5 D5 C5"

Note names are scientific pitch: C4 is middle C, so Eb5 is the E flat above
it. Accidentals may be written b or #. Use --note-value to say what each note
is worth (8 for quavers, 16 for semiquavers) so the tempo features see the
real speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engine.events import build_events, new_note_fingers, notes_from_hand_data  # noqa: E402
from engine.features import FeatureTensors  # noqa: E402
from engine.fingering import assign_fingering  # noqa: E402
from engine.hand import hand_profile  # noqa: E402
from engine.keyboard import midi_to_name, name_to_midi  # noqa: E402
from engine.viterbi import decode, event_best_costs  # noqa: E402
from engine.weights import Weights  # noqa: E402

NEAR_MISS = 1.5      # an alternative this much dearer is worth trying
TIGHT = 0.75         # below this the engine is close to indifferent


def parse_notes(text: str):
    tokens = [t for t in text.replace(",", " ").split() if t]
    return [name_to_midi(t) for t in tokens]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--notes", help="Space-separated note names, e.g. 'Eb6 C6 Bb5'")
    source.add_argument("--musicxml", type=Path,
                        help="A .musicxml or .mxl file; real note times and tempo are used")
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--span-cm", type=float, default=21.0, help="Your little-finger-to-thumb spread")
    parser.add_argument("--bpm", type=float, default=60.0, help="Beats per minute")
    parser.add_argument("--note-value", type=int, default=8,
                        help="8 for quavers, 16 for semiquavers, 32 for demisemiquavers")
    parser.add_argument("--legato", action="store_true", default=True)
    parser.add_argument("--staccato", dest="legato", action="store_false")
    parser.add_argument("--goal", default=None, choices=[None, "beginner", "expression", "speed"])
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)
    args.bpm_given = any(a.startswith("--bpm") for a in (argv if argv is not None else sys.argv[1:]))

    if args.musicxml:
        from omr.parser import parse_musicxml

        parsed = parse_musicxml(args.musicxml, tempo_bpm=args.bpm if args.bpm_given else None)
        data = [{k: v for k, v in n.items() if k != "finger"} for n in parsed[f"{args.hand}_hand"]]
        if not data:
            other = "left" if args.hand == "right" else "right"
            print(f"No notes on the {args.hand} staff. The file has "
                  f"{len(parsed[f'{other}_hand'])} on the {other}. Try --hand {other}.", file=sys.stderr)
            return 1
        for i, note in enumerate(data):
            note["note_id"] = i
        pitches = [n["pitch"] for n in data]
        onsets = sorted({round(n["start_time_sec"], 4) for n in data})
        gaps = [b - a for a, b in zip(onsets, onsets[1:])]
        ioi = min(gaps) if gaps else 0.5
        tempo_note = (f"score tempo {parsed['score_tempo_bpm']:g} bpm"
                      + (f", overridden to {args.bpm:g}" if args.bpm_given else ""))
    else:
        pitches = parse_notes(args.notes)
        if len(pitches) < 2:
            print("Give at least two notes.", file=sys.stderr)
            return 1
        # A note of value N at B bpm lasts (4/N) * 60/B seconds.
        ioi = (4.0 / args.note_value) * 60.0 / args.bpm
        duration = ioi if args.legato else ioi * 0.5
        data = [{"note_id": i, "pitch": p, "start_time_sec": i * ioi, "duration_sec": duration}
                for i, p in enumerate(pitches)]
        tempo_note = f"{args.note_value}ths at {args.bpm:g} bpm"

    weights = Weights.load(args.weights) if args.weights else None
    result = assign_fingering(data, args.hand, hand_span_cm=args.span_cm, goal=args.goal, weights=weights)

    profile = hand_profile(args.hand, args.span_cm)
    events = build_events(notes_from_hand_data(data), profile)
    tensors = FeatureTensors(events, profile)
    w = (weights or Weights.default()).with_preset(args.goal) if weights else Weights.default(args.goal)
    path = decode(tensors, w).path
    best = event_best_costs(tensors, w)
    optimal = min(float(b.min()) for b in best if b.size)

    print(f"{len(pitches)} notes, {args.hand} hand, span {args.span_cm} cm, {tempo_note}, "
          f"fastest note {ioi*1000:.0f} ms apart ({1/ioi:.1f} notes per second)")
    print(f"total strain cost {result.total_cost:.2f}\n")
    print(f"{'#':>3s}  {'note':<5s} {'finger':>6s}   {'alternatives (extra cost)':<34s}  notes")
    print("-" * 96)

    tight_spots = []
    for k, ev in enumerate(events):
        assign = ev.assignments[path[k]]
        note, finger = new_note_fingers(ev, assign)[0]
        extras = []
        for a, cost in enumerate(best[k]):
            alt = ev.assignments[a]
            if alt == assign or not (cost < float("inf")):
                continue
            extras.append((cost - optimal, alt[0]))
        extras.sort()
        shown = "  ".join(f"{f}:+{d:.2f}" for d, f in extras[:3] if d < NEAR_MISS * 4) or "none close"
        reasons = "; ".join(result.explanations.get(note.note_id, [])[:2])
        flag = ""
        if extras and extras[0][0] < TIGHT:
            flag = "  <- nearly a coin flip"
            tight_spots.append((k + 1, midi_to_name(note.midi), finger, extras[0][1], extras[0][0]))
        print(f"{k+1:3d}  {midi_to_name(note.midi):<5s} {finger:>6d}   {shown:<34s}  {reasons}{flag}")

    print()
    if tight_spots:
        print("Worth trying both ways at the piano, the engine is close to indifferent here:")
        for index, name, chosen, alt, margin in tight_spots:
            print(f"  note {index} ({name}): it chose {chosen}, but {alt} costs only {margin:+.2f} more")
    else:
        print("No close calls: every note had a clear best finger.")
    if result.unfingered_note_ids:
        print(f"\nUnfingered notes: {result.unfingered_note_ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
