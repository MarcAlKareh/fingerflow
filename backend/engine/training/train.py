"""Learn feature weights from human fingerings with a structured perceptron.

The engine's cost is linear in the weights, cost(y) = w . Phi(x, y), and
the decoder returns argmin_y cost(y). Given a human fingering y* for a
piece x, each training step runs a loss-augmented decode (Taskar et al.
2005) and a passive-aggressive update (Crammer et al. 2006):

    y_hat = argmin_y [ w . Phi(x, y) - rho * Hamming(y, y*) ]
    loss  = w . Phi(x, y*) - w . Phi(x, y_hat) + rho * Hamming(y_hat, y*)
    if loss > 0:
        tau = min(C, loss / |Phi(x, y_hat) - Phi(x, y*)|^2)
        w  <- w + tau * (Phi(x, y_hat) - Phi(x, y*))

so that the human fingering becomes cheaper than its strongest rival by
a margin proportional to how many notes differ, using the smallest
weight change that achieves it. Weights are averaged over all steps
(Collins 2002), which is what makes the result generalise.

Features are computed once per piece and cached on disk, so each epoch
only runs the numpy Viterbi.

Usage:

    python -m engine.training.train --pig-dir PIG/FingeringFiles \
        --out weights_learned.json --epochs 8 --cache .feature_cache

The PIG dataset must be downloaded separately (see pig.py).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import build_events, new_note_fingers, notes_from_hand_data
from ..features import FeatureTensors, N_SECOND, N_STATE, N_TRANS, TRANSITION_FEATURES
from ..fingering import path_from_fingers
from ..hand import hand_profile
from ..viterbi import decode
from ..weights import Weights
from .pig import PigSequence, load_pig_dir, split_by_piece

FEATURE_VERSION = "ff-features-2026-09-01"


@dataclass
class PreparedSequence:
    key: str
    piece: str
    annotator: str
    hand: str
    tensors: FeatureTensors
    gold_path: List[Optional[int]]
    # per event: number of struck notes, and gold finger per struck note
    gold_fingers: Dict[int, int]
    n_notes: int


def _cache_key(seq: PigSequence, span_cm: float) -> str:
    payload = f"{seq.path.name}|{seq.hand}|{span_cm}|{FEATURE_VERSION}|{len(seq.hand_data)}"
    return hashlib.sha1(payload.encode()).hexdigest()


def prepare_sequence(seq: PigSequence, span_cm: float, cache_dir: Optional[Path]) -> PreparedSequence:
    key = _cache_key(seq, span_cm)
    if cache_dir is not None:
        cached = cache_dir / f"{key}.pkl"
        if cached.exists():
            with cached.open("rb") as fh:
                return pickle.load(fh)

    profile = hand_profile(seq.hand, span_cm)
    notes = notes_from_hand_data(seq.hand_data)
    events = build_events(notes, profile)
    tensors = FeatureTensors(events, profile)
    gold_fingers = {int(item["note_id"]): int(f) for item, f in zip(seq.hand_data, seq.fingers)}
    gold_path = path_from_fingers(events, gold_fingers)
    prepared = PreparedSequence(
        key=key, piece=seq.piece, annotator=seq.annotator, hand=seq.hand,
        tensors=tensors, gold_path=gold_path, gold_fingers=gold_fingers, n_notes=len(notes),
    )
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with (cache_dir / f"{key}.pkl").open("wb") as fh:
            pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared


def _prepare_worker(args):
    seq, span_cm, cache_dir = args
    return prepare_sequence(seq, span_cm, cache_dir)


def prepare_all(sequences: Sequence[PigSequence], span_cm: float, cache_dir: Optional[Path], workers: int = 1) -> List[PreparedSequence]:
    jobs = [(s, span_cm, cache_dir) for s in sequences]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            return list(pool.imap(_prepare_worker, jobs, chunksize=1))
    out = []
    for i, job in enumerate(jobs):
        out.append(_prepare_worker(job))
        if (i + 1) % 20 == 0:
            print(f"  prepared {i + 1}/{len(jobs)}", file=sys.stderr)
    return out


def hamming_augment(prepared: PreparedSequence, rho: float) -> List[Optional[np.ndarray]]:
    """Per-event bonus: rho times the number of struck notes fingered unlike the human."""
    if rho <= 0:
        return [None] * len(prepared.tensors.events)
    out: List[Optional[np.ndarray]] = []
    for k, ev in enumerate(prepared.tensors.events):
        gold = prepared.gold_path[k]
        if gold is None:
            out.append(None)
            continue
        bonus = np.zeros(len(ev.assignments))
        for a, assign in enumerate(ev.assignments):
            mismatches = 0
            for note, finger in new_note_fingers(ev, assign):
                if prepared.gold_fingers.get(note.note_id) != finger:
                    mismatches += 1
            bonus[a] = rho * mismatches
        out.append(bonus)
    return out


def note_matches(prepared: PreparedSequence, path: Sequence[int]) -> Tuple[int, int]:
    """(matched notes, total notes with a gold finger)."""
    matched = 0
    total = 0
    for k, ev in enumerate(prepared.tensors.events):
        assign = ev.assignments[path[k]]
        for note, finger in new_note_fingers(ev, assign):
            gold = prepared.gold_fingers.get(note.note_id)
            if gold is None:
                continue
            total += 1
            if gold == finger:
                matched += 1
    return matched, total


def hamming_notes(prepared: PreparedSequence, path_a: Sequence[int], path_b: Sequence[int]) -> int:
    """Number of struck notes fingered differently by two assignment paths."""
    count = 0
    for k, ev in enumerate(prepared.tensors.events):
        if path_a[k] == path_b[k]:
            continue
        fa = ev.assignments[path_a[k]]
        fb = ev.assignments[path_b[k]]
        new_ids = {n.note_id for n in ev.new_notes}
        for note, x, y in zip(ev.sounding, fa, fb):
            if note.note_id in new_ids and x != y:
                count += 1
    return count


def _complete_gold(prepared: PreparedSequence, predicted: Sequence[int]) -> List[int]:
    """Gold path with illegal events (finger substitutions, odd chords) filled from the prediction."""
    return [g if g is not None else int(p) for g, p in zip(prepared.gold_path, predicted)]


def evaluate(prepared_list: Sequence[PreparedSequence], weights: Weights) -> Dict[str, float]:
    """General, highest and soft match rates in the sense of Nakamura et al. (2020)."""
    per_seq_matches: Dict[Tuple[str, str], List[Tuple[int, int, List[int]]]] = {}
    matched_total = 0
    notes_total = 0
    for prep in prepared_list:
        result = decode(prep.tensors, weights)
        m, t = note_matches(prep, result.path)
        matched_total += m
        notes_total += t
        per_seq_matches.setdefault((prep.piece, prep.hand), []).append((m, t, result.path))

    highest_num = 0.0
    highest_den = 0
    for (piece, hand), items in per_seq_matches.items():
        best_rate = max((m / t) if t else 0.0 for m, t, _ in items)
        n = max(t for _, t, _ in items)
        highest_num += best_rate * n
        highest_den += n
    return {
        "general_match_rate": matched_total / notes_total if notes_total else 0.0,
        "highest_match_rate": highest_num / highest_den if highest_den else 0.0,
        "notes": notes_total,
        "sequences": len(prepared_list),
    }


def train(
    prepared_train: Sequence[PreparedSequence],
    initial: Weights,
    epochs: int = 8,
    lr: float = 0.5,
    rho: float = 0.5,
    l2_to_prior: float = 0.0,
    nonnegative: bool = False,
    seed: int = 0,
    prepared_dev: Optional[Sequence[PreparedSequence]] = None,
    verbose: bool = True,
) -> Weights:
    rng = random.Random(seed)
    w = initial.copy()
    w0 = initial.copy()
    avg = initial.copy()
    n_updates = 0
    n_steps = 0
    order = list(range(len(prepared_train)))

    # Only the physically meaningful features are clipped when asked to;
    # categorical finger-pair features and the black-key crossing
    # discount may take either sign.
    signed_trans = np.array([
        name.startswith("pair_") or name == "cross_other_on_black" for name in TRANSITION_FEATURES
    ])

    for epoch in range(epochs):
        rng.shuffle(order)
        errors = 0
        matched = 0
        total = 0
        t0 = time.time()
        for idx in order:
            prep = prepared_train[idx]
            augment = hamming_augment(prep, rho)
            pred = decode(prep.tensors, w, augment=augment).path
            gold = _complete_gold(prep, pred)
            n_steps += 1
            m, t = note_matches(prep, pred)
            matched += m
            total += t
            if pred != gold:
                errors += 1
                fs_p, ft_p, fq_p = prep.tensors.path_features(pred)
                fs_g, ft_g, fq_g = prep.tensors.path_features(gold)
                d_s, d_t, d_q = fs_p - fs_g, ft_p - ft_g, fq_p - fq_g
                # Passive-aggressive step (PA-I, Crammer et al. 2006): the
                # smallest change that makes the human fingering cheaper
                # than the rival by the required margin, capped by lr.
                margin = rho * hamming_notes(prep, pred, gold)
                gap = float(w.state @ fs_g + w.trans @ ft_g + w.second @ fq_g
                            - (w.state @ fs_p + w.trans @ ft_p + w.second @ fq_p))
                loss = gap + margin
                norm2 = float(d_s @ d_s + d_t @ d_t + d_q @ d_q)
                if loss > 0 and norm2 > 0:
                    step = min(lr, loss / norm2)
                    w.state += step * d_s
                    w.trans += step * d_t
                    w.second += step * d_q
                if l2_to_prior > 0:
                    w.state -= lr * l2_to_prior * (w.state - w0.state)
                    w.trans -= lr * l2_to_prior * (w.trans - w0.trans)
                    w.second -= lr * l2_to_prior * (w.second - w0.second)
                if nonnegative:
                    w.state = np.maximum(w.state, 0.0)
                    w.second = np.maximum(w.second, 0.0)
                    w.trans = np.where(signed_trans, w.trans, np.maximum(w.trans, 0.0))
                n_updates += 1
            # Running average of the weight vector over all steps.
            avg.state += (w.state - avg.state) / n_steps
            avg.trans += (w.trans - avg.trans) / n_steps
            avg.second += (w.second - avg.second) / n_steps
        if verbose:
            msg = (f"epoch {epoch + 1}/{epochs}: {errors}/{len(order)} sequences updated, "
                   f"train match (augmented decode) {matched / max(total, 1):.3f}, {time.time() - t0:.1f}s")
            if prepared_dev:
                dev = evaluate(prepared_dev, avg)
                msg += f", dev general {dev['general_match_rate']:.3f} highest {dev['highest_match_rate']:.3f}"
            print(msg, file=sys.stderr)
    return avg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", required=True, type=Path, help="Directory containing *_fingering.txt files")
    parser.add_argument("--out", required=True, type=Path, help="Where to write the learned weights JSON")
    parser.add_argument("--init", type=Path, default=None, help="Initial weights JSON (default: built-in)")
    parser.add_argument("--cache", type=Path, default=None, help="Feature cache directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.5, help="Maximum passive-aggressive step size C")
    parser.add_argument("--rho", type=float, default=0.5, help="Loss-augmentation margin per mismatched note")
    parser.add_argument("--l2-to-prior", type=float, default=0.01, help="Pull learned weights toward the initial (physical) weights")
    parser.add_argument("--nonnegative", action="store_true", help="Keep physical feature weights non-negative")
    parser.add_argument("--span-cm", type=float, default=21.0, help="Assumed hand span of the annotators")
    parser.add_argument("--hands", default="both", choices=["both", "right", "left"])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N sequences (smoke tests)")
    args = parser.parse_args(argv)

    hands = ("right", "left") if args.hands == "both" else (args.hands,)
    sequences = load_pig_dir(args.pig_dir, hands=hands)
    if args.limit:
        sequences = sequences[: args.limit]
    if not sequences:
        print("No fingering files found", file=sys.stderr)
        return 1
    train_seqs, test_seqs = split_by_piece(sequences, args.test_fraction)
    print(f"{len(train_seqs)} training and {len(test_seqs)} test sequences", file=sys.stderr)

    print("Preparing features ...", file=sys.stderr)
    prepared_train = prepare_all(train_seqs, args.span_cm, args.cache, args.workers)
    prepared_test = prepare_all(test_seqs, args.span_cm, args.cache, args.workers)
    illegal = sum(sum(1 for g in p.gold_path if g is None) for p in prepared_train)
    events = sum(len(p.gold_path) for p in prepared_train)
    print(f"{illegal}/{events} training events have a human fingering the state space cannot express "
          f"(substitutions, non-monotone chords); they are filled from the model's own prediction.", file=sys.stderr)

    initial = Weights.load(args.init) if args.init else Weights.default()
    before = evaluate(prepared_test, initial) if prepared_test else None
    if before:
        print(f"Before training: general {before['general_match_rate']:.3f}, highest {before['highest_match_rate']:.3f}", file=sys.stderr)

    learned = train(
        prepared_train, initial, epochs=args.epochs, lr=args.lr, rho=args.rho,
        l2_to_prior=args.l2_to_prior, nonnegative=args.nonnegative,
        prepared_dev=prepared_test or None,
    )
    learned.save(args.out)
    after = evaluate(prepared_test, learned) if prepared_test else None
    if after:
        print(f"After training: general {after['general_match_rate']:.3f}, highest {after['highest_match_rate']:.3f}", file=sys.stderr)
    print(f"Saved weights to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
