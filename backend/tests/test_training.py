"""Smoke tests for the PIG loader, trainer and evaluator on synthetic fixtures."""

from pathlib import Path

from engine.training.pig import load_pig_dir, parse_finger, split_by_piece
from engine.training.train import evaluate, prepare_all, train
from engine.viterbi import decode
from engine.weights import Weights

FIXTURES = Path(__file__).parent / "fixtures" / "pig"


def test_parse_finger_tokens():
    assert parse_finger("3") == 3
    assert parse_finger("-2") == 2
    assert parse_finger("1_2") == 1
    assert parse_finger("-1_-2") == 1
    assert parse_finger("x") is None


def test_loader_splits_hands_and_pieces():
    seqs = load_pig_dir(FIXTURES)
    assert {(s.piece, s.annotator, s.hand) for s in seqs} == {
        ("001", "1", "right"), ("001", "1", "left"),
        ("001", "2", "right"), ("001", "2", "left"),
        ("002", "1", "right"), ("002", "1", "left"),
    }
    rh = next(s for s in seqs if s.piece == "001" and s.annotator == "1" and s.hand == "right")
    assert rh.fingers[:8] == [1, 2, 3, 1, 2, 3, 4, 5]
    assert all(1 <= f <= 5 for s in seqs for f in s.fingers)
    train_seqs, test_seqs = split_by_piece(seqs, test_fraction=0.5)
    assert {s.piece for s in train_seqs}.isdisjoint({s.piece for s in test_seqs})


def test_default_weights_match_standard_fingerings():
    seqs = [s for s in load_pig_dir(FIXTURES) if s.annotator == "1"]
    prepared = prepare_all(seqs, 21.0, None)
    metrics = evaluate(prepared, Weights.default())
    assert metrics["general_match_rate"] >= 0.9


def test_training_can_fit_an_unusual_fingering():
    seqs = [s for s in load_pig_dir(FIXTURES, hands=("right",)) if s.annotator == "2"]
    prepared = prepare_all(seqs, 21.0, None)
    before = evaluate(prepared, Weights.default())["general_match_rate"]
    learned = train(prepared, Weights.default(), epochs=15, lr=1.0, rho=0.5, verbose=False)
    after = evaluate(prepared, learned)["general_match_rate"]
    assert after > before
    assert after == 1.0


def test_feature_cache_roundtrip(tmp_path):
    seqs = load_pig_dir(FIXTURES, hands=("left",))[:1]
    first = prepare_all(seqs, 21.0, tmp_path)
    second = prepare_all(seqs, 21.0, tmp_path)
    assert first[0].key == second[0].key
    assert decode(first[0].tensors, Weights.default()).path == decode(second[0].tensors, Weights.default()).path
