"""Adapt the weights to one player's hand from preferences, not labels.

The PIG-trained weights answer "what would a pianist write here?". They
cannot answer "what suits *your* hand?", because nothing in PIG records the
annotator's hand. The Pathetique case study made the gap concrete: the
learned model prefers 3-5-4-3-2 on a turn where the player's hand prefers
3-4-3-2-1, and the two differ by 1.04, which is inside the model's own
noise. No amount of further PIG training resolves that. Only the player can.

So the training signal here is a *comparison*, not a label. The player plays
two complete fingerings of the same passage and says which one they would
keep. That is far easier to give honestly than a fingering written from
scratch, and it is the only kind of judgement about comfort that a person
can make reliably.

Learning from it is the mirror image of the structured perceptron in
``train.py``. Where that makes a human's fingering cheaper than the model's
rival, this makes the preferred fingering cheaper than the rejected one:

    d     = Phi(preferred) - Phi(rejected)
    loss  = w . d + margin
    if loss > 0:
        tau = min(C, loss / |d|^2)
        w  <- w - tau * d

The passive-aggressive step (Crammer et al. 2006) takes the smallest weight
change that satisfies the comparison, which matters enormously here: with
perhaps twenty preferences against ninety-nine weights, the only thing
keeping the result sane is that every step is minimal and pulled back
toward the population prior. Weights are averaged over all steps (Collins
2002), and reported against a leave-one-out estimate, because agreement
with the preferences used for fitting means nothing at this sample size.

    python scripts/prefer.py record --musicxml passage.mxl --hand right \\
        --span-cm 23 --segment 9-13 --prefer 3-4-3-2-1 --over 3-5-4-3-2 \\
        --why "2 to 4 up a tone is uncomfortable"

    python scripts/prefer.py fit --prefs my_prefs.jsonl \\
        --base backend/weights_learned.json --out weights_kevin.json

    python scripts/prefer.py check --prefs my_prefs.jsonl --weights weights_kevin.json
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..passage import Passage, PassageSpec, format_fingering
from ..weights import Weights

# How much cheaper the preferred fingering must become. In the same strain
# units as the cost function, where a whole passage runs to tens and the
# gaps that matter are around one.
DEFAULT_MARGIN = 0.75
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.5
DEFAULT_L2 = 0.02


@dataclass
class Preference:
    """One judgement: on this passage, this fingering beats that one."""

    source: PassageSpec
    segment: Tuple[int, int]        # 1-based, inclusive, as the player counted
    prefer: List[int]
    over: List[int]
    why: str = ""
    weight: float = 1.0
    recorded: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "segment": list(self.segment),
            "prefer": list(self.prefer),
            "over": list(self.over),
            "why": self.why,
            "weight": self.weight,
            "recorded": self.recorded or date.today().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Preference":
        return cls(
            source=PassageSpec.from_dict(data["source"]),
            segment=(int(data["segment"][0]), int(data["segment"][1])),
            prefer=[int(f) for f in data["prefer"]],
            over=[int(f) for f in data["over"]],
            why=data.get("why", ""),
            weight=float(data.get("weight", 1.0)),
            recorded=data.get("recorded", ""),
        )

    @property
    def label(self) -> str:
        lo, hi = self.segment
        return (f"{self.source.label or 'passage'} {lo}-{hi} "
                f"{format_fingering(self.prefer)} over {format_fingering(self.over)}")


def load_preferences(path: Path) -> List[Preference]:
    out: List[Preference] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(Preference.from_dict(json.loads(line)))
        except Exception as exc:  # noqa: BLE001 - the line number is the useful part
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return out


def append_preference(path: Path, pref: Preference) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pref.to_dict()) + "\n")


@dataclass
class Comparison:
    """A preference resolved into two complete fingerings and their feature gap."""

    pref: Preference
    passage: Passage
    path_prefer: List[int]
    path_over: List[int]
    delta_state: np.ndarray
    delta_trans: np.ndarray
    delta_second: np.ndarray

    def cost_gap(self, weights: Weights) -> float:
        """Cost of the preferred fingering minus the rejected one; negative is satisfied."""
        return float(weights.state @ self.delta_state
                     + weights.trans @ self.delta_trans
                     + weights.second @ self.delta_second)

    @property
    def norm2(self) -> float:
        return float(self.delta_state @ self.delta_state
                     + self.delta_trans @ self.delta_trans
                     + self.delta_second @ self.delta_second)


def resolve(pref: Preference, weights: Weights) -> Comparison:
    """Complete both fingerings of the passage and take their feature difference.

    Only the segment is fixed; the rest of the passage is re-optimised
    around it under ``weights``, so each side is a complete, playable
    fingering. The completions are computed once, against the base weights,
    and then held fixed while fitting. Re-completing them at every step
    would chase its own tail: the model's opinion of the context is exactly
    what is being adjusted.
    """
    passage = pref.source.build()
    lo, hi = pref.segment[0] - 1, pref.segment[1] - 1
    span = list(range(lo, hi + 1))
    for name, fingers in (("prefer", pref.prefer), ("over", pref.over)):
        if len(fingers) != len(span):
            raise ValueError(f"{pref.label}: {name} has {len(fingers)} fingers "
                             f"for a {len(span)}-note segment")

    got_p = passage.constrained(weights, dict(zip(span, pref.prefer)))
    got_o = passage.constrained(weights, dict(zip(span, pref.over)))
    if got_p is None:
        raise ValueError(f"{pref.label}: the preferred fingering is not playable here")
    if got_o is None:
        raise ValueError(f"{pref.label}: the rejected fingering is not playable here")

    fs_p, ft_p, fq_p = passage.path_features(got_p[0])
    fs_o, ft_o, fq_o = passage.path_features(got_o[0])
    return Comparison(
        pref=pref, passage=passage, path_prefer=got_p[0], path_over=got_o[0],
        delta_state=fs_p - fs_o, delta_trans=ft_p - ft_o, delta_second=fq_p - fq_o,
    )


def resolve_all(prefs: Sequence[Preference], weights: Weights) -> List[Comparison]:
    return [resolve(p, weights) for p in prefs]


def satisfied(comparisons: Sequence[Comparison], weights: Weights) -> List[bool]:
    """Which comparisons the weights now get right (preferred is cheaper)."""
    return [c.cost_gap(weights) < 0 for c in comparisons]


def fit(
    comparisons: Sequence[Comparison],
    base: Weights,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    margin: float = DEFAULT_MARGIN,
    l2_to_prior: float = DEFAULT_L2,
    seed: int = 0,
) -> Weights:
    """Passive-aggressive fit to a set of comparisons, averaged over all steps."""
    rng = random.Random(seed)
    w = base.copy()
    w0 = base.copy()
    avg = base.copy()
    steps = 0
    order = list(range(len(comparisons)))

    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            comp = comparisons[index]
            steps += 1
            loss = comp.cost_gap(w) + margin * comp.pref.weight
            norm2 = comp.norm2
            if loss > 0 and norm2 > 0:
                tau = min(lr, loss / norm2)
                w.state -= tau * comp.delta_state
                w.trans -= tau * comp.delta_trans
                w.second -= tau * comp.delta_second
            if l2_to_prior > 0:
                w.state -= lr * l2_to_prior * (w.state - w0.state)
                w.trans -= lr * l2_to_prior * (w.trans - w0.trans)
                w.second -= lr * l2_to_prior * (w.second - w0.second)
            avg.state += (w.state - avg.state) / steps
            avg.trans += (w.trans - avg.trans) / steps
            avg.second += (w.second - avg.second) / steps
    return avg


def leave_one_out(
    comparisons: Sequence[Comparison],
    base: Weights,
    **kwargs,
) -> List[bool]:
    """For each comparison, fit on the others and test on it.

    With a handful of preferences this is the only honest score. Fitting is
    pure vector arithmetic once the comparisons are resolved, so running it
    n times costs nothing.
    """
    out: List[bool] = []
    for i in range(len(comparisons)):
        rest = [c for j, c in enumerate(comparisons) if j != i]
        if not rest:
            out.append(False)
            continue
        adapted = fit(rest, base, **kwargs)
        out.append(comparisons[i].cost_gap(adapted) < 0)
    return out


def drift(base: Weights, adapted: Weights, top: int = 8) -> Tuple[float, List[Tuple[str, float, float]]]:
    """How far the weights moved, and which features moved most."""
    a, b = base.to_dict(), adapted.to_dict()
    changes = [(name, a[name], b[name]) for name in a]
    distance = float(np.sqrt(sum((b[n] - a[n]) ** 2 for n in a)))
    changes.sort(key=lambda row: -abs(row[2] - row[1]))
    return distance, changes[:top]
