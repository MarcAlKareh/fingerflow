"""Second-order Viterbi decoding over event finger assignments.

State at event k is an assignment s_k of fingers to all notes sounding
at that event (see events.py). The dynamic programme runs over pairs
(s_{k-1}, s_k) so that costs may depend on three consecutive
assignments:

    D_k(p, a) = min_h [ D_{k-1}(h, p) + Q_k(h, p, a) ] + T_k(p, a) + S_k(a)

with S, T, Q the weighted state, transition and second-order costs.
Transitions that would change the finger on a held note are infeasible.
Complexity is O(K * n^3) for n assignments per event (n <= 10), which
is negligible for any score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .features import FeatureTensors
from .weights import Weights

INF = float("inf")
FORCED_RELEASE_PENALTY = 50.0


@dataclass
class DecodeResult:
    path: List[int]
    cost: float


def _costs(tensors: FeatureTensors, weights: Weights, augment: Optional[Sequence[Optional[np.ndarray]]]):
    """Weighted cost arrays per event, with infeasible transitions at +inf."""
    cS: List[np.ndarray] = []
    cT: List[Optional[np.ndarray]] = []
    cQ: List[Optional[np.ndarray]] = []
    for k in range(len(tensors.events)):
        s = tensors.state[k].astype(np.float64) @ weights.state
        if augment is not None and augment[k] is not None:
            s = s - augment[k]
        cS.append(s)
        if k == 0 or tensors.trans[k] is None:
            cT.append(None)
            cQ.append(None)
            continue
        t = tensors.trans[k].astype(np.float64) @ weights.trans
        feas = tensors.feasible[k]
        if feas is not None:
            # An assignment with no feasible predecessor is one a held note has
            # painted into a corner. Rather than failing, allow reaching it by
            # releasing the held note early, at a fixed penalty. Only those
            # assignments are relaxed; every other infeasible transition stays
            # prohibited, so a held note is never dropped for convenience.
            unreachable = ~feas.any(axis=0)
            if unreachable.any():
                t = np.where(feas, t, np.where(unreachable[None, :], t + FORCED_RELEASE_PENALTY, INF))
            else:
                t = np.where(feas, t, INF)
        cT.append(t)
        if tensors.second[k] is not None:
            cQ.append(tensors.second[k].astype(np.float64) @ weights.second)
        else:
            cQ.append(None)
    return cS, cT, cQ


def event_best_costs(tensors: FeatureTensors, weights: Weights) -> List[np.ndarray]:
    """For every event k and assignment a, the cost of the cheapest complete
    path that uses assignment a at event k.

    This is the min-sum forward-backward companion to :func:`decode`. The
    forward table is the same one the decoder builds,

        F_k(p, a) = min cost of s_1..s_k with s_{k-1} = p, s_k = a,

    and the backward table is its mirror,

        B_k(p, a) = min_b [ Q_{k+1}(p,a,b) + T_{k+1}(a,b) + S_{k+1}(b) + B_{k+1}(a,b) ],

    with B_K = 0. The backward state is the pair (p, a) rather than a alone
    because the second-order term Q_{k+1} reaches back to s_{k-1}. Then the
    cheapest path through assignment a at event k is min_p [F_k(p,a) + B_k(p,a)].

    Subtracting the optimal cost gives the price of changing the fingering at
    one event, which is what :mod:`engine.training.select` uses to find the
    passages where the model is close to indifferent.
    """
    K = len(tensors.events)
    if K == 0:
        return []
    cS, cT, cQ = _costs(tensors, weights, None)
    if K == 1:
        return [cS[0].copy()]

    # Forward.
    F: List[Optional[np.ndarray]] = [None] * K
    F[1] = cS[0][:, None] + cT[1] + cS[1][None, :]
    for k in range(2, K):
        if cQ[k] is not None:
            candidate = F[k - 1][:, :, None] + cQ[k]
        else:
            candidate = np.broadcast_to(F[k - 1][:, :, None], F[k - 1].shape + (len(cS[k]),))
        F[k] = candidate.min(axis=0) + cT[k] + cS[k][None, :]

    # Backward.
    B: List[Optional[np.ndarray]] = [None] * K
    B[K - 1] = np.zeros_like(F[K - 1])
    for k in range(K - 2, 0, -1):
        nxt = k + 1
        term = cT[nxt][None, :, :] + cS[nxt][None, None, :] + B[nxt][None, :, :]
        if cQ[nxt] is not None:
            term = term + cQ[nxt]
        else:
            term = np.broadcast_to(term, (len(cS[k - 1]),) + term.shape[1:])
        B[k] = term.min(axis=2)

    best: List[np.ndarray] = [np.empty(0)] * K
    for k in range(1, K):
        best[k] = (F[k] + B[k]).min(axis=0)
    # Event 0 is the first member of the pair at k = 1.
    best[0] = (F[1] + B[1]).min(axis=1)
    return best


def path_cost(tensors: FeatureTensors, weights: Weights, path: Sequence[int]) -> float:
    """Cost of one assignment path.

    :func:`decode` returns the cost of the path it found, which is not the
    same thing when the decode was constrained or loss-augmented: those add
    penalties that are not part of the model. This recomputes the honest
    cost from the path alone, so costs from different constrained decodes
    are comparable with each other and with the unconstrained optimum.
    """
    if not path:
        return 0.0
    cS, cT, cQ = _costs(tensors, weights, None)
    total = float(cS[0][path[0]])
    for k in range(1, len(path)):
        total += float(cS[k][path[k]])
        if cT[k] is not None:
            total += float(cT[k][path[k - 1], path[k]])
        if k >= 2 and cQ[k] is not None:
            total += float(cQ[k][path[k - 2], path[k - 1], path[k]])
    return total


def event_margins(tensors: FeatureTensors, weights: Weights, path: Sequence[int]) -> np.ndarray:
    """Cost penalty for changing the fingering at each event, one value per event.

    A small margin means the model has a near-equally-good alternative there,
    which is exactly where a human judgement is worth asking for. Events with
    only one possible assignment get ``inf``.
    """
    best = event_best_costs(tensors, weights)
    if not best:
        return np.zeros(0)
    optimal = min(float(np.min(b)) for b in best if b.size)
    margins = np.full(len(best), np.inf)
    for k, b in enumerate(best):
        if b.size <= 1:
            continue
        alternatives = np.delete(b, path[k])
        if alternatives.size:
            margins[k] = float(np.min(alternatives)) - optimal
    return margins


def decode(
    tensors: FeatureTensors,
    weights: Weights,
    augment: Optional[Sequence[Optional[np.ndarray]]] = None,
) -> DecodeResult:
    """Minimum-cost assignment path. ``augment`` subtracts a per-assignment
    bonus from the state cost (used for loss-augmented decoding in training)."""
    K = len(tensors.events)
    if K == 0:
        return DecodeResult([], 0.0)

    cS, cT, cQ = _costs(tensors, weights, augment)

    if K == 1:
        a = int(np.argmin(cS[0]))
        return DecodeResult([a], float(cS[0][a]))

    # D_1(p, a)
    dp = cS[0][:, None] + cT[1] + cS[1][None, :]
    backpointers: List[Optional[np.ndarray]] = [None, None]

    for k in range(2, K):
        # candidate[h, p, a] = D_{k-1}(h, p) + Q_k(h, p, a)
        if cQ[k] is not None:
            candidate = dp[:, :, None] + cQ[k]
        else:
            candidate = np.broadcast_to(dp[:, :, None], dp.shape + (len(cS[k]),))
        bp = np.argmin(candidate, axis=0)                        # (n_{k-1}, n_k)
        best = np.take_along_axis(candidate, bp[None, :, :], axis=0)[0]
        dp = best + cT[k] + cS[k][None, :]
        if not np.isfinite(dp).any():
            raise RuntimeError(f"No feasible fingering at event {k}")
        backpointers.append(bp)

    p, a = np.unravel_index(int(np.argmin(dp)), dp.shape)
    total = float(dp[p, a])
    path = [int(a), int(p)]
    for k in range(K - 1, 1, -1):
        h = int(backpointers[k][p, a])
        path.append(h)
        p, a = h, p
    path.reverse()
    return DecodeResult(path, total)
