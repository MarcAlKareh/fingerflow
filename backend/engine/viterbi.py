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
            if not feas.any(axis=0).all():
                # Some current assignment is unreachable from every previous
                # one (a pathological held-note situation): allow a forced
                # release at a fixed penalty rather than failing.
                t = np.where(feas, t, t + FORCED_RELEASE_PENALTY)
            else:
                t = np.where(feas, t, INF)
        cT.append(t)
        if tensors.second[k] is not None:
            cQ.append(tensors.second[k].astype(np.float64) @ weights.second)
        else:
            cQ.append(None)
    return cS, cT, cQ


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
