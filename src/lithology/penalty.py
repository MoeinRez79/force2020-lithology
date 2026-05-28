"""The penalty matrix is the heart of why this project is different.

FORCE 2020 was *not* scored on accuracy. It used a 12x12 penalty matrix A where
A[true, pred] is the cost of predicting `pred` when the truth is `true`.
Confusing chalk for limestone is cheap; confusing basement for coal is not.

Most public solutions trained on plain cross-entropy and only used A at the very
end to report a score. We instead minimise *expected penalty* directly, so the
training objective matches the metric. This module provides:
  * load_penalty_matrix : official matrix if present, else a documented fallback
  * competition_score   : the official mean-penalty metric (lower is better)
  * ExpectedCostLoss    : a differentiable expected-penalty loss for training
"""
from __future__ import annotations

import os
import numpy as np

from .constants import N_CLASSES, IGNORE_INDEX

try:  # torch is only needed for the loss, not for scoring
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def fallback_penalty_matrix() -> np.ndarray:
    """A geologically-motivated stand-in for smoke tests ONLY.

    Diagonal = 0. Off-diagonal defaults to 1.0, but confusions *within* a
    coarse lithology family (clastics / carbonates / evaporites) are discounted
    to 0.5 because they are geologically less wrong. Replace this with the
    official `penalty_matrix.npy` before reporting any leaderboard number.
    """
    families = {
        "clastic": [0, 1, 2, 3],          # sandstone, sst/shale, shale, marl
        "carbonate": [4, 5, 6],           # dolomite, limestone, chalk
        "evaporite": [7, 8],              # halite, anhydrite
        "other": [9, 10, 11],             # tuff, coal, basement
    }
    fam_of = {c: f for f, cs in families.items() for c in cs}
    A = np.ones((N_CLASSES, N_CLASSES), dtype=np.float64)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if i == j:
                A[i, j] = 0.0
            elif fam_of[i] == fam_of[j]:
                A[i, j] = 0.5
    return A


def load_penalty_matrix(path: str | None) -> np.ndarray:
    """Load the official penalty matrix from .npy, else return the fallback."""
    if path and os.path.exists(path):
        A = np.load(path)
        assert A.shape == (N_CLASSES, N_CLASSES), f"bad penalty shape {A.shape}"
        return A.astype(np.float64)
    print("[penalty] official matrix not found -> using FALLBACK (smoke-test only)")
    return fallback_penalty_matrix()


def competition_score(y_true: np.ndarray, y_pred: np.ndarray, A: np.ndarray) -> float:
    """Official metric: mean penalty over labelled samples (lower is better)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    valid = y_true != IGNORE_INDEX
    yt, yp = y_true[valid].astype(int), y_pred[valid].astype(int)
    return float(A[yt, yp].mean())


if _HAS_TORCH:

    class ExpectedCostLoss(nn.Module):
        """Differentiable expected penalty + cross-entropy stabiliser.

        For predicted probabilities p (softmax of logits) and true class y, the
        expected cost is  sum_j p_j * A[y, j].  Minimising it aligns gradients
        with the competition metric. We blend in a little CE for early-training
        stability and weight each sample by its label confidence.

        logits : (B, C, L)   per-depth class scores
        target : (B, L)      class indices, IGNORE_INDEX for padded/unlabelled
        weight : (B, L)      per-sample confidence weight (optional)
        """

        def __init__(self, penalty: np.ndarray, ce_lambda: float = 0.3,
                     cost_lambda: float = 1.0):
            super().__init__()
            self.register_buffer("A", torch.tensor(penalty, dtype=torch.float32))
            self.ce_lambda = ce_lambda
            self.cost_lambda = cost_lambda  # set 0.0 for a plain cross-entropy ablation

        def forward(self, logits, target, weight=None):
            B, C, L = logits.shape
            logits = logits.permute(0, 2, 1).reshape(-1, C)   # (B*L, C)
            target = target.reshape(-1)                        # (B*L,)
            mask = target != IGNORE_INDEX
            if mask.sum() == 0:
                return logits.sum() * 0.0
            logits, tgt = logits[mask], target[mask]
            probs = torch.softmax(logits, dim=1)               # (M, C)
            cost_rows = self.A[tgt]                             # (M, C) = A[y, :]
            exp_cost = (probs * cost_rows).sum(dim=1)           # (M,)
            ce = nn.functional.cross_entropy(logits, tgt, reduction="none")
            per_sample = self.cost_lambda * exp_cost + self.ce_lambda * ce
            if weight is not None:
                w = weight.reshape(-1)[mask]
                per_sample = per_sample * w
                return per_sample.sum() / (w.sum() + 1e-8)
            return per_sample.mean()
