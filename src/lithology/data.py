"""Loading and preparing FORCE 2020 well-log data.

Real data: download the competition CSVs (see README) and point the config at
`train.csv`. The loader maps lithology codes to dense indices, builds a
*missingness mask*, and reports the spatial grouping (by well) used for honest
cross-validation.

No data yet? `make_synthetic(...)` produces a schema-faithful dataset with
realistic structured missingness so the whole pipeline runs end-to-end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    LOG_COLUMNS, DEPTH_COL, WELL_COL, X_COL, Y_COL, LABEL_COL, CONF_COL,
    CODE_TO_IDX, IGNORE_INDEX, N_CLASSES, CONFIDENCE_WEIGHTS, DEFAULT_CONF_WEIGHT,
)


def load_force_csv(path: str) -> pd.DataFrame:
    """Read a competition CSV and map lithology codes -> dense 0..11 labels."""
    df = pd.read_csv(path, sep=";") if _is_semicolon(path) else pd.read_csv(path)
    if LABEL_COL in df.columns:
        df["label"] = df[LABEL_COL].map(CODE_TO_IDX).fillna(IGNORE_INDEX).astype(int)
    else:
        df["label"] = IGNORE_INDEX
    df = df.sort_values([WELL_COL, DEPTH_COL]).reset_index(drop=True)
    return df


def _is_semicolon(path: str) -> bool:
    with open(path, "r") as f:
        head = f.readline()
    return head.count(";") > head.count(",")


def confidence_weights(df: pd.DataFrame) -> np.ndarray:
    if CONF_COL in df.columns:
        return df[CONF_COL].map(CONFIDENCE_WEIGHTS).fillna(DEFAULT_CONF_WEIGHT).to_numpy()
    return np.full(len(df), DEFAULT_CONF_WEIGHT)


def build_features(df: pd.DataFrame, logs: list[str] | None = None):
    """Return (values, mask, present_logs).

    values : (N, C) float array, NaNs left in place (filled later, post-norm)
    mask   : (N, C) 1.0 where the log is present, 0.0 where missing
    """
    logs = logs or [c for c in LOG_COLUMNS if c in df.columns]
    values = df[logs].to_numpy(dtype=np.float64)
    mask = (~np.isnan(values)).astype(np.float64)
    return values, mask, logs


class Normalizer:
    """Per-channel standardisation fit on TRAIN ONLY (no leakage)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, values: np.ndarray) -> "Normalizer":
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0) + 1e-6
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.mean_) / self.std_
        return np.nan_to_num(z, nan=0.0)  # missing -> 0 == channel mean


def make_synthetic(n_wells: int = 12, seed: int = 0) -> pd.DataFrame:
    """Schema-faithful synthetic data with structured missingness + spatial drift.

    Each "well" is a depth-ordered sequence whose lithology is a smooth random
    walk (geology is vertically correlated). Logs are crude functions of
    lithology + noise, and whole logs are randomly absent per well -- exactly
    the failure mode that breaks naive imputation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        n = int(rng.integers(800, 2500))
        depth0 = rng.uniform(500, 3000)
        depth = depth0 + np.arange(n) * 0.15
        # vertically-correlated lithology via a clipped random walk
        walk = np.cumsum(rng.normal(0, 0.25, size=n))
        lith = (np.clip((walk - walk.min()) / (np.ptp(walk) + 1e-9), 0, 1)
                * (N_CLASSES - 1)).round().astype(int)
        x, y = rng.uniform(4.3e5, 5.5e5), rng.uniform(6.4e6, 6.8e6)
        present = {c: rng.random() > 0.35 for c in LOG_COLUMNS}
        present["GR"] = True  # guaranteed
        rec = {WELL_COL: f"SYN-{w:02d}", DEPTH_COL: depth,
               X_COL: x, Y_COL: y}
        for c in LOG_COLUMNS:
            base = lith * rng.uniform(0.5, 1.5) + rng.normal(0, 0.4, size=n)
            col = base + rng.normal(0, 0.6, size=n)
            if not present[c]:
                col = np.full(n, np.nan)
            rec[c] = col
        from .constants import IDX_TO_CODE
        rec[LABEL_COL] = np.vectorize(IDX_TO_CODE.get)(lith)
        rec[CONF_COL] = rng.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1])
        rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    df["label"] = df[LABEL_COL].map(CODE_TO_IDX).fillna(IGNORE_INDEX).astype(int)
    return df.sort_values([WELL_COL, DEPTH_COL]).reset_index(drop=True)
