"""An honest gradient-boosting baseline.

Strong, simple, and what most of the field used. We keep it so the deep model
has to *earn* its complexity by beating this under identical spatial CV. A
self-respecting result section reports both numbers, not just the shiny one.
Adds light signal-aware features (rolling mean/grad over depth) per well.
"""
from __future__ import annotations

import warnings
import numpy as np
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import pandas as pd
import lightgbm as lgb

from .constants import WELL_COL, IGNORE_INDEX, N_CLASSES
from .data import build_features


def _windowed_features(df, logs, win=15):
    v, m, _ = build_features(df, logs)
    feats = [v, m]
    g = df.groupby(WELL_COL).cumcount()  # placeholder to keep per-well grouping intent
    vdf = pd.DataFrame(v, columns=logs)
    vdf[WELL_COL] = df[WELL_COL].to_numpy()
    roll = vdf.groupby(WELL_COL)[logs].transform(
        lambda s: s.rolling(win, min_periods=1, center=True).mean())
    grad = vdf.groupby(WELL_COL)[logs].transform(lambda s: s.diff().fillna(0.0))
    feats += [roll.to_numpy(), grad.to_numpy()]
    X = np.nan_to_num(np.concatenate(feats, axis=1), nan=0.0)
    return X


def train_baseline(train_df, val_df, logs):
    Xtr = _windowed_features(train_df, logs)
    Xva = _windowed_features(val_df, logs)
    ytr = train_df["label"].to_numpy()
    yva = val_df["label"].to_numpy()
    keep = ytr != IGNORE_INDEX
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr[keep], ytr[keep])
    pred = model.predict(Xva)
    return pred.astype(int), yva.astype(int)
