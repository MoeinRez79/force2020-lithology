"""Train and evaluate under geologically-honest spatial cross-validation.

Smoke test on synthetic data (no download needed):
    python train.py --synthetic --epochs 2 --folds 2 --window 128

Real data once downloaded (see README):
    python train.py --csv data/train.csv --penalty data/penalty_matrix.npy --folds 5

Most important methodological choice: we split by WELL with GroupKFold. Random
row splits leak (neighbouring depths in a well are near-identical), so naive CV
looks great then collapses on the hidden wells. This protects against that.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.lithology.data import load_force_csv, make_synthetic, build_features
from src.lithology.penalty import load_penalty_matrix, competition_score
from src.lithology.baseline import train_baseline
from src.lithology.engine import RunConfig, train_eval_fold
from src.lithology.constants import WELL_COL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--penalty", default="data/penalty_matrix.npy")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    df = make_synthetic() if (args.synthetic or args.csv is None) else load_force_csv(args.csv)
    logs = build_features(df)[2]
    A = load_penalty_matrix(args.penalty)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(epochs=args.epochs, batch=args.batch, window=args.window, lr=args.lr)
    print(f"[setup] wells={df[WELL_COL].nunique()} samples={len(df)} "
          f"logs={len(logs)} device={device}")

    groups = df[WELL_COL].to_numpy()
    gkf = GroupKFold(n_splits=args.folds)
    deep, base = [], []
    for k, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n=== Fold {k+1}/{args.folds} "
              f"(val wells: {sorted(set(groups[va_i]))}) ===")
        tr_df, va_df = df.iloc[tr_i].copy(), df.iloc[va_i].copy()
        d, _ = train_eval_fold(tr_df, va_df, logs, A, cfg, device)
        bp, bt = train_baseline(tr_df, va_df, logs)
        b = competition_score(bt, bp, A)
        print(f"  -> U-Net penalty={d:.4f}   LightGBM penalty={b:.4f}   (lower is better)")
        deep.append(d); base.append(b)

    print("\n========== SPATIAL CV SUMMARY ==========")
    print(f"U-Net    mean penalty: {np.mean(deep):.4f} +/- {np.std(deep):.4f}")
    print(f"LightGBM mean penalty: {np.mean(base):.4f} +/- {np.std(base):.4f}")


if __name__ == "__main__":
    main()
