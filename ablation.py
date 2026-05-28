"""Ablation study -- the table that proves each design choice earns its place.

We re-run identical spatial cross-validation while toggling one idea at a time:
  * Full          : missingness mask + confidence weighting + cost-sensitive loss
  * -- mask       : drop the presence-mask channels
  * -- confidence : ignore the label-confidence weights
  * -- cost (CE)  : train on plain cross-entropy instead of expected penalty

Lower penalty is better. If an ablation is NOT clearly worse than Full, that
component is not pulling its weight and you should say so honestly.

Run:
    python ablation.py --synthetic --epochs 3 --folds 3
    python ablation.py --csv data/train.csv --penalty data/penalty_matrix.npy --folds 5
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import torch

from src.lithology.data import load_force_csv, make_synthetic, build_features
from src.lithology.penalty import load_penalty_matrix
from src.lithology.engine import RunConfig, cross_validate
from src.lithology.constants import WELL_COL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--penalty", default="data/penalty_matrix.npy")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    df = make_synthetic() if (args.synthetic or args.csv is None) else load_force_csv(args.csv)
    logs = build_features(df)[2]
    A = load_penalty_matrix(args.penalty)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = dict(epochs=args.epochs, window=args.window, batch=args.batch)
    print(f"[setup] wells={df[WELL_COL].nunique()} samples={len(df)} "
          f"logs={len(logs)} device={device}")

    variants = {
        "Full (mask+conf+cost)": RunConfig(**base),
        "-- mask":               RunConfig(**base, use_mask=False),
        "-- confidence":         RunConfig(**base, use_confidence=False),
        "-- cost (plain CE)":    RunConfig(**base, cost_lambda=0.0, ce_lambda=1.0),
    }

    rows = []
    for name, cfg in variants.items():
        print(f"\n########## {name} ##########")
        scores = cross_validate(df, logs, A, cfg, args.folds, device, quiet=True)
        rows.append({"variant": name,
                     "mean_penalty": round(float(scores.mean()), 4),
                     "std": round(float(scores.std()), 4),
                     "folds": args.folds})
        print(f"  {name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    res = pd.DataFrame(rows).sort_values("mean_penalty").reset_index(drop=True)
    full = res.loc[res.variant.str.startswith("Full"), "mean_penalty"].iloc[0]
    res["delta_vs_full"] = (res["mean_penalty"] - full).round(4)

    os.makedirs(args.outdir, exist_ok=True)
    res.to_csv(os.path.join(args.outdir, "ablation.csv"), index=False)
    with open(os.path.join(args.outdir, "ablation.md"), "w") as f:
        f.write("# Ablation: spatial-CV mean penalty (lower is better)\n\n")
        f.write(res.to_markdown(index=False))
        f.write("\n\n*Positive `delta_vs_full` means removing that component "
                "made the model worse, i.e. the component helps.*\n")

    print("\n========== ABLATION TABLE ==========")
    print(res.to_string(index=False))
    print(f"\nSaved -> {args.outdir}/ablation.csv  and  {args.outdir}/ablation.md")


if __name__ == "__main__":
    main()
