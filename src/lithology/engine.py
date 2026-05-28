"""One training routine, driven by a config, reused by train.py and ablation.py.

Keeping a single `train_eval_fold` means the ablations differ from the main run
by *configuration only* -- never by a second, subtly-different copy of the loop.
That is exactly the discipline that makes an ablation table trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from .data import build_features, Normalizer
from .dataset import WellWindowDataset
from .model import UNet1D
from .penalty import competition_score, ExpectedCostLoss
from .constants import WELL_COL


@dataclass
class RunConfig:
    epochs: int = 8
    batch: int = 32
    window: int = 256
    lr: float = 1e-3
    use_mask: bool = True          # feed presence mask as channels
    use_confidence: bool = True    # weight loss by label confidence
    cost_lambda: float = 1.0       # 0.0 -> plain cross-entropy ablation
    ce_lambda: float = 0.3


def train_eval_fold(tr_df, va_df, logs, A, cfg: RunConfig, device: str):
    """Train a U-Net on one fold, return (val_penalty, n_val_samples)."""
    norm = Normalizer().fit(build_features(tr_df, logs)[0])
    tr = WellWindowDataset(tr_df, norm, logs, cfg.window, cfg.window // 2, cfg.use_mask)
    va = WellWindowDataset(va_df, norm, logs, cfg.window, cfg.window, cfg.use_mask)
    tl = DataLoader(tr, batch_size=cfg.batch, shuffle=True)
    vl = DataLoader(va, batch_size=cfg.batch, shuffle=False)

    in_ch = (2 if cfg.use_mask else 1) * len(logs)
    model = UNet1D(in_ch=in_ch).to(device)
    loss_fn = ExpectedCostLoss(A, ce_lambda=cfg.ce_lambda,
                               cost_lambda=cfg.cost_lambda).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    for ep in range(cfg.epochs):
        model.train()
        running = 0.0
        for x, y, w in tl:
            x, y = x.to(device), y.to(device)
            w = w.to(device) if cfg.use_confidence else None
            opt.zero_grad()
            loss = loss_fn(model(x), y, w)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"    epoch {ep+1}/{cfg.epochs}  train_loss={running/len(tl):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y, _ in vl:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy().ravel())
            trues.append(y.numpy().ravel())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return competition_score(yt, yp, A), len(yt)


def cross_validate(df, logs, A, cfg: RunConfig, folds: int, device: str, quiet=False):
    """Spatial (group-by-well) CV. Returns array of per-fold val penalties."""
    groups = df[WELL_COL].to_numpy()
    gkf = GroupKFold(n_splits=folds)
    scores = []
    for k, (tr_i, va_i) in enumerate(gkf.split(df, groups=groups)):
        if not quiet:
            print(f"\n=== Fold {k+1}/{folds} "
                  f"(val wells: {sorted(set(groups[va_i]))}) ===")
        s, _ = train_eval_fold(df.iloc[tr_i].copy(), df.iloc[va_i].copy(),
                               logs, A, cfg, device)
        if not quiet:
            print(f"  -> U-Net penalty={s:.4f} (lower is better)")
        scores.append(s)
    return np.array(scores)
