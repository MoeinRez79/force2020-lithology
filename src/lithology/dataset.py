"""Turn each well into overlapping fixed-length depth windows.

The model labels every depth sample in a window (semantic segmentation of the
borehole), so adjacent-depth correlation is exploited instead of thrown away.
Each window carries its channel-presence mask and per-sample confidence weight.
Short tails are right-padded; padded positions get IGNORE_INDEX and mask 0.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import WELL_COL, IGNORE_INDEX
from .data import build_features, confidence_weights


class WellWindowDataset(Dataset):
    def __init__(self, df, normalizer, logs, window: int = 256, stride: int = 128,
                 use_mask: bool = True):
        self.window, self.stride = window, stride
        self.logs = logs
        self.use_mask = use_mask
        values, mask, _ = build_features(df, logs)
        values = normalizer.transform(values)            # (N, C), missing -> 0
        labels = df["label"].to_numpy().astype(np.int64)
        conf = confidence_weights(df).astype(np.float32)
        wells = df[WELL_COL].to_numpy()

        self.samples = []  # (values, mask, labels, conf) per window
        for w in np.unique(wells):
            idx = np.where(wells == w)[0]
            v, m, y, c = values[idx], mask[idx], labels[idx], conf[idx]
            n = len(idx)
            starts = list(range(0, max(1, n - window + 1), stride))
            if starts[-1] + window < n:
                starts.append(n - window)
            for s in starts:
                e = min(s + window, n)
                pad = window - (e - s)
                vv = np.pad(v[s:e], ((0, pad), (0, 0)))
                mm = np.pad(m[s:e], ((0, pad), (0, 0)))
                yy = np.pad(y[s:e], (0, pad), constant_values=IGNORE_INDEX)
                cc = np.pad(c[s:e], (0, pad))
                self.samples.append((vv, mm, yy, cc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        v, m, y, c = self.samples[i]
        # channel-first; optionally concat presence mask -> missingness-aware input
        if self.use_mask:
            x = np.concatenate([v.T, m.T], axis=0)       # (2C, W)
        else:
            x = v.T                                       # (C, W), mask ablated
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(c, dtype=torch.float32))
