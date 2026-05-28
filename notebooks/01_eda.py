"""Exploratory Data Analysis for FORCE 2020 (script version).

Run from the repo root:   python notebooks/01_eda.py
Saves four figures into figures/. Uses the real CSV if data/train.csv exists,
otherwise schema-faithful synthetic data. The point is to understand the
*missingness structure*, which is the core modelling problem in this dataset.
"""
from __future__ import annotations

import sys, os
ROOT = next(p for p in [os.getcwd(), os.path.abspath("..")]
            if os.path.isdir(os.path.join(p, "src")))
sys.path.insert(0, ROOT)
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
DATA_CSV = os.path.join(ROOT, "data", "train.csv")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt

from src.lithology.data import load_force_csv, make_synthetic, build_features
from src.lithology.constants import IDX_TO_NAME, WELL_COL, DEPTH_COL

df = load_force_csv(DATA_CSV) if os.path.exists(DATA_CSV) else make_synthetic()
values, mask, logs = build_features(df)
print(f"{df[WELL_COL].nunique()} wells | {len(df):,} samples | {len(logs)} logs")


def save(fig, name):
    path = os.path.join(ROOT, "figures", name)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    print("saved", path)


# 1. Log availability per well
mdf = pd.DataFrame(mask, columns=logs); mdf[WELL_COL] = df[WELL_COL].to_numpy()
avail = mdf.groupby(WELL_COL)[logs].mean()
fig, ax = plt.subplots(figsize=(12, max(4, 0.32 * len(avail))))
im = ax.imshow(avail.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(logs))); ax.set_xticklabels(logs, rotation=90)
ax.set_yticks(range(len(avail))); ax.set_yticklabels(avail.index, fontsize=7)
ax.set_title("Log availability per well (1 = always present)")
fig.colorbar(im, ax=ax, label="fraction present")
save(fig, "log_availability.png")

# 2. Overall completeness per log
overall = pd.Series(mask.mean(0), index=logs).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(overall.index, overall.values, color="#3b6ea5")
ax.set_xlabel("fraction of all samples present"); ax.set_xlim(0, 1)
ax.set_title("Overall log completeness")
save(fig, "log_completeness.png")

# 3. Lithology class balance
counts = df.loc[df.label >= 0, "label"].value_counts().sort_index()
names = [IDX_TO_NAME[i] for i in counts.index]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(names, counts.values, color="#a5683b")
ax.set_ylabel("samples"); ax.set_yscale("log")
ax.set_title("Lithology class distribution (log scale)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
save(fig, "class_balance.png")

# 4. One well as a petrophysical log plot
well = df[WELL_COL].value_counts().index[0]
w = df[df[WELL_COL] == well].sort_values(DEPTH_COL)
tracks = [t for t in ["GR", "RHOB", "NPHI", "DTC"] if t in w and w[t].notna().any()][:3]
fig, axes = plt.subplots(1, len(tracks) + 1, figsize=(2.4 * (len(tracks) + 1), 8), sharey=True)
for ax, t in zip(axes, tracks):
    ax.plot(w[t], w[DEPTH_COL], lw=0.5, color="#2a2a2a")
    ax.set_xlabel(t); ax.grid(alpha=.3)
axes[0].set_ylabel("Depth (m)"); axes[0].invert_yaxis()
litho = w["label"].to_numpy().reshape(-1, 1)
axes[-1].imshow(litho, aspect="auto", cmap="tab20", vmin=0, vmax=11,
                extent=[0, 1, w[DEPTH_COL].max(), w[DEPTH_COL].min()])
axes[-1].set_xlabel("lithology"); axes[-1].set_xticks([])
fig.suptitle(f"Well {well}: logs + interpreted lithology")
save(fig, "well_track.png")

print("\nTakeaways: missingness is per-well/per-log (mask is informative), "
      "classes are imbalanced (penalty matrix matters on rare ones), "
      "lithology is vertically smooth (sequence framing + spatial CV).")
