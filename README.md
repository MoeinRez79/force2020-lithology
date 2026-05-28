# Subsurface Semantic Segmentation — FORCE 2020 Lithology

> Predicting lithology from well logs, framed not as row-wise classification but
> as **cost-sensitive, uncertainty-aware semantic segmentation of partially-observed
> depth signals**, validated with geologically-honest spatial cross-validation.

This is a portfolio project built on the open [FORCE 2020 Machine Predicted
Lithology dataset](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
(offshore Norway wireline logs, lithofacies labels released under CC-BY-4.0 /
NLOD-2.0). It is deliberately designed to *not* look like the ~300 public
gradient-boosting notebooks.

---

## Why this is hard (and most solutions sidestep it)

Four properties of this dataset are genuinely difficult. The standard approach
ignores three of them.

1. **The metric is a penalty matrix, not accuracy.** Each misclassification has
   a different cost (`A[true, pred]`). Confusing chalk with limestone is cheap;
   confusing basement with coal is expensive. Almost everyone trains on plain
   cross-entropy and only touches the matrix at scoring time. **We minimise
   expected penalty directly**, so gradients point at the real objective
   (`src/lithology/penalty.py::ExpectedCostLoss`).

2. **Missingness is structured and informative.** Only `DEPTH_MD` and `GR` are
   guaranteed present; whole logs are absent for whole wells, correlated with
   operator and era. Mean-imputing destroys that signal. **We feed the network
   the presence mask as extra channels** so it learns from *which* logs exist.

3. **Depth is a sequence, not i.i.d. rows.** Lithology is vertically correlated.
   Treating each depth independently throws away the strongest prior in the data.
   **We segment whole depth windows with a 1D U-Net**, combining thin-bed and
   formation-scale context the way a petrophysicist reads a log by eye.

4. **Validation leaks if you let it.** Neighbouring depths in one well are nearly
   identical, so random splits inflate CV scores that then collapse on the hidden
   wells. **We split by well with GroupKFold** (leave-one-well-out at the limit).
   This is the difference between an honest result and a leaderboard mirage.

A fifth, optional lever: the underused `FORCE_2020_LITHOFACIES_CONFIDENCE`
column becomes a **per-sample loss weight**, so the model trusts clean labels
more than noisy ones.

## The thesis in one line

> *Treat the borehole as a signal to be segmented, optimise the cost you're
> actually scored on, prove it with spatial validation, and only keep
> complexity that beats an honest LightGBM baseline.*

That last clause matters: every reported number compares the U-Net against a
strong gradient-boosting baseline under identical folds. Complexity has to earn
its place.

---

## Architecture

```
logs (N, C) ──► standardise (train-only) ──► concat presence mask ──► (2C, W) windows
                                                                          │
                                                          1D U-Net (encoder/skip/decoder)
                                                                          │
                                                       per-depth logits (n_classes, W)
                                                                          │
                                  ExpectedCostLoss  =  E[penalty] + λ·CE,  weighted by confidence
```

- **Model:** `UNet1D` — a segmentation U-Net transplanted to the depth axis.
  Skip connections fuse multi-scale context; input is logs ⊕ mask.
- **Loss:** `ExpectedCostLoss` — `Σ_j p_j · A[y, j]` plus a cross-entropy
  stabiliser, weighted by label confidence.
- **Baseline:** `LightGBM` on signal-aware features (rolling mean + depth
  gradient), same spatial folds.
- **Validation:** `GroupKFold` by `WELL`.

## Repository map

```
force2020-lithology/
├── README.md                     ← this plan
├── requirements.txt
├── train.py                      ← spatial-CV training: U-Net vs baseline
├── data/                         ← put train.csv + penalty_matrix.npy here
└── src/lithology/
    ├── constants.py              ← 12-class vocabulary, log columns, conf weights
    ├── data.py                   ← loader, synthetic generator, normaliser, masks
    ├── penalty.py                ← penalty matrix, official metric, cost-sensitive loss
    ├── dataset.py                ← windowed sequence Dataset (per-depth labels + mask)
    ├── model.py                  ← missingness-aware 1D U-Net
    └── baseline.py               ← LightGBM comparison
```

---

## Roadmap

**Phase 0 — Run the skeleton (today).** Smoke-test on synthetic data, no
download required:
```bash
pip install -r requirements.txt
python train.py --synthetic --epochs 2 --folds 2 --window 128
```
You should see per-fold U-Net vs LightGBM penalties and a spatial-CV summary.

**Phase 1 — Real data + faithful EDA (week 1).**
Download the competition CSVs and the official `penalty_matrix.npy` from the
[FORCE repo](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
(`lithology_competition/data/`) into `data/`. Replace the fallback penalty
matrix. Build a per-well log-availability heatmap and a depth-vs-lithology
panel — understanding the missingness pattern *is* the modelling work.
```bash
python train.py --csv data/train.csv --penalty data/penalty_matrix.npy --folds 5
```

**Phase 2 — Beat the baseline honestly (weeks 2–3).** Tune window/stride, the
`ce_lambda` blend, and class handling for rare lithologies (Tuff, Basement).
Report mean ± std penalty across folds for *both* models. Target: U-Net
consistently below LightGBM under leave-one-well-out.

**Phase 3 — Differentiators (weeks 3–4).** Add the confidence-weighting ablation
(on/off), and a missingness ablation (mask channels on/off) — these two tables
are what make the write-up publishable-looking.

**Phase 4 — Stretch (optional, weeks 5+).**
- **Spatial context:** add well `X_LOC/Y_LOC` as a coordinate channel, or build
  a graph over wells by geographic proximity and message-pass between them.
- **Calibrated uncertainty:** Monte-Carlo dropout to flag low-confidence depths
  for human review — directly useful in real interpretation workflows.
- **Transformer-over-windows** as a second sequence backbone to compare.
- **LAS ingestion:** swap the CSV loader for `lasio` to read raw `.LAS` files,
  and `segyio` if you extend toward seismic facies later.

## Evaluation protocol (non-negotiable)

- Metric: **mean penalty** from the official matrix (lower is better).
- Splitting: **GroupKFold by well**; never random rows.
- Always report U-Net **and** baseline, with fold-wise std.
- An ablation table (confidence weighting, missingness mask, cost-sensitive vs
  CE-only loss) is the deliverable, not a single leaderboard number.

## Data & licensing

Download from the official competition repository. Labels are CC-BY-4.0; the
well logs are released under the Norwegian License for Open Government Data
(NLOD) 2.0. Cite the FORCE 2020 organisers in any write-up.

