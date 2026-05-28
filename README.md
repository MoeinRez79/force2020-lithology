# Subsurface Semantic Segmentation — FORCE 2020 Lithology

> Predicting lithology from well logs, framed not as row-wise classification but
> as **cost-sensitive, uncertainty-aware semantic segmentation of partially-observed
> depth signals**, validated with geologically-honest spatial cross-validation.

Built on the open [FORCE 2020 Machine Predicted Lithology dataset](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
(offshore Norway wireline logs; ~1.17M depth samples across 98 wells, 18 logs).
It is deliberately designed to *not* look like the ~300 public gradient-boosting
notebooks, and every claim is backed by a spatial-CV number against a tuned
baseline.

## Why this is hard (and most solutions sidestep it)

1. **The metric is a penalty matrix, not accuracy.** Each misclassification has a
   different cost (`A[true, pred]`). Most people train on plain cross-entropy and
   only touch the matrix at scoring time. **This project minimises expected
   penalty directly**, so gradients point at the real objective
   (`src/lithology/penalty.py::ExpectedCostLoss`).
2. **Missingness is structured and informative.** Only `DEPTH_MD` and `GR` are
   guaranteed present; some logs (e.g. DTS) appear in under half the wells, and
   the test set shares that distribution. **The presence mask is fed to the model
   as extra channels** instead of imputing the gaps away.
3. **Depth is a sequence, not i.i.d. rows.** Lithology is vertically correlated,
   so **whole depth windows are segmented with a 1D U-Net**, combining thin-bed
   and formation-scale context.
4. **Validation leaks if you let it.** Neighbouring depths in one well are nearly
   identical, so random splits inflate scores that collapse on hidden wells.
   **Splitting is by well (GroupKFold).**

A fifth lever: the underused `FORCE_2020_LITHOFACIES_CONFIDENCE` column becomes a
**per-sample loss weight**, so the model trusts clean labels more than noisy ones.

## Results

5-fold **spatial** cross-validation (group-by-well, i.e. every validation well is
unseen in training). Metric is the official **mean penalty — lower is better**.
Run config: `--folds 5 --epochs 8 --window 256`, on a single GPU.

| Fold | U-Net (cost-sensitive) | LightGBM baseline | Winner |
|-----:|:----------------------:|:-----------------:|:------:|
| 1 | 0.817 | 0.888 | U-Net |
| 2 | 1.177 | 1.159 | LightGBM |
| 3 | 0.965 | 0.994 | U-Net |
| 4 | **0.786** | 1.033 | U-Net |
| 5 | 1.117 | 0.948 | LightGBM |
| **Mean ± std** | **0.972 ± 0.156** | 1.004 ± 0.091 | — |

**Honest interpretation.** The cost-sensitive U-Net edges out a strong, tuned
LightGBM baseline on the mean (0.972 vs 1.004) and wins 3 of 5 folds — but its
lead is smaller than its fold-to-fold variance. The result is best described as
*competitive with, and marginally ahead of, the baseline*, not a decisive win.

The high variance is itself the interesting finding: U-Net performance depends
strongly on **which** wells are held out (it excels on fold 4, trails on fold 5),
which points to genuine geological heterogeneity — some held-out wells contain
lithologies or log-availability patterns underrepresented in the training wells.
This is exactly the behaviour that random (non-spatial) cross-validation would
have hidden.

Training loss was still decreasing at epoch 8 on every fold, so the U-Net is
**under-trained**; longer schedules, per-well normalisation, or fold-ensembling
are the natural next steps to sharpen the margin. The headline takeaway is
methodological: under *honest* spatial validation, optimising the real penalty
metric with a sequence model is competitive with the field-standard approach.

### Ablation study

`ablation.py` re-runs the identical spatial CV while toggling one design choice at
a time — missingness mask, confidence weighting, and cost-sensitive loss vs plain
cross-entropy — and writes a comparison table to `results/ablation.md`. A positive
delta versus the full model means removing that component hurt, i.e. it earns its
place.

<!-- ABLATION TABLE: paste the table from results/ablation.md here once the run completes -->

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# smoke test on synthetic data (no download needed):
python train.py --synthetic --epochs 2 --folds 2 --window 128

# EDA figures (saved into figures/):
python notebooks/01_eda.py
```

For real results, download `train.zip` (unzip to `train.csv`) and
`penalty_matrix.npy` from the official
[FORCE repo](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)
(`lithology_competition/data/`) into a `data/` folder, then:

```bash
python train.py    --csv data/train.csv --penalty data/penalty_matrix.npy --folds 5 --epochs 8
python ablation.py --csv data/train.csv --penalty data/penalty_matrix.npy --folds 5 --epochs 6
```

A GPU is strongly recommended (minutes vs hours). The code auto-detects CUDA.

## Repository layout

```
src/lithology/
  constants.py   12-class vocabulary, log columns, confidence weights
  data.py        loader, synthetic generator, normaliser, missingness mask
  penalty.py     penalty matrix, official metric, cost-sensitive loss
  dataset.py     windowed sequence Dataset (per-depth labels + mask)
  model.py       missingness-aware 1D U-Net
  baseline.py    LightGBM comparison
  engine.py      shared training/eval engine (one loop, config-driven)
train.py         spatial-CV training: U-Net vs baseline
ablation.py      ablation study -> results/ablation.md
notebooks/01_eda.py   data-availability and lithology EDA
```

## Data & licensing

Download from the official competition repository. Labels are CC-BY-4.0; the well
logs are released under the Norwegian License for Open Government Data (NLOD) 2.0.
Any publication using the data must cite: *"Lithofacies data was provided by the
FORCE Machine Learning competition with well logs and seismic 2020"* (Bormann et
al.). The `data/` folder is gitignored — it is not redistributed here.

## What this project demonstrates

Reframed an open well-log benchmark as cost-sensitive 1D semantic segmentation: a
missingness-aware U-Net trained to minimise the competition's penalty matrix
directly, validated leave-wells-out, and benchmarked against a tuned LightGBM
baseline under identical folds. Tools: PyTorch, LightGBM, scikit-learn, pandas,
NumPy.
