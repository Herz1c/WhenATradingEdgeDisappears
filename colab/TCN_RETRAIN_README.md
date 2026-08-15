# TCN retrain v2 — Colab run (TTC 15–150, seed sweep)

Goal: retrain the residual TCN on dataset v2 — a proper multi-day validation split, with
the TTC band widened to (15, 150] so it covers the 75–120 early slot with margin — and
measure seed variance. The bundle is produced by `tools/make_tcn_retrain_bundle.py`.

Recommended runtime: A100 or L4; a T4 is enough (batch 64, roughly 10–20 min per model).

## 1. Mount Drive and unpack

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
ZIP_PATH = "/content/drive/MyDrive/btc_tcn_retrain_v2_bundle.zip"
WORKDIR = "/content/tcn_retrain_v2"

import os, zipfile, shutil
if os.path.exists(WORKDIR):
    shutil.rmtree(WORKDIR)
os.makedirs(WORKDIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(WORKDIR)
os.chdir(WORKDIR)
print("cwd =", os.getcwd())
```

## 2. Dependencies and GPU check

```python
!pip -q install -r requirements_colab.txt
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
```

## 3. Seed sweep — baseline capacity c64/b7, TTC 15–150

```python
SEEDS = [11, 7, 23, 42, 101]
for seed in SEEDS:
    out = f"artifacts/tcn_v2_c64_b7_ttc15_150_seed{seed}"
    !python tools/train_episode_tcn.py \
        --dataset data/datasets/btc_5m_episodes_v2_200ms \
        --out {out} \
        --ttc-min 15 --ttc-max 150 \
        --channels 64 --blocks 7 \
        --epochs 30 --patience 6 \
        --batch-size 64 --seed {seed}
```

## 4. Capacity probe — c96/b8 (seed 11 only)

```python
!python tools/train_episode_tcn.py \
    --dataset data/datasets/btc_5m_episodes_v2_200ms \
    --out artifacts/tcn_v2_c96_b8_ttc15_150_seed11 \
    --ttc-min 15 --ttc-max 150 \
    --channels 96 --blocks 8 \
    --epochs 30 --patience 6 \
    --batch-size 64 --seed 11
```

## 5. Pack results back to Drive

```python
import shutil
shutil.make_archive("/content/drive/MyDrive/tcn_v2_results", "zip", "artifacts")
print("done -> MyDrive/tcn_v2_results.zip")
```

## 6. Back on the local machine

Download `tcn_v2_results.zip`, unpack it into `artifacts/`, and run:

```
py tools/backtest_locked_strategy.py --tcn-artifacts artifacts/tcn_v2_c64_b7_ttc15_150_seed11 --dataset data/datasets/btc_5m_episodes_v2_200ms --split test
```

The seed-variance evaluation runs locally afterwards. The metric that matters there is
the spread of policy PnL across seeds, not Brier score.

## Notes

- The `test.npz` in the bundle is the reporting split 06-28 → 07-02, which is still
  selection-era data. **The 07-03+ holdout is deliberately not in the bundle** — it is
  touched only at the final gate, locally.
- The trainer early-stops on the validation split (06-23 → 06-27, five days). Against
  v1's two-day validation split this is an order of magnitude more robust, both for
  calibration and for choosing the epoch.
- Downstream policy evaluation needs deltas across the *whole* episode; the
  `predictions_*.npz` files the trainer writes cover only the loss band, so the local
  engine recomputes the deltas itself.
