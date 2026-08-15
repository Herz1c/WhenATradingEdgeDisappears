"""Fit the strategy's isotonic calibrator on the OOS calib days (04-22, 04-30)
and save it so the live bot scores EXACTLY like the validated backtest.
"""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa

ART = TOOLS.parent / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]

model, feats, _ = load_model(ART)
df = load_days(CALIB)
p = score_model_p(model, feats, df)
y = df["label_up"].to_numpy().astype(np.float64)
iso = fit_iso(p, y)
out = ART / "calibrator_strategy.pkl"
joblib.dump(iso, out)
print(f"fit isotonic on {CALIB} ({len(y):,} rows), saved -> {out}")
print(f"  sample map: p=0.20->{iso.predict([0.20])[0]:.3f}  0.40->{iso.predict([0.40])[0]:.3f}  "
      f"0.60->{iso.predict([0.60])[0]:.3f}")
