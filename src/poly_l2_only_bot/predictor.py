"""Thin wrapper around the LightGBM v2 model. Designed for ONE-row inference
in the hot path with minimum overhead.

Loads once at startup. Pre-allocates the numpy buffer. The hot path just
fills the buffer and calls predict_one().
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np


_LOGIT_CLIP = (1e-4, 1 - 1e-4)


class V2Predictor:
    """LightGBM residual-head predictor. Apply isotonic calibration on top.

    Hot path:
        feats_dict = state_to_features(state, ts_ns)
        for i, c in enumerate(predictor.feature_order):
            predictor.buf[0, i] = feats_dict.get(c, 0.0)
        p_up = predictor.predict_one()
    """

    __slots__ = (
        "model", "calibrator", "calibrator_path", "feature_order", "best_iter",
        "_implied_p_up_idx", "buf",
    )

    def __init__(self, artifact_dir: Path) -> None:
        self.model = joblib.load(artifact_dir / "model.pkl")
        self.feature_order: List[str] = json.loads(
            (artifact_dir / "features.json").read_text()
        )
        # Prefer the STRATEGY calibrator (isotonic fit on the OOS calib days
        # 04-22/04-30) so live scoring matches the validated backtest exactly.
        # Fall back to the shipped training-time calibrator.
        strat_cal = artifact_dir / "calibrator_strategy.pkl"
        ship_cal = artifact_dir / "calibrator.pkl"
        cal_path = strat_cal if strat_cal.exists() else ship_cal
        self.calibrator = joblib.load(cal_path) if cal_path.exists() else None
        self.calibrator_path = str(cal_path) if self.calibrator is not None else None
        self.best_iter = int(self.model.best_iteration)
        self._implied_p_up_idx = self.feature_order.index("implied_p_up")
        # Pre-allocated feature buffer — reused on every predict.
        self.buf = np.zeros((1, len(self.feature_order)), dtype=np.float32)
        # Sanity check: model must have the init_score residual-head flag.
        if not bool(getattr(self.model, "_uses_init_score", False)):
            raise RuntimeError(
                f"Model at {artifact_dir} is not the v2 residual head "
                "(missing _uses_init_score flag). This bot only supports v2."
            )

    def fill_features(self, feats_dict: Dict[str, float]) -> None:
        """Fill self.buf from a feature dict. ~600 ns per call."""
        order = self.feature_order
        buf = self.buf
        for i, c in enumerate(order):
            v = feats_dict.get(c, 0.0)
            buf[0, i] = v

    def predict_one(self) -> float:
        """Run the v2 residual-head predict on self.buf. ~1-3 ms (LightGBM
        single-row inference overhead)."""
        mp_val = float(self.buf[0, self._implied_p_up_idx])
        mp_clip = mp_val if mp_val > _LOGIT_CLIP[0] else _LOGIT_CLIP[0]
        if mp_clip > _LOGIT_CLIP[1]:
            mp_clip = _LOGIT_CLIP[1]
        init_z = math.log(mp_clip / (1.0 - mp_clip))
        raw = float(self.model.predict(self.buf, raw_score=True,
                                        num_iteration=self.best_iter)[0])
        p = 1.0 / (1.0 + math.exp(-(init_z + raw)))
        if self.calibrator is not None:
            p = float(self.calibrator.transform([p])[0])
        return p
