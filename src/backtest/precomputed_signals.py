"""Batch-precompute signals for a whole week, then expose a fast SignalEngine.

The default SignalEngine calls 9 LGBM models per anchor (~30ms each on CPU
with single-thread caps). For a week of 7000 anchors that's ~210s per week,
which is too slow when running ~600 weeks across a 30-trial search.

We instead:
  1. Run each model in ONE batched .predict()/.predict_proba() call on the
     entire week's feature matrix (LGBM batches 7k rows in ~50ms).
  2. Append predicted columns (_pred_fair_value, _pred_p_severe, ...) to the df.
  3. Provide a PrecomputedSignalEngine that reads those columns instead of
     calling the models. Strategy code uses it transparently via duck typing.

This converts ~270 model calls per anchor (9 models * 30 weeks * sample size)
into 9 batched calls per week. Net speedup ~100x.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading.entities import MarketState, SignalSnapshot
from trading.signals import SignalEngine


# Mapping from SignalEngine internal model-key to the SignalSnapshot field it
# populates, and whether it's a proba (classifier) or scalar (regression) model.
_MODEL_TO_FIELD = [
    ("fair_value",       "fair_value",            "proba"),
    ("mispricing",       "mispricing_predicted",  "scalar"),
    ("fill_prob",        "fill_prob_bid",         "proba"),   # also used for fill_prob_ask
    ("defense_severe",   "p_severe_7c_3s",        "proba"),
    ("defense_moderate", "p_moderate_5c_5s",      "proba"),
    ("defense_extreme",  "p_extreme_10c_5s",      "proba"),
    ("direction",        "p_up_5s",               "proba"),
    ("closing_flip",     "p_closing_flip",        "proba"),
    ("adverse_selection","expected_markout_1s",   "scalar"),
    ("mkt_to_close",     "expected_markout_to_close", "scalar"),
]


def batch_precompute(df: pl.DataFrame, engine: SignalEngine,
                     pred_col_prefix: str = "_pred_") -> pl.DataFrame:
    """Run every loaded model once on the whole df; append _pred_* columns.

    Missing models / NaN features get filled with sentinel (None as null).
    """
    if len(df) == 0:
        return df

    new_cols: dict[str, np.ndarray] = {}

    for key, field, kind in _MODEL_TO_FIELD:
        model = engine.models.get(key)
        feat_names = engine.feature_names.get(key, [])
        out_name = pred_col_prefix + field
        if model is None or not feat_names:
            new_cols[out_name] = np.full(len(df), np.nan, dtype=np.float32)
            continue
        # Build X by selecting columns; fill missing or NaN or non-numeric with 0
        X = np.zeros((len(df), len(feat_names)), dtype=np.float32)
        for i, fname in enumerate(feat_names):
            if fname not in df.columns:
                continue
            series = df.get_column(fname)
            if not series.dtype.is_numeric():
                continue   # categorical / string column — treat as zero
            try:
                v = series.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                v = np.asarray(v, dtype=np.float32)
                v = np.where(np.isfinite(v), v, 0.0)
                X[:, i] = v
            except Exception:
                continue
        try:
            if kind == "proba" and hasattr(model, "predict_proba"):
                raw = model.predict_proba(X)[:, 1]
                cal = getattr(model, "_calibrator", None)
                if cal is not None:
                    raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
                new_cols[out_name] = np.asarray(raw, dtype=np.float32)
            else:
                new_cols[out_name] = np.asarray(model.predict(X), dtype=np.float32)
        except Exception:
            new_cols[out_name] = np.full(len(df), np.nan, dtype=np.float32)

    # Add an alias for fill_prob_ask = fill_prob_bid (single-model symmetric)
    if (pred_col_prefix + "fill_prob_bid") in new_cols:
        new_cols[pred_col_prefix + "fill_prob_ask"] = new_cols[pred_col_prefix + "fill_prob_bid"]

    return df.with_columns([
        pl.Series(name=name, values=arr) for name, arr in new_cols.items()
    ])


class PrecomputedSignalEngine:
    """Duck-typed SignalEngine that reads from precomputed _pred_* fields in a feature_dict.

    Strategy code calls `self.signals.predict(state, feature_dict, ...)`.
    The feature_dict carries the row's columns, including _pred_*; we just
    look them up and build a SignalSnapshot.
    """

    def __init__(self, pred_col_prefix: str = "_pred_"):
        self.prefix = pred_col_prefix

    def predict(self, market: MarketState, feature_dict: dict,
                event_features: Optional[dict] = None) -> SignalSnapshot:
        f = feature_dict

        def g(name: str) -> Optional[float]:
            v = f.get(self.prefix + name)
            if v is None:
                return None
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            if v != v:  # NaN
                return None
            return v

        return SignalSnapshot(
            market_id=market.market_id,
            ts_ns=market.book.ts_ns,
            fair_value=g("fair_value"),
            mispricing_predicted=g("mispricing_predicted"),
            fill_prob_bid=g("fill_prob_bid"),
            fill_prob_ask=g("fill_prob_ask"),
            p_severe_7c_3s=g("p_severe_7c_3s"),
            p_moderate_5c_5s=g("p_moderate_5c_5s"),
            p_extreme_10c_5s=g("p_extreme_10c_5s"),
            p_up_5s=g("p_up_5s"),
            p_closing_flip=g("p_closing_flip"),
            expected_markout_1s=g("expected_markout_1s"),
            expected_markout_to_close=g("expected_markout_to_close"),
        )
