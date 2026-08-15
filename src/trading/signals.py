"""SignalEngine — wraps all trained models and produces SignalSnapshots.

Loads cleaned artifacts from `artifacts_cleaned/` by default.  Each
predict_* method handles missing features / partial data gracefully and
returns None rather than raising.

For backtesting, this can be subclassed/mocked to return deterministic
signals from precomputed parquet outputs.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import joblib
import polars as pl

from .entities import SignalSnapshot, MarketState, MarketId


# Path conventions
ARTIFACT_PATHS_DEFAULT = {
    "fair_value":          "artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl",
    "mispricing":          "artifacts_cleaned/model_06_mispricing/dense_close/lightgbm/model.pkl",
    "fill_prob":           "artifacts_cleaned/model_03_fill_probability/lightgbm/model.pkl",
    "defense_severe":      "artifacts_cleaned/model_08c_maker_defense_v3_eventtime/severe_7c_3s/agg_plus_event/model.pkl",
    "defense_moderate":    "artifacts_cleaned/model_08c_maker_defense_v3_eventtime/moderate_5c_5s/agg_plus_event/model.pkl",
    "defense_extreme":     "artifacts_cleaned/model_08c_maker_defense_v3_eventtime/extreme_10c_5s/agg_plus_event/model.pkl",
    "direction":           "artifacts_cleaned/model_07_microstructure_direction/tabular/lightgbm/model.pkl",
    "closing_flip":        "artifacts_cleaned/model_05_closing_flip/dense_close/lightgbm/model.pkl",
    "adverse_selection":   "artifacts_cleaned/model_04_adverse_selection/lightgbm/model.pkl",
    "mkt_to_close":        "artifacts_cleaned/model_04b_markout_to_close/lightgbm/model.pkl",
}


class SignalEngine:
    """Loads all production models; computes a SignalSnapshot from a MarketState."""

    def __init__(self, artifact_root: str = "artifacts_cleaned",
                 paths: Optional[dict[str, str]] = None):
        self.artifact_root = artifact_root
        self.paths = paths or ARTIFACT_PATHS_DEFAULT
        self.models: dict[str, object] = {}
        self.feature_names: dict[str, list[str]] = {}
        self._load_all()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        for key, path in self.paths.items():
            p = Path(path)
            if not p.exists():
                print(f"  [SignalEngine] WARN missing {key}: {p}")
                self.models[key] = None
                continue
            try:
                self.models[key] = joblib.load(p)
            except Exception as e:
                print(f"  [SignalEngine] ERROR loading {key}: {e}")
                self.models[key] = None
                continue
            # Try to recover feature names from sibling JSON.
            # Different model families store this in different files:
            #   - framework LGBM:  feature_importance.json
            #   - 08c v1/v2/v3:    metrics.json["feature_importance"]
            #   - 08b:             results.json["feature_importances"] paired w/ config.features
            feat_names: list[str] = []
            fi_path = p.parent / "feature_importance.json"
            metrics_path = p.parent / "metrics.json"
            results_path = p.parent / "results.json"
            if fi_path.exists():
                try:
                    feat_names = list(json.loads(fi_path.read_text()).keys())
                except Exception:
                    pass
            if not feat_names and metrics_path.exists():
                try:
                    d = json.loads(metrics_path.read_text())
                    fi_raw = d.get("feature_importance", {})
                    if fi_raw:
                        feat_names = list(fi_raw.keys())
                except Exception:
                    pass
            if not feat_names and results_path.exists():
                try:
                    d = json.loads(results_path.read_text())
                    cfg_feats = d.get("config", {}).get("features") or d.get("features")
                    if cfg_feats:
                        feat_names = list(cfg_feats)
                except Exception:
                    pass
            if feat_names:
                self.feature_names[key] = feat_names
            print(f"  [SignalEngine] loaded {key} from {p}  "
                  f"({len(self.feature_names.get(key, []))} features)")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _predict_proba_safe(model, X: np.ndarray) -> Optional[float]:
        if model is None or X is None or X.size == 0:
            return None
        try:
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(X)[:, 1]
                cal = getattr(model, "_calibrator", None)
                if cal is not None:
                    return float(np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)[0])
                return float(raw[0])
            return float(model.predict(X)[0])
        except Exception:
            return None

    @staticmethod
    def _predict_safe(model, X: np.ndarray) -> Optional[float]:
        if model is None or X is None or X.size == 0:
            return None
        try:
            return float(model.predict(X)[0])
        except Exception:
            return None

    @staticmethod
    def _build_X(features: list[str], feature_dict: dict[str, float]) -> Optional[np.ndarray]:
        """Build a 1×N input matrix from a feature dict, in the order expected."""
        if not features:
            return None
        vals = []
        for f in features:
            v = feature_dict.get(f, 0.0)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                v = 0.0
            vals.append(float(v))
        return np.array([vals], dtype=np.float32)

    # ── Main entrypoint ──────────────────────────────────────────────────────

    def predict(self, market: MarketState,
                feature_dict: dict[str, float],
                event_features: Optional[dict[str, float]] = None,
                ) -> SignalSnapshot:
        """Compute a SignalSnapshot for a market.

        Parameters
        ----------
        market : MarketState
            Current market state (book, position, timing).
        feature_dict : dict
            All aggregate features (anchor_source, event_count_*, spread_*,
            imbalance_*, mid_return_*, etc.).  Anything missing is zero-filled.
        event_features : dict, optional
            evt_* features computed from the L2 event array (for 08c v3
            agg_plus_event models).  If None, defense uses agg_only signals.
        """
        feature_dict = dict(feature_dict or {})
        if event_features:
            feature_dict.update(event_features)

        # Fair value (model 02)
        X_fv = self._build_X(self.feature_names.get("fair_value", []), feature_dict)
        fair_value = self._predict_proba_safe(self.models.get("fair_value"), X_fv)

        # Mispricing (model 06) — regression
        X_mp = self._build_X(self.feature_names.get("mispricing", []), feature_dict)
        mispricing_predicted = self._predict_safe(self.models.get("mispricing"), X_mp)

        # Fill prob (model 03) — need bid-specific and ask-specific feature dicts
        # For v1 we use the same prediction for both sides as a rough proxy;
        # a future refinement is to score the model separately conditioned on
        # the order price.
        X_fp = self._build_X(self.feature_names.get("fill_prob", []), feature_dict)
        fill_p = self._predict_proba_safe(self.models.get("fill_prob"), X_fp)
        fill_prob_bid = fill_prob_ask = fill_p

        # Defense (08c v3 agg_plus_event)
        X_def = self._build_X(self.feature_names.get("defense_severe", []), feature_dict)
        p_severe = self._predict_proba_safe(self.models.get("defense_severe"), X_def)
        p_moderate = self._predict_proba_safe(self.models.get("defense_moderate"), X_def)
        p_extreme = self._predict_proba_safe(self.models.get("defense_extreme"), X_def)

        # Direction (07)
        X_dir = self._build_X(self.feature_names.get("direction", []), feature_dict)
        p_up = self._predict_proba_safe(self.models.get("direction"), X_dir)

        # Closing flip (05)
        X_cf = self._build_X(self.feature_names.get("closing_flip", []), feature_dict)
        p_flip = self._predict_proba_safe(self.models.get("closing_flip"), X_cf)

        # Adverse selection (04) — regression
        X_as = self._build_X(self.feature_names.get("adverse_selection", []), feature_dict)
        e_markout = self._predict_safe(self.models.get("adverse_selection"), X_as)

        # Markout to close (04b) — regression
        X_mc = self._build_X(self.feature_names.get("mkt_to_close", []), feature_dict)
        e_markout_close = self._predict_safe(self.models.get("mkt_to_close"), X_mc)

        return SignalSnapshot(
            market_id=market.market_id,
            ts_ns=market.book.ts_ns,
            fair_value=fair_value,
            mispricing_predicted=mispricing_predicted,
            fill_prob_bid=fill_prob_bid,
            fill_prob_ask=fill_prob_ask,
            p_severe_7c_3s=p_severe,
            p_moderate_5c_5s=p_moderate,
            p_extreme_10c_5s=p_extreme,
            p_up_5s=p_up,
            p_closing_flip=p_flip,
            expected_markout_1s=e_markout,
            expected_markout_to_close=e_markout_close,
        )
