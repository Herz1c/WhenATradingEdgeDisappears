"""
Baseline predictors for Model 1 gate evaluation.

Each returns (y_prob, name) given a DataFrame and the train base rate.
Used to verify the trained model beats simple heuristics.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from model_factory.evaluation.metrics import compute_brier_score, compute_ece, compute_log_loss


def evaluate_baselines(
    frame: pl.DataFrame,
    target_col: str,
    train_base_rate: float,
) -> dict[str, dict[str, float]]:
    """Compute all baseline metrics for a given split frame."""
    y_true = frame[target_col].to_numpy().astype(int)
    n = len(y_true)
    results: dict[str, dict[str, float]] = {}

    def _score(name: str, probs: np.ndarray) -> None:
        p = np.clip(probs, 1e-6, 1 - 1e-6)
        results[name] = {
            "log_loss": compute_log_loss(y_true, p),
            "brier_score": compute_brier_score(y_true, p),
            "ece": compute_ece(y_true, p),
            "accuracy": float((np.round(p) == y_true).mean()),
        }

    # 1. Constant base rate from training set
    _score("constant_base_rate", np.full(n, train_base_rate))

    # 2. Always predict UP (optimistic upper bound on coverage)
    _score("always_up", np.full(n, 0.99))

    # 3. Streak-based: resolved_up_ratio_last_12
    if "resolved_up_ratio_last_12" in frame.columns:
        p = frame["resolved_up_ratio_last_12"].fill_null(train_base_rate).to_numpy()
        _score("streak_ratio_last_12", p.astype(float))

    # 4. Delta-to-strike heuristic (sigmoid of scaled delta)
    if "delta_to_strike" in frame.columns:
        delta = frame["delta_to_strike"].fill_null(0.0).to_numpy().astype(float)
        p_delta = 1.0 / (1.0 + np.exp(-delta * 0.5))
        _score("delta_to_strike_sigmoid", p_delta)

    # 5. Sign of BTC 60s return as direction proxy
    if "btc_return_60s" in frame.columns:
        ret = frame["btc_return_60s"].fill_null(0.0).to_numpy().astype(float)
        # Positive return → UP more likely; scale modestly
        p_ret = 1.0 / (1.0 + np.exp(-ret * 50.0))
        _score("btc_return_60s_sigmoid", p_ret)

    return results


def gate_check(
    model_log_loss: float,
    baselines: dict[str, dict[str, float]],
    gate_name: str = "delta_to_strike_sigmoid",
) -> dict[str, Any]:
    """Return pass/fail for the must_beat_delta_to_strike_heuristic gate."""
    if gate_name not in baselines:
        return {"pass": None, "reason": f"gate baseline '{gate_name}' not available"}
    bl_ll = baselines[gate_name]["log_loss"]
    delta = model_log_loss - bl_ll
    passed = model_log_loss < bl_ll
    return {
        "pass": passed,
        "model_log_loss": model_log_loss,
        "baseline_log_loss": bl_ll,
        "delta": delta,
        "baseline": gate_name,
        "reason": "PASS" if passed else f"Model log_loss {model_log_loss:.5f} >= baseline {bl_ll:.5f}",
    }
