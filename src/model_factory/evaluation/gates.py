from __future__ import annotations

from typing import Any, Callable


def _safe(metrics: dict[str, Any], *path: str, default: float = 0.0) -> float:
    current: Any = metrics
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    try:
        return float(current)
    except Exception:
        return default


PROMOTE_KILL_GATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "model_01_bias": lambda metrics: _safe(metrics, "val", "brier_score", default=1.0) < 0.25 and _safe(metrics, "val", "ece", default=1.0) < 0.05 and _safe(metrics, "val", "fee_adjusted_edge") > 0.005,
    "model_02_fair_resolution": lambda metrics: _safe(metrics, "val", "brier_score", default=1.0) < 0.24 and _safe(metrics, "val", "ece", default=1.0) < 0.04 and _safe(metrics, "val", "fee_adjusted_edge") > 0.008,
    "model_03_fill_probability": lambda metrics: _safe(metrics, "val", "brier_score", default=1.0) < 0.24 and _safe(metrics, "val", "ece", default=1.0) < 0.05,
    "model_04_adverse_selection": lambda metrics: _safe(metrics, "val", "adverse_flag_precision") > 0.55 and _safe(metrics, "val", "adverse_flag_recall") > 0.30,
    "model_05_closing_flip": lambda metrics: _safe(metrics, "val", "flip_precision") > 0.55 and _safe(metrics, "val", "brier_score", default=1.0) < 0.22,
    "model_06_mispricing": lambda metrics: _safe(metrics, "val", "edge_sign_accuracy") > 0.55 and _safe(metrics, "val", "mae", default=1.0) < 0.05,
    "model_07_microstructure_direction": lambda metrics: _safe(metrics, "val", "brier_score", default=1.0) < 0.25 and _safe(metrics, "val", "brier_score", default=1.0) < _safe(metrics, "baseline", "brier_score", default=1.0) - 0.005,
    "model_08_move_size": lambda metrics: _safe(metrics, "val", "mae", default=1.0) < 0.02,
    "model_09_regime_encoder": lambda metrics: float(metrics.get("downstream_lift", 0.0)) > 0.002,
    "model_10_action_policy": lambda metrics: _safe(metrics, "val", "replay_paper_net_pnl") > 0.0 and _safe(metrics, "val", "adverse_fill_rate", default=1.0) < 0.25,
}


def evaluate_gate(model_id: str, metrics: dict[str, Any]) -> str:
    passed = PROMOTE_KILL_GATES[model_id](metrics)
    return "PROMOTE" if passed else "WATCH"

