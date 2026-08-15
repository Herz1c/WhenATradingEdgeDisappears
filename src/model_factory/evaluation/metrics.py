from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = _arr(y_true)
    p = np.clip(_arr(y_prob), 1e-15, 1.0 - 1e-15)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = _arr(y_true)
    p = _arr(y_prob)
    return float(np.mean((p - y) ** 2))


def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    y = _arr(y_true)
    p = _arr(y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frac_pos = []
    mean_pred = []
    sizes = []
    for index in range(n_bins):
        lo, hi = edges[index], edges[index + 1]
        mask = (p >= lo) & (p <= hi if index == n_bins - 1 else p < hi)
        sizes.append(int(mask.sum()))
        if mask.any():
            frac_pos.append(float(y[mask].mean()))
            mean_pred.append(float(p[mask].mean()))
        else:
            frac_pos.append(None)
            mean_pred.append(None)
    return {"fraction_positive": frac_pos, "mean_predicted": mean_pred, "bin_sizes": sizes}


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    curve = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    total = max(sum(curve["bin_sizes"]), 1)
    ece = 0.0
    for frac, pred, size in zip(curve["fraction_positive"], curve["mean_predicted"], curve["bin_sizes"], strict=False):
        if frac is None or pred is None:
            continue
        ece += (size / total) * abs(float(frac) - float(pred))
    return float(ece)


def compute_implied_edge(y_prob: np.ndarray, market_mid: np.ndarray) -> float:
    return float(np.nanmean(_arr(y_prob) - _arr(market_mid)))


def compute_edge_by_time_to_close(y_true: np.ndarray, y_prob: np.ndarray, t_to_close_s: np.ndarray, buckets: list = [0, 10, 30, 60, 120, 300]) -> dict:
    y, p, t = _arr(y_true), _arr(y_prob), _arr(t_to_close_s)
    result = {}
    for lo, hi in zip(buckets[:-1], buckets[1:], strict=False):
        mask = (t >= lo) & (t < hi)
        key = f"{lo}_{hi}"
        result[key] = {"rows": int(mask.sum())}
        if mask.any():
            result[key].update({"log_loss": compute_log_loss(y[mask], p[mask]), "brier_score": compute_brier_score(y[mask], p[mask])})
    return result


def compute_edge_by_regime(y_true: np.ndarray, y_prob: np.ndarray, regime_label: np.ndarray) -> dict:
    labels = np.asarray(regime_label)
    return {str(label): compute_brier_score(_arr(y_true)[labels == label], _arr(y_prob)[labels == label]) for label in sorted(set(labels))}


def compute_edge_by_session(y_true: np.ndarray, y_prob: np.ndarray, hour_utc: np.ndarray) -> dict:
    # BTC trades 24/7 — these are UTC hour buckets, not equity sessions.
    # Labels reflect Polymarket activity levels, not exchange sessions.
    hours = np.asarray(hour_utc, dtype=int)
    buckets = {
        "utc_00_08": (hours >= 0) & (hours < 8),   # thin Polymarket books
        "utc_08_16": (hours >= 8) & (hours < 16),   # EU hours, moderate activity
        "utc_16_24": (hours >= 16) & (hours < 24),  # US hours, peak activity
    }
    return {name: compute_brier_score(_arr(y_true)[mask], _arr(y_prob)[mask]) for name, mask in buckets.items() if mask.any()}


def compute_fee_adjusted_edge(y_prob: np.ndarray, market_mid: np.ndarray, fee_rate: float = 0.018) -> float:
    return float(compute_implied_edge(y_prob, market_mid) - fee_rate)


def compute_fill_adjusted_edge(y_prob: np.ndarray, fill_prob: np.ndarray, market_mid: np.ndarray, fee_rate: float = 0.0, rebate_rate: float = 0.20) -> float:
    edge = _arr(y_prob) - _arr(market_mid) - float(fee_rate) + float(rebate_rate) * float(fee_rate)
    return float(np.nanmean(edge * _arr(fill_prob)))


def compute_markout_stats(markout_1s: np.ndarray, markout_3s: np.ndarray, markout_5s: np.ndarray, markout_to_close: np.ndarray) -> dict:
    result = {}
    for name, values in {"1s": markout_1s, "3s": markout_3s, "5s": markout_5s, "to_close": markout_to_close}.items():
        arr = _arr(values)
        result[name] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)), "p5": float(np.nanpercentile(arr, 5)), "p95": float(np.nanpercentile(arr, 95))}
    return result


def compute_pnl_stats(gross_pnl: np.ndarray, net_pnl: np.ndarray, fill_mask: np.ndarray) -> dict:
    mask = np.asarray(fill_mask, dtype=bool)
    gross = _arr(gross_pnl)
    net = _arr(net_pnl)
    return {"gross_total": float(np.nansum(gross)), "net_total": float(np.nansum(net)), "filled_rows": int(mask.sum()), "net_per_fill": float(np.nanmean(net[mask])) if mask.any() else 0.0}


def compute_inventory_stress(position_series: np.ndarray, resolution_series: np.ndarray, max_position: float = 1000.0) -> dict:
    pos = _arr(position_series)
    res = _arr(resolution_series)
    exposure = np.abs(pos)
    return {"max_abs_position": float(np.nanmax(exposure)) if exposure.size else 0.0, "breach_rate": float(np.mean(exposure > max_position)) if exposure.size else 0.0, "stress_pnl": float(np.nanmin(pos * res)) if pos.size else 0.0}


def compute_stability_by_day(y_true: np.ndarray, y_prob: np.ndarray, date_series: np.ndarray, metric_fn: Callable) -> dict:
    labels = np.asarray(date_series)
    values = [float(metric_fn(_arr(y_true)[labels == label], _arr(y_prob)[labels == label])) for label in sorted(set(labels))]
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "by_day": values}


def compute_stability_by_phase(y_true: np.ndarray, y_prob: np.ndarray, phase_bucket: np.ndarray, metric_fn: Callable) -> dict:
    labels = np.asarray(phase_bucket)
    return {str(label): float(metric_fn(_arr(y_true)[labels == label], _arr(y_prob)[labels == label])) for label in sorted(set(labels))}


def compute_fill_rate_by_offset(filled: np.ndarray, offset_bps: np.ndarray, buckets: list = [0, 5, 10, 20, 50, 100]) -> dict:
    f = np.asarray(filled, dtype=bool)
    o = _arr(offset_bps)
    result = {}
    for lo, hi in zip(buckets[:-1], buckets[1:], strict=False):
        mask = (o >= lo) & (o < hi)
        result[f"{lo}_{hi}"] = float(f[mask].mean()) if mask.any() else None
    return result


def compute_adverse_flag_precision_recall(adverse_flag_pred: np.ndarray, adverse_flag_true: np.ndarray) -> dict:
    pred = np.asarray(adverse_flag_pred, dtype=bool)
    true = np.asarray(adverse_flag_true, dtype=bool)
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_flip_precision_recall(flip_pred_prob: np.ndarray, flip_happened: np.ndarray, threshold: float = 0.5) -> dict:
    return compute_adverse_flag_precision_recall(_arr(flip_pred_prob) >= threshold, np.asarray(flip_happened, dtype=bool))


def compute_flip_rate_by_delta_bucket(flip_happened: np.ndarray, abs_delta_to_strike: np.ndarray, buckets: list = [0, 5, 10, 25, 50, 100, float("inf")]) -> dict:
    flip = np.asarray(flip_happened, dtype=bool)
    delta = _arr(abs_delta_to_strike)
    result = {}
    for lo, hi in zip(buckets[:-1], buckets[1:], strict=False):
        mask = (delta >= lo) & (delta < hi)
        result[f"{lo}_{hi}"] = float(flip[mask].mean()) if mask.any() else None
    return result


# ---------------------------------------------------------------------------
# Regression metrics (Model 6 and future regression models)
# ---------------------------------------------------------------------------

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(_arr(y_true) - _arr(y_pred))))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((_arr(y_true) - _arr(y_pred)) ** 2)))


def compute_edge_sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of rows where sign(y_pred) matches sign(y_true). Excludes y_true == 0."""
    y, p = _arr(y_true), _arr(y_pred)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y[mask]) == np.sign(p[mask])))


def compute_mispricing_by_delta_bucket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abs_delta: np.ndarray,
    buckets: list = [0, 5, 10, 25, 50, 100],
) -> dict:
    """MAE, mean actual/predicted edge, and sign accuracy by |delta_to_strike| bucket."""
    y, p, d = _arr(y_true), _arr(y_pred), _arr(abs_delta)
    result = {}
    edges = buckets + [float("inf")]
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (d >= lo) & (d < hi)
        label = f"{lo:.0f}_{hi:.0f}" if hi != float("inf") else f"{lo:.0f}_inf"
        if mask.sum() == 0:
            continue
        result[label] = {
            "rows": int(mask.sum()),
            "mae": compute_mae(y[mask], p[mask]),
            "mean_actual_edge": float(np.nanmean(y[mask])),
            "mean_predicted_edge": float(np.nanmean(p[mask])),
            "sign_accuracy": compute_edge_sign_accuracy(y[mask], p[mask]),
        }
    return result


def compute_mispricing_by_time_to_close(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_to_close_s: np.ndarray,
    buckets: list = [0, 10, 30, 60, 120, 300],
) -> dict:
    """MAE and sign accuracy by time-to-close bucket."""
    y, p, t = _arr(y_true), _arr(y_pred), _arr(t_to_close_s)
    result = {}
    for lo, hi in zip(buckets[:-1], buckets[1:], strict=False):
        mask = (t >= lo) & (t < hi)
        key = f"{lo}_{hi}"
        result[key] = {"rows": int(mask.sum())}
        if mask.any():
            result[key].update({
                "mae": compute_mae(y[mask], p[mask]),
                "mean_actual_edge": float(np.nanmean(y[mask])),
                "sign_accuracy": compute_edge_sign_accuracy(y[mask], p[mask]),
            })
    return result

