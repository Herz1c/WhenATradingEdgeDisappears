"""Optuna search space for HybridStrategy hyperparameters.

Trimmed first pass: only the `hybrid` mode (maker + taker fallback).
Ranges chosen from existing defaults in src/trading/config.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading.config import StrategyConfig


def suggest_hybrid_config(trial) -> tuple[StrategyConfig, dict[str, Any]]:
    """Sample a hyperparameter point for the hybrid mode.

    Returns (StrategyConfig, params_dict_for_logging).
    """
    # Use market_maker preset as the base — already tuned for maker-first;
    # we re-enable taker by setting max_taker_trades_per_hour > 0.
    cfg = StrategyConfig.market_maker()
    params: dict[str, Any] = {}

    # ── Quoting geometry ────────────────────────────────────────────────────
    # Wider range — adverse selection means tight quotes lose money on PolyMM
    params["quote_edge_ticks"] = trial.suggest_float("quote_edge_ticks", 0.5, 10.0)
    cfg.signals.quote_edge_ticks = params["quote_edge_ticks"]

    # Edge check gate — uniform mode now respects min_edge_required_ticks
    params["enforce_edge_check"] = trial.suggest_categorical("enforce_edge_check", [True, False])
    cfg.risk.enforce_edge_check_in_uniform_mode = params["enforce_edge_check"]
    # Allow negative threshold (still quote if expected adverse cost is small).
    # Half-spread on Polymarket is ~0.005 USD = 0.5 ticks; rebate at p=0.5
    # is ~0.0036 USD. So break-even adverse selection is ~0.36 ticks.
    params["min_edge_required_ticks"] = trial.suggest_float("min_edge_required_ticks", -1.0, 2.0)
    cfg.risk.min_edge_required_ticks = params["min_edge_required_ticks"]

    # ── Taker gating ────────────────────────────────────────────────────────
    params["max_taker_trades_per_hour"] = trial.suggest_int("max_taker_trades_per_hour", 0, 20)
    cfg.risk.max_taker_trades_per_hour = params["max_taker_trades_per_hour"]
    params["taker_min_edge_ticks"] = trial.suggest_float("taker_min_edge_ticks", 1.0, 5.0)
    cfg.signals.taker_min_edge_ticks = params["taker_min_edge_ticks"]
    params["taker_min_fill_prob"] = trial.suggest_float("taker_min_fill_prob", 0.50, 0.90)
    cfg.signals.taker_min_fill_prob = params["taker_min_fill_prob"]

    # ── Defense thresholds ──────────────────────────────────────────────────
    # Wider range — observed p_severe values are mostly 0.05-0.30, so the
    # default 0.40 floor never fires. Allow 0.05-0.65 to find the right cutoff.
    params["severe_defense_threshold_frac"] = trial.suggest_float(
        "severe_defense_threshold_frac", 0.05, 0.65)
    cfg.signals.severe_defense_threshold_frac = params["severe_defense_threshold_frac"]
    params["extreme_freeze_threshold_frac"] = trial.suggest_float(
        "extreme_freeze_threshold_frac", 0.05, 0.55)
    cfg.signals.extreme_freeze_threshold_frac = params["extreme_freeze_threshold_frac"]
    params["moderate_lean_threshold_frac"] = trial.suggest_float(
        "moderate_lean_threshold_frac", 0.10, 0.70)
    cfg.signals.moderate_lean_threshold_frac = params["moderate_lean_threshold_frac"]

    # ── Direction skew ──────────────────────────────────────────────────────
    params["direction_skew_threshold_frac"] = trial.suggest_float(
        "direction_skew_threshold_frac", 0.55, 0.80)
    cfg.signals.direction_skew_threshold_frac = params["direction_skew_threshold_frac"]
    params["direction_skew_ticks"] = trial.suggest_float("direction_skew_ticks", 0.0, 2.0)
    cfg.signals.direction_skew_ticks = params["direction_skew_ticks"]

    # ── Directional maker gate (Model 04b markout-to-close) ─────────────────
    # The big finding from analysis: maker quote on the side Model 04b
    # predicts is profitable yields >$0/trade. Gate 0.0 = disabled.
    params["markout_to_close_directional_gate_usd"] = trial.suggest_float(
        "markout_to_close_directional_gate_usd", 0.0, 0.30)
    cfg.signals.markout_to_close_directional_gate_usd = params["markout_to_close_directional_gate_usd"]

    # ── Closing flip defense ────────────────────────────────────────────────
    params["closing_flip_window_seconds"] = trial.suggest_int(
        "closing_flip_window_seconds", 30, 120)
    cfg.signals.closing_flip_window_seconds = params["closing_flip_window_seconds"]
    params["closing_flip_defense_threshold_frac"] = trial.suggest_float(
        "closing_flip_defense_threshold_frac", 0.45, 0.70)
    cfg.signals.closing_flip_defense_threshold_frac = params["closing_flip_defense_threshold_frac"]

    # ── Post-fill flatten ───────────────────────────────────────────────────
    params["flatten_if_predicted_markout_ticks"] = trial.suggest_float(
        "flatten_if_predicted_markout_ticks", -2.0, -0.3)
    cfg.signals.flatten_if_predicted_markout_below_ticks = params["flatten_if_predicted_markout_ticks"]

    # ── Sizing ──────────────────────────────────────────────────────────────
    params["uniform_quote_size"] = trial.suggest_int("uniform_quote_size", 1, 5)
    cfg.risk.uniform_quote_size = float(params["uniform_quote_size"])
    cfg.risk.max_size_per_quote_contracts = max(2.0, 2.0 * float(params["uniform_quote_size"]))

    # ── Market selection ────────────────────────────────────────────────────
    params["no_quote_in_last_n_seconds"] = trial.suggest_int(
        "no_quote_in_last_n_seconds", 5, 60)
    cfg.risk.no_quote_in_last_n_seconds = params["no_quote_in_last_n_seconds"]
    params["max_concurrent_markets"] = trial.suggest_int("max_concurrent_markets", 2, 20)
    cfg.selection.max_concurrent_markets = params["max_concurrent_markets"]
    params["require_min_book_depth_contracts"] = trial.suggest_float(
        "require_min_book_depth_contracts", 5.0, 100.0)
    cfg.selection.require_min_book_depth_contracts = params["require_min_book_depth_contracts"]

    # Tighten halt limits for $100 bankroll
    cfg.risk.max_daily_loss_usd = 20.0
    cfg.risk.max_drawdown_from_session_peak_usd = 10.0

    # Accurate Polymarket fee/rebate approximation for edge calc.
    # Real rebate at p=0.5 = 0.20 * 0.072 * 0.25 = 0.0036 USD = 0.36 ticks.
    # Real taker fee at p=0.5 = 0.072 * 0.25 = 0.018 USD = 1.8 ticks.
    cfg.execution.maker_rebate_per_contract_ticks = 0.36
    cfg.execution.taker_fee_per_contract_ticks = 1.8

    # Backtest-only: disable all wall-clock-based caps. RiskManager uses
    # time.time() for blackouts and rate limits, but the backtest replays
    # historical data in fast-forward — a 30-min wall-clock blackout or a
    # 200/min rate cap would kill the rest of the week. Bankruptcy + dd
    # checks (not wall-clock based) still gate exits.
    cfg.risk.session_blackout_after_loss_minutes = 0
    cfg.risk.market_blackout_after_losses_seconds = 0
    cfg.risk.max_consecutive_session_losses = 10_000
    cfg.risk.max_consecutive_losses_per_market = 10_000
    cfg.risk.max_trades_per_minute = 10_000_000
    cfg.risk.max_taker_trades_per_hour = 10_000_000  # search may still set lower; this is just the wall-clock-cap kill

    # Quiet
    cfg.log_every_decision = False
    cfg.log_skipped_quotes = False

    return cfg, params


def fixed_baseline_config() -> StrategyConfig:
    """A deterministic baseline using the existing market_maker preset."""
    cfg = StrategyConfig.market_maker()
    cfg.risk.max_daily_loss_usd = 20.0
    cfg.risk.max_drawdown_from_session_peak_usd = 10.0
    cfg.risk.session_blackout_after_loss_minutes = 0
    cfg.risk.market_blackout_after_losses_seconds = 0
    cfg.risk.max_consecutive_session_losses = 10_000
    cfg.risk.max_consecutive_losses_per_market = 10_000
    cfg.risk.max_trades_per_minute = 10_000_000
    cfg.risk.max_taker_trades_per_hour = 10_000_000
    cfg.execution.maker_rebate_per_contract_ticks = 0.36
    cfg.execution.taker_fee_per_contract_ticks = 1.8
    cfg.log_every_decision = False
    cfg.log_skipped_quotes = False
    return cfg


def fixed_directional_maker_config(gate_usd: float = 0.05) -> StrategyConfig:
    """The strategy that the data analysis pointed to: maker quotes on the
    side that Model 04b (markout-to-close) predicts is profitable, skipping
    quotes on the unfavorable side.

    On a $100 bankroll, $0.05 gate yielded ~$67 PnL/week in the prototype
    (single contract per fill, no risk halts).
    """
    cfg = fixed_baseline_config()
    cfg.signals.markout_to_close_directional_gate_usd = gate_usd
    cfg.risk.enforce_edge_check_in_uniform_mode = False  # gate alone — no edge gate
    cfg.signals.quote_edge_ticks = 0.5  # quote at best bid/ask (tight) to maximize fills
    cfg.selection.max_concurrent_markets = 20
    cfg.risk.uniform_quote_size = 1.0
    return cfg


def fixed_defensive_config() -> StrategyConfig:
    """A hand-tuned defensive baseline: enforce edge check, lower defense
    thresholds matched to observed data, wider quotes, fewer markets."""
    cfg = fixed_baseline_config()
    cfg.risk.enforce_edge_check_in_uniform_mode = True
    cfg.risk.min_edge_required_ticks = 0.0     # break-even or better
    cfg.signals.severe_defense_threshold_frac = 0.20
    cfg.signals.extreme_freeze_threshold_frac = 0.15
    cfg.signals.moderate_lean_threshold_frac = 0.25
    cfg.signals.quote_edge_ticks = 3.0
    cfg.selection.max_concurrent_markets = 5
    cfg.selection.require_min_book_depth_contracts = 30.0
    cfg.risk.uniform_quote_size = 1.0
    return cfg
