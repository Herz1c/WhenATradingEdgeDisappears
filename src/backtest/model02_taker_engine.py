"""Model 02 taker engine with proper margin lifecycle.

Replays resolution_dense_close snapshots in time order. At each snapshot:
  1. Run Model 02 → predicted P(UP).
  2. Compute taker edge per side:
       edge_UP = p_up - up_ask         (buy UP at ask)
       edge_DN = up_bid - p_up         (sell UP at bid ≡ buy DOWN)
  3. If edge >= threshold AND ttc in range AND a side is allowed:
       Place a taker order, lock margin until market close.
  4. At market close, realize PnL = payoff - entry - fee.

Margin model: each open position consumes `entry_price * size` of bankroll
until that market's close_ts_ns. Heap-based release on each new snapshot.
"""
from __future__ import annotations

import gc, heapq, json, math, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from backtest.fees import FeeCalculator
from feature_cleanup import clean_features


DATASET = _REPO_ROOT / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
ARTIFACT = _REPO_ROOT / "artifacts_cleaned" / "model_02_fair_resolution" / "dense_close" / "lightgbm" / "model.pkl"


def _load_model():
    model = joblib.load(ARTIFACT)
    feats = list(json.loads((ARTIFACT.parent / "feature_importance.json").read_text()).keys())
    return model, feats


def _predict_p_up(model, feats, df):
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def load_dense_week(dates, max_per_day=20_000, seed=42):
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists(): continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        if max_per_day and len(df) > max_per_day:
            df = df.sample(n=max_per_day, seed=seed, with_replacement=False)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    full = pl.concat(parts, how="diagonal")
    if "snapshot_ts_ns" in full.columns:
        full = full.sort("snapshot_ts_ns")
    return full


def run_taker_backtest(
    dates,
    edge_threshold_usd: float = 0.10,
    allow_up_side: bool = False,           # DOWN side is the profitable side per audit
    allow_dn_side: bool = True,
    min_t_to_close_s: float = 10,
    max_t_to_close_s: float = 60,
    starting_usd: float = 100.0,
    max_margin_usd: float = 100.0,
    quote_size: float = 1.0,
    max_per_day: int = 20_000,
    fee_scale: float = 1.0,
    seed: int = 42,
    max_positions_per_market: int = 1,    # at most this many concurrent positions per (market, side)
    min_seconds_between_entries_per_market: float = 0.0,  # min gap between subsequent entries
):
    df = load_dense_week(dates, max_per_day=max_per_day, seed=seed)
    if len(df) == 0:
        return {"error": "no data", "starting_usd": starting_usd, "ending_usd": starting_usd,
                "net_pnl_usd": 0, "max_drawdown_usd": 0, "n_fills": 0, "win_rate": 0,
                "daily_pnl": {}, "worst_day_pnl_usd": 0, "best_day_pnl_usd": 0, "dates": dates}
    df = clean_features(df)

    model, feats = _load_model()
    p_up = _predict_p_up(model, feats, df)

    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    dates_arr = df["date"].to_numpy()
    market_slugs = df["market_slug"].to_numpy()
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)

    # Pending PnL realisations: heap of (close_ts, date, realized_pnl, margin_freed)
    pending: list[tuple] = []
    margin_in_use = 0.0
    balance = starting_usd
    peak = trough = starting_usd
    max_dd = 0.0
    daily = defaultdict(float)
    n_fills = n_wins = n_losses = 0
    n_skipped_margin = 0
    n_skipped_no_signal = 0
    n_skipped_market_cap = 0
    fee_calcs: dict[str, FeeCalculator] = {}
    # Track how many positions we already have per (market, side)
    open_positions_per_key: dict[tuple, int] = defaultdict(int)
    # Track last entry ts per (market, side) for the min-gap rule
    last_entry_ts_per_key: dict[tuple, int] = {}
    n_skipped_gap = 0
    min_gap_ns = int(min_seconds_between_entries_per_market * 1e9)

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up
    down_ask = 1.0 - up_bid

    for i in range(len(df)):
        # Realize pending positions whose close has passed
        while pending and pending[0][0] <= snap_ts[i]:
            _, d, real, freed, mk = heapq.heappop(pending)
            balance += real
            daily[d] += real
            margin_in_use -= freed
            open_positions_per_key[mk] = max(0, open_positions_per_key[mk] - 1)
            if real > 0: n_wins += 1
            elif real < 0: n_losses += 1
            peak = max(peak, balance)
            trough = min(trough, balance)
            max_dd = max(max_dd, peak - balance)

        if not (min_t_to_close_s <= ttc[i] <= max_t_to_close_s):
            continue

        # Pick side
        side = None
        entry = None
        if allow_up_side and edge_up[i] >= edge_threshold_usd:
            side = "UP"; entry = up_ask[i]
        elif allow_dn_side and edge_dn[i] >= edge_threshold_usd:
            side = "DOWN"; entry = down_ask[i]
        else:
            n_skipped_no_signal += 1
            continue

        if entry is None or not math.isfinite(entry) or entry <= 0 or entry >= 1:
            continue

        # Per-market position cap — avoid trading the same market many times
        # off many snapshots within its 5-min life
        market_key = (str(market_slugs[i]), side)
        if open_positions_per_key[market_key] >= max_positions_per_market:
            n_skipped_market_cap += 1
            continue

        # Min-gap rule: when allowing multiple entries per market, require
        # at least N seconds since the last entry on that (market, side)
        if min_gap_ns > 0:
            last_ts = last_entry_ts_per_key.get(market_key)
            if last_ts is not None and (snap_ts[i] - last_ts) < min_gap_ns:
                n_skipped_gap += 1
                continue

        margin = entry * quote_size
        if margin_in_use + margin > max_margin_usd:
            n_skipped_margin += 1
            continue

        # Compute realised PnL (at close)
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        date = str(dates_arr[i])
        fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date, fee_scale=fee_scale))
        fee = fc.taker_fee_usd(price=entry, size=quote_size)
        payoff = 1.0 if won else 0.0
        realized = (payoff - entry) * quote_size - fee

        # Schedule realisation at close_ts (also pass market_key so we can
        # release the per-market slot when the position closes)
        heapq.heappush(pending, (int(close_ts[i]), date, realized, margin, market_key))
        margin_in_use += margin
        open_positions_per_key[market_key] += 1
        last_entry_ts_per_key[market_key] = int(snap_ts[i])
        n_fills += 1

    # Flush remaining positions (they would resolve after our last snapshot)
    while pending:
        _, d, real, freed, mk = heapq.heappop(pending)
        balance += real
        daily[d] += real
        margin_in_use -= freed
        open_positions_per_key[mk] = max(0, open_positions_per_key[mk] - 1)
        if real > 0: n_wins += 1
        elif real < 0: n_losses += 1
        peak = max(peak, balance)
        trough = min(trough, balance)
        max_dd = max(max_dd, peak - balance)

    return {
        "starting_usd": starting_usd,
        "ending_usd": balance,
        "net_pnl_usd": balance - starting_usd,
        "max_drawdown_usd": max_dd,
        "peak_balance_usd": peak,
        "trough_balance_usd": trough,
        "n_fills": n_fills, "n_wins": n_wins, "n_losses": n_losses,
        "win_rate": (n_wins / n_fills) if n_fills else 0.0,
        "daily_pnl": dict(daily),
        "worst_day_pnl_usd": min(daily.values()) if daily else 0.0,
        "best_day_pnl_usd": max(daily.values()) if daily else 0.0,
        "dates": dates,
        "n_skipped_margin": n_skipped_margin,
        "n_skipped_no_signal": n_skipped_no_signal,
        "n_skipped_market_cap": n_skipped_market_cap,
        "n_skipped_gap": n_skipped_gap,
        "config": {
            "edge_threshold_usd": edge_threshold_usd,
            "allow_up_side": allow_up_side,
            "allow_dn_side": allow_dn_side,
            "min_t_to_close_s": min_t_to_close_s,
            "max_t_to_close_s": max_t_to_close_s,
            "max_margin_usd": max_margin_usd,
            "quote_size": quote_size,
            "fee_scale": fee_scale,
            "max_positions_per_market": max_positions_per_market,
            "min_seconds_between_entries_per_market": min_seconds_between_entries_per_market,
        },
    }
