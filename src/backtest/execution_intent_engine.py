"""Backtest engine that replays execution_intent_dataset_v1 with real fill outcomes.

This dataset is the model training substrate — every row has:
  - All features the models were trained on (price_level, state_*, imbalance, etc.)
  - The actual `filled` outcome of the maker quote (whether it filled in real life)
  - `markout_to_close` (realized P&L per contract if held to close)
  - `t_to_close_s`, `order_side` ('buy'/'sell'), `state_mid`, `state_best_bid/ask`

Strategy filter is a thin wrapper: given a row, decide whether to PLACE the quote.
If our decision is "place" AND the row's `filled == True`, we book the realized PnL.

PnL per filled trade:
    direction = +1 if order_side == 'buy' else -1
    gross = markout_to_close * direction
    rebate = 0.20 * 0.072 * (fill_price * (1 - fill_price))
    net = gross + rebate
"""
from __future__ import annotations

import gc
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from .fees import FeeCalculator
from .pnl_tracker import PnLTracker
from .precomputed_signals import batch_precompute

try:
    from feature_cleanup import clean_features
except Exception:
    def clean_features(df):
        return df


DATASET = _REPO_ROOT / "data" / "datasets" / "execution_intent_dataset_v1"


@dataclass
class IntentFilter:
    """A ruleset for choosing which execution intents to accept.

    Most thresholds default to "accept everything" — disable a filter by leaving
    its threshold at the boundary value. The two band thresholds are exceptions:
    `signed_pred_min` and `signed_pred_max` bound `_pred_markout_to_close * direction`
    (with direction = +1 for buy, -1 for sell). The default (-inf, +inf) is
    no constraint; a narrow band like (0.05, 0.10] targets the empirical sweet
    spot observed in the data.
    """
    min_fill_prob: float = 0.0          # require _pred_fill_prob >= this
    max_t_to_close_s: float = 1e9       # require t_to_close_s <= this
    min_t_to_close_s: float = 0.0       # require t_to_close_s >= this (avoid last-second)
    max_p_severe: float = 1.0           # require _pred_p_severe < this
    max_p_extreme: float = 1.0
    max_p_moderate: float = 1.0
    signed_pred_min: float = -1e9       # require markout*direction >= this
    signed_pred_max: float = 1e9        # require markout*direction <= this
    use_direction_consistency: bool = False  # only allow if pred_direction matches order_side
    max_concurrent_positions: int = 100      # global limit (~1 contract per slot)
    quote_size: float = 1.0                  # contracts per fill

    def accept(self, row: dict) -> bool:
        fp = row.get("_pred_fill_prob_bid")
        if fp is None or not math.isfinite(fp):
            fp = 0.0
        if fp < self.min_fill_prob:
            return False
        ttc = row.get("t_to_close_s")
        if ttc is not None and math.isfinite(ttc):
            if ttc > self.max_t_to_close_s or ttc < self.min_t_to_close_s:
                return False
        for key, thr in [("_pred_p_severe_7c_3s", self.max_p_severe),
                         ("_pred_p_extreme_10c_5s", self.max_p_extreme),
                         ("_pred_p_moderate_5c_5s", self.max_p_moderate)]:
            v = row.get(key)
            if v is not None and math.isfinite(v) and v >= thr:
                return False
        # Signed-prediction band — direction-aware markout filter
        side = row.get("order_side") or "buy"
        mk = row.get("_pred_expected_markout_to_close")
        if mk is None or not math.isfinite(mk):
            if self.signed_pred_min > -1e8 or self.signed_pred_max < 1e8:
                return False
        else:
            direction = 1.0 if side == "buy" else -1.0
            signed = float(mk) * direction
            if signed < self.signed_pred_min or signed > self.signed_pred_max:
                return False
            if self.use_direction_consistency and signed < 0:
                return False
        return True


# ── Data loader ──────────────────────────────────────────────────────────────

def load_intent_week(dates: list[str], max_intents_per_day: Optional[int] = None,
                     seed: int = 0, filled_only: bool = False) -> pl.DataFrame:
    """Load execution_intent shards. Subsample per day if needed."""
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if filled_only:
            df = df.filter(pl.col("filled") == True)
        df = df.with_columns(pl.lit(d).alias("date"))
        if max_intents_per_day is not None and len(df) > max_intents_per_day:
            df = df.sample(n=max_intents_per_day, seed=seed, with_replacement=False)
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal").sort("intent_arrival_ts" if "intent_arrival_ts" in parts[0].columns else "target_date")


# ── Engine ───────────────────────────────────────────────────────────────────

class ExecutionIntentEngine:
    """Streaming backtester over execution_intent_dataset_v1."""

    def __init__(self, signals, filt: IntentFilter,
                 starting_balance_usd: float = 100.0, seed: int = 0,
                 fee_scale: float = 1.0):
        self.signals = signals
        self.filt = filt
        self.pnl = PnLTracker(starting_balance_usd)
        self.fee_calcs: dict[str, FeeCalculator] = {}
        self.fee_scale = fee_scale
        self.n_eligible_intents = 0
        self.n_accepted_intents = 0
        self.n_real_fills = 0

    def _fee(self, date: str) -> FeeCalculator:
        fc = self.fee_calcs.get(date)
        if fc is None:
            fc = FeeCalculator.for_date(date, fee_scale=self.fee_scale)
            self.fee_calcs[date] = fc
        return fc

    def run(self, dates: list[str], max_intents_per_day: int = 30_000,
            filled_only: bool = True, max_margin_usd: Optional[float] = None) -> dict:
        """filled_only=True is recommended — we only need filled rows to grade
        accepted intents, and this gives 3x the effective sample for the same
        memory/time budget.

        max_margin_usd: cap on the total USD value of open positions at any
        time. If None, falls back to `filt.max_concurrent_positions * 1.0`
        (treating each contract as $1 notional). Positions consume entry-price
        margin and free it at market_close_ts_ns.
        """
        df = load_intent_week(dates, max_intents_per_day=max_intents_per_day,
                              seed=0, filled_only=filled_only)
        if len(df) == 0:
            return {"error": "no data", **self.pnl.summary()}
        df = clean_features(df)
        df = batch_precompute(df, self.signals)
        self.n_eligible_intents = len(df)

        # Sort by entry time so position lifecycle is realistic
        if "order_ts_ns" in df.columns:
            df = df.sort("order_ts_ns")

        if max_margin_usd is None:
            max_margin_usd = float(self.filt.max_concurrent_positions) * 1.0

        # heap of (close_ts_ns, margin_freed_usd) sorted by close time
        import heapq
        open_heap: list[tuple[int, float]] = []
        margin_in_use = 0.0
        n_skipped_margin = 0
        t0 = time.time()
        for row in df.iter_rows(named=True):
            entry_ts = int(row.get("order_ts_ns") or 0)
            # Free any positions that have closed by now
            while open_heap and open_heap[0][0] <= entry_ts:
                _, freed = heapq.heappop(open_heap)
                margin_in_use -= freed

            if not self.filt.accept(row):
                continue
            self.n_accepted_intents += 1

            filled = row.get("filled")
            if filled is not True:
                continue
            mk = row.get("markout_to_close")
            if mk is None or not math.isfinite(mk):
                continue
            side = row.get("order_side") or "buy"
            direction = 1.0 if side == "buy" else -1.0
            fill_price = float(row.get("fill_price") or row.get("limit_price") or 0.5)
            fill_price = max(1e-6, min(1.0 - 1e-6, fill_price))
            entry_margin = fill_price * self.filt.quote_size

            # Check margin headroom
            if margin_in_use + entry_margin > max_margin_usd:
                n_skipped_margin += 1
                continue

            date = str(row.get("date") or row.get("target_date") or "?")
            fc = self._fee(date)
            rebate = fc.maker_rebate_usd(price=fill_price, size=self.filt.quote_size)
            realized = (float(mk) * direction * self.filt.quote_size) + rebate

            self.pnl.balance += realized
            self.pnl.daily_pnl[date] += realized
            self.pnl.gross_rebates += rebate
            self.pnl.n_fills += 1
            if realized > 0: self.pnl.n_wins += 1
            elif realized < 0: self.pnl.n_losses += 1
            self.pnl.peak_balance = max(self.pnl.peak_balance, self.pnl.balance)
            self.pnl.trough_balance = min(self.pnl.trough_balance, self.pnl.balance)
            self.pnl.max_drawdown_usd = max(self.pnl.max_drawdown_usd,
                                            self.pnl.peak_balance - self.pnl.balance)
            self.n_real_fills += 1

            # Schedule margin release at market close
            close_ts = int(row.get("market_close_ts_ns") or (entry_ts + int(self.filt.max_t_to_close_s * 1e9)))
            heapq.heappush(open_heap, (close_ts, entry_margin))
            margin_in_use += entry_margin

            if self.pnl.balance <= -100.0:   # 100% drawdown protection
                break

        s = self.pnl.summary()
        s["dates"] = dates
        s["elapsed_sec"] = time.time() - t0
        s["n_eligible_intents"] = self.n_eligible_intents
        s["n_accepted_intents"] = self.n_accepted_intents
        s["n_real_fills"] = self.n_real_fills
        s["n_skipped_for_margin"] = n_skipped_margin
        s["acceptance_rate"] = self.n_accepted_intents / max(1, self.n_eligible_intents)
        s["fill_rate"] = self.n_real_fills / max(1, self.n_accepted_intents)
        s["max_margin_usd"] = max_margin_usd
        return s


def cleanup():
    gc.collect()
