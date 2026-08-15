"""Multi-market replay engine.

Runs the HybridStrategy on bucketed anchors so multiple concurrent markets
can compete for capital from a single shared RiskManager / PnLTracker.

Key differences vs the inline loop in backtest_pnl.py:
  - Anchors are grouped by `(date, second-bucket)` and processed in time order.
  - RiskManager is global (one instance for the whole week).
  - PnLTracker is global (single $100 bankroll).
  - Strategy state (MarketState dict, modes, etc.) is reused across the week.
  - Quiet by default — only summary printed unless `verbose=True`.
"""
from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading.config import StrategyConfig
from trading.entities import (
    Fill, FlattenInventory, MarketEvent, OrderBookSnapshot, PlaceQuote, TakerOrder,
)
from trading.risk import RiskManager
from trading.signals import SignalEngine
from trading.strategy import HybridStrategy

from .fees import FeeCalculator
from .fill_model import FillSimulator
from .pnl_tracker import PnLTracker
from .precomputed_signals import PrecomputedSignalEngine, batch_precompute

# Optional feature cleanup (drops trojan features)
try:
    from feature_cleanup import clean_features
except Exception:
    def clean_features(df):  # type: ignore
        return df


DATASET = _REPO_ROOT / "data" / "datasets" / "microstructure_sequence_dataset_v1_tabular"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_week(dates: list[str], max_anchors_per_day: Optional[int] = None,
              seed: int = 0,
              require_markout_field: str = "hold_to_close_edge_vs_mid") -> pl.DataFrame:
    """Load a list of date shards. Subsample within each day if requested."""
    parts: list[pl.DataFrame] = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p)
        if "sequence_feature_eligible" in df.columns:
            df = df.filter(pl.col("sequence_feature_eligible"))
        # Keep rows where the chosen markout field is present.
        if require_markout_field in df.columns:
            df = df.filter(
                pl.col(require_markout_field).is_not_null()
                & pl.col(require_markout_field).is_not_nan()
            )
        else:
            df = df.filter(pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan())
        df = df.with_columns(pl.lit(d).alias("date"))
        if max_anchors_per_day is not None and len(df) > max_anchors_per_day:
            df = df.sample(n=max_anchors_per_day, seed=seed, with_replacement=False)
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    full = pl.concat(parts, how="diagonal")
    return full.sort(["anchor_ts_ns"])


# ── Event construction ───────────────────────────────────────────────────────

def row_to_event_and_features(row: dict) -> tuple[MarketEvent, dict]:
    mid = float(row.get("price_level") or row.get("microprice") or 0.5)
    spread = float(row.get("spread_mean_5s") or 0.02)
    bid = max(1e-6, mid - spread / 2)
    ask = min(1.0 - 1e-6, mid + spread / 2)
    book = OrderBookSnapshot(
        market_id=str(row["market_slug"]),
        token_side=str(row.get("token_side_normalized", "YES")),
        best_bid=bid, best_ask=ask,
        bid_depth=float(row.get("state_bid_depth_total") or 100.0),
        ask_depth=float(row.get("state_ask_depth_total") or 100.0),
        mid=mid,
        microprice=float(row.get("microprice") or mid),
        spread=spread,
        ts_ns=int(row["anchor_ts_ns"]),
    )
    ev = MarketEvent(
        ts_ns=int(row["anchor_ts_ns"]),
        market_id=str(row["market_slug"]),
        token_side=str(row.get("token_side_normalized", "YES")),
        event_type="book_update",
        book=book,
    )
    feat = {k: v for k, v in row.items()
            if v is not None and isinstance(v, (int, float))}
    feat["price_level"] = mid
    return ev, feat


# ── Engine ───────────────────────────────────────────────────────────────────

class MultiMarketEngine:
    """Runs one week of anchors through HybridStrategy with a shared bankroll."""

    def __init__(self, cfg: StrategyConfig, signals: SignalEngine,
                 starting_balance_usd: float = 100.0,
                 verbose: bool = False, seed: int = 0,
                 use_precomputed_signals: bool = True,
                 markout_field: str = "hold_to_close_edge_vs_mid"):
        """markout_field: which dataset column to use as the realized P&L per fill.
        Defaults to `hold_to_close_edge_vs_mid` (true binary resolution payoff
        relative to current mid) — appropriate when the strategy is built on
        close-time signals (Models 02, 04b, 06). Use `mid_move_5s` for a
        short-horizon round-trip model.
        """
        self.cfg = cfg
        self.signals = signals  # Heavy engine — used for batch precompute
        self.use_precomputed = use_precomputed_signals
        self.markout_field = markout_field
        # Strategy uses the fast precomputed engine (reads _pred_* from feature dict)
        strategy_signals = PrecomputedSignalEngine() if use_precomputed_signals else signals
        self.risk = RiskManager(cfg.risk)
        self.strategy = HybridStrategy(cfg, strategy_signals, self.risk)
        self.pnl = PnLTracker(starting_balance_usd)
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.fee_calcs: dict[str, FeeCalculator] = {}

    def _fee_calc_for_date(self, date_str: str) -> FeeCalculator:
        fc = self.fee_calcs.get(date_str)
        if fc is None:
            fc = FeeCalculator.for_date(date_str, fee_scale=self._fee_scale)
            self.fee_calcs[date_str] = fc
        return fc

    @property
    def _fee_scale(self) -> float:
        return getattr(self, "fee_scale_override", 1.0)

    def run(self, dates: list[str],
            max_anchors_per_day: int = 5_000,
            fee_scale: float = 1.0) -> dict:
        """Process the week. Returns a summary dict."""
        self.fee_scale_override = fee_scale
        df = load_week(dates, max_anchors_per_day=max_anchors_per_day, seed=self.rng.randint(0, 2**31 - 1))
        if len(df) == 0:
            return {"error": "no data", "dates": dates, **self.pnl.summary()}

        df = clean_features(df)

        # Batch-precompute all 9 model predictions for the whole week (massive speedup)
        if self.use_precomputed:
            df = batch_precompute(df, self.signals)

        t0 = time.time()
        for row in df.iter_rows(named=True):
            date_str = str(row["date"])
            fee_calc = self._fee_calc_for_date(date_str)
            sim = FillSimulator(self.cfg.execution, fee_calc, self.rng)

            try:
                event, feat = row_to_event_and_features(row)
            except Exception:
                continue

            # Ask strategy (calls signals.predict internally; we don't recompute)
            intents = self.strategy.on_market_event(event, feat)

            for it in intents:
                if isinstance(it, FlattenInventory):
                    pos = self.risk.positions.get(it.market_id, 0.0)
                    fill = sim._simulate_flatten(it, row, pos)
                else:
                    # Pass sigs=None → fill model uses static heuristic for p_random
                    # (saves a full SignalEngine.predict per anchor).
                    fill = sim.simulate(it, row, sigs=None)
                if fill is None:
                    continue
                mid = float(row.get("price_level") or 0.5)
                markout = row.get(self.markout_field)
                if markout is None or markout != markout:   # None or NaN
                    markout = float(row.get("mid_move_5s") or 0.0)   # fallback
                future_mid = mid + float(markout)
                future_mid = max(1e-6, min(1.0 - 1e-6, future_mid))
                self.pnl.apply_fill(fill, date_str, future_mid)
                state = self.strategy.market_states.get(event.market_id)
                if state:
                    self.strategy.on_fill(fill, state, feat)
                if isinstance(it, PlaceQuote):
                    self.risk.on_quote_cancelled()

            # Simulate TIF expiry — quotes auto-cancel each anchor in this sim
            self.risk.active_quotes_count = 0

            if self.pnl.is_busted():
                if self.verbose:
                    print(f"  busted on {date_str}", file=sys.stderr)
                break

        elapsed = time.time() - t0
        s = self.pnl.summary()
        s["dates"] = dates
        s["elapsed_sec"] = elapsed
        s["n_anchors_processed"] = len(df)
        return s


def cleanup():
    """Force GC + clear polars caches to keep memory bounded across runs."""
    gc.collect()
