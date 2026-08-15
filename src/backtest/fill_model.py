"""Maker/taker fill simulator with realistic Polymarket fees.

Replaces the inline FillSimulator in backtest_pnl.py. Two improvements:

  1. Fees/rebates computed in USD via FeeCalculator (uses fee_schedule_registry).
  2. Optionally blends Model 03 fill-probability (passed via `sigs.fill_prob_*`)
     with the static 2-channel heuristic for more realistic maker fills.

The fill model has two channels:
  - p_random: benign uninformed flow (Model 03 baseline if available, else heuristic).
              Markout ~= 0 in expectation.
  - p_adverse: price moves through the quote. Markout = full subsequent mid_move_5s.
"""
from __future__ import annotations

import random
from typing import Optional

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from trading.config import ExecutionConfig
from trading.entities import (
    Fill, PlaceQuote, TakerOrder, FlattenInventory, SignalSnapshot,
)

from .fees import FeeCalculator


class FillSimulator:
    """Decides whether and at what price each order intent fills.

    Parameters
    ----------
    cfg : ExecutionConfig
    fee_calc : FeeCalculator
        Day-specific fee calculator (call FeeCalculator.for_date(...) once per day).
    rng : random.Random
    use_model_fill_prob : bool
        If True and a SignalSnapshot is passed to `simulate`, use the model's
        fill_prob_bid/ask as the random-flow baseline instead of the static
        heuristic. Defaults to True.
    """

    def __init__(self, cfg: ExecutionConfig, fee_calc: FeeCalculator,
                 rng: random.Random, use_model_fill_prob: bool = True):
        self.cfg = cfg
        self.fee_calc = fee_calc
        self.rng = rng
        self.use_model_fill_prob = use_model_fill_prob

    # ── Public ───────────────────────────────────────────────────────────────

    def simulate(self, intent, row: dict, sigs: Optional[SignalSnapshot] = None,
                 position: float = 0.0) -> Optional[Fill]:
        if isinstance(intent, PlaceQuote):
            return self._simulate_maker(intent, row, sigs)
        if isinstance(intent, TakerOrder):
            return self._simulate_taker(intent, row)
        if isinstance(intent, FlattenInventory):
            return self._simulate_flatten(intent, row, position)
        return None

    # ── Maker ────────────────────────────────────────────────────────────────

    def _simulate_maker(self, q: PlaceQuote, row: dict,
                        sigs: Optional[SignalSnapshot]) -> Optional[Fill]:
        mid = float(row.get("price_level") or 0.5)
        next_mid = mid + float(row.get("mid_move_5s") or 0.0)
        dist_ticks = abs(q.price - mid) / max(1e-9, self.cfg.tick_size)

        # Channel 1: benign uninformed flow
        if self.use_model_fill_prob and sigs is not None:
            base = sigs.fill_prob_bid if q.order_side == "BID" else sigs.fill_prob_ask
            if base is None or not (0.0 <= float(base) <= 1.0):
                base = 0.20
            else:
                base = float(base)
        else:
            base = 0.20
        p_random = base * (0.6 ** max(0, dist_ticks - 1))

        # Channel 2: adverse (price moves through our quote)
        adverse = ((q.order_side == "BID" and next_mid < q.price)
                   or (q.order_side == "ASK" and next_mid > q.price))
        p_adverse = 0.90 if adverse else 0.0

        p_fill = 1.0 - (1.0 - p_random) * (1.0 - p_adverse)
        if self.rng.random() > p_fill:
            return None

        rebate = self.fee_calc.maker_rebate_usd(price=q.price, size=q.size)
        return Fill(
            market_id=q.market_id, token_side=q.token_side,
            order_side=q.order_side, price=q.price, size=q.size,
            fees_paid=0.0, rebate_earned=rebate,
            client_order_id=q.client_order_id, ts_ns=int(row["anchor_ts_ns"]),
        )

    # ── Taker ────────────────────────────────────────────────────────────────

    def _simulate_taker(self, t: TakerOrder, row: dict) -> Fill:
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        fill_price = (mid + spread / 2) if t.order_side == "BID" else (mid - spread / 2)
        fill_price = max(1e-6, min(1.0 - 1e-6, fill_price))
        fees = self.fee_calc.taker_fee_usd(price=fill_price, size=t.size)
        return Fill(
            market_id=t.market_id, token_side=t.token_side,
            order_side=t.order_side, price=fill_price, size=t.size,
            fees_paid=fees, rebate_earned=0.0,
            client_order_id=t.client_order_id, ts_ns=int(row["anchor_ts_ns"]),
        )

    # ── Flatten ──────────────────────────────────────────────────────────────

    def _simulate_flatten(self, f: FlattenInventory, row: dict,
                          position: float) -> Optional[Fill]:
        if abs(position) < 1e-9:
            return None
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        order_side = "ASK" if position > 0 else "BID"
        fill_price = (mid - spread / 2) if order_side == "ASK" else (mid + spread / 2)
        fill_price = max(1e-6, min(1.0 - 1e-6, fill_price))
        size = abs(position)
        fees = self.fee_calc.taker_fee_usd(price=fill_price, size=size)
        return Fill(
            market_id=f.market_id, token_side=f.token_side,
            order_side=order_side, price=fill_price, size=size,
            fees_paid=fees, rebate_earned=0.0,
            client_order_id=f"flatten_{int(row['anchor_ts_ns'])}",
            ts_ns=int(row["anchor_ts_ns"]),
        )
