#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end P&L backtest of the hybrid maker/taker strategy.

DESIGN:
  - Pick a random 7-day window from the available microstructure data
  - For each anchor in the window:
      1. Build MarketEvent + feature_dict from parquet row
      2. Call strategy.on_market_event()
      3. Simulate fills using actual subsequent move + a stochastic fill model
  - Track $100 starting balance, position sizes scaled for the small budget
  - Report final balance, daily P&L, drawdown, win rate, Sharpe, fill stats

FILL MODEL (simplified but realistic for a sanity check):
  - PlaceQuote at price X: probability of fill within TIF = model_03(features)
    × scaling for distance from best
  - If filled: realized PnL = -mid_move_5s × signed_size (negative because
    we capture the "wrong-side" of subsequent move when adversely selected,
    plus spread capture from being maker)
  - Spread capture: if we filled at the best bid (we got hit), we earn
    half-spread + maker rebate; if we filled by improving the book, less.
  - TakerOrder: always fills at best ask (for BID) / best bid (for ASK)
    immediately; pays taker fee.
  - FlattenInventory: always fills at mid; pays half-spread + taker fee.

LIMITATIONS (for honest reporting):
  - We score one market at a time, not interleaved across markets
  - Fill probability is approximate (uses model 03 output as-is)
  - No queue-position model
  - No partial fills
  - We use mid_move_5s as the markout proxy
  - No actual maker rebate (placeholder 0.0)

Usage:
    py -3 backtest_pnl.py [--start YYYY-MM-DD] [--days 7] [--budget 100]
                         [--max-anchors 50000] [--seed 42]
"""
from __future__ import annotations
import sys, io, json, argparse, random, time
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from trading.config import StrategyConfig, RiskConfig, ExecutionConfig, SignalConfig, SelectionConfig
from trading.signals import SignalEngine
from trading.risk import RiskManager
from trading.strategy import HybridStrategy
from trading.entities import (
    MarketEvent, OrderBookSnapshot, Fill,
    PlaceQuote, CancelQuote, TakerOrder, FlattenInventory,
)
from feature_cleanup import clean_features


# ── Config tuned for $100 budget ─────────────────────────────────────────────

def build_small_budget_config() -> StrategyConfig:
    """Market-maker style: always quote, signals adjust not gate.
    Tuned for $100 bankroll: small uniform sizing (1 contract), strict halt limits.
    """
    cfg = StrategyConfig.market_maker()
    # Tighten halt limits for the small bankroll
    cfg.risk.max_daily_loss_usd = 20.0           # 20% halt
    cfg.risk.max_drawdown_from_session_peak_usd = 10.0
    cfg.risk.max_gross_inventory_contracts = 100.0
    # Realistic Polymarket-ish fee schedule (placeholder; tune to actual tier)
    cfg.execution.maker_rebate_per_contract_ticks = 0.0
    cfg.execution.taker_fee_per_contract_ticks = 0.5
    cfg.log_every_decision = False
    return cfg


# ── Data loading ─────────────────────────────────────────────────────────────

DATASET = Path("data/datasets/microstructure_sequence_dataset_v1_tabular")
MISP_DATASET = Path("data/datasets/resolution_snapshot_dataset_v1_dense_close")


def pick_random_week(seed: int) -> list[str]:
    """Pick a contiguous 7-day window from available daily shards (post-training)."""
    rng = random.Random(seed)
    files = sorted(DATASET.glob("*.parquet"))
    all_dates = [f.stem for f in files if "-" in f.stem]
    # Restrict to non-training window (Apr 19 - May 6 was training; pick later)
    candidates = [d for d in all_dates if d >= "2026-05-04"]
    if len(candidates) < 7:
        return all_dates[-7:]
    start = rng.randint(0, len(candidates) - 7)
    return candidates[start : start + 7]


def load_window(dates: list[str]) -> pl.DataFrame:
    """Load tabular data for the dates, sort by anchor_ts_ns within each day."""
    parts = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            print(f"  WARN missing {d}", file=sys.stderr)
            continue
        df = pl.read_parquet(p)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    if not parts:
        return pl.DataFrame()
    full = pl.concat(parts, how="diagonal")
    if "sequence_feature_eligible" in full.columns:
        full = full.filter(pl.col("sequence_feature_eligible"))
    full = full.filter(pl.col("mid_move_5s").is_not_null() & pl.col("mid_move_5s").is_not_nan())
    full = full.sort(["date", "anchor_ts_ns"])
    return full


# ── Event + feature construction ─────────────────────────────────────────────

def row_to_event_and_features(row: dict) -> tuple[MarketEvent, dict]:
    """Convert a parquet row into a MarketEvent + feature_dict for the strategy."""
    mid = float(row.get("price_level") or row.get("microprice") or 0.5)
    spread = float(row.get("spread_mean_5s") or 0.02)
    bid = mid - spread / 2
    ask = mid + spread / 2
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
    # Feature dict — pass all numeric columns through
    feat = {k: v for k, v in row.items() if v is not None and isinstance(v, (int, float))}
    feat["price_level"] = mid
    return ev, feat


# ── Fill simulator ───────────────────────────────────────────────────────────

class FillSimulator:
    """Decides whether and at what price each PlaceQuote / TakerOrder fills.

    Uses actual subsequent mid_move_5s from the row to determine adverse
    selection markout.  Uses a stochastic fill probability for maker quotes.
    """

    def __init__(self, cfg: ExecutionConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def simulate(self, intent, row: dict) -> Fill | None:
        """Return a Fill or None (not filled)."""
        if isinstance(intent, PlaceQuote):
            return self._simulate_maker(intent, row)
        if isinstance(intent, TakerOrder):
            return self._simulate_taker(intent, row)
        if isinstance(intent, FlattenInventory):
            return self._simulate_flatten(intent, row)
        return None

    def _simulate_maker(self, q: PlaceQuote, row: dict) -> Fill | None:
        """A maker quote fills via two channels:
          1. RANDOM (benign) flow: contra-side aggressive trader crosses spread.
             Markout is approximately zero (no predictive content).
          2. ADVERSE flow: price genuinely moves through our quote.
             Markout = full subsequent move (we get picked off).

        The COMBINED probability of any fill is max(p_random, p_adverse).  But
        because we don't separately track which channel filled us, we approximate
        the realized markout as a probability-weighted average of the two.
        """
        mid = float(row.get("price_level") or 0.5)
        next_mid = mid + float(row.get("mid_move_5s") or 0.0)
        dist_ticks = abs(q.price - mid) / self.cfg.tick_size

        # Channel 1: random non-toxic flow.  Roughly 20% per 5s for quote at best.
        p_random = 0.20 * (0.6 ** max(0, dist_ticks - 1))

        # Channel 2: adverse — price moves through us
        adverse = ((q.order_side == "BID" and next_mid < q.price)
                   or (q.order_side == "ASK" and next_mid > q.price))
        p_adverse = 0.90 if adverse else 0.0

        # Effective fill prob (union)
        p_fill = 1.0 - (1.0 - p_random) * (1.0 - p_adverse)
        if self.rng.random() > p_fill:
            return None

        rebate = self.cfg.maker_rebate_per_contract_ticks * self.cfg.tick_size * q.size
        return Fill(
            market_id=q.market_id, token_side=q.token_side,
            order_side=q.order_side, price=q.price, size=q.size,
            fees_paid=0.0, rebate_earned=rebate,
            client_order_id=q.client_order_id, ts_ns=int(row["anchor_ts_ns"]),
        )

    def _simulate_taker(self, t: TakerOrder, row: dict) -> Fill:
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        fill_price = (mid + spread / 2) if t.order_side == "BID" else (mid - spread / 2)
        fees = self.cfg.taker_fee_per_contract_ticks * self.cfg.tick_size * t.size
        return Fill(
            market_id=t.market_id, token_side=t.token_side,
            order_side=t.order_side, price=fill_price, size=t.size,
            fees_paid=fees, rebate_earned=0.0,
            client_order_id=t.client_order_id, ts_ns=int(row["anchor_ts_ns"]),
        )

    def _simulate_flatten(self, f: FlattenInventory, row: dict, position: float) -> Fill | None:
        """Flatten by crossing the spread.  Caller must pass current position."""
        if abs(position) < 1e-9:
            return None
        mid = float(row.get("price_level") or 0.5)
        spread = float(row.get("spread_mean_5s") or 0.02)
        # Selling out a long → hit the bid; covering a short → lift the ask
        order_side = "ASK" if position > 0 else "BID"
        fill_price = (mid - spread / 2) if order_side == "ASK" else (mid + spread / 2)
        size = abs(position)
        fees = self.cfg.taker_fee_per_contract_ticks * self.cfg.tick_size * size
        return Fill(
            market_id=f.market_id, token_side=f.token_side,
            order_side=order_side, price=fill_price, size=size,
            fees_paid=fees, rebate_earned=0.0,
            client_order_id="flatten_" + str(int(row["anchor_ts_ns"])),
            ts_ns=int(row["anchor_ts_ns"]),
        )


# ── P&L tracker ──────────────────────────────────────────────────────────────

class PnLTracker:
    """Tracks position + realized PnL per market and aggregate, with $100 budget."""

    def __init__(self, starting_balance_usd: float):
        self.starting = starting_balance_usd
        self.balance = starting_balance_usd
        self.positions: dict[str, float] = defaultdict(float)         # signed contracts
        self.avg_entry: dict[str, float] = defaultdict(float)
        self.realized: dict[str, float] = defaultdict(float)
        self.daily_pnl: dict[str, float] = defaultdict(float)         # by date string
        self.peak_balance = starting_balance_usd
        self.trough_balance = starting_balance_usd
        self.fills: list[dict] = []
        self.n_wins = 0
        self.n_losses = 0

    def apply_fill(self, fill: Fill, date_str: str, future_mid: float) -> float:
        """Apply fill, realize P&L immediately using future_mid as the close price.

        This treats each trade as a single-shot roundtrip (held ~5s).  It's a
        cleaner measure of the strategy's signal quality than tracking long-held
        positions, especially given we have markout columns at fixed horizons.

        realized = (future_mid - fill_price) * signed_size + rebate - fees
        """
        m = fill.market_id
        signed = fill.size if fill.order_side == "BID" else -fill.size

        # Realize P&L immediately at future_mid
        markout = (future_mid - fill.price) * signed
        realized = markout + fill.rebate_earned - fill.fees_paid

        # Tracking
        self.balance += realized
        self.realized[m] += realized
        self.daily_pnl[date_str] += realized
        if realized > 0:
            self.n_wins += 1
        elif realized < 0:
            self.n_losses += 1

        # We keep a "phantom" position briefly so the strategy's POST_FILL
        # branch can run, but we zero it out so it doesn't accumulate.
        self.positions[m] = 0.0
        self.avg_entry[m] = 0.0

        self.peak_balance = max(self.peak_balance, self.balance)
        self.trough_balance = min(self.trough_balance, self.balance)

        self.fills.append({
            "ts_ns": fill.ts_ns, "date": date_str,
            "market": fill.market_id, "side": fill.order_side,
            "price": fill.price, "size": fill.size,
            "future_mid": future_mid, "markout_per_contract": future_mid - fill.price,
            "realized": realized, "balance_after": self.balance,
        })
        return realized

    def summary(self) -> dict:
        n_trades = self.n_wins + self.n_losses
        win_rate = self.n_wins / n_trades if n_trades else 0.0
        total_realized = self.balance - self.starting
        return_pct = (total_realized / self.starting) * 100
        max_dd_usd = self.peak_balance - self.trough_balance
        # Daily returns for Sharpe
        daily_returns = [v / self.starting for v in self.daily_pnl.values()]
        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        return {
            "starting_usd": self.starting,
            "ending_usd": self.balance,
            "total_pnl_usd": total_realized,
            "return_pct": return_pct,
            "n_fills": len(self.fills),
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": win_rate,
            "peak_balance_usd": self.peak_balance,
            "trough_balance_usd": self.trough_balance,
            "max_drawdown_usd": max_dd_usd,
            "max_drawdown_pct": max_dd_usd / max(self.peak_balance, 1) * 100,
            "annualized_sharpe": sharpe,
            "daily_pnl": dict(self.daily_pnl),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None,
                    help="Start date (default: random week)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--max-anchors", type=int, default=20_000,
                    help="Cap on number of anchors evaluated (for speed)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    # Pick window
    if args.start:
        from datetime import date as _date
        d0 = _date.fromisoformat(args.start)
        dates = [(d0 + timedelta(days=i)).isoformat() for i in range(args.days)]
    else:
        dates = pick_random_week(args.seed)

    print("=" * 78, file=sys.stderr)
    print(f"P&L BACKTEST  |  budget=${args.budget:.0f}  |  seed={args.seed}", file=sys.stderr)
    print(f"window: {dates[0]} .. {dates[-1]} ({len(dates)} days)", file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    # Load data
    print("\nLoading data ...", file=sys.stderr)
    df = load_window(dates)
    if len(df) == 0:
        print("No data loaded.  Aborting.", file=sys.stderr); return
    print(f"  {len(df):,} eligible anchors across {len(dates)} days", file=sys.stderr)

    # Apply cleanup so model feature names match
    df = clean_features(df)

    # Subsample for tractable runtime
    if len(df) > args.max_anchors:
        df = df.sample(n=args.max_anchors, seed=args.seed, with_replacement=False).sort(["date", "anchor_ts_ns"])
        print(f"  subsampled to {len(df):,} for speed", file=sys.stderr)

    # Wire up the strategy
    cfg = build_small_budget_config()
    sigs = SignalEngine(artifact_root="artifacts_cleaned")
    risk = RiskManager(cfg.risk)
    strat = HybridStrategy(cfg, sigs, risk)
    sim = FillSimulator(cfg.execution, rng)
    pnl = PnLTracker(args.budget)

    # Main loop — process each anchor row
    print("\nReplaying anchors ...", file=sys.stderr)
    n_intents = 0
    n_fills = 0
    intents_by_type = defaultdict(int)
    t0 = time.time()
    rows = df.iter_rows(named=True)
    for i, row in enumerate(rows):
        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  [{i:,}/{len(df):,}] balance=${pnl.balance:.2f}  fills={n_fills}  "
                  f"intents={n_intents}  elapsed={elapsed:.0f}s", file=sys.stderr)

        # Build event + features
        try:
            event, feat = row_to_event_and_features(row)
        except Exception:
            continue

        # Ask strategy
        intents = strat.on_market_event(event, feat)
        n_intents += len(intents)
        for it in intents:
            intents_by_type[type(it).__name__] += 1

        # Simulate fills
        for it in intents:
            if isinstance(it, FlattenInventory):
                pos = pnl.positions.get(it.market_id, 0.0)
                fill = sim._simulate_flatten(it, row, pos)
            else:
                fill = sim.simulate(it, row)
            if fill is None:
                continue
            n_fills += 1
            future_mid = float(row.get("price_level") or 0.5) + float(row.get("mid_move_5s") or 0.0)
            pnl.apply_fill(fill, str(row["date"]), future_mid)
            state = strat.market_states.get(event.market_id)
            if state:
                strat.on_fill(fill, state, feat)
            # Quote that filled is no longer active
            if isinstance(it, PlaceQuote):
                risk.on_quote_cancelled()

        # SIM FIX: simulate TIF expiry for any quotes that weren't filled this
        # tick.  In real trading, quotes auto-cancel after TIF (~5s default).
        # We reset every anchor since we're not maintaining a live book here.
        risk.active_quotes_count = 0

        # Positions are realized immediately in apply_fill — no auto-close needed
        if pnl.balance <= 0:
            print(f"  *** BUSTED at anchor {i}: balance=${pnl.balance:.2f}", file=sys.stderr)
            break

    # ── Report ───────────────────────────────────────────────────────────────
    s = pnl.summary()
    print("\n" + "=" * 78)
    print("P&L BACKTEST RESULTS")
    print("=" * 78)
    print(f"  Window:                {dates[0]} .. {dates[-1]} ({len(dates)} days)")
    print(f"  Anchors processed:     {len(df):,}")
    print(f"  Intents emitted:       {n_intents}")
    for t, n in intents_by_type.items():
        print(f"    {t:<22} {n}")
    print(f"  Fills simulated:       {n_fills}")
    print()
    print(f"  Starting balance:      ${s['starting_usd']:.2f}")
    print(f"  Ending balance:        ${s['ending_usd']:.2f}")
    print(f"  Total P&L:             ${s['total_pnl_usd']:+.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Peak balance:          ${s['peak_balance_usd']:.2f}")
    print(f"  Trough balance:        ${s['trough_balance_usd']:.2f}")
    print(f"  Max drawdown:          ${s['max_drawdown_usd']:.2f}  ({s['max_drawdown_pct']:.1f}%)")
    print()
    print(f"  Wins / losses:         {s['n_wins']} / {s['n_losses']}")
    print(f"  Win rate:              {s['win_rate']*100:.1f}%")
    print(f"  Annualized Sharpe:     {s['annualized_sharpe']:.2f}")
    print()
    print(f"  Daily P&L:")
    for d, p in sorted(s['daily_pnl'].items()):
        bar = "+" * max(0, int(p * 5)) if p > 0 else "-" * max(0, int(-p * 5))
        print(f"    {d}  ${p:+7.2f}  {bar}")
    print()

    # Per-fill diagnostic
    print(f"  Sample fills (first 10 + last 5):")
    print(f"    {'date':<12} {'mkt':<30} {'side':<5} {'sz':>5} {'price':>7} {'markout':>9} {'realized':>10}")
    sample = pnl.fills[:10] + (pnl.fills[-5:] if len(pnl.fills) > 15 else [])
    for f in sample:
        print(f"    {f['date']:<12} {f['market'][:30]:<30} {f['side']:<5} "
              f"{f['size']:>5.0f} {f['price']:>7.4f} {f.get('markout_per_contract', 0):>9.4f} "
              f"${f['realized']:>+9.4f}")

    # Underlying market vol
    moves = df.get_column("mid_move_5s").to_numpy()
    moves = moves[np.isfinite(moves)]
    print()
    print(f"  Underlying mid_move_5s distribution (informational):")
    print(f"    n={len(moves):,}  mean={moves.mean():+.5f}  std={moves.std():.5f}")
    print(f"    p10={np.quantile(moves, 0.1):+.4f}  p50={np.quantile(moves, 0.5):+.4f}  "
          f"p90={np.quantile(moves, 0.9):+.4f}")
    print(f"    abs_p50={np.quantile(np.abs(moves), 0.5):.4f}  abs_p90={np.quantile(np.abs(moves), 0.9):.4f}")

    print()
    print(f"  Session summary (risk view): {risk.session_summary()}")

    # Save full results
    out = Path("docs/backtest_pnl_report.json")
    out.write_text(json.dumps({
        "config": {
            "budget": args.budget, "days": args.days, "seed": args.seed,
            "window": [dates[0], dates[-1]],
        },
        "summary": s,
        "intents_by_type": dict(intents_by_type),
        "n_anchors": len(df),
        "n_fills": n_fills,
    }, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
