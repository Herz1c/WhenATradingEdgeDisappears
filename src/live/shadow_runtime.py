"""Shadow-mode runtime — runs the FULL production decision pipeline
(model -> filters -> sizing -> risk state) over a dense_close parquet
and logs every snapshot + decision to JSONL.

This is the exact code path the live bot will use. The only difference
between shadow and live is the source of the parquet:

  - shadow  : reads a parquet built once for [start, end] window
              (tools/build_dataset_window.py)
  - live    : reads a parquet that is rebuilt every N seconds from
              recorder tails (TBD — Phase 2)

By keeping shadow and live identical from `predict_p_up()` onwards, the
diff against the historical parquet's expected decisions catches all
strategy-wiring bugs (wrong threshold, wrong sizing, wrong feature
order, calibrator accidentally re-attached, etc.) in seconds.

Strategy parameters are hard-coded here to match LiveTradingBotPlan.md.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
# Only rewrap stdout when run as a script — importing this module from
# another script would otherwise close the importer's stdout handle.
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import joblib
import numpy as np
import polars as pl

from feature_cleanup import clean_features
from backtest.fees import FeeCalculator


# ──────────────────────────────────────────────────────────────────────────────
# Strategy parameters — keep in lockstep with LiveTradingBotPlan.md §1
# ──────────────────────────────────────────────────────────────────────────────
THRESHOLD                    = 0.3067
EDGE_K                       = 8.0
MIN_NOTIONAL                 = 1.00
MAX_NOTIONAL                 = 2.50          # FINAL choice
MAX_POSITIONS_PER_MARKET     = 2
MIN_SECONDS_BETWEEN_ENTRIES  = 10.0
MAX_MARGIN_USD               = 30.00
TTC_MIN_S, TTC_MAX_S         = 10.0, 60.0
SECOND_NS                    = 1_000_000_000

ART = Path("artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm")


# ──────────────────────────────────────────────────────────────────────────────
# Risk state (the live bot will share this exact dataclass)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskState:
    margin_in_use: float = 0.0
    positions_per_market: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_entry_ns_per_market: dict[str, int] = field(default_factory=dict)
    # closed-market sweep — list of (close_ts_ns, notional)
    open_lots: list[tuple[int, float]] = field(default_factory=list)

    def release_closed(self, now_ns: int) -> None:
        keep: list[tuple[int, float]] = []
        for close_ns, notional in self.open_lots:
            if close_ns <= now_ns:
                self.margin_in_use -= notional
            else:
                keep.append((close_ns, notional))
        self.open_lots = keep

    def add_lot(self, close_ns: int, notional: float) -> None:
        self.open_lots.append((close_ns, notional))
        self.margin_in_use += notional


# ──────────────────────────────────────────────────────────────────────────────
# Decision engine (mirror of LiveTradingBotPlan.md §3.4)
# ──────────────────────────────────────────────────────────────────────────────
def size_for_edge(edge_dn: float) -> float:
    raw = 1.0 + EDGE_K * max(0.0, edge_dn - THRESHOLD)
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, raw))


def decide(*, snap: dict, p_up: float, state: RiskState) -> dict:
    """Return a dict describing what would happen for this snapshot."""
    up_bid = snap.get("up_token_best_bid")
    up_ask = snap.get("up_token_best_ask")
    out: dict = {"decision": "skip", "reason": "", "notional": 0.0,
                 "edge_dn": None, "p_up": float(p_up)}

    if up_bid is None or up_ask is None:
        out["reason"] = "no_quote"; return out
    if up_ask <= 0.01 or up_bid >= 0.99:
        out["reason"] = "degenerate_quote"; return out

    ttc = float(snap["t_to_close_s"])
    if not (TTC_MIN_S <= ttc <= TTC_MAX_S):
        out["reason"] = "ttc_band"; return out

    edge_dn = float(up_bid) - float(p_up)
    out["edge_dn"] = edge_dn
    if edge_dn < THRESHOLD:
        out["reason"] = "below_threshold"; return out

    mk = str(snap["market_slug"])
    if state.positions_per_market[mk] >= MAX_POSITIONS_PER_MARKET:
        out["reason"] = "market_cap_full"; return out
    last = state.last_entry_ns_per_market.get(mk)
    snap_ts = int(snap["snapshot_ts_ns"])
    if last is not None and (snap_ts - last) < int(MIN_SECONDS_BETWEEN_ENTRIES * SECOND_NS):
        out["reason"] = "within_gap"; return out

    notional = size_for_edge(edge_dn)
    avail = MAX_MARGIN_USD - state.margin_in_use
    if avail < MIN_NOTIONAL:
        out["reason"] = "margin_full"; return out
    if notional > avail:
        notional = avail

    out["decision"] = "ENTER"
    out["reason"]   = "ENTER"
    out["notional"] = float(notional)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    if os.environ.get("FEATURE_CLEANUP_ENABLED") != "1":
        print("WARN: FEATURE_CLEANUP_ENABLED is not '1' — setting it now for safety.")
        os.environ["FEATURE_CLEANUP_ENABLED"] = "1"

    model = joblib.load(ART / "model.pkl")
    if hasattr(model, "_calibrator") and getattr(model, "_calibrator") is not None:
        raise RuntimeError(
            "Model has a non-null _calibrator. The deployed model must have the calibrator "
            "stripped (see LiveTradingBotPlan.md hard rule #1)."
        )
    feats = list(json.loads((ART / "feature_importance.json").read_text()).keys())
    print(f"model loaded: {ART}  (features={len(feats)}, calibrator=None)")
    return model, feats


def build_X(df_clean: pl.DataFrame, feats: list[str]) -> np.ndarray:
    X = np.zeros((len(df_clean), len(feats)), dtype=np.float32)
    for i, f in enumerate(feats):
        if f in df_clean.columns:
            s = df_clean.get_column(f)
            if s.dtype.is_numeric():
                v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
                X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-parquet", required=True, type=Path,
                    help="Path to the dense_close parquet to replay through the strategy.")
    ap.add_argument("--output", required=True, type=Path,
                    help="JSONL output path — one line per processed snapshot.")
    ap.add_argument("--log-all", action="store_true",
                    help="Log every snapshot (default: only ttc-band rows, which is enough "
                         "to diff decisions and edge values).")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model, feats = load_model()

    df = pl.read_parquet(args.source_parquet)
    print(f"source rows: {len(df):,}  markets: {df['market_slug'].n_unique()}")

    df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
    df = df.sort(["snapshot_ts_ns", "market_slug"])

    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up_all = model.predict_proba(X)[:, 1]

    # Pre-extract columns we'll touch in the loop
    up_bid_arr     = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask_arr     = df["up_token_best_ask"].to_numpy().astype(float)
    ttc_arr        = df["t_to_close_s"].to_numpy().astype(float)
    market_arr     = df["market_slug"].to_numpy()
    snap_ts_arr    = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts_arr = (
        df["market_close_ts_ns"].to_numpy().astype(np.int64)
        if "market_close_ts_ns" in df.columns
        else snap_ts_arr + (300 * SECOND_NS)
    )
    # resolved_side_label is present in backtest parquets but won't be in true-live ones
    # — guard accordingly so the same code runs in both modes
    resolved_arr = (
        df["resolved_side_label"].to_numpy().astype(int)
        if "resolved_side_label" in df.columns
        else np.full(len(df), -1, dtype=int)
    )

    state = RiskState()
    fee_calcs: dict[str, FeeCalculator] = {}

    n_total = 0
    n_logged = 0
    n_entered = 0
    n_realized = 0
    realized_pnl = 0.0
    reason_counts: dict[str, int] = defaultdict(int)

    t0 = datetime.now(UTC)
    with args.output.open("w", encoding="utf-8") as fh:
        for i in range(len(df)):
            n_total += 1
            now_ns = int(snap_ts_arr[i])
            state.release_closed(now_ns)

            snap = {
                "snapshot_ts_ns": now_ns,
                "market_slug": str(market_arr[i]),
                "up_token_best_bid": float(up_bid_arr[i]),
                "up_token_best_ask": float(up_ask_arr[i]),
                "t_to_close_s": float(ttc_arr[i]),
            }
            decision = decide(snap=snap, p_up=float(p_up_all[i]), state=state)
            reason_counts[decision["reason"]] += 1

            # Log: every decision in ttc band (we always want them) plus all enters,
            # plus everything if --log-all.
            in_band = TTC_MIN_S <= snap["t_to_close_s"] <= TTC_MAX_S
            should_log = args.log_all or in_band or decision["decision"] == "ENTER"
            if should_log:
                rec = {
                    "snapshot_ts_ns":      now_ns,
                    "market_slug":         snap["market_slug"],
                    "market_close_ts_ns":  int(close_ts_arr[i]),
                    "t_to_close_s":        snap["t_to_close_s"],
                    "up_token_best_bid":   snap["up_token_best_bid"],
                    "up_token_best_ask":   snap["up_token_best_ask"],
                    "p_up":                decision["p_up"],
                    "edge_dn":             decision["edge_dn"],
                    "decision":            decision["decision"],
                    "reason":              decision["reason"],
                    "notional":            decision["notional"],
                    "margin_in_use_before": state.margin_in_use,
                }
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n_logged += 1

            if decision["decision"] == "ENTER":
                n_entered += 1
                entry_price = 1.0 - snap["up_token_best_bid"]   # DOWN ask
                shares = decision["notional"] / entry_price
                d_str = datetime.fromtimestamp(now_ns / 1e9, UTC).date().isoformat()
                fc = fee_calcs.setdefault(d_str, FeeCalculator.for_date(d_str))
                fee = fc.taker_fee_usd(price=entry_price, size=shares)
                # If we have the resolved label, compute realized PnL (shadow can
                # only do this when running on historical/resolved markets; the
                # true-live bot will track via position_manager instead)
                if resolved_arr[i] >= 0:
                    win = (resolved_arr[i] == 0)   # DOWN wins
                    payoff = shares * 1.0 if win else 0.0
                    pnl = payoff - decision["notional"] - fee
                    realized_pnl += pnl
                    n_realized += 1
                state.add_lot(close_ns=int(close_ts_arr[i]), notional=decision["notional"])
                state.positions_per_market[snap["market_slug"]] += 1
                state.last_entry_ns_per_market[snap["market_slug"]] = now_ns

    elapsed = (datetime.now(UTC) - t0).total_seconds()
    print(f"\n=== SHADOW REPLAY DONE ===")
    print(f"  source           : {args.source_parquet}")
    print(f"  output           : {args.output}")
    print(f"  processed rows   : {n_total:,}")
    print(f"  logged rows      : {n_logged:,}")
    print(f"  enters           : {n_entered}")
    print(f"  realized fills   : {n_realized}  (only for markets with resolved label)")
    print(f"  realized PnL     : ${realized_pnl:+.2f}")
    print(f"  elapsed          : {elapsed:.1f}s ({n_total/max(elapsed,1e-3):,.0f} rows/s)")
    print(f"\n  decision reasons (top 10):")
    for reason, cnt in sorted(reason_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {reason:<25} {cnt:>7,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
