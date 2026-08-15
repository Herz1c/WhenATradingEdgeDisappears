"""Decision engine — copy of src/live/shadow_runtime.py logic, exposed
as a clean function for the live bot. The numbers below MUST stay in
lockstep with LiveTradingBotPlan.md §1 and the shadow runtime."""
from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field


def _is_valid_price(x) -> bool:
    """True iff x is a finite float, not None, not NaN, not Inf."""
    if x is None: return False
    try:
        f = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)

# Edge threshold for ENTER. Env-overridable so the maker strategy (which
# extracts edge from a midpoint-priced limit order, not from a stale taker
# fill) can run at a lower threshold without a code change. OOS sweep on
# 2026-05-{21,22,26,29} with midpoint maker shows total PnL is essentially
# flat across thr 0.02-0.20 (the model has consistent edge across that
# band); lowering from 0.30 to 0.05-0.07 ~doubles fill volume while
# keeping per-day PnL the same or slightly higher.
THRESHOLD                    = float(os.environ.get("LIVE_BOT_EDGE_THRESHOLD", "0.3067"))
EDGE_K                       = 8.0
# Sizing bounds (env-var overridable for paper-trade / micro-live).
# Set MIN==MAX to lock a constant size regardless of edge magnitude.
MIN_NOTIONAL                 = float(os.environ.get("LIVE_BOT_MIN_NOTIONAL", "1.00"))
MAX_NOTIONAL                 = float(os.environ.get("LIVE_BOT_MAX_NOTIONAL", "2.00"))
# Never buy the DOWN token when it is a cheap longshot priced below this.
# Uses the real DOWN best-ask when available, else the up-book-implied price.
# Raised from 0.15 -> 0.30 after the OOS fill audit showed the [0, 0.20]
# entry-price band had 0 wins / 14 fills (pure -PnL contribution), and the
# [0.20, 0.30] band was 28% win and net negative too. Below ~0.30 the
# model's edge collapses -- those signals only fire on markets that have
# already moved so far that DOWN is implausible.
MIN_DOWN_PRICE               = float(os.environ.get("LIVE_BOT_MIN_DOWN_PRICE", "0.40"))
# Also a floor for the UP side -- never buy UP token cheaper than this.
MIN_UP_PRICE                 = float(os.environ.get("LIVE_BOT_MIN_UP_PRICE", "0.40"))
# Upper caps -- never buy a near-certain token: payoff per share is too
# small to justify the fee + slippage. Symmetric on both sides.
MAX_DOWN_PRICE               = float(os.environ.get("LIVE_BOT_MAX_DOWN_PRICE", "0.90"))
MAX_UP_PRICE                 = float(os.environ.get("LIVE_BOT_MAX_UP_PRICE", "0.90"))
# Refined-rule filter #1 (data-driven, from OOS analysis 2026-05-21..29):
# UP-side entries with up_ask in [0.36, 0.46) lost money badly across all
# four test days (40% win rate, -$9.25 PnL contribution at $1 size). Block
# the band entirely. Env-overridable so it can be disabled for A/B testing.
SKIP_UP_PRICE_BAND_LO        = float(os.environ.get("LIVE_BOT_SKIP_UP_PRICE_LO", "0.36"))
SKIP_UP_PRICE_BAND_HI        = float(os.environ.get("LIVE_BOT_SKIP_UP_PRICE_HI", "0.46"))
# Refined-rule filter #2: when raw delta-to-strike (binance_spot_mid -
# price_to_beat) lands in the [+$51, +$86] band, the market is "moderately
# above strike" -- close enough that noise flips the outcome. In OOS, win
# rate cratered to 27-39% in these buckets. Skip any-side entries here.
# Sign-aware: a positive delta means BTC is above strike, negative means
# below. The bad zone observed empirically was the POSITIVE band; we apply
# the same magnitude to the negative side as a symmetric safety guard.
SKIP_DELTA_MAG_LO            = float(os.environ.get("LIVE_BOT_SKIP_DELTA_LO", "51.0"))
SKIP_DELTA_MAG_HI            = float(os.environ.get("LIVE_BOT_SKIP_DELTA_HI", "86.0"))
MAX_POSITIONS_PER_MARKET     = int(os.environ.get("LIVE_BOT_MAX_POS_PER_MARKET", "2"))
# Per-(market, side) cap. Default 1 = at most one UP and one DOWN per market.
# Combined with MAX_POSITIONS_PER_MARKET=2 this gives "max 2 entries per
# market, max 1 per side". User-tunable for asymmetric configurations.
MAX_POSITIONS_PER_SIDE       = int(os.environ.get("LIVE_BOT_MAX_POS_PER_SIDE", "1"))
MIN_SECONDS_BETWEEN_ENTRIES  = float(os.environ.get("LIVE_BOT_MIN_GAP_SECONDS", "10.0"))
MAX_MARGIN_USD               = float(os.environ.get("LIVE_BOT_MAX_MARGIN_USD", "30.00"))
TTC_MIN_S                    = 10.0
TTC_MAX_S                    = 60.0
# Bias-freshness gate: bot must have loaded a canonical bias whose
# anchor second is within this many seconds of NOW, otherwise refuse
# to trade. Empirically the backtest still produces sensible p_up
# with bias up to 3 days old (chainlink_public_delayed has been broken
# since 2026-05-24 and canonical falls back to bias_carried_forward).
# 172_800s = 2 days (conservative — backtest tested up to 3 days).
MAX_BIAS_AGE_S               = float(os.environ.get("LIVE_BOT_MAX_BIAS_AGE_S", "172800"))
SECOND_NS                    = 1_000_000_000


@dataclass
class RiskState:
    margin_in_use: float = 0.0
    positions_per_market: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Per-(market_slug, side) counter -- enforces "at most 1 UP + at most 1
    # DOWN per market" when MAX_POSITIONS_PER_SIDE=1.
    positions_per_market_side: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    last_entry_ns_per_market: dict[str, int] = field(default_factory=dict)
    # (close_ts_ns, notional) — released when current ns >= close_ts
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


def size_for_edge(edge_dn: float) -> float:
    raw = 1.0 + EDGE_K * max(0.0, edge_dn - THRESHOLD)
    return min(MAX_NOTIONAL, max(MIN_NOTIONAL, raw))


def decide(*, snap: dict, p_up: float, state: RiskState,
           bias_age_seconds: float | None = None) -> dict:
    """Returns a decision dict. New keys vs the legacy DOWN-only contract:
      - `side`  : "UP" or "DOWN" (or absent when decision != "ENTER")
      - `edge`  : the side-specific edge value used to gate this trade
      - `entry_price`: the ask of the chosen side (= the taker fill price)

    Legacy keys kept for backwards compatibility:
      - `edge_dn`: down-side edge (always populated when edge_dn is computed,
                   even if we ended up entering UP-side, so the existing
                   decisions log doesn't lose history)
      - `down_price`: when the DOWN-side is the chosen side (or considered
                      but rejected by floor)

    `bias_age_seconds` = wall-clock seconds since the canonical parquet's
    most recent valid second. None means "never loaded". The bot's
    feature_runtime passes this on every call.  We refuse to ENTER if
    bias is missing or older than MAX_BIAS_AGE_S — this prevents the
    2026-05-26 failure mode where the bot traded for hours on the
    wrong synthetic price.
    """
    out: dict = {"decision": "skip", "reason": "", "notional": 0.0,
                 "edge_dn": None, "p_up": float(p_up) if _is_valid_price(p_up) else None}

    if bias_age_seconds is None:
        out["reason"] = "bias_not_loaded"; return out
    if bias_age_seconds > MAX_BIAS_AGE_S:
        out["reason"] = f"bias_stale_{int(bias_age_seconds)}s"; return out

    up_bid = snap.get("up_token_best_bid")
    up_ask = snap.get("up_token_best_ask")
    if not (_is_valid_price(up_bid) and _is_valid_price(up_ask)):
        out["reason"] = "no_quote"; return out
    up_bid = float(up_bid); up_ask = float(up_ask)
    if up_ask <= 0.01 or up_bid >= 0.99:
        out["reason"] = "degenerate_quote"; return out

    if not _is_valid_price(p_up):
        out["reason"] = "bad_p_up"; return out
    p_up = float(p_up); out["p_up"] = p_up

    ttc = float(snap["t_to_close_s"])
    if not (TTC_MIN_S <= ttc <= TTC_MAX_S):
        out["reason"] = "ttc_band"; return out

    # ── Compute side-specific edges ─────────────────────────────────────
    edge_dn = float(up_bid) - float(p_up)            # buy DOWN edge: up_bid - p_up
    edge_up = float(p_up)    - float(up_ask)         # buy UP   edge: p_up - up_ask
    out["edge_dn"] = edge_dn
    out["edge_up"] = edge_up

    # Pick whichever side has the larger positive edge over THRESHOLD.
    side: str | None = None
    if edge_up >= THRESHOLD and edge_up >= edge_dn:
        side = "UP"
    elif edge_dn >= THRESHOLD:
        side = "DOWN"
    if side is None:
        out["reason"] = "below_threshold"; return out
    out["side"] = side
    out["edge"] = edge_up if side == "UP" else edge_dn

    # ── Refined rule #2: skip if raw delta-to-strike is in the
    #    "danger zone" [+/-$51, +/-$86]. From the OOS audit on 2026-05-21..29
    #    this band has 27-39% win rate -- noisy markets where BTC is close-
    #    but-not-fixed-on a side. Sign-symmetric.
    binance_mid = snap.get("binance_spot_mid")
    strike      = snap.get("price_to_beat")
    if _is_valid_price(binance_mid) and _is_valid_price(strike):
        raw_delta_mag = abs(float(binance_mid) - float(strike))
        out["raw_delta"] = float(binance_mid) - float(strike)
        if SKIP_DELTA_MAG_LO <= raw_delta_mag <= SKIP_DELTA_MAG_HI:
            out["reason"] = (f"skip_danger_delta_{SKIP_DELTA_MAG_LO:.0f}"
                             f"_{SKIP_DELTA_MAG_HI:.0f}_({raw_delta_mag:.2f})")
            return out

    # ── Side-specific price floors and the UP-side band filter ─────────
    if side == "UP":
        entry_price = up_ask
        out["entry_price"] = entry_price
        # Refined rule #1: skip UP when up_token_best_ask is in [0.36, 0.46).
        # This band lost money on every test day in OOS (40% win rate).
        if SKIP_UP_PRICE_BAND_LO <= entry_price < SKIP_UP_PRICE_BAND_HI:
            out["reason"] = (f"skip_up_band_{SKIP_UP_PRICE_BAND_LO:.2f}"
                             f"_{SKIP_UP_PRICE_BAND_HI:.2f}_({entry_price:.3f})")
            return out
        if entry_price < MIN_UP_PRICE:
            out["reason"] = f"up_below_{MIN_UP_PRICE:.2f}_{entry_price:.3f}"; return out
        if entry_price > MAX_UP_PRICE:
            out["reason"] = f"up_above_{MAX_UP_PRICE:.2f}_{entry_price:.3f}"; return out
    else:   # DOWN
        down_ask = snap.get("down_token_best_ask")
        down_price = float(down_ask) if _is_valid_price(down_ask) else (1.0 - up_bid)
        out["down_price"] = down_price
        out["entry_price"] = down_price
        if down_price < MIN_DOWN_PRICE:
            out["reason"] = f"down_below_{MIN_DOWN_PRICE:.2f}_{down_price:.3f}"; return out
        if down_price > MAX_DOWN_PRICE:
            out["reason"] = f"down_above_{MAX_DOWN_PRICE:.2f}_{down_price:.3f}"; return out

    # ── Per-market caps / gap ─────────────────────────────────────────
    mk = str(snap["market_slug"])
    if state.positions_per_market[mk] >= MAX_POSITIONS_PER_MARKET:
        out["reason"] = "market_cap_full"; return out
    # Per-(market, side) cap -- "at most 1 UP and 1 DOWN per market"
    if state.positions_per_market_side[(mk, side)] >= MAX_POSITIONS_PER_SIDE:
        out["reason"] = f"market_side_cap_full_{side.lower()}"; return out

    snap_ts = int(snap["snapshot_ts_ns"])
    last = state.last_entry_ns_per_market.get(mk)
    if last is not None and (snap_ts - last) < int(MIN_SECONDS_BETWEEN_ENTRIES * SECOND_NS):
        out["reason"] = "within_gap"; return out

    notional = size_for_edge(out["edge"])
    avail = MAX_MARGIN_USD - state.margin_in_use
    if avail < MIN_NOTIONAL:
        out["reason"] = "margin_full"; return out
    if notional > avail:
        notional = avail

    out["decision"] = "ENTER"
    out["reason"]   = "ENTER"
    out["notional"] = float(notional)
    return out
