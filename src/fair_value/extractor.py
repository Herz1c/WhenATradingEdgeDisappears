"""Fair-value feature extraction: Polymarket L2 + Binance spot + RTDS Chainlink.

SINGLE SOURCE OF TRUTH — the offline builder (tools/build_fair_value_dataset.py)
and the live bot both drive the same state machine, so train ≈ live.

Three sub-states, fed by an interleaved time-ordered frame stream:
  * pm  : Polymarket book state (delegated to poly_l2_only.extractor — proven)
  * bin : Binance spot top-of-book + trade flow (the LEADING indicator)
  * cl  : Chainlink nowcast state — latest RTDS price + a CAUSAL EMA of the
          basis (rtds_price − binance_mid@chainlink_ts), so we can nowcast the
          Chainlink price ahead of RTDS's own ~2.7 s delivery lag.

Leakage rule: every read uses only frames already applied (recv_ts ≤ emit time).
recv_ts_ns in the recorded data already encodes real feed latency, so emitting
on a wall-clock grid using recv_ts ordering reproduces what live would have seen.

Thesis (docs/fair_value_dataset_architecture.md): Chainlink cexprice is an
aggregate of CEX prices, so Binance leads it; the calibrated nowcast resolves the
outcome before RTDS delivers it and before the Polymarket book reprices.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

from poly_l2_only.extractor import (  # proven PM state machine
    MarketState,
    state_to_features as pm_state_to_features,
    update_state as pm_update_state,
)

NS = 1_000_000_000
SQRT2 = math.sqrt(2.0)

# ---- emit gating / config (overridable by the builder) ----------------------
BINANCE_STALE_S = 10.0     # last Binance bookTicker must be within this of emit
RTDS_MAX_AGE_S = 60.0      # latest RTDS recv must be within this of emit
RTDS_SOURCE_MAX_AGE_S = 60.0  # latest Chainlink publish timestamp must also be fresh
SEC_MID_MAXLEN = 45        # ~45 s of 1-s-sampled Binance mids
BASIS_ALPHA = 0.02         # EMA weight for the Binance→Chainlink basis
VOL_WIN_S = 15             # window for realized vol used in delta_over_vol
CROSS_WIN_S = 30           # window for strike-crossing / time-above features


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def _f(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ----- Binance state ---------------------------------------------------------

@dataclass
class BinanceState:
    last_bid: float = 0.0
    last_ask: float = 0.0
    last_bid_qty: float = 0.0
    last_ask_qty: float = 0.0
    last_mid: float = 0.0
    last_micro: float = 0.0
    last_ts_ns: int = 0

    # 1-second-sampled mid history for returns/vol/crossings
    sec_mids: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=SEC_MID_MAXLEN))
    _last_sec: int = -1

    # signed-trade-flow windows (aggressor: +1 buy, -1 sell)
    tr1: Deque[Tuple[int, float]] = field(default_factory=deque)   # (ts_ns, signed_qty)
    tr5: Deque[Tuple[int, float]] = field(default_factory=deque)
    sig1: float = 0.0
    sig5: float = 0.0
    absv1: float = 0.0
    absv5: float = 0.0

    def update_bookticker(self, ts_ns: int, bid: float, ask: float,
                          bid_qty: float, ask_qty: float) -> None:
        if not (bid > 0 and ask > 0):
            return
        self.last_bid = bid
        self.last_ask = ask
        self.last_bid_qty = bid_qty
        self.last_ask_qty = ask_qty
        self.last_mid = (bid + ask) * 0.5
        denom = bid_qty + ask_qty
        # microprice: ask-weighted by bid size and vice-versa
        self.last_micro = ((bid * ask_qty + ask * bid_qty) / denom) if denom > 0 else self.last_mid
        self.last_ts_ns = ts_ns
        sec = ts_ns // NS
        if sec != self._last_sec:
            self.sec_mids.append((sec, self.last_mid))
            self._last_sec = sec

    def update_trade(self, ts_ns: int, qty: float, side: float) -> None:
        # 1 s window
        cutoff = ts_ns - NS
        dq = self.tr1
        while dq and dq[0][0] < cutoff:
            _, sq = dq.popleft()
            self.sig1 -= sq
            self.absv1 -= abs(sq)
        sq = side * qty
        dq.append((ts_ns, sq))
        self.sig1 += sq
        self.absv1 += abs(sq)
        # 5 s window
        cutoff5 = ts_ns - 5 * NS
        dq5 = self.tr5
        while dq5 and dq5[0][0] < cutoff5:
            _, osq = dq5.popleft()
            self.sig5 -= osq
            self.absv5 -= abs(osq)
        dq5.append((ts_ns, sq))
        self.sig5 += sq
        self.absv5 += abs(sq)

    def _decay(self, now_ns: int) -> None:
        cutoff = now_ns - NS
        while self.tr1 and self.tr1[0][0] < cutoff:
            _, sq = self.tr1.popleft(); self.sig1 -= sq; self.absv1 -= abs(sq)
        cutoff5 = now_ns - 5 * NS
        while self.tr5 and self.tr5[0][0] < cutoff5:
            _, sq = self.tr5.popleft(); self.sig5 -= sq; self.absv5 -= abs(sq)

    def mid_at(self, target_ts_ns: int) -> float:
        """Most recent 1-s-sampled mid at or before target time (for basis)."""
        target_sec = target_ts_ns // NS
        best = 0.0
        for sec, mid in self.sec_mids:
            if sec <= target_sec:
                best = mid
            else:
                break
        return best or self.last_mid

    def mid_n_sec_ago(self, now_sec: int, n: int) -> float:
        want = now_sec - n
        best = 0.0
        for sec, mid in self.sec_mids:
            if sec <= want:
                best = mid
            else:
                break
        return best

    def realized_vol_frac(self, now_sec: int, win: int) -> float:
        """Std of 1-s log returns over the last `win` seconds (fractional)."""
        pts = [(s, m) for (s, m) in self.sec_mids if s > now_sec - win and m > 0]
        if len(pts) < 3:
            return 0.0
        rets = []
        for i in range(1, len(pts)):
            p0, p1 = pts[i - 1][1], pts[i][1]
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)


# ----- Chainlink nowcast state ----------------------------------------------

@dataclass
class ChainlinkState:
    latest_price: float = 0.0
    latest_cl_ts_ns: int = 0
    latest_recv_ns: int = 0
    basis_ema: Optional[float] = None
    alpha: float = BASIS_ALPHA
    sec_prices: Deque[Tuple[int, float]] = field(default_factory=lambda: deque(maxlen=SEC_MID_MAXLEN))
    _last_sec: int = -1

    def update(self, recv_ts_ns: int, cl_ts_ns: int, price: float,
               binance: BinanceState) -> None:
        if not (price > 0):
            return
        self.latest_price = price
        self.latest_cl_ts_ns = cl_ts_ns
        self.latest_recv_ns = recv_ts_ns
        sec = recv_ts_ns // NS
        if sec != self._last_sec:
            self.sec_prices.append((sec, price))
            self._last_sec = sec
        # basis = chainlink price − Binance mid AT the chainlink timestamp
        mid_then = binance.mid_at(cl_ts_ns)
        if mid_then > 0:
            basis = price - mid_then
            if self.basis_ema is None:
                self.basis_ema = basis
            else:
                self.basis_ema += self.alpha * (basis - self.basis_ema)

    def price_n_sec_ago(self, now_sec: int, n: int) -> float:
        want = now_sec - n
        best = 0.0
        for sec, price in self.sec_prices:
            if sec <= want:
                best = price
            else:
                break
        return best

    def realized_vol_frac(self, now_sec: int, win: int) -> float:
        pts = [(s, p) for (s, p) in self.sec_prices if s > now_sec - win and p > 0]
        if len(pts) < 3:
            return 0.0
        rets = []
        for i in range(1, len(pts)):
            p0, p1 = pts[i - 1][1], pts[i][1]
            if p0 > 0 and p1 > 0:
                rets.append(math.log(p1 / p0))
        if len(rets) < 2:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        return math.sqrt(var)


# ----- combined state --------------------------------------------------------

@dataclass
class FairValueState:
    pm: MarketState = field(default_factory=MarketState)
    bin: BinanceState = field(default_factory=BinanceState)
    cl: ChainlinkState = field(default_factory=ChainlinkState)
    strike: float = 0.0
    open_s: int = 0
    close_s: int = 0
    require_cex: bool = True

    # ---- update entry points (typed core; dict wrappers for live) ----
    def update_pm(self, frame: Dict[str, Any]) -> bool:
        return pm_update_state(self.pm, frame)

    def update_binance_bookticker(self, ts_ns: int, bid: float, ask: float,
                                  bid_qty: float, ask_qty: float) -> None:
        self.bin.update_bookticker(ts_ns, bid, ask, bid_qty, ask_qty)

    def update_binance_trade(self, ts_ns: int, qty: float, side: float) -> None:
        self.bin.update_trade(ts_ns, qty, side)

    def update_rtds(self, recv_ts_ns: int, cl_ts_ns: int, price: float) -> None:
        self.cl.update(recv_ts_ns, cl_ts_ns, price, self.bin)

    def update_binance_frame(self, rec: Dict[str, Any]) -> None:
        """Live convenience: parse a raw Binance ws record."""
        et = rec.get("event_type")
        ts = int(rec.get("recv_ts_ns") or 0)
        p = rec.get("payload") or {}
        if et == "bookTicker":
            self.update_binance_bookticker(ts, _f(p.get("b")), _f(p.get("a")),
                                           _f(p.get("B")), _f(p.get("A")))
        elif et == "aggTrade":
            # m=True -> buyer is maker -> aggressor SELL (-1)
            side = -1.0 if p.get("m") else 1.0
            self.update_binance_trade(ts, _f(p.get("q")), side)

    def update_rtds_frame(self, rec: Dict[str, Any]) -> None:
        try:
            self.update_rtds(int(rec["recv_ts_ns"]), int(rec["chainlink_ts_ms"]) * 1_000_000,
                             float(rec["btc_usd_price"]))
        except (KeyError, TypeError, ValueError):
            pass

    # ---- feature emit ----
    def features(self, now_ns: int) -> Optional[Dict[str, float]]:
        """Return the full feature row at `now_ns`, or None if a required source
        is stale/absent (gap-safe — caller simply skips the row)."""
        b = self.bin
        cl = self.cl
        # ---- gating ----
        if self.strike <= 0:
            return None
        if cl.latest_recv_ns == 0 or (now_ns - cl.latest_recv_ns) > RTDS_MAX_AGE_S * NS:
            return None
        rtds_age_s = (now_ns - cl.latest_cl_ts_ns) / NS
        if cl.latest_cl_ts_ns == 0 or rtds_age_s > RTDS_SOURCE_MAX_AGE_S:
            return None
        if self.require_cex:
            if b.last_ts_ns == 0 or (now_ns - b.last_ts_ns) > BINANCE_STALE_S * NS:
                return None
            if cl.basis_ema is None or b.last_mid <= 0:
                return None
        pm = pm_state_to_features(self.pm, now_ns)
        if not (pm["up_mid"] > 0 and pm["down_mid"] > 0):
            return None

        now_sec = now_ns // NS
        ttc_s = max(0.0, (self.close_s * NS - now_ns) / NS)

        # ---- oracle / fair-value core ----
        delta_rtds = cl.latest_price - self.strike
        if self.require_cex:
            b._decay(now_ns)
            nowcast = b.last_mid + (cl.basis_ema or 0.0)
            vol_frac = b.realized_vol_frac(now_sec, VOL_WIN_S)
            vol_price = b.last_mid
            basis = cl.basis_ema or 0.0
        else:
            # PM+RTDS mode deliberately has no Binance/Coinbase dependency.
            # The legacy "nowcast" field names are kept for artifact
            # compatibility, but here they are pure RTDS values.
            nowcast = cl.latest_price
            vol_frac = cl.realized_vol_frac(now_sec, VOL_WIN_S)
            vol_price = cl.latest_price
            basis = 0.0
        delta_nowcast = nowcast - self.strike
        sigma_dollar = vol_frac * vol_price        # $ per sqrt(second)
        horizon_sd = sigma_dollar * math.sqrt(ttc_s) if ttc_s > 0 else 0.0
        if horizon_sd > 1e-9:
            delta_over_vol = delta_nowcast / horizon_sd
        else:
            # essentially no remaining time/vol: outcome ~ determined by sign
            delta_over_vol = math.copysign(8.0, delta_nowcast) if delta_nowcast != 0 else 0.0
        delta_over_vol = max(-8.0, min(8.0, delta_over_vol))
        p_bs = _phi(delta_over_vol)

        # ---- CEX/RTDS microstructure ----
        ret1 = ret3 = ret5 = ret10 = 0.0
        m0 = b.last_mid if self.require_cex else cl.latest_price
        for n in (1, 3, 5, 10):
            mn = b.mid_n_sec_ago(now_sec, n) if self.require_cex else cl.price_n_sec_ago(now_sec, n)
            r = ((m0 - mn) / mn) if mn > 0 else 0.0
            if n == 1: ret1 = r
            elif n == 3: ret3 = r
            elif n == 5: ret5 = r
            else: ret10 = r
        if self.require_cex:
            bdenom = b.last_bid_qty + b.last_ask_qty
            btc_depth_imb = ((b.last_bid_qty - b.last_ask_qty) / bdenom) if bdenom > 0 else 0.0
            ofi1 = (b.sig1 / b.absv1) if b.absv1 > 0 else 0.0
            ofi5 = (b.sig5 / b.absv5) if b.absv5 > 0 else 0.0
        else:
            btc_depth_imb = ofi1 = ofi5 = 0.0

        # time-above/below strike + crossings over CROSS_WIN_S
        above = below = 0
        crossings = 0
        prev_side = 0
        price_points = b.sec_mids if self.require_cex else cl.sec_prices
        for sec, mid in price_points:
            if sec <= now_sec - CROSS_WIN_S:
                continue
            s = 1 if mid >= self.strike else -1
            if s > 0: above += 1
            else: below += 1
            if prev_side != 0 and s != prev_side:
                crossings += 1
            prev_side = s
        tot_sec = above + below
        frac_above = (above / tot_sec) if tot_sec else 0.0

        # ---- the explicit mispricing / arb signal ----
        implied_p_up = pm["implied_p_up"]
        book_vs_oracle_gap = implied_p_up - p_bs

        # ---- assemble (PM features + the new groups) ----
        feats: Dict[str, float] = dict(pm)
        feats.update({
            # oracle core
            "delta_to_strike_rtds": delta_rtds,
            "rtds_age_s": rtds_age_s,
            "synthetic_chainlink_nowcast": nowcast,
            "delta_to_strike_nowcast": delta_nowcast,
            "basis_binance_chainlink": basis,
            "btc_realized_vol_15s": vol_frac,
            "delta_over_vol": delta_over_vol,
            "p_bs": p_bs,
            # CEX microstructure (source configurable: binance/coinbase/...)
            "cex_mid": b.last_mid if self.require_cex else cl.latest_price,
            "cex_microprice_tilt": (b.last_micro - b.last_mid) if self.require_cex else 0.0,
            "cex_spread": (b.last_ask - b.last_bid) if self.require_cex else 0.0,
            "btc_ret_1s": ret1, "btc_ret_3s": ret3, "btc_ret_5s": ret5, "btc_ret_10s": ret10,
            "btc_depth_imbalance": btc_depth_imb,
            "btc_ofi_1s": ofi1, "btc_ofi_5s": ofi5,
            "time_frac_above_strike_30s": frac_above,
            "strike_crossings_30s": float(crossings),
            # arb signal
            "book_vs_oracle_gap": book_vs_oracle_gap,
            "strike": self.strike,
        })
        return feats
