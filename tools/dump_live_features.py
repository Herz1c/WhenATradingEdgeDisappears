"""Read-only live feature diagnostic. Connects to the Polymarket market WS the
same way the bot does, builds book state, and for every in-window tick computes
the 55-feature vector + model p_up + the would-be decision. NO ORDERS.

Replicates the FIXED feature path: market close = slug(open) + 300, and the
ttc_s feature is taken from that close (re-asserted after update_state, since
the direct-WS frames don't carry market_close_s).

Prints the first few full feature vectors, then a summary so we can compare
live behaviour (entry rate, p_up level, DOWN/UP skew) against the backtest.

Run:
  py -3 -u tools/dump_live_features.py --minutes 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from live_bot.poly_ws_direct import PolyDirectWS  # noqa: E402
from poly_l2_only.extractor import (  # noqa: E402
    EMIT_EVENT_TYPES, MarketState, NS_PER_S, state_to_features, update_state,
)
from poly_l2_only_bot.discovery import BTCUpDownDiscovery, DiscoveredMarket  # noqa: E402
from poly_l2_only_bot.predictor import V2Predictor  # noqa: E402
from poly_l2_only_bot.strategy import (  # noqa: E402
    FILL_MAX, FILL_MIN, MIN_EDGE, TTC_MAX_S, TTC_MIN_S,
)

ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"


class Ctx:
    __slots__ = ("slug", "close_s", "close_ns", "open_s", "up_id", "down_id", "state", "entered")
    def __init__(self, dm: DiscoveredMarket):
        self.slug = dm.market_slug
        self.close_s = dm.market_close_s
        self.open_s = dm.market_open_s
        self.close_ns = dm.market_close_s * NS_PER_S
        self.up_id = dm.up_asset_id
        self.down_id = dm.down_asset_id
        self.state = MarketState()
        self.state.event_slug = dm.market_slug
        self.state.market_open_s = dm.market_open_s
        self.state.market_close_s = dm.market_close_s
        self.state.asset_to_side[dm.up_asset_id] = "Up"
        self.state.asset_to_side[dm.down_asset_id] = "Down"
        self.entered = False


class Diag:
    def __init__(self, minutes: float, max_samples: int):
        self.pred = V2Predictor(ART)
        self.by_slug: Dict[str, Ctx] = {}
        self.by_asset: Dict[str, Ctx] = {}
        self.samples = []          # (ttc, p_up, up_ask, dn_ask, ev_up, ev_dn, side_or_None)
        self.full_dumps = 0
        self.cnt = Counter()
        self.deadline = time.time() + minutes * 60
        self.max_samples = max_samples
        self.shutdown = asyncio.Event()
        self.ws = PolyDirectWS(on_l2=self._on_frame)
        self.disc = BTCUpDownDiscovery(on_new_market=self._register)

    async def _register(self, dm: DiscoveredMarket):
        if dm.market_slug in self.by_slug:
            return
        c = Ctx(dm)
        self.by_slug[dm.market_slug] = c
        self.by_asset[dm.up_asset_id] = c
        self.by_asset[dm.down_asset_id] = c
        self.ws.add_market(market_slug=dm.market_slug, market_id=dm.market_id,
                           condition_id=dm.condition_id, up_asset_id=dm.up_asset_id,
                           down_asset_id=dm.down_asset_id)

    def _on_frame(self, frame: dict):
        self.cnt["frames"] += 1
        et = frame.get("event_type")
        if et not in EMIT_EVENT_TYPES:
            self.cnt["bad_event"] += 1
            return
        slug = frame.get("selected_market_slug")
        c = self.by_slug.get(slug) if slug else None
        if c is None:
            aid = frame.get("token_asset_id")
            c = self.by_asset.get(str(aid)) if aid else None
        if c is None:
            self.cnt["no_ctx"] += 1
            return
        if not update_state(c.state, frame):
            self.cnt["update_fail"] += 1
            return
        # FIX: re-assert correct close (frames don't carry market_close_s).
        c.state.market_close_s = c.close_s
        c.state.market_open_s = c.open_s
        ts_ns = int(frame.get("recv_ts_ns") or time.time_ns())
        ttc = (c.close_ns - ts_ns) / NS_PER_S
        if ttc >= TTC_MAX_S:
            self.cnt["too_early"] += 1
            return
        if ttc < TTC_MIN_S:
            self.cnt["too_late"] += 1
            return
        self.cnt["in_window"] += 1
        feats = state_to_features(c.state, ts_ns)
        self.pred.fill_features(feats)
        p_up = self.pred.predict_one()
        up_ask = feats.get("up_best_ask", 0.0)
        dn_ask = feats.get("down_best_ask", 0.0)
        ev_up = (p_up - up_ask) if 0 < up_ask < 1 else -1
        ev_dn = ((1 - p_up) - dn_ask) if 0 < dn_ask < 1 else -1
        side = None
        if ev_up >= ev_dn and ev_up >= MIN_EDGE and FILL_MIN <= up_ask <= FILL_MAX:
            side = "UP"
        elif ev_dn > ev_up and ev_dn >= MIN_EDGE and FILL_MIN <= dn_ask <= FILL_MAX:
            side = "DOWN"
        self.samples.append((round(ttc, 1), round(p_up, 4), round(up_ask, 3),
                             round(dn_ask, 3), round(ev_up, 3), round(ev_dn, 3), side))
        if side and not c.entered:
            c.entered = True
        if self.full_dumps < 3:
            self.full_dumps += 1
            order = self.pred.feature_order
            nz = sum(1 for k in order if feats.get(k, 0.0))
            print(f"\n=== FULL FEATURE DUMP #{self.full_dumps} | {c.slug} | ttc={ttc:.1f} "
                  f"p_up={p_up:.4f} | nonzero={nz}/{len(order)} ===", flush=True)
            print(json.dumps({k: round(float(feats.get(k, 0.0)), 5) for k in order}, indent=0), flush=True)
        if len(self.samples) >= self.max_samples or time.time() > self.deadline:
            self.shutdown.set()

    async def _heartbeat(self):
        while not self.shutdown.is_set():
            await asyncio.sleep(20)
            print(f"[hb] markets={len(self.by_slug)} ws={dict(self.ws.stats)} "
                  f"frame_breakdown={dict(self.cnt)}", flush=True)

    async def run(self):
        tasks = [asyncio.create_task(self.disc.run()), asyncio.create_task(self.ws.run()),
                 asyncio.create_task(self._heartbeat())]
        try:
            await asyncio.wait_for(self.shutdown.wait(),
                                   timeout=max(1, self.deadline - time.time()))
        except asyncio.TimeoutError:
            pass
        self.disc.stop(); self.ws.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._summary()

    def _summary(self):
        s = self.samples
        print(f"\n================ SUMMARY ({len(s)} in-window ticks) ================")
        if not s:
            print("no in-window ticks captured (run longer / off-peak)"); return
        pu = [x[1] for x in s]
        markets = len(self.by_slug)
        entered = [c for c in self.by_slug.values() if c.entered]
        sides = Counter(x[6] for x in s if x[6])
        ent_sides = Counter()
        for c in self.by_slug.values():
            if c.entered:
                # first qualifying side
                pass
        import statistics as st
        print(f"markets registered      : {markets}")
        print(f"markets that would ENTER: {len(entered)}  ({100*len(entered)/max(1,markets):.0f}% of markets)")
        print(f"p_up  mean={st.mean(pu):.3f}  min={min(pu):.3f}  max={max(pu):.3f}  "
              f"median={st.median(pu):.3f}")
        print(f"frac p_up<0.5 (DOWN-lean): {100*sum(1 for v in pu if v<0.5)/len(pu):.0f}%")
        print(f"qualifying-tick side split: {dict(sides)}")
        print(f"ttc range of ticks: {min(x[0] for x in s):.0f}..{max(x[0] for x in s):.0f}s")
        print("\nfirst 12 in-window ticks (ttc, p_up, up_ask, dn_ask, ev_up, ev_dn, side):")
        for x in s[:12]:
            print(f"  ttc={x[0]:>4} p_up={x[1]:.3f} up_ask={x[2]:.2f} dn_ask={x[3]:.2f} "
                  f"ev_up={x[4]:+.3f} ev_dn={x[5]:+.3f} -> {x[6] or '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--max-samples", type=int, default=400)
    args = ap.parse_args()
    print(f"live feature diagnostic: up to {args.minutes} min / {args.max_samples} ticks. "
          f"window ttc ({TTC_MIN_S:.0f},{TTC_MAX_S:.0f}]s, edge>={MIN_EDGE}, fill[{FILL_MIN},{FILL_MAX}]",
          flush=True)
    asyncio.run(Diag(args.minutes, args.max_samples).run())


if __name__ == "__main__":
    main()
