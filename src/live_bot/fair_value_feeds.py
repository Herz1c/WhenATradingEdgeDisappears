"""Direct WebSocket feeds for the fair-value bot — RTDS (Chainlink) + Coinbase.

Mirrors the latency rationale of live_bot.poly_ws_direct: connect straight from
the bot process instead of tailing recorder files, so the oracle (RTDS) and the
leading CEX (Coinbase) reach the decision loop sub-second instead of ~10-18s late
(the file-tail re-decompresses the huge growing L2 file every poll and starves
everything). Connection details copied from the proven recorders:
  * RTDS:     src/polymarket_recorder/rtds_recorder.py
  * Coinbase: src/coinbase_recorder/recorder.py
"""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import Callable, Optional

import orjson
from websockets import connect

from market_recorders.time_utils import utc_now_ns


def _flt(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _iso_to_ns(text: object) -> int | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


def _coinbase_message_ts_ns(msg: dict) -> int:
    # The v2 source-time dataset uses Coinbase's message/payload timestamp as
    # the replay clock. Direct live features must use the same clock, not local
    # receive time, or CEX spread/OFI/returns can drift just enough to flip an
    # otherwise marginal action.
    return _iso_to_ns(msg.get("timestamp")) or utc_now_ns()


class RtdsDirectWS:
    """Polymarket RTDS Chainlink BTC/USD stream — the resolution-price feed.
    Calls on_price(recv_ts_ns, chainlink_ts_ns, price) for every tick."""

    def __init__(self, *, on_price: Callable[[int, int, float], None],
                 logger: logging.Logger | None = None,
                 ws_url: str = "wss://ws-live-data.polymarket.com",
                 topic: str = "crypto_prices_chainlink", symbol: str = "btc/usd",
                 reconnect_min_seconds: float = 1.0, reconnect_max_seconds: float = 30.0,
                 ping_interval_seconds: float = 5.0, recv_timeout_seconds: float = 20.0) -> None:
        self._on_price = on_price
        self.logger = logger or logging.getLogger("rtds_ws")
        self._ws_url = ws_url
        self._topic = topic
        self._symbol = symbol
        self._reconnect_min = reconnect_min_seconds
        self._reconnect_max = reconnect_max_seconds
        self._ping_interval = ping_interval_seconds
        self._recv_timeout = recv_timeout_seconds
        self._shutdown = asyncio.Event()
        self.stats = {"connects": 0, "reconnects": 0, "messages": 0, "last_error": None}

    def stop(self) -> None:
        self._shutdown.set()

    async def run(self) -> None:
        delay = self._reconnect_min
        while not self._shutdown.is_set():
            try:
                await self._consume()
                delay = self._reconnect_min
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["reconnects"] += 1
                self.stats["last_error"] = repr(exc)
                self.logger.warning("rtds_ws error %r — backoff %.1fs", exc, delay)
                await asyncio.sleep(delay + random.random() * 1.5)
                delay = min(self._reconnect_max, max(self._reconnect_min, delay * 2))

    async def _consume(self) -> None:
        sub = {"action": "subscribe", "subscriptions": [{
            "topic": self._topic, "type": "*",
            "filters": orjson.dumps({"symbol": self._symbol}).decode("utf-8")}]}
        async with connect(self._ws_url, open_timeout=15.0, close_timeout=5.0,
                           max_queue=2048, ping_interval=None) as ws:
            await ws.send(orjson.dumps(sub).decode("utf-8"))
            self.stats["connects"] += 1
            self.logger.info("rtds_ws connected (%s %s)", self._topic, self._symbol)
            hb = asyncio.create_task(self._heartbeat(ws))
            try:
                while not self._shutdown.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=self._recv_timeout)
                    except asyncio.TimeoutError:
                        continue
                    if msg in ("PONG", "{}", b"{}"):
                        continue
                    try:
                        m = orjson.loads(msg)
                    except Exception:
                        continue
                    if not isinstance(m, dict):
                        continue
                    payload = m.get("payload") if isinstance(m.get("payload"), dict) else {}
                    val = payload.get("value")
                    cl_ts_ms = payload.get("timestamp") or m.get("timestamp")
                    if val is None or cl_ts_ms is None:
                        continue
                    try:
                        price = float(val)
                        cl_ts_ns = int(cl_ts_ms) * 1_000_000
                    except (TypeError, ValueError):
                        continue
                    self._on_price(utc_now_ns(), cl_ts_ns, price)
                    self.stats["messages"] += 1
            finally:
                hb.cancel()
                try: await hb
                except asyncio.CancelledError: pass

    async def _heartbeat(self, ws) -> None:
        try:
            while not self._shutdown.is_set():
                await asyncio.sleep(self._ping_interval)
                try: await ws.send("PING")
                except Exception: return
        except asyncio.CancelledError:
            return


class CoinbaseDirectWS:
    """Coinbase advanced-trade public WS — ticker (top-of-book) + market_trades.
    Calls on_ticker(ts,bid,ask,bid_qty,ask_qty) and on_trade(ts,qty,side)."""

    def __init__(self, *, on_ticker: Callable[[int, float, float, float, float], None],
                 on_trade: Callable[[int, float, float], None],
                 logger: logging.Logger | None = None,
                 ws_url: str = "wss://advanced-trade-ws.coinbase.com",
                 product_id: str = "BTC-USD",
                 channels: tuple[str, ...] = ("ticker", "market_trades"),
                 reconnect_min_seconds: float = 1.0, reconnect_max_seconds: float = 30.0) -> None:
        self._on_ticker = on_ticker
        self._on_trade = on_trade
        self.logger = logger or logging.getLogger("coinbase_ws")
        self._ws_url = ws_url
        self._product_id = product_id
        self._channels = channels
        self._reconnect_min = reconnect_min_seconds
        self._reconnect_max = reconnect_max_seconds
        self._shutdown = asyncio.Event()
        self.stats = {"connects": 0, "reconnects": 0, "messages": 0, "last_error": None}

    def stop(self) -> None:
        self._shutdown.set()

    async def run(self) -> None:
        delay = self._reconnect_min
        while not self._shutdown.is_set():
            try:
                await self._consume()
                delay = self._reconnect_min
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.stats["reconnects"] += 1
                self.stats["last_error"] = repr(exc)
                self.logger.warning("coinbase_ws error %r — backoff %.1fs", exc, delay)
                await asyncio.sleep(delay + random.random() * 1.5)
                delay = min(self._reconnect_max, max(self._reconnect_min, delay * 2))

    async def _consume(self) -> None:
        async with connect(self._ws_url, ping_interval=20.0, ping_timeout=20.0,
                           max_size=4_000_000) as ws:
            for ch in self._channels:
                await ws.send(orjson.dumps({"type": "subscribe", "channel": ch,
                                            "product_ids": [self._product_id]}).decode("utf-8"))
            self.stats["connects"] += 1
            self.logger.info("coinbase_ws connected (%s %s)", self._product_id, self._channels)
            async for message in ws:
                if self._shutdown.is_set():
                    break
                try:
                    m = orjson.loads(message)
                except Exception:
                    continue
                if not isinstance(m, dict):
                    continue
                ch = m.get("channel")
                ts = _coinbase_message_ts_ns(m)
                if ch == "ticker":
                    for ev in m.get("events", ()):
                        for tk in ev.get("tickers", ()):
                            bb, ba = tk.get("best_bid"), tk.get("best_ask")
                            if bb and ba:
                                self._on_ticker(ts, float(bb), float(ba),
                                                _flt(tk.get("best_bid_quantity")),
                                                _flt(tk.get("best_ask_quantity")))
                elif ch == "market_trades":
                    for ev in m.get("events", ()):
                        if ev.get("type") == "snapshot":
                            continue
                        for tr in ev.get("trades", ()):
                            sz = tr.get("size")
                            if sz:
                                side = tr.get("side")
                                s = 1.0 if side == "BUY" else (-1.0 if side == "SELL" else 0.0)
                                self._on_trade(ts, float(sz), s)
                self.stats["messages"] += 1
