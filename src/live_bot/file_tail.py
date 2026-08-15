"""Async tail of zstd-compressed JSONL files written by the live
recorders.

The recorders (`market_recorders run-all`) write streaming zst-encoded
JSONL with a 2-second block flush. We read them as they grow, surface
new lines as records, and roll over to fresh files as the recorder
opens new ones (e.g. when a new Polymarket market opens or a new
hourly Binance file starts).

Latency budget: typical end-to-end is ~2-3 s (recorder flush + our
poll interval). Acceptable for ttc 10-60s trading; for sub-second
deployments we'd need direct WS consumers instead.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable

import orjson
import zstandard as zstd


@dataclass
class _FileCursor:
    path: Path
    file_handle: object | None = None
    reader: object | None = None
    text_stream: io.TextIOWrapper | None = None
    eof_reached: bool = False
    bytes_read: int = 0
    lines_emitted: int = 0
    last_growth_ns: int = 0


class ZstdJsonlTail:
    """Tail a single zst file. Re-opens on growth; survives partial
    final lines by buffering them until the next flush completes them."""

    def __init__(self, path: Path, *, poll_interval_s: float = 0.5,
                 logger: logging.Logger | None = None) -> None:
        self.path = path
        self.poll_interval_s = poll_interval_s
        self.logger = logger or logging.getLogger(f"tail[{path.name}]")
        self._closed = False
        self._line_buffer = b""
        # last byte position in the RAW compressed file we've decompressed past
        self._last_size = 0
        self._emitted_count = 0

    async def __aiter__(self) -> AsyncIterator[dict]:
        while not self._closed:
            if not self.path.exists():
                await asyncio.sleep(self.poll_interval_s)
                continue
            current_size = self.path.stat().st_size
            if current_size <= self._last_size:
                await asyncio.sleep(self.poll_interval_s)
                continue
            # Re-decompress from scratch each time (zstd block-flushes don't
            # let us seek into the middle of a stream safely). Track which
            # lines we've already emitted by count, not by file offset.
            yielded_in_pass, records = await asyncio.to_thread(
                self._read_growth_pass, self._emitted_count)
            self._emitted_count = max(self._emitted_count, yielded_in_pass)
            self._last_size = current_size
            for rec in records:
                yield rec
            await asyncio.sleep(self.poll_interval_s)

    def _read_growth_pass(self, emitted_count: int) -> tuple[int, list[dict]]:
        yielded_in_pass = 0
        records: list[dict] = []
        try:
            with self.path.open("rb") as fh:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(fh) as r:
                    buf = b""
                    while True:
                        try:
                            chunk = r.read(65_536)
                        except zstd.ZstdError:
                            break   # truncated frame at end; ok for tail
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            nl = buf.find(b"\n")
                            if nl < 0:
                                break
                            line = buf[:nl].strip()
                            buf = buf[nl + 1:]
                            if not line:
                                continue
                            yielded_in_pass += 1
                            if yielded_in_pass <= emitted_count:
                                continue
                            try:
                                records.append(orjson.loads(line))
                            except orjson.JSONDecodeError:
                                pass
        except FileNotFoundError:
            pass
        return yielded_in_pass, records

    def close(self) -> None:
        self._closed = True


class DirectoryTail:
    """Tail every matching file in a directory; as new files are
    created, start tailing them too.

    Yields (path, record) pairs in roughly file-creation order; within
    a single file we preserve write order (since zst is append-only)."""

    def __init__(self, root: Path, *, pattern: str = "*.jsonl.zst",
                 poll_interval_s: float = 1.0,
                 max_age_seconds: float = 900.0,
                 logger: logging.Logger | None = None) -> None:
        self.root = root
        self.pattern = pattern
        self.poll_interval_s = poll_interval_s
        # Files whose mtime is older than this are NOT tailed — they're
        # static (already-closed market files don't grow). Saves a huge
        # amount of needless decompression on directories with hundreds
        # of historical files.
        self.max_age_seconds = max_age_seconds
        self.logger = logger or logging.getLogger(f"dirtail[{root.name}]")
        self._tails: dict[Path, ZstdJsonlTail] = {}
        self._closed = False
        self._record_queue: asyncio.Queue[tuple[Path, dict]] = asyncio.Queue(maxsize=10_000)

    async def _scan_loop(self) -> None:
        import time as _time
        while not self._closed:
            try:
                if self.root.exists():
                    now = _time.time()
                    for p in sorted(self.root.glob(self.pattern)):
                        if p in self._tails:
                            continue
                        try:
                            age = now - p.stat().st_mtime
                        except OSError:
                            continue
                        if age > self.max_age_seconds:
                            # Static historical file — skip
                            self._tails[p] = None  # sentinel so we don't re-check
                            continue
                        tail = ZstdJsonlTail(p, poll_interval_s=self.poll_interval_s,
                                             logger=self.logger)
                        self._tails[p] = tail
                        asyncio.create_task(self._drain(tail), name=f"tail-{p.name}")
            except Exception as exc:
                self.logger.warning("scan_loop error: %r", exc)
            await asyncio.sleep(self.poll_interval_s)

    async def _drain(self, tail: ZstdJsonlTail) -> None:
        try:
            async for rec in tail:
                await self._record_queue.put((tail.path, rec))
        except Exception as exc:
            self.logger.warning("drain error %s: %r", tail.path.name, exc)

    async def records(self) -> AsyncIterator[tuple[Path, dict]]:
        asyncio.create_task(self._scan_loop(), name=f"scan-{self.root.name}")
        while not self._closed:
            try:
                item = await asyncio.wait_for(self._record_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            yield item

    def close(self) -> None:
        self._closed = True
        for t in self._tails.values():
            if t is not None:
                t.close()


class MultiSourceTail:
    """Wires up the three input streams the bot needs:
      - Polymarket L2 events       (per-market files, slug-filtered)
      - Polymarket REST snapshots  (per-market gamma_discovery rows)
      - Binance bookTicker         (per-hour files for live BTC mid)
      - Hyperliquid bookTicker     (per-hour files for live BTC mid)
    Each is yielded as a (source_kind, record) tuple in roughly
    arrival order."""

    SOURCE_POLY_L2     = "poly_l2"
    SOURCE_POLY_REST   = "poly_rest"
    SOURCE_POLY_STRIKE = "poly_strike"
    SOURCE_BINANCE     = "binance"
    SOURCE_HYPERLIQ    = "hyperliquid"
    SOURCE_BINUSDM     = "binance_usdm"
    SOURCE_CHAINLINK   = "chainlink_onchain"
    SOURCE_RTDS        = "rtds"        # Polymarket RTDS Chainlink price (resolution feed)
    SOURCE_COINBASE    = "coinbase"    # Coinbase advanced ticker/trades (leading CEX)

    def __init__(self, *, raw_root: Path,
                 poll_interval_s: float = 0.5,
                 logger: logging.Logger | None = None,
                 today_iso: str | None = None,
                 fair_value_sources: bool = False,
                 registration_only: bool = False,
                 polymarket_only: bool = False) -> None:
        """`today_iso` is ignored — kept only for backwards compatibility.
        The tail now ALWAYS scans both yesterday's and today's date
        directories on each poll, so it auto-handles UTC midnight rollover
        without needing the caller to restart anything."""
        self.raw_root = raw_root
        self.logger = logger or logging.getLogger("multi-tail")
        self.poll_interval_s = poll_interval_s
        self._fair_value_sources = fair_value_sources
        self._registration_only = registration_only
        self._polymarket_only = polymarket_only
        self._queue: asyncio.Queue[tuple[str, Path, dict]] = asyncio.Queue(maxsize=20_000)
        self._dirs: list[tuple[str, DirectoryTail]] = []
        self._closed = False
        self._last_active_date: str = ""   # track for logging
        self._drainer_starter = None       # set in stream()
        self._watched_keys: set = set()    # (source, dir_path_str) we've already created tails for

    def _active_dates(self) -> list[str]:
        """Yesterday + today (UTC). DirectoryTail's mtime filter drops
        files older than max_age_seconds so we don't keep re-decompressing
        yesterday's static files."""
        from datetime import datetime as _dt, timedelta as _td, UTC as _UTC
        today = _dt.now(_UTC).date()
        return [(today - _td(days=1)).isoformat(), today.isoformat()]

    def _build_dirs(self) -> None:
        # Build one DirectoryTail per (source × date) — DirectoryTail's
        # internal scan_loop watches its own dir, so as new files appear
        # in today's dir (including AFTER midnight), they get tailed
        # automatically. After midnight a new "today" appears and we
        # extend the watcher list to include it.
        from datetime import UTC as _UTC, datetime as _dt
        now_utc = _dt.now(_UTC)
        today_iso = now_utc.date().isoformat()
        current_hour_pattern = f"{now_utc.hour:02d}.ws.jsonl.zst"
        existing_paths = {id(t) for _, t in self._dirs}
        for d in self._active_dates():
            poly_dir = self.raw_root / "raw" / "polymarket" / "btc_updown_5m" / d
            bin_spot = self.raw_root / "raw" / "binance" / "spot" / "BTCUSDT" / d
            bin_usdm = self.raw_root / "raw" / "binance" / "usdm" / "BTCUSDT" / d
            hl_dir   = self.raw_root / "raw" / "hyperliquid" / "linear" / "BTC-USD" / d
            cl_dir   = self.raw_root / "raw" / "chainlink" / "onchain" / "BTCUSD" / d
            rtds_dir = self.raw_root / "raw" / "polymarket" / "rtds" / "crypto_prices_chainlink" / "btc_usd" / d
            cb_dir   = self.raw_root / "raw" / "coinbase" / "advanced" / "BTC-USD" / d
            if self._registration_only:
                # Hot data (book/RTDS/Coinbase) comes via direct WS; from files
                # we only need the low-frequency market discovery + strike.
                sources = [
                    (self.SOURCE_POLY_REST,   poly_dir, "*.rest.jsonl.zst"),
                    (self.SOURCE_POLY_STRIKE, poly_dir, "*.strike.jsonl.zst"),
                ]
                for source, dirpath, pattern in sources:
                    key = (source, str(dirpath), pattern)
                    if key in self._watched_keys: continue
                    tail = DirectoryTail(dirpath, pattern=pattern,
                                         poll_interval_s=self.poll_interval_s, logger=self.logger)
                    self._dirs.append((source, tail))
                    self._watched_keys.add(key)
                    if self._drainer_starter is not None:
                        self._drainer_starter(source, tail)
                self._last_active_date = d
                continue
            sources = [
                (self.SOURCE_POLY_L2,     poly_dir, "*.l2.jsonl.zst"),
                (self.SOURCE_POLY_REST,   poly_dir, "*.rest.jsonl.zst"),
                (self.SOURCE_POLY_STRIKE, poly_dir, "*.strike.jsonl.zst"),
            ]
            if not self._polymarket_only and not self._fair_value_sources:
                sources += [
                (self.SOURCE_BINANCE,     bin_spot, "*.ws.jsonl.zst"),
                (self.SOURCE_BINUSDM,     bin_usdm, "*.ws.jsonl.zst"),
                (self.SOURCE_HYPERLIQ,    hl_dir,   "*.ws.jsonl.zst"),
                (self.SOURCE_CHAINLINK,   cl_dir,   "*.onchain.jsonl.zst"),
                ]
            if self._fair_value_sources and d == today_iso:
                sources += [
                    (self.SOURCE_RTDS,     rtds_dir, current_hour_pattern),
                    (self.SOURCE_COINBASE, cb_dir,   current_hour_pattern),
                ]
            for source, dirpath, pattern in sources:
                key = (source, str(dirpath), pattern)
                if key in self._watched_keys: continue
                max_age = 480.0 if self._fair_value_sources else 900.0
                tail = DirectoryTail(dirpath, pattern=pattern,
                                     poll_interval_s=self.poll_interval_s,
                                     max_age_seconds=max_age,
                                     logger=self.logger)
                self._dirs.append((source, tail))
                self._watched_keys.add(key)
                # Start drainer immediately if event loop is already running
                if self._drainer_starter is not None:
                    self._drainer_starter(source, tail)
            self._last_active_date = d
        # Note: cleanup of old dirs not needed — DirectoryTail's mtime
        # filter (max_age_seconds=900s) means old dirs decay to no-op.

    async def _drain_dir(self, source: str, tail: DirectoryTail) -> None:
        async for path, rec in tail.records():
            await self._queue.put((source, path, rec))

    async def _rebuild_loop(self) -> None:
        """Re-scan active dates every minute so the newly-rolled-over
        UTC date directory gets a fresh DirectoryTail without restarting
        the bot."""
        while not self._closed:
            await asyncio.sleep(60.0)
            try:
                self._build_dirs()
            except Exception as exc:
                self.logger.warning("rebuild_loop error: %r", exc)

    async def stream(self) -> AsyncIterator[tuple[str, Path, dict]]:
        # Capture a function that the (sync) _build_dirs can call to
        # start a drainer task for any newly-created DirectoryTail.
        def _start(source: str, tail: DirectoryTail) -> None:
            asyncio.create_task(self._drain_dir(source, tail), name=f"drain-{source}")
        self._drainer_starter = _start
        self._watched_keys: set = set()
        self._build_dirs()
        asyncio.create_task(self._rebuild_loop(), name="multi-tail-rebuild")
        while not self._closed:
            try:
                yield await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

    def close(self) -> None:
        self._closed = True
        for _, t in self._dirs:
            t.close()
