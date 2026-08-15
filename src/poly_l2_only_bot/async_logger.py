"""Lock-free async logger for the trading hot path.

Hot path calls log(dict) → queue.put_nowait(dict). Background daemon
thread drains the queue, serializes with orjson, writes to a daily
JSONL shard. File rotates at UTC midnight.

queue.put_nowait() is ~200 ns. orjson + fwrite happens off the hot path
and never blocks it.

If the queue is full (>QUEUE_HIGH_WATER), the hot path DROPS the message
(non-blocking drop) and increments a counter. We log the drop count once
per second so we know if we're losing telemetry under load. Trading
decisions are NEVER blocked on logging.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import orjson

# Queue capacity: 65k messages buffer. At ~100 events/s, that's 10+ min
# of buffer for catastrophic disk stalls without losing recent telemetry.
QUEUE_HIGH_WATER = 65536


class AsyncJsonlLogger:
    """A single-file (rotated daily) async JSONL writer."""

    def __init__(self, *, log_dir: Path, basename: str) -> None:
        self.log_dir = log_dir
        self.basename = basename
        log_dir.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[Dict[str, Any] | None] = queue.Queue(maxsize=QUEUE_HIGH_WATER)
        self._dropped = 0
        self._written = 0
        self._stop = threading.Event()
        self._current_date = ""
        self._current_fh = None
        self._writer = threading.Thread(target=self._run, name=f"async_log:{basename}",
                                         daemon=True)
        self._writer.start()

    def log(self, record: Dict[str, Any]) -> None:
        """Hot-path entry point. Non-blocking. ~200 ns when queue has space."""
        try:
            self.q.put_nowait(record)
        except queue.Full:
            self._dropped += 1   # not thread-safe but we only read this in writer thread

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self.q.put(None)  # sentinel to wake the writer
        self._writer.join(timeout=timeout)
        if self._current_fh is not None:
            try:
                self._current_fh.close()
            except Exception:
                pass

    def _path_for_date(self, date_str: str) -> Path:
        return self.log_dir / f"{self.basename}_{date_str}.jsonl"

    def _ensure_fh(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._current_fh is not None:
                try:
                    self._current_fh.close()
                except Exception:
                    pass
            self._current_fh = self._path_for_date(today).open("ab", buffering=0)
            self._current_date = today

    def _run(self) -> None:
        # Reuse newline byte to avoid one allocation per record.
        NL = b"\n"
        last_stats_report = time.monotonic()
        local_dropped = 0
        while True:
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                # Periodic flush of the dropped-message counter so we don't lose visibility.
                now = time.monotonic()
                if now - last_stats_report > 10.0 and self._dropped > local_dropped:
                    self._ensure_fh()
                    self._current_fh.write(orjson.dumps({
                        "_logger_event": "dropped_messages",
                        "dropped_total": self._dropped,
                        "since_last": self._dropped - local_dropped,
                        "ts_iso": datetime.now(timezone.utc).isoformat(),
                    }) + NL)
                    local_dropped = self._dropped
                    last_stats_report = now
                if self._stop.is_set():
                    break
                continue
            if rec is None:
                break
            try:
                self._ensure_fh()
                self._current_fh.write(orjson.dumps(rec) + NL)
                self._written += 1
            except Exception as exc:
                # Last-ditch: print to stderr. Never raise.
                try:
                    import sys
                    print(f"[async_logger] write failed: {exc!r}", file=sys.stderr)
                except Exception:
                    pass

    @property
    def stats(self) -> Dict[str, int]:
        return {"written": self._written, "dropped": self._dropped,
                "queue_size": self.q.qsize()}
