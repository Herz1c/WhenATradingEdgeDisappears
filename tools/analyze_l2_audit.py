"""Compare direct-WS L2 audit events against recorder L2 events.

Run after starting the fair-value bot with FV_L2_AUDIT=1. The tool answers the
question that feature-level parity cannot: is the bot receiving the same
Polymarket L2 event stream as the recorder, or are there duplicate/missing/
different direct-WS events before features are even computed?

Usage:
  py -3 tools/analyze_l2_audit.py 2026-06-29 06:19
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import zstandard as zstd

ROOT = Path(__file__).resolve().parents[1]
RECV_RE = re.compile(br'"recv_ts_ns":\s*(\d+)')


def _digest(rec: dict) -> str:
    changed = rec.get("changed_levels") or []
    payload = rec.get("event_payload") or {}
    src = json.dumps(changed or payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(src.encode("utf-8"), digest_size=8).hexdigest()


def _iter_zst_lines(path: Path):
    with path.open("rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as zr:
        buf = b""
        while True:
            chunk = zr.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    yield line
        if buf.strip():
            yield buf


def _compact_recorder_row(rec: dict) -> dict:
    lc = rec.get("level_counts") or {}
    dt = rec.get("depth_totals") or {}
    return {
        "ts": int(rec.get("recv_ts_ns") or 0),
        "slug": rec.get("selected_market_slug"),
        "conn": rec.get("connection_id"),
        "seq": rec.get("local_msg_seq"),
        "event_type": rec.get("event_type"),
        "token_outcome": rec.get("token_outcome"),
        "token_asset_id": rec.get("token_asset_id"),
        "book_state_seq": rec.get("book_state_seq"),
        "snapshot_origin": rec.get("snapshot_origin"),
        "book_hash": rec.get("book_hash"),
        "event_digest": _digest(rec),
        "best_bid": rec.get("best_bid"),
        "best_ask": rec.get("best_ask"),
        "level_bid": lc.get("bid"),
        "level_ask": lc.get("ask"),
        "depth_bid": dt.get("bid_size_total"),
        "depth_ask": dt.get("ask_size_total"),
    }


def _event_key(row: dict) -> tuple:
    return (
        row.get("event_type"),
        row.get("token_asset_id"),
        row.get("event_digest"),
        row.get("best_bid"),
        row.get("best_ask"),
    )


def _load_live(date: str, t0: int) -> list[dict]:
    path = ROOT / "logs" / "fair_value_bot" / f"l2_audit_{date}.jsonl"
    if not path.exists():
        print(f"missing {path} (restart bot with FV_L2_AUDIT=1)")
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("ts") or 0) >= t0:
            rows.append(row)
    return rows


def _load_recorder(date: str, slug: str, lo: int, hi: int) -> list[dict]:
    pm_dir = ROOT / "data" / "raw" / "polymarket" / "btc_updown_5m" / date
    files = sorted(pm_dir.glob(f"*{slug}*.l2.jsonl.zst"))
    if not files:
        return []
    rows = []
    for line in _iter_zst_lines(files[0]):
        m = RECV_RE.search(line)
        if not m:
            continue
        ts = int(m.group(1))
        if ts < lo or ts > hi:
            continue
        try:
            rows.append(_compact_recorder_row(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return rows


def _summarize(name: str, rows: list[dict]) -> None:
    by_type = Counter(r.get("event_type") for r in rows)
    keys = [_event_key(r) for r in rows]
    uniq = len(set(keys))
    dup = len(keys) - uniq
    books = sum(1 for r in rows if r.get("event_type") == "book")
    hashes = len({r.get("book_hash") for r in rows if r.get("book_hash")})
    print(
        f"{name:8s} rows={len(rows):7d} unique={uniq:7d} dup={dup:7d} "
        f"books={books:4d} book_hashes={hashes:4d} types={dict(by_type)}"
    )


def main() -> int:
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone.utc).date().isoformat()
    hhmm = sys.argv[2] if len(sys.argv) > 2 else "00:00"
    h, m = map(int, hhmm.split(":"))
    t0 = int(datetime(*map(int, date.split("-")), h, m, tzinfo=timezone.utc).timestamp() * 1e9)

    live = _load_live(date, t0)
    if not live:
        return 1
    print(f"live direct L2 audit rows: {len(live)} since {date} {hhmm} UTC")

    for slug in sorted({r["slug"] for r in live if r.get("slug")}):
        lr = [r for r in live if r.get("slug") == slug]
        # Recorder rows are filtered on recv_ts_ns; when the recorder is
        # draining a backlog, events the bot saw at T are written minutes
        # later. Extend the upper bound so those still match by content key
        # instead of being reported as recorder-missing.
        lo = min(int(r["ts"]) for r in lr) - 5_000_000_000
        hi = max(int(r["ts"]) for r in lr) + 330_000_000_000
        rr = _load_recorder(date, slug, lo, hi)
        print(f"\n{slug}")
        _summarize("live", lr)
        _summarize("rec", rr)
        live_keys = Counter(_event_key(r) for r in lr)
        rec_keys = Counter(_event_key(r) for r in rr)
        common_unique = len(set(live_keys) & set(rec_keys))
        live_only_unique = len(set(live_keys) - set(rec_keys))
        rec_only_unique = len(set(rec_keys) - set(live_keys))
        live_extra_repeats = sum(max(0, live_keys[k] - rec_keys.get(k, 0)) for k in live_keys)
        print(
            f"key overlap: common_unique={common_unique} "
            f"live_only_unique={live_only_unique} rec_only_unique={rec_only_unique} "
            f"live_extra_repeats={live_extra_repeats}"
        )
        if live_only_unique:
            print("first live-only keys:")
            for key in list(set(live_keys) - set(rec_keys))[:5]:
                print("  ", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
