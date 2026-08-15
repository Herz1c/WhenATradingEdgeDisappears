#!/usr/bin/env python3
"""Rebuild a corrupted .ws.manifest.json from the raw .jsonl.zst file.

Scans the raw file to extract first/last recv_ts_ns and record count, then
writes a manifest matching the schema of an adjacent (good) manifest.
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path
from datetime import datetime, timezone
import zstandard as zstd

raw_path = Path(sys.argv[1])               # data/raw/.../06.ws.jsonl.zst
template_path = Path(sys.argv[2])          # an adjacent good manifest
out_path = Path(sys.argv[3])               # data/manifests/.../06.ws.manifest.json

template = json.loads(template_path.read_text(encoding="utf-8"))

# Decompress & scan
first_ts_ns = None
last_ts_ns = None
record_count = 0
field_counts: dict[str, dict[str, int]] = {"record_type": {}, "stream": {}}

decomp = zstd.ZstdDecompressor()
with raw_path.open("rb") as f:
    with decomp.stream_reader(f) as reader:
        text_reader = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text_reader:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # last partial line during crash — stop
                break
            recv_ts = rec.get("recv_ts_ns")
            if isinstance(recv_ts, int):
                if first_ts_ns is None:
                    first_ts_ns = recv_ts
                last_ts_ns = recv_ts
            record_count += 1
            # tally record_type
            rt = str(rec.get("record_type") or "ws_event")
            field_counts["record_type"][rt] = field_counts["record_type"].get(rt, 0) + 1
            # tally stream
            stream = rec.get("stream")
            if isinstance(stream, str):
                field_counts["stream"][stream] = field_counts["stream"].get(stream, 0) + 1

if first_ts_ns is None or last_ts_ns is None:
    print(f"ERROR: no recv_ts_ns found in {raw_path}", file=sys.stderr)
    sys.exit(2)

# Build new manifest using template as schema
file_size = raw_path.stat().st_size

new_manifest = dict(template)  # shallow copy
new_manifest["file_path"] = str(raw_path).replace("\\", "/")
new_manifest["file_size"] = file_size
new_manifest["first_recv_ts_iso"] = datetime.fromtimestamp(first_ts_ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")
new_manifest["first_recv_ts_ns"] = first_ts_ns
new_manifest["last_recv_ts_iso"] = datetime.fromtimestamp(last_ts_ns / 1e9, tz=timezone.utc).isoformat().replace("+00:00", "Z")
new_manifest["last_recv_ts_ns"] = last_ts_ns
new_manifest["record_count"] = record_count
new_manifest["readable_record_count"] = record_count
new_manifest["field_counts"] = field_counts
new_manifest["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
new_manifest["manifest_written_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
new_manifest["file_state"] = "recovered"
new_manifest["quality_class"] = "recovered"
new_manifest["recovered_from_raw"] = True
new_manifest["tail_status"] = "tail_truncated"   # crash-ended file
new_manifest["finalized"] = True
new_manifest["manifest_record_count_source"] = "readable_rows"
# Carry over fields from template that should differ:
# - keep config_snapshot, archive_kind, recorder_service, source, source_tier, symbol, venue, etc.

out_path.write_text(json.dumps(new_manifest, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote rebuilt manifest: {out_path}")
print(f"  records: {record_count}")
print(f"  first_recv_ts: {new_manifest['first_recv_ts_iso']}")
print(f"  last_recv_ts:  {new_manifest['last_recv_ts_iso']}")
print(f"  file_size:    {file_size:,}")
