"""Load Polymarket resolution outcomes into a slug -> outcome lookup.

Resolution lives under data/raw/polymarket/resolution/btc_updown_5m/YYYY-MM-DD/HH.resolution.jsonl.zst.
Each line is one resolved market. We key by `market_slug` (e.g.
"btc-updown-5m-1780181100") because the slug appears verbatim in the L2
file metadata too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

import orjson
import zstandard as zstd


def _iter_jsonl_zst(path: Path) -> Iterator[dict]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while True:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl]
                    buf = buf[nl + 1:]
                    if line:
                        try:
                            yield orjson.loads(line)
                        except Exception:
                            continue
            if buf.strip():
                try:
                    yield orjson.loads(buf)
                except Exception:
                    pass


def load_resolution_map(resolution_root: Path, verbose: bool = False) -> Dict[str, str]:
    """Walk all resolution zst files under `resolution_root` and return
    {market_slug: "Up"|"Down"}. Last-write-wins on duplicate slugs.

    Corrupted zst files are skipped with a printed warning."""
    out: Dict[str, str] = {}
    if not resolution_root.exists():
        return out
    files = sorted(resolution_root.glob("*/*.resolution.jsonl.zst"))
    bad = 0
    for p in files:
        try:
            for rec in _iter_jsonl_zst(p):
                slug = rec.get("market_slug")
                outcome = rec.get("winning_outcome")
                if not slug or outcome not in ("Up", "Down"):
                    continue
                out[str(slug)] = str(outcome)
        except Exception as e:  # noqa: BLE001
            bad += 1
            if verbose:
                print(f"  [resolution] skipping {p.name}: {e}")
    if bad and verbose:
        print(f"  [resolution] {bad} files skipped due to errors")
    return out
