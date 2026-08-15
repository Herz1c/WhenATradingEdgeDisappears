"""Pack the TCN v2 retrain bundle for Colab (Phase 4).

Contents: dataset v2 split npz + metadata, the trainer, requirements, README.
The growing holdout (2026-07-03+) is deliberately NOT included — it belongs to
the final gate only.

Usage:
    py tools/make_tcn_retrain_bundle.py
    # -> artifacts/colab_bundles/btc_tcn_retrain_v2_bundle.zip (upload to Drive)
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms"
OUT = ROOT / "artifacts" / "colab_bundles" / "btc_tcn_retrain_v2_bundle.zip"

FILES = [
    (V2 / "train.npz", "data/datasets/btc_5m_episodes_v2_200ms/train.npz"),
    (V2 / "val.npz", "data/datasets/btc_5m_episodes_v2_200ms/val.npz"),
    (V2 / "test.npz", "data/datasets/btc_5m_episodes_v2_200ms/test.npz"),
    (V2 / "normalization.json", "data/datasets/btc_5m_episodes_v2_200ms/normalization.json"),
    (V2 / "splits_v2.json", "data/datasets/btc_5m_episodes_v2_200ms/splits_v2.json"),
    (V2 / "feature_names.json", "data/datasets/btc_5m_episodes_v2_200ms/feature_names.json"),
    (V2 / "quote_names.json", "data/datasets/btc_5m_episodes_v2_200ms/quote_names.json"),
    (V2 / "audit_names.json", "data/datasets/btc_5m_episodes_v2_200ms/audit_names.json"),
    (ROOT / "tools" / "train_episode_tcn.py", "tools/train_episode_tcn.py"),
    (ROOT / "requirements_colab.txt", "requirements_colab.txt"),
    (ROOT / "colab" / "TCN_RETRAIN_README.md", "TCN_RETRAIN_README.md"),
]


def main() -> int:
    missing = [str(src) for src, _ in FILES if not src.exists()]
    if missing:
        print("missing inputs (run build_splits_v2.py first):")
        for m in missing:
            print(" ", m)
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # npz are already zip containers -> ZIP_STORED avoids double compression
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_STORED) as z:
        for src, arc in FILES:
            z.write(src, arc)
            print(f"  + {arc} ({src.stat().st_size / 1e6:.1f} MB)")
    print(f"-> {OUT} ({OUT.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
