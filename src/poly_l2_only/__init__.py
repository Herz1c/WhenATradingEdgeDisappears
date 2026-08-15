"""Polymarket-only L2 feature extractor.

Single source of truth for raw L2 -> feature vector. Used by both:
- Offline trainer (tools/build_poly_l2_last60s.py): replays recorded
  .l2.jsonl.zst frames through the extractor.
- Live bot: feeds the same WS frames through the same extractor in real time.

Inputs: Polymarket WS only. No external (Chainlink / Binance / oracle) features.

Design principle: every feature emitted at train time must be derivable
from the live WS stream alone, computed by the same code path.
"""
