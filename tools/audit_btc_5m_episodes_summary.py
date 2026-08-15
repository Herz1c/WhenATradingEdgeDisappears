"""Finalize the episode-dataset audit: enrich audit_report.json with caveats /
recommendations and render audit_summary.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "btc_5m_episodes_v1_200ms_audit"
r = json.loads((OUT / "audit_report.json").read_text(encoding="utf-8"))
ss = r["split_stats"]
cov = r["coverage"]
base = r["baseline_p_market"]

r["caveats"] = [
    "Coverage gaps: 20 missing calendar days inside 2026-04-19..07-02 (recorder downtime; incl. known 2026-05-24 no-resolution day). Gaps are BETWEEN days, so no episode is truncated; chronological split semantics remain valid.",
    "5 small days (<50 episodes): 2026-04-19 (22), 05-01 (24), 06-11 (12, valid_frac 0.139 - known weak day, no_rtds_window skips), 06-15 (35), 06-30 (33).",
    "valid_frac is nonstationary: ~0.22 (train era) -> ~0.29 (test era) -> 0.75-0.81 on post-recorder-fix days (06-30/07-01). Mask-aware sampling/weighting recommended; coverage itself is a (mild) era signal a model could sniff - do not feed mask stats as features.",
    "val split is thin: 2 days / 395 markets / 194k valid steps. Adequate for smoke checks and early stopping at best; do NOT fit calibration layers or select architectures on it alone (see paper_v1 section 5.2 for the documented hazard). Prefer day-blocked CV within train.",
    "test (4,667 markets) is larger than train (3,841): fine for honest evaluation, but training data is on the small side for large sequence models.",
    "2026-07-02 is a partial day (51 episodes; recorders still running at build time).",
    "Episode floor is permissive (min 20 valid steps / 4 s): p10 of valid steps per episode is 129-159. Weight episodes by valid count or set a stricter floor at training time.",
    "now_ns grid carries <=18.3 us jitter from float32 grid rounding (cosmetic; starts exactly at open_s).",
    "pm_recv/delivery lags on valid steps reach 286 s (recorder backlog) - this is why source-time replay was mandatory; recv-clock consumers of the same raw data would be invalid. Audit-only, not in features.",
]

r["suitability"] = {
    "shape": "1500 x 69 @ 5 Hz per 300 s market - reasonable for GRU/TCN; transformers should patch (e.g., 30-60 step patches) rather than attend over 1500 raw steps",
    "volume": {
        "train_valid_steps": ss["train"]["valid_steps_total"],
        "val_valid_steps": ss["val"]["valid_steps_total"],
        "test_valid_steps": ss["test"]["valid_steps_total"],
        "assessment": "1.29M masked train steps across 3,841 episodes: enough for small/medium GRU/TCN (0.1-2M params); thin for large transformers",
    },
    "recommended_models": [
        "GRU/LSTM (1-2 layers, 64-128 hidden) with valid_mask as extra input channel and hidden-state carry-through across invalid steps",
        "TCN with mask-aware convolutions (zero-input + mask channel) - cheap, strong baseline",
        "patch Transformer (patch 30-60 steps => 25-50 tokens) only after GRU/TCN baselines; Perceiver-style pooling if attending across full episode",
    ],
    "loss_and_sampling": [
        "loss ONLY on valid_mask steps; predict per-step P(UP) with BCE against episode label, optionally TTC-weighted",
        "anchor on the market: use logit(p_market) offset/residual head - paper_v1 shows from-scratch models start ~1 Brier point behind",
        "sample episodes proportional to valid_count (or fixed per-episode step subsample) to stop high-coverage days from dominating",
        "evaluate on TTC 15-90 s as headline (matches decision window and prior work); train on all valid steps",
        "downweight (0.25-0.5x) rather than exclude the 5 small/weak days; exclude nothing else",
    ],
    "expected_baseline_to_beat": {
        "test_ttc_15_90": base["test"]["ttc_15_90"],
        "note": "market p Brier 0.1104 / AUC 0.926 - the tabular study could not beat this; treat matching it as success criterion #1",
    },
}

r["live_replicability"] = {
    "verdict": "deployable in principle - all 69 features are causally computable live from PM book + RTDS + Coinbase ticker via the shared FairValueState extractor; clock policy matches the live bot's direct feeds",
    "requirements": [
        "ship normalization.json (mean/std) with the model; live features must be normalized identically",
        "live gating must replicate freshness policy exactly: pm_source_age<=2s, cex_age<=2s, rtds_source_age<=60s, else mask the step (zeros) exactly as in training",
        "live loop must run the same 200 ms grid from market open_s and the same warmup semantics",
        "RTDS availability clock = chainlink_ts + 2.5 s latency; the live bot consumes RTDS earlier - either delay the feature clock to match or rebuild with live-measured latency",
    ],
    "suspicious_features": "none - no recv/delivery/absolute-level/outcome columns in feature_names; audit columns are segregated",
    "bot_must_log_for_parity": [
        "per 200 ms step (for traded markets): now_ns, the 69 raw feature values pre-normalization, valid flag + which gate failed",
        "pm/cex/rtds source ages at each step (the audit vector)",
        "quotes (up/down bid/ask, implied_p_up) at decision time",
        "model input-window hash + model p at decision time, for exact replay via verify_fv_decision_parity-style tooling",
    ],
}

(OUT / "audit_report.json").write_text(json.dumps(r, indent=1), encoding="utf-8")

# ------------------------------------------------------------- summary md
def pct(x):
    return f"{100 * x:.1f}%"

lines = [
    "# Audit: btc_5m_episodes_v1_200ms",
    "",
    f"*Generated {r['generated_at']} by `tools/audit_btc_5m_episodes.py` — "
    f"{len(r['checks'])} checks, {len(r['blocking'])} blocking failures.*",
    "",
    f"## Verdict: **READY (with caveats)** — all {len(r['checks'])} strict checks pass",
    "",
    "Integrity, chronological splits, market disjointness, train-only masked normalization "
    "(recomputed independently, agreement 1e-7; split X byte-reproducible from raw shards), "
    "source-clock freshness policy (0 violations on valid steps), label/slug/date alignment, "
    "and market-baseline sanity all verified against the raw daily shards, not the manifest.",
    "",
    "## Key stats",
    "",
    "| | train | val | test |",
    "|---|---|---|---|",
    f"| markets | {ss['train']['markets']:,} | {ss['val']['markets']:,} | {ss['test']['markets']:,} |",
    f"| days | {ss['train']['n_dates']} (04-19..05-11) | 2 (05-12..13) | {ss['test']['n_dates']} (05-14..07-02) |",
    f"| P(UP) | {ss['train']['y_mean']} | {ss['val']['y_mean']} | {ss['test']['y_mean']} |",
    f"| valid_frac | {pct(ss['train']['valid_frac'])} | {pct(ss['val']['valid_frac'])} | {pct(ss['test']['valid_frac'])} |",
    f"| present_frac | {pct(ss['train']['present_frac'])} | {pct(ss['val']['present_frac'])} | {pct(ss['test']['present_frac'])} |",
    f"| valid steps | {ss['train']['valid_steps_total']:,} | {ss['val']['valid_steps_total']:,} | {ss['test']['valid_steps_total']:,} |",
    f"| valid steps / episode p50 | {ss['train']['valid_steps_per_episode']['p50']:.0f} | "
    f"{ss['val']['valid_steps_per_episode']['p50']:.0f} | {ss['test']['valid_steps_per_episode']['p50']:.0f} |",
    f"| market Brier (TTC 15-90, valid) | {base['train']['ttc_15_90']['brier']} | "
    f"{base['val']['ttc_15_90']['brier']} | {base['test']['ttc_15_90']['brier']} |",
    f"| market AUC (TTC 15-90, valid) | {base['train']['ttc_15_90']['auc']} | "
    f"{base['val']['ttc_15_90']['auc']} | {base['test']['ttc_15_90']['auc']} |",
    "",
    "Clock policy on valid steps (test): pm_source_age p50 3 ms / p95 74 ms (cap 2 s), "
    "cex_age p50 0.12 s (cap 2 s), rtds_source_age p50 3.0 s (cap 60 s). "
    "99.7% of present-but-invalid steps fail on PM freshness (historical recorder backlog) — "
    "pm_recv/delivery lags up to 286 s are visible in the audit channels and correctly excluded "
    "from validity by the source-clock policy.",
    "",
    "## Blocking issues",
    "",
    "None.",
    "",
    "## Caveats (non-blocking)",
    "",
]
lines += [f"{i + 1}. {c}" for i, c in enumerate(r["caveats"])]
lines += [
    "",
    "## Training recommendations",
    "",
    "1. **Models:** start GRU (1-2 x 64-128) and TCN with valid_mask input channel + masked BCE; "
    "patch-Transformer (30-60-step patches) only after those baselines.",
    "2. **Anchor on the market:** residual head over logit(p_market); from-scratch sequence models "
    "must re-learn price formation and start far behind (see paper_v1 §5.1).",
    "3. **Loss/sampling:** masked per-step BCE vs episode label; sample by valid_count; "
    "headline eval on TTC 15-90 s; day-block bootstrap for CIs; downweight the 5 weak days.",
    "4. **Model selection:** do NOT trust the 2-day val split for anything beyond early stopping; "
    "use day-blocked CV within train (documented calibration hazard).",
    "5. **Success criterion #1:** match market Brier 0.1104 / AUC 0.926 on test TTC 15-90 valid steps. "
    "The tabular study tied it; a sequence model that merely ties it is expected, beating it is news.",
    "",
    "## Live-replicability",
    "",
    f"{r['live_replicability']['verdict']}.",
    "",
]
lines += [f"- {x}" for x in r["live_replicability"]["requirements"]]
lines += ["", "Bot must log for parity: " + "; ".join(r["live_replicability"]["bot_must_log_for_parity"]) + ".", ""]
(OUT / "audit_summary.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT / "audit_summary.md")
print("updated", OUT / "audit_report.json")
