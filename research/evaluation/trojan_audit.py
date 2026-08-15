#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trojan-horse feature audit across all 33 model artifacts.

Looking for: models that lean heavily on environmental features that have
catastrophic distribution shift between training and OOS. From the diagnosis:

  TROJAN FEATURES (PSI > 1 in 18-day train vs 4-day OOS window):
    live_btc_usd                     PSI 7.19
    price_to_beat                    PSI 7.85   (proxy for BTC; same shift)
    live_applied_bias_age_seconds    PSI 2.43

Output: per-model report with total trojan-importance share, ranked by risk.

Also includes a softer tier of features with moderate PSI (0.05-0.25):
    spread_mean_5s, spread_max_5s, trade_count_5s, etc.
"""

from __future__ import annotations
import io, sys, json, joblib
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

# ── Trojan tiers ──────────────────────────────────────────────────────────────

TROJAN_HARD = {  # PSI > 1.0 — catastrophic shift
    "live_btc_usd",
    "price_to_beat",
    "live_applied_bias_age_seconds",
}

TROJAN_SOFT = {  # PSI 0.05 - 0.25 — moderate shift, watch for over-reliance
    "spread_mean_5s", "spread_max_5s", "spread_min_5s",
    "trade_count_5s", "trade_count_3s", "trade_count_1s",
    "absolute_trade_size_5s", "absolute_trade_size_3s",
    "signed_trade_size_5s", "signed_trade_size_3s",
    "quote_update_count_1s", "quote_update_count_3s", "quote_update_count_5s",
    "event_count_1s", "event_count_3s", "event_count_5s", "event_count_15s",
    "depth_change_rate_5s", "quote_churn_rate_5s",
}

# ── Feature importance extractors ─────────────────────────────────────────────

def load_lgbm_importance(path: Path) -> dict[str, float]:
    """Load feature_importance.json (LightGBM) and normalize to sum=1."""
    d = json.loads(path.read_text())
    total = sum(d.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in d.items()}


def load_metrics_importance(path: Path) -> dict[str, float]:
    """Load metrics.json's feature_importance dict (08c family) and normalize."""
    raw = json.loads(path.read_text()).get("feature_importance", {})
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def load_results_importance(path: Path) -> dict[str, float] | None:
    """Load results.json's feature_importances (08b) and normalize."""
    d = json.loads(path.read_text())
    fi = d.get("feature_importances") or d.get("feature_importance")
    feats = d.get("config", {}).get("features") or d.get("features")
    if fi is None or feats is None:
        return None
    if isinstance(fi, list) and len(fi) == len(feats):
        named = dict(zip(feats, fi))
    elif isinstance(fi, dict):
        named = fi
    else:
        return None
    total = sum(named.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in named.items()}


def load_lr_importance(model_path: Path) -> dict[str, float] | None:
    """For LR/linreg: |coef| * std as proxy importance.  Std unavailable here →
    fall back to |coef| share which approximates importance for standardized features."""
    try:
        m = joblib.load(model_path)
    except Exception:
        return None
    # Sklearn LogisticRegression / LinearRegression
    coef = getattr(m, "coef_", None)
    if coef is None:
        # Pipeline?
        if hasattr(m, "named_steps"):
            for step in m.named_steps.values():
                coef = getattr(step, "coef_", None)
                if coef is not None:
                    break
    if coef is None:
        return None
    coef = np.asarray(coef).ravel()
    feat_names = getattr(m, "feature_names_in_", None)
    if feat_names is None:
        # Try to find from manifest
        return None
    if len(feat_names) != len(coef):
        return None
    abs_coef = np.abs(coef)
    total = abs_coef.sum()
    if total <= 0:
        return {}
    return {n: float(c / total) for n, c in zip(feat_names, abs_coef)}


# ── Model registry — map artifact paths to (label, importance source) ─────────

def discover_models() -> list[tuple[str, str, Path, str]]:
    """
    Return list of (model_label, family_tag, importance_path, source_type).
    source_type: 'lgbm_fi' | 'metrics_fi' | 'results_fi' | 'lr_coef'
    """
    out = []
    root = Path("artifacts")
    # 1. All LGBM models with feature_importance.json
    for fi in sorted(root.rglob("feature_importance.json")):
        rel = fi.relative_to(root)
        label = str(rel.parent).replace("\\", "/")
        out.append((label, label.split("/")[0], fi, "lgbm_fi"))

    # 2. 08c family — metrics.json contains feature_importance
    for fi in sorted(root.rglob("metrics.json")):
        rel = fi.relative_to(root)
        label = str(rel.parent).replace("\\", "/")
        # avoid double-counting if there's also a feature_importance.json sibling
        if (fi.parent / "feature_importance.json").exists():
            continue
        out.append((label, label.split("/")[0], fi, "metrics_fi"))

    # 3. 08b — results.json
    for fi in sorted(root.rglob("results.json")):
        rel = fi.relative_to(root)
        label = str(rel.parent).replace("\\", "/")
        out.append((label, label.split("/")[0], fi, "results_fi"))

    # 4. LR / linreg models (no feature_importance.json) — fall back to |coef|
    for mpkl in sorted(root.rglob("model.pkl")):
        if "logistic_regression" not in str(mpkl) and "linear_regression" not in str(mpkl):
            continue
        rel = mpkl.relative_to(root)
        label = str(rel.parent).replace("\\", "/")
        out.append((label, label.split("/")[0], mpkl, "lr_coef"))
    return out


def get_importance(path: Path, src: str) -> dict[str, float] | None:
    if src == "lgbm_fi":
        return load_lgbm_importance(path)
    if src == "metrics_fi":
        return load_metrics_importance(path)
    if src == "results_fi":
        return load_results_importance(path)
    if src == "lr_coef":
        return load_lr_importance(path)
    return None


# ── Risk scoring ──────────────────────────────────────────────────────────────

def score_model(imp: dict[str, float]) -> dict:
    if not imp:
        return {
            "n_features": 0, "hard_share": None, "soft_share": None,
            "top_trojan": [], "any_in_top5": False,
        }
    hard = sum(v for k, v in imp.items() if k in TROJAN_HARD)
    soft = sum(v for k, v in imp.items() if k in TROJAN_SOFT)
    top5 = sorted(imp.items(), key=lambda kv: -kv[1])[:5]
    any_in_top5 = any(k in TROJAN_HARD for k, _ in top5)
    trojan_contribs = sorted(
        [(k, v) for k, v in imp.items() if k in TROJAN_HARD],
        key=lambda kv: -kv[1],
    )
    return {
        "n_features": len(imp),
        "hard_share": hard,
        "soft_share": soft,
        "top_trojan": trojan_contribs,
        "any_in_top5": any_in_top5,
        "top5": top5,
    }


def risk_label(s: dict) -> str:
    h = s["hard_share"]
    if h is None:
        return "n/a"
    if h >= 0.30 or s["any_in_top5"]:
        return "CRITICAL"
    if h >= 0.15:
        return "HIGH"
    if h >= 0.05:
        return "MODERATE"
    return "LOW"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    models = discover_models()
    results = []
    for label, family, path, src in models:
        imp = get_importance(path, src)
        score = score_model(imp or {})
        results.append({
            "label": label, "family": family, "source": src,
            "path": str(path), "importance": imp, "score": score,
        })

    # Sort by hard_share desc, putting None at bottom
    results.sort(key=lambda r: (r["score"]["hard_share"] or -1), reverse=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print("=" * 110)
    print("TROJAN-HORSE FEATURE AUDIT — 33 model artifacts")
    print("=" * 110)
    print(f"Trojan-HARD features (PSI > 1.0): {sorted(TROJAN_HARD)}")
    print()
    print(f"{'MODEL':<60} {'N':>4} {'HARD%':>8} {'SOFT%':>8} {'RISK':>10}  TOP TROJANS")
    print("-" * 130)
    for r in results:
        s = r["score"]
        h = f"{s['hard_share']*100:.1f}%" if s["hard_share"] is not None else "  —  "
        sft = f"{s['soft_share']*100:.1f}%" if s["soft_share"] is not None else "  —  "
        n = s["n_features"] if s["n_features"] else "—"
        risk = risk_label(s)
        tj = ", ".join(f"{k}={v*100:.1f}%" for k, v in s.get("top_trojan", []))
        print(f"{r['label']:<60} {str(n):>4} {h:>8} {sft:>8} {risk:>10}  {tj}")
    print()

    # ── Build markdown report ─────────────────────────────────────────────────
    lines = []
    lines.append("# Trojan-Horse Feature Audit\n")
    lines.append("Generated: 2026-05-17\n\n")
    lines.append("**Hypothesis under test**: Are there other models (beyond the already-broken `moderate_5c_5s` / `extreme_10c_5s`) ")
    lines.append("that over-rely on features with catastrophic distribution drift between training and OOS?\n\n")
    lines.append("**Trojan-HARD features** (PSI > 1.0 in train vs OOS):\n")
    for f in sorted(TROJAN_HARD):
        lines.append(f"- `{f}`\n")
    lines.append("\n**Trojan-SOFT features** (PSI 0.05–0.25 — watch for over-reliance):\n")
    lines.append(f"- {', '.join('`' + f + '`' for f in sorted(TROJAN_SOFT))}\n\n")
    lines.append("**Risk tiers** (based on share of total importance on HARD features):\n")
    lines.append("- **CRITICAL**: ≥30% on HARD features OR any HARD feature in top 5 → expect failure on regime shift\n")
    lines.append("- **HIGH**:     15–30% → vulnerable\n")
    lines.append("- **MODERATE**: 5–15% → monitor\n")
    lines.append("- **LOW**:      <5% → robust\n\n")

    lines.append("## Ranked Vulnerability\n\n")
    lines.append("| Model | N feat | HARD% | SOFT% | Risk | Top trojan features | Top-5 features overall |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in results:
        s = r["score"]
        h = f"{s['hard_share']*100:.1f}%" if s["hard_share"] is not None else "—"
        sft = f"{s['soft_share']*100:.1f}%" if s["soft_share"] is not None else "—"
        n = s["n_features"] if s["n_features"] else "—"
        risk = risk_label(s)
        tj = ", ".join(f"`{k}` ({v*100:.0f}%)" for k, v in s.get("top_trojan", [])[:3])
        top5 = ", ".join(f"`{k}` ({v*100:.0f}%)" for k, v in s.get("top5", [])[:5])
        lines.append(f"| `{r['label']}` | {n} | **{h}** | {sft} | **{risk}** | {tj or '—'} | {top5} |\n")

    # ── Summary by tier ───────────────────────────────────────────────────────
    lines.append("\n## Risk Distribution\n\n")
    tier_counts = {"CRITICAL": [], "HIGH": [], "MODERATE": [], "LOW": [], "n/a": []}
    for r in results:
        tier_counts[risk_label(r["score"])].append(r["label"])
    for tier in ["CRITICAL", "HIGH", "MODERATE", "LOW", "n/a"]:
        lines.append(f"### {tier} ({len(tier_counts[tier])})\n")
        if not tier_counts[tier]:
            lines.append("_(none)_\n\n")
            continue
        for label in tier_counts[tier]:
            lines.append(f"- `{label}`\n")
        lines.append("\n")

    out = Path("docs/trojan_horse_audit_report.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Full report saved to {out}")

    # console tier summary
    print("\nRISK DISTRIBUTION:")
    for tier in ["CRITICAL", "HIGH", "MODERATE", "LOW", "n/a"]:
        print(f"  {tier:10}  {len(tier_counts[tier]):3d}  {tier_counts[tier][:3]}{'...' if len(tier_counts[tier])>3 else ''}")


if __name__ == "__main__":
    main()
