"""Claim-verification tests.

The rest of the suite tests infrastructure. These tests guard the numbers that
appear in README.md, docs/RESULTS.md and the manuscript, so a published claim
cannot drift away from the artifact it came from without the suite going red.

They deliberately read only committed artifacts, so they run on a fresh clone
with no data download.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------- forward test

def test_shadow_summary_matches_published_numbers() -> None:
    """The post-lock replay figures quoted in RESULTS.md and README.md."""
    s = _json(ARTIFACTS / "tcn_shadow_parity" / "shadow_pnl_summary.json")
    assert s["days"] == 5
    assert s["total"]["trades"] == 262
    assert s["total"]["pnl"] == pytest.approx(23.90, abs=0.01)
    lo, hi = s["bootstrap"]["total_pnl_ci90"]
    assert (lo, hi) == pytest.approx((-15.95, 60.36), abs=0.01)
    # The gate is not met, and the documents must not claim otherwise.
    assert s["bootstrap"]["gate_ci90_excludes_0"] is False


def test_shadow_entries_are_replay_derived() -> None:
    """RESULTS.md states the scored entries come from an offline replay.

    If this ever changes to live decision logs the wording has to change with it.
    """
    day = _json(ARTIFACTS / "tcn_shadow_parity" / "shadow_pnl_2026-07-04.json")
    assert day["entry_source"] == "parity"
    assert {t["entry_source"] for t in day["trades"]} == {"parity_expected"}


# ---------------------------------------------------------------- audit gates

def test_snooping_verdict_is_kill_for_both_slots() -> None:
    r = _json(ARTIFACTS / "audit_v1" / "snooping_report.json")
    for slot, verdict in r["d2_verdicts"].items():
        assert verdict["survives_d2"] is False, slot
    assert r["whites_reality_check"]["passes_wrc_10pct"] is False


def test_wrc_recomputes_above_any_rejection_threshold() -> None:
    """The corrected audit includes the combined reported winner."""
    blob = np.load(ARTIFACTS / "audit_v1" / "wrc_universe_daily_pnl.npz")
    old_universe = blob["daily_pnl"]
    old_report = _json(ARTIFACTS / "audit_v1" / "snooping_report.json")
    combined_by_day = old_report["slots"]["combined"]["summary"]["by_day"]
    combined = np.asarray([combined_by_day.get(day, 0.0) for day in old_report["days"]])
    universe = np.vstack((old_universe, combined))
    n_days = universe.shape[1]
    observed = float(universe.mean(axis=1).max())
    centred = universe - universe.mean(axis=1, keepdims=True)

    rng = np.random.default_rng(20260809)
    b, p_restart = 2_000, 1.0 / 3.0
    idx = np.empty((b, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_days, size=b)
    for t in range(1, n_days):
        restart = rng.random(b) < p_restart
        idx[:, t] = np.where(restart, rng.integers(0, n_days, size=b),
                             (idx[:, t - 1] + 1) % n_days)
    null_max = np.array([centred[:, row].mean(axis=1).max() for row in idx])
    p_value = float(((null_max >= observed).sum() + 1) / (b + 1))

    assert universe.shape == (847, 32)
    assert p_value > 0.10, f"WRC p={p_value:.3f} would reverse the published verdict"


def test_deflated_sharpe_below_selection_benchmark() -> None:
    """The claim is that observed Sharpe sits *below* the expected max of N."""
    r = _json(ARTIFACTS / "audit_v1" / "snooping_report.json")
    for slot in ("early_tcn_75_120_utc02_13", "combined"):
        entry = next(d for d in r["slots"][slot]["dsr_sensitivity"] if d["n_trials"] == 846)
        assert entry["sr_daily"] < entry["sr_benchmark_expected_max"], slot
        assert entry["passes_dsr_gt_0.95"] is False, slot


def test_walkforward_intervals_all_contain_zero() -> None:
    r = _json(ARTIFACTS / "audit_v1" / "walkforward_policy_report.json")
    for rule, res in r["results"].items():
        lo, hi = res["oos_total_ci95"]
        assert lo < 0 < hi, f"{rule} interval no longer contains zero"


# ------------------------------------------------------- committed-output consistency

def test_committed_backtest_output_matches_the_lock() -> None:
    """D1 is artifact consistency; the absent data/model prevent a public rerun."""
    s = _json(ARTIFACTS / "backtest_repro_v1" / "summary_test_cap_drop.json")
    c = s["combined"]
    assert c["trades"] == 706
    assert c["total_pnl"] == pytest.approx(214.1461, abs=1e-4)
    assert c["worst_day"] == pytest.approx(-18.0556, abs=1e-4)


def test_fill_worse_changes_the_trade_set() -> None:
    """RESULTS.md downgrades D4 partly because the stress is not decision-matched.

    If someone later fixes the stress to hold the decision set fixed, this test
    fails and the D4 wording should be revisited.
    """
    base = _json(ARTIFACTS / "backtest_repro_v1" / "summary_test_cap_drop.json")
    worse = _json(ARTIFACTS / "backtest_repro_v1" / "summary_test_fill_worse.json")
    assert base["combined"]["trades"] != worse["combined"]["trades"]


# ------------------------------------------------------------- calibration

def test_brier_summary_is_current_and_supports_the_published_wording() -> None:
    summary = _json(ARTIFACTS / "tcn_v2_eval" / "brier_summary.json")
    rows = {r["calibrator"]: r for r in summary["rows"]}
    baseline_test = summary["market_baseline"]["test"]["brier"]

    platt = rows["val_platt_l2"]["test"]
    assert summary["calibrator_count"] == len(rows) == 7
    assert summary["lowest_observed_test_brier_post_hoc"] == "val_platt_l2"
    assert platt["mean_brier_across_seeds"] == pytest.approx(0.12773, abs=1e-5)
    assert baseline_test == pytest.approx(0.12828, abs=1e-5)
    assert platt["seeds_beating_market"] == 4
    lo, hi = platt["market_cluster_bootstrap"]["ci95"]
    assert lo < 0 < hi
    assert platt["market_cluster_ci_includes_zero"] is True
    assert all(
        row["test"]["market_cluster_ci_includes_zero"] is True
        for row in summary["rows"]
    )

    # The published caution rests on isotonic winning validation and losing test.
    assert summary["lowest_validation_brier"] == "val_isotonic"
    assert rows["val_isotonic"]["test"]["delta_vs_market"] > 0
    assert rows["val_isotonic"]["test"]["seeds_beating_market"] == 0

    # The advantage must stay small enough that "suggestive, not established" holds.
    assert abs(platt["delta_vs_market"]) < 0.001


def test_brier_table_regenerates_identically() -> None:
    """`build_brier_table.py --check` is part of the release check."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_brier_table.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------- canonical claim register

def test_canonical_claim_register_matches_committed_artifacts() -> None:
    register = _json(ARTIFACTS / "publication_claims.json")
    claims = {item["id"]: item for item in register["claims"]}

    assert claims["C1_DATASET_CLOCK"]["verdict"] == "SOURCE-TIME COUNTERFACTUAL"
    assert claims["C2_BRIER"]["verdict"] == "EXPLORATORY; IMPROVEMENT NOT DEMONSTRATED"
    assert claims["C3_LOCKED_BACKTEST"]["verdict"] == "INVALID AS EDGE EVIDENCE"
    assert claims["C4_OUTPUT_CONSISTENCY"]["verdict"] == "PUBLIC PARTIAL REPRODUCTION: PASS"
    assert claims["C5_EXECUTION"]["verdict"] == "UNRESOLVED / INVALID AUDIT"
    assert claims["C6_POST_LOCK_REPLAY"]["verdict"] == (
        "INCONCLUSIVE SOURCE-TIME OFFLINE POST-LOCK REPLAY"
    )
    assert claims["C11_INTELLECTUAL_OWNERSHIP"]["verdict"] == (
        "AUTHOR-CONCEIVED AND DIRECTED"
    )
    assert claims["C10_OVERALL"]["verdict"] == "NO DEMONSTRATED TRADING EDGE"
    brier = _json(ARTIFACTS / "tcn_v2_eval" / "brier_summary.json")
    platt = next(row for row in brier["rows"] if row["calibrator"] == "val_platt_l2")
    cm = claims["C2_BRIER"]["metrics"]
    assert cm["model_test_brier"] == pytest.approx(
        platt["test"]["mean_brier_across_seeds"]
    )
    assert cm["market_test_brier"] == pytest.approx(brier["market_baseline"]["test"]["brier"])
    assert cm["delta_model_minus_market"] == pytest.approx(platt["test"]["delta_vs_market"])
    assert cm["seeds_below_market"] == platt["test"]["seeds_beating_market"]
    assert cm["calibrators_inspected"] == len(brier["rows"]) == 7
    assert cm["market_cluster_ci95"] == pytest.approx(
        platt["test"]["market_cluster_bootstrap"]["ci95"]
    )

    audit = _json(ARTIFACTS / "audit_v1" / "snooping_report.json")
    corrected = _json(ARTIFACTS / "audit_v2" / "selection_audit.json")
    ranks = audit["whites_reality_check"]["locked_config_ranks"]
    am = claims["C3_LOCKED_BACKTEST"]["metrics"]
    assert am["corrected_wrc_candidates"] == 847
    assert am["corrected_wrc_p_value"] == pytest.approx(
        corrected["corrected_public_reproduction"]["finite_sample_corrected_p_value"]
    )
    assert am["locked_early_rank_in_original_matrix"] == (
        ranks["u_early_hours_2-13"]["rank_by_total_pnl"]
    ) == 5
    assert am["locked_late_rank_in_original_matrix"] == (
        ranks["u_b1.25_ttc50-75_ev0.125_px0.2-0.8"]["rank_by_total_pnl"]
    ) == 16
    assert am["separate_all_day_early_candidate_rank"] == (
        ranks["u_b1_ttc75-120_ev0.075_px0.2-0.8"]["rank_by_total_pnl"]
    ) == 2

    replay = _json(ARTIFACTS / "tcn_shadow_parity" / "shadow_pnl_summary.json")
    rm = claims["C6_POST_LOCK_REPLAY"]["metrics"]
    assert rm["days"] == replay["days"]
    assert rm["trades"] == replay["total"]["trades"]
    assert rm["pnl"] == pytest.approx(replay["total"]["pnl"])
    assert rm["ci90"] == pytest.approx(replay["bootstrap"]["total_pnl_ci90"])


def test_generated_claim_register_is_current() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_publication_claims.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_paper_matches_its_source_and_claim_register() -> None:
    """The builder owns the expanded quantitative source-hash contract."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "build_public_paper.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_historical_episode_builder_requires_counterfactual_opt_in() -> None:
    source = (ROOT / "tools" / "build_btc_5m_episode_dataset.py").read_text(encoding="utf-8")
    assert "--allow-source-time-counterfactual" in source
    assert "source_time_counterfactual_not_live_achievable" in source
    assert '"causal_availability": False' in source


def test_removed_generated_clutter_does_not_return() -> None:
    assert not (ROOT / "paper" / "figures").exists()
    assert list((ROOT / "config" / "schemas").glob("*.tmp")) == []


# ------------------------------------------------------------------ documents

def test_documents_do_not_reclaim_a_demonstrated_edge() -> None:
    """Guard against the overclaims this project has already had to retract.

    Each pattern must match the phrase *as an assertion*. The same words are
    allowed inside an explicit retraction ("I previously wrote that ..."), which
    is why these are anchored at a sentence start rather than matched anywhere.
    """
    import re

    banned = [
        r"(?:^|[.!?]\s+)genuinely beats\b",
        r"(?:^|[.!?]\s+)Execution is not the binding constraint",
        r"(?:^|[.!?]\s+)That is a real informational advantage",
        # Calling our own gates "pre-registered" claims verifiable ordering we
        # cannot demonstrate. Saying they were *not* publicly pre-registered is
        # exactly the disclaimer we want, so only the claiming forms are banned.
        r"\bpre-registered (?:audit|gate|requirement|criteri|protocol|threshold)",
        r"(?:^|[.!?]\s+)(?:The|This) (?:result|evaluation) is a live forward test\b",
        r"\ball joins are causal\b",
        r"\bExecution (?:passes|passed) comfortably\b",
    ]
    for name in ("README.md", "docs/RESULTS.md", "docs/METHODOLOGY.md", "paper/manuscript.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for pattern in banned:
            m = re.search(pattern, text, flags=re.MULTILINE)
            assert m is None, f"{name} asserts: {m.group(0)[:60]!r}"


def test_results_states_the_source_time_caveat() -> None:
    text = (DOCS / "RESULTS.md").read_text(encoding="utf-8")
    assert "zero-latency" in text
    assert "source time" in text or "source timestamps" in text


def test_data_availability_lists_the_published_evaluation_artifacts() -> None:
    text = (DOCS / "DATA_AVAILABILITY.md").read_text(encoding="utf-8")
    assert "tcn_v2_eval" in text
    assert (ARTIFACTS / "tcn_v2_eval" / "brier_summary.json").exists()
    assert "evaluation_repro_v2" in text
    assert "audit_v2" in text


def test_no_broken_relative_links_in_top_level_docs() -> None:
    import re

    pattern = re.compile(r"\[[^\]]*\]\(([^)#][^)]*?)\)")
    # Globbed rather than hard-coded so renaming or removing a top-level document
    # cannot leave this guard pointing at a file that no longer exists.
    for md in [*sorted(ROOT.glob("*.md")), *sorted(DOCS.glob("*.md"))]:
        for m in pattern.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (md.parent / target).exists(), f"{md.name} -> {target}"
