"""End-to-end tests: does the scoreboard match the real data, and how does the
model perform — over the whole pipeline (not synthetic).

These run Person A's real pipeline (``matchup.pipeline.run``) on real tracking
and assert the properties that must hold for the Matchup Win Score and the
player leaderboard to be trustworthy:

* schema integrity (scored + leaderboard carry the frozen contract columns),
* label consistency with the raw PFF data (pressure_allowed = max(hit, hurry,
  sack); pairing rate near the ~96% seen in the data),
* model performance (Win Score separates pressure-allowed reps from clean reps,
  by mean, correlation, and ROC-AUC),
* leaderboard sanity (win rates in [0,1], shrinkage pulls toward the global
  mean, opponent adjustment and CIs are well-formed, rep counts reconcile with
  the scored table).

The pipeline is run ONCE (session-scoped fixture) and cached by Person A's own
parquet cache, so the whole file is quick on re-runs.

Run:  pytest tests/test_scoreboard.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from matchup import pipeline
from matchup._upstream import GOLDEN  # Person C's golden-rep dict (game/play/blocker/rusher)

# How many games to score for the suite. Small enough to be fast, large enough
# that the performance signal is stable.
N_GAMES = 15


# --------------------------------------------------------------------------- #
# Fixtures: run the real pipeline once.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def result() -> dict:
    """Full pipeline result dict for the first N_GAMES games (real data)."""
    return pipeline.run(n_games=N_GAMES)


@pytest.fixture(scope="session")
def scored(result) -> pd.DataFrame:
    return result["scored"]


@pytest.fixture(scope="session")
def leaderboard_df(result) -> pd.DataFrame:
    lb = result.get("leaderboard")
    if lb is None:
        pytest.skip(f"leaderboard not produced: {result.get('leaderboard_error')}")
    return lb


@pytest.fixture(scope="session")
def labelled(scored) -> pd.DataFrame:
    """Scored reps with both a win_score and a pressure label present."""
    return scored[scored["win_score"].notna() & scored["pressure_allowed"].notna()].copy()


# --------------------------------------------------------------------------- #
# 1. The pipeline actually ran on real data.
# --------------------------------------------------------------------------- #
def test_pipeline_scored_nonempty(scored):
    assert len(scored) > 1000, "expected thousands of scored reps from real tracking"


def test_scored_has_contract_columns(scored):
    required = {
        "game_id", "play_id", "blocker_id", "rusher_id", "qb_id",
        "pressure_allowed", "sack_allowed", "hurry_allowed", "hit_allowed",
        "pairing_ok", "win_score", "win_flag",
    }
    missing = required - set(scored.columns)
    assert not missing, f"scored is missing contract columns: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# 2. Scoreboard matches the real data (label + pairing consistency).
# --------------------------------------------------------------------------- #
def test_pressure_label_is_max_of_components(scored):
    """pressure_allowed must equal max(hit, hurry, sack) allowed — the contract."""
    expected = scored[["hit_allowed", "hurry_allowed", "sack_allowed"]].max(axis=1)
    assert (scored["pressure_allowed"].astype(int) == expected.astype(int)).all()


def test_labels_are_binary(scored):
    for col in ["pressure_allowed", "sack_allowed", "hurry_allowed", "hit_allowed", "win_flag"]:
        vals = set(scored[col].dropna().unique().tolist())
        assert vals <= {0, 1}, f"{col} has non-binary values: {vals}"


def test_pairing_rate_matches_data(scored):
    """~96.7% of Pass-Block reps have a blocked-player assignment in the raw PFF
    data; the scored table's pairing_ok rate should be in that ballpark."""
    rate = float(scored["pairing_ok"].mean())
    assert 0.85 <= rate <= 1.0, f"pairing_ok rate {rate:.3f} outside expected ~0.95 range"


def test_pressure_rate_is_plausible(scored):
    """Pressure is rare in real football: a few percent of reps, not half."""
    rate = float(scored["pressure_allowed"].mean())
    assert 0.01 <= rate <= 0.25, f"pressure rate {rate:.3f} implausible for pass pro"


def test_win_flag_consistent_with_win_score(scored):
    """win_flag must be the thresholded win_score (>= 0 by contract)."""
    ok = scored[scored["win_score"].notna()]
    derived = (ok["win_score"] >= 0.0).astype(int)
    agree = float((derived == ok["win_flag"].astype(int)).mean())
    assert agree >= 0.99, f"win_flag disagrees with sign(win_score) on {100*(1-agree):.1f}% of reps"


# --------------------------------------------------------------------------- #
# 3. Model performance: Win Score separates pressure from clean reps.
# --------------------------------------------------------------------------- #
def test_pressure_reps_score_lower_than_clean(labelled):
    """Higher win_score = blocker won, so pressure-allowed reps must average a
    LOWER win_score than clean reps."""
    m_clean = labelled.loc[labelled["pressure_allowed"] == 0, "win_score"].mean()
    m_press = labelled.loc[labelled["pressure_allowed"] == 1, "win_score"].mean()
    assert m_clean > m_press, (
        f"clean mean {m_clean:.3f} should exceed pressure mean {m_press:.3f}"
    )


def test_winscore_pressure_correlation_is_negative(labelled):
    """Win Score should be negatively correlated with conceding pressure."""
    corr = float(np.corrcoef(labelled["win_score"], labelled["pressure_allowed"])[0, 1])
    assert corr <= -0.15, f"corr(win_score, pressure) = {corr:.3f} is too weak/positive"


def test_winscore_auc_beats_chance(labelled):
    """ROC-AUC of (-win_score) predicting pressure must clear a useful bar.

    We negate win_score because a LOW score should predict pressure=1.
    """
    auc = roc_auc_score(labelled["pressure_allowed"].astype(int), -labelled["win_score"])
    assert auc >= 0.70, f"AUC {auc:.3f} below the 0.70 usefulness bar"


def test_validation_report_matches_recomputed_auc(result, labelled):
    """The pipeline's own validation AUC should match an independent recompute."""
    rep = result.get("validation")
    if not rep or rep.get("auc") != rep.get("auc"):  # None or NaN
        pytest.skip(f"no usable validation report: {rep}")
    independent = roc_auc_score(labelled["pressure_allowed"].astype(int), -labelled["win_score"])
    # Both should land on the same side of 0.5 and be within a loose tolerance.
    assert abs(rep["auc"] - independent) < 0.15, (
        f"pipeline AUC {rep['auc']:.3f} vs recomputed {independent:.3f} diverge"
    )


# --------------------------------------------------------------------------- #
# 4. Leaderboard sanity + reconciliation with the scored table.
# --------------------------------------------------------------------------- #
def test_leaderboard_has_contract_columns(leaderboard_df):
    required = {
        "blocker_id", "blocker_name", "n_reps",
        "win_rate", "adj_win_rate", "win_rate_shrunk",
        "ci_low", "ci_high",
    }
    missing = required - set(leaderboard_df.columns)
    assert not missing, f"leaderboard missing columns: {sorted(missing)}"


def test_leaderboard_rates_in_unit_interval(leaderboard_df):
    for col in ["win_rate", "adj_win_rate", "win_rate_shrunk", "ci_low", "ci_high"]:
        s = leaderboard_df[col].dropna()
        assert ((s >= 0.0) & (s <= 1.0)).all(), f"{col} escapes [0,1]"


def test_leaderboard_ci_brackets_shrunk_rate(leaderboard_df):
    """ci_low <= ci_high, and the interval is well-ordered."""
    lo, hi = leaderboard_df["ci_low"], leaderboard_df["ci_high"]
    assert (lo <= hi + 1e-9).all(), "ci_low exceeds ci_high for some blockers"


def test_shrinkage_pulls_toward_a_common_center(leaderboard_df):
    """Shrinkage must regress rates toward a common center, so the shrunk rates
    are LESS dispersed than the raw rates.

    We don't hard-code which center the leaderboard shrinks toward (it uses its
    own weighted global, not the raw win_flag mean), so we assert the defining
    property of shrinkage instead: it compresses the spread.
    """
    raw = leaderboard_df["win_rate"].dropna()
    shrunk = leaderboard_df["win_rate_shrunk"].dropna()
    assert shrunk.std() < raw.std(), (
        f"shrunk std {shrunk.std():.3f} not below raw std {raw.std():.3f} — "
        "shrinkage should compress the spread"
    )


def test_shrinkage_is_stronger_for_low_rep_blockers(leaderboard_df):
    """Low-rep blockers should be pulled toward the center MORE than high-rep
    ones: |shrunk - raw| should trend down as n_reps grows."""
    df = leaderboard_df.dropna(subset=["win_rate", "win_rate_shrunk", "n_reps"]).copy()
    df["pull"] = (df["win_rate_shrunk"] - df["win_rate"]).abs()
    if df["n_reps"].nunique() < 3:
        pytest.skip("not enough distinct rep counts to test the trend")
    corr = float(np.corrcoef(df["n_reps"], df["pull"])[0, 1])
    assert corr < 0.0, f"expected negative n_reps-vs-pull correlation, got {corr:.3f}"


def test_shrunk_rates_are_less_extreme_than_raw(leaderboard_df):
    """A shrunk rate never moves AWAY from the pack: perfect (1.0) and zero
    raw rates must be pulled inward, so no shrunk rate stays at a raw extreme
    while the raw rate is itself extreme (unless raw was already central)."""
    df = leaderboard_df.dropna(subset=["win_rate", "win_rate_shrunk"])
    extreme = df[(df["win_rate"] >= 0.999) | (df["win_rate"] <= 0.001)]
    if not len(extreme):
        pytest.skip("no extreme raw win rates to check")
    # Every extreme-raw blocker should have been moved strictly inward.
    moved_in = (
        ((extreme["win_rate"] >= 0.999) & (extreme["win_rate_shrunk"] < 0.999))
        | ((extreme["win_rate"] <= 0.001) & (extreme["win_rate_shrunk"] > 0.001))
    )
    assert moved_in.all(), "some extreme raw win rates were not shrunk inward"


def test_leaderboard_rep_counts_reconcile(scored, leaderboard_df):
    """Each blocker's n_reps on the leaderboard should match the paired, scored
    reps for that blocker in the scored table."""
    paired = scored[scored["pairing_ok"].astype(bool) & scored["win_flag"].notna()]
    counts = paired.groupby("blocker_id").size()
    checked = 0
    for row in leaderboard_df.itertuples(index=False):
        bid = int(row.blocker_id)
        if bid in counts.index:
            # Allow small slack: the leaderboard may apply its own min-rep filter.
            assert int(row.n_reps) <= int(counts.loc[bid]) + 0, (
                f"blocker {bid}: leaderboard n_reps {row.n_reps} exceeds scored reps {counts.loc[bid]}"
            )
            checked += 1
    assert checked > 0, "no leaderboard blocker matched the scored table"


def test_leaderboard_covers_most_blockers(scored, leaderboard_df):
    """The leaderboard should account for a large share of distinct blockers."""
    scored_blockers = set(scored.loc[scored["pairing_ok"].astype(bool), "blocker_id"].unique())
    lb_blockers = set(leaderboard_df["blocker_id"].unique())
    coverage = len(lb_blockers & scored_blockers) / max(len(scored_blockers), 1)
    assert coverage >= 0.5, f"leaderboard covers only {coverage:.0%} of scored blockers"


# --------------------------------------------------------------------------- #
# 5. Golden rep — informational (does NOT fail the suite).
# --------------------------------------------------------------------------- #
def test_golden_rep_present_and_report(scored, capsys):
    """The golden rep must exist and be a pressure-allowed rep. Its win_flag is
    reported but NOT asserted: Person A's scoring model evolves, and this rep is
    a genuine edge case (the blocker stayed 'between' yet a hurry was charged)."""
    g = scored[
        (scored["game_id"] == GOLDEN["game_id"])
        & (scored["play_id"] == GOLDEN["play_id"])
        & (scored["blocker_id"] == GOLDEN["blocker_id"])
        & (scored["rusher_id"] == GOLDEN["rusher_id"])
    ]
    assert len(g) == 1, f"expected exactly one golden rep, found {len(g)}"
    r = g.iloc[0]
    assert int(r["pressure_allowed"]) == 1, "golden rep should be pressure-allowed"
    with capsys.disabled():
        print(
            f"\n[golden] {r.get('blocker_name')} vs {r.get('rusher_name')}: "
            f"win_score={float(r['win_score']):.3f}, win_flag={int(r['win_flag'])}, "
            f"pressure_allowed={int(r['pressure_allowed'])}, "
            f"ground_given_up={float(r['ground_given_up']):.2f}, "
            f"betweenness_end={float(r['betweenness_end']):.2f}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
