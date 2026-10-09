"""Task 8 — opponent-quality adjustment and the player win-rate leaderboard.

This module turns per-rep scores into the headline deliverable: a blocker
win-rate leaderboard that is adjusted for the quality of the rushers each blocker
faced and stabilised with empirical-Bayes shrinkage and Wilson confidence bounds.

Functions (all pure DataFrame transforms):

* :func:`rusher_strength` — per-rusher difficulty (higher = harder to block).
* :func:`opponent_adjust` — add ``adj_win_score`` via rusher fixed-effect removal.
* :func:`player_leaderboard` — per-blocker aggregate with shrinkage + Wilson CI.

Sign convention: ``win_score`` and ``adj_win_score`` are higher when the blocker
won; ``win_flag`` / ``adj_win_flag`` are 1 when the blocker won. ``pairing_ok`` is
honoured so that only cleanly paired reps enter the aggregates, while counts stay
honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchup.config import WIN_SCORE_THRESHOLD, _make_synthetic_scored

# 95% Wilson interval z-score.
_Z_95 = 1.959963984540054


def _paired(df: pd.DataFrame) -> pd.DataFrame:
    """Return only cleanly paired reps when ``pairing_ok`` exists; else the frame."""
    if "pairing_ok" in df.columns:
        return df[df["pairing_ok"].astype(bool)].copy()
    return df.copy()


def rusher_strength(scored: pd.DataFrame) -> pd.DataFrame:
    """Estimate per-rusher difficulty (higher ``strength`` = harder to block).

    For each rusher we compute how often blockers facing him conceded pressure and
    how low their win scores were, then blend the two into a single z-scored index::

        pressure_rate = mean(pressure_allowed) over the rusher's reps
        neg_win       = mean(-win_score) over the rusher's reps   (higher = tougher)
        strength      = mean( z(pressure_rate), z(neg_win) )

    Both components point the same way (a harder rusher yields more pressure and
    lower blocker win scores), so averaging their z-scores gives a balanced,
    unit-free difficulty score centred near 0.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame.

    Returns
    -------
    pandas.DataFrame
        Columns ``rusher_id, rusher_name, n_reps, pressure_rate, strength``.
    """
    work = _paired(scored)
    grp = work.groupby("rusher_id", sort=False)
    out = grp.agg(
        rusher_name=("rusher_name", "first"),
        n_reps=("rusher_id", "size"),
        pressure_rate=("pressure_allowed", "mean"),
        neg_win=("win_score", lambda s: float(-np.mean(s))),
    ).reset_index()

    def _z(col: pd.Series) -> pd.Series:
        std = col.std(ddof=0)
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - col.mean()) / std

    out["strength"] = (_z(out["pressure_rate"]) + _z(out["neg_win"])) / 2.0
    out = out.drop(columns=["neg_win"])
    return out[["rusher_id", "rusher_name", "n_reps", "pressure_rate", "strength"]]


def opponent_adjust(scored: pd.DataFrame) -> pd.DataFrame:
    """Add ``adj_win_score``: ``win_score`` with the rusher fixed effect removed.

    Each rusher's mean ``win_score`` captures how easy or hard he is to block. We
    subtract that rusher mean from every rep and add back the global mean so the
    population mean is preserved (pure re-centring, not rescaling)::

        adj_win_score = win_score - mean_win_score[rusher] + global_mean_win_score

    A blocker who posts an average score against a tough rusher (low rusher mean)
    is credited above population average, and vice versa. The original ``win_score``
    is left untouched.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame containing ``win_score`` and ``rusher_id``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``scored`` with an added ``adj_win_score`` column.
    """
    out = scored.copy()
    global_mean = float(out["win_score"].mean())
    rusher_mean = out.groupby("rusher_id")["win_score"].transform("mean")
    out["adj_win_score"] = out["win_score"] - rusher_mean + global_mean
    return out


def _wilson_interval(wins: np.ndarray, n: np.ndarray, z: float = _Z_95) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Wilson score interval for a binomial proportion.

    The Wilson interval is chosen over the normal approximation because it stays
    within ``[0, 1]`` and behaves well for small ``n`` and extreme rates — exactly
    the regime of blockers with few reps.
    """
    n = np.where(n == 0, np.nan, n.astype(float))
    phat = wins / n
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    low = centre - margin
    high = centre + margin
    return np.clip(low, 0.0, 1.0), np.clip(high, 0.0, 1.0)


def player_leaderboard(scored_adj: pd.DataFrame) -> pd.DataFrame:
    """Build the per-blocker win-rate leaderboard with shrinkage and Wilson CIs.

    If ``adj_win_score`` is absent, :func:`opponent_adjust` is called first.
    ``adj_win_flag`` is derived from ``adj_win_score`` using the SAME numeric
    boundary as ``win_flag`` (``>= WIN_SCORE_THRESHOLD``, i.e. 0), keeping the two
    flag definitions consistent.

    Empirical-Bayes shrinkage
    -------------------------
    Per-blocker win rates are shrunk toward the global win rate with a Beta-Binomial
    prior whose strength ``k`` (pseudo-count) is estimated by method-of-moments from
    the spread of per-blocker rates::

        global_rate = mean(win_flag) over all reps
        var_obs     = variance of per-blocker win rates
        var_binom   = global_rate*(1-global_rate) / mean(n_reps)   (sampling noise)
        var_true    = max(var_obs - var_binom, tiny)
        k           = global_rate*(1-global_rate)/var_true - 1       (prior strength)
        win_rate_shrunk = (wins + k*global_rate) / (n_reps + k)

    This pulls low-rep blockers toward the global mean while leaving high-rep
    blockers close to their observed rate, so every ``win_rate_shrunk`` lies between
    the blocker's raw ``win_rate`` and the global rate.

    Parameters
    ----------
    scored_adj:
        A ``scored``-schema frame, ideally already carrying ``adj_win_score``.

    Returns
    -------
    pandas.DataFrame
        Columns ``blocker_id, blocker_name, blocker_pos, team, n_reps, win_rate,
        adj_win_rate, win_rate_shrunk, mean_win_score, adj_mean_win_score, ci_low,
        ci_high``, sorted by ``win_rate_shrunk`` descending.
    """
    df = scored_adj
    if "adj_win_score" not in df.columns:
        df = opponent_adjust(df)
    df = _paired(df)
    df = df.copy()
    df["adj_win_flag"] = (df["adj_win_score"] >= WIN_SCORE_THRESHOLD).astype(int)

    grp = df.groupby("blocker_id", sort=False)
    lb = grp.agg(
        blocker_name=("blocker_name", "first"),
        blocker_pos=("blocker_pos", "first"),
        team=("team", "first"),
        n_reps=("blocker_id", "size"),
        win_rate=("win_flag", "mean"),
        adj_win_rate=("adj_win_flag", "mean"),
        mean_win_score=("win_score", "mean"),
        adj_mean_win_score=("adj_win_score", "mean"),
    ).reset_index()

    # Global win rate across all paired reps (rep-weighted).
    global_rate = float(df["win_flag"].mean())

    # Method-of-moments prior strength k.
    rates = lb["win_rate"].to_numpy(dtype=float)
    n_reps = lb["n_reps"].to_numpy(dtype=float)
    var_obs = float(np.var(rates, ddof=0)) if len(rates) > 1 else 0.0
    mean_n = float(np.mean(n_reps)) if len(n_reps) else 1.0
    var_binom = global_rate * (1.0 - global_rate) / max(mean_n, 1.0)
    var_true = max(var_obs - var_binom, 1e-9)
    denom = var_true
    k = global_rate * (1.0 - global_rate) / denom - 1.0
    if not np.isfinite(k) or k <= 0.0:
        k = max(mean_n, 1.0)  # fall back to a mild prior if the estimate degenerates

    wins = (rates * n_reps)
    lb["win_rate_shrunk"] = (wins + k * global_rate) / (n_reps + k)

    low, high = _wilson_interval(wins, n_reps)
    lb["ci_low"] = low
    lb["ci_high"] = high

    cols = [
        "blocker_id",
        "blocker_name",
        "blocker_pos",
        "team",
        "n_reps",
        "win_rate",
        "adj_win_rate",
        "win_rate_shrunk",
        "mean_win_score",
        "adj_mean_win_score",
        "ci_low",
        "ci_high",
    ]
    lb = lb[cols].sort_values("win_rate_shrunk", ascending=False).reset_index(drop=True)
    return lb


if __name__ == "__main__":
    scored = _make_synthetic_scored(n_reps=400, seed=0)
    print("leaderboard.py demo")
    print("  scored shape:", scored.shape)

    rs = rusher_strength(scored)
    print("\n  rusher_strength (head):")
    print(rs.head().to_string(index=False))

    adj = opponent_adjust(scored)
    print(
        "\n  opponent_adjust: added adj_win_score; "
        f"mean win_score={adj.win_score.mean():.4f}, mean adj_win_score={adj.adj_win_score.mean():.4f} "
        "(should match)"
    )

    lb = player_leaderboard(adj)
    print("\n  player_leaderboard (top 10 by win_rate_shrunk):")
    print(lb.head(10).to_string(index=False))

    # --- invariant checks ---------------------------------------------------- #
    global_rate = float(scored.loc[scored["pairing_ok"].astype(bool), "win_flag"].mean())
    between = (
        ((lb["win_rate_shrunk"] >= np.minimum(lb["win_rate"], global_rate) - 1e-9)
         & (lb["win_rate_shrunk"] <= np.maximum(lb["win_rate"], global_rate) + 1e-9))
    )
    assert between.all(), "win_rate_shrunk must lie between raw win_rate and global rate"
    print(f"\n  OK: every win_rate_shrunk between raw win_rate and global rate ({global_rate:.3f})")

    golden = scored[(scored.blocker_id == 42377) & (scored.rusher_id == 42403)]
    assert int(golden.win_flag.iloc[0]) == 0, "golden row must be a blocker loss"
    print("  OK: golden row (42377 vs 42403) win_flag == 0")

    schema = {
        "blocker_id", "blocker_name", "blocker_pos", "team", "n_reps", "win_rate",
        "adj_win_rate", "win_rate_shrunk", "mean_win_score", "adj_mean_win_score",
        "ci_low", "ci_high",
    }
    assert schema.issubset(lb.columns), "leaderboard missing schema columns"
    print("  OK: leaderboard has all schema columns")
