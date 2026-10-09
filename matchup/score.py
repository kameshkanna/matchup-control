"""Task 4 (Person A) — Matchup Win Score (per rep).

One interpretable number per rep. Higher = the blocker won. In one sentence:
a blocker wins when he keeps the rusher far from the QB, stays glued to him,
and shields the QB — so the score rewards low ground-given-up, tight
separation, strong between-ness/mirroring, and a decelerating rusher.

The weights here are sensible defaults; Task 6 (validate.calibrate_weights)
overwrites DEFAULT_WEIGHTS by fitting against real PFF pressure labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as feat

# ---------------------------------------------------------------------------
# Standardisation reference (rough population scales so the weighted sum is
# balanced before calibration). Updated by calibration if desired.
# Each entry: (center, scale). Feature is z = (value - center)/scale.
# ---------------------------------------------------------------------------
FEATURE_STATS = {
    "ground_given_up":       (2.5, 2.5),   # yards; lower is better -> negate weight
    "penetration_past_los":  (4.0, 3.0),   # yards; lower better -> negate
    "sep_mean":              (1.2, 1.0),   # yards; lower better -> negate
    "betweenness_mean":      (0.75, 0.15), # higher better
    "mirroring_mean":        (0.85, 0.15), # higher better
    "rusher_speed_late":     (-2.0, 1.0),  # already negated (higher=better)
    "rusher_accel_late":     (-1.5, 1.0),  # already negated (higher=better)
}

# Positive weight = contributes to a WIN. Features where "low is good" get a
# negative weight so the standardised sum is consistently "higher = win".
DEFAULT_WEIGHTS = {
    "ground_given_up":      -1.4,
    "penetration_past_los": -0.6,
    "sep_mean":             -0.9,
    "betweenness_mean":      0.5,
    "mirroring_mean":        0.4,
    "rusher_speed_late":     0.5,
    "rusher_accel_late":     0.3,
}


def win_score(features: dict, weights: dict | None = None) -> float:
    """Single continuous score, higher = blocker won.

    Standardise each feature, multiply by its weight, sum. Interpretable:
    a positive score means the blocker controlled the rep. Roughly centered
    near 0 for a league-average rep.
    """
    w = weights or DEFAULT_WEIGHTS
    total = 0.0
    used = 0
    for k, wk in w.items():
        v = features.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        center, scale = FEATURE_STATS.get(k, (0.0, 1.0))
        z = (v - center) / (scale if scale else 1.0)
        total += wk * z
        used += 1
    if used == 0:
        return np.nan
    return float(total)


def win_flag(score: float, threshold: float = 0.0) -> int:
    """1 = blocker won the rep, 0 = lost. NaN scores -> 0 (treated as lost/unknown)."""
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return 0
    return int(score >= threshold)


# ---------------------------------------------------------------------------
# Position-relative scoring
# ---------------------------------------------------------------------------
# Tackles block edge rushers on an island, so their rusher naturally travels
# further and closer to the QB than an interior rusher who hits a wall of
# bodies. Judged on one absolute scale, every tackle looks bad. We therefore
# standardise the raw score WITHIN position group so a tackle is compared to
# other tackles, a guard to guards, a center to centers.
POSITION_GROUP = {
    "T": "T", "OT": "T", "LT": "T", "RT": "T",
    "G": "G", "OG": "G", "LG": "G", "RG": "G",
    "C": "C",
}


def position_group(pos) -> str:
    """Map an official position / lined-up position to T / G / C (else 'OTHER')."""
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return "OTHER"
    return POSITION_GROUP.get(str(pos).upper(), "OTHER")


def relativize_by_position(
    scored: pd.DataFrame, raw_col: str = "win_score_abs",
    out_col: str = "win_score", pos_col: str = "blocker_pos",
) -> pd.DataFrame:
    """Standardise raw scores within position group.

    Keeps the absolute score in `raw_col`; writes the position-relative
    z-score into `out_col`. A value of +1 means "one SD better than a typical
    rep at this position". win_flag is recomputed from the relative score.
    """
    df = scored.reset_index(drop=True).copy()
    df["pos_group"] = df[pos_col].apply(position_group)

    raw = pd.to_numeric(df[raw_col], errors="coerce")
    # Per-group mean/std, broadcast back to each row (NaN raw stays NaN).
    grp = df["pos_group"]
    mu = raw.groupby(grp).transform("mean")
    sd = raw.groupby(grp).transform("std").replace(0, np.nan).fillna(1.0)
    rel = (raw - mu) / sd

    df[out_col] = rel
    df["win_flag"] = np.where(rel.isna(), 0, (rel >= 0.0).astype(int))
    return df


def score_matchups(
    matchups_with_features: pd.DataFrame, weights: dict | None = None,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Adds win_score, win_flag columns. SCHEMA: scored.

    Input must already carry the feature columns (see build_scored for the
    convenience path that computes features from a matchups table)."""
    df = matchups_with_features.copy()
    w = weights or DEFAULT_WEIGHTS

    # Calibrated weights (validate.calibrate_weights) PREDICT pressure, i.e. a
    # blocker LOSS: a positive weight there means "more of this feature => more
    # pressure". win_score is the opposite polarity (higher = blocker WON), so
    # such weights must be negated before they drive the score. We detect the
    # calibrated form by its "intercept" key; DEFAULT_WEIGHTS has none and is
    # already in win polarity, so it is used as-is.
    if "intercept" in w:
        w = {k: -v for k, v in w.items() if k != "intercept"}

    # Vectorised standardised weighted sum over the feature columns.
    total = np.zeros(len(df), dtype=float)
    used = np.zeros(len(df), dtype=int)
    for k, wk in w.items():
        if k not in df.columns:
            continue
        center, scale = FEATURE_STATS.get(k, (0.0, 1.0))
        scale = scale if scale else 1.0
        vals = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
        z = (vals - center) / scale
        present = ~np.isnan(z)
        total[present] += wk * z[present]
        used[present] += 1

    score = np.where(used > 0, total, np.nan)
    df["win_score"] = score
    df["win_flag"] = np.where(np.isnan(score), 0, (score >= threshold).astype(int))
    return df


def build_scored(
    matchups: pd.DataFrame, weights: dict | None = None, threshold: float = 0.0,
    relative: bool = True,
) -> pd.DataFrame:
    """Convenience: take a matchups table (SCHEMA: matchups), compute features
    for each pairing_ok rep, and return the SCHEMA: scored table.

    Rows with pairing_ok=False are kept but get NaN features / win_flag=0.
    """
    from . import io_load

    feat_rows = []
    # Group by play so we read + index each play's frames ONCE.
    for (gid, pid), grp in matchups.groupby(["game_id", "play_id"]):
        try:
            play = io_load.get_play(int(gid), int(pid))
            pindex = feat._player_index(play)
        except Exception as e:
            print(f"scored skip play {gid}/{pid}: {e}")
            play, pindex = None, None
        snap = int(grp.snap_frame.iloc[0])
        end = int(grp.end_frame.iloc[0])
        for _, r in grp.iterrows():
            base = {k: r[k] for k in r.index}
            if pindex is not None and bool(r["pairing_ok"]):
                try:
                    fr = feat.rep_features(
                        int(gid), int(pid), int(r.blocker_id), int(r.rusher_id),
                        qb_id=int(r.qb_id), window=(snap, end), pindex=pindex,
                    )
                    base.update({k: fr[k] for k in feat._FEATURE_KEYS if k in fr})
                    base["n_frames"] = fr.get("n_frames")
                except Exception as e:
                    print(f"feat fail {gid}/{pid}/{r.blocker_id}: {e}")
            feat_rows.append(base)

    df = pd.DataFrame(feat_rows)
    df = score_matchups(df, weights=weights, threshold=threshold)
    if relative:
        # Keep the absolute weighted score, make win_score position-relative.
        df["win_score_abs"] = df["win_score"]
        df = relativize_by_position(df, raw_col="win_score_abs", out_col="win_score")
    return df


# ---------------------------------------------------------------------------
# Smoke test — 5 hand-pickable reps on the golden play, ordering should be sane
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from . import pairing

    m = pairing.get_matchups(C.GOLDEN_GAME, C.GOLDEN_PLAY)
    scored = build_scored(m)
    cols = [
        "blocker_id", "blocker_pos", "rusher_id", "pressure_allowed",
        "ground_given_up", "sep_mean", "win_score", "win_flag",
    ]
    scored = scored.sort_values("win_score", ascending=False)
    lines = [
        "scored golden play (sorted best->worst block):",
        scored[cols].to_string(index=False),
        "",
        "CHECK: pressure-allowed reps should trend to lower win_score.",
        f"mean win_score | pressure=0: {scored[scored.pressure_allowed==0].win_score.mean():.3f}",
        f"mean win_score | pressure=1: {scored[scored.pressure_allowed==1].win_score.mean():.3f}",
    ]
    msg = "\n".join(lines)
    print(msg)
    (C.CACHE_DIR / "_smoke_score.txt").write_text(msg, encoding="utf-8")
