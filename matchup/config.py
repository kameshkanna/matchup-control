"""Shared configuration, frozen schema, and synthetic-data generator for Matchup Control.

This is the one module every Person B analytics module imports. It owns:

* The field/sampling constants (:data:`FIELD_LEN`, :data:`FIELD_WID`, :data:`HZ`).
* The resolved data/cache directories (:data:`DATA_DIR`, :data:`CACHE_DIR`), derived
  from ``__file__`` — never hardcoded absolute paths.
* The frozen ``scored`` schema (:data:`MATCHUPS_COLS`, :data:`FEATURES_KEYS`,
  :data:`SCORED_COLS`) that downstream modules code strictly against.
* :func:`_make_synthetic_scored`, the shared generator that lets every module run
  in isolation today — before Person A's upstream (``io_load``/``pairing``/
  ``features``/``score``/``pipeline``) exists.

Canonical identifier column names (snake_case, used throughout the package)
-------------------------------------------------------------------------
* ``game_id`` — integer game identifier.
* ``play_id`` — integer play identifier (unique within a game).
* ``nfl_id`` — integer player identifier. The ball uses the sentinel ``-1``
  (never ``NaN``) so joins and dtype stay integral.
* ``frame_id`` — integer 1-indexed tracking frame within a play (10 Hz).

The raw CSVs under ``data/`` use camelCase (``gameId``, ``nflId``, ...); the
camelCase -> snake_case rename is Person A's ``io_load`` responsibility. Person B
never reads the raw CSVs: it codes against the snake_case frozen schema below and
the synthetic generator emits snake_case directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Field / sampling constants (NFL Big Data Bowl 2023 regional set, 10 Hz).
# --------------------------------------------------------------------------- #
FIELD_LEN: float = 120.0  # yards, end zone to end zone (including both end zones)
FIELD_WID: float = 53.3   # yards, sideline to sideline
HZ: int = 10              # tracking sample rate in frames per second

# --------------------------------------------------------------------------- #
# Directories, resolved relative to this file (no absolute paths committed).
# DATA_DIR -> <workspace>/data (read-only), CACHE_DIR -> <workspace>/cache.
# --------------------------------------------------------------------------- #
DATA_DIR: Path = (Path(__file__).resolve().parent.parent / "data")
CACHE_DIR: Path = (Path(__file__).resolve().parent.parent / "cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Frozen schema. Downstream modules must never rename or drop these columns.
# Extra columns are permitted. `game_id/play_id/blocker_id/rusher_id` are shared
# between the matchups and features groups and appear once in `scored`.
# --------------------------------------------------------------------------- #
MATCHUPS_COLS: list[str] = [
    "game_id",
    "play_id",
    "blocker_id",
    "blocker_name",
    "blocker_pos",
    "rusher_id",
    "rusher_name",
    "rusher_pos",
    "qb_id",
    "snap_frame",
    "end_frame",
    "block_type",
    "is_switch",
    "is_chip",
    "is_double",
    "pressure_allowed",  # int 0/1 = max(hit_allowed, hurry_allowed, sack_allowed)
    "sack_allowed",
    "hurry_allowed",
    "hit_allowed",
    "pairing_ok",  # bool
    "team",        # blocker's team abbreviation
]

# Control-feature keys. Convention: every feature is SIGNED so that a larger
# value means the blocker won the rep. "Toward the QB" is a negative change in
# rusher_to_qb distance, so ground_given_up / penetration / rusher speed+accel
# are losses for the blocker and are kept small when the blocker wins.
FEATURES_KEYS: list[str] = [
    "game_id",
    "play_id",
    "blocker_id",
    "rusher_id",
    "ground_given_up",       # yards of rusher->QB distance surrendered (small = good)
    "min_rusher_to_qb_dist",  # closest the rusher got to the QB (large = good)
    "penetration_past_los",   # yards the rusher pushed past the LOS (small = good)
    "sep_mean",               # mean blocker-rusher separation (large = good)
    "sep_min",                # minimum blocker-rusher separation (large = good)
    "betweenness_mean",       # mean "stays between rusher and QB" (large = good)
    "betweenness_end",        # end-of-rep betweenness (large = good)
    "mirroring_mean",         # mean direction-vector agreement (large = good)
    "rusher_speed_late",      # rusher speed in final 0.5s (small = good)
    "rusher_accel_late",      # rusher accel in final 0.5s (small = good)
    "rep_duration_sec",       # rep length in seconds (context feature)
]

# `scored` = matchups + features + win_score + win_flag, de-duplicated and ordered.
_FEATURE_ONLY_KEYS: list[str] = [
    k
    for k in FEATURES_KEYS
    if k not in ("game_id", "play_id", "blocker_id", "rusher_id")
]
SCORED_COLS: list[str] = MATCHUPS_COLS + _FEATURE_ONLY_KEYS + ["win_score", "win_flag"]

# Threshold convention for turning the continuous `win_score` into `win_flag`.
# `win_flag == 1` (blocker won) when `win_score >= WIN_SCORE_THRESHOLD`. The
# synthetic generator centers `win_score` on 0, so the natural boundary is 0.
# `leaderboard.adj_win_flag` reuses this exact numeric boundary on `adj_win_score`.
WIN_SCORE_THRESHOLD: float = 0.0


# --------------------------------------------------------------------------- #
# Shared synthetic generator.
# --------------------------------------------------------------------------- #
def _make_synthetic_scored(n_reps: int = 400, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic ``scored`` DataFrame with realistic, self-consistent signal.

    The frame matches the frozen ``scored`` schema exactly (see :data:`SCORED_COLS`)
    and is used by every Person B module's ``__main__`` demo and inline tests so the
    modules run in isolation before Person A's upstream exists.

    Signal construction
    --------------------
    1. Draw a latent Bernoulli ``blocker_won`` per rep (~55% base win rate).
    2. Derive each control feature from the latent with Gaussian noise, honouring the
       sign convention: when the blocker won, "good-for-blocker" features
       (``min_rusher_to_qb_dist``, ``sep_mean``, ``sep_min``, ``betweenness_*``,
       ``mirroring_mean``) are larger and "bad-for-blocker" features
       (``ground_given_up``, ``penetration_past_los``, ``rusher_speed_late``,
       ``rusher_accel_late``) are smaller. All features are finite.
    3. Derive ``pressure_allowed`` via a logistic link on the latent, so a blocker
       loss is much more likely to concede pressure. Split a conceded pressure into
       hit/hurry/sack such that ``pressure_allowed == max(hit, hurry, sack)`` always.
    4. Build ``win_score`` as a monotone function of the latent plus a feature blend
       with mild noise (higher = blocker won); ``win_flag = win_score >= threshold``
       with the threshold recentred to 0 so ``win_flag`` closely mirrors the latent.

    Golden row
    ----------
    One deterministic row is always injected regardless of seed:
    ``game_id=2021090900, play_id=97, blocker_id=42377 (Tristan Wirfs),
    rusher_id=42403, sack_allowed=1, pressure_allowed=1, win_flag==0`` — the real
    PFF-confirmed blocker loss used as the project's golden test.

    Parameters
    ----------
    n_reps:
        Number of synthetic reps to generate (before adding the golden row).
    seed:
        Seed for ``numpy.random.default_rng`` for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A ``scored``-schema frame with ``n_reps + 1`` rows (the extra row is the
        golden row), columns ordered per :data:`SCORED_COLS`.
    """
    rng = np.random.default_rng(seed)
    n = int(n_reps)

    # 1. Latent outcome.
    blocker_won = rng.random(n) < 0.55  # bool array, ~55% wins
    won = blocker_won.astype(float)
    # Signed latent strength: positive => blocker in control.
    latent = (won - 0.5) * 2.0 + rng.normal(0.0, 0.6, n)  # roughly centred, correlated

    def _feat(base_win: float, base_loss: float, scale: float) -> np.ndarray:
        """Feature with a win mean, a loss mean, and Gaussian noise."""
        mean = np.where(blocker_won, base_win, base_loss)
        return mean + rng.normal(0.0, scale, n)

    # 2. Features. "Good-for-blocker" larger when won; "bad-for-blocker" larger when lost.
    min_rusher_to_qb_dist = np.clip(_feat(4.5, 1.8, 0.8), 0.0, None)
    sep_mean = np.clip(_feat(1.6, 0.6, 0.3), 0.0, None)
    sep_min = np.clip(_feat(0.9, 0.25, 0.2), 0.0, None)
    betweenness_mean = np.clip(_feat(0.8, 0.35, 0.12), 0.0, 1.0)
    betweenness_end = np.clip(_feat(0.78, 0.3, 0.14), 0.0, 1.0)
    mirroring_mean = np.clip(_feat(0.7, 0.2, 0.15), -1.0, 1.0)

    ground_given_up = np.clip(_feat(1.2, 4.0, 0.9), 0.0, None)
    penetration_past_los = np.clip(_feat(0.4, 2.5, 0.7), 0.0, None)
    rusher_speed_late = np.clip(_feat(1.8, 4.2, 0.7), 0.0, None)
    rusher_accel_late = np.clip(_feat(1.2, 3.0, 0.6), 0.0, None)

    rep_duration_sec = np.clip(rng.normal(2.6, 0.5, n), 1.0, None)

    # 3. pressure_allowed via logistic on the latent (loss => more pressure).
    pressure_logit = -1.6 * latent - 0.4 + rng.normal(0.0, 0.5, n)
    p_pressure = 1.0 / (1.0 + np.exp(-pressure_logit))
    pressure_allowed = (rng.random(n) < p_pressure).astype(int)

    # Split a conceded pressure into hit/hurry/sack with pressure = max(...).
    sack_allowed = np.zeros(n, dtype=int)
    hurry_allowed = np.zeros(n, dtype=int)
    hit_allowed = np.zeros(n, dtype=int)
    conceded = np.flatnonzero(pressure_allowed == 1)
    if conceded.size:
        # Each conceded pressure is a sack (10%), a hit (25%), else a hurry.
        kind = rng.choice(["sack", "hit", "hurry"], size=conceded.size, p=[0.10, 0.25, 0.65])
        sack_allowed[conceded[kind == "sack"]] = 1
        hit_allowed[conceded[kind == "hit"]] = 1
        hurry_allowed[conceded[kind == "hurry"]] = 1
    # Guarantee the invariant pressure_allowed == max(hit, hurry, sack).
    pressure_allowed = np.maximum.reduce([hit_allowed, hurry_allowed, sack_allowed])

    # 4. win_score: monotone in latent + a standardized feature blend.
    feat_blend = (
        0.4 * (min_rusher_to_qb_dist - min_rusher_to_qb_dist.mean())
        + 0.3 * (sep_mean - sep_mean.mean())
        - 0.4 * (ground_given_up - ground_given_up.mean())
        + 0.3 * (betweenness_mean - betweenness_mean.mean())
    )
    win_score = 1.2 * latent + 0.5 * feat_blend + rng.normal(0.0, 0.3, n)
    win_score = win_score - np.median(win_score)  # recentre so threshold 0 ~ median
    win_flag = (win_score >= WIN_SCORE_THRESHOLD).astype(int)

    # Identifier / categorical fields. The real golden ids (blocker 42377,
    # rusher 42403) are deliberately excluded from the random pools so that the
    # only row matching the golden filter is the deterministic injected row.
    blocker_pool = np.array([44801, 46100, 47800, 52001, 53900, 48200, 41900])
    rusher_pool = np.array([44990, 46700, 51200, 53100, 49800, 47300])
    blocker_id = rng.choice(blocker_pool, size=n)
    rusher_id = rng.choice(rusher_pool, size=n)
    blocker_pos = rng.choice(["LT", "LG", "C", "RG", "RT", "TE", "RB"], size=n)
    rusher_pos = rng.choice(["LE", "RE", "DT", "ROLB", "LOLB"], size=n)
    block_type = rng.choice(["PP", "PA", "SW", "CH", "BH"], size=n, p=[0.6, 0.2, 0.08, 0.08, 0.04])
    team = rng.choice(["TB", "DAL", "KC", "BUF", "SF", "PHI"], size=n)

    is_switch = (block_type == "SW")
    is_chip = (block_type == "CH")
    is_double = rng.random(n) < 0.06
    pairing_ok = rng.random(n) >= 0.03  # ~97% paired cleanly

    snap_frame = np.ones(n, dtype=int)
    end_frame = snap_frame + np.rint(rep_duration_sec * HZ).astype(int)

    game_id = rng.integers(2021_09_09_00, 2021_09_09_00 + 50, size=n)
    play_id = rng.integers(50, 4000, size=n)

    df = pd.DataFrame(
        {
            "game_id": game_id,
            "play_id": play_id,
            "blocker_id": blocker_id,
            "blocker_name": [f"Blocker{b}" for b in blocker_id],
            "blocker_pos": blocker_pos,
            "rusher_id": rusher_id,
            "rusher_name": [f"Rusher{r}" for r in rusher_id],
            "rusher_pos": rusher_pos,
            "qb_id": 30000,
            "snap_frame": snap_frame,
            "end_frame": end_frame,
            "block_type": block_type,
            "is_switch": is_switch,
            "is_chip": is_chip,
            "is_double": is_double,
            "pressure_allowed": pressure_allowed,
            "sack_allowed": sack_allowed,
            "hurry_allowed": hurry_allowed,
            "hit_allowed": hit_allowed,
            "pairing_ok": pairing_ok,
            "team": team,
            "ground_given_up": ground_given_up,
            "min_rusher_to_qb_dist": min_rusher_to_qb_dist,
            "penetration_past_los": penetration_past_los,
            "sep_mean": sep_mean,
            "sep_min": sep_min,
            "betweenness_mean": betweenness_mean,
            "betweenness_end": betweenness_end,
            "mirroring_mean": mirroring_mean,
            "rusher_speed_late": rusher_speed_late,
            "rusher_accel_late": rusher_accel_late,
            "rep_duration_sec": rep_duration_sec,
            "win_score": win_score,
            "win_flag": win_flag,
        }
    )

    # --- GOLDEN ROW: deterministic, overrides any sampled values. --------------
    golden = {
        "game_id": 2021090900,
        "play_id": 97,
        "blocker_id": 42377,
        "blocker_name": "Tristan Wirfs",
        "blocker_pos": "LT",
        "rusher_id": 42403,
        "rusher_name": "Rusher42403",
        "rusher_pos": "ROLB",
        "qb_id": 30000,
        "snap_frame": 1,
        "end_frame": 1 + int(round(2.3 * HZ)),
        "block_type": "PP",
        "is_switch": False,
        "is_chip": False,
        "is_double": False,
        "pressure_allowed": 1,
        "sack_allowed": 1,
        "hurry_allowed": 1,
        "hit_allowed": 0,
        "pairing_ok": True,
        "team": "TB",
        # Clear-loss feature values (blocker gave up a lot of ground, little separation).
        "ground_given_up": 5.0,
        "min_rusher_to_qb_dist": 0.8,
        "penetration_past_los": 3.2,
        "sep_mean": 0.4,
        "sep_min": 0.1,
        "betweenness_mean": 0.2,
        "betweenness_end": 0.1,
        "mirroring_mean": 0.05,
        "rusher_speed_late": 4.8,
        "rusher_accel_late": 3.4,
        "rep_duration_sec": 2.3,
        "win_score": -2.5,          # well below threshold => loss
        "win_flag": 0,
    }
    df = pd.concat([df, pd.DataFrame([golden])], ignore_index=True)

    # Column order per the frozen schema (extra cols, if any, would trail).
    df = df[SCORED_COLS + [c for c in df.columns if c not in SCORED_COLS]]
    return df


if __name__ == "__main__":
    _demo = _make_synthetic_scored()
    print("config demo")
    print("  FIELD_LEN / FIELD_WID / HZ:", FIELD_LEN, FIELD_WID, HZ)
    print("  DATA_DIR :", DATA_DIR)
    print("  CACHE_DIR:", CACHE_DIR, "exists?", CACHE_DIR.exists())
    print("  synthetic scored shape:", _demo.shape)
    print("  schema columns present:", set(SCORED_COLS).issubset(_demo.columns))
    _g = _demo[(_demo.blocker_id == 42377) & (_demo.rusher_id == 42403)]
    print("  golden row win_flag:", int(_g.win_flag.iloc[0]), "(expect 0)")
