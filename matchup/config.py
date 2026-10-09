"""Shared configuration, frozen schema, cache helpers, and synthetic generator.

This is the ONE module every other module imports from. It owns:

* Field/sampling constants (:data:`FIELD_LEN`, :data:`FIELD_WID`, :data:`HZ`).
* Resolved data/cache directories, derived from ``__file__`` (never hardcoded).
* Event tags, PFF role labels, and the golden-rep constants (Person A engine).
* The frozen ``scored`` schema (:data:`MATCHUPS_COLS`, :data:`FEATURES_KEYS`,
  :data:`SCORED_COLS`) downstream analytics code strictly against (Person B).
* parquet-or-CSV cache helpers so a missing parquet engine never blocks anyone.
* :func:`_make_synthetic_scored`, so Person B modules run in isolation before the
  real upstream pipeline is wired in.

Canonical identifier column names (snake_case, used throughout the package):
``game_id``, ``play_id``, ``nfl_id`` (ball = sentinel ``-1``, never NaN),
``frame_id`` (1-indexed, 10 Hz). The raw CSVs use camelCase; the rename is
``io_load``'s responsibility (Person A). Person B codes against the snake_case
frozen schema and the synthetic generator, which emits snake_case directly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths — resolved relative to this file (no absolute paths committed).
# This file lives at <workspace>/matchup/config.py, so the workspace root is
# two parents up. The dataset lives in nfl-big-data-bowl-regional-event-data/.
# --------------------------------------------------------------------------- #
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "nfl-big-data-bowl-regional-event-data" / "data"
TRACKING_DIR = DATA_DIR / "tracking"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Field / tracking constants (NFL Big Data Bowl 2023 regional set, 10 Hz).
# --------------------------------------------------------------------------- #
FIELD_LEN: float = 120.0   # x axis, yards (includes both 10-yard end zones)
FIELD_WID: float = 53.3    # y axis, yards (sideline to sideline)
HZ: int = 10               # tracking frequency (frames per second)

BALL_ID = -1               # ball rows (raw nflId is NA) get this sentinel

# --------------------------------------------------------------------------- #
# Event tags
# --------------------------------------------------------------------------- #
SNAP_EVENTS = ("ball_snap",)
# Rep ends at the first of these (ball leaves QB hand / play breaks down).
REP_END_EVENTS = (
    "pass_forward",
    "autoevent_passforward",
    "qb_sack",
    "qb_strip_sack",
    "pass_shovel",
    "run",            # scramble handoff point — protection rep effectively ends
    "pass_tipped",
)

# --------------------------------------------------------------------------- #
# PFF role labels
# --------------------------------------------------------------------------- #
ROLE_BLOCK = "Pass Block"
ROLE_RUSH = "Pass Rush"
ROLE_PASS = "Pass"          # the QB / passer
ROLE_ROUTE = "Pass Route"
ROLE_COVERAGE = "Coverage"

# Stunt-relevant block types (pff_blockType)
BLOCKTYPE_SWITCH = "SW"
BLOCKTYPE_CHIP = "CH"

# --------------------------------------------------------------------------- #
# The golden test rep (everyone validates against this)
# --------------------------------------------------------------------------- #
GOLDEN_GAME = 2021090900
GOLDEN_PLAY = 97
GOLDEN_BLOCKER = 42377   # LT
GOLDEN_RUSHER = 42403    # ROLB — pressure allowed on this rep

# --------------------------------------------------------------------------- #
# Frozen schema. Downstream modules must never rename or drop these columns.
# Extra columns are permitted. game_id/play_id/blocker_id/rusher_id are shared
# between the matchups and features groups and appear once in `scored`.
# --------------------------------------------------------------------------- #
MATCHUPS_COLS: list[str] = [
    "game_id", "play_id",
    "blocker_id", "blocker_name", "blocker_pos",
    "rusher_id", "rusher_name", "rusher_pos",
    "qb_id", "snap_frame", "end_frame",
    "block_type", "is_switch", "is_chip", "is_double",
    "pressure_allowed",  # int 0/1 = max(hit_allowed, hurry_allowed, sack_allowed)
    "sack_allowed", "hurry_allowed", "hit_allowed",
    "pairing_ok",        # bool
    "team",              # blocker's team abbreviation
]

# Control-feature keys. Convention: every feature is SIGNED so a larger value
# means the blocker won the rep.
FEATURES_KEYS: list[str] = [
    "game_id", "play_id", "blocker_id", "rusher_id",
    "ground_given_up", "min_rusher_to_qb_dist", "penetration_past_los",
    "sep_mean", "sep_min", "betweenness_mean", "betweenness_end",
    "mirroring_mean", "rusher_speed_late", "rusher_accel_late",
    "rep_duration_sec",
]

_FEATURE_ONLY_KEYS: list[str] = [
    k for k in FEATURES_KEYS
    if k not in ("game_id", "play_id", "blocker_id", "rusher_id")
]
SCORED_COLS: list[str] = MATCHUPS_COLS + _FEATURE_ONLY_KEYS + ["win_score", "win_flag"]

# win_flag == 1 (blocker won) when win_score >= WIN_SCORE_THRESHOLD.
WIN_SCORE_THRESHOLD: float = 0.0


# --------------------------------------------------------------------------- #
# Cache helpers — parquet if pyarrow/fastparquet is available, else CSV, so a
# missing parquet engine never blocks anyone.
# --------------------------------------------------------------------------- #
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


HAS_PARQUET = _has_parquet()


def cache_path(stem: str):
    """Cache file path for a stem, with the best available extension."""
    ext = "parquet" if HAS_PARQUET else "csv"
    return CACHE_DIR / f"{stem}.{ext}"


def cache_write(df, stem: str) -> None:
    p = cache_path(stem)
    if HAS_PARQUET:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def cache_read(stem: str):
    p = cache_path(stem)
    if not p.exists():
        return None
    if HAS_PARQUET:
        return pd.read_parquet(p)
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
# Shared synthetic generator — lets Person B modules run before the real
# upstream pipeline exists. Emits the frozen `scored` schema with realistic,
# self-consistent signal and a deterministic golden row.
# --------------------------------------------------------------------------- #
def _make_synthetic_scored(n_reps: int = 400, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic ``scored`` DataFrame matching :data:`SCORED_COLS`."""
    rng = np.random.default_rng(seed)
    n = int(n_reps)

    blocker_won = rng.random(n) < 0.55
    won = blocker_won.astype(float)
    latent = (won - 0.5) * 2.0 + rng.normal(0.0, 0.6, n)

    def _feat(base_win: float, base_loss: float, scale: float) -> np.ndarray:
        mean = np.where(blocker_won, base_win, base_loss)
        return mean + rng.normal(0.0, scale, n)

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

    pressure_logit = -1.6 * latent - 0.4 + rng.normal(0.0, 0.5, n)
    p_pressure = 1.0 / (1.0 + np.exp(-pressure_logit))
    pressure_allowed = (rng.random(n) < p_pressure).astype(int)

    sack_allowed = np.zeros(n, dtype=int)
    hurry_allowed = np.zeros(n, dtype=int)
    hit_allowed = np.zeros(n, dtype=int)
    conceded = np.flatnonzero(pressure_allowed == 1)
    if conceded.size:
        kind = rng.choice(["sack", "hit", "hurry"], size=conceded.size, p=[0.10, 0.25, 0.65])
        sack_allowed[conceded[kind == "sack"]] = 1
        hit_allowed[conceded[kind == "hit"]] = 1
        hurry_allowed[conceded[kind == "hurry"]] = 1
    pressure_allowed = np.maximum.reduce([hit_allowed, hurry_allowed, sack_allowed])

    feat_blend = (
        0.4 * (min_rusher_to_qb_dist - min_rusher_to_qb_dist.mean())
        + 0.3 * (sep_mean - sep_mean.mean())
        - 0.4 * (ground_given_up - ground_given_up.mean())
        + 0.3 * (betweenness_mean - betweenness_mean.mean())
    )
    win_score = 1.2 * latent + 0.5 * feat_blend + rng.normal(0.0, 0.3, n)
    win_score = win_score - np.median(win_score)
    win_flag = (win_score >= WIN_SCORE_THRESHOLD).astype(int)

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
    pairing_ok = rng.random(n) >= 0.03

    snap_frame = np.ones(n, dtype=int)
    end_frame = snap_frame + np.rint(rep_duration_sec * HZ).astype(int)
    game_id = rng.integers(2021_09_09_00, 2021_09_09_00 + 50, size=n)
    play_id = rng.integers(50, 4000, size=n)

    df = pd.DataFrame(
        {
            "game_id": game_id, "play_id": play_id,
            "blocker_id": blocker_id,
            "blocker_name": [f"Blocker{b}" for b in blocker_id],
            "blocker_pos": blocker_pos,
            "rusher_id": rusher_id,
            "rusher_name": [f"Rusher{r}" for r in rusher_id],
            "rusher_pos": rusher_pos,
            "qb_id": 30000, "snap_frame": snap_frame, "end_frame": end_frame,
            "block_type": block_type, "is_switch": is_switch, "is_chip": is_chip,
            "is_double": is_double, "pressure_allowed": pressure_allowed,
            "sack_allowed": sack_allowed, "hurry_allowed": hurry_allowed,
            "hit_allowed": hit_allowed, "pairing_ok": pairing_ok, "team": team,
            "ground_given_up": ground_given_up,
            "min_rusher_to_qb_dist": min_rusher_to_qb_dist,
            "penetration_past_los": penetration_past_los,
            "sep_mean": sep_mean, "sep_min": sep_min,
            "betweenness_mean": betweenness_mean, "betweenness_end": betweenness_end,
            "mirroring_mean": mirroring_mean,
            "rusher_speed_late": rusher_speed_late,
            "rusher_accel_late": rusher_accel_late,
            "rep_duration_sec": rep_duration_sec,
            "win_score": win_score, "win_flag": win_flag,
        }
    )

    golden = {
        "game_id": 2021090900, "play_id": 97, "blocker_id": 42377,
        "blocker_name": "Tristan Wirfs", "blocker_pos": "LT",
        "rusher_id": 42403, "rusher_name": "Rusher42403", "rusher_pos": "ROLB",
        "qb_id": 30000, "snap_frame": 1, "end_frame": 1 + int(round(2.3 * HZ)),
        "block_type": "PP", "is_switch": False, "is_chip": False, "is_double": False,
        "pressure_allowed": 1, "sack_allowed": 1, "hurry_allowed": 1, "hit_allowed": 0,
        "pairing_ok": True, "team": "TB",
        "ground_given_up": 5.0, "min_rusher_to_qb_dist": 0.8,
        "penetration_past_los": 3.2, "sep_mean": 0.4, "sep_min": 0.1,
        "betweenness_mean": 0.2, "betweenness_end": 0.1, "mirroring_mean": 0.05,
        "rusher_speed_late": 4.8, "rusher_accel_late": 3.4, "rep_duration_sec": 2.3,
        "win_score": -2.5, "win_flag": 0,
    }
    df = pd.concat([df, pd.DataFrame([golden])], ignore_index=True)
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
