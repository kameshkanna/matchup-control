"""Person C's adapter onto Person A's upstream API (io_load / pairing / features).

Person C's modules (``viz``, ``receiver``, ``story``) need a play's frames, its
rep window, its blocker->rusher matchups, and the synced rep trajectories. Those
are Person A's functions and may not be merged yet.

Every public accessor here (:func:`load_play`, :func:`rep_window`,
:func:`matchups`, :func:`rep_tracks`, :func:`control_ts`) first tries Person A's
real ``matchup.*`` module and only falls back to the local, contract-faithful
implementation if that import fails. The fallbacks return the SAME frozen schemas
(``tracking``, ``matchups``, ``rep_tracks``, ``control_ts``), read the raw CSVs
from :data:`matchup.config.DATA_DIR`, and normalise direction exactly once. When
Person A merges, Person C's code uses A's code with zero changes.

Constants that Person C needs but the shared ``config`` does not define yet
(event names, role strings, ball sentinel, golden row) live here so that the
shared ``config`` is never edited from Person C's side.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from matchup.config import DATA_DIR, FIELD_LEN, FIELD_WID, HZ, MATCHUPS_COLS

# --------------------------------------------------------------------------- #
# Constants verified against the raw data (not yet in the shared config).
# --------------------------------------------------------------------------- #
SNAP_EVENTS: tuple[str, ...] = ("ball_snap", "autoevent_ballsnap")
# There is NO 'pass_release' event in the tracking files; 'qb_strip_sack' exists.
END_EVENTS: tuple[str, ...] = ("pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack")

# pff_role values are title-case in the CSV (the README's lower-case is wrong).
ROLE_PASS = "Pass"
ROLE_BLOCK = "Pass Block"
ROLE_RUSH = "Pass Rush"
ROLE_ROUTE = "Pass Route"
ROLE_COVERAGE = "Coverage"

BALL_ID: int = -1

# Golden rep, as the raw PFF data records it: Donovan Smith (T) vs Randy Gregory
# (DE); hurry_allowed=1, sack_allowed=0 -> pressure_allowed=1, expect win_flag 0.
GOLDEN: dict = dict(game_id=2021090900, play_id=97, blocker_id=42377, rusher_id=42403)


# --------------------------------------------------------------------------- #
# Raw loaders (cached; camelCase -> snake_case happens here, once).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _plays() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "plays.csv")


@lru_cache(maxsize=1)
def _players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "players.csv")


@lru_cache(maxsize=1)
def _pff() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "pffScoutingData.csv")


_TRACK_RENAME = {
    "gameId": "game_id",
    "playId": "play_id",
    "nflId": "nfl_id",
    "frameId": "frame_id",
    "jerseyNumber": "jersey_number",
    "playDirection": "play_direction",
}


def _normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip left-moving plays so the offense always moves toward +x."""
    df = df.copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = FIELD_LEN - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_WID - df.loc[left, "y"]
    for col in ("dir", "o"):
        df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0
    return df


@lru_cache(maxsize=4)
def _load_tracking_cached(game_id: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "tracking" / f"tracking_{game_id}.csv")
    df = df.rename(columns=_TRACK_RENAME)
    df["nfl_id"] = df["nfl_id"].fillna(BALL_ID).astype(int)
    return _normalize_direction(df)


def _fb_load_tracking(game_id: int) -> pd.DataFrame:
    return _load_tracking_cached(int(game_id)).copy()


def _fb_get_play(game_id: int, play_id: int) -> pd.DataFrame:
    """Merged tracking + roles + names for ONE play (SCHEMA: tracking)."""
    trk = _fb_load_tracking(game_id)
    trk = trk[trk["play_id"] == play_id].copy()

    pff = _pff()
    pff = pff[(pff["gameId"] == game_id) & (pff["playId"] == play_id)]
    roles = pff[["nflId", "pff_role", "pff_positionLinedUp"]].rename(
        columns={"nflId": "nfl_id", "pff_positionLinedUp": "pff_position_lined_up"}
    )
    names = _players()[["nflId", "displayName", "officialPosition"]].rename(
        columns={"nflId": "nfl_id", "displayName": "display_name", "officialPosition": "position"}
    )
    return trk.merge(roles, on="nfl_id", how="left").merge(names, on="nfl_id", how="left")


def _fb_get_rep_window(play_frames: pd.DataFrame) -> tuple[int, int]:
    ev = play_frames.dropna(subset=["event"])
    snap = ev.loc[ev["event"].isin(SNAP_EVENTS), "frame_id"]
    snap_frame = int(snap.min()) if len(snap) else int(play_frames["frame_id"].min())
    end = ev.loc[ev["event"].isin(END_EVENTS) & (ev["frame_id"] >= snap_frame), "frame_id"]
    end_frame = int(end.min()) if len(end) else int(play_frames["frame_id"].max())
    return snap_frame, end_frame


def _fb_get_matchups(game_id: int, play_id: int) -> pd.DataFrame:
    """Blocker->rusher matchups from PFF assignment (SCHEMA: matchups, incl. team)."""
    play = _fb_get_play(game_id, play_id)
    snap_frame, end_frame = _fb_get_rep_window(play)

    pff = _pff()
    pff = pff[(pff["gameId"] == game_id) & (pff["playId"] == play_id)]
    qb = pff.loc[pff["pff_role"] == ROLE_PASS, "nflId"]
    qb_id = int(qb.iloc[0]) if len(qb) else BALL_ID

    plays = _plays()
    prow = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)]
    team = str(prow["possessionTeam"].iloc[0]) if len(prow) else None  # blockers are on offense

    names = _players().set_index("nflId")

    def _nm(pid, col):
        return names.at[pid, col] if pid in names.index else None

    rows = []
    for b in pff[pff["pff_role"] == ROLE_BLOCK].itertuples(index=False):
        bid = int(b.nflId)
        paired = pd.notna(b.pff_nflIdBlockedPlayer)
        rid = int(b.pff_nflIdBlockedPlayer) if paired else BALL_ID
        hit, hur, sck = (int(v == 1) for v in (b.pff_hitAllowed, b.pff_hurryAllowed, b.pff_sackAllowed))
        rows.append(
            dict(
                game_id=int(game_id), play_id=int(play_id),
                blocker_id=bid, blocker_name=_nm(bid, "displayName"), blocker_pos=_nm(bid, "officialPosition"),
                rusher_id=rid, rusher_name=_nm(rid, "displayName"), rusher_pos=_nm(rid, "officialPosition"),
                qb_id=qb_id, snap_frame=snap_frame, end_frame=end_frame,
                block_type=b.pff_blockType,
                is_switch=b.pff_blockType == "SW", is_chip=b.pff_blockType == "CH", is_double=False,
                pressure_allowed=max(hit, hur, sck),
                sack_allowed=sck, hurry_allowed=hur, hit_allowed=hit,
                pairing_ok=bool(paired), team=team,
            )
        )
    return pd.DataFrame(rows, columns=MATCHUPS_COLS)


def _fb_get_rep_tracks(game_id: int, play_id: int, blocker_id: int, rusher_id: int) -> pd.DataFrame:
    """Frame-aligned blocker/rusher/qb trajectories over the rep (SCHEMA: rep_tracks)."""
    play = _fb_get_play(game_id, play_id)
    snap_frame, end_frame = _fb_get_rep_window(play)
    qb = play.loc[play["pff_role"] == ROLE_PASS, "nfl_id"]
    qb_id = int(qb.iloc[0]) if len(qb) else BALL_ID

    win = play[(play["frame_id"] >= snap_frame) & (play["frame_id"] <= end_frame)]

    def _seat(pid, cols, prefix):
        return win[win["nfl_id"] == pid].set_index("frame_id")[cols].add_prefix(prefix)

    out = (
        _seat(blocker_id, ["x", "y", "s", "dir"], "blk_")
        .join(_seat(rusher_id, ["x", "y", "s", "a", "dir"], "rsh_"), how="outer")
        .join(_seat(qb_id, ["x", "y"], "qb_"), how="outer")
        .reset_index()
    )
    out["t_sec"] = (out["frame_id"] - snap_frame) / HZ
    cols = ["frame_id", "t_sec", "blk_x", "blk_y", "blk_s", "blk_dir",
            "rsh_x", "rsh_y", "rsh_s", "rsh_a", "rsh_dir", "qb_x", "qb_y"]
    return out.reindex(columns=cols)


def _fb_control_timeseries(rt: pd.DataFrame) -> pd.DataFrame:
    """Per-frame control signals (SCHEMA: control_ts). Higher = blocker in control."""
    rusher_to_qb = np.hypot(rt["rsh_x"] - rt["qb_x"], rt["rsh_y"] - rt["qb_y"])
    sep = np.hypot(rt["blk_x"] - rt["rsh_x"], rt["blk_y"] - rt["rsh_y"])

    # Betweenness: is the blocker on the rusher->QB segment, and how close to it?
    sx, sy = rt["qb_x"] - rt["rsh_x"], rt["qb_y"] - rt["rsh_y"]
    bx, by = rt["blk_x"] - rt["rsh_x"], rt["blk_y"] - rt["rsh_y"]
    seg2 = sx**2 + sy**2
    safe = np.where(seg2 > 0, seg2, 1.0)
    proj = (bx * sx + by * sy) / safe
    perp = np.abs(bx * sy - by * sx) / np.sqrt(safe)
    betweenness = np.where((proj >= 0) & (proj <= 1), 1.0 / (1.0 + perp), 0.0)

    mirroring = np.cos(np.deg2rad(rt["blk_dir"]) - np.deg2rad(rt["rsh_dir"]))
    return pd.DataFrame(
        {
            "frame_id": rt["frame_id"],
            "t_sec": rt["t_sec"],
            "rusher_to_qb_dist": rusher_to_qb,
            "blocker_rusher_sep": sep,
            "betweenness": betweenness,
            "mirroring": mirroring,
        }
    )


# --------------------------------------------------------------------------- #
# Public accessors: Person A's real module first, local fallback second.
# --------------------------------------------------------------------------- #
def load_tracking(game_id: int) -> pd.DataFrame:
    try:
        from matchup.io_load import load_tracking as real
    except ImportError:
        return _fb_load_tracking(game_id)
    return real(game_id)


def load_play(game_id: int, play_id: int) -> pd.DataFrame:
    try:
        from matchup.io_load import get_play as real
    except ImportError:
        return _fb_get_play(game_id, play_id)
    return real(game_id, play_id)


def rep_window(play_frames: pd.DataFrame) -> tuple[int, int]:
    try:
        from matchup.pairing import get_rep_window as real
    except ImportError:
        return _fb_get_rep_window(play_frames)
    return real(play_frames)


def matchups(game_id: int, play_id: int) -> pd.DataFrame:
    try:
        from matchup.pairing import get_matchups as real
    except ImportError:
        return _fb_get_matchups(game_id, play_id)
    return real(game_id, play_id)


def rep_tracks(game_id: int, play_id: int, blocker_id: int, rusher_id: int) -> pd.DataFrame:
    try:
        from matchup.features import get_rep_tracks as real
    except ImportError:
        return _fb_get_rep_tracks(game_id, play_id, blocker_id, rusher_id)
    return real(game_id, play_id, blocker_id, rusher_id)


def control_ts(rep_tracks_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from matchup.features import control_timeseries as real
    except ImportError:
        return _fb_control_timeseries(rep_tracks_df)
    return real(rep_tracks_df)


def plays() -> pd.DataFrame:
    """Raw plays table (camelCase), used for LOS / coverage / playDescription lookups.

    Deliberately reads the CSV rather than ``io_load.load_plays`` so the column
    names Person C relies on stay stable regardless of A's renaming.
    """
    return _plays()


def upstream_source() -> str:
    """'person_a' if Person A's modules are importable, else 'fallback'."""
    try:
        import matchup.features  # noqa: F401
        import matchup.io_load  # noqa: F401
        import matchup.pairing  # noqa: F401
    except ImportError:
        return "fallback"
    return "person_a"
