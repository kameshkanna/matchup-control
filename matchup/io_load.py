"""Task 1 (Person A) — loaders + direction normalisation.

The ONE place camelCase CSV columns become snake_case, and the ONE place
play direction is normalised (offense always moves toward +x). Downstream
code assumes this has happened and never re-flips.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------------------
# Column renames: raw camelCase -> canonical snake_case
# ---------------------------------------------------------------------------
_GAMES_REN = {
    "gameId": "game_id", "homeTeamAbbr": "home_team",
    "visitorTeamAbbr": "visitor_team", "gameDate": "game_date",
    "gameTimeEastern": "game_time",
}
_PLAYS_REN = {
    "gameId": "game_id", "playId": "play_id", "playDescription": "play_description",
    "possessionTeam": "possession_team", "defensiveTeam": "defensive_team",
    "yardlineSide": "yardline_side", "yardlineNumber": "yardline_number",
    "passResult": "pass_result", "playResult": "play_result",
    "absoluteYardlineNumber": "absolute_yardline",
    "offenseFormation": "offense_formation", "personnelO": "personnel_o",
    "personnelD": "personnel_d", "defendersInBox": "defenders_in_box",
    "dropBackType": "dropback_type", "pff_playAction": "pff_play_action",
    "pff_passCoverage": "pff_pass_coverage",
    "pff_passCoverageType": "pff_pass_coverage_type",
    "yardsToGo": "yards_to_go",
}
_PLAYERS_REN = {
    "nflId": "nfl_id", "collegeName": "college_name",
    "officialPosition": "position", "displayName": "display_name",
    "birthDate": "birth_date",
}
_PFF_REN = {
    "gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
    "pff_positionLinedUp": "pff_position_lined_up",
    "pff_nflIdBlockedPlayer": "pff_blocked_player_id",
    "pff_blockType": "pff_block_type",
    "pff_hitAllowed": "pff_hit_allowed", "pff_hurryAllowed": "pff_hurry_allowed",
    "pff_sackAllowed": "pff_sack_allowed",
    "pff_beatenByDefender": "pff_beaten_by_defender",
    "pff_backFieldBlock": "pff_backfield_block",
}
_TRACK_REN = {
    "gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
    "frameId": "frame_id", "jerseyNumber": "jersey_number",
    "playDirection": "play_direction",
}


# ---------------------------------------------------------------------------
# Static tables (memoised — loaded once per process)
#
# These return a SHARED cached frame for speed. Treat as read-only; call
# .copy() if you need to mutate. Everything in this codebase treats them as
# read-only, which is why memoisation is safe and makes scaling tractable.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_games() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / "games.csv")
    return df.rename(columns=_GAMES_REN)


@lru_cache(maxsize=1)
def load_plays() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / "plays.csv")
    return df.rename(columns=_PLAYS_REN)


@lru_cache(maxsize=1)
def load_players() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / "players.csv")
    return df.rename(columns=_PLAYERS_REN)


@lru_cache(maxsize=1)
def load_pff() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_DIR / "pffScoutingData.csv")
    df = df.rename(columns=_PFF_REN)
    # blocked player id -> nullable Int so we can keep NA but compare cleanly
    df["pff_blocked_player_id"] = df["pff_blocked_player_id"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Direction normalisation
# ---------------------------------------------------------------------------
def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates so offense ALWAYS moves toward +x.

    For rows where play_direction == 'left':
      x -> FIELD_LEN - x
      y -> FIELD_WID - y   (keep the field a mirror image, not just x)
      dir, o -> (angle + 180) mod 360   (angles are 0=+y clockwise in this data,
                 but a 180-degree rotation is correct for a point reflection)

    Adds no rows. Idempotent is NOT guaranteed — call exactly once.
    """
    df = df.copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = C.FIELD_LEN - df.loc[left, "x"]
    df.loc[left, "y"] = C.FIELD_WID - df.loc[left, "y"]
    for col in ("dir", "o"):
        if col in df.columns:
            df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0
    return df


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def load_tracking(game_id: int) -> pd.DataFrame:
    """One game's tracking: snake_case, ball nfl_id = -1, direction-normalised.

    Memoised (last 4 games) so the per-play loop in build_scored doesn't
    re-read and re-normalise a ~30MB CSV hundreds of times. Treat as read-only.
    """
    path = C.TRACKING_DIR / f"tracking_{game_id}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns=_TRACK_REN)
    # Ball rows have NA nflId -> sentinel BALL_ID, then make column plain int.
    df["nfl_id"] = df["nfl_id"].fillna(C.BALL_ID).astype(int)
    df = normalize_direction(df)
    return df


@lru_cache(maxsize=4)
def _game_play_table(game_id: int) -> pd.DataFrame:
    """Whole game: tracking with player + role columns already merged in.

    Built once per game and sliced per play by get_play. This is the single
    biggest scaling win — role joins happen once per game, not once per play.
    """
    trk = load_tracking(game_id)

    players = load_players()[["nfl_id", "display_name", "position"]]
    trk = trk.merge(players, on="nfl_id", how="left")

    pff = load_pff()
    pff_game = pff[pff["game_id"] == game_id][
        ["play_id", "nfl_id", "pff_role", "pff_position_lined_up"]
    ]
    trk = trk.merge(pff_game, on=["play_id", "nfl_id"], how="left")
    return trk.sort_values(["play_id", "nfl_id", "frame_id"]).reset_index(drop=True)


def get_play(game_id: int, play_id: int) -> pd.DataFrame:
    """Merged tracking + player roles for ONE play, direction-normalised.

    THE entry point everyone uses to get a play's frames. Joins in
    display_name, position (players.csv) and pff_role, pff_position_lined_up
    (pffScoutingData.csv). Ball row keeps NaN role. Backed by a per-game cache.
    """
    game = _game_play_table(game_id)
    play = game[game["play_id"] == play_id]
    return play.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out = []
    play = get_play(C.GOLDEN_GAME, C.GOLDEN_PLAY)
    n_players = play["nfl_id"].nunique()
    out.append(f"players+ball on golden play: {n_players}")
    out.append(f"has ball row (nfl_id=-1): {(play['nfl_id'] == C.BALL_ID).any()}")

    # Direction check: offense (possession team) should move toward +x on average.
    plays = load_plays()
    row = plays[(plays.game_id == C.GOLDEN_GAME) & (plays.play_id == C.GOLDEN_PLAY)].iloc[0]
    poss = row["possession_team"]
    off = play[play["team"] == poss]
    # mean x at last frame minus first frame, per player, averaged
    first = off.groupby("nfl_id")["x"].first()
    last = off.groupby("nfl_id")["x"].last()
    mean_dx = float((last - first).mean())
    raw = load_tracking(C.GOLDEN_GAME)
    pdir = raw[raw.play_id == C.GOLDEN_PLAY]["play_direction"].iloc[0]
    out.append(f"play_direction raw: {pdir}; mean offense dx after normalise: {mean_dx:.2f}")
    out.append(f"roles present: {sorted(play['pff_role'].dropna().unique())}")

    msg = "\n".join(out)
    print(msg)
    (C.CACHE_DIR / "_smoke_io_load.txt").write_text(msg, encoding="utf-8")
