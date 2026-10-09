"""Task 2 (Person A) — rep windows + matchup pairing (pass-block head).

Turns a play into one row per blocker->rusher matchup, with the rep window,
stunt flags, and the labeled PFF outcome. Unmatched reps are KEPT with
pairing_ok=False, never dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import io_load


def get_rep_window(play_frames: pd.DataFrame) -> tuple[int, int]:
    """Return (snap_frame_id, end_frame_id) for a play.

    snap  = first frame tagged ball_snap (fallback: min frame_id)
    end   = first frame at/after snap tagged with any REP_END_EVENTS
            (fallback: max frame_id)
    """
    events = play_frames[["frame_id", "event"]].dropna(subset=["event"])
    events = events[events["event"] != "None"]

    snap = events[events["event"].isin(C.SNAP_EVENTS)]["frame_id"]
    snap_frame = int(snap.min()) if len(snap) else int(play_frames["frame_id"].min())

    end_candidates = events[
        events["event"].isin(C.REP_END_EVENTS) & (events["frame_id"] >= snap_frame)
    ]["frame_id"]
    end_frame = (
        int(end_candidates.min())
        if len(end_candidates)
        else int(play_frames["frame_id"].max())
    )
    if end_frame <= snap_frame:
        end_frame = int(play_frames["frame_id"].max())
    return snap_frame, end_frame


def get_matchups(game_id: int, play_id: int) -> pd.DataFrame:
    """One row per blocker->rusher matchup on the play. SCHEMA: matchups."""
    play = io_load.get_play(game_id, play_id)
    pff = io_load.load_pff()
    pff = pff[(pff.game_id == game_id) & (pff.play_id == play_id)].copy()
    players = io_load.load_players()[["nfl_id", "display_name", "position"]]

    snap_frame, end_frame = get_rep_window(play)

    # QB reference (the passer). If multiple, take the one tagged Pass.
    qb_rows = pff[pff.pff_role == C.ROLE_PASS]["nfl_id"].tolist()
    qb_id = int(qb_rows[0]) if qb_rows else C.BALL_ID

    name = players.set_index("nfl_id")["display_name"].to_dict()
    pos = players.set_index("nfl_id")["position"].to_dict()

    blockers = pff[pff.pff_role == C.ROLE_BLOCK]
    rushers = set(pff[pff.pff_role == C.ROLE_RUSH]["nfl_id"].tolist())

    # Count how many blockers were assigned to each rusher (double-team detection).
    assign_counts = (
        blockers["pff_blocked_player_id"].dropna().astype(int).value_counts().to_dict()
    )

    rows = []
    for _, b in blockers.iterrows():
        bid = int(b["nfl_id"])
        assigned = b["pff_blocked_player_id"]
        has_assign = pd.notna(assigned)
        rid = int(assigned) if has_assign else C.BALL_ID
        block_type = b.get("pff_block_type")
        block_type = None if pd.isna(block_type) else block_type

        pairing_ok = bool(has_assign and rid in rushers)
        rows.append(
            {
                "game_id": game_id,
                "play_id": play_id,
                "blocker_id": bid,
                "blocker_name": name.get(bid),
                "blocker_pos": pos.get(bid),
                "rusher_id": rid,
                "rusher_name": name.get(rid),
                "rusher_pos": pos.get(rid),
                "qb_id": qb_id,
                "snap_frame": snap_frame,
                "end_frame": end_frame,
                "block_type": block_type,
                "is_switch": block_type == C.BLOCKTYPE_SWITCH,
                "is_chip": block_type == C.BLOCKTYPE_CHIP,
                "is_double": bool(has_assign and assign_counts.get(rid, 0) > 1),
                "sack_allowed": _to01(b.get("pff_sack_allowed")),
                "hurry_allowed": _to01(b.get("pff_hurry_allowed")),
                "hit_allowed": _to01(b.get("pff_hit_allowed")),
                "pairing_ok": pairing_ok,
            }
        )

    df = pd.DataFrame(rows)
    if len(df):
        df["pressure_allowed"] = df[
            ["hit_allowed", "hurry_allowed", "sack_allowed"]
        ].max(axis=1)
    return df


def get_all_matchups(game_ids: list[int], use_cache: bool = True) -> pd.DataFrame:
    """get_matchups over many games, concatenated. Cached to parquet."""
    stem = f"matchups_{len(game_ids)}games"
    if use_cache:
        cached = C.cache_read(stem)
        if cached is not None:
            return cached

    plays = io_load.load_plays()
    frames = []
    for gid in game_ids:
        gplays = plays[plays.game_id == gid]["play_id"].unique()
        for pid in gplays:
            try:
                frames.append(get_matchups(int(gid), int(pid)))
            except Exception as e:  # keep going; log which play failed
                print(f"skip {gid}/{pid}: {e}")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if use_cache and len(out):
        C.cache_write(out, stem)
    return out


def _to01(v) -> int:
    """PFF labels are 0/1/NA. NA (role not applicable) -> 0."""
    if pd.isna(v):
        return 0
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


# ---------------------------------------------------------------------------
# Smoke test — golden rep must appear with sack_allowed=1
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = get_matchups(C.GOLDEN_GAME, C.GOLDEN_PLAY)
    cols = [
        "blocker_id", "blocker_pos", "rusher_id", "rusher_pos",
        "block_type", "pressure_allowed", "sack_allowed", "pairing_ok",
    ]
    golden = m[(m.blocker_id == C.GOLDEN_BLOCKER)]
    lines = [
        f"matchups on golden play: {len(m)}",
        f"pairing_ok rate: {m.pairing_ok.mean():.2f}",
        f"qb_id: {m.qb_id.iloc[0]}  snap/end: {m.snap_frame.iloc[0]}/{m.end_frame.iloc[0]}",
        "golden blocker row:",
        golden[cols].to_string(index=False),
        "",
        "all matchups:",
        m[cols].to_string(index=False),
    ]
    msg = "\n".join(lines)
    print(msg)
    (C.CACHE_DIR / "_smoke_pairing.txt").write_text(msg, encoding="utf-8")
