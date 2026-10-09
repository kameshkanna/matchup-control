"""Task 7 (Person A) — stunt / chip / double-team handling.

PFF's pff_nflIdBlockedPlayer is the FIRST defender a blocker engaged. On stunts
and twists the matchup crosses: the rusher a blocker started on ends up handled
by a teammate. We flag and segment these so a lineman who cleanly passed off a
looper isn't charged for the rusher who then beat a different blocker.

Pragmatic, not a full stunt model: we detect pass-offs geometrically (did the
assigned rusher get closer to a DIFFERENT blocker than to the assigned one by
the end of the rep) and annotate attribution rather than silently reassigning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import io_load
from . import pairing


def tag_stunts(matchups: pd.DataFrame, play_frames: pd.DataFrame | None = None) -> pd.DataFrame:
    """Adds/updates is_switch, is_chip, is_double, passed_off (bool).

    is_switch / is_chip / is_double already come from pairing.get_matchups;
    we (re)assert them and add passed_off, computed geometrically per play.
    """
    df = matchups.copy()
    for col, default in (("is_switch", False), ("is_chip", False),
                         ("is_double", False), ("passed_off", False)):
        if col not in df.columns:
            df[col] = default

    # passed_off needs trajectories; compute per play.
    passed = []
    for (gid, pid), grp in df.groupby(["game_id", "play_id"]):
        pf = play_frames
        if pf is None:
            try:
                pf = io_load.get_play(int(gid), int(pid))
            except Exception:
                pf = None
        snap = int(grp.snap_frame.iloc[0])
        end = int(grp.end_frame.iloc[0])
        blocker_ids = grp.blocker_id.astype(int).tolist()
        for _, r in grp.iterrows():
            flag = False
            if pf is not None and bool(r.get("pairing_ok", False)):
                flag = _rusher_passed_off(
                    pf, int(r.blocker_id), int(r.rusher_id), blocker_ids, snap, end
                )
            passed.append(flag)
    df["passed_off"] = passed
    return df


def _rusher_passed_off(play_frames, blocker_id, rusher_id, all_blocker_ids,
                       snap, end) -> bool:
    """True if, by the end of the rep, the assigned rusher is markedly closer to
    a DIFFERENT blocker than to his assigned blocker (a twist/pass-off)."""
    last = play_frames[play_frames.frame_id == end]
    rush = last[last.nfl_id == rusher_id]
    if rush.empty:
        return False
    rx, ry = float(rush.x.iloc[0]), float(rush.y.iloc[0])

    def dist(bid):
        b = last[last.nfl_id == bid]
        if b.empty:
            return np.inf
        return float(np.hypot(b.x.iloc[0] - rx, b.y.iloc[0] - ry))

    own = dist(blocker_id)
    others = [dist(b) for b in all_blocker_ids if b != blocker_id]
    if not others:
        return False
    nearest_other = min(others)
    # Passed off if a teammate is clearly the one on this rusher now.
    return (nearest_other + 1.5) < own and nearest_other < 2.0


def resolve_attribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Reassign/segment credit on crossed reps. Adds attribution_note (str).

    Conservative policy for a hackathon:
      - If a blocker passed_off his assigned rusher cleanly (switch/twist) and
        allowed no pressure, we DAMP the penalty: his win_score is floored so a
        geometric 'ground given up' from the looper leaving doesn't sink him.
      - Double-team members share: neither is solely charged.
    The raw win_score is preserved in win_score_raw; win_score becomes the
    attribution-adjusted value.
    """
    df = scored.copy()
    if "win_score" not in df.columns:
        return df
    df["win_score_raw"] = df["win_score"]
    notes = []
    for _, r in df.iterrows():
        note = ""
        s = r.get("win_score")
        if bool(r.get("passed_off", False)):
            note = "passed_off: looper handed to teammate"
            # Don't penalise a clean pass-off that allowed no pressure.
            if int(r.get("pressure_allowed", 0)) == 0 and pd.notna(s):
                df.loc[r.name, "win_score"] = max(float(s), 0.0)
        elif bool(r.get("is_double", False)):
            note = "double_team: shared responsibility"
        elif bool(r.get("is_switch", False)):
            note = "switch block"
        elif bool(r.get("is_chip", False)):
            note = "chip then released"
        notes.append(note)
    df["attribution_note"] = notes
    df["win_flag"] = df["win_score"].apply(
        lambda x: 0 if pd.isna(x) else int(x >= 0.0)
    )
    return df


# ---------------------------------------------------------------------------
# Smoke test — golden play has two SW (switch) reps on a stunt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from . import score

    m = pairing.get_matchups(C.GOLDEN_GAME, C.GOLDEN_PLAY)
    m = tag_stunts(m)
    scored = score.build_scored(m)
    scored = resolve_attribution(scored)
    cols = [
        "blocker_id", "blocker_pos", "rusher_id", "block_type", "is_double",
        "is_switch", "passed_off", "pressure_allowed", "win_score_raw",
        "win_score", "win_flag", "attribution_note",
    ]
    lines = [
        "golden play with stunt attribution:",
        scored[cols].to_string(index=False),
    ]
    msg = "\n".join(lines)
    print(msg)
    (C.CACHE_DIR / "_smoke_stunts.txt").write_text(msg, encoding="utf-8")
