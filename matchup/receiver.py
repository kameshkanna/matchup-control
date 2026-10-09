"""Task 11 — bonus receiver head.

The engine does not care whether a pair is blocker-vs-rusher or
receiver-vs-defender, so the receiver head produces the SAME ``matchups`` schema
and reuses ``features`` / ``score`` / ``viz`` unchanged. Here:

* attacker (``blocker_*`` columns) = the receiver (``pff_role == 'Pass Route'``),
* chaser   (``rusher_*`` columns)  = the nearest coverage defender at the snap
  (``pff_role == 'Coverage'``),
* scope    = man coverage (``pff_passCoverageType == 'Man'``); zone rows are kept
  but flagged ``pairing_ok = False`` because nearest-at-snap is a poor proxy there.

There is no PFF ground-truth label for coverage, so ``pressure_allowed`` and the
raw labels are 0; ``is_target`` (parsed from ``playDescription``) marks the
receiver the ball went to, which the story uses instead of a pressure label.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from matchup import _upstream as up
from matchup.config import MATCHUPS_COLS

# "...pass [short|deep] [left|middle|right] to C.Godwin ..." -> capture "C.Godwin".
_TARGET_RE = re.compile(r"\bto\s+([A-Z][.\-A-Za-z']*\.[A-Za-z][\-A-Za-z']*)")


def parse_target(play_description: str) -> str | None:
    """Extract the targeted receiver token (e.g. ``'C.Godwin'``) or ``None``.

    Anchors on the ``pass`` clause so trailing ``to <yardline>`` phrases ("to DAL
    39") are ignored, and only accepts an ``Initial.Lastname`` shaped token.

    Parameters
    ----------
    play_description:
        The raw ``playDescription`` string.

    Returns
    -------
    str | None
    """
    if not isinstance(play_description, str):
        return None
    low = play_description.lower()
    if "pass" not in low:
        return None
    scope = play_description[low.find("pass"):]
    m = _TARGET_RE.search(scope)
    return m.group(1).rstrip(".") if m else None


def _match_name_to_id(target: str | None, play: pd.DataFrame) -> int | None:
    """Resolve ``'C.Godwin'`` to an ``nfl_id`` among this play's route runners."""
    if not target:
        return None
    initial, _, last = target.partition(".")
    routes = play[play["pff_role"] == up.ROLE_ROUTE].dropna(subset=["display_name"])
    last_lower = last.lower()
    for r in routes.itertuples(index=False):
        parts = str(r.display_name).split()
        if len(parts) >= 2 and parts[-1].lower() == last_lower \
                and parts[0][:1].lower() == initial[:1].lower():
            return int(r.nfl_id)
    # Surname-only fallback (handles middle names / Jr. suffixes).
    for r in routes.itertuples(index=False):
        parts = str(r.display_name).split()
        if parts and parts[-1].lower().rstrip(".") == last_lower:
            return int(r.nfl_id)
    return None


def get_receiver_matchups(game_id: int, play_id: int) -> pd.DataFrame:
    """Receiver->nearest-coverage-defender pairs for a play (``matchups`` schema).

    The ``blocker_*`` columns carry the receiver, ``rusher_*`` the coverage
    defender, so ``features`` / ``score`` / ``viz`` consume this unchanged. Two
    extra columns are appended (allowed by the frozen schema): ``is_target`` and
    ``coverage_type``.

    Parameters
    ----------
    game_id, play_id:
        The play.

    Returns
    -------
    pandas.DataFrame
        One row per route runner (empty frame if the play is unknown).
        ``pairing_ok`` is ``True`` only when a defender was found AND the play is
        man coverage.
    """
    plays = up.plays()
    prow = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)]
    if not len(prow):
        return pd.DataFrame(columns=MATCHUPS_COLS + ["is_target", "coverage_type"])

    cov_type = str(prow["pff_passCoverageType"].iloc[0])
    man_scoped = cov_type.lower().startswith("man")
    team = str(prow["possessionTeam"].iloc[0])  # receivers are on offense
    target_name = parse_target(prow["playDescription"].iloc[0])

    play = up.load_play(game_id, play_id)
    snap_frame, end_frame = up.rep_window(play)
    target_id = _match_name_to_id(target_name, play)

    snap = play[play["frame_id"] == snap_frame]
    receivers = snap[snap["pff_role"] == up.ROLE_ROUTE]
    defenders = snap[snap["pff_role"] == up.ROLE_COVERAGE]
    qb = snap.loc[snap["pff_role"] == up.ROLE_PASS, "nfl_id"]
    qb_id = int(qb.iloc[0]) if len(qb) else up.BALL_ID

    rows = []
    claimed: set[int] = set()
    for rec in receivers.itertuples(index=False):
        best, best_d = None, np.inf
        for d in defenders.itertuples(index=False):
            if d.nfl_id in claimed:
                continue
            dist = float(np.hypot(rec.x - d.x, rec.y - d.y))
            if dist < best_d:
                best_d, best = dist, d
        if best is not None:
            claimed.add(int(best.nfl_id))
        def_id = int(best.nfl_id) if best is not None else up.BALL_ID
        rows.append(
            {
                "game_id": int(game_id), "play_id": int(play_id),
                "blocker_id": int(rec.nfl_id), "blocker_name": rec.display_name, "blocker_pos": rec.position,
                "rusher_id": def_id,
                "rusher_name": (best.display_name if best is not None else None),
                "rusher_pos": (best.position if best is not None else None),
                "qb_id": qb_id, "snap_frame": snap_frame, "end_frame": end_frame,
                "block_type": "COVERAGE",
                "is_switch": False, "is_chip": False, "is_double": False,
                "pressure_allowed": 0, "sack_allowed": 0, "hurry_allowed": 0, "hit_allowed": 0,
                "pairing_ok": bool(best is not None and man_scoped),
                "team": team,
                "is_target": bool(target_id is not None and int(rec.nfl_id) == target_id),
                "coverage_type": cov_type,
            }
        )
    return pd.DataFrame(rows, columns=MATCHUPS_COLS + ["is_target", "coverage_type"])


if __name__ == "__main__":
    g = up.GOLDEN
    print("receiver.py demo —  upstream source:", up.upstream_source())
    print("\nparse_target checks:")
    for desc in (
        "(13:33) (Shotgun) T.Brady pass incomplete deep right to C.Godwin.",
        "(12:23) (Shotgun) D.Prescott pass short middle to D.Schultz to DAL 39 for 5 yards (D.White).",
        "(2:00) D.Prescott kneels to DAL 20 for -1 yards.",
    ):
        print(f"  {parse_target(desc)!r:14} <- {desc[:70]}")

    rm = get_receiver_matchups(g["game_id"], g["play_id"])
    print(f"\nreceiver matchups on {g['game_id']}/{g['play_id']}  (coverage: "
          f"{rm['coverage_type'].iloc[0] if len(rm) else 'n/a'}):")
    cols = ["blocker_name", "blocker_pos", "rusher_name", "pairing_ok", "is_target"]
    print(rm[cols].to_string(index=False) if len(rm) else "  (none)")
