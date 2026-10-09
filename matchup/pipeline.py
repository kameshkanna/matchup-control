"""Task 9 (Person A) — end-to-end pipeline wiring.

One command: raw CSVs -> normalised tracking -> matchups -> features -> scores
-> (optional) label calibration -> (optional) opponent-adjusted leaderboard,
with parquet caching of the expensive intermediates.

Person B's validate.py and leaderboard.py are imported defensively: the core
pipeline (through scored reps) runs even before those modules exist.

Usage:
    python -m matchup.pipeline --games 15
    python -m matchup.pipeline --all
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from . import io_load
from . import pairing
from . import score
from . import stunts


def select_game_ids(n: int | None) -> list[int]:
    games = io_load.load_games().sort_values(["week", "game_id"])
    ids = games["game_id"].astype(int).tolist()
    return ids if n is None else ids[:n]


def run(n_games: int | None = 15, use_cache: bool = True) -> dict:
    """Build the full scored table (and leaderboard if Person B's modules are
    present). Returns a dict of the key DataFrames and writes parquet caches."""
    game_ids = select_game_ids(n_games)
    tag = "all" if n_games is None else str(len(game_ids))

    # 1-2: matchups
    matchups = pairing.get_all_matchups(game_ids, use_cache=use_cache)

    # 7: stunt flags
    matchups = stunts.tag_stunts(matchups)

    # 3-4: features + score
    scored_stem = f"scored_{tag}games"
    cached_scored = C.cache_read(scored_stem) if use_cache else None
    if cached_scored is not None:
        scored = cached_scored
    else:
        scored = score.build_scored(matchups)
        scored = stunts.resolve_attribution(scored)
        C.cache_write(scored, scored_stem)

    result = {"matchups": matchups, "scored": scored}

    # 6: calibrate weights against PFF labels, re-score (Person B)
    try:
        from . import validate
        weights = validate.calibrate_weights(scored)
        scored = score.score_matchups(scored, weights=weights)
        scored = stunts.resolve_attribution(scored)
        result["scored"] = scored
        result["weights"] = weights
        result["validation"] = validate.validation_report(scored)
    except Exception as e:
        result["validation_error"] = str(e)

    # 8: opponent adjustment + leaderboard (Person B)
    try:
        from . import leaderboard
        adj = leaderboard.opponent_adjust(result["scored"])
        board = leaderboard.player_leaderboard(adj)
        result["leaderboard"] = board
        C.cache_write(board, f"leaderboard_{tag}games")
    except Exception as e:
        result["leaderboard_error"] = str(e)

    return result


def _summarise(result: dict) -> str:
    scored = result["scored"]
    lines = [
        f"matchups: {len(result['matchups'])}",
        f"scored reps: {len(scored)}",
        f"pairing_ok rate: {result['matchups'].pairing_ok.mean():.3f}",
    ]
    if "pressure_allowed" in scored and scored.win_score.notna().any():
        ok = scored[scored.win_score.notna()]
        lines += [
            f"mean win_score | pressure=0: {ok[ok.pressure_allowed==0].win_score.mean():.3f}",
            f"mean win_score | pressure=1: {ok[ok.pressure_allowed==1].win_score.mean():.3f}",
        ]
    if "validation" in result:
        lines.append(f"validation: {result['validation']}")
    if "validation_error" in result:
        lines.append(f"(validate.py not ready yet: {result['validation_error']})")
    if "leaderboard" in result:
        lines.append("top 10 blockers (by available ranking col):")
        lines.append(result["leaderboard"].head(10).to_string(index=False))
    if "leaderboard_error" in result:
        lines.append(f"(leaderboard.py not ready yet: {result['leaderboard_error']})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=15, help="number of games")
    ap.add_argument("--all", action="store_true", help="use all games")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    n = None if args.all else args.games
    res = run(n_games=n, use_cache=not args.no_cache)
    msg = _summarise(res)
    print(msg)
    (C.CACHE_DIR / "_pipeline_summary.txt").write_text(msg, encoding="utf-8")
