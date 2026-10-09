"""Simple end-to-end check for the Matchup Control engine.

Run:  python run_demo.py            # scores 5 games (default)
      python run_demo.py 10         # scores 10 games

Prints a readable summary and writes the scored table to cache/scored_demo.parquet.
No fancy pipeline, no optional modules — just: matchups -> features -> score,
so we can confirm the core engine works and is useful before building more.
"""
import sys
import time

import pandas as pd

from matchup import config as C
from matchup import io_load, pairing, score, stunts


def main(n_games: int = 5):
    t_start = time.time()
    games = io_load.load_games().sort_values(["week", "game_id"])
    game_ids = games["game_id"].astype(int).tolist()[:n_games]
    print(f"Scoring {len(game_ids)} games: {game_ids}")

    # 1-2: build the blocker->rusher matchup table
    t0 = time.time()
    matchups = pairing.get_all_matchups(game_ids, use_cache=False)
    matchups = stunts.tag_stunts(matchups)
    print(f"  matchups built: {len(matchups)} reps  ({time.time()-t0:.1f}s)")

    # 3-4: features + win score
    t0 = time.time()
    scored = score.build_scored(matchups)
    scored = stunts.resolve_attribution(scored)
    print(f"  scored:          {len(scored)} reps  ({time.time()-t0:.1f}s)")

    scored.to_csv(C.CACHE_DIR / "scored_demo.csv", index=False)

    # ---- Readable summary -------------------------------------------------
    ok = scored[scored["win_score"].notna()].copy()
    print("\n================= RESULTS =================")
    print(f"total reps: {len(scored)}   scored reps: {len(ok)}")
    print(f"pairing_ok rate: {matchups['pairing_ok'].mean():.1%}")

    # Does the geometric score agree with real PFF pressure labels?
    # Use the ABSOLUTE score for this (position-relative is zero-mean by group).
    score_col = "win_score_abs" if "win_score_abs" in ok.columns else "win_score"
    clean = ok[ok["pressure_allowed"] == 0][score_col]
    pressure = ok[ok["pressure_allowed"] == 1][score_col]
    print(f"\n--- does the score track reality? ({score_col}, higher = blocker won) ---")
    print(f"mean, CLEAN reps (no pressure allowed): {clean.mean():+.2f}  (n={len(clean)})")
    print(f"mean, PRESSURE allowed reps:            {pressure.mean():+.2f}  (n={len(pressure)})")
    print(f"gap (should be clearly positive): {clean.mean() - pressure.mean():+.2f}")

    # Simple per-player leaderboard (unadjusted) as a sanity check
    board = (
        ok.groupby(["blocker_id", "blocker_name", "blocker_pos"])
        .agg(n_reps=("win_flag", "size"),
             win_rate=("win_flag", "mean"),
             mean_score=("win_score", "mean"),
             mean_abs=("win_score_abs", "mean") if "win_score_abs" in ok.columns
                      else ("win_score", "mean"))
        .reset_index()
    )
    board = board[board["n_reps"] >= 10].sort_values("mean_score", ascending=False)
    print("\n--- top 10 blockers (position-relative score, >=10 reps) ---")
    print("    (win_rate/score are now RELATIVE to position group)")
    print(board.head(10).to_string(index=False))
    print("\n--- bottom 5 blockers (position-relative score, >=10 reps) ---")
    print(board.tail(5).to_string(index=False))

    print(f"\nTOTAL TIME: {time.time()-t_start:.1f}s")
    print("saved: cache/scored_demo.csv")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n)
