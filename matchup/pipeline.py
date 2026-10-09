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


_OL_POS = {"T", "OT", "LT", "RT", "G", "OG", "LG", "RG", "C"}


def _ol_only(scored):
    """Keep only true offensive linemen (T/G/C) so stray RB/TE pass-protection
    reps don't pollute the lineman leaderboard."""
    if "blocker_pos" not in scored.columns:
        return scored
    pos = scored["blocker_pos"].astype(str).str.upper()
    return scored[pos.isin(_OL_POS)]


def _save_importance_chart(imp_df, path):
    """Horizontal bar chart of permutation feature importance (the 'what makes
    a block win or lose' story)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = imp_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d["feature"], d["importance"], color="C0")
    ax.set_xlabel("permutation importance (drop in accuracy)")
    ax.set_title("What separates a won block from a lost one")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


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

    # 6: fit the gradient-boosting Control Score against PFF pressure labels.
    try:
        from . import validate
        # feature_frame carries the GBM features + label (output of build_scored),
        # BEFORE production gbm columns are applied. Both the production fit and
        # the OOF validation consume this same frame.
        feature_frame = scored
        model, feats = score.fit_gbm_scorer(feature_frame)
        # PRODUCTION scoring: one model fit on all reps grades every rep. This is
        # the deliverable the leaderboard ranks on.
        scored = score.gbm_win_score(feature_frame, model, feats)
        result["scored"] = scored
        result["gbm_model"] = model
        result["gbm_features"] = feats
        result["feature_importance"] = score.gbm_feature_importance(scored, model, feats)

        # In-sample validation (optimistic): the model has seen every rep it is
        # graded on, so this AUC is inflated. Kept only for comparison.
        result["validation_insample"] = validate.validation_report(scored)

        # OOF validation (honest): each rep is scored by a model that did not
        # train on it. This is the number to trust.
        try:
            oof = score.oof_gbm_scores(feature_frame, features=feats)
            result["validation"] = validate.validation_report(oof)
            result["cv_strategy"] = oof.attrs.get("cv_strategy", "unknown")
        except Exception as e:
            result["validation_oof_error"] = str(e)
    except Exception as e:
        result["validation_error"] = str(e)

    # 8: opponent adjustment + leaderboard (Person B), OL-only (T/G/C).
    try:
        from . import leaderboard
        board_input = _ol_only(result["scored"])
        adj = leaderboard.opponent_adjust(board_input)
        board = leaderboard.player_leaderboard(adj)
        # Rank by continuous score, not the saturated win-rate flag.
        rank_col = "adj_mean_win_score" if "adj_mean_win_score" in board.columns \
            else ("mean_win_score" if "mean_win_score" in board.columns else None)
        if rank_col:
            board = board.sort_values(rank_col, ascending=False).reset_index(drop=True)
        result["leaderboard"] = board
        C.cache_write(board, f"leaderboard_{tag}games")
    except Exception as e:
        result["leaderboard_error"] = str(e)

    # 10: evidence charts + validation + feature-importance -> PNGs
    figs = []
    try:
        from . import evidence
        evidence.plot_score_distribution(result["scored"]).savefig(
            C.CACHE_DIR / "fig_score_distribution.png", dpi=120)
        figs.append("fig_score_distribution.png")
    except Exception as e:
        result["figure_error_dist"] = str(e)
    try:
        from . import validate
        validate.plot_validation(result["scored"]).savefig(
            C.CACHE_DIR / "fig_validation.png", dpi=120)
        figs.append("fig_validation.png")
    except Exception as e:
        result["figure_error_val"] = str(e)
    if "feature_importance" in result:
        try:
            _save_importance_chart(result["feature_importance"],
                                   C.CACHE_DIR / "fig_feature_importance.png")
            figs.append("fig_feature_importance.png")
        except Exception as e:
            result["figure_error_imp"] = str(e)
    result["figures"] = figs

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
        lines.append(f"validation (OOF, honest):     {result['validation']}")
    if "cv_strategy" in result:
        lines.append(f"OOF cv_strategy: {result['cv_strategy']}")
    if "validation_insample" in result:
        lines.append(f"validation (in-sample, optimistic): {result['validation_insample']}")
    if "validation_oof_error" in result:
        lines.append(f"(OOF validation failed: {result['validation_oof_error']})")
    if "validation_error" in result:
        lines.append(f"(scoring step failed: {result['validation_error']})")
    if "feature_importance" in result:
        lines.append("\nwhat predicts a lost block (GBM permutation importance):")
        for _, row in result["feature_importance"].head(8).iterrows():
            lines.append(f"    {row['feature']:22s} {row['importance']:+.4f}")
    if "leaderboard" in result:
        lines.append("\ntop 10 blockers:")
        lines.append(result["leaderboard"].head(10).to_string(index=False))
        lines.append("\nbottom 5 blockers:")
        lines.append(result["leaderboard"].tail(5).to_string(index=False))
    if "leaderboard_error" in result:
        lines.append(f"(leaderboard step failed: {result['leaderboard_error']})")
    if "figures" in result:
        lines.append(f"\nsaved charts to cache/: {', '.join(result['figures'])}")
    if "figure_error" in result:
        lines.append(f"(chart step failed: {result['figure_error']})")
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
