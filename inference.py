"""Inference / demo harness for the Matchup Control visualizations (Person C).

Run this to SEE what every visual in the project produces, end to end, on real
tracking data (with synthetic fallback for the population panels when Person A's
pipeline is not merged yet). It does not test correctness with asserts — it
exercises each entry point, saves the figure, and prints a short, readable
description of the output so you can eyeball what the graphics show.

Run:
    python inference.py                      # golden rep + full story
    python inference.py --game 2021090900 --play 97 --blocker 42377
    python inference.py --receiver           # also render the receiver head

Outputs land in ``cache/inference/`` (git-ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; save PNGs, never pop a window

import pandas as pd

from matchup import _upstream as up
from matchup import evidence, leaderboard, receiver, story, validate, viz
from matchup.config import CACHE_DIR, _make_synthetic_scored

OUT = CACHE_DIR / "inference"
OUT.mkdir(parents=True, exist_ok=True)

_RULE = "-" * 72


def _save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def _describe_matchup(game_id: int, play_id: int, blocker_id: int,
                      rusher_id: int | None) -> None:
    """Render plot_matchup and report what the two panels contain."""
    print(_RULE)
    print(f"[1] plot_matchup — one rep  (game {game_id}, play {play_id}, blocker {blocker_id})")

    mu = up.matchups(game_id, play_id)
    row = mu[mu["blocker_id"] == blocker_id]
    if rusher_id is None and len(row):
        rusher_id = int(row["rusher_id"].iloc[0])
    if len(row):
        r = row.iloc[0]
        print(f"    pairing   : {r['blocker_name']} ({r['blocker_pos']}) "
              f"vs {r['rusher_name']} ({r['rusher_pos']})")
        outcome = ("SACK" if r["sack_allowed"] else "HIT" if r["hit_allowed"]
                   else "HURRY" if r["hurry_allowed"] else "clean")
        print(f"    label     : pressure_allowed={r['pressure_allowed']} ({outcome}), "
              f"pairing_ok={bool(r['pairing_ok'])}")

    fig = viz.plot_matchup(game_id, play_id, blocker_id, rusher_id)
    path = _save(fig, f"matchup_{game_id}_{play_id}_{blocker_id}.png")

    # Describe the control curve the bottom panel drew.
    rt = up.rep_tracks(game_id, play_id, blocker_id, rusher_id).dropna(subset=["blk_x", "rsh_x"])
    cts = up.control_ts(rt)
    d0, d1 = cts["rusher_to_qb_dist"].iloc[0], cts["rusher_to_qb_dist"].iloc[-1]
    b0, b1 = cts["betweenness"].iloc[0], cts["betweenness"].iloc[-1]
    verdict = "rusher CLOSED on the QB (blocker losing ground)" if d1 < d0 else \
              "rusher HELD OFF (blocker kept distance)"
    print(f"    top panel : {len(rt)} frames over {rt['t_sec'].iloc[-1]:.1f}s; "
          f"blue=blocker, red=rusher, white=QB, yellow dashed=LOS")
    print(f"    bottom    : rusher→QB dist {d0:.1f} → {d1:.1f} yd  [{verdict}]")
    print(f"                betweenness {b0:.2f} → {b1:.2f}  (1 = perfectly between)")
    print(f"    saved     : {path}")


def _describe_receiver(game_id: int, play_id: int) -> None:
    """Render the receiver head's matchups and plot the targeted receiver."""
    print(_RULE)
    print(f"[2] receiver head — get_receiver_matchups  (game {game_id}, play {play_id})")
    rm = receiver.get_receiver_matchups(game_id, play_id)
    if not len(rm):
        print("    (no receiver matchups on this play)")
        return
    cov = rm["coverage_type"].iloc[0]
    print(f"    coverage  : {cov}  (man-scoped pairing_ok only when Man)")
    cols = ["blocker_name", "blocker_pos", "rusher_name", "pairing_ok", "is_target"]
    print(rm[cols].to_string(index=False).replace("\n", "\n    "))

    tgt = rm[rm["is_target"]]
    if len(tgt):
        t = tgt.iloc[0]
        print(f"    targeted  : {t['blocker_name']} vs {t['rusher_name']} — rendering reuse of plot_matchup")
        try:
            fig = viz.plot_matchup(game_id, play_id, int(t["blocker_id"]), int(t["rusher_id"]))
            path = _save(fig, f"receiver_{game_id}_{play_id}_{int(t['blocker_id'])}.png")
            print(f"    saved     : {path}")
        except ValueError as exc:
            print(f"    (receiver visual skipped: {exc})")


def _describe_evidence() -> None:
    """Render the two population evidence figures and the validation report."""
    print(_RULE)
    print("[3] population evidence + validation")
    scored, src = None, "synthetic"
    try:
        from matchup.pipeline import run as pipeline_run
        result = pipeline_run(n_games=15)
        cand = result.get("scored")
        if cand is not None and len(cand):
            scored, src = cand, "pipeline"
    except Exception:
        pass
    if scored is None:
        scored = _make_synthetic_scored(n_reps=600, seed=0)
    print(f"    scored    : {len(scored)} reps  (source: {src})")

    rep = validate.validation_report(scored)
    print(f"    validation: AUC={rep.get('auc'):.3f}  corr={rep.get('corr'):.3f}  n={rep.get('n')}")
    print(f"                (AUC>0.5 means Win Score ranks clean reps above pressure reps)")

    f1 = evidence.plot_score_distribution(scored)
    p1 = _save(f1, "evidence_score_distribution.png")
    clean = (scored["pressure_allowed"] == 0).sum()
    press = (scored["pressure_allowed"] == 1).sum()
    print(f"    fig score_distribution: 2 overlaid histograms — clean (n={clean}) vs "
          f"pressure (n={press}); dashed lines = medians")
    print(f"    saved     : {p1}")

    by = "rusher_pos" if "rusher_pos" in scored.columns else "block_type"
    try:
        f2 = evidence.plot_headline_comparison(scored, by=by)
        p2 = _save(f2, f"evidence_headline_{by}.png")
        print(f"    fig headline_comparison: blocker win_rate bars grouped by '{by}' with 95% Wilson CI")
        print(f"    saved     : {p2}")
    except Exception as exc:  # Person B's evidence bug on real data — report, don't crash
        print(f"    fig headline_comparison: SKIPPED — evidence.plot_headline_comparison raised "
              f"{type(exc).__name__}: {exc}")
        print(f"      -> likely Person B bug: yerr can go negative when the Wilson CI is clipped "
              f"to [0,1] but `rate` is not; clamp yerr with np.clip(..., 0, None).")

    lb = leaderboard.player_leaderboard(scored)
    sort_col = "adj_win_rate" if "adj_win_rate" in lb.columns else "win_rate"
    top = lb.sort_values(sort_col, ascending=False).head(5)
    show = [c for c in ["blocker_name", "n_reps", "win_rate", "adj_win_rate", "win_rate_shrunk"]
            if c in top.columns]
    print("    leaderboard top-5:")
    print(top[show].to_string(index=False).replace("\n", "\n      "))


def _describe_story() -> None:
    print(_RULE)
    print("[4] full story assembly — build_story()")
    result = story.build_story(out_dir=OUT / "story")
    m = result["meta"]
    print(f"    scored source : {m['scored_source']}   upstream: {m['upstream_source']}")
    print(f"    hero visual   : ok={m['hero_visual_ok']} {m['hero_visual_note']}")
    print(f"    validation    : {result['validation']}")
    for name, path in result["figures"].items():
        print(f"    figure[{name}] -> {path}")
    print(f"    narrative     -> {OUT / 'story' / 'story.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Matchup Control visualization inference harness")
    ap.add_argument("--game", type=int, default=up.GOLDEN["game_id"])
    ap.add_argument("--play", type=int, default=up.GOLDEN["play_id"])
    ap.add_argument("--blocker", type=int, default=up.GOLDEN["blocker_id"])
    ap.add_argument("--rusher", type=int, default=None, help="defaults to the blocker's PFF assignment")
    ap.add_argument("--receiver", action="store_true", help="also render the receiver head")
    ap.add_argument("--no-story", action="store_true", help="skip the full story assembly")
    args = ap.parse_args()

    print("=" * 72)
    print("MATCHUP CONTROL — visualization inference")
    print(f"  upstream source : {up.upstream_source()}  "
          f"({'Person A merged' if up.upstream_source() == 'person_a' else 'using local fallback'})")
    print(f"  output dir      : {OUT}")
    print("=" * 72)

    _describe_matchup(args.game, args.play, args.blocker, args.rusher)
    if args.receiver:
        _describe_receiver(args.game, args.play)
    _describe_evidence()
    if not args.no_story:
        _describe_story()

    print(_RULE)
    print(f"done — open the PNGs under {OUT} to view the graphics.")


if __name__ == "__main__":
    main()
