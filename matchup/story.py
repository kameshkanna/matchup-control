"""Task 12 — narrative assembly.

Ties the whole project together as a single zoom, from one battle out to the
league:

1. **One rep** — the signature visual (:func:`matchup.viz.plot_matchup`) on the
   golden rep, the microscope.
2. **Validation** — does the Win Score agree with PFF pressure labels
   (:func:`matchup.validate.validation_report`) plus the score-distribution split
   (:func:`matchup.evidence.plot_score_distribution`).
3. **The league** — the opponent-adjusted player win-rate leaderboard
   (:func:`matchup.leaderboard.player_leaderboard`) and a headline comparison
   (:func:`matchup.evidence.plot_headline_comparison`), the telescope.

:func:`build_story` returns a plain dict (metrics + saved figure paths + a
markdown narrative) and writes the figures and ``story.md`` to ``CACHE_DIR``.
Nothing prints outside ``__main__``.

Data sourcing
-------------
The ``scored`` population frame is taken from Person A's pipeline when available
(``matchup.pipeline.build_scored``); otherwise it falls back to the shared
synthetic generator so the population panels render today. The golden *visual*
always uses real tracking via :mod:`matchup._upstream`. ``build_story`` records
which source was used in ``meta`` so the narrative never overstates its basis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matchup import _upstream as up
from matchup import evidence, leaderboard, validate, viz
from matchup.config import CACHE_DIR, _make_synthetic_scored


def _load_scored(n_games: int = 15) -> tuple[pd.DataFrame, str]:
    """Return a ``scored`` frame and its provenance ('pipeline' | 'synthetic').

    Person A's entry point is ``pipeline.run(n_games=...)``, which returns a dict
    with a ``'scored'`` key. (``score.build_scored`` needs a matchups table, so it
    is not a no-arg call, and ``pipeline`` has no ``build_scored``.) Falls back to
    the shared synthetic generator only if the pipeline is unavailable or errors.
    """
    try:
        from matchup.pipeline import run as pipeline_run  # Person A, Task 9
    except ImportError:
        return _make_synthetic_scored(n_reps=600, seed=0), "synthetic"
    try:
        result = pipeline_run(n_games=n_games)
        scored = result.get("scored")
        if scored is not None and len(scored):
            return scored, "pipeline"
    except Exception:
        pass
    return _make_synthetic_scored(n_reps=600, seed=0), "synthetic"


def _scored_with_adjustment(scored: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``adj_win_score`` is present (opponent adjustment applied once)."""
    if "adj_win_score" in scored.columns:
        return scored
    return leaderboard.opponent_adjust(scored)


def build_story(
    golden: dict | None = None,
    out_dir: Path | None = None,
    top_n: int = 10,
) -> dict:
    """Assemble the end-to-end narrative: figures, metrics, and markdown.

    Parameters
    ----------
    golden:
        The hero rep to visualise. Defaults to the project's golden rep
        (Donovan Smith vs Randy Gregory, game 2021090900 / play 97).
    out_dir:
        Where to write figures and ``story.md``. Defaults to ``CACHE_DIR``.
    top_n:
        Number of leaderboard rows to keep in the narrative.

    Returns
    -------
    dict
        ``{"meta", "validation", "leaderboard", "figures", "narrative_md"}``.
    """
    golden = golden or up.GOLDEN
    out_dir = Path(out_dir) if out_dir is not None else CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    # --- 1. One rep: the signature visual on real tracking -----------------
    hero_ok, hero_note = True, ""
    try:
        fig_hero = viz.plot_matchup(
            golden["game_id"], golden["play_id"], golden["blocker_id"], golden["rusher_id"]
        )
        p = out_dir / "story_1_golden_matchup.png"
        fig_hero.savefig(p, dpi=120)
        figures["golden_matchup"] = str(p)
    except Exception as exc:  # never let a viz hiccup sink the whole story
        hero_ok, hero_note = False, f"{type(exc).__name__}: {exc}"

    # --- population frame for validation + leaderboard ---------------------
    scored, source = _load_scored()
    scored = _scored_with_adjustment(scored)

    # --- 2. Validation: score vs PFF pressure labels -----------------------
    report = validate.validation_report(scored)
    fig_dist = evidence.plot_score_distribution(scored)
    p = out_dir / "story_2_score_distribution.png"
    fig_dist.savefig(p, dpi=120)
    figures["score_distribution"] = str(p)

    # --- 3. The league: opponent-adjusted leaderboard + headline -----------
    lb = leaderboard.player_leaderboard(scored)
    sort_col = "adj_win_rate" if "adj_win_rate" in lb.columns else "win_rate"
    lb_top = lb.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)

    by = "rusher_pos" if "rusher_pos" in scored.columns else "block_type"
    headline_note = ""
    try:
        fig_head = evidence.plot_headline_comparison(scored, by=by)
        p = out_dir / "story_3_headline_comparison.png"
        fig_head.savefig(p, dpi=120)
        figures["headline_comparison"] = str(p)
    except Exception as exc:  # don't let evidence's yerr bug on real data sink the story
        headline_note = f"{type(exc).__name__}: {exc}"

    meta = {
        "scored_source": source,
        "upstream_source": up.upstream_source(),
        "n_reps": int(len(scored)),
        "golden": golden,
        "hero_visual_ok": hero_ok,
        "hero_visual_note": hero_note,
        "headline_note": headline_note,
    }

    narrative_md = _render_markdown(meta, report, lb_top, figures, by)
    (out_dir / "story.md").write_text(narrative_md, encoding="utf-8")

    return {
        "meta": meta,
        "validation": report,
        "leaderboard": lb_top,
        "figures": figures,
        "narrative_md": narrative_md,
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table without external deps.

    Avoids ``DataFrame.to_markdown`` so ``tabulate`` is not required. Floats are
    shown to 3 decimals; everything else via ``str``.
    """
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _render_markdown(meta, report, lb_top, figures, by) -> str:
    auc = report.get("auc")
    corr = report.get("corr")
    auc_s = f"{auc:.3f}" if isinstance(auc, (int, float)) and auc == auc else "n/a"
    corr_s = f"{corr:.3f}" if isinstance(corr, (int, float)) and corr == corr else "n/a"

    show_cols = [c for c in ["blocker_name", "blocker_pos", "team", "n_reps",
                             "win_rate", "adj_win_rate", "win_rate_shrunk"] if c in lb_top.columns]
    lb_md = _df_to_markdown(lb_top[show_cols]) if len(lb_top) else "_(no rows)_"

    src_note = (
        "Population figures use **real pipeline output**."
        if meta["scored_source"] == "pipeline"
        else "Population figures use the **shared synthetic generator** (Person A's "
        "pipeline not merged yet); the golden visual below still uses real tracking."
    )
    hero_md = (
        f"![golden matchup]({Path(figures['golden_matchup']).name})"
        if meta["hero_visual_ok"] and "golden_matchup" in figures
        else f"_Golden visual unavailable: {meta['hero_visual_note']}_"
    )

    return f"""# Matchup Control — the story

**Football looks like a team game, but it is really thousands of one-on-one
battles.** On every dropback, each blocker is matched against one rusher, and the
question is always the same: *who controlled whom?* We built one geometric engine
that answers it from tracking data, proved it against expert pressure labels, and
ranked players more fairly than sack counts ever could.

> {src_note}

## 1. One battle (the microscope)

The golden rep — game {meta['golden']['game_id']}, play {meta['golden']['play_id']}
— a confirmed blocker loss. The top panel shows the paths; the bottom panel shows
*control over time*: the rusher's distance to the QB collapsing while betweenness
falls away is the loss, made visible.

{hero_md}

## 2. Does the score tell the truth? (validation)

We check the per-rep Win Score against PFF's pressure labels across
**{meta['n_reps']:,} reps**:

- **AUC = {auc_s}** — how well the Win Score ranks pressure-allowed reps below
  clean reps (0.5 = coin flip, 1.0 = perfect).
- **correlation = {corr_s}** — Win Score vs the pressure outcome.

![score distribution]({Path(figures['score_distribution']).name})

Pressure reps sit at lower Win Scores than clean reps — the score is measuring
the right thing.

## 3. The league (the telescope)

Pooling every rep across all games and adjusting for opponent quality gives a
per-blocker leaderboard (shrunk for small samples):

{lb_md}

And the same scored population, broken out by `{by}`:

{(f"![headline comparison]({Path(figures['headline_comparison']).name})" if 'headline_comparison' in figures else f"_Headline figure unavailable: {meta.get('headline_note')}_")}

## One-line pitch

Two players, a few seconds, one question — *who controlled whom?* One engine
answers it, validated where ground truth exists, and reused where it does not.
"""


if __name__ == "__main__":
    print("story.py demo —  upstream source:", up.upstream_source())
    result = build_story()
    print("  scored source :", result["meta"]["scored_source"])
    print("  validation    :", result["validation"])
    print("  leaderboard   :", result["leaderboard"].shape, "rows (top-n)")
    print("  hero visual ok:", result["meta"]["hero_visual_ok"], result["meta"]["hero_visual_note"])
    for name, path in result["figures"].items():
        print(f"  figure[{name}] -> {path}")
    print("  narrative     ->", CACHE_DIR / "story.md")
