"""Matchup Control — basic Streamlit dashboard.

Shows everything the pipeline produces: validation metrics, the blocker
leaderboard, the evidence charts, and a scored-reps explorer. Reads cached
results instantly and only recomputes when you click "Run pipeline".

Launch from the repo root:
    streamlit run matchup/app.py

Built to be resilient to the pipeline's exact internals: every result key is
read defensively, so this keeps working as teammates change scoring/validation.
"""
from __future__ import annotations

import sys
from pathlib import Path

# This file lives at <repo_root>/matchup/app.py. Streamlit puts the script's own
# folder (matchup/) on sys.path, not the repo root, so `import matchup` can fail.
# Add the repo root (one level up) to sys.path so the package resolves no matter
# where `streamlit run` is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from matchup import config as C
from matchup import pipeline

# Person C's modules — imported defensively so the app still loads if one is
# mid-change on a teammate's branch.
try:
    from matchup import viz
except Exception:  # pragma: no cover
    viz = None
try:
    from matchup import receiver
except Exception:  # pragma: no cover
    receiver = None
try:
    from matchup import story
except Exception:  # pragma: no cover
    story = None

st.set_page_config(page_title="Matchup Control", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_pipeline(n_games: int) -> dict:
    """Run the full pipeline. Cached by n_games so re-selecting is instant."""
    return pipeline.run(n_games=n_games, use_cache=True)


def load_cached_leaderboard(n_games):
    """Try to load a previously-saved leaderboard without recomputing.

    The pipeline tags caches 'all' when n_games is None, else the game count.
    """
    tag = "all" if n_games is None else str(n_games)
    return C.cache_read(f"leaderboard_{tag}games")


def chart_path(name: str):
    p = C.CACHE_DIR / name
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# Human-readable column labels (so no raw snake_case / ids reach the viewer)
# --------------------------------------------------------------------------- #
LEADERBOARD_LABELS = {
    "blocker_name": "Player",
    "blocker_pos": "Pos",
    "team": "Team",
    "n_reps": "Blocks graded",
    "win_rate_shrunk": "Win rate",            # headline (sample-size corrected)
    "adj_win_rate": "Win rate (vs opp.)",     # opponent-adjusted
}
REPS_LABELS = {
    "blocker_name": "Blocker",
    "rusher_name": "Rusher",
    "blocker_pos": "Pos",
    "outcome": "Outcome",
    "win_score": "Control grade",
    "ground_given_up": "Ground given up (yds)",
    "sep_min": "Closest the blocker got (yds)",
    "min_rusher_to_qb_dist": "Rusher's closest to QB (yds)",
    "betweenness_mean": "Shielded QB (0-1)",
}


def rename_for_display(df, labels):
    """Return a copy with only the labelled columns present, renamed to labels."""
    cols = [c for c in labels if c in df.columns]
    return df[cols].rename(columns={c: labels[c] for c in cols})


def rep_outcome(r):
    """Human outcome string for a scored rep row."""
    if r.get("sack_allowed"):
        return "Sack"
    if r.get("hit_allowed"):
        return "Hit"
    if r.get("hurry_allowed"):
        return "Hurry"
    return "Clean"


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Matchup Control")
st.sidebar.caption("Grading one-on-one pass-protection battles from tracking data")

# Always analyse all games; no games slider.
n_games = None  # None => all games in pipeline.run()
run_clicked = st.sidebar.button("Run pipeline (all games)", type="primary")
st.sidebar.caption(
    "Analyses all available games. Loads cached results instantly; click "
    "**Run pipeline** to (re)compute from scratch."
)

# Decide what to show: a fresh run, or whatever is cached.
result = None
if run_clicked:
    with st.spinner("Running pipeline on all games… this can take a few minutes."):
        result = run_pipeline(n_games)
    st.session_state["last_result"] = result
elif "last_result" in st.session_state:
    result = st.session_state["last_result"]


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🏈 Matchup Control — Pass-Protection Evaluator")
st.markdown(
    "A tracking-derived score for **who won each one-on-one block**, validated "
    "against PFF pressure labels and adjusted for the rusher faced."
)

if result is None:
    st.info(
        "No results loaded yet. Click **Run pipeline (all games)** in the sidebar. "
        "Showing any cached leaderboard below if present."
    )
    board = load_cached_leaderboard(n_games)
    if board is not None:
        st.subheader("Cached leaderboard")
        st.dataframe(board, use_container_width=True)
    st.stop()


# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #
scored = result.get("scored")
matchups = result.get("matchups")
validation = result.get("validation", {}) or {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Scored reps", f"{len(scored):,}" if scored is not None else "—")
if matchups is not None and "pairing_ok" in matchups:
    c2.metric("Pairing success", f"{matchups['pairing_ok'].mean():.1%}")
else:
    c2.metric("Pairing success", "—")
auc = validation.get("auc")
c3.metric("Validation AUC", f"{auc:.3f}" if isinstance(auc, (int, float)) else "pending")
corr = validation.get("corr")
c4.metric("Score vs pressure corr", f"{corr:.3f}" if isinstance(corr, (int, float)) else "—")

# Surface any step errors so the UI is honest about what ran.
for key, label in [
    ("validation_error", "Scoring/validation step"),
    ("leaderboard_error", "Leaderboard step"),
    ("figure_error_dist", "Score-distribution chart"),
    ("figure_error_val", "Validation chart"),
    ("figure_error_imp", "Feature-importance chart"),
]:
    if key in result:
        st.warning(f"{label} did not complete: {result[key]}")


# --------------------------------------------------------------------------- #
# Tabs: three question-driven views + the two bonus heads
#   Players    → "who is good?"
#   How it works → "does the metric work?" (validation + method)
#   Film room  → "what happened on this block?" (pick a rep, see the visual)
# --------------------------------------------------------------------------- #
tab_players, tab_how, tab_film, tab_recv, tab_story = st.tabs(
    ["🏆 Players", "📊 How it works", "🎬 Film room", "🏃 Receiver head", "📖 Story"]
)

# ===== TAB 1: PLAYERS =======================================================
with tab_players:
    board = result.get("leaderboard")
    if board is None or len(board) == 0:
        st.info("No leaderboard produced (leaderboard step may not have run).")
    else:
        st.subheader("Who protects the quarterback best?")
        st.caption(
            "Each lineman graded across every pass-block rep. **Win rate** is the "
            "share of blocks he won, corrected for how many reps we have on him. "
            "**Win rate (vs opp.)** adjusts for how good the rushers he faced were — "
            "higher than his raw win rate means he beat tough competition."
        )

        # ---- Filters: scope, team, position --------------------------------
        f1, f2, f3 = st.columns(3)
        scope = f1.selectbox("Scope", ["All games", "By team"], index=0,
                             help="Narrow the leaderboard to a single team's linemen.")
        teams = sorted(board["team"].dropna().unique().tolist()) \
            if "team" in board.columns else []
        team_pick = None
        if scope == "By team" and teams:
            team_pick = f2.selectbox("Team", teams)
        else:
            f2.selectbox("Team", ["(all teams)"], disabled=True)

        pos_opts = ["All positions"] + (
            sorted(board["blocker_pos"].dropna().unique().tolist())
            if "blocker_pos" in board.columns else []
        )
        pos_pick = f3.selectbox("Position", pos_opts, index=0)

        min_reps = st.slider("Minimum blocks graded", 1, 60, 10,
                             help="Hide small-sample players whose numbers are noisy.")

        view = board
        if "n_reps" in view.columns:
            view = view[view["n_reps"] >= min_reps]
        if team_pick is not None and "team" in view.columns:
            view = view[view["team"] == team_pick]
        if pos_pick != "All positions" and "blocker_pos" in view.columns:
            view = view[view["blocker_pos"] == pos_pick]

        display = rename_for_display(view, LEADERBOARD_LABELS)
        # Show win rates as percentages for readability.
        for col in ("Win rate", "Win rate (vs opp.)"):
            if col in display.columns:
                display[col] = (display[col] * 100).round(1)
        st.dataframe(
            display, use_container_width=True, height=520, hide_index=True,
            column_config={
                "Win rate": st.column_config.NumberColumn(format="%.1f%%"),
                "Win rate (vs opp.)": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        with st.expander("ℹ️ How to read this table"):
            st.markdown(
                "- **Win rate** — % of one-on-one blocks this lineman won. The headline "
                "number, already corrected so a player with very few reps isn't flattered.\n"
                "- **Win rate (vs opp.)** — the same, but crediting him more for beating "
                "strong rushers and less for beating weak ones.\n"
                "- **Blocks graded** — his sample size. Trust high-rep players more.\n\n"
                "A block is 'won' when the lineman kept the rusher away from the QB and "
                "stayed attached to him — measured purely from tracking geometry, then "
                "validated against real PFF pressure outcomes (see *How it works*)."
            )
            with st.expander("Advanced columns (margins, confidence range)"):
                adv_cols = [c for c in ["blocker_name", "mean_win_score",
                                        "adj_mean_win_score", "ci_low", "ci_high"]
                            if c in view.columns]
                st.caption("Control margin = how *decisively* he wins, not just how often. "
                           "CI = the range his true win rate likely falls in.")
                st.dataframe(view[adv_cols].rename(columns={
                    "blocker_name": "Player", "mean_win_score": "Control margin",
                    "adj_mean_win_score": "Control margin (vs opp.)",
                    "ci_low": "CI low", "ci_high": "CI high"}),
                    use_container_width=True, hide_index=True)

# ===== TAB 2: HOW IT WORKS (validation + method) ============================
with tab_how:
    st.subheader("How the score is built")

    st.markdown("""
**The unit is one rep: one blocker against the one rusher PFF says he was
assigned to.** We take both players' tracking from the snap to the moment the
ball leaves the QB's hand and compute a handful of geometric features that
describe pass protection:

- **Ground given up** — how much the rusher closed the distance to the QB over
  the rep. The core feature: a lost block is one where the rusher gets to the QB.
- **Closest the rusher got to the QB** — the minimum rusher-to-QB distance.
- **Separation** — the blocker-to-rusher gap (mean and minimum). Staying attached
  is good; a growing gap means the rusher is slipping the block.
- **Betweenness** — whether the blocker's body stays on the line between the
  rusher and the QB (1 = perfectly shielding, 0 = beaten to a side).
- **Mirroring** — how well the blocker's movement direction matches the rusher's.
- **Rusher speed / acceleration late in the rep** — a rusher still accelerating at
  the QB near the end is winning.

All features are signed so that **higher = the blocker won**.
""")

    st.markdown("""
**Turning features into one number.** We don't hand-pick weights for the final
grade. Instead a gradient-boosted tree model (sklearn `HistGradientBoostingClassifier`)
is trained to predict the real outcome — did this rep concede a hit, hurry, or
sack (PFF's `pressure_allowed`). The model's predicted pressure probability is
flipped into the **control grade**: high grade = low modelled chance of pressure
= the blocker won. Using the model, rather than fixed weights, lets the metric
learn interactions — e.g. giving up ground only matters when separation also
collapses — that a simple weighted sum would miss.
""")

    auc_v = validation.get("auc")
    if isinstance(auc_v, (int, float)):
        st.markdown(
            f"**Validation — AUC = {auc_v:.2f}.** Take one rep that gave up pressure and "
            "one that didn't; the tracking-only grade ranks the worse block lower "
            f"about **{auc_v*100:.0f}%** of the time (50% = a coin flip). The grade is "
            "built from geometry alone and never sees the PFF label, so this is a real "
            "check that it measures blocking."
        )
    else:
        st.info("Run the pipeline to compute validation.")

    # Charts, constrained to a sensible width (not full-page).
    cols = st.columns([1, 1])
    v = chart_path("fig_validation.png")
    if v:
        cols[0].image(str(v), caption="Grade vs real PFF pressure outcomes")
    d = chart_path("fig_score_distribution.png")
    if d:
        cols[1].image(str(d), caption="Clean blocks grade higher than pressure blocks")

    st.divider()
    st.markdown("**Which features drive the grade?**")
    st.caption("Permutation importance: how much the model's accuracy drops when each "
               "feature is scrambled. The taller bars are the geometry that most "
               "separates a won block from a lost one.")
    imp = result.get("feature_importance")
    if imp is not None and {"feature", "importance"}.issubset(getattr(imp, "columns", [])):
        # Keep the chart compact by placing it in a narrower column.
        ic = st.columns([2, 1])
        ic[0].bar_chart(imp.set_index("feature")["importance"], height=320)
    else:
        ip = chart_path("fig_feature_importance.png")
        if ip:
            st.columns([2, 1])[0].image(str(ip))

    st.markdown("""
**Opponent adjustment.** A win rate is inflated by feasting on weak rushers. We
estimate each rusher's difficulty from how blockers fare against him on average,
then re-center every rep by that rusher's effect. The *vs opp.* columns on the
Players tab use this adjusted grade, so beating a star edge rusher counts for
more than stonewalling a backup.
""")

# ===== TAB 3: FILM ROOM (pick a rep → see the visual) =======================
with tab_film:
    if viz is None or scored is None or len(scored) == 0:
        st.info("Run the pipeline first, then pick a block to study.")
    else:
        st.subheader("Study one block")
        st.caption("Pick a matchup to see the two-panel breakdown: the players' paths on "
                   "the field (top) and who was winning moment-to-moment (bottom). "
                   "● = snap, ✕ = end of rep.")

        paired = scored[scored.get("pairing_ok", True) == True] \
            if "pairing_ok" in scored.columns else scored
        if "win_score" in paired.columns:
            paired = paired.sort_values("win_score")

        col_f1, col_f2 = st.columns([3, 1])
        worst_first = col_f2.checkbox("Worst-blocked first", value=True)
        opts = (paired if worst_first else paired.iloc[::-1]).head(300).reset_index(drop=True)

        def _label(r):
            bn = r.get("blocker_name", r.get("blocker_id"))
            rn = r.get("rusher_name", r.get("rusher_id"))
            return f"{bn} vs {rn}  —  {rep_outcome(r)}  (game {int(r['game_id'])})"

        if len(opts) == 0:
            st.info("No paired reps to visualise.")
        else:
            labels = [_label(r) for _, r in opts.iterrows()]
            pick = col_f1.selectbox("Choose a block", range(len(labels)),
                                    format_func=lambda i: labels[i])
            row = opts.iloc[pick]
            try:
                fig = viz.plot_matchup(
                    int(row["game_id"]), int(row["play_id"]),
                    int(row["blocker_id"]), int(row["rusher_id"]),
                )
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not render this rep: {e}")

            with st.expander("🔧 Under the hood — the geometry for this block"):
                st.caption("The raw tracking measurements this block was graded on.")
                feat_row = {k: row.get(k) for k in REPS_LABELS if k in row.index}
                st.dataframe(
                    pd.DataFrame([feat_row]).rename(columns=REPS_LABELS),
                    use_container_width=True, hide_index=True,
                )

            st.divider()
            st.markdown("**Browse all blocks**")
            only_pressure = st.checkbox("Only blocks that gave up pressure", value=False)
            tbl = scored
            if only_pressure and "pressure_allowed" in tbl.columns:
                tbl = tbl[tbl["pressure_allowed"] == 1]
            if "win_score" in tbl.columns:
                tbl = tbl.sort_values("win_score")
            tbl = tbl.assign(outcome=tbl.apply(rep_outcome, axis=1))
            st.caption(f"{len(tbl):,} blocks (worst-graded first).")
            st.dataframe(rename_for_display(tbl, REPS_LABELS),
                         use_container_width=True, height=360, hide_index=True)


# --------------------------------------------------------------------------- #
# Receiver head — same engine, receiver vs coverage defender
# --------------------------------------------------------------------------- #
with tab_recv:
    if receiver is None or viz is None:
        st.info("receiver/viz modules not available.")
    else:
        st.subheader("Bonus: the same engine on receivers")
        st.caption("Receiver vs nearest coverage defender (man coverage). "
                   "The identical control engine, applied to route-running.")
        colg, colp = st.columns(2)
        g_in = colg.number_input("game_id", value=int(C.GOLDEN_GAME), step=1, format="%d")
        p_in = colp.number_input("play_id", value=int(C.GOLDEN_PLAY), step=1, format="%d")
        if st.button("Load receiver matchups"):
            try:
                rm = receiver.get_receiver_matchups(int(g_in), int(p_in))
                if len(rm) == 0:
                    st.info("No receiver matchups found for that play.")
                else:
                    st.caption(f"Coverage: {rm['coverage_type'].iloc[0]}")
                    show = ["blocker_name", "blocker_pos", "rusher_name",
                            "pairing_ok", "is_target"]
                    show = [c for c in show if c in rm.columns]
                    st.dataframe(rm[show], use_container_width=True)
                    paired_r = rm[rm["pairing_ok"] == True]
                    tgt = paired_r[paired_r.get("is_target", False) == True]
                    pick_row = (tgt.iloc[0] if len(tgt) else
                                (paired_r.iloc[0] if len(paired_r) else None))
                    if pick_row is not None:
                        fig = viz.plot_matchup(
                            int(pick_row["game_id"]), int(pick_row["play_id"]),
                            int(pick_row["blocker_id"]), int(pick_row["rusher_id"]),
                        )
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("No man-coverage pair to visualise on this play.")
            except Exception as e:
                st.error(f"Receiver head failed: {e}")


# --------------------------------------------------------------------------- #
# Story — the end-to-end narrative
# --------------------------------------------------------------------------- #
with tab_story:
    if story is None:
        st.info("story module not available.")
    else:
        st.subheader("The end-to-end story")
        st.caption("One battle → validation → the league. Builds figures and a "
                   "narrative tying the project together.")
        if st.button("Build story"):
            with st.spinner("Assembling the story…"):
                try:
                    out = story.build_story()
                    md = out.get("narrative_md", "")
                    # Render markdown, but swap image refs for absolute paths so
                    # Streamlit can display the figures.
                    st.markdown(md)
                    for name, path in out.get("figures", {}).items():
                        if C.CACHE_DIR.joinpath(path).exists() or __import__("os").path.exists(path):
                            st.image(path, caption=name, use_container_width=True)
                except Exception as e:
                    st.error(f"Story build failed: {e}")
