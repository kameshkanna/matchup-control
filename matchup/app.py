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


def load_cached_leaderboard(n_games: int):
    """Try to load a previously-saved leaderboard without recomputing."""
    return C.cache_read(f"leaderboard_{n_games}games")


def chart_path(name: str):
    p = C.CACHE_DIR / name
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("Matchup Control")
st.sidebar.caption("Grading one-on-one pass-protection battles from tracking data")

n_games = st.sidebar.slider("Games to analyse", min_value=1, max_value=122, value=5, step=1)
run_clicked = st.sidebar.button("Run pipeline", type="primary")
st.sidebar.caption(
    "Loads cached results instantly. Click **Run pipeline** to (re)compute — "
    "that can take a while for many games."
)

# Decide what to show: a fresh run, or whatever is cached.
result = None
if run_clicked:
    with st.spinner(f"Running pipeline on {n_games} games…"):
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
        "No results loaded yet. Pick the number of games in the sidebar and "
        "click **Run pipeline**. Showing any cached leaderboard below if present."
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
# Tabs: Leaderboard · Charts · Reps explorer · What predicts a loss
# --------------------------------------------------------------------------- #
tab_board, tab_charts, tab_reps, tab_imp, tab_play, tab_recv, tab_story = st.tabs(
    ["🏆 Leaderboard", "📊 Charts", "🔍 Reps explorer", "🧠 What predicts a loss",
     "🎯 Play visual", "🏃 Receiver head", "📖 Story"]
)

with tab_board:
    board = result.get("leaderboard")
    if board is None or len(board) == 0:
        st.info("No leaderboard produced (leaderboard step may not have run).")
    else:
        st.subheader("Blocker leaderboard")
        st.caption("Offensive linemen only, ranked by opponent-adjusted score.")
        min_reps = st.slider("Minimum reps", 1, 60, 10)
        if "n_reps" in board.columns:
            view = board[board["n_reps"] >= min_reps]
        else:
            view = board
        st.dataframe(view, use_container_width=True, height=520)

with tab_charts:
    charts = [
        ("fig_score_distribution.png", "Win-score distribution: clean vs pressure reps"),
        ("fig_feature_importance.png", "What separates a won block from a lost one"),
        ("fig_validation.png", "Validation vs PFF labels (ROC + score separation)"),
    ]
    any_chart = False
    for fname, caption in charts:
        p = chart_path(fname)
        if p is not None:
            st.image(str(p), caption=caption, use_container_width=True)
            any_chart = True
    if not any_chart:
        st.info("No charts found yet — run the pipeline to generate them.")

with tab_reps:
    if scored is None or len(scored) == 0:
        st.info("No scored reps available.")
    else:
        st.subheader("Per-rep explorer")
        cols_pref = [
            "game_id", "play_id", "blocker_name", "blocker_pos", "rusher_name",
            "block_type", "pressure_allowed", "win_score", "win_flag",
            "ground_given_up", "sep_min", "min_rusher_to_qb_dist",
        ]
        show_cols = [c for c in cols_pref if c in scored.columns]
        only_pressure = st.checkbox("Only reps where pressure was allowed", value=False)
        view = scored
        if only_pressure and "pressure_allowed" in view.columns:
            view = view[view["pressure_allowed"] == 1]
        if "win_score" in view.columns:
            view = view.sort_values("win_score")
        st.caption(f"{len(view):,} reps (sorted worst-blocked first).")
        st.dataframe(view[show_cols] if show_cols else view,
                     use_container_width=True, height=520)

with tab_imp:
    imp = result.get("feature_importance")
    if imp is None or len(imp) == 0:
        st.info("No feature-importance table available.")
    else:
        st.subheader("What predicts a lost block")
        st.caption("Permutation importance: how much each tracking feature "
                   "matters for predicting a conceded pressure.")
        st.dataframe(imp, use_container_width=True)
        if {"feature", "importance"}.issubset(imp.columns):
            st.bar_chart(imp.set_index("feature")["importance"])


# --------------------------------------------------------------------------- #
# Play visual — the signature two-panel figure for one rep
# --------------------------------------------------------------------------- #
with tab_play:
    if viz is None:
        st.info("viz module not available.")
    elif scored is None or len(scored) == 0:
        st.info("Run the pipeline first to pick a rep.")
    else:
        st.subheader("One rep: field paths + control over time")
        st.caption("Pick a blocker-vs-rusher rep. ● = snap, ✕ = end of rep. "
                   "On a lost rep the rusher→QB distance (red, bottom) collapses.")

        paired = scored[scored.get("pairing_ok", True) == True] \
            if "pairing_ok" in scored.columns else scored

        # Default to the worst-blocked (lowest win_score) rep for drama.
        default_idx = 0
        if "win_score" in paired.columns and len(paired):
            paired = paired.sort_values("win_score")

        # Build human-readable options: "blocker vs rusher — game/play (outcome)"
        def _label(r):
            out = "clean"
            if r.get("sack_allowed"): out = "SACK"
            elif r.get("hit_allowed"): out = "HIT"
            elif r.get("hurry_allowed"): out = "HURRY"
            bn = r.get("blocker_name", r.get("blocker_id"))
            rn = r.get("rusher_name", r.get("rusher_id"))
            return f"{bn} vs {rn} — {int(r['game_id'])}/{int(r['play_id'])} ({out})"

        opts = paired.head(300).reset_index(drop=True)
        if len(opts) == 0:
            st.info("No paired reps to visualise.")
        else:
            labels = [_label(r) for _, r in opts.iterrows()]
            pick = st.selectbox("Choose a rep (worst-blocked first)", range(len(labels)),
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
