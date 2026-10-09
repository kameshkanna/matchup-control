# Matchup Control

<<<<<<< HEAD
We have grades every one-on-one pass-protection rep — one blocker
against the rusher PFF assigned him — using player tracking from the NFL Big
Data Bowl 2023 regional set. For each rep we measure the geometry of the block
(how much ground the rusher gained toward the QB, whether the blocker stayed
attached and between the rusher and the passer) and turn it into a single
control grade, which we validate against PFF's real hit/hurry/sack labels. A
Streamlit dashboard displays the results: a per-lineman leaderboard filterable
by team and position, a "how it works" page with the validation and feature
importance, a film-room view that draws any single block as field paths plus a
control-over-time curve, a team view, and a bonus receiver-vs-defender head. To
run it, `pip install -r requirements.txt`, make sure the data folder is in place
(see below), then `streamlit run matchup/app.py` from the repo root and click
**Run pipeline** in the sidebar.
=======
**A unified engine for grading one-on-one battles from NFL tracking data**
(NFL Big Data Bowl 2023 regional set — 122 games, 2021 Weeks 1-8, 10 Hz tracking).
>>>>>>> e26ce1c (fix: Rewite README,md)

Football looks like a team game, but it's really thousands of one-on-one battles.
On every dropback each blocker is matched to one rusher, and the only question that
matters is: **who controlled whom?** Matchup Control answers that geometrically from
tracking data, scores every rep, and validates the score against real PFF pressure
outcomes.

- **Primary head — pass protection** (blocker vs rusher, QB as the protected point).
  Pairing comes from PFF's `pff_nflIdBlockedPlayer`; the score is validated against the
  real hit/hurry/sack labels.
- **Bonus head — receiver vs coverage** (receiver vs nearest man-coverage defender).
  The *identical* engine, reused with no retraining. No ground-truth label here, so it's
  flagged as a bonus, not a headline claim.

See `SHARED_CONTRACT.md` for the cross-team code API (module layout, function signatures,
dataframe schemas) and `PRESENTATION.md` for the demo script.

## What we built

The pipeline turns raw tracking into a validated, opponent-adjusted leaderboard:

```
raw CSVs
  → normalise direction + rep window        (io_load, pairing)
  → pair blocker↔rusher, sync trajectories  (pairing, features)
  → control features over the rep           (features)
  → per-rep Matchup Win Score               (score, gradient-boosted on PFF labels)
  → validate vs hit/hurry/sack labels        (validate)   ·  AUC ≈ 0.90
  → opponent-quality adjustment              (leaderboard)
  → shrinkage-corrected player leaderboard   (leaderboard)
```

**Headline results (15-game slice):** 5,782 scored reps · 95.5% pairing success ·
**AUC 0.90** of the geometry-only score against PFF pressure · top credible pass
protectors are centers (Creed Humphrey, Corey Linsley, ~95% win rate).

## Setup

```bash
pip install -r requirements.txt
```

## Data (NOT in this repo — 833 MB)

Copy the dataset so the data folder sits next to `matchup/`:

```
<repo-root>/
├─ matchup/                              # the code (in git)
└─ nfl-big-data-bowl-regional-event-data/
   └─ data/
      ├─ games.csv  plays.csv  players.csv  pffScoutingData.csv
      └─ tracking/tracking_<gameId>.csv   (122 files)
```

Paths are resolved in `matchup/config.py` — nothing is hardcoded elsewhere.

## See the graphs — two ways

### 1. Interactive dashboard (UI) — `matchup/app.py`

A Streamlit app with everything clickable.

```bash
pip install streamlit                 # if not already installed
streamlit run matchup/app.py          # launch from the repo root
```

Click **Run pipeline (all games)** once; results cache, so every tab is then instant.
Five tabs:

- **🏆 Players** — the opponent-adjusted, shrinkage-corrected blocker leaderboard, with
  team/position filters and a min-reps slider.
- **📊 How it works** — the method, the validation chart, and feature importance.
- **🎬 Film room** — pick any block and see the two-panel signature visual (field paths
  + control-over-time curve), with a plain-English story written from that rep's geometry.
- **🤝 Team view** — the match read *as a team*: individual scores aggregated per team,
  per game or across all games (e.g. TB 90% vs DAL 77% in Week 1).
- **🏃 Receiver head** — the same engine on receiver-vs-defender, selected by searchable
  **team → game → play** dropdowns (man-coverage plays only).

### 2. Static figures — `inference.py` (no UI needed)

Prefer not to run the web app? `inference.py` regenerates every graph as PNGs and prints
a plain-language description of what each shows.

```bash
python inference.py              # golden rep + validation + leaderboard + story
python inference.py --receiver   # also render the receiver head
python inference.py --game 2021090900 --play 97 --blocker 42377   # any rep
```

Figures are written to `cache/inference/` (git-ignored). Open the PNGs directly, e.g.
`open cache/inference/matchup_2021090900_97_42377.png`.

## Quick check

```bash
python -m matchup.score      # scores the golden play (fast)
python run_demo.py 5         # scores 5 games, prints validation + leaderboard
```

## Tests

End-to-end tests validate the scoreboard and model performance against the **real**
pipeline (not synthetic): label consistency, score separation, AUC, and leaderboard
sanity.

```bash
./run_tests.sh               # finds the venv automatically
# or:  python -m pytest
```

<<<<<<< HEAD
Opens in the browser. Click **Run pipeline (all games)** in the sidebar (a cold
run takes a while; results are then cached). Tabs: **Players** (leaderboard,
filter by team/position), **How it works** (validation + feature importance),
**Film room** (pick a block, see its field + control-over-time visual), **Team
view**, **Receiver head**, and **Story**.

=======
>>>>>>> e26ce1c (fix: Rewite README,md)
## Who owns what (workstreams)

- **Person A** — engine/metric: `io_load`, `pairing`, `features`, `score`, `stunts`, `pipeline`
- **Person B** — validation/leaderboard: `validate`, `leaderboard`, `evidence`
- **Person C** — visuals, story, receiver head, labels: `viz`, `receiver`, `story`,
  `labels`, `inference`, dashboard tabs for Film room / Team view / Receiver head, and the
  `tests/`

> `matchup/_upstream.py` is a thin adapter Person C's modules use to reach the engine; it
> prefers the real `matchup.*` modules and only falls back to a local implementation if
> they're unavailable, so the visuals could be built before the engine landed.
