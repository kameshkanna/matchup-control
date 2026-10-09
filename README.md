# Matchup Control

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

See `SHARED_CONTRACT.md` for the cross-team code API (module layout,
function signatures, dataframe schemas). **Read it before writing code.**

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
The dataset's own column dictionary ships inside that data folder.

## Quick check

```bash
python -m matchup.score      # scores the golden play (fast)
python run_demo.py 5         # scores 5 games, prints validation + leaderboard
```

The demo writes `cache/scored_demo.csv`.

## Dashboard (UI)

```bash
pip install streamlit        # if not already installed
streamlit run matchup/app.py # launch from the repo root
```

Opens in the browser. Click **Run pipeline (all games)** in the sidebar (a cold
run takes a while; results are then cached). Tabs: **Players** (leaderboard,
filter by team/position), **How it works** (validation + feature importance),
**Film room** (pick a block, see its field + control-over-time visual), **Team
view**, **Receiver head**, and **Story**.

## Who owns what (workstreams)

- **Person A** — engine/metric: `io_load`, `pairing`, `features`, `score`, `stunts`, `pipeline`
- **Person B** — validation/leaderboard: `validate`, `leaderboard`, `evidence`
- **Person C** — visuals/story + receiver head: `viz`, `receiver`, story
