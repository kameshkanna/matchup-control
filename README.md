# Matchup Control

A unified engine for grading one-on-one battles from NFL tracking data
(NFL Big Data Bowl 2023 regional set). Primary head: pass protection
(blocker vs rusher), validated against PFF pressure labels.

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

## Who owns what (workstreams)

- **Person A** — engine/metric: `io_load`, `pairing`, `features`, `score`, `stunts`, `pipeline`
- **Person B** — validation/leaderboard: `validate`, `leaderboard`, `evidence`
- **Person C** — visuals/story + receiver head: `viz`, `receiver`, story
