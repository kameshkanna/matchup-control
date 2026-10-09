# Matchup Control — Shared Code Contract

**Read this before writing any code.** This is the agreement between the 3 laptops.
Depend on each other's **function signatures** and **dataframe columns** — never on
internals. If everyone honors the names below, merging is copy-paste, not debugging.

---

## 0. Project layout (one package, one module per concern)

```
NFL Hackathon/
├─ nfl-big-data-bowl-regional-event-data/data/   # the raw data (already here)
├─ matchup/                      # the shared package — EVERYONE imports from here
│  ├─ __init__.py
│  ├─ config.py                  # paths, constants (Person A owns)
│  ├─ io_load.py                 # Task 1  — loaders + direction normalisation (A)
│  ├─ pairing.py                 # Task 2  — rep windows + matchup pairing (A)
│  ├─ features.py                # Task 3  — trajectories + control features (A)
│  ├─ score.py                   # Task 4  — Matchup Win Score (A)
│  ├─ stunts.py                  # Task 7  — stunt/chip/double-team handling (A)
│  ├─ pipeline.py                # Task 9  — end-to-end wiring (A)
│  ├─ validate.py                # Task 6  — label validation / calibration (B)
│  ├─ leaderboard.py             # Task 8  — opponent adjustment + ranking (B)
│  ├─ evidence.py                # Task 10 — population charts (B)
│  ├─ viz.py                     # Task 5  — plot_matchup signature visual (C)
│  ├─ receiver.py                # Task 11 — bonus receiver head (C)
│  └─ story.py / notebook        # Task 12 — narrative assembly (C)
└─ cache/                        # parquet intermediates (gitignored-ish)
```

Rule: **all cross-person imports go `from matchup.<module> import <fn>`.**
Nobody imports from a file sitting on their own laptop by a different name.

---

## 1. Canonical identifiers & constants (`config.py`)

Everyone uses these exact names. No synonyms.

| name | meaning |
|------|---------|
| `game_id`   | int, the gameId |
| `play_id`   | int, the playId |
| `nfl_id`    | int, player id (ball = `-1`, NOT NaN — we fill it) |
| `frame_id`  | int, 1-based frame |
| `FIELD_LEN = 120.0` | x axis length (yards) |
| `FIELD_WID = 53.3`  | y axis width (yards) |
| `HZ = 10` | tracking frequency |
| `DATA_DIR` | Path to `.../data` |
| `CACHE_DIR`| Path to `./cache` |

**Snake_case everywhere.** We rename the raw CSV camelCase columns ONCE, inside
`io_load`, to the names in the schemas below. After that, no camelCase in the codebase.

---

## 2. Function signatures (the public API — do not change these names/args)

### Task 1 — `matchup/io_load.py` (Person A)
```python
def load_games() -> pd.DataFrame: ...
def load_plays() -> pd.DataFrame: ...
def load_players() -> pd.DataFrame: ...
def load_pff() -> pd.DataFrame: ...
def load_tracking(game_id: int) -> pd.DataFrame:
    """One game's tracking, direction-normalised, snake_case, ball nfl_id=-1."""

def get_play(game_id: int, play_id: int) -> pd.DataFrame:
    """Merged tracking + player roles for ONE play, direction-normalised.
    This is THE entry point everyone uses to get a play's frames."""

def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip x->120-x, y->53.3-y, and reflect dir/o by 180 where play_direction=='left',
    so offense ALWAYS moves toward +x. Adds no new rows."""
```

### Task 2 — `matchup/pairing.py` (Person A)
```python
def get_rep_window(play_frames: pd.DataFrame) -> tuple[int, int]:
    """Return (snap_frame_id, end_frame_id). end = first of
    pass_forward / autoevent_passforward / qb_sack / pass_release."""

def get_matchups(game_id: int, play_id: int) -> pd.DataFrame:
    """One row per blocker->rusher matchup on the play. See SCHEMA: matchups."""

def get_all_matchups(game_ids: list[int]) -> pd.DataFrame:
    """get_matchups over many games, concatenated. Cached to parquet."""
```

### Task 3 — `matchup/features.py` (Person A)
```python
def get_rep_tracks(game_id: int, play_id: int, blocker_id: int,
                   rusher_id: int) -> pd.DataFrame:
    """Frame-aligned blocker/rusher/qb trajectories over the rep. SCHEMA: rep_tracks."""

def control_timeseries(rep_tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-frame control signals (for the viz bottom panel). SCHEMA: control_ts."""

def rep_features(game_id: int, play_id: int, blocker_id: int,
                 rusher_id: int) -> dict:
    """Collapse one rep to a flat dict of features. SCHEMA: features (keys)."""
```

### Task 4 — `matchup/score.py` (Person A)
```python
def win_score(features: dict, weights: dict | None = None) -> float:
    """Single continuous score, higher = blocker won. Explainable in one sentence."""

def win_flag(score: float, threshold: float = 0.0) -> int:
    """1 = blocker won the rep, 0 = lost."""

def score_matchups(matchups_with_features: pd.DataFrame) -> pd.DataFrame:
    """Adds columns win_score, win_flag. SCHEMA: scored."""

DEFAULT_WEIGHTS: dict   # module-level, overwritten by calibration in Task 6
```

### Task 7 — `matchup/stunts.py` (Person A)
```python
def tag_stunts(matchups: pd.DataFrame, play_frames: pd.DataFrame) -> pd.DataFrame:
    """Adds/updates is_switch, is_chip, is_double, passed_off (bool) columns."""
def resolve_attribution(scored: pd.DataFrame) -> pd.DataFrame:
    """Reassign/segment credit on crossed reps. Adds attribution_note (str)."""
```

### Task 6 — `matchup/validate.py` (Person B)
```python
def calibrate_weights(scored_or_features: pd.DataFrame) -> dict:
    """Fit logistic model: features -> pressure_allowed label. Returns weights dict
    (same keys as features) to feed back into score.win_score."""
def validation_report(scored: pd.DataFrame) -> dict:
    """Returns {'auc':..., 'corr':..., 'n':...} of win_score vs pressure_allowed."""
```

### Task 8 — `matchup/leaderboard.py` (Person B)
```python
def rusher_strength(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-rusher difficulty. SCHEMA: rusher_strength."""
def opponent_adjust(scored: pd.DataFrame) -> pd.DataFrame:
    """Adds adj_win_score column (opponent-quality adjusted)."""
def player_leaderboard(scored_adj: pd.DataFrame) -> pd.DataFrame:
    """Per-blocker aggregate with shrinkage. SCHEMA: leaderboard."""
```

### Task 10 — `matchup/evidence.py` (Person B)
```python
def plot_score_distribution(scored: pd.DataFrame): ...
def plot_headline_comparison(scored: pd.DataFrame, by: str): ...
```

### Task 5 — `matchup/viz.py` (Person C)
```python
def draw_field(ax=None): ...
def plot_matchup(game_id: int, play_id: int, blocker_id: int,
                 rusher_id: int | None = None):
    """Two-panel: field (paths/arrows/LOS/events) + control-over-time curve.
    Consumes get_rep_tracks() and control_timeseries()."""
```

### Task 11 — `matchup/receiver.py` (Person C)
```python
def get_receiver_matchups(game_id: int, play_id: int) -> pd.DataFrame:
    """Same schema shape as matchups but receiver vs coverage defender,
    man-coverage-scoped. Reuses features/score/viz unchanged."""
def parse_target(play_description: str) -> str | None:
    """'...to C.Godwin...' -> 'C.Godwin' or None."""
```

---

## 3. Dataframe column schemas (FROZEN — the real integration contract)

If two people's dataframes share these columns, their code composes. Add extra
columns freely, but **never rename or drop** a column listed here.

### SCHEMA: tracking (output of `load_tracking` / `get_play`)
```
game_id, play_id, nfl_id, frame_id, time, jersey_number, team,
play_direction, x, y, s, a, dis, o, dir, event,
display_name, position, pff_role, pff_position_lined_up   # roles joined in get_play
```
- `nfl_id == -1` for the ball.
- after normalisation, offense moves toward +x; `x,y,dir,o` are the normalised values.

### SCHEMA: matchups (output of `get_matchups`)
```
game_id, play_id,
blocker_id, blocker_name, blocker_pos,          # the OL (pff_role == 'Pass block')
rusher_id,  rusher_name,  rusher_pos,            # pff_nflIdBlockedPlayer target
qb_id,                                           # the passer (pff_role == 'Pass')
snap_frame, end_frame,
block_type,                                      # pff_blockType (PP, SW, CH, PT, ...)
is_switch, is_chip, is_double,                   # bool flags
pressure_allowed,                                # int label: 1 if hurry/hit/sack allowed
sack_allowed, hurry_allowed, hit_allowed,        # the raw PFF labels (0/1)
pairing_ok                                        # bool: assignment resolved cleanly
```
- `pressure_allowed = max(hit_allowed, hurry_allowed, sack_allowed)` — the Task 6 label.

### SCHEMA: rep_tracks (output of `get_rep_tracks`)
```
frame_id, t_sec,                                 # t_sec = (frame_id - snap_frame)/HZ
blk_x, blk_y, blk_s, blk_dir,                    # blocker
rsh_x, rsh_y, rsh_s, rsh_a, rsh_dir,             # rusher
qb_x, qb_y                                        # reference point
```

### SCHEMA: control_ts (output of `control_timeseries`)
```
frame_id, t_sec,
rusher_to_qb_dist,          # distance rusher -> qb (shrinks = losing)
blocker_rusher_sep,         # gap between blocker and rusher
betweenness,                # 1 = blocker perfectly between rusher and qb, 0 = beaten
mirroring                   # cosine(blocker_dir, rusher_dir), 1 = mirrored
```

### SCHEMA: features (keys of `rep_features` dict)
```
game_id, play_id, blocker_id, rusher_id,
ground_given_up,            # start rusher_to_qb_dist - min rusher_to_qb_dist
min_rusher_to_qb_dist,
penetration_past_los,       # max rusher x beyond LOS (normalised frame)
sep_mean, sep_min,          # blocker_rusher_sep stats
betweenness_mean, betweenness_end,
mirroring_mean,
rusher_speed_late, rusher_accel_late,   # mean over final 0.5s
rep_duration_sec
```

### SCHEMA: scored (output of `score_matchups`)
```
<all matchups columns> + <all features keys> + win_score, win_flag
```

### SCHEMA: rusher_strength (`rusher_strength`)
```
rusher_id, rusher_name, n_reps, pressure_rate, strength   # strength: higher = harder
```

### SCHEMA: leaderboard (`player_leaderboard`)
```
blocker_id, blocker_name, blocker_pos, team,
n_reps, win_rate, adj_win_rate, win_rate_shrunk,
mean_win_score, adj_mean_win_score, ci_low, ci_high
```

---

## 4. Conventions (prevents the silly merge bugs)

1. **Toward the QB = negative change in `rusher_to_qb_dist`.** Blocker winning
   keeps this distance LARGE. Everyone signs features so **higher = blocker won**.
2. **Ball is `nfl_id == -1`**, never NaN. `load_tracking` fills this.
3. **Direction is normalised exactly once**, in `load_tracking`. Downstream code
   assumes offense moves +x and NEVER re-flips.
4. **Caching:** `get_all_matchups`, `score_matchups` write parquet to `CACHE_DIR`
   with deterministic filenames (e.g. `cache/matchups_<n>games.parquet`). Reads
   check cache first. Lets people work without rerunning each other's slow steps.
5. **Missing/edge reps:** keep the row, set `pairing_ok=False`. Never silently drop.
6. **No hardcoded absolute paths in functions** — read from `config.DATA_DIR`.
7. **Each module has an `if __name__ == '__main__':` smoke test** that runs its
   own demo on game 2021090900, play 97 (the known LT 42377->rusher 42403
   sack-allowed rep) so anyone can verify their piece in isolation.
8. Return **DataFrames/dicts**, not printed output. Print only in `__main__`.

---

## 5. The golden test row (everyone validates against this)

```
game_id=2021090900, play_id=97
blocker_id=42377 (LT), rusher_id=42403 (ROLB), sack_allowed=1
```
Expect: high `ground_given_up`, low `betweenness_end`, `win_flag == 0`.
If your module disagrees on this rep, it's wrong — fix before merging.
