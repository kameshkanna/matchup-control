# Verification — Person B modules (validate / leaderboard / evidence + config)

First iteration (no `review.json` present). All commands run from the workspace root
`c:\Users\Kamesh\aws\nfl-big-data-bowl-regional-event-data`.

## 1. Dependencies

No install needed — all deps already present (reported via `python -c "import ..."`):

| package | version |
|---|---|
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| matplotlib | 3.11.1 |
| scikit-learn (sklearn) | 1.9.0 |
| pyarrow | 25.0.1 |

Python 3.12.10. Nothing was `pip install`ed.

## 2. Modules run as scripts (all exit 0, demos print)

- `python -m matchup.config`     → exit 0; synthetic scored shape `(401, 34)`; golden row `win_flag == 0`.
- `python -m matchup.validate`   → exit 0.
- `python -m matchup.leaderboard`→ exit 0 (confirmed with output redirected; a `-1` seen earlier was a PowerShell broken-pipe from `Select-Object -First 3`, not a module error).
- `python -m matchup.evidence`   → exit 0.

## 3. validation_report numbers (synthetic, n_reps=400, seed=0)

```
validation_report: {'auc': 0.8617777896166858, 'corr': -0.6331739765790829, 'n': 390}
```
- AUC = **0.862** (> 0.5, exceeds the > 0.7 target).
- corr = **-0.633** (negative, as required — higher win_score => less pressure).
- n = **390** reps used (paired, finite rows out of 401; the ~11 dropped are `pairing_ok == False` reps, honestly excluded).

Calibrated raw-feature weights (predict *pressure*; negate to feed win_score) printed in the demo,
e.g. `ground_given_up +0.156`, `sep_min -0.923`, `mirroring_mean -1.114`, `intercept +1.127`.

## 4. player_leaderboard schema + shrinkage invariant

All 12 leaderboard SCHEMA columns present: `blocker_id, blocker_name, blocker_pos, team,
n_reps, win_rate, adj_win_rate, win_rate_shrunk, mean_win_score, adj_mean_win_score, ci_low, ci_high`.

Assertion in the demo passes: every `win_rate_shrunk` lies between the row's raw `win_rate`
and the global mean (global rate = 0.500). Example rows:

```
 blocker_id  blocker_name blocker_pos team  n_reps  win_rate  adj_win_rate  win_rate_shrunk  mean_win_score  adj_mean_win_score   ci_low  ci_high
      48200  Blocker48200          LG   TB      48  0.583333      0.583333         0.570032        0.380452            0.416527 0.442813 0.711503
      53900  Blocker53900          RT   KC      51  0.529412      0.529412         0.524952        0.338038            0.304471 0.395233 0.659470
      47800  Blocker47800          LT   KC      57  0.526316      0.526316         0.522687        0.339306            0.336352 0.399180 0.650128
      42377 Tristan Wirfs          LT   TB       1  0.000000      1.000000         0.450576       -2.500000            0.181347 0.000000 0.793451
```
The golden blocker (42377, 1 rep) is correctly shrunk from raw 0.0 toward the global 0.5 (0.451).

## 5. Golden synthetic row

`scored[(blocker_id==42377) & (rusher_id==42403)]` has `win_flag == 0` (blocker loss),
`sack_allowed == 1`, `pressure_allowed == 1`. Asserted in both `config` and `leaderboard` demos.
The real golden ids were removed from the random id pools so only the injected row matches.

## 6. PNGs written to cache/

```
cache/validation.png            53509 bytes   ROC (-win_score vs pressure) + win_score boxplots by outcome
cache/score_distribution.png    27160 bytes   win_score histograms overlaid by pressure_allowed (0 vs 1)
cache/headline_rusher_pos.png   25613 bytes   win_rate by rusher_pos with 95% Wilson CI + sample sizes
```

## 7. Raw data untouched

`git status --short data` → no output (no files under `data/` modified).

## Files created

- `matchup/__init__.py` — package docstring, import-light.
- `matchup/config.py` — constants (FIELD_LEN/FIELD_WID/HZ), DATA_DIR/CACHE_DIR (from `__file__`), frozen schema lists, `_make_synthetic_scored` shared generator with the golden row.
- `matchup/validate.py` — `DEFAULT_FEATURE_KEYS`, `calibrate_weights`, `validation_report`, `plot_validation` (Task 6).
- `matchup/leaderboard.py` — `rusher_strength`, `opponent_adjust`, `player_leaderboard` with EB shrinkage + Wilson CI (Task 8).
- `matchup/evidence.py` — `plot_score_distribution`, `plot_headline_comparison` (Task 10).
