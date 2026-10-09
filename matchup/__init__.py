"""Matchup Control — a unified engine for grading one-on-one battles from NFL
tracking data.

Grades pass-protection reps geometrically and validates the per-rep "Matchup
Win Score" against PFF hurry/sack labels. See SHARED_CONTRACT.md for the
cross-team API (module layout, function signatures, dataframe schemas).

Workstreams: Person A owns the engine (io_load, pairing, features, score,
stunts, pipeline); Person B owns validation, leaderboard, evidence; Person C
owns visuals and the receiver head. The package is intentionally import-light
at the top level so ``import matchup`` succeeds before any heavy submodule.
"""

__version__ = "0.1.0"
