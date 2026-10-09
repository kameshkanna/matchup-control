"""Matchup Control — a unified engine for grading one-on-one battles from NFL tracking data.

This package grades pass-protection reps geometrically and validates the resulting
per-rep "Matchup Win Score" against PFF hurry/sack labels. Person B owns the
validation, leaderboard, and evidence (visual) modules plus the shared ``config``.

The package is intentionally import-light at the top level so that
``import matchup`` succeeds before any heavy submodule is loaded.
"""
