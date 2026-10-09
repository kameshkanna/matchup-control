#!/usr/bin/env bash
# Run the Matchup Control test suite using the project's virtualenv.
#
# Why this exists: on this machine a bare `pytest` / `python` is not on PATH
# (only `python3` is), so running `pytest` directly fails with
# "command not found" — which looks like a missing-module error. This wrapper
# always calls the venv's interpreter, so you never have to activate anything
# or remember the venv path.
#
# Usage:
#   ./run_tests.sh                 # run the whole suite
#   ./run_tests.sh -k leaderboard  # pass any pytest args through
#   ./run_tests.sh tests/test_scoreboard.py::test_winscore_auc_beats_chance

set -euo pipefail

# Directory this script lives in (the repo root), regardless of where it's called from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the venv python: prefer ../.venv (sibling of the repo), then ./.venv.
if [[ -x "$ROOT/../.venv/bin/python" ]]; then
  PY="$ROOT/../.venv/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  echo "error: could not find the project virtualenv (.venv)." >&2
  echo "       expected at $ROOT/../.venv or $ROOT/.venv" >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" -m pytest "$@"
