"""Shared paths and constants. Person A owns this; everyone imports from it.

No hardcoded absolute paths anywhere else in the codebase — read from here.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at  <workspace>/matchup/config.py
# so the workspace root is two parents up.
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = ROOT_DIR / "nfl-big-data-bowl-regional-event-data" / "data"
TRACKING_DIR = DATA_DIR / "tracking"
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Field / tracking constants
# ---------------------------------------------------------------------------
FIELD_LEN = 120.0   # x axis, yards (includes both 10-yard end zones)
FIELD_WID = 53.3    # y axis, yards
HZ = 10             # tracking frequency (frames per second)

BALL_ID = -1        # we fill ball rows (raw nflId is NA) with this sentinel

# ---------------------------------------------------------------------------
# Event tags
# ---------------------------------------------------------------------------
SNAP_EVENTS = ("ball_snap",)
# Rep ends at the first of these (ball leaves QB hand / play breaks down).
REP_END_EVENTS = (
    "pass_forward",
    "autoevent_passforward",
    "qb_sack",
    "qb_strip_sack",
    "pass_shovel",
    "run",            # scramble handoff point — rep of protection effectively ends
    "pass_tipped",
)

# ---------------------------------------------------------------------------
# PFF role labels
# ---------------------------------------------------------------------------
ROLE_BLOCK = "Pass Block"
ROLE_RUSH = "Pass Rush"
ROLE_PASS = "Pass"          # the QB / passer
ROLE_ROUTE = "Pass Route"
ROLE_COVERAGE = "Coverage"

# Stunt-relevant block types (pff_blockType)
BLOCKTYPE_SWITCH = "SW"
BLOCKTYPE_CHIP = "CH"

# ---------------------------------------------------------------------------
# The golden test rep (everyone validates against this)
# ---------------------------------------------------------------------------
GOLDEN_GAME = 2021090900
GOLDEN_PLAY = 97
GOLDEN_BLOCKER = 42377   # LT
GOLDEN_RUSHER = 42403    # ROLB — sack_allowed = 1


# ---------------------------------------------------------------------------
# Cache helpers — use parquet if pyarrow/fastparquet is available, else CSV.
# This means NO ONE is ever blocked by a missing parquet engine.
# ---------------------------------------------------------------------------
def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


HAS_PARQUET = _has_parquet()


def cache_path(stem: str):
    """Return the cache file path for a given stem, with the best available
    extension (.parquet if possible, else .csv)."""
    ext = "parquet" if HAS_PARQUET else "csv"
    return CACHE_DIR / f"{stem}.{ext}"


def cache_write(df, stem: str) -> None:
    import pandas as pd  # local import to keep config import-light
    p = cache_path(stem)
    if HAS_PARQUET:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def cache_read(stem: str):
    import pandas as pd
    p = cache_path(stem)
    if not p.exists():
        return None
    if HAS_PARQUET:
        return pd.read_parquet(p)
    return pd.read_csv(p)
