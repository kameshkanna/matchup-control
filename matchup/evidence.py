"""Task 10 — evidence visuals that tell the Matchup Control story.

Two presentation-ready figures:

* :func:`plot_score_distribution` — the ``win_score`` distribution split by
  ``pressure_allowed`` (0 vs 1), showing that conceded-pressure reps sit at lower
  win scores than clean reps.
* :func:`plot_headline_comparison` — grouped ``win_rate`` with 95% Wilson CI error
  bars and sample sizes, broken out by any categorical column (e.g. ``rusher_pos``
  or ``block_type``).

Both return a :class:`matplotlib.figure.Figure`; neither shows nor prints.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matchup.config import _make_synthetic_scored
from matchup.leaderboard import _wilson_interval


def plot_score_distribution(scored: pd.DataFrame) -> matplotlib.figure.Figure:
    """Plot the ``win_score`` distribution overlaid by ``pressure_allowed``.

    Two translucent histograms are overlaid — clean reps (``pressure_allowed == 0``)
    and pressure reps (``pressure_allowed == 1``) — with their medians marked. A
    working score shows the pressure distribution shifted toward lower win scores.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame containing ``win_score`` and ``pressure_allowed``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ws = pd.to_numeric(scored["win_score"], errors="coerce")
    y = pd.to_numeric(scored["pressure_allowed"], errors="coerce")
    mask = np.isfinite(ws) & np.isfinite(y)
    ws = ws[mask]
    y = y[mask].astype(int)

    clean = ws[y == 0].to_numpy()
    pressure = ws[y == 1].to_numpy()

    if ws.size:
        bins = np.linspace(float(ws.min()), float(ws.max()), 30)
    else:
        bins = 30

    if clean.size:
        ax.hist(clean, bins=bins, alpha=0.55, color="C0", label=f"clean (n={clean.size})")
        ax.axvline(np.median(clean), color="C0", ls="--", lw=1.5)
    if pressure.size:
        ax.hist(pressure, bins=bins, alpha=0.55, color="C3", label=f"pressure (n={pressure.size})")
        ax.axvline(np.median(pressure), color="C3", ls="--", lw=1.5)

    ax.set_xlabel("win_score  (higher = blocker won)")
    ax.set_ylabel("rep count")
    ax.set_title("Win Score distribution by outcome")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_headline_comparison(scored: pd.DataFrame, by: str) -> matplotlib.figure.Figure:
    """Plot blocker ``win_rate`` grouped by a categorical column with 95% CI bars.

    Each bar is the mean ``win_flag`` within a level of ``by``; error bars are the
    95% Wilson score interval (matching the leaderboard's CI choice because it stays
    in ``[0, 1]`` and behaves for small samples), and each bar is annotated with its
    sample size ``n``.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame containing ``win_flag`` and the ``by`` column.
    by:
        Name of the categorical column to group on (e.g. ``"rusher_pos"``,
        ``"block_type"``).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``by`` is not a column of ``scored``.
    """
    if by not in scored.columns:
        raise ValueError(
            f"column {by!r} not found in scored; available columns include "
            f"{sorted(scored.columns)[:8]}..."
        )

    work = scored
    if "pairing_ok" in work.columns:
        work = work[work["pairing_ok"].astype(bool)]

    grp = work.groupby(by, sort=True)
    stats = grp.agg(n=("win_flag", "size"), wins=("win_flag", "sum"), rate=("win_flag", "mean"))
    stats = stats.reset_index()

    wins = stats["wins"].to_numpy(dtype=float)
    n = stats["n"].to_numpy(dtype=float)
    rate = stats["rate"].to_numpy(dtype=float)
    low, high = _wilson_interval(wins, n)
    yerr = np.vstack([rate - low, high - rate])

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(stats)), 5))
    x = np.arange(len(stats))
    ax.bar(x, rate, yerr=yerr, capsize=5, color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(stats[by].astype(str), rotation=0)
    ax.set_ylabel("win_rate (mean win_flag)")
    ax.set_xlabel(by)
    ax.set_ylim(0, 1)
    ax.set_title(f"Blocker win rate by {by} (95% Wilson CI)")
    for xi, ri, ni in zip(x, rate, n.astype(int)):
        ax.text(xi, min(ri + 0.03, 0.97), f"n={ni}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from matchup.config import CACHE_DIR

    scored = _make_synthetic_scored(n_reps=400, seed=0)
    print("evidence.py demo")
    print("  scored shape:", scored.shape)

    fig1 = plot_score_distribution(scored)
    p1 = CACHE_DIR / "score_distribution.png"
    fig1.savefig(p1, dpi=120)
    print("  saved:", p1)

    fig2 = plot_headline_comparison(scored, by="rusher_pos")
    p2 = CACHE_DIR / "headline_rusher_pos.png"
    fig2.savefig(p2, dpi=120)
    print("  saved:", p2)

    # Demonstrate the missing-`by` guard.
    try:
        plot_headline_comparison(scored, by="not_a_column")
    except ValueError as exc:
        print("  guard OK: ValueError raised for missing `by` ->", str(exc)[:60], "...")
