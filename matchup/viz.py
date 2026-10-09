"""Task 5 — the signature visual for one rep.

:func:`plot_matchup` draws a two-panel figure for ONE blocker-vs-rusher rep:

* top — the field (normalised: offense moves +x) with both players' paths,
  snap/end markers, the QB path, and the line of scrimmage.
* bottom — "control over time": rusher->QB distance, blocker-rusher gap, and
  betweenness, so the moment the blocker wins or loses is visible.

"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matchup import _upstream as up
from matchup.config import FIELD_LEN, FIELD_WID

_BLK_COLOR = "cornflowerblue"
_RSH_COLOR = "tomato"
_QB_COLOR = "white"


def draw_field(ax: matplotlib.axes.Axes | None = None) -> matplotlib.axes.Axes:
    """Draw a plain field in normalised coordinates (offense moves toward +x).

    Parameters
    ----------
    ax:
        Axes to draw on. A new figure/axes is created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("forestgreen")
    ax.axvspan(0, 10, color="darkgreen")
    ax.axvspan(FIELD_LEN - 10, FIELD_LEN, color="darkgreen")
    for x in range(5, int(FIELD_LEN), 5):
        ax.axvline(x, color="white", lw=1.0 if x % 10 == 0 else 0.4, alpha=0.4)
    ax.set_xlim(0, FIELD_LEN)
    ax.set_ylim(0, FIELD_WID)
    ax.set_aspect("equal")
    ax.set_xlabel("x (yards) — offense moves right →")
    ax.set_ylabel("y (yards)")
    return ax


def _los_x(game_id: int, play_id: int, play_frames: pd.DataFrame) -> float | None:
    """Line of scrimmage in normalised x, or ``None`` if it can't be resolved.

    ``absoluteYardlineNumber`` is in raw field coordinates, so it gets the same
    left-play flip as tracking ``x``.
    """
    plays = up.plays()
    row = plays[(plays["gameId"] == game_id) & (plays["playId"] == play_id)]
    if not len(row) or pd.isna(row["absoluteYardlineNumber"].iloc[0]):
        return None
    los = float(row["absoluteYardlineNumber"].iloc[0])
    direction = play_frames["play_direction"].dropna()
    if len(direction) and direction.iloc[0] == "left":
        los = FIELD_LEN - los
    return los


def _outcome_label(meta: pd.Series | None) -> str:
    if meta is None:
        return "outcome unknown"
    if meta.get("sack_allowed"):
        return "SACK allowed"
    if meta.get("hit_allowed"):
        return "HIT allowed"
    if meta.get("hurry_allowed"):
        return "HURRY allowed"
    return "clean rep (no pressure)"


def plot_matchup(
    game_id: int,
    play_id: int,
    blocker_id: int,
    rusher_id: int | None = None,
) -> matplotlib.figure.Figure:
    """Two-panel signature visual for one rep: field paths + control over time.

    Parameters
    ----------
    game_id, play_id:
        The play.
    blocker_id:
        The blocker (or, for the receiver head, the receiver).
    rusher_id:
        The opponent. If ``None``, resolved from the play's matchups table via
        the blocker's PFF assignment.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``rusher_id`` is ``None`` and the blocker has no resolved matchup on
        the play, or if the rep has no tracking frames for the pair.
    """
    mu = up.matchups(game_id, play_id)
    row = mu[mu["blocker_id"] == blocker_id] if len(mu) else mu
    if rusher_id is None:
        if not len(row) or not bool(row["pairing_ok"].iloc[0]):
            raise ValueError(f"no resolved matchup for blocker {blocker_id} on {game_id}/{play_id}")
        rusher_id = int(row["rusher_id"].iloc[0])

    meta = None
    if len(row):
        hit = row[row["rusher_id"] == rusher_id]
        meta = (hit.iloc[0] if len(hit) else row.iloc[0])

    rt = up.rep_tracks(game_id, play_id, blocker_id, rusher_id)
    rt = rt.dropna(subset=["blk_x", "rsh_x"])
    if rt.empty:
        raise ValueError(f"no tracking frames for {blocker_id} vs {rusher_id} on {game_id}/{play_id}")
    cts = up.control_ts(rt)
    play_frames = up.load_play(game_id, play_id)

    blk_name = (meta["blocker_name"] if meta is not None else None) or str(blocker_id)
    rsh_name = (meta["rusher_name"] if meta is not None else None) or str(rusher_id)

    fig, (ax_f, ax_c) = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2, 1]}
    )

    # ---- top: field + paths ------------------------------------------------
    draw_field(ax_f)
    los = _los_x(game_id, play_id, play_frames)
    if los is not None:
        ax_f.axvline(los, color="yellow", lw=1.8, ls="--", label="line of scrimmage")

    ax_f.plot(rt["blk_x"], rt["blk_y"], "-o", color=_BLK_COLOR, ms=3, label=f"blocker: {blk_name}")
    ax_f.plot(rt["rsh_x"], rt["rsh_y"], "-o", color=_RSH_COLOR, ms=3, label=f"rusher: {rsh_name}")
    if rt["qb_x"].notna().any():
        ax_f.plot(rt["qb_x"], rt["qb_y"], "-s", color=_QB_COLOR, ms=3, label="QB")

    for cx, cy, color in (("blk_x", "blk_y", _BLK_COLOR), ("rsh_x", "rsh_y", _RSH_COLOR)):
        ax_f.scatter(rt[cx].iloc[0], rt[cy].iloc[0], s=80, color=color, edgecolor="black", zorder=5)
        ax_f.scatter(rt[cx].iloc[-1], rt[cy].iloc[-1], s=120, marker="X", color=color,
                     edgecolor="black", zorder=5)

    xs = pd.concat([rt["blk_x"], rt["rsh_x"], rt["qb_x"]]).dropna().to_numpy()
    ys = pd.concat([rt["blk_y"], rt["rsh_y"], rt["qb_y"]]).dropna().to_numpy()
    if los is not None:
        xs = np.append(xs, los)
    pad = 3.0
    ax_f.set_xlim(max(0.0, xs.min() - pad), min(FIELD_LEN, xs.max() + pad))
    ax_f.set_ylim(max(0.0, ys.min() - pad), min(FIELD_WID, ys.max() + pad))
    ax_f.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_f.set_title(
        f"{blk_name} vs {rsh_name} — {_outcome_label(meta)}\n"
        f"game {game_id}, play {play_id}  (● snap, ✕ end of rep)",
        fontsize=11,
    )

    # ---- bottom: control over time ----------------------------------------
    ax_c.plot(cts["t_sec"], cts["rusher_to_qb_dist"], color=_RSH_COLOR, lw=2,
              label="rusher → QB distance (shrinking = blocker losing)")
    ax_c.plot(cts["t_sec"], cts["blocker_rusher_sep"], color=_BLK_COLOR, lw=2,
              label="blocker–rusher gap")
    ax_c.set_xlabel("seconds since snap")
    ax_c.set_ylabel("yards")
    ax_b = ax_c.twinx()
    ax_b.plot(cts["t_sec"], cts["betweenness"], color="orange", ls="--", lw=1.8,
              label="betweenness (1 = blocker between rusher and QB)")
    ax_b.set_ylim(-0.05, 1.05)
    ax_b.set_ylabel("betweenness")
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_b.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7, framealpha=0.85)
    ax_c.set_title("Control over time", fontsize=10)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    from matchup.config import CACHE_DIR

    g = up.GOLDEN
    print("viz.py demo —  upstream source:", up.upstream_source())
    fig = plot_matchup(g["game_id"], g["play_id"], g["blocker_id"], g["rusher_id"])
    out = CACHE_DIR / "golden_matchup.png"
    fig.savefig(out, dpi=120)
    print("  saved:", out)
