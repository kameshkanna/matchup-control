"""Task 3 (Person A) — synchronized trajectories + core control features.

Everything is signed so HIGHER = BLOCKER WON. The blocker's whole job is to
keep his body between the rusher and the QB and stop the rusher gaining ground
toward the QB; the features below measure exactly that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import io_load
from . import pairing


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
def _player_index(play: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split a play's frames into {nfl_id -> frame-indexed slice} ONCE.

    Pass the result as `pindex` to get_rep_tracks/rep_features to avoid
    re-masking the full play frame for every player on every matchup. This is
    the hot-path optimisation that lets scoring scale to all 122 games.
    """
    idx = {}
    for nid, grp in play.groupby("nfl_id"):
        idx[int(nid)] = grp.set_index("frame_id")[["x", "y", "s", "a", "dir"]].sort_index()
    return idx


def get_rep_tracks(
    game_id: int, play_id: int, blocker_id: int, rusher_id: int,
    qb_id: int | None = None, play: pd.DataFrame | None = None,
    window: tuple[int, int] | None = None,
    pindex: dict | None = None,
) -> pd.DataFrame:
    """Frame-aligned blocker / rusher / qb trajectories over the rep.

    SCHEMA: rep_tracks. One row per frame between snap and end (inclusive).
    `play`/`window`/`pindex` can be passed in to avoid recomputation when
    scoring many matchups on the same play.
    """
    if play is None and pindex is None:
        play = io_load.get_play(game_id, play_id)
    if window is None:
        window = pairing.get_rep_window(play)
    snap_frame, end_frame = window

    if qb_id is None:
        if play is None:
            raise ValueError("qb_id must be given when only pindex is provided")
        qb_rows = play[play.pff_role == C.ROLE_PASS]["nfl_id"].unique()
        qb_id = int(qb_rows[0]) if len(qb_rows) else C.BALL_ID

    if pindex is None:
        pindex = _player_index(play)

    empty = pd.DataFrame(columns=["x", "y", "s", "a", "dir"])

    def _slice(nid):
        d = pindex.get(int(nid), empty)
        if len(d) == 0:
            return d
        return d.loc[(d.index >= snap_frame) & (d.index <= end_frame)]

    blk = _slice(blocker_id)
    rsh = _slice(rusher_id)
    qb = _slice(qb_id)

    if len(blk) == 0 or len(rsh) == 0 or len(qb) == 0:
        return pd.DataFrame(
            columns=["frame_id", "t_sec", "blk_x", "blk_y", "blk_s", "blk_dir",
                     "rsh_x", "rsh_y", "rsh_s", "rsh_a", "rsh_dir", "qb_x", "qb_y"]
        )

    frames = blk.index.intersection(rsh.index).intersection(qb.index).sort_values()

    out = pd.DataFrame(index=frames)
    out["frame_id"] = frames
    out["t_sec"] = (frames - snap_frame) / C.HZ
    out["blk_x"], out["blk_y"] = blk.loc[frames, "x"], blk.loc[frames, "y"]
    out["blk_s"], out["blk_dir"] = blk.loc[frames, "s"], blk.loc[frames, "dir"]
    out["rsh_x"], out["rsh_y"] = rsh.loc[frames, "x"], rsh.loc[frames, "y"]
    out["rsh_s"], out["rsh_a"] = rsh.loc[frames, "s"], rsh.loc[frames, "a"]
    out["rsh_dir"] = rsh.loc[frames, "dir"]
    out["qb_x"], out["qb_y"] = qb.loc[frames, "x"], qb.loc[frames, "y"]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-frame control signals
# ---------------------------------------------------------------------------
def control_timeseries(rep_tracks: pd.DataFrame) -> pd.DataFrame:
    """Per-frame control signals (feeds the viz bottom panel). SCHEMA: control_ts."""
    t = rep_tracks
    rusher_to_qb = np.hypot(t.rsh_x - t.qb_x, t.rsh_y - t.qb_y)
    blocker_rusher_sep = np.hypot(t.blk_x - t.rsh_x, t.blk_y - t.rsh_y)
    betweenness = _betweenness(t)
    mirroring = _mirroring(t.blk_dir.to_numpy(), t.rsh_dir.to_numpy())

    return pd.DataFrame(
        {
            "frame_id": t.frame_id,
            "t_sec": t.t_sec,
            "rusher_to_qb_dist": rusher_to_qb,
            "blocker_rusher_sep": blocker_rusher_sep,
            "betweenness": betweenness,
            "mirroring": mirroring,
        }
    )


def _betweenness(t: pd.DataFrame) -> np.ndarray:
    """How well the blocker sits ON the rusher->qb segment.

    1 = blocker exactly on the line between rusher and QB (perfect shield);
    decays with perpendicular distance from that segment. Only counts when the
    blocker's projection falls between rusher and QB (0<=proj<=1), else low.
    """
    rx, ry = t.rsh_x.to_numpy(), t.rsh_y.to_numpy()
    qx, qy = t.qb_x.to_numpy(), t.qb_y.to_numpy()
    bx, by = t.blk_x.to_numpy(), t.blk_y.to_numpy()

    vx, vy = qx - rx, qy - ry
    seg_len2 = vx * vx + vy * vy
    seg_len2 = np.where(seg_len2 < 1e-6, 1e-6, seg_len2)
    # projection parameter of blocker onto rusher->qb segment
    proj = ((bx - rx) * vx + (by - ry) * vy) / seg_len2
    proj_c = np.clip(proj, 0.0, 1.0)
    # closest point on segment
    cx, cy = rx + proj_c * vx, ry + proj_c * vy
    perp = np.hypot(bx - cx, by - cy)
    # shield quality: close to the line AND between the two players
    on_line = np.exp(-perp / 1.5)              # 1.5 yd scale
    in_span = ((proj >= -0.1) & (proj <= 1.1)).astype(float)
    return on_line * in_span


def _mirroring(blk_dir_deg: np.ndarray, rsh_dir_deg: np.ndarray) -> np.ndarray:
    """Cosine similarity of the two players' motion directions. 1 = mirrored."""
    b = np.deg2rad(blk_dir_deg)
    r = np.deg2rad(rsh_dir_deg)
    # unit vectors (dir is 0=+y clockwise, but cosine of the angle diff is invariant)
    return np.cos(b - r)


# ---------------------------------------------------------------------------
# Collapse one rep to flat features
# ---------------------------------------------------------------------------
def rep_features(
    game_id: int, play_id: int, blocker_id: int, rusher_id: int,
    qb_id: int | None = None, play: pd.DataFrame | None = None,
    window: tuple[int, int] | None = None, los_x: float | None = None,
    pindex: dict | None = None,
) -> dict:
    """Collapse one rep to a flat dict of features. SCHEMA: features (keys).

    Signed so HIGHER = BLOCKER WON for every component.
    """
    tracks = get_rep_tracks(
        game_id, play_id, blocker_id, rusher_id, qb_id=qb_id, play=play,
        window=window, pindex=pindex,
    )
    f = {
        "game_id": game_id, "play_id": play_id,
        "blocker_id": blocker_id, "rusher_id": rusher_id,
    }
    if len(tracks) < 2:
        # Degenerate rep — return NaN features but keep the row.
        for k in _FEATURE_KEYS:
            f[k] = np.nan
        f["n_frames"] = len(tracks)
        return f

    ts = control_timeseries(tracks)
    r2qb = ts.rusher_to_qb_dist.to_numpy()
    sep = ts.blocker_rusher_sep.to_numpy()

    # Ground given up toward the QB: how much the rusher closed on the QB.
    # Blocker winning => rusher never gets much closer than he started.
    f["ground_given_up"] = float(r2qb[0] - np.nanmin(r2qb))
    f["min_rusher_to_qb_dist"] = float(np.nanmin(r2qb))

    # Penetration past LOS. los_x defaults to rusher's x at snap (DL start).
    if los_x is None:
        los_x = float(tracks.rsh_x.iloc[0])
    # offense moves +x, so rusher moving to SMALLER x = penetrating backfield.
    f["penetration_past_los"] = float(los_x - np.nanmin(tracks.rsh_x.to_numpy()))

    f["sep_mean"] = float(np.nanmean(sep))
    f["sep_min"] = float(np.nanmin(sep))
    f["betweenness_mean"] = float(np.nanmean(ts.betweenness))
    f["betweenness_end"] = float(np.nanmean(ts.betweenness.to_numpy()[-3:]))
    f["mirroring_mean"] = float(np.nanmean(ts.mirroring))

    # Rusher kinematics in the final 0.5s (5 frames). Still accelerating toward
    # the QB late = rusher winning, so we NEGATE for "higher = blocker won".
    n_late = min(5, len(tracks))
    f["rusher_speed_late"] = -float(np.nanmean(tracks.rsh_s.to_numpy()[-n_late:]))
    f["rusher_accel_late"] = -float(np.nanmean(tracks.rsh_a.to_numpy()[-n_late:]))

    f["rep_duration_sec"] = float(tracks.t_sec.iloc[-1])
    f["n_frames"] = len(tracks)
    return f


_FEATURE_KEYS = [
    "ground_given_up", "min_rusher_to_qb_dist", "penetration_past_los",
    "sep_mean", "sep_min", "betweenness_mean", "betweenness_end",
    "mirroring_mean", "rusher_speed_late", "rusher_accel_late",
    "rep_duration_sec",
]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    play = io_load.get_play(C.GOLDEN_GAME, C.GOLDEN_PLAY)
    win = pairing.get_rep_window(play)

    # Golden LOST rep (pressure allowed) vs a clean rep on the same play.
    lost = rep_features(C.GOLDEN_GAME, C.GOLDEN_PLAY, 42377, 42403, play=play, window=win)
    clean = rep_features(C.GOLDEN_GAME, C.GOLDEN_PLAY, 42404, 44955, play=play, window=win)

    ts = control_timeseries(
        get_rep_tracks(C.GOLDEN_GAME, C.GOLDEN_PLAY, 42377, 42403, play=play, window=win)
    )

    lines = [
        "GOLDEN LOST rep (LT 42377 vs 42403, pressure_allowed=1):",
        *[f"  {k}: {lost[k]:.3f}" for k in _FEATURE_KEYS if not np.isnan(lost[k])],
        "",
        "clean rep (LG 42404 vs 44955, pressure_allowed=0):",
        *[f"  {k}: {clean[k]:.3f}" for k in _FEATURE_KEYS if not np.isnan(clean[k])],
        "",
        "control timeseries (lost rep), first/last 3 frames:",
        pd.concat([ts.head(3), ts.tail(3)]).to_string(index=False),
        "",
        f"CHECK ground_given_up lost>clean: {lost['ground_given_up'] > clean['ground_given_up']}",
        f"CHECK betweenness_end lost<clean: {lost['betweenness_end'] < clean['betweenness_end']}",
    ]
    msg = "\n".join(lines)
    print(msg)
    (C.CACHE_DIR / "_smoke_features.txt").write_text(msg, encoding="utf-8")
