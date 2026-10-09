"""Human-readable labels + team views for the dashboard (Person C).

The raw data keys everything on numeric ``gameId`` / ``playId``, which are
meaningless to a viewer. This module turns those into:

* **game labels** — "TB vs DAL (Week 1, 2021-09-09)" with the matching id,
* **team listing** — every team abbreviation, for a searchable dropdown,
* **play labels** — "Q1 13:33 · 3rd & 2 · T.Brady pass deep right to C.Godwin",
* **team scoreboard** — individual win scores aggregated to a per-team view for
  a single game, so a match can be read "as a team", not rep by rep.

Everything is derived from ``games.csv`` / ``plays.csv`` via
:mod:`matchup._upstream`; no new data sources.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from matchup import _upstream as up


# --------------------------------------------------------------------------- #
# Games / teams
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _games() -> pd.DataFrame:
    return pd.read_csv(up.DATA_DIR / "games.csv")


def list_teams() -> list[str]:
    """All team abbreviations present in the data, sorted (for a dropdown)."""
    g = _games()
    teams = set(g["homeTeamAbbr"]) | set(g["visitorTeamAbbr"])
    return sorted(teams)


def games_for_team(team: str | None = None) -> pd.DataFrame:
    """Games involving ``team`` (or all games if ``team`` is None).

    Returns columns: ``game_id, week, game_date, home, visitor, label`` where
    ``label`` is "HOME vs VISITOR (Week W, date)".
    """
    g = _games().copy()
    if team:
        g = g[(g["homeTeamAbbr"] == team) | (g["visitorTeamAbbr"] == team)]
    g = g.sort_values(["week", "gameId"])
    out = pd.DataFrame(
        {
            "game_id": g["gameId"].astype(int),
            "week": g["week"].astype(int),
            "game_date": g["gameDate"].astype(str),
            "home": g["homeTeamAbbr"],
            "visitor": g["visitorTeamAbbr"],
        }
    )
    out["label"] = (
        out["home"] + " vs " + out["visitor"]
        + " (Week " + out["week"].astype(str) + ", " + out["game_date"] + ")"
    )
    return out.reset_index(drop=True)


def game_label(game_id: int) -> str:
    """"TB vs DAL (Week 1, 09/09/2021)" for a single game id."""
    g = _games()
    row = g[g["gameId"] == game_id]
    if not len(row):
        return str(game_id)
    r = row.iloc[0]
    return f"{r['homeTeamAbbr']} vs {r['visitorTeamAbbr']} (Week {int(r['week'])}, {r['gameDate']})"


def game_teams(game_id: int) -> tuple[str, str]:
    """(home, visitor) team abbreviations for a game."""
    g = _games()
    row = g[g["gameId"] == game_id]
    if not len(row):
        return ("", "")
    r = row.iloc[0]
    return (str(r["homeTeamAbbr"]), str(r["visitorTeamAbbr"]))


# --------------------------------------------------------------------------- #
# Plays
# --------------------------------------------------------------------------- #
_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def _down_dist(down, togo) -> str:
    try:
        d = _ORDINAL.get(int(down), f"{int(down)}th")
        return f"{d} & {int(togo)}"
    except (ValueError, TypeError):
        return ""


def _short_desc(desc: str, limit: int = 60) -> str:
    if not isinstance(desc, str):
        return ""
    # Strip the leading "(clock) " and any "(Shotgun)"/"(No Huddle)" prefix noise.
    s = desc
    for _ in range(2):
        if s.startswith("("):
            close = s.find(")")
            if 0 < close < 20:  # only strip short parentheticals (clock/formation)
                s = s[close + 1:].lstrip()
    return s[:limit].rstrip()


def plays_for_game(game_id: int, man_only: bool = False) -> pd.DataFrame:
    """Dropback plays for a game with readable labels.

    Returns columns: ``play_id, quarter, clock, down_dist, possession,
    defense, coverage, desc, label`` where ``label`` is e.g.
    "Q1 13:33 · 3rd & 2 · TB · T.Brady pass deep right to C.Godwin".

    Parameters
    ----------
    game_id:
        The game.
    man_only:
        If True, keep only man-coverage plays (``pff_passCoverageType`` starting
        with "Man"). The receiver head only produces a visualisable pair on man
        coverage, so this restricts the dropdown to plays that will yield a
        result instead of the "no man-coverage pair" message.
    """
    plays = up.plays()
    pr = plays[plays["gameId"] == game_id].copy()
    if man_only and "pff_passCoverageType" in pr.columns:
        pr = pr[pr["pff_passCoverageType"].astype(str).str.lower().str.startswith("man")]
    if not len(pr):
        return pd.DataFrame(
            columns=["play_id", "quarter", "clock", "down_dist", "possession",
                     "defense", "coverage", "desc", "label"]
        )
    pr = pr.sort_values(["quarter", "playId"])
    out = pd.DataFrame(
        {
            "play_id": pr["playId"].astype(int),
            "quarter": pr["quarter"],
            "clock": pr["gameClock"].astype(str),
            "down_dist": [_down_dist(d, t) for d, t in zip(pr["down"], pr["yardsToGo"])],
            "possession": pr["possessionTeam"],
            "defense": pr["defensiveTeam"],
            "coverage": pr.get("pff_passCoverageType", pd.Series(index=pr.index, dtype=object)),
            "desc": [_short_desc(d) for d in pr["playDescription"]],
        }
    )

    def _mk(r):
        bits = [f"Q{int(r['quarter'])} {r['clock']}" if pd.notna(r["quarter"]) else ""]
        if r["down_dist"]:
            bits.append(r["down_dist"])
        bits.append(str(r["possession"]))
        if r["desc"]:
            bits.append(r["desc"])
        return " · ".join(b for b in bits if b)

    out["label"] = out.apply(_mk, axis=1)
    return out.reset_index(drop=True)


def play_label(game_id: int, play_id: int) -> str:
    """Readable one-line label for a single play."""
    pf = plays_for_game(game_id)
    row = pf[pf["play_id"] == play_id]
    return row["label"].iloc[0] if len(row) else f"play {play_id}"


# --------------------------------------------------------------------------- #
# Team scoreboard — the "match as a team" view
# --------------------------------------------------------------------------- #
def team_scoreboard(scored: pd.DataFrame, game_id: int | None = None) -> pd.DataFrame:
    """Aggregate individual win scores to a per-team view.

    Each scored rep belongs to a team (the blocker's offense). This rolls the
    per-rep ``win_score`` / ``win_flag`` up to one row per team (optionally
    within a single game), so a match reads as a team result rather than a list
    of individual battles.

    Parameters
    ----------
    scored:
        A ``scored``-schema frame (needs ``team``, ``win_score``, ``win_flag``,
        ``pressure_allowed``; ``game_id`` required if filtering by game).
    game_id:
        If given, restrict to that game before aggregating.

    Returns
    -------
    pandas.DataFrame
        Columns: ``team, n_reps, win_rate, mean_win_score, pressures_allowed,
        pressure_rate`` sorted by ``mean_win_score`` descending (best line
        first). Empty frame if there is nothing to aggregate.
    """
    df = scored
    if game_id is not None and "game_id" in df.columns:
        df = df[df["game_id"] == game_id]
    if "pairing_ok" in df.columns:
        df = df[df["pairing_ok"].astype(bool)]
    df = df[df["win_score"].notna()] if "win_score" in df.columns else df
    if not len(df) or "team" not in df.columns:
        return pd.DataFrame(
            columns=["team", "n_reps", "win_rate", "mean_win_score",
                     "pressures_allowed", "pressure_rate"]
        )

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        return pd.Series(
            {
                "n_reps": n,
                "win_rate": float(group["win_flag"].mean()) if "win_flag" in group else float("nan"),
                "mean_win_score": float(group["win_score"].mean()),
                "pressures_allowed": int(group["pressure_allowed"].sum())
                if "pressure_allowed" in group else 0,
                "pressure_rate": float(group["pressure_allowed"].mean())
                if "pressure_allowed" in group else float("nan"),
            }
        )

    board = df.groupby("team", sort=False).apply(_agg, include_groups=False).reset_index()
    return board.sort_values("mean_win_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("labels.py demo")
    print("  teams:", list_teams()[:8], "...")
    tb = games_for_team("TB")
    print(f"\n  TB games ({len(tb)}):")
    print(tb[["game_id", "label"]].head().to_string(index=False))

    gid = int(tb["game_id"].iloc[0])
    pf = plays_for_game(gid)
    print(f"\n  first plays of {game_label(gid)}:")
    for lab in pf["label"].head(4):
        print("   ", lab)

    from matchup import pipeline
    scored = pipeline.run(n_games=5)["scored"]
    print("\n  team scoreboard (all 5 games):")
    print(team_scoreboard(scored).to_string(index=False))
    print(f"\n  team scoreboard (game {gid} only):")
    print(team_scoreboard(scored, game_id=gid).to_string(index=False))
