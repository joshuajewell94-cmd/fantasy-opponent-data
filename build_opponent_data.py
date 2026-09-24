#!/usr/bin/env python3
"""
Build the weekly opponent-data JSON for the 4th & Go Season Command Center.

Pulls free public data from nflverse (the same project the dashboard already
uses for rosters and player stats), scores everything with your league's rules,
and writes one weekly file the dashboard imports with one click:
  - who each NFL team plays this week and the fantasy points each defense allows
    to QB/RB/WR/TE (fills Opponent / Opp Pos Avg)
  - every player's points from the week just finished (fills Actual)
  - season-to-date usage: snap %, target/carry/red-zone/air-yards share (fills Snap %, RZ %)
Shape:

    {
      "week": 3,
      "matchups":        {"ATL": "GB", "GB": "ATL", ...},
      "defenseAverages": {"GB": {"QB": 16.6, "RB": 22.8, "WR": 33.2, "TE": 11.5}, ...},
      "vegasImplied":    {"ATL": 19.0, "GB": 23.5, ...},    # optional extra
      "playerPoints":    {"week": 2, "players": [{"name": "Lamar Jackson", "team": "BAL", "pos": "QB", "pts": 16.8}, ...]},
      "source": "...",
      "updated": "2026-09-23T12:00:00Z"
    }

Usage:
    python build_opponent_data.py                 # auto-detect season + upcoming week
    python build_opponent_data.py --week 3        # force a week
    python build_opponent_data.py --season 2026 --week 3 --out data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Data sources (all free, no API key, published by nflverse)
# ---------------------------------------------------------------------------
SCHEDULE_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
SNAPS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "snap_counts/snap_counts_{season}.csv"
)
PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.csv.gz"
)
STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)

POSITIONS = ["QB", "RB", "WR", "TE"]
POSITION_MAP = {"FB": "RB"}  # count fullbacks as RB

# How many "games" of last season's average to blend in. Early in the year a
# defense has only 1-2 games of data, which is very noisy. With PRIOR_WEIGHT=4,
# a team with 2 games played is 2/6 this season + 4/6 last season; by week 12
# it is ~73% this season. Set to 0 to use current season only.
PRIOR_WEIGHT = 4

# The dashboard's player pool mixes team-code conventions (nflverse uses JAX and
# LA; the built-in pool uses JAC for Jacksonville; other sites use LAR and WSH).
# Every alias is written into both maps so lookups work whichever code a
# player carries.
TEAM_ALIASES = {"JAX": ["JAC"], "LA": ["LAR"], "WAS": ["WSH"]}

# Your Yahoo league's scoring rules (from League > Settings). Used for both the
# defense averages and the weekly actual points, so the two always compare fairly.
SCORING = {
    # offense
    "pass_yds_per_pt": 25, "pass_td": 4, "pass_int": -1,
    "rush_att": 0.25, "rush_yds_per_pt": 10, "rush_td": 6,
    "reception": 0.5, "rec_yds_per_pt": 10, "rec_td": 6,
    "return_td": 6, "two_pt": 2, "fumble_lost": -2, "off_fumble_return_td": 6,
    # kickers
    "fg_0_19": 3, "fg_20_29": 3, "fg_30_39": 3, "fg_40_49": 4, "fg_50_plus": 5,
    "fg_miss_0_19": -4, "fg_miss_20_29": -3, "fg_miss_30_39": -2, "fg_miss_40_49": -1,
    "pat_made": 1, "pat_missed": -2,
    # defensive players (IDP)
    "tackle_solo": 1, "tackle_assist": 0.5, "sack": 2, "def_int": 3,
    "fumble_forced": 2, "fumble_recovery": 2, "def_td": 4, "safety": 2,
    "pass_defended": 1, "blocked_kick": 2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric column, or zeros if nflverse ever renames/drops it."""
    if name not in df.columns:
        log(f"  warning: column '{name}' missing from stats file; treating as 0")
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def league_points(df: pd.DataFrame) -> pd.Series:
    s = SCORING
    fumbles = col(df, "sack_fumbles_lost") + col(df, "rushing_fumbles_lost") + col(df, "receiving_fumbles_lost")
    two_pt = col(df, "passing_2pt_conversions") + col(df, "rushing_2pt_conversions") + col(df, "receiving_2pt_conversions")
    offense = (
        col(df, "passing_yards") / s["pass_yds_per_pt"] + col(df, "passing_tds") * s["pass_td"]
        + col(df, "passing_interceptions") * s["pass_int"]
        + col(df, "carries") * s["rush_att"] + col(df, "rushing_yards") / s["rush_yds_per_pt"]
        + col(df, "rushing_tds") * s["rush_td"]
        + col(df, "receptions") * s["reception"] + col(df, "receiving_yards") / s["rec_yds_per_pt"]
        + col(df, "receiving_tds") * s["rec_td"]
        + col(df, "special_teams_tds") * s["return_td"] + two_pt * s["two_pt"] + fumbles * s["fumble_lost"]
    )
    kicking = (
        col(df, "fg_made_0_19") * s["fg_0_19"] + col(df, "fg_made_20_29") * s["fg_20_29"]
        + col(df, "fg_made_30_39") * s["fg_30_39"] + col(df, "fg_made_40_49") * s["fg_40_49"]
        + (col(df, "fg_made_50_59") + col(df, "fg_made_60_")) * s["fg_50_plus"]
        + col(df, "fg_missed_0_19") * s["fg_miss_0_19"] + col(df, "fg_missed_20_29") * s["fg_miss_20_29"]
        + col(df, "fg_missed_30_39") * s["fg_miss_30_39"] + col(df, "fg_missed_40_49") * s["fg_miss_40_49"]
        + col(df, "pat_made") * s["pat_made"] + col(df, "pat_missed") * s["pat_missed"]
    )
    defense = (
        col(df, "def_tackles_solo") * s["tackle_solo"] + col(df, "def_tackle_assists") * s["tackle_assist"]
        + col(df, "def_sacks") * s["sack"] + col(df, "def_interceptions") * s["def_int"]
        + col(df, "def_fumbles_forced") * s["fumble_forced"] + col(df, "fumble_recovery_opp") * s["fumble_recovery"]
        + col(df, "def_tds") * s["def_td"] + col(df, "def_safeties") * s["safety"]
        + col(df, "def_pass_defended") * s["pass_defended"]
        + (col(df, "def_fg_blocks") + col(df, "def_punt_blocks") + col(df, "def_pat_blocks")) * s["blocked_kick"]
    )
    # A fumble recovered by the offense and returned for a TD shows up here for skill players
    off_fr_td = col(df, "fumble_recovery_tds") * df["position"].isin(["QB", "RB", "WR", "TE", "FB"])
    return offense + kicking + defense + off_fr_td * s["off_fumble_return_td"]


def _norm_name(n: str) -> str:
    n = str(n).lower()
    n = re.sub(r"[.'\u2019`]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n)
    return re.sub(r"[^a-z]", "", n)


def _canon_team(t: str) -> str:
    t = str(t).upper()
    for canon, aliases in TEAM_ALIASES.items():
        if t in aliases:
            return canon
    return t


def _load_optional(url: str, **kw) -> pd.DataFrame | None:
    try:
        return pd.read_csv(url, low_memory=False, **kw)
    except Exception as e:
        log(f"  could not load {url.rsplit('/', 1)[-1]} ({e}); skipping that part")
        return None


NON_FANTASY_POS = {"T", "G", "C", "OT", "OG", "OL", "LS", "P"}


def usage_through(stats: pd.DataFrame, season: int, before_week: int) -> dict | None:
    """Season-to-date usage for every player, from games before `before_week`:
    snap %, target share, carry share, red-zone share, air-yards share, and per-game volume."""
    st = stats[(stats["week"] < before_week) & stats["player_display_name"].notna()].copy()
    st = st[~st["position"].isin(NON_FANTASY_POS)]
    if st.empty:
        return None
    st["team"] = st["team"].map(_canon_team)
    st["tgt"], st["car"] = col(st, "targets"), col(st, "carries")
    st["air"] = col(st, "receiving_air_yards").clip(lower=0)
    team_wk = st.groupby(["team", "week"])[["tgt", "car", "air"]].sum().add_prefix("team_")
    st = st.join(team_wk, on=["team", "week"])

    # red-zone opportunities (carries + targets inside the 20) from play-by-play
    pbp = _load_optional(PBP_URL.format(season=season), compression="gzip",
                         usecols=["week", "season_type", "posteam", "yardline_100", "play_type",
                                  "two_point_attempt", "rush_attempt", "rusher_player_id", "receiver_player_id"])
    have_rz = pbp is not None
    if have_rz:
        pbp = pbp[(pbp["season_type"] == "REG") & (pbp["week"] < before_week) & (pbp["yardline_100"] <= 20)
                  & (pbp["play_type"] != "no_play") & (pbp["two_point_attempt"] != 1)]
        pbp = pbp.assign(posteam=pbp["posteam"].map(_canon_team))
        rush = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()][["week", "posteam", "rusher_player_id"]]
        rec = pbp[pbp["receiver_player_id"].notna()][["week", "posteam", "receiver_player_id"]]
        opps = pd.concat([rush.set_axis(["week", "team", "player_id"], axis=1),
                          rec.set_axis(["week", "team", "player_id"], axis=1)])
        pl_rz = opps.groupby(["player_id", "week"]).size().rename("rz")
        tm_rz = opps.groupby(["team", "week"]).size().rename("team_rz")
        st = st.join(pl_rz, on=["player_id", "week"]).join(tm_rz, on=["team", "week"])
        st[["rz", "team_rz"]] = st[["rz", "team_rz"]].fillna(0)

    # snap % from snap counts (matched by name + team)
    snaps = _load_optional(SNAPS_URL.format(season=season))
    snap_map = {}
    if snaps is not None:
        snaps = snaps[(snaps["game_type"] == "REG") & (snaps["week"] < before_week)].copy()
        snaps["k"] = snaps["player"].map(_norm_name)
        snaps["team"] = snaps["team"].map(_canon_team)
        agg = snaps.groupby(["k", "team"])[["offense_pct", "defense_pct"]].mean() * 100
        for (k, t), r in agg.iterrows():
            snap_map[(k, t)] = r
        by_name = agg.reset_index().groupby("k")
        snap_single = {k: g.iloc[0] for k, g in by_name if len(g) == 1}

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    out = []
    for pid, g in st.groupby("player_id"):
        last = g.sort_values("week").iloc[-1]
        n = len(g)
        rec = {
            "name": str(last["player_display_name"]), "team": str(last["team"]), "pos": str(last["position"]),
            "games": int(n),
            "targets": round(g["tgt"].sum() / n, 1), "carries": round(g["car"].sum() / n, 1),
            "receptions": round(col(g, "receptions").sum() / n, 1),
            "opportunities": round((g["tgt"].sum() + g["car"].sum()) / n, 1),
            "tackles": round((col(g, "def_tackles_solo").sum() + col(g, "def_tackle_assists").sum()) / n, 1),
            "sacks": round(col(g, "def_sacks").sum() / n, 2),
            "targetShare": pct(g["tgt"].sum(), g["team_tgt"].sum()),
            "carryShare": pct(g["car"].sum(), g["team_car"].sum()),
            "airShare": pct(g["air"].sum(), g["team_air"].sum()),
            "rzShare": pct(g["rz"].sum(), g["team_rz"].sum()) if have_rz else None,
        }
        k = _norm_name(rec["name"])
        sr = snap_map.get((k, rec["team"])) if snap_map else None
        if sr is None and snap_map:
            sr = snap_single.get(k)
        if sr is not None:
            rec["snap"] = round(float(sr["offense_pct"]), 1) if sr["offense_pct"] > 0 else None
            rec["defSnap"] = round(float(sr["defense_pct"]), 1) if sr["defense_pct"] > 0 else None
        out.append({k2: v for k2, v in rec.items() if v is not None})
    return {"throughWeek": before_week - 1, "count": len(out), "players": out}


def player_points_for_week(stats: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int) -> dict | None:
    """Every player's league-scored points for one finished week, or None if that
    week's stats aren't complete yet (then the dashboard simply gets no points)."""
    games = schedule[(schedule["season"] == season) & (schedule["week"] == week) & (schedule["game_type"] == "REG")]
    played = set(games.loc[games["home_score"].notna(), "home_team"]) | set(games.loc[games["home_score"].notna(), "away_team"])
    wk = stats[stats["week"] == week]
    if not played or len(games) != len(games[games["home_score"].notna()]):
        log(f"  Week {week} games aren't all final yet; skipping player points.")
        return None
    missing = played - set(wk["team"])
    if missing:
        log(f"  Week {week} stats not posted yet for {sorted(missing)}; skipping player points.")
        return None
    wk = wk[wk["player_display_name"].notna()]  # a few rows have no name; skip them
    wk = wk.assign(pts=league_points(wk).round(2))
    players = [
        {"name": str(r.player_display_name), "team": str(r.team) if pd.notna(r.team) else "",
         "pos": str(r.position) if pd.notna(r.position) else "", "pts": float(r.pts)}
        for r in wk.itertuples()
    ]
    return {"week": week, "count": len(players), "players": players}


def load_stats(season: int) -> pd.DataFrame | None:
    url = STATS_URL.format(season=season)
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as e:  # file won't exist before a season starts
        log(f"  could not load {season} stats ({e}); skipping")
        return None
    return df[df["season_type"] == "REG"].copy()


def points_allowed_per_game(stats: pd.DataFrame, before_week: int | None = None) -> pd.DataFrame:
    """Rows: defense team, Columns: QB/RB/WR/TE, Values: avg league-scored points allowed per game,
    plus a 'games' column with how many games that average is based on."""
    df = stats
    if before_week is not None:
        df = df[df["week"] < before_week]
    if df.empty:
        return pd.DataFrame(columns=POSITIONS + ["games"])
    df = df.assign(pos=df["position"].replace(POSITION_MAP), pts=league_points(df))
    df = df[df["pos"].isin(POSITIONS)]
    # total allowed to each position in each game, then average across games
    per_game = df.groupby(["opponent_team", "week", "pos"])["pts"].sum().unstack("pos").fillna(0.0)
    per_game = per_game.reindex(columns=POSITIONS, fill_value=0.0)
    avg = per_game.groupby(level="opponent_team").mean()
    avg["games"] = per_game.groupby(level="opponent_team").size()
    return avg


def detect_week(schedule: pd.DataFrame, season: int) -> int:
    """First regular-season week that still has an unplayed game."""
    reg = schedule[(schedule["season"] == season) & (schedule["game_type"] == "REG")]
    unplayed = reg[reg["home_score"].isna()]
    if unplayed.empty:
        raise SystemExit(f"No unplayed regular-season games left in {season}.")
    return int(unplayed["week"].min())


def current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 3 else now.year - 1  # Jan/Feb belong to last season


def expand_aliases(mapping: dict) -> dict:
    out = dict(mapping)
    for canon, aliases in TEAM_ALIASES.items():
        if canon in mapping:
            for a in aliases:
                out[a] = mapping[canon]
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------
def build(season: int, week: int | None) -> dict:
    log("Loading schedule...")
    schedule = pd.read_csv(SCHEDULE_URL, low_memory=False)
    if week is None:
        week = detect_week(schedule, season)
    log(f"Building Season {season}, Week {week}")

    games = schedule[(schedule["season"] == season) & (schedule["week"] == week) & (schedule["game_type"] == "REG")]
    if games.empty:
        raise SystemExit(f"No regular-season games found for {season} week {week}.")

    matchups, implied = {}, {}
    for g in games.itertuples():
        matchups[g.home_team], matchups[g.away_team] = g.away_team, g.home_team
        # nfldata spread_line is from the home team's view (positive = home favored)
        if pd.notna(g.spread_line) and pd.notna(g.total_line):
            implied[g.home_team] = round((g.total_line + g.spread_line) / 2, 1)
            implied[g.away_team] = round((g.total_line - g.spread_line) / 2, 1)

    log("Loading player stats...")
    cur_stats = load_stats(season)
    cur = points_allowed_per_game(cur_stats, before_week=week) if cur_stats is not None else points_allowed_per_game(pd.DataFrame())
    prior = pd.DataFrame(columns=POSITIONS + ["games"])
    if PRIOR_WEIGHT > 0:
        prior_stats = load_stats(season - 1)
        if prior_stats is not None:
            prior = points_allowed_per_game(prior_stats)

    teams = sorted(set(cur.index) | set(prior.index) | set(matchups))
    defense = {}
    for t in teams:
        n = float(cur.loc[t, "games"]) if t in cur.index else 0.0
        k = float(PRIOR_WEIGHT) if t in prior.index else 0.0
        if n + k == 0:
            continue
        defense[t] = {
            p: round(
                ((cur.loc[t, p] * n if n else 0.0) + (prior.loc[t, p] * k if k else 0.0)) / (n + k), 1
            )
            for p in POSITIONS
        }

    points = None
    if cur_stats is not None and week > 1:
        log(f"Scoring Week {week - 1} player points...")
        points = player_points_for_week(cur_stats, schedule, season, week - 1)

    usage = None
    if cur_stats is not None and week > 1:
        log("Building usage percentages...")
        usage = usage_through(cur_stats, season, week)

    cur_games = int(cur["games"].max()) if not cur.empty else 0
    blend = f", blended with {season - 1} at a {PRIOR_WEIGHT}-game weight" if PRIOR_WEIGHT else ""
    data = {
        "week": week,
        "matchups": expand_aliases(matchups),
        "defenseAverages": expand_aliases(defense),
        "vegasImplied": expand_aliases(implied),
        "source": (
            f"nflverse player stats (league-scoring FPA through {season} Week {week - 1}, "
            f"up to {cur_games} game(s){blend}) + nflverse schedule/lines. Auto-generated."
        ),
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if points:
        data["playerPoints"] = points
    if usage:
        data["usage"] = usage
    return data


def validate(data: dict) -> list[str]:
    """Return a list of problems. Any problem fails the run, so bad data never ships."""
    problems = []
    m, d = data["matchups"], data["defenseAverages"]
    canon = [t for t in m if t not in {a for al in TEAM_ALIASES.values() for a in al}]
    if len(canon) < 20:
        problems.append(f"Only {len(canon)} teams have a matchup (expected 26-32; byes reduce it).")
    for team, opp in m.items():
        if opp not in d:
            problems.append(f"{team}'s opponent {opp} has no defense averages.")
        if m.get(opp) != team and team in canon:
            problems.append(f"Matchup not symmetric: {team} -> {opp} -> {m.get(opp)}")
    for team, row in d.items():
        for p in POSITIONS:
            v = row.get(p)
            if v is None or not (0 <= v <= 60):
                problems.append(f"{team} {p} value looks wrong: {v}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=current_season())
    ap.add_argument("--week", type=int, default=None, help="default: next week with unplayed games")
    ap.add_argument("--out", default="data", help="output folder")
    args = ap.parse_args()

    data = build(args.season, args.week)
    problems = validate(data)
    if problems:
        log("VALIDATION FAILED - no file written:")
        for p in problems:
            log("  - " + p)
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, allow_nan=False)  # fail loudly rather than write invalid JSON
    (out / f"week{data['week']}_opponent_data.json").write_text(text)
    (out / "latest.json").write_text(text)
    log(f"Wrote {out}/week{data['week']}_opponent_data.json and {out}/latest.json "
        f"({len(data['matchups'])} matchup keys, {len(data['defenseAverages'])} defense keys, "
        f"{data['playerPoints']['count'] if 'playerPoints' in data else 0} player scores, "
        f"{data['usage']['count'] if 'usage' in data else 0} usage profiles)")


if __name__ == "__main__":
    main()
