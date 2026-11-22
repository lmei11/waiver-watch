#!/usr/bin/env python3
"""
Find the best free-agent streamers for a given day.

For every NBA team playing that day, players who are OUT (or day-to-day) free
up minutes. This estimates how those minutes get split among teammates, scales
the teammates' stats up to match, scores the result across your league's scoring
categories, and ranks the free agents who play that day.

Usage:
    python3 streamers.py                     # today (US Eastern), top 15
    python3 streamers.py --date 2026-11-23   # another day
    python3 streamers.py --top 25 --punt "TO,FT%"
    python3 streamers.py --weights "3PM=3,FG%=0"   # need threes, ignore FG%
    python3 streamers.py --markdown          # table for a GitHub Actions summary
    python3 streamers.py --push 3            # ntfy the top 3 (needs NTFY_TOPIC)

Reads ESPN_LEAGUE_ID / ESPN_S2 / ESPN_SWID / NTFY_TOPIC from .env like waiver_watch.py.

How the numbers work (all the knobs are the constants below):
  - Baseline per-game line = last 15 days blended with the season (or projections
    early on). Recent games already include absences that started a while ago, so
    a player only frees up minutes in proportion to how much of the recent window
    he actually played (NEW_FRACTION).
  - Vacated minutes are shared among teammates by minutes already played, position
    similarity to the missing player, and room left before MAX_MPG.
  - Extra minutes turn into extra stats at MINUTES_EFFICIENCY (< 1: marginal
    minutes are a bit less productive than average ones).
  - Each category is scored as a z-score against the top POOL_SIZE most-owned
    players (percentages by their effect on the team's made shots, not raw %).
  - ESPN only reports ACTIVE / DAY_TO_DAY / OUT, so DAY_TO_DAY counts as DTD_P_OUT
    chance of sitting. Re-run later in the day once the news is firmer.

Requires: pip install requests
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from waiver_watch import load_dotenv, normalize, query_players

# --- Tunable assumptions -----------------------------------------------------
DTD_P_OUT = 0.5            # chance a DAY_TO_DAY player sits
MAX_MPG = 38.0             # nobody absorbs minutes past this
BENCH_FLOOR = 4.0          # deep-bench players still absorb a little
MINUTES_EFFICIENCY = 0.9   # value of an extra minute vs an average one
RECENT_WEIGHT = 0.6        # last-15-days weight in the baseline line
MIN_RECENT_GP = 4          # games needed to trust the last-15-days line
POOL_SIZE = 150            # most-owned players used for z-score scale
PLAYER_LIMIT = 600         # players pulled from ESPN (most owned first)

# ESPN stat id -> (label, kind). kind "count" = per-game total, "pct" = made/attempted.
STAT_NAMES = {0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 6: "REB", 11: "TO", 17: "3PM", 19: "FG%", 20: "FT%"}
DEFAULT_CATS = [0, 6, 3, 2, 1, 17, 11, 19, 20]
COUNTING = ["0", "1", "2", "3", "6", "11", "13", "14", "15", "16", "17"]  # scale with minutes
MINUTES = "28"
GP = "42"
POSITIONS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0"}


def log(msg):
    print(msg, flush=True)


# --- ESPN data ---------------------------------------------------------------
def league_categories(league, season, cookies):
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}/segments/0/leagues/{league}?view=mSettings"
    try:
        r = requests.get(url, headers=UA, cookies=cookies, timeout=20)
        r.raise_for_status()
        ids = [i["statId"] for i in r.json()["settings"]["scoringSettings"]["scoringItems"]]
    except Exception as e:
        log(f"(couldn't read league scoring settings, assuming standard 9-cat: {e})")
        return DEFAULT_CATS
    unknown = [i for i in ids if i not in STAT_NAMES]
    if unknown:
        log(f"(ignoring scoring stats this script doesn't model: {unknown})")
    return [i for i in ids if i in STAT_NAMES]


def nba_games(season, day):
    """{pro team id: (opponent abbrev, is_home, tip time)} for games on `day` (ET), plus abbrevs."""
    url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}?view=proTeamSchedules_wl"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    teams = r.json()["settings"]["proTeams"]
    abbrev = {t["id"]: t["abbrev"] for t in teams}
    games = {}
    for t in teams:
        for period_games in t.get("proGamesByScoringPeriod", {}).values():
            for g in period_games:
                tip = datetime.fromtimestamp(g["date"] / 1000, timezone.utc).astimezone(ET)
                if tip.date() != day:
                    continue
                home, away = g["homeProTeamId"], g["awayProTeamId"]
                games[home] = (abbrev.get(away, "?"), True, tip)
                games[away] = (abbrev.get(home, "?"), False, tip)
    return games, abbrev


def pull_players(league, season, cookies):
    f = {"limit": PLAYER_LIMIT, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}
    return query_players(league, season, cookies, f)


# --- Player profiles ---------------------------------------------------------
def mix(a, b, wa):
    return {k: wa * a.get(k, 0.0) + (1 - wa) * b.get(k, 0.0) for k in set(a) | set(b)}


def window(blocks, key):
    st = blocks.get(key) or {}
    avg = {k: float(v) for k, v in (st.get("averageStats") or {}).items()}
    gp = float((st.get("stats") or {}).get(GP, 0) or 0)
    return avg, gp


def build_profile(p, season):
    pl = p["player"]
    blocks = {st["id"]: st for st in pl.get("stats", [])}
    season_avg, gp_s = window(blocks, f"00{season}")
    recent_avg, gp_l = window(blocks, f"02{season}")
    proj_avg, _ = window(blocks, f"10{season}")

    if gp_s >= 10 or (gp_s >= 3 and not proj_avg):
        base = season_avg
    elif proj_avg:
        base = mix(season_avg, proj_avg, min(gp_s / 10, 1.0)) if gp_s else proj_avg
    else:
        return None
    line = mix(recent_avg, base, RECENT_WEIGHT) if gp_l >= MIN_RECENT_GP else base
    if line.get(MINUTES, 0) < 1:
        return None

    inj = pl.get("injuryStatus")
    if inj in (None, "ACTIVE"):
        p_out = 0.0
    elif inj == "DAY_TO_DAY":
        p_out = DTD_P_OUT
    else:  # OUT, suspensions, anything else
        p_out = 1.0
    return {
        "id": p["id"], "name": pl["fullName"], "team": pl.get("proTeamId", 0),
        "pos": pl.get("defaultPositionId", 3), "status": p.get("status"),
        "inj": inj or "ACTIVE", "p_out": p_out, "line": line, "gp_recent": gp_l,
        "own": (pl.get("ownership") or {}).get("percentOwned", 0.0) or 0.0,
    }


def pos_similarity(a, b):
    d = abs(a - b)
    return 1.0 if d == 0 else 0.6 if d == 1 else 0.3


# --- Minutes redistribution --------------------------------------------------
def team_boosts(roster, p_out):
    """Extra minutes each teammate absorbs. roster: profiles of one team; p_out: {id: chance out}.

    Returns {player id: (extra minutes, {absent player name: minutes from him})}.
    """
    team_games = max((m["gp_recent"] for m in roster), default=0)
    vacated = []  # (profile, minutes freed)
    for m in roster:
        po = p_out[m["id"]]
        if po <= 0:
            continue
        new_frac = min(1.0, m["gp_recent"] / team_games) if team_games else 1.0
        vacated.append((m, m["line"][MINUTES] * po * new_frac))
    total = sum(v for _, v in vacated)
    if total < 1.0:
        return {}

    weights, gains = {}, defaultdict(float)
    for m in roster:
        avail = 1.0 - p_out[m["id"]]
        if avail <= 0:
            continue
        sim = sum(v * pos_similarity(m["pos"], o["pos"]) for o, v in vacated) / total
        weights[m["id"]] = (m["line"][MINUTES] + BENCH_FLOOR) * sim * avail
    by_id = {m["id"]: m for m in roster}

    remaining = total
    for _ in range(5):  # hand out minutes, re-sharing whatever players are too capped to take
        room = {i: max(0.0, MAX_MPG - by_id[i]["line"][MINUTES] - gains[i]) for i in weights}
        open_w = sum(w for i, w in weights.items() if room[i] > 0)
        if remaining < 0.05 or open_w <= 0:
            break
        given = 0.0
        for i, w in weights.items():
            if room[i] <= 0:
                continue
            take = min(remaining * w / open_w, room[i])
            gains[i] += take
            given += take
        remaining -= given

    out = {}
    for i, g in gains.items():
        if g > 0.05:
            out[i] = (g, {o["name"]: g * v / total for o, v in vacated if o["id"] != i})
    return out


def boosted_line(m, extra):
    line = dict(m["line"])
    scale = 1.0 + MINUTES_EFFICIENCY * extra / line[MINUTES]
    for k in COUNTING:
        if k in line:
            line[k] *= scale
    line[MINUTES] += extra
    return line


# --- Scoring -----------------------------------------------------------------
def cat_values(line, cats, lg_fg, lg_ft):
    v = {}
    for c in cats:
        if c == 19:
            v[c] = line.get("13", 0) - lg_fg * line.get("14", 0)
        elif c == 20:
            v[c] = line.get("15", 0) - lg_ft * line.get("16", 0)
        elif c == 11:
            v[c] = -line.get("11", 0)
        else:
            v[c] = line.get(str(c), 0)
    return v


class Scorer:
    """z-scores a stat line against the most-owned players' baseline lines."""

    def __init__(self, pool, cats, weights=None):
        self.cats = cats
        self.weights = weights or {}  # category id -> multiplier (default 1, 0 = ignore)
        fgm = sum(m["line"].get("13", 0) for m in pool)
        fga = sum(m["line"].get("14", 0) for m in pool) or 1
        ftm = sum(m["line"].get("15", 0) for m in pool)
        fta = sum(m["line"].get("16", 0) for m in pool) or 1
        self.lg_fg, self.lg_ft = fgm / fga, ftm / fta
        rows = [cat_values(m["line"], cats, self.lg_fg, self.lg_ft) for m in pool]
        self.mean, self.std = {}, {}
        for c in cats:
            xs = [r[c] for r in rows]
            mu = sum(xs) / len(xs)
            self.mean[c] = mu
            self.std[c] = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0

    def z(self, line):
        """Weighted z-score per category (weights apply to every player alike)."""
        v = cat_values(line, self.cats, self.lg_fg, self.lg_ft)
        return {c: self.weights.get(c, 1.0) * (v[c] - self.mean[c]) / self.std[c] for c in self.cats}


# --- Main analysis -----------------------------------------------------------
def analyze(profiles, games, cats, scorer, out_override=None):
    """Score every free agent playing today. out_override: {player id: chance out} for what-ifs."""
    by_team = defaultdict(list)
    for m in profiles:
        if m["team"] in games:
            by_team[m["team"]].append(m)

    results = []
    for team, roster in by_team.items():
        p_out = {m["id"]: m["p_out"] for m in roster}
        if out_override:
            p_out.update({i: v for i, v in out_override.items() if i in p_out})
        boosts = team_boosts(roster, p_out)
        for m in roster:
            if m["status"] != "FREEAGENT" or p_out[m["id"]] >= 1.0:
                continue
            extra, causes = boosts.get(m["id"], (0.0, {}))
            base_z = scorer.z(m["line"])
            adj_z = scorer.z(boosted_line(m, extra)) if extra else base_z
            base, adj = sum(base_z.values()), sum(adj_z.values())
            score = adj * (1 - p_out[m["id"]]) if adj > 0 else adj  # a possible scratch only hurts
            best = sorted(adj_z, key=adj_z.get, reverse=True)[:2]
            results.append({"m": m, "extra": extra, "causes": causes, "base": base, "adj": adj,
                            "score": score, "best": [STAT_NAMES[c] for c in best]})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def pending_scenarios(profiles, games, cats, scorer, results, top_n=3, limit=6):
    """For key day-to-day players on teams playing today: who gains most if he's ruled out."""
    current = {r["m"]["id"]: r["score"] for r in results}
    dtd = [m for m in profiles if m["inj"] == "DAY_TO_DAY" and m["team"] in games
           and m["line"][MINUTES] >= 18]
    dtd.sort(key=lambda m: m["line"][MINUTES], reverse=True)
    scenarios = []
    for m in dtd[:limit]:
        alt = analyze(profiles, games, cats, scorer, out_override={m["id"]: 1.0})
        gain = [(r["score"] - current.get(r["m"]["id"], r["score"]), r) for r in alt if r["m"]["team"] == m["team"]]
        gain = [g for g in sorted(gain, key=lambda x: x[0], reverse=True) if g[0] > 0.05][:top_n]
        if gain:
            scenarios.append((m, gain))
    return scenarios


by_name = {}  # filled in main: name -> profile, used by reason()


def reason(r):
    if not r["causes"]:
        return "no teammate absences"
    top = sorted(r["causes"].items(), key=lambda kv: kv[1], reverse=True)[:2]
    return ", ".join(f"{n.split()[-1]} {'DTD' if by_name[n]['inj'] == 'DAY_TO_DAY' else 'OUT'}" for n, _ in top)


def matchup(m, games):
    opp, home, tip = games[m["team"]]
    return f"{'vs' if home else '@'} {opp} {tip.strftime('%I:%M%p').lstrip('0').lower()}"


def parse_weights(text, cats):
    """'3pm=3,fg%=0' -> {category id: weight}. Names are case- and punctuation-insensitive."""
    names = {normalize(STAT_NAMES[c]): c for c in cats}
    weights = {}
    for part in text.split(","):
        if not part.strip():
            continue
        key, eq, val = part.partition("=")
        if normalize(key) not in names:
            sys.exit(f'Unknown category "{key.strip()}". Use: {", ".join(STAT_NAMES[c] for c in cats)}')
        try:
            if not eq:
                raise ValueError
            weights[names[normalize(key)]] = float(val)
        except ValueError:
            sys.exit(f'Bad weight "{part.strip()}". Use CATEGORY=NUMBER, e.g. 3PM=3 or FG%=0.')
    return weights


def render(results, scenarios, games, abbrev, day, top, markdown, weights):
    teams_by_id = abbrev
    lines = []
    flag = lambda m: " (DTD)" if m["inj"] == "DAY_TO_DAY" else ""
    title = f"Streamers for {day:%a %b %d}"
    custom = ", ".join(f"{STAT_NAMES[c]} x{w:g}" for c, w in weights.items() if w != 1)
    note = f" Weights: {custom}." if custom else ""
    if markdown:
        lines += [f"## {title}", "", f"{len(games) // 2} games. Ranked by expected value over your categories "
                  f"after shifting vacated minutes to teammates.{note}", "",
                  "| # | Player | Game | Own% | +Min | Base | Adj | Helps | Why |", "|--|--|--|--|--|--|--|--|--|"]
    else:
        lines += [title, f"{len(games) // 2} games, free agents only, ranked by adjusted value.{note}", ""]
    for i, r in enumerate(results[:top], 1):
        m = r["m"]
        name = f"{m['name']} ({POSITIONS.get(m['pos'], '?')}, {teams_by_id.get(m['team'], '?')}){flag(m)}"
        if markdown:
            lines.append(f"| {i} | {name} | {matchup(m, games)} | {m['own']:.0f} | {r['extra']:+.1f} | "
                         f"{r['base']:+.1f} | {r['adj']:+.1f} | {', '.join(r['best'])} | {reason(r)} |")
        else:
            lines.append(f"{i:>2}. {name:<42} {matchup(m, games):<14} own {m['own']:>3.0f}%  "
                         f"{r['extra']:+.1f} min  base {r['base']:+.1f} -> {r['adj']:+.1f}  "
                         f"[{', '.join(r['best'])}]  {reason(r)}")
    if scenarios:
        lines += ["", "### Still undecided (day-to-day)" if markdown else "Still undecided (day-to-day):"]
        for m, gains in scenarios:
            who = ", ".join(f"{r['m']['name']} ({g:+.1f})" for g, r in gains)
            row = f"If {m['name']} ({teams_by_id.get(m['team'], '?')}) is ruled out: {who}"
            lines.append(f"- {row}" if markdown else f"  {row}")
    return "\n".join(lines)


def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    env_league = os.getenv("ESPN_LEAGUE_ID")
    ap = argparse.ArgumentParser(description="Rank free-agent streamers by injury-driven minutes.")
    ap.add_argument("--league", type=int, default=int(env_league) if env_league else None,
                    required=not env_league, help="ESPN league ID (defaults to ESPN_LEAGUE_ID in .env)")
    ap.add_argument("--season", type=int, default=2027, help="ESPN season year (2026-27 = 2027)")
    ap.add_argument("--date", help="Day to plan for, YYYY-MM-DD in US Eastern (default: today)")
    ap.add_argument("--top", type=int, default=15, help="How many free agents to list")
    ap.add_argument("--weights", default="",
                    help='Category multipliers, e.g. "3PM=3,FG%%=0" (default 1 each). Applied to every player.')
    ap.add_argument("--punt", default="", help='Shorthand for weight 0, e.g. "TO,FT%%"')
    ap.add_argument("--markdown", action="store_true", help="Markdown output (for a GitHub Actions summary)")
    ap.add_argument("--push", type=int, default=0, metavar="N", help="Send the top N to your phone via ntfy")
    ap.add_argument("--ntfy", default=os.getenv("NTFY_TOPIC"), help="ntfy topic (defaults to NTFY_TOPIC)")
    args = ap.parse_args()

    day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.now(ET).date()
    cookies = {}
    if os.getenv("ESPN_S2") and os.getenv("ESPN_SWID"):
        cookies = {"espn_s2": os.environ["ESPN_S2"], "SWID": os.environ["ESPN_SWID"]}

    cats = league_categories(args.league, args.season, cookies)
    spec = args.weights + "," + ",".join(f"{c.strip()}=0" for c in args.punt.split(",") if c.strip())
    weights = parse_weights(spec, cats)

    games, abbrev = nba_games(args.season, day)
    if not games:
        log(f"No NBA games on {day:%a %b %d}.")
        return

    try:
        raw = pull_players(args.league, args.season, cookies)
    except requests.RequestException as e:
        sys.exit(f"Couldn't load players from ESPN: {e}\nIf your league is private, set ESPN_S2 and ESPN_SWID.")
    profiles = [m for m in (build_profile(p, args.season) for p in raw) if m and m["team"]]
    by_name.update({m["name"]: m for m in profiles})

    pool = sorted(profiles, key=lambda m: m["own"], reverse=True)[:POOL_SIZE]
    scorer = Scorer(pool, cats, weights)
    results = analyze(profiles, games, cats, scorer)
    scenarios = pending_scenarios(profiles, games, cats, scorer, results)

    text = render(results, scenarios, games, abbrev, day, args.top, args.markdown, weights)
    log(text)

    if args.push and args.ntfy and results:
        top = results[:args.push]
        body = "\n".join(f"{r['m']['name']} ({abbrev.get(r['m']['team'], '?')}) {r['extra']:+.1f} min, {reason(r)}"
                         for r in top)
        try:
            requests.post(f"https://ntfy.sh/{args.ntfy}", data=body.encode(), timeout=10,
                          headers={"Title": f"Streamers for {day:%a %b %d}", "Priority": "default"})
        except requests.RequestException as e:
            log(f"(ntfy push failed: {e})")


if __name__ == "__main__":
    main()
