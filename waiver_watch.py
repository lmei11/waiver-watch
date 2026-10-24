#!/usr/bin/env python3
"""
Watch an ESPN Fantasy Basketball player on waivers and notify you the moment
your league's waivers process (status flips from WAIVERS to FREEAGENT or ONTEAM).

Usage:
    python3 waiver_watch.py --league 123456 --player 4277905
    python3 waiver_watch.py --league 123456 --player 4277905 --ntfy my-secret-topic
    python3 waiver_watch.py --name "jokic"      # search players on waivers by name

Polling only happens between 3:00 and 5:00 AM Eastern, wherever your computer
is. Started earlier, the script checks the player once, then waits for 3:00 AM
ET. Use --now to poll right away, --window 2-6 (or WAIVER_WINDOW=2-6 in .env)
to change the hours, or --tz / WAIVER_TZ to change the time zone.

Optional .env file next to this script:
    ESPN_LEAGUE_ID=123456     # lets you omit --league
    NTFY_TOPIC=my-topic       # phone push on every run (skip with --no-ntfy)
    ESPN_S2=...               # private leagues only
    ESPN_SWID={...}           # private leagues only
(Real environment variables, if set, take priority over .env.)

Requires: pip install requests
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

POLL_SECONDS = 30
BACKOFF_SECONDS = 300  # wait this long after a 429 / 5xx before retrying


def load_dotenv(path):
    """Minimal .env loader: KEY=VALUE lines, # comments, optional quotes."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def parse_window(s):
    """'3-5' -> (3, 5). Hours on a 24h clock; '23-2' wraps past midnight."""
    try:
        a, b = (int(x) for x in str(s).split("-"))
        if not (0 <= a <= 23 and 0 <= b <= 24) or a == b:
            raise ValueError
        return a, b
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid window "{s}" - use start-end hours, like 3-5')


def hour_label(h):
    h %= 24
    return f"{h % 12 or 12}{'am' if h < 12 else 'pm'}"


def in_window(dt, start_h, end_h):
    if start_h < end_h:
        return start_h <= dt.hour < end_h
    return dt.hour >= start_h or dt.hour < end_h  # wraps past midnight


def next_window_start(dt, start_h):
    start = dt.replace(hour=start_h, minute=0, second=0, microsecond=0)
    if start <= dt:
        start += timedelta(days=1)
    return start


def sleep_until(target):
    # Sleep in short chunks so a laptop waking from sleep doesn't overshoot badly.
    while (remaining := (target - datetime.now(target.tzinfo)).total_seconds()) > 0:
        time.sleep(min(remaining, 60))


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def query_players(league, season, cookies, player_filter):
    url = (
        f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season}"
        f"/segments/0/leagues/{league}?view=kona_player_info"
    )
    headers = {
        "X-Fantasy-Filter": json.dumps({"players": player_filter}),
        "User-Agent": "Mozilla/5.0",
    }
    r = requests.get(url, headers=headers, cookies=cookies, timeout=15)
    if not r.ok:
        # Include ESPN's response body: it usually says what was wrong with the request.
        body = r.text.strip().replace("\n", " ")[:300]
        raise requests.HTTPError(f"HTTP {r.status_code}: {body or '(empty response)'}", response=r)
    return r.json().get("players", [])


def fetch_player(league, player, season, cookies):
    players = query_players(league, season, cookies, {"filterIds": {"value": [player]}})
    if not players:
        raise ValueError("Player not found in response - check the player ID.")
    return players[0]


def normalize(s):
    """Lowercase and strip accents/punctuation so 'jokic' matches 'Nikola Jokić'."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum() or c == " ").strip()


def find_waiver_player_by_name(league, season, cookies, name):
    """Search the players currently on waivers in this league by (partial) name."""
    # ESPN's API is undocumented, so try a few filter formats, from narrowest to broadest.
    sort = {"sortPercOwned": {"sortPriority": 1, "sortAsc": False}}
    attempts = [
        {"filterStatus": {"value": ["WAIVERS"]}, "limit": 100},
        {"filterStatus": {"value": ["FREEAGENT", "WAIVERS"]}, "limit": 100, "offset": 0, **sort},
        {"filterStatus": {"value": ["FREEAGENT", "WAIVERS"]}, "limit": 50, **sort},
    ]
    errors = []
    waiver_players = None
    for i, f in enumerate(attempts, 1):
        try:
            waiver_players = query_players(league, season, cookies, f)
            break
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in (401, 403):
                raise  # auth problem: trying other formats won't help
            errors.append(f"  format {i}: {e}")
    if waiver_players is None:
        sys.exit("Player search failed - ESPN rejected every request format:\n"
                 + "\n".join(errors)
                 + "\nTry --player with the ID instead, and send me this output.")

    waiver_players = [p for p in waiver_players if p.get("status") == "WAIVERS"]

    query = normalize(name).split()
    matches = [
        p for p in waiver_players
        if all(word in normalize(p.get("player", {}).get("fullName", "")) for word in query)
    ]

    def full_name(p):
        return p.get("player", {}).get("fullName", str(p.get("id")))

    if len(matches) == 1:
        return matches[0]["id"]

    if not matches:
        msg = f'No player on waivers in this league matches "{name}".'
        if waiver_players:
            names = ", ".join(sorted(full_name(p) for p in waiver_players))
            msg += f"\nCurrently on waivers: {names}"
        else:
            msg += "\nNo players are on waivers in this league right now."
        sys.exit(msg)

    # Prefer an exact full-name match if one exists
    exact = [p for p in matches if normalize(full_name(p)) == normalize(name)]
    if len(exact) == 1:
        return exact[0]["id"]

    lines = "\n".join(f"  {full_name(p)} (id {p['id']})" for p in matches)
    sys.exit(f'"{name}" matches more than one player on waivers:\n{lines}\n'
             "Be more specific, or use --player with the ID.")


def notify(title, message, ntfy_topic):
    # Terminal bell + print
    print("\a", end="", flush=True)
    log(f"*** {title}: {message} ***")

    # Desktop notification
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{message}" with title "{title}" sound name "Glass"'],
                check=False,
            )
        elif system == "Linux":
            if shutil.which("notify-send"):  # absent on servers like GitHub Actions
                subprocess.run(["notify-send", title, message], check=False)
        elif system == "Windows":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n=New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon=[System.Drawing.SystemIcons]::Information;"
                f"$n.Visible=$true;$n.ShowBalloonTip(10000,'{title}','{message}','Info');"
                "Start-Sleep -s 10"
            )
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps])
    except Exception as e:
        log(f"(desktop notification failed: {e})")

    # Phone push via ntfy.sh (optional)
    if ntfy_topic:
        try:
            requests.post(
                f"https://ntfy.sh/{ntfy_topic}",
                data=message.encode(),
                headers={"Title": title, "Priority": "urgent"},
                timeout=10,
            )
        except Exception as e:
            log(f"(ntfy push failed: {e})")


def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    env_league = os.getenv("ESPN_LEAGUE_ID")

    ap = argparse.ArgumentParser(description="Notify when ESPN fantasy NBA waivers process.")
    ap.add_argument("--league", type=int, default=int(env_league) if env_league else None,
                    required=not env_league,
                    help="ESPN league ID (defaults to ESPN_LEAGUE_ID in .env)")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--player", type=int, help="ESPN player ID currently on waivers")
    who.add_argument("--name", help='Player name, full or partial (e.g. "jokic"), searched among players on waivers')
    ap.add_argument("--season", type=int, default=2027, help="ESPN season year (2026-27 season = 2027)")
    ap.add_argument("--ntfy", default=os.getenv("NTFY_TOPIC"),
                    help="ntfy.sh topic for phone push notifications (defaults to NTFY_TOPIC in .env)")
    ap.add_argument("--no-ntfy", action="store_true", help="Skip the phone push for this run")
    ap.add_argument("--window", type=parse_window, default=os.getenv("WAIVER_WINDOW", "3-5"),
                    help='Hours to poll in, 24h clock, in --tz time (default "3-5" = 3:00-5:00 AM)')
    ap.add_argument("--now", action="store_true", help="Poll immediately, ignoring the time window")
    ap.add_argument("--one-window", action="store_true",
                    help="Exit (and notify) if the window ends with the player still on waivers, "
                         "instead of waiting for the next night")
    ap.add_argument("--tz", default=os.getenv("WAIVER_TZ", "America/New_York"),
                    help="Time zone the window is in, regardless of where you are (default America/New_York)")
    args = ap.parse_args()
    if isinstance(args.window, str):  # default value isn't run through type= by every Python version
        args.window = parse_window(args.window)
    if args.no_ntfy:
        args.ntfy = None

    cookies = {}
    if os.getenv("ESPN_S2") and os.getenv("ESPN_SWID"):
        cookies = {"espn_s2": os.environ["ESPN_S2"], "SWID": os.environ["ESPN_SWID"]}
        log("ESPN cookies loaded.")
    else:
        log("No ESPN cookies found (fine for public leagues).")

    if args.name:
        try:
            args.player = find_waiver_player_by_name(args.league, args.season, cookies, args.name)
        except requests.RequestException as e:
            sys.exit(f"Player search failed: {e}\n"
                     "If your league is private, set ESPN_S2 and ESPN_SWID.")

    # Initial check
    try:
        p = fetch_player(args.league, args.player, args.season, cookies)
    except Exception as e:
        sys.exit(f"Initial request failed: {e}\n"
                 "If your league is private, set ESPN_S2 and ESPN_SWID.")

    name = p.get("player", {}).get("fullName", f"Player {args.player}")
    status = p.get("status")
    if status != "WAIVERS":
        sys.exit(f"{name} is not on waivers (status: {status}). Nothing to watch.")

    clear_ms = p.get("waiverProcessDate")
    if clear_ms:
        log(f"{name} is on waivers, scheduled to clear {datetime.fromtimestamp(clear_ms / 1000):%a %b %d %I:%M %p}.")

    try:
        tz = ZoneInfo(args.tz)
    except Exception:
        sys.exit(f'Unknown time zone "{args.tz}". Use a name like America/New_York.\n'
                 "(On Windows you may need: pip install tzdata)")

    start_h, end_h = args.window
    window_label = f"{hour_label(start_h)}-{hour_label(end_h)} {datetime.now(tz):%Z}"
    if args.now:
        log(f"Polling every {POLL_SECONDS}s (ignoring the time window). Ctrl+C to stop.")
    else:
        log(f"Polling every {POLL_SECONDS}s during {window_label} only. Ctrl+C to stop.")

    last_check = time.time()  # the initial check above counts
    while True:
        if not args.now and not in_window(datetime.now(tz), start_h, end_h):
            start = next_window_start(datetime.now(tz), start_h)
            local = start.astimezone()
            when = f"{start:%a %I:%M %p %Z}"
            if local.utcoffset() != start.utcoffset():
                when += f" ({local:%I:%M %p} your time)"
            log(f"Outside the {window_label} window. Waiting until {when} to start polling.")
            sleep_until(start)
            log(f"Window open - polling every {POLL_SECONDS}s.")
        else:
            wait = POLL_SECONDS - (time.time() - last_check)
            if wait > 0:
                time.sleep(wait)
            if not args.now and not in_window(datetime.now(tz), start_h, end_h):
                log(f"The {window_label} window ended and {name} is still on waivers.")
                if args.one_window:
                    notify("Waiver watch stopped",
                           f"The {window_label} window ended and {name} is still on waivers.", args.ntfy)
                    return
                continue

        last_check = time.time()
        try:
            p = fetch_player(args.league, args.player, args.season, cookies)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 429 or (code and code >= 500):
                log(f"HTTP {code} - backing off {BACKOFF_SECONDS // 60} min.")
                time.sleep(BACKOFF_SECONDS)
            else:
                log(f"HTTP error: {e}")
            continue
        except Exception as e:
            log(f"Request error: {e}")
            continue

        status = p.get("status")
        if status == "WAIVERS":
            log("Still on waivers.")
            continue

        if status == "FREEAGENT":
            notify("Waivers processed", f"{name} is now a FREE AGENT - go grab him!", args.ntfy)
        elif status == "ONTEAM":
            notify("Waivers processed", f"{name} was claimed by another team.", args.ntfy)
        else:
            notify("Waivers processed", f"{name} status changed to {status}.", args.ntfy)
        break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped.")
