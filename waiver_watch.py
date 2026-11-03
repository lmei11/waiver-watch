#!/usr/bin/env python3
"""
Watch an ESPN Fantasy Basketball player on waivers and notify you the moment
your league's waivers process (status flips from WAIVERS to FREEAGENT or ONTEAM).

Usage:
    python3 waiver_watch.py --league 123456 --player 4277905
    python3 waiver_watch.py --league 123456 --player 4277905 --ntfy my-secret-topic
    python3 waiver_watch.py --name "jokic"      # search players on waivers by name
    python3 waiver_watch.py --auto              # just log tonight's processing time

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
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

POLL_SECONDS = 30
DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "waiver_log.csv")
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


# No league ID or anything account-specific: the log is committed to the repo.
LOG_FIELDS = [
    "night", "player_id", "player_name", "outcome",
    "last_waivers_utc", "detected_utc", "window_seconds", "processed_est_local",
]


def upgrade_log_format(path):
    """Rewrite a log with an older column layout (e.g. one with league_id) into LOG_FIELDS."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == LOG_FIELDS:
            return
        rows = list(reader)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log("Updated the log file to the current format.")


def record_result(path, player, name, outcome, last_waivers_at, detected_at, tz):
    """Append one night's result to the CSV log.

    The real processing moment is somewhere between the last poll that still showed
    WAIVERS and the first poll that didn't, so both ends are stored. The midpoint is
    the best single estimate. NOT_IN_WINDOW rows mean the window ended with no change.
    """
    ref = detected_at or last_waivers_at
    est = last_waivers_at + (detected_at - last_waivers_at) / 2 if detected_at else None
    row = {
        "night": ref.astimezone(tz).date().isoformat(),
        "player_id": player,
        "player_name": name,
        "outcome": outcome,
        "last_waivers_utc": last_waivers_at.isoformat(timespec="seconds"),
        "detected_utc": detected_at.isoformat(timespec="seconds") if detected_at else "",
        "window_seconds": round((detected_at - last_waivers_at).total_seconds()) if detected_at else "",
        "processed_est_local": est.astimezone(tz).isoformat(timespec="seconds") if est else "",
    }
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new_file:
            upgrade_log_format(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        if est:
            log(f"Logged: processed between {last_waivers_at.astimezone(tz):%I:%M:%S} and "
                f"{detected_at.astimezone(tz):%I:%M:%S %p %Z}.")
        else:
            log("Logged: no processing during the window.")
    except OSError as e:
        log(f"(couldn't write log file {path}: {e})")


def window_night(now, start_h, end_h):
    """Calendar date (in now's time zone) of the window that's open now or opens next."""
    if in_window(now, start_h, end_h):
        start = next_window_start(now - timedelta(days=1), start_h)  # the one that opened
    else:
        start = next_window_start(now, start_h)
    return start.date()


def already_logged(path, night):
    if not path or not os.path.exists(path):
        return False
    with open(path, newline="", encoding="utf-8") as f:
        return any(r.get("night") == night.isoformat() for r in csv.DictReader(f))


def pick_tracking_player(league, season, cookies, tz, night):
    """For data collection: pick any player on waivers who's due to clear on `night`.

    Every player due that night clears in the same processing run, so any one of
    them shows when it happened. Returns a player ID, or None if nobody clears.
    """
    players = list_waiver_players(league, season, cookies)
    if not players:
        log("No players are on waivers in this league right now.")
        return None

    dated = [p for p in players if p.get("waiverProcessDate")]
    if not dated:
        # ESPN didn't say when they clear - watch one anyway (may log NOT_IN_WINDOW).
        log(f"ESPN gave no clear dates; watching {full_name(players[0])} anyway.")
        return players[0]["id"]

    def clears(p):
        return datetime.fromtimestamp(p["waiverProcessDate"] / 1000, tz)

    tonight = [p for p in dated if clears(p).date() <= night]
    if not tonight:
        upcoming = ", ".join(f"{full_name(p)} ({clears(p):%a})" for p in sorted(dated, key=clears)[:5])
        log(f"Nobody on waivers clears on {night:%a %b %d}. On waivers: {upcoming}")
        return None
    p = min(tonight, key=clears)
    log(f"Tracking with {full_name(p)} (due to clear {clears(p):%a %I:%M %p %Z}).")
    return p["id"]


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


def list_waiver_players(league, season, cookies):
    """All players currently on waivers in this league."""
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

    return [p for p in waiver_players if p.get("status") == "WAIVERS"]


def full_name(p):
    return p.get("player", {}).get("fullName", str(p.get("id")))


def find_waiver_player_by_name(league, season, cookies, name):
    """Search the players currently on waivers in this league by (partial) name."""
    waiver_players = list_waiver_players(league, season, cookies)

    query = normalize(name).split()
    matches = [
        p for p in waiver_players
        if all(word in normalize(p.get("player", {}).get("fullName", "")) for word in query)
    ]

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
    who.add_argument("--auto", action="store_true",
                     help="Data-collection mode: pick any player due to clear tonight, log the "
                          "processing time, and send no phone push. Exits if nobody clears tonight.")
    ap.add_argument("--skip-if-logged", action="store_true",
                    help="Exit right away if tonight is already in the log")
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
    ap.add_argument("--log", default=os.getenv("WAIVER_LOG", DEFAULT_LOG),
                    help=f"CSV file to record processing times in (default {DEFAULT_LOG})")
    ap.add_argument("--no-log", action="store_true", help="Don't record this run")
    args = ap.parse_args()
    if isinstance(args.window, str):  # default value isn't run through type= by every Python version
        args.window = parse_window(args.window)
    if args.no_ntfy or args.auto:
        args.ntfy = None

    try:
        tz = ZoneInfo(args.tz)
    except Exception:
        sys.exit(f'Unknown time zone "{args.tz}". Use a name like America/New_York.\n'
                 "(On Windows you may need: pip install tzdata)")
    start_h, end_h = args.window
    log_path = None if args.no_log else args.log
    night = window_night(datetime.now(tz), start_h, end_h)

    if args.skip_if_logged and already_logged(log_path, night):
        log(f"{night:%a %b %d} is already logged. Nothing to do.")
        return

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
    elif args.auto:
        args.one_window = True  # data collection is per night; never roll over
        try:
            args.player = pick_tracking_player(args.league, args.season, cookies, tz, night)
        except requests.RequestException as e:
            sys.exit(f"Couldn't list players on waivers: {e}")
        if args.player is None:
            log("Nothing to track tonight.")
            return

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

    window_label = f"{hour_label(start_h)}-{hour_label(end_h)} {datetime.now(tz):%Z}"
    if args.now:
        log(f"Polling every {POLL_SECONDS}s (ignoring the time window). Ctrl+C to stop.")
    else:
        log(f"Polling every {POLL_SECONDS}s during {window_label} only. Ctrl+C to stop.")

    if log_path:
        log(f"Results will be logged to {log_path}")

    last_check = time.time()  # the initial check above counts
    last_waivers_at = datetime.now(timezone.utc)
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
                    if log_path:
                        record_result(log_path, args.player, name, "NOT_IN_WINDOW",
                                      last_waivers_at, None, tz)
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
            last_waivers_at = datetime.now(timezone.utc)
            log("Still on waivers.")
            continue

        # Waivers processed somewhere between the last WAIVERS poll and now.
        if log_path:
            record_result(log_path, args.player, name, status,
                          last_waivers_at, datetime.now(timezone.utc), tz)

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