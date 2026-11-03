#!/usr/bin/env python3
"""
Summarize when your league's waivers have processed, using the log written by
waiver_watch.py (data/waiver_log.csv), and fit a simple distribution.

Usage:
    python3 analyze_waivers.py                  # text report
    python3 analyze_waivers.py --markdown       # markdown (used for the GitHub run summary)
    python3 analyze_waivers.py --log path.csv --tz America/New_York

No extra packages needed (standard library only).
"""

import argparse
import csv
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "waiver_log.csv")
PRECISE_SECONDS = 5 * 60   # intervals wider than this are too vague to use as a data point
MIN_FOR_FIT = 5            # fewer nights than this: just list them
BIN_MINUTES = 10


def minutes_of_night(dt):
    """Minutes after midnight, with evening times negative (11:30 PM -> -30)."""
    m = dt.hour * 60 + dt.minute + dt.second / 60
    return m - 1440 if dt.hour >= 12 else m


def clock(m):
    m = round(m) % 1440
    h, mm = divmod(m, 60)
    return f"{h % 12 or 12}:{mm:02d} {'AM' if h < 12 else 'PM'}"


def load(path, tz, league=None):
    if not os.path.exists(path):
        sys.exit(f"No log found at {path} yet. It's created after the first watched night.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if league:
        rows = [r for r in rows if r["league_id"] == str(league)]

    # One result per night: if several runs watched the same night, keep the tightest.
    nights = {}
    for r in rows:
        r["last"] = datetime.fromisoformat(r["last_waivers_utc"]).astimezone(tz)
        r["det"] = datetime.fromisoformat(r["detected_utc"]).astimezone(tz) if r["detected_utc"] else None
        r["width"] = (r["det"] - r["last"]).total_seconds() if r["det"] else None
        key = (r["league_id"], r["night"])
        best = nights.get(key)
        if best is None or (r["det"] and (best["det"] is None or r["width"] < best["width"])):
            nights[key] = r
    return sorted(nights.values(), key=lambda r: r["night"])


def quantile(sorted_vals, q):
    """Linear-interpolated quantile of an already sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def analyze(nights):
    precise, vague, missed = [], [], []
    for r in nights:
        if r["det"] is None:
            missed.append(r)
        elif r["width"] <= PRECISE_SECONDS:
            mid = r["last"] + (r["det"] - r["last"]) / 2
            precise.append((r, minutes_of_night(mid)))
        else:
            vague.append(r)

    res = {"precise": precise, "vague": vague, "missed": missed, "total": len(nights)}
    times = sorted(m for _, m in precise)
    res["times"] = times
    if not times:
        return res

    res["min"], res["max"] = times[0], times[-1]
    res["median"] = statistics.median(times)
    res["mean"] = statistics.fmean(times)
    res["sd"] = statistics.stdev(times) if len(times) > 1 else 0.0
    res["p10"], res["p90"] = quantile(times, 0.10), quantile(times, 0.90)

    if len(times) >= MIN_FOR_FIT:
        nd = statistics.NormalDist(res["mean"], max(res["sd"], 0.5))
        res["normal"] = nd
        # Suggested window: covers the normal fit's middle 98% AND every night seen,
        # rounded out to 5-minute marks, with a 5-minute cushion on each side.
        lo = min(nd.inv_cdf(0.01), res["min"]) - 5
        hi = max(nd.inv_cdf(0.99), res["max"]) + 5
        res["suggest"] = (5 * (lo // 5), 5 * -(-hi // 5))

    by_day = defaultdict(list)
    for r, m in precise:
        by_day[datetime.fromisoformat(r["night"]).strftime("%a")].append(m)
    res["by_day"] = by_day
    res["day_model"] = day_model(by_day, res)
    return res


def window_for(center, sd, lo_seen, hi_seen):
    """98% range of a normal, widened to cover what was seen, +5 min cushion, 5-min marks."""
    nd = statistics.NormalDist(center, max(sd, 0.5))
    lo = min(nd.inv_cdf(0.01), lo_seen) - 5
    hi = max(nd.inv_cdf(0.99), hi_seen) + 5
    return 5 * (lo // 5), 5 * -(-hi // 5)


def day_model(by_day, res):
    """Model each weekday separately, with partial pooling.

    A weekday with only 2-3 nights has a very noisy average, so each day's estimate
    is pulled toward the overall average - strongly when that day has little data or
    the days look alike, barely when it has plenty of data and clearly differs.
    (Normal-normal hierarchical model, variances estimated by method of moments.)
    Also runs a permutation test: how often would shuffling the weekday labels
    produce differences between days at least this big? Small p => days really differ.
    """
    all_times = [t for v in by_day.values() for t in v]
    n_total, n_days = len(all_times), len(by_day)
    if n_total < 10 or sum(1 for v in by_day.values() if len(v) >= 2) < 2:
        return None

    grand = statistics.fmean(all_times)
    means = {d: statistics.fmean(v) for d, v in by_day.items()}

    # Within-day spread (pooled across days) and between-day spread of the true day means.
    ss_within = sum((t - means[d]) ** 2 for d, v in by_day.items() for t in v)
    var_w = ss_within / max(n_total - n_days, 1)
    if n_days > 1:
        var_of_means = statistics.variance(means.values())
        tau2 = max(0.0, var_of_means - statistics.fmean(var_w / len(v) for v in by_day.values()))
    else:
        tau2 = 0.0

    days = {}
    for d, v in by_day.items():
        n = len(v)
        se2 = var_w / n
        weight = tau2 / (tau2 + se2) if tau2 + se2 > 0 else 0.0   # how much to trust this day's own data
        est = weight * means[d] + (1 - weight) * grand
        pred_sd = (var_w + weight * se2 + (1 - weight) * var_w / n_total) ** 0.5
        days[d] = {
            "n": n, "median": statistics.median(v), "est": est, "weight": weight,
            "window": window_for(est, pred_sd, min(v), max(v)),
        }

    # Permutation test on the between-day sum of squares.
    def between_ss(groups):
        return sum(len(g) * (statistics.fmean(g) - grand) ** 2 for g in groups)

    sizes = [len(v) for v in by_day.values()]
    observed = between_ss(list(by_day.values()))
    rng = random.Random(0)
    pool = list(all_times)
    trials, hits = 5000, 0
    for _ in range(trials):
        rng.shuffle(pool)
        groups, i = [], 0
        for s in sizes:
            groups.append(pool[i:i + s])
            i += s
        if between_ss(groups) >= observed - 1e-9:
            hits += 1

    return {"days": days, "p": (hits + 1) / (trials + 1), "tau": tau2 ** 0.5, "within_sd": var_w ** 0.5}


def histogram(times, width=30):
    lo = BIN_MINUTES * int(min(times) // BIN_MINUTES)
    hi = BIN_MINUTES * int(max(times) // BIN_MINUTES)
    bins = {b: 0 for b in range(lo, hi + BIN_MINUTES, BIN_MINUTES)}
    for t in times:
        bins[BIN_MINUTES * int(t // BIN_MINUTES)] += 1
    top = max(bins.values())
    lines = []
    for b, n in bins.items():
        bar = "#" * max(1 if n else 0, round(width * n / top))
        lines.append(f"{clock(b):>8} - {clock(b + BIN_MINUTES):>8} | {bar} {n if n else ''}")
    return lines


def report(res, tzname, md=False):
    out = []
    h = (lambda s: out.append(f"### {s}")) if md else (lambda s: out.extend([s, "-" * len(s)]))
    zone = datetime.now(ZoneInfo(tzname)).strftime("%Z")

    h("Waiver processing times")
    out.append(f"Nights logged: {res['total']}  |  timed precisely: {len(res['precise'])}  |  "
               f"no processing seen in window: {len(res['missed'])}  |  "
               f"too imprecise to use: {len(res['vague'])}")
    out.append("")

    times = res["times"]
    if not times:
        out.append("No precisely timed nights yet.")
        if res["vague"]:
            out.append("Some nights had already processed when polling started - try an earlier window.")
        return "\n".join(out)

    if "normal" not in res:
        out.append(f"Only {len(times)} night(s) so far - need {MIN_FOR_FIT} before fitting a distribution.")
        for r, m in res["precise"]:
            out.append(f"  {r['night']}: ~{clock(m)} {zone}")
        return "\n".join(out)

    nd = res["normal"]
    stats = [
        ("Earliest", clock(res["min"])),
        ("10th percentile", clock(res["p10"])),
        ("Median", clock(res["median"])),
        ("90th percentile", clock(res["p90"])),
        ("Latest", clock(res["max"])),
        ("Normal fit", f"mean {clock(res['mean'])}, std dev {res['sd']:.1f} min"),
        ("Normal fit 95% range", f"{clock(nd.inv_cdf(0.025))} - {clock(nd.inv_cdf(0.975))}"),
    ]
    if md:
        out += ["| | Time (" + zone + ") |", "|---|---|"] + [f"| {k} | {v} |" for k, v in stats]
    else:
        out += [f"  {k:<22}{v}" for k, v in stats]
    out.append("")

    out.append("Distribution (10-minute bins):")
    hist = histogram(times)
    out += (["```"] + hist + ["```"]) if md else ["  " + l for l in hist]
    out.append("")

    lo, hi = res["suggest"]
    out.append(f"Suggested polling window (all days): {clock(lo)} - {clock(hi)} {zone}")
    out.append("")

    dm = res["day_model"]
    h("By day of week")
    out.append("Day = the morning waivers ran (\"Tue\" = overnight Mon into Tue).")
    if dm is None:
        out.append("Not enough data yet - need 10+ timed nights, with at least 2 on two or more weekdays.")
    else:
        rows = []
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            if d in dm["days"]:
                x = dm["days"][d]
                wlo, whi = x["window"]
                rows.append((d, str(x["n"]), clock(x["median"]), clock(x["est"]),
                             f"{x['weight']:.0%}", f"{clock(wlo)} - {clock(whi)}"))
        head = ("Day", "Nights", "Median seen", "Model estimate", "Own-data weight", "Suggested window")
        if md:
            out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            out += ["| " + " | ".join(r) + " |" for r in rows]
        else:
            widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
            out.append("  " + "  ".join(head[i].ljust(widths[i]) for i in range(len(head))))
            out += ["  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(head))) for r in rows]
        out.append("")
        p = dm["p"]
        if p < 0.05:
            verdict = (f"Days look genuinely different (p = {p:.3f}). Typical gap between days: "
                       f"~{dm['tau']:.0f} min; night-to-night spread within a day: ~{dm['within_sd']:.0f} min.")
        elif p < 0.2:
            verdict = (f"Some hint that days differ (p = {p:.2f}), but it could still be noise. "
                       "More nights will tell.")
        else:
            verdict = (f"No real evidence that days differ yet (p = {p:.2f}) - the gaps between "
                       "weekday medians are about what random night-to-night variation would produce.")
        out.append(verdict)
        out.append("Model estimates are pulled toward the overall average, more so for days with "
                   "few nights; \"own-data weight\" shows how much each day's own nights count.")
    out.append("")
    if res["vague"]:
        out.append(f"Note: on {len(res['vague'])} night(s) the change was only seen after a long gap between "
                   "polls (usually: it had already processed when the window opened). If that keeps "
                   "happening, start the window earlier.")
    if res["missed"]:
        out.append(f"Note: on {len(res['missed'])} night(s) nothing processed during the window - "
                   "either the window is too narrow or waivers didn't run that night.")
    if len(times) < 15:
        out.append(f"(Based on {len(times)} nights - treat as rough until you have 15+.)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Analyze logged waiver processing times.")
    ap.add_argument("--log", default=os.getenv("WAIVER_LOG", DEFAULT_LOG))
    ap.add_argument("--tz", default=os.getenv("WAIVER_TZ", "America/New_York"))
    ap.add_argument("--league", help="Only include this league ID")
    ap.add_argument("--markdown", action="store_true", help="Output markdown")
    args = ap.parse_args()

    nights = load(args.log, ZoneInfo(args.tz), args.league)
    print(report(analyze(nights), args.tz, md=args.markdown))


if __name__ == "__main__":
    main()
