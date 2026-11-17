"""Forecast a launch day's Twitch viewers, hour by hour, from its first hour.

Entry for make.fun's gta6-twitch-ccu-forecast-01 (program/v1, Python 3.13,
standard library only). For every case file in the input directory the program
writes one forecast: five quantiles of the category's concurrent viewers for
each of hours 2 to 24 of the release day, scored by pinball loss on the log
scale.

The model is fitted at run time from the case itself, so nothing is memorised
for one game:

* The anchor is hour 1's print, p1. Every forecast is p1 times a ratio.
* The case's `comparables` are past big launches with full hourly curves.
  Each one contributes, for every hour h, the log ratio ln(v_h / v_1) of its
  own release day: the shape a launch day takes after its first hour.
* A comparable counts more the closer its own hour-1 audience is to this
  case's: a Gaussian kernel on the log hour-1 distance (bandwidth 1.1). Big
  launches behave like big launches; a launch that started at 400 viewers
  because its unlock was still hours away says little about one that opened
  at a million.
* Hour h's median is the weighted median of those log ratios, lightly
  smoothed over neighbouring hours. Its spread is two-sided: the weighted
  median deviation below the median and the one above it, each rescaled to
  a sigma, floored, capped and smoothed over a wider window — a launch day
  can surprise further upward (a bigger region wakes up) than downward.
  Quantiles are the normal quantiles of that log-scale distribution, so
  they are non-decreasing by construction.
* Every case this contest grades is a console-only launch, so a comparable
  whose platforms include PC counts 0.6: its simultaneous global unlock
  bends the day's shape away from the rolling local-midnight pattern the
  targets follow.

With no usable comparable the program falls back to persistence: every hour
at hour 1's level with the house baseline's widening spread. The program is
deterministic: same case in, same bytes out.
"""

import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))

# Kernel bandwidth on the log hour-1 distance between a comparable and the case.
SCALE_BANDWIDTH = 1.1
# A comparable never quite vanishes, however far its scale is from the case's.
WEIGHT_FLOOR = 0.001
# Weight kept by a comparable that also launched on PC (see the docstring).
PC_FACTOR = 0.6
# Spread guard rails, in log units.
SIGMA_FLOOR = 0.10
SIGMA_CAP = 1.20
# Persistence fallback spread per square-root hour (the published baseline's).
FALLBACK_SIGMA = 0.15


def parse_time(text):
    """Parse an ISO 8601 UTC stamp like 2026-11-19T00:00:00Z."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_curve(comparable):
    """A comparable's release-day curve: hour offset from day_start -> viewers.

    Offset 0 is hour 1. Points that do not fall on a whole hour are ignored,
    as are repeat points for an hour already seen.
    """
    start = parse_time(comparable["day_start"])
    curve = {}
    for point in comparable.get("hourly") or []:
        offset = (parse_time(point["from"]) - start).total_seconds() / 3600.0
        whole = round(offset)
        if abs(offset - whole) > 1e-6 or whole in curve:
            continue
        curve[int(whole)] = float(point["viewers"])
    return curve


def weighted_quantile(pairs, level):
    """The weighted quantile of (value, weight) pairs, linearly interpolated."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    walked = 0.0
    prev_value, prev_share = ordered[0][0], 0.0
    for value, weight in ordered:
        walked += weight
        share = walked / total
        if share >= level:
            if share == prev_share:
                return value
            return prev_value + (value - prev_value) * (level - prev_share) / (share - prev_share)
        prev_value, prev_share = value, share
    return ordered[-1][0]


def smoothed(values, radius):
    """Each hour's value averaged with the hours within `radius` of it."""
    hours = sorted(values)
    out = {}
    for hour in hours:
        near = [values[other] for other in hours if abs(other - hour) <= radius]
        out[hour] = sum(near) / len(near)
    return out


def has_pc(platforms):
    """True when a comparable's platform list names PC as its own token."""
    tokens = "".join(ch if ch.isalnum() else " " for ch in platforms or "").split()
    return "PC" in tokens


def training_rows(case):
    """(hour-1 viewers, {hour: log ratio}, on-PC) per comparable with an hour 1."""
    rows = []
    for comparable in case.get("comparables") or []:
        curve = day_curve(comparable)
        first = curve.get(0)
        if not first or first <= 0:
            continue
        log_first = math.log(max(first, 1.0))
        ratios = {}
        for hour in HOURS:
            viewers = curve.get(hour - 1)
            if viewers is not None:
                ratios[hour] = math.log(max(viewers, 1.0)) - log_first
        if ratios:
            rows.append((first, ratios, has_pc(comparable.get("platforms"))))
    return rows


def weights_for(rows, p1):
    """Gaussian kernel weights on the log hour-1 distance to this case."""
    log_p1 = math.log(max(p1, 1.0))
    weights = []
    for first, _, on_pc in rows:
        distance = math.log(max(first, 1.0)) - log_p1
        weight = math.exp(-0.5 * (distance / SCALE_BANDWIDTH) ** 2)
        if on_pc:
            weight *= PC_FACTOR
        weights.append(max(weight, WEIGHT_FLOOR))
    return weights


def model_quantiles(case, p1, levels):
    """{hour: [quantiles]} from the comparables, or None when there are none."""
    rows = training_rows(case)
    if not rows:
        return None
    weights = weights_for(rows, p1)
    medians = {}
    sigmas_below = {}
    sigmas_above = {}
    for hour in HOURS:
        pairs = [(ratios[hour], weight) for (_, ratios, _), weight in zip(rows, weights) if hour in ratios]
        if not pairs:
            continue
        median = weighted_quantile(pairs, 0.5)
        below = [(median - value, weight) for value, weight in pairs if value < median]
        above = [(value - median, weight) for value, weight in pairs if value > median]
        sigma_below = 1.4826 * weighted_quantile(below, 0.5) if below else None
        sigma_above = 1.4826 * weighted_quantile(above, 0.5) if above else None
        if sigma_below is None:
            sigma_below = sigma_above
        if sigma_above is None:
            sigma_above = sigma_below
        medians[hour] = median
        sigmas_below[hour] = min(max(sigma_below, SIGMA_FLOOR), SIGMA_CAP)
        sigmas_above[hour] = min(max(sigma_above, SIGMA_FLOOR), SIGMA_CAP)
    if not medians:
        return None
    medians = smoothed(medians, 1)
    sigmas_below = smoothed(sigmas_below, 2)
    sigmas_above = smoothed(sigmas_above, 2)
    normal = statistics.NormalDist()
    out = {}
    for hour, median in medians.items():
        row = []
        for level in levels:
            z = normal.inv_cdf(level)
            sigma = sigmas_below[hour] if z < 0 else sigmas_above[hour]
            value = p1 * math.exp(median + z * sigma)
            row.append(min(max(value, 0.01), 1_000_000_000.0))
        out[hour] = row
    return out


def fallback_quantiles(p1, levels):
    """Persistence: every hour at hour 1's level, spread widening by root hour."""
    normal = statistics.NormalDist()
    out = {}
    for hour in HOURS:
        sigma = FALLBACK_SIGMA * math.sqrt(hour - 1)
        out[hour] = [min(p1 * math.exp(normal.inv_cdf(level) * sigma), 1_000_000_000.0) for level in levels]
    return out


def round2(value):
    """Round to 2 decimals, half up, as the forecast format allows fractions."""
    return math.floor(value * 100 + 0.5) / 100


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    first = next((row for row in case.get("known") or [] if row.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    quantiles = model_quantiles(case, p1, levels)
    if quantiles is None:
        quantiles = fallback_quantiles(p1, levels)
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        values = quantiles[hour]
        # Non-decreasing, as the format demands (they are by construction).
        for i in range(1, len(values)):
            values[i] = max(values[i], values[i - 1])
        hours.append({"hour": hour, "q": [round2(value) for value in values]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
