"""Features for a one-day-ahead forecast.

The forecast for day D is issued at 4:00 a.m. local time and covers the hours
5:00 to 23:00 of D. Every feature for D is built only from data available at
4:00 a.m. on D, so the backtest is honest: no peeking at the day being
predicted. tests/test_features.py checks this.
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from forecast.data import TZ

ISSUE_HOUR = 4
FORECAST_HOURS = list(range(5, 24))
TARGETS = ["empty", "no_ebike"]

# shrink station-level rates toward the overall rate when a station has little
# history (k = how many "pseudo hours" of the overall rate to blend in)
SMOOTH_K = 10

FEATURES = [
    "hour",
    "weekend",
    "hours_ahead",
    "bikes_now",
    "electric_now",
    "docks_now",
    "capacity",
    "fill_now",
    "starts_per_hour",
    "ebike_starts_per_hour",
    "days_of_history",
] + [f"{t}_{k}" for t in TARGETS for k in ("station_rate", "station_hour_rate", "station_daytype_rate", "last7_rate", "yesterday", "now")]


def issue_time(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(d, time(ISSUE_HOUR)), tz=TZ)


def _smoothed(group_sum: pd.Series, group_n: pd.Series, prior: float) -> pd.Series:
    return (group_sum + SMOOTH_K * prior) / (group_n + SMOOTH_K)


def build_day(d: date, snaps: pd.DataFrame, hours: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Feature rows (station x forecast hour) for day d, plus targets if known."""
    cutoff = issue_time(d)
    weekend = int(d.weekday() >= 5)
    day_type = "weekend" if weekend else "weekday"

    hist = hours[hours["hour_start"] + pd.Timedelta(hours=1) <= cutoff]
    past = snaps[snaps["time"] <= cutoff]
    if hist.empty or past.empty:
        raise ValueError(f"no history before {cutoff}")

    # station state at issue time = last snapshot at or before 4:00
    last = past.sort_values("time").groupby("station_id").tail(1).set_index("station_id")
    stale = last["time"] < cutoff - pd.Timedelta(hours=3)
    last = last[~stale]

    stations = last.index
    grid = pd.MultiIndex.from_product([stations, FORECAST_HOURS], names=["station_id", "hour"]).to_frame(index=False)
    grid["date"] = d
    grid["weekend"] = weekend
    grid["hours_ahead"] = grid["hour"] - ISSUE_HOUR
    grid["bikes_now"] = grid["station_id"].map(last["bikes"])
    grid["electric_now"] = grid["station_id"].map(last["electric"])
    grid["docks_now"] = grid["station_id"].map(last["docks"])

    cap = (past["bikes"] + past["docks"]).groupby(past["station_id"]).median()
    grid["capacity"] = grid["station_id"].map(cap)
    grid["fill_now"] = grid["bikes_now"] / grid["capacity"].replace(0, np.nan)

    prof = profile[profile["day_type"] == day_type].set_index(["station_id", "hour"])
    idx = pd.MultiIndex.from_frame(grid[["station_id", "hour"]])
    grid["starts_per_hour"] = prof["starts_per_hour"].reindex(idx).to_numpy()
    grid["ebike_starts_per_hour"] = prof["ebike_starts_per_hour"].reindex(idx).to_numpy()

    hist = hist.assign(weekend=pd.to_datetime(hist["date"]).dt.dayofweek.ge(5).astype(int))
    grid["days_of_history"] = grid["station_id"].map(hist.groupby("station_id")["date"].nunique()).fillna(0)
    yesterday = d - timedelta(days=1)
    week_ago = d - timedelta(days=7)

    for t in TARGETS:
        prior = hist[t].mean()

        g = hist.groupby("station_id")[t]
        grid[f"{t}_station_rate"] = grid["station_id"].map(_smoothed(g.sum(), g.count(), prior)).fillna(prior)

        g = hist.groupby(["station_id", "hour"])[t]
        sh = _smoothed(g.sum(), g.count(), prior).reindex(idx).to_numpy()
        grid[f"{t}_station_hour_rate"] = np.where(np.isnan(sh), grid[f"{t}_station_rate"], sh)

        same = hist[hist["weekend"] == weekend].groupby("station_id")[t]
        grid[f"{t}_station_daytype_rate"] = (
            grid["station_id"].map(_smoothed(same.sum(), same.count(), prior)).fillna(grid[f"{t}_station_rate"])
        )

        recent = hist[hist["date"] >= week_ago].groupby("station_id")[t]
        grid[f"{t}_last7_rate"] = grid["station_id"].map(_smoothed(recent.sum(), recent.count(), prior)).fillna(prior)

        y = hist[hist["date"] == yesterday].set_index(["station_id", "hour"])[t]
        grid[f"{t}_yesterday"] = y.reindex(idx).to_numpy()  # NaN if missing; the model handles it

    grid["empty_now"] = (grid["bikes_now"] == 0).astype(int)
    grid["no_ebike_now"] = (grid["electric_now"] == 0).astype(int)

    # attach the answers if the day has already happened
    actual = hours[hours["date"] == d].set_index(["station_id", "hour"])[TARGETS]
    for t in TARGETS:
        grid[t] = actual[t].reindex(idx).to_numpy()

    return grid


def build_days(days, snaps, hours, profile) -> pd.DataFrame:
    return pd.concat([build_day(d, snaps, hours, profile) for d in days], ignore_index=True)
