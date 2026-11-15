"""The forecast for a day must not use anything recorded after 4 a.m. that day."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast.data import TZ, hourly
from forecast.features import FEATURES, FORECAST_HOURS, TARGETS, build_day, issue_time


def fake_snapshots(days=6, stations=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-11-02", tz=TZ)
    times = pd.date_range(start, periods=days * 96, freq="15min")
    rows = []
    for t in times:
        for s in stations:
            bikes = int(rng.integers(0, 4))
            rows.append({"time": t, "station_id": s, "bikes": bikes, "electric": int(rng.integers(0, bikes + 1)), "docks": 10 - bikes})
    return pd.DataFrame(rows)


@pytest.fixture
def setup():
    snaps = fake_snapshots()
    profile = pd.DataFrame(
        [{"station_id": s, "day_type": dt, "hour": h, "starts_per_hour": 1.0, "ebike_starts_per_hour": 0.5}
         for s in (1, 2, 3) for dt in ("weekday", "weekend") for h in range(24)]
    )
    return snaps, profile


def test_no_leakage(setup):
    snaps, profile = setup
    d = date(2026, 11, 6)
    full = build_day(d, snaps, hourly(snaps), profile)

    # wipe out everything after the issue time and rebuild: features must not change
    cut = snaps[snaps["time"] <= issue_time(d)]
    trimmed = build_day(d, cut, hourly(cut), profile)
    pd.testing.assert_frame_equal(full[FEATURES], trimmed[FEATURES])


def test_targets_attached_for_past_days(setup):
    snaps, profile = setup
    out = build_day(date(2026, 11, 6), snaps, hourly(snaps), profile)
    assert len(out) == 3 * len(FORECAST_HOURS)
    for t in TARGETS:
        assert out[t].notna().all()


def test_future_day_has_no_targets(setup):
    snaps, profile = setup
    cut = snaps[snaps["time"] <= issue_time(date(2026, 11, 7))]
    out = build_day(date(2026, 11, 7), cut, hourly(cut), profile)
    for t in TARGETS:
        assert out[t].isna().all()


def test_hourly_flags_empty_if_any_snapshot_is_zero():
    t0 = pd.Timestamp("2026-11-05 08:00", tz=TZ)
    snaps = pd.DataFrame(
        {"time": [t0, t0 + timedelta(minutes=15), t0 + timedelta(minutes=30)],
         "station_id": [1, 1, 1], "bikes": [3, 0, 2], "electric": [1, 0, 1], "docks": [7, 10, 8]}
    )
    h = hourly(snaps)
    assert h["empty"].tolist() == [1]
    assert h["no_ebike"].tolist() == [1]
