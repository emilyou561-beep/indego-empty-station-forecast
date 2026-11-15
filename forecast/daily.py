"""The daily run (GitHub Actions, early every morning Philadelphia time).

    python -m forecast.daily --archive ../indego-station-status-archive

1. Score every earlier forecast whose day is now over.
2. Retrain on all complete days so far and forecast today, 5 a.m. to midnight.
3. Refresh the chart and the "today" section of the README.
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from forecast.backtest import eligible_days
from forecast.data import TZ, complete_days, hourly, load_snapshots, load_stations
from forecast.features import TARGETS, build_day, issue_time
from forecast.model import fit, predict, score
from forecast.report import daily_chart, readme_block, update_readme

FORECASTS = Path("forecasts")
SCORES = Path("results/daily_scores.csv")


def score_pending(hours: pd.DataFrame, done_days: list, today: date) -> pd.DataFrame:
    old = pd.read_csv(SCORES) if SCORES.exists() else pd.DataFrame()
    scored = set(old["date"]) if len(old) else set()
    new = []
    for f in sorted(FORECASTS.glob("*.csv")):
        d = date.fromisoformat(f.stem)
        if d >= today or str(d) in scored or d not in done_days:
            continue
        fc = pd.read_csv(f)
        actual = hours[hours["date"] == d][["station_id", "hour"] + TARGETS]
        fc = fc.drop(columns=[t for t in TARGETS if t in fc]).merge(actual, on=["station_id", "hour"], how="left")
        for t in TARGETS:
            for r in score(fc, t):
                new.append({"date": str(d), **r})
        print("scored", d)
    out = pd.concat([old, pd.DataFrame(new)], ignore_index=True)
    if len(out):
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        out.sort_values(["date", "target", "method"]).to_csv(SCORES, index=False, float_format="%.4f")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--today", type=date.fromisoformat, default=None, help="for reruns; defaults to today in Philadelphia")
    args = ap.parse_args()

    today = args.today or pd.Timestamp.now(tz=TZ).date()
    snaps = load_snapshots(args.archive)
    hours = hourly(snaps)
    stations = load_stations(args.archive)
    profile = pd.read_csv("data/station_hour_profile.csv")

    if snaps["time"].max() < issue_time(today) - pd.Timedelta(hours=1):
        raise SystemExit(f"archive is stale (latest snapshot {snaps['time'].max()}), not forecasting {today}")

    scores = score_pending(hours, complete_days(snaps), today)

    train_days = eligible_days(snaps, today)
    train = pd.concat([build_day(d, snaps, hours, profile) for d in train_days], ignore_index=True)
    models = {t: fit(train, t) for t in TARGETS}
    pred = predict(models, build_day(today, snaps, hours, profile))

    keep = ["station_id", "hour", "bikes_now", "electric_now"] + [c for c in pred.columns if c.startswith(("p_", "base_"))]
    fc = pred[keep].merge(stations[["station_id", "name"]], on="station_id", how="left")
    fc = fc[["station_id", "name", "hour"] + keep[2:]]
    FORECASTS.mkdir(exist_ok=True)
    fc.to_csv(FORECASTS / f"{today}.csv", index=False, float_format="%.4f")
    print(f"forecast for {today}: {fc['station_id'].nunique()} stations, trained on {len(train_days)} days")

    if len(scores):
        daily_chart(scores, Path("figures/daily_scores.png"))
    update_readme(Path("README.md"), readme_block(fc, today, scores))


if __name__ == "__main__":
    main()
