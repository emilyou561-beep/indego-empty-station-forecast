"""Rolling backtest: for each past day, train only on the days before it, then score.

    python -m forecast.backtest --archive ../indego-station-status-archive

Writes results/backtest.csv (one row per day x target x method).
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from forecast.data import TZ, complete_days, hourly, load_snapshots
from forecast.features import FEATURES, TARGETS, build_day
from forecast.model import fit, predict, score
from forecast.report import backtest_chart

MIN_TRAIN_DAYS = 3


def eligible_days(snaps: pd.DataFrame, today) -> list:
    days = [d for d in complete_days(snaps) if d < today]
    first = snaps["time"].min().date()
    return [d for d in days if d > first]  # need at least some history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/backtest.csv"))
    args = ap.parse_args()

    snaps = load_snapshots(args.archive)
    hours = hourly(snaps)
    profile = pd.read_csv("data/station_hour_profile.csv")
    today = pd.Timestamp.now(tz=TZ).date()

    days = eligible_days(snaps, today)
    # each day's features only use data from before that day, so building them
    # all up front doesn't leak anything
    feats = {d: build_day(d, snaps, hours, profile) for d in days}

    rows = []
    for i, d in enumerate(days):
        if i < MIN_TRAIN_DAYS:
            continue
        train = pd.concat([feats[x] for x in days[:i]], ignore_index=True)
        models = {t: fit(train, t) for t in TARGETS}
        pred = predict(models, feats[d])
        for t in TARGETS:
            for r in score(pred, t):
                rows.append({"date": d, "train_days": i, **r})
        print(d, "done")

    # which features matter, measured on the last backtest day
    imp = []
    for t in TARGETS:
        test = feats[days[-1]].dropna(subset=[t])
        r = permutation_importance(models[t], test[FEATURES], test[t].astype(int), scoring="roc_auc", n_repeats=5, random_state=0)
        imp += [{"target": t, "feature": f, "auc_drop": m} for f, m in zip(FEATURES, r.importances_mean)]
    pd.DataFrame(imp).sort_values(["target", "auc_drop"], ascending=[True, False]).to_csv(
        "results/feature_importance.csv", index=False, float_format="%.4f"
    )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.4f")
    backtest_chart(out, Path("figures/backtest.png"))
    print(out.groupby(["target", "method"])[["brier", "auc", "avg_precision", "top25_hit_rate"]].mean().round(3))


if __name__ == "__main__":
    main()
