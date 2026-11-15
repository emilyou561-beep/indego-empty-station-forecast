"""Load the station status archive and turn it into one row per station per hour.

The archive (github.com/emilyou561-beep/indego-station-status-archive) has a
snapshot of every station every ~15 minutes. Here a station-hour counts as
"empty" if at least one snapshot in that hour showed zero bikes, and
"no e-bike" if at least one snapshot showed zero electric bikes.

All times are converted to Philadelphia local time, since that's what riders
(and the hour-of-day patterns) follow.
"""

from pathlib import Path

import pandas as pd

TZ = "America/New_York"


def load_snapshots(archive: Path) -> pd.DataFrame:
    files = sorted((archive / "data" / "station_status").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no snapshot files under {archive}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["time"] = pd.to_datetime(df["snapshot_utc"], utc=True).dt.tz_convert(TZ)
    # a station that isn't renting can't be "empty" in a useful sense
    df = df[(df["is_installed"] == 1) & (df["is_renting"] == 1)]
    return df[["time", "station_id", "bikes", "electric", "docks"]].reset_index(drop=True)


def load_stations(archive: Path) -> pd.DataFrame:
    return pd.read_csv(archive / "data" / "station_information.csv")[["station_id", "name", "lat", "lon"]]


def hourly(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per station per local hour, with the two targets."""
    s = snaps.assign(hour_start=snaps["time"].dt.floor("h"))
    out = (
        s.groupby(["station_id", "hour_start"])
        .agg(
            min_bikes=("bikes", "min"),
            min_electric=("electric", "min"),
            n_snaps=("bikes", "size"),
        )
        .reset_index()
    )
    out["empty"] = (out["min_bikes"] == 0).astype(int)
    out["no_ebike"] = (out["min_electric"] == 0).astype(int)
    out["date"] = out["hour_start"].dt.date
    out["hour"] = out["hour_start"].dt.hour
    return out


def complete_days(snaps: pd.DataFrame, min_snaps: int = 60) -> list:
    """Local dates with enough snapshots to trust (GitHub's scheduler skips some runs)."""
    per_day = snaps.groupby(snaps["time"].dt.date)["time"].nunique()
    return sorted(per_day[per_day >= min_snaps].index)
