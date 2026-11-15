"""Model, baselines and scoring.

Three baselines to beat:
- history: the station's own smoothed rate for that hour of day
- yesterday: did it happen at the same station and hour yesterday
- now: is the station already empty at 4 a.m. (assume it stays that way)

The model is a gradient-boosted tree classifier on the features in
forecast/features.py, one per target.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from forecast.features import FEATURES, TARGETS

TOP_K = 25
# the column that says whether the station still had what we're predicting at 4 a.m.
STOCK_AT_ISSUE = {"empty": "bikes_now", "no_ebike": "electric_now"}


def fit(train: pd.DataFrame, target: str) -> HistGradientBoostingClassifier:
    rows = train.dropna(subset=[target])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    model.fit(rows[FEATURES], rows[target].astype(int))
    return model


def predict(models: dict, day: pd.DataFrame) -> pd.DataFrame:
    out = day.copy()
    for t in TARGETS:
        out[f"p_{t}"] = models[t].predict_proba(day[FEATURES])[:, 1]
        out[f"base_history_{t}"] = day[f"{t}_station_hour_rate"]
        out[f"base_yesterday_{t}"] = day[f"{t}_yesterday"].fillna(day[f"{t}_station_hour_rate"])
        # ties broken by the station's history so the ranking isn't arbitrary
        out[f"base_now_{t}"] = 0.9 * day[f"{t}_now"] + 0.1 * day[f"{t}_station_hour_rate"]
    return out


def score(pred: pd.DataFrame, target: str) -> list[dict]:
    """Metrics for the model and both baselines on one or more scored days."""
    rows = pred.dropna(subset=[target])
    y = rows[target].astype(int)
    results = []
    for name, col in [
        ("model", f"p_{target}"),
        ("history", f"base_history_{target}"),
        ("yesterday", f"base_yesterday_{target}"),
        ("now", f"base_now_{target}"),
    ]:
        p = rows[col].clip(0, 1)
        ranked = rows.assign(p=p)
        top = ranked.nlargest(TOP_K, "p")
        # the harder, more useful version: only stations that still had bikes at
        # 4 a.m., so a station that's broken or empty all day can't pad the score
        stocked = ranked[ranked[STOCK_AT_ISSUE[target]] > 0].nlargest(TOP_K, "p")
        results.append(
            {
                "target": target,
                "method": name,
                "n": len(rows),
                "base_rate": y.mean(),
                "brier": brier_score_loss(y, p),
                "auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
                "avg_precision": average_precision_score(y, p) if y.nunique() == 2 else np.nan,
                f"top{TOP_K}_hit_rate": top[target].mean(),
                f"top{TOP_K}_stocked_hit_rate": stocked[target].mean(),
            }
        )
    return results
