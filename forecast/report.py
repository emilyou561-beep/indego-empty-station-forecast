"""Charts and the auto-updated part of the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAVY = "#00274C"
GRAY = "#A3ACB9"
INK = "#1F2933"
MUTED = "#6B7280"

LABELS = {
    "model": "Model",
    "now": "Empty at 4 a.m. stays empty",
    "history": "Station's usual rate at that hour",
    "yesterday": "Same as yesterday",
}

START = "<!-- daily:start -->"
END = "<!-- daily:end -->"


def _style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def backtest_chart(bt: pd.DataFrame, out: Path) -> None:
    """Top-25 hit rate among stations that still had bikes at 4 a.m., by method."""
    order = ["model", "now", "history", "yesterday"]
    titles = {"empty": "No bikes at all", "no_ebike": "No e-bikes"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, target in zip(axes, ["empty", "no_ebike"]):
        m = bt[bt["target"] == target].groupby("method")["top25_stocked_hit_rate"].mean().reindex(order)
        colors = [NAVY if k == "model" else GRAY for k in order]
        ax.barh([LABELS[k] for k in order], m.values, color=colors, height=0.55)
        for i, v in enumerate(m.values):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", color=INK, fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.set_title(titles[target], loc="left", color=INK, fontsize=11)
        _style(ax)
    axes[0].invert_yaxis()  # shared y axis, so once is enough
    fig.suptitle(
        "Stations that had bikes at 4 a.m.: of the 25 station-hours each method flagged\nas most likely to run out later that day, how many did?",
        x=0.01, ha="left", color=INK, fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def daily_chart(scores: pd.DataFrame, out: Path) -> None:
    """Live record: model vs. the best baseline, one point per forecast day."""
    s = scores[scores["target"] == "empty"].copy()
    s["date"] = pd.to_datetime(s["date"])
    model = s[s["method"] == "model"].set_index("date")["top25_stocked_hit_rate"]
    base = s[s["method"] != "model"].groupby("date")["top25_stocked_hit_rate"].max()

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(base.index, base.values, color=GRAY, linewidth=2, marker="o", markersize=5, label="Best baseline that day")
    ax.plot(model.index, model.values, color=NAVY, linewidth=2, marker="o", markersize=5, label="Model")
    if len(model):
        ax.annotate("Model", (model.index[-1], model.values[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
        ax.annotate("Best baseline", (base.index[-1], base.values[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_title("Live forecasts, scored the next day: top-25 hit rate for 'no bikes' (stations with bikes at 4 a.m.)", loc="left", color=INK, fontsize=11.5)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.tick_params(colors=MUTED, length=0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _hours(hs) -> str:
    hs = sorted(hs)
    def fmt(h):
        return f"{h % 12 or 12}{'a' if h < 12 else 'p'}"
    return ", ".join(fmt(h) for h in hs[:4]) + ("…" if len(hs) > 4 else "")


def readme_block(forecast: pd.DataFrame, day, scores: pd.DataFrame) -> str:
    lines = [START, "", f"### Today's forecast ({day:%a %b %-d, %Y})", ""]
    lines.append("Stations that had e-bikes at 4 a.m. but are most likely to **run out of e-bikes** during the morning rush (7–10 a.m.):")
    lines.append("")
    lines.append("| Station | Chance of no e-bike | Hours above 50% |")
    lines.append("|---|---:|---|")
    rush = forecast[forecast["hour"].between(7, 9) & (forecast["electric_now"] > 0)]
    top = rush.groupby(["station_id", "name"])["p_no_ebike"].max().nlargest(8).reset_index()
    for _, r in top.iterrows():
        hrs = rush[(rush["station_id"] == r["station_id"]) & (rush["p_no_ebike"] >= 0.5)]["hour"]
        lines.append(f"| {r['name']} | {r['p_no_ebike']:.0%} | {_hours(hrs) if len(hrs) else '–'} |")
    lines.append("")
    lines.append(f"Full forecast: [`forecasts/{day}.csv`](forecasts/{day}.csv)")
    lines.append("")

    if len(scores):
        s = scores[scores["method"] == "model"]
        last7 = s[pd.to_datetime(s["date"]) >= pd.to_datetime(s["date"]).max() - pd.Timedelta(days=6)]
        n_days = s["date"].nunique()
        lines.append(f"### Live track record ({n_days} day{'s' if n_days != 1 else ''} scored so far)")
        lines.append("")
        lines.append("![Live track record](figures/daily_scores.png)")
        lines.append("")
        lines.append("| Last 7 days | Top-25 hit rate | Same, stations with bikes at 4 a.m. | AUC |")
        lines.append("|---|---:|---:|---:|")
        for t, label in [("empty", "No bikes"), ("no_ebike", "No e-bikes")]:
            r = last7[last7["target"] == t]
            lines.append(
                f"| {label} | {r['top25_hit_rate'].mean():.0%} | {r['top25_stocked_hit_rate'].mean():.0%} | {r['auc'].mean():.2f} |"
            )
        lines.append("")
    else:
        lines.append("The first forecasts get scored the morning after they're made; the track record shows up here once they are.")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    a, b = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:a] + block + text[b:])
