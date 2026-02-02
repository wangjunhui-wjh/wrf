import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_eval_dir(name):
    match = re.match(r"pred_(.+)_(\d{4}-\d{2})$", name)
    if not match:
        return name, ""
    return match.group(1), match.group(2)


def _plot_bar(df, metric, out_path, title):
    if df.empty:
        return
    df = df.copy()
    df["label"] = df.apply(lambda r: f"{r['model']}\n{r['month']}" if r["month"] else r["model"], axis=1)
    plt.figure(figsize=(8, 4))
    plt.bar(df["label"], df[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_grouped(df, index_col, metric, out_path, title):
    if df.empty:
        return
    pivot = df.pivot_table(index=index_col, columns="model", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    plt.figure(figsize=(8, 4))
    for model in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[model], marker="o", label=model)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation outputs into tables and plots.")
    parser.add_argument("--eval-dir", default="processed_data/eval", help="Evaluation output directory")
    parser.add_argument("--out-dir", default="processed_data/summary", help="Summary output directory")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows = []
    season_rows = []
    wind_rows = []
    div_rows = []
    radar_paths = []

    for eval_sub in sorted(eval_dir.glob("pred_*")):
        if not eval_sub.is_dir():
            continue
        model, month = _parse_eval_dir(eval_sub.name)

        overall_path = eval_sub / "metrics_overall.csv"
        if overall_path.exists():
            df = pd.read_csv(overall_path)
            df["model"] = model
            df["month"] = month
            overall_rows.append(df)

        season_path = eval_sub / "metrics_by_season.csv"
        if season_path.exists():
            df = pd.read_csv(season_path)
            df["model"] = model
            df["month"] = month
            season_rows.append(df)

        wind_path = eval_sub / "metrics_by_windbin.csv"
        if wind_path.exists():
            df = pd.read_csv(wind_path)
            df["model"] = model
            df["month"] = month
            wind_rows.append(df)

        div_path = eval_sub / "divergence_stats.csv"
        if div_path.exists():
            df = pd.read_csv(div_path)
            df["model"] = model
            df["month"] = month
            div_rows.append(df)

        for radar_plot in eval_sub.glob("*_timeseries.png"):
            radar_paths.append(str(radar_plot))

    if overall_rows:
        overall = pd.concat(overall_rows, ignore_index=True)
        overall.to_csv(out_dir / "metrics_overall_all.csv", index=False)

        overall_all = overall[overall["station_id"] == "ALL"].copy()
        overall_all.to_csv(out_dir / "metrics_overall_all_only.csv", index=False)

        if not overall_all.empty:
            _plot_bar(overall_all, "rmse", out_dir / "overall_rmse.png", "Overall RMSE (ALL)")
            _plot_bar(overall_all, "mae", out_dir / "overall_mae.png", "Overall MAE (ALL)")
            _plot_bar(overall_all, "r", out_dir / "overall_r.png", "Overall R (ALL)")
            _plot_bar(overall_all, "bias", out_dir / "overall_bias.png", "Overall Bias (ALL)")

    if season_rows:
        season = pd.concat(season_rows, ignore_index=True)
        season.to_csv(out_dir / "metrics_by_season_all.csv", index=False)
        _plot_grouped(season, "season", "rmse", out_dir / "season_rmse.png", "RMSE by Season")
        _plot_grouped(season, "season", "mae", out_dir / "season_mae.png", "MAE by Season")
        _plot_grouped(season, "season", "r", out_dir / "season_r.png", "R by Season")

    if wind_rows:
        wind = pd.concat(wind_rows, ignore_index=True)
        wind.to_csv(out_dir / "metrics_by_windbin_all.csv", index=False)
        _plot_grouped(wind, "wind_bin", "rmse", out_dir / "windbin_rmse.png", "RMSE by Wind Bin")
        _plot_grouped(wind, "wind_bin", "mae", out_dir / "windbin_mae.png", "MAE by Wind Bin")
        _plot_grouped(wind, "wind_bin", "r", out_dir / "windbin_r.png", "R by Wind Bin")

    if div_rows:
        div = pd.concat(div_rows, ignore_index=True)
        div.to_csv(out_dir / "divergence_stats_all.csv", index=False)

    if radar_paths:
        radar_index = out_dir / "radar_plots_index.txt"
        with open(radar_index, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(radar_paths)))

    print(f"Summary outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
