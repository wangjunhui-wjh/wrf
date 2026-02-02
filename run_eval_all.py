import argparse
import re
from pathlib import Path

from src.evaluation import (
    _open_pred_dataset,
    _resolve_uv_vars,
    compute_station_metrics,
    evaluate_radar_timeseries,
    divergence_stats,
)


def _parse_name(stem):
    match = re.match(r"pred_(.+)_(\d{4}-\d{2})$", stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for all prediction zarrs.")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr directory")
    parser.add_argument("--months", default="", help="Comma-separated months filter (e.g., 2025-10)")
    parser.add_argument("--models", default="", help="Comma-separated model filter (e.g., espcn,unet)")
    parser.add_argument("--obs-station", default="processed_data/obs/station_30min.csv", help="Station 30min CSV")
    parser.add_argument("--station-meta", default="processed_data/obs/station_meta.csv", help="Station meta CSV")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr path")
    parser.add_argument("--radar-dir", default="processed_data/obs", help="Radar products directory")
    parser.add_argument("--out-dir", default="processed_data/eval", help="Base output directory")
    parser.add_argument("--wind-bins", default="0,2,4,6,8,10,12,15,20", help="Wind bins (comma-separated)")
    parser.add_argument("--skip-radar", action="store_true", help="Skip radar plots")
    parser.add_argument("--skip-divergence", action="store_true", help="Skip divergence stats")
    parser.add_argument("--divergence-max-times", type=int, default=200, help="Max time steps for divergence stats")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred dir not found: {pred_dir}")

    months = {m.strip() for m in args.months.split(",") if m.strip()}
    models = {m.strip() for m in args.models.split(",") if m.strip()}

    wind_bins = [float(x) for x in args.wind_bins.split(",") if x.strip()]
    wind_bins.append(float("inf"))

    pred_paths = sorted(pred_dir.glob("pred_*.zarr"))
    if not pred_paths:
        print("No pred zarr files found.")
        return

    for pred_path in pred_paths:
        model, month = _parse_name(pred_path.stem)
        if months and month and month not in months:
            continue
        if models and model and model not in models:
            continue

        model_name = pred_path.stem
        out_dir = Path(args.out_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating {pred_path} -> {out_dir}")
        metrics_overall, metrics_by_season, metrics_by_windbin, merged = compute_station_metrics(
            args.obs_station,
            args.station_meta,
            pred_path,
            grid_static=args.grid_static,
            wind_bins=wind_bins,
        )
        metrics_overall.to_csv(out_dir / "metrics_overall.csv", index=False)
        metrics_by_season.to_csv(out_dir / "metrics_by_season.csv", index=False)
        metrics_by_windbin.to_csv(out_dir / "metrics_by_windbin.csv", index=False)
        merged.to_csv(out_dir / "station_matches.csv", index=False)

        ds = _open_pred_dataset(pred_path)
        u_name, v_name = _resolve_uv_vars(ds, None, None)

        if not args.skip_radar:
            radar_dir = Path(args.radar_dir)
            for radar_csv in radar_dir.glob("radar_*_30min.csv"):
                meta_path = radar_csv.with_name(radar_csv.stem + "_meta.json")
                evaluate_radar_timeseries(ds, u_name, v_name, radar_csv, meta_path, args.grid_static, out_dir)

        if not args.skip_divergence:
            divergence_stats(ds, u_name, v_name, out_dir, max_times=args.divergence_max_times)

    print("Batch evaluation done.")


if __name__ == "__main__":
    main()
