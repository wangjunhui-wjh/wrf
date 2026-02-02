import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import (
    _open_pred_dataset,
    _resolve_uv_vars,
    compute_station_metrics,
    evaluate_radar_timeseries,
    divergence_stats,
)


def main():
    parser = argparse.ArgumentParser(description="One-click evaluation for station/radar observations.")
    parser.add_argument("--pred", required=True, help="Prediction dataset path (.nc/.zarr)")
    parser.add_argument("--obs-station", default="processed_data/obs/station_30min.csv", help="Station 30min CSV")
    parser.add_argument("--station-meta", default="processed_data/obs/station_meta.csv", help="Station meta CSV")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr path")
    parser.add_argument("--radar-dir", default="processed_data/obs", help="Radar products directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for evaluation results")
    parser.add_argument("--u-var", default=None, help="U variable name in pred dataset")
    parser.add_argument("--v-var", default=None, help="V variable name in pred dataset")
    parser.add_argument("--wind-bins", default="0,2,4,6,8,10,12,15,20", help="Wind bins (comma-separated)")
    parser.add_argument("--skip-radar", action="store_true", help="Skip radar plots")
    parser.add_argument("--skip-divergence", action="store_true", help="Skip divergence stats")
    parser.add_argument("--divergence-max-times", type=int, default=200, help="Max time steps for divergence stats")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    model_name = pred_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else Path("processed_data/eval") / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    wind_bins = [float(x) for x in args.wind_bins.split(",") if x.strip()]
    wind_bins.append(float("inf"))

    print("Computing station metrics...")
    metrics_overall, metrics_by_season, metrics_by_windbin, merged = compute_station_metrics(
        args.obs_station,
        args.station_meta,
        args.pred,
        grid_static=args.grid_static,
        u_var=args.u_var,
        v_var=args.v_var,
        wind_bins=wind_bins,
    )

    metrics_overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    metrics_by_season.to_csv(out_dir / "metrics_by_season.csv", index=False)
    metrics_by_windbin.to_csv(out_dir / "metrics_by_windbin.csv", index=False)

    # optional: save matched station time series
    merged.to_csv(out_dir / "station_matches.csv", index=False)

    ds = _open_pred_dataset(args.pred)
    u_name, v_name = _resolve_uv_vars(ds, args.u_var, args.v_var)

    if not args.skip_radar:
        print("Generating radar plots...")
        radar_dir = Path(args.radar_dir)
        for radar_csv in radar_dir.glob("radar_*_30min.csv"):
            meta_path = radar_csv.with_name(radar_csv.stem + "_meta.json")
            evaluate_radar_timeseries(ds, u_name, v_name, radar_csv, meta_path, args.grid_static, out_dir)

    if not args.skip_divergence:
        print("Computing divergence stats...")
        divergence_stats(ds, u_name, v_name, out_dir, max_times=args.divergence_max_times)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
