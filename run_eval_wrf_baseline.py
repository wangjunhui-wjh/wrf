import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from src.evaluation import (
    _open_pred_dataset,
    _resolve_uv_vars,
    compute_station_metrics,
    evaluate_radar_timeseries,
    divergence_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate WRF HR/LR baseline against station observations.")
    parser.add_argument("--month", required=True, help="Month to evaluate (e.g., 2025-10)")
    parser.add_argument("--use", choices=["hr", "lr"], default="hr", help="Use WRF HR or LR")
    parser.add_argument("--obs-station", default="processed_data/obs/station_30min.csv", help="Station 30min CSV")
    parser.add_argument("--station-meta", default="processed_data/obs/station_meta.csv", help="Station meta CSV")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr path")
    parser.add_argument("--radar-dir", default="processed_data/obs", help="Radar products directory")
    parser.add_argument("--out-dir", default="processed_data/eval", help="Base output directory")
    parser.add_argument("--wind-bins", default="0,2,4,6,8,10,12,15,20", help="Wind bins (comma-separated)")
    parser.add_argument("--manifest", default="raw_manifest/manifest_2026.csv", help="Manifest CSV for time override")
    parser.add_argument("--round", default="30min", help="Floor time to this freq, empty to disable")
    parser.add_argument("--skip-radar", action="store_true", help="Skip radar plots")
    parser.add_argument("--skip-divergence", action="store_true", help="Skip divergence stats")
    parser.add_argument("--divergence-max-times", type=int, default=200, help="Max time steps for divergence stats")
    args = parser.parse_args()

    month = args.month
    src_dir = Path("processed_data") / args.use
    pred_path = src_dir / f"{month}.zarr"
    if args.use == "lr":
        bicubic_path = Path("processed_data/pred") / f"pred_bicubic_{month}.zarr"
        if bicubic_path.exists():
            pred_path = bicubic_path
            print(f"Using bicubic-upsampled LR as baseline: {pred_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {args.use} zarr for {month}: {pred_path}")

    out_dir = Path(args.out_dir) / f"wrf_{args.use}_{month}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wind_bins = [float(x) for x in args.wind_bins.split(",") if x.strip()]
    wind_bins.append(float("inf"))

    print(f"Evaluating WRF-{args.use.upper()} {pred_path} -> {out_dir}")
    # Override time with manifest timestamps (WRF zarr time units are not reliable)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    manifest["time"] = pd.to_datetime(manifest["time"], errors="coerce")
    month_mask = manifest["time"].dt.strftime("%Y-%m") == month
    month_times = manifest.loc[month_mask, "time"].sort_values().reset_index(drop=True)
    if args.round:
        month_times = month_times.dt.floor(args.round)

    ds = xr.open_zarr(pred_path, consolidated=True, decode_times=False)
    if ds.sizes.get("time") != len(month_times):
        raise ValueError(
            f"Time length mismatch: {ds.sizes.get('time')} (zarr) vs {len(month_times)} (manifest)"
        )
    ds = ds.assign_coords(time=("time", month_times.to_numpy()))
    metrics_overall, metrics_by_season, metrics_by_windbin, merged = compute_station_metrics(
        args.obs_station,
        args.station_meta,
        ds,
        grid_static=args.grid_static,
        wind_bins=wind_bins,
    )

    metrics_overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    metrics_by_season.to_csv(out_dir / "metrics_by_season.csv", index=False)
    metrics_by_windbin.to_csv(out_dir / "metrics_by_windbin.csv", index=False)
    merged.to_csv(out_dir / "station_matches.csv", index=False)

    u_name, v_name = _resolve_uv_vars(ds, None, None)

    if not args.skip_radar:
        radar_dir = Path(args.radar_dir)
        for radar_csv in radar_dir.glob("radar_*_30min.csv"):
            meta_path = radar_csv.with_name(radar_csv.stem + "_meta.json")
            evaluate_radar_timeseries(ds, u_name, v_name, radar_csv, meta_path, args.grid_static, out_dir)

    if not args.skip_divergence:
        divergence_stats(ds, u_name, v_name, out_dir, max_times=args.divergence_max_times)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
