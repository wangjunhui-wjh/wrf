import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.sr_dataset import list_months


def _combine_stats(stats_list):
    total_count = 0
    total_mean = 0.0
    total_m2 = 0.0
    for count, mean, var in stats_list:
        if count == 0:
            continue
        total_count += count
        total_mean += count * mean
        total_m2 += count * (var + mean ** 2)
    if total_count == 0:
        return 0.0, 1.0
    mean = total_mean / total_count
    var = total_m2 / total_count - mean ** 2
    std = float(np.sqrt(max(var, 1e-12)))
    return float(mean), std


def _collect_stats(ds, var_name):
    da = ds[var_name]
    count = int(da.size)
    mean = float(da.mean().compute().item())
    var = float(da.var().compute().item())
    return count, mean, var


def main():
    parser = argparse.ArgumentParser(description="Compute mean/std stats for SR training.")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="HR zarr directory")
    parser.add_argument("--lr-dir", default="processed_data/lr", help="LR zarr directory")
    parser.add_argument("--input-vars", default="U10,V10,T2,PSFC,PBLH", help="LR input vars")
    parser.add_argument("--target-vars", default="U10,V10", help="HR target vars")
    parser.add_argument("--static-vars", default="", help="Static vars from grid_static (e.g., HGT)")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr")
    parser.add_argument("--exclude-months", default="2025-09", help="Comma-separated months to exclude")
    parser.add_argument("--out", default="processed_data/stats.json", help="Output stats json")
    args = parser.parse_args()

    input_vars = [v.strip() for v in args.input_vars.split(",") if v.strip()]
    target_vars = [v.strip() for v in args.target_vars.split(",") if v.strip()]
    static_vars = [v.strip() for v in args.static_vars.split(",") if v.strip()]
    exclude = {m.strip() for m in args.exclude_months.split(",") if m.strip()}

    months = [m for m in list_months(args.hr_dir, args.lr_dir) if m not in exclude]
    if not months:
        raise ValueError("No months to compute stats.")

    input_stats = {}
    target_stats = {}

    for var in input_vars:
        stats_list = []
        for month in months:
            ds_lr = xr.open_zarr(Path(args.lr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
            stats_list.append(_collect_stats(ds_lr, var))
        mean, std = _combine_stats(stats_list)
        input_stats[var] = {"mean": mean, "std": std}

    for var in target_vars:
        stats_list = []
        for month in months:
            ds_hr = xr.open_zarr(Path(args.hr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
            stats_list.append(_collect_stats(ds_hr, var))
        mean, std = _combine_stats(stats_list)
        target_stats[var] = {"mean": mean, "std": std}

    static_stats = {}
    if static_vars:
        gs_path = Path(args.grid_static)
        if not gs_path.exists():
            raise FileNotFoundError(f"grid_static not found: {gs_path}")
        ds_static = xr.open_zarr(gs_path)
        for var in static_vars:
            if var not in ds_static:
                raise KeyError(f"Static var {var} not found in grid_static.")
            count, mean, var_ = _collect_stats(ds_static, var)
            std = float(np.sqrt(max(var_, 1e-12)))
            static_stats[var] = {"mean": float(mean), "std": std}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"inputs": input_stats, "targets": target_stats, "static": static_stats},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved stats to {out_path}")


if __name__ == "__main__":
    main()
