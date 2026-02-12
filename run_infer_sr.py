import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr

from src.sr_models import build_model


def load_stats(stats_path):
    if not stats_path:
        return {}
    path = Path(stats_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_norm(arr, var_names, stats):
    if not stats:
        return arr
    out = arr.copy()
    for i, name in enumerate(var_names):
        info = stats.get(name)
        if not info:
            continue
        mean = info.get("mean", 0.0)
        std = info.get("std", 1.0)
        if std == 0:
            std = 1.0
        out[:, i, :, :] = (out[:, i, :, :] - mean) / std
    return out


def _apply_denorm(arr, var_names, stats):
    if not stats:
        return arr
    out = arr.copy()
    for i, name in enumerate(var_names):
        info = stats.get(name)
        if not info:
            continue
        mean = info.get("mean", 0.0)
        std = info.get("std", 1.0)
        out[:, i, :, :] = out[:, i, :, :] * std + mean
    return out


def _get_stats(stats, name):
    info = stats.get(name, {})
    mean = info.get("mean", 0.0)
    std = info.get("std", 1.0)
    if std == 0:
        std = 1.0
    return mean, std


def _lr_target_norm(x, input_vars, target_vars, input_stats, target_stats):
    chunks = []
    for var in target_vars:
        if var not in input_vars:
            raise ValueError(f"Target var {var} not in input_vars, required for residual.")
        idx = input_vars.index(var)
        x_var = x[:, idx:idx + 1]
        in_mean, in_std = _get_stats(input_stats, var)
        tar_mean, tar_std = _get_stats(target_stats, var)
        x_phys = x_var * in_std + in_mean
        x_tar = (x_phys - tar_mean) / tar_std
        chunks.append(x_tar)
    return torch.cat(chunks, dim=1)


def _load_static_lr(grid_static, static_vars, scale, static_stats):
    if not static_vars:
        return None
    gs = xr.open_zarr(grid_static)
    static_arrays = []
    for name in static_vars:
        if name not in gs:
            raise KeyError(f"Static var {name} not found in grid_static.")
        da = gs[name]
        y_dim, x_dim = [d for d in da.dims]
        lr_da = da.coarsen({y_dim: scale, x_dim: scale}, boundary="trim").mean()
        arr = lr_da.values.astype(np.float32)
        stats = static_stats.get(name)
        if stats:
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0) or 1.0
            arr = (arr - mean) / std
        static_arrays.append(arr)
    return np.stack(static_arrays, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Run SR inference and export pred_test.zarr.")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="HR zarr directory")
    parser.add_argument("--lr-dir", default="processed_data/lr", help="LR zarr directory")
    parser.add_argument("--months", default="", help="Comma-separated months (e.g., 2025-10,2025-11)")
    parser.add_argument("--model", default="unet", choices=["bicubic", "espcn", "unet", "physr"], help="Model type")
    parser.add_argument("--weights", default="", help="Path to trained model weights (not needed for bicubic)")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    parser.add_argument("--input-vars", default="U10,V10,T2,PSFC,PBLH", help="LR input vars")
    parser.add_argument("--target-vars", default="U10,V10", help="Output vars")
    parser.add_argument("--batch", type=int, default=8, help="Inference batch size")
    parser.add_argument("--stats", default="", help="JSON stats for normalization")
    parser.add_argument("--out-dir", default="processed_data/pred", help="Output directory")
    parser.add_argument("--out", default="", help="Optional output zarr path (or directory)")
    parser.add_argument("--static-vars", default="", help="Static vars from grid_static (e.g., HGT)")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="Grid static zarr")
    parser.add_argument("--residual", action="store_true", help="Predict residual on top of bicubic upsample")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    input_vars = [v.strip() for v in args.input_vars.split(",") if v.strip()]
    target_vars = [v.strip() for v in args.target_vars.split(",") if v.strip()]
    static_vars = [v.strip() for v in args.static_vars.split(",") if v.strip()]
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    if not months:
        raise ValueError("Provide --months for inference.")

    stats = load_stats(args.stats)
    input_stats = stats.get("inputs", {})
    target_stats = stats.get("targets", {})
    static_stats = stats.get("static", {})

    model = None
    if args.model != "bicubic":
        if not args.weights:
            raise ValueError("Provide --weights for learned models.")
        model = build_model(args.model, len(input_vars) + len(static_vars), len(target_vars), scale=args.scale)
        ckpt = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        model = model.to(args.device)
        model.eval()

    out_mode = "dir"
    if args.out:
        out_base = Path(args.out)
        if out_base.suffix == ".zarr":
            out_mode = "file"
        else:
            out_mode = "dir"
    else:
        out_base = Path(args.out_dir)
        if out_base.suffix == ".zarr":
            out_mode = "file"

    if out_mode == "dir":
        out_root = Path(out_base)
        out_root.mkdir(parents=True, exist_ok=True)

    static_lr = _load_static_lr(args.grid_static, static_vars, args.scale, static_stats) if static_vars else None

    for month in months:
        hr_path = Path(args.hr_dir) / f"{month}.zarr"
        lr_path = Path(args.lr_dir) / f"{month}.zarr"
        if not hr_path.exists() or not lr_path.exists():
            print(f"Skip {month}: missing hr/lr zarr")
            continue

        ds_hr = xr.open_zarr(hr_path, consolidated=True, decode_times=False)
        ds_lr = xr.open_zarr(lr_path, consolidated=True, decode_times=False)

        time_vals = ds_hr["time"].values
        target_var = target_vars[0]
        y_dim = [d for d in ds_hr[target_var].dims if d != "time"][0]
        x_dim = [d for d in ds_hr[target_var].dims if d != "time"][1]
        hr_h = ds_hr.sizes[y_dim]
        hr_w = ds_hr.sizes[x_dim]
        t_len = ds_hr.sizes["time"]
        if args.model == "bicubic":
            lr_ref = ds_lr[target_vars[0]]
        else:
            lr_ref = ds_lr[input_vars[0]]
        lr_y_dim = [d for d in lr_ref.dims if d != "time"][0]
        lr_x_dim = [d for d in lr_ref.dims if d != "time"][1]
        lr_h = ds_lr.sizes[lr_y_dim]
        lr_w = ds_lr.sizes[lr_x_dim]
        out_h = lr_h * args.scale
        out_w = lr_w * args.scale
        write_h = min(out_h, hr_h)
        write_w = min(out_w, hr_w)
        if out_h != hr_h or out_w != hr_w:
            print(f"Warning: pred size {out_h}x{out_w} != HR {hr_h}x{hr_w}. Writing top-left {write_h}x{write_w}.")

        if out_mode == "file":
            if len(months) == 1:
                out_path = Path(out_base)
            else:
                out_path = Path(out_base).with_name(f"{Path(out_base).stem}_{month}.zarr")
        else:
            out_path = out_root / f"pred_{args.model}_{month}.zarr"
        store = zarr.DirectoryStore(str(out_path))
        root = zarr.group(store=store, overwrite=True)
        root.create_dataset("time", data=time_vals, dtype=time_vals.dtype)
        root.create_dataset("U10", shape=(t_len, hr_h, hr_w), chunks=(1, 64, 64), dtype="f4", fill_value=np.nan)
        root.create_dataset("V10", shape=(t_len, hr_h, hr_w), chunks=(1, 64, 64), dtype="f4", fill_value=np.nan)
        # xarray requires dimension metadata to open zarr datasets
        root["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
        root["U10"].attrs["_ARRAY_DIMENSIONS"] = ["time", y_dim, x_dim]
        root["V10"].attrs["_ARRAY_DIMENSIONS"] = ["time", y_dim, x_dim]

        for t0 in range(0, t_len, args.batch):
            t1 = min(t0 + args.batch, t_len)
            if args.model == "bicubic":
                lr_block = ds_lr[target_vars].isel(time=slice(t0, t1)).to_array().values
                lr_block = np.transpose(lr_block, (1, 0, 2, 3))
                x = torch.from_numpy(lr_block.astype(np.float32)).to(args.device)
                with torch.no_grad():
                    pred = torch.nn.functional.interpolate(
                        x, scale_factor=args.scale, mode="bicubic", align_corners=False
                    ).cpu().numpy()
            else:
                lr_block = ds_lr[input_vars].isel(time=slice(t0, t1)).to_array().values
                lr_block = np.transpose(lr_block, (1, 0, 2, 3))
                lr_block = _apply_norm(lr_block, input_vars, input_stats)
                lr_dynamic = lr_block
                if static_lr is not None:
                    static_block = np.broadcast_to(static_lr, (lr_block.shape[0],) + static_lr.shape)
                    lr_block = np.concatenate([lr_block, static_block], axis=1)
                x = torch.from_numpy(lr_block.astype(np.float32)).to(args.device)
                with torch.no_grad():
                    pred = model(x).cpu().numpy()
                if args.residual:
                    lr_dyn_t = torch.from_numpy(lr_dynamic.astype(np.float32)).to(args.device)
                    base = _lr_target_norm(lr_dyn_t, input_vars, target_vars, input_stats, target_stats)
                    base = torch.nn.functional.interpolate(
                        base, scale_factor=args.scale, mode="bilinear", align_corners=False
                    ).cpu().numpy()
                    pred = pred + base
                pred = _apply_denorm(pred, target_vars, target_stats)

            root["U10"][t0:t1, 0:write_h, 0:write_w] = pred[:, 0, :write_h, :write_w]
            root["V10"][t0:t1, 0:write_h, 0:write_w] = pred[:, 1, :write_h, :write_w]

            print(f"{month}: {t1}/{t_len}")

        zarr.consolidate_metadata(store)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
