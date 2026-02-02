import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import _open_pred_dataset


def _parse_pred_name(path):
    match = re.match(r"pred_(.+)_(\d{4}-\d{2})$", path.stem)
    if match:
        return match.group(1), match.group(2)
    return path.stem, ""


def _find_spatial_dims(var):
    dims = list(var.dims)
    if "time" in dims:
        dims.remove("time")
    if len(dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {dims}")
    return dims[0], dims[1]


def _to_time_yx(da):
    y_dim, x_dim = _find_spatial_dims(da)
    return da.transpose("time", y_dim, x_dim)


def _pick_case_times_by_diff(u_hr, v_hr, pred_map, n_cases=3):
    t_len = u_hr.sizes["time"]
    spreads = []
    batch = 8
    for t0 in range(0, t_len, batch):
        t1 = min(t0 + batch, t_len)
        uo = u_hr.isel(time=slice(t0, t1)).values
        vo = v_hr.isel(time=slice(t0, t1)).values
        so = np.sqrt(uo ** 2 + vo ** 2)
        model_errs = []
        for _, (u_da, v_da) in pred_map.items():
            up = u_da.isel(time=slice(t0, t1)).values
            vp = v_da.isel(time=slice(t0, t1)).values
            sp = np.sqrt(up ** 2 + vp ** 2)
            err = np.abs(sp - so).reshape(sp.shape[0], -1).mean(axis=1)
            model_errs.append(err)
        if not model_errs:
            spreads.extend([0.0] * (t1 - t0))
            continue
        model_errs = np.stack(model_errs, axis=1)
        spread = model_errs.max(axis=1) - model_errs.min(axis=1)
        spreads.extend(spread.tolist())
    spreads = np.array(spreads)
    idx_sorted = np.argsort(spreads)
    picks = list(idx_sorted[-n_cases:][::-1])
    return picks


def _plot_transect(hr, lr, model_lines, diff_lines, out_path, title):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax0, ax1 = axes

    ax0.plot(hr[0], hr[1], label="HR", linewidth=2)
    if lr is not None:
        ax0.plot(lr[0], lr[1], label="LR-bilinear", linewidth=1.5, linestyle="--")
    for name, (x, y) in model_lines.items():
        ax0.plot(x, y, label=name, linewidth=1.2)
    ax0.set_ylabel("Speed (m/s)")
    ax0.legend(ncol=2, fontsize=8)
    ax0.set_title(title)

    ax1.axhline(0.0, color="k", linewidth=0.8)
    for name, (x, y) in diff_lines.items():
        ax1.plot(x, y, label=f"{name}-HR", linewidth=1.2)
    ax1.set_xlabel("x")
    ax1.set_ylabel("Diff (m/s)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _calc_stats(diff):
    return {
        "mae": float(np.nanmean(np.abs(diff))),
        "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
        "max_abs": float(np.nanmax(np.abs(diff))),
        "mean": float(np.nanmean(diff)),
    }


def _load_calibration(calib_dir, month):
    calib_index = Path(calib_dir) / f"cases_{month}_index.csv"
    if not calib_index.exists():
        return {}
    df = pd.read_csv(calib_index)
    out = {}
    for _, row in df.drop_duplicates(["model"]).iterrows():
        out[row["model"]] = (row["a"], row["b"])
    return out


def _eval_station_calibration(eval_dir, month, station_ids, models):
    rows = []
    for m in models:
        csv = Path(eval_dir) / f"pred_{m}_{month}" / "station_matches.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df = df[df["station_id"].isin(station_ids)]
        if df.empty:
            continue
        obs = df["ws_mean"].to_numpy()
        pred = df["pred_speed"].to_numpy()
        mask = np.isfinite(obs) & np.isfinite(pred)
        if mask.sum() < 2:
            continue
        a, b = np.polyfit(pred[mask], obs[mask], 1)
        pred_corr = a * pred + b
        diff = pred - obs
        diff_corr = pred_corr - obs
        rows.append({
            "model": m,
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(diff[mask]))),
            "rmse": float(np.sqrt(np.mean(diff[mask] ** 2))),
            "bias": float(np.mean(diff[mask])),
            "mae_corr": float(np.mean(np.abs(diff_corr[mask]))),
            "rmse_corr": float(np.sqrt(np.mean(diff_corr[mask] ** 2))),
            "bias_corr": float(np.mean(diff_corr[mask])),
            "a": float(a),
            "b": float(b),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate transect figures with HR/LR/models and diff lines.")
    parser.add_argument("--month", required=True)
    parser.add_argument("--pred-dir", default="processed_data/pred")
    parser.add_argument("--hr-dir", default="processed_data/hr")
    parser.add_argument("--lr-dir", default="processed_data/lr")
    parser.add_argument("--eval-dir", default="processed_data/eval")
    parser.add_argument("--out-dir", default="processed_data/summary/fig3_profiles")
    parser.add_argument("--calib-dir", default="processed_data/summary/fig3_calibrated")
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--models", default="bicubic,espcn,unet,physr")
    parser.add_argument("--use-calibrated", action="store_true", help="use calibrated outputs in plots")
    args = parser.parse_args()

    month = args.month
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_hr = xr.open_zarr(Path(args.hr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
    ds_lr = xr.open_zarr(Path(args.lr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
    u_hr = _to_time_yx(ds_hr["U10"])
    v_hr = _to_time_yx(ds_hr["V10"])
    u_lr = _to_time_yx(ds_lr["U10"])
    v_lr = _to_time_yx(ds_lr["V10"])

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    pred_map = {}
    for pred_path in sorted(Path(args.pred_dir).glob(f"pred_*_{month}.zarr")):
        model, _ = _parse_pred_name(pred_path)
        if model not in models:
            continue
        ds_pred = _open_pred_dataset(pred_path)
        pred_map[model] = (_to_time_yx(ds_pred["U10"]), _to_time_yx(ds_pred["V10"]))

    if not pred_map:
        raise ValueError("No predictions found.")

    case_idxs = _pick_case_times_by_diff(u_hr, v_hr, pred_map, n_cases=args.n_cases)

    calib = _load_calibration(args.calib_dir, month)

    summary_rows = []

    for idx in case_idxs:
        uo = u_hr.isel(time=idx).values
        vo = v_hr.isel(time=idx).values
        speed_hr = np.sqrt(uo ** 2 + vo ** 2)

        # pick patch around max mean diff
        diff_mean = None
        speed_maps = {}
        for m, (u_da, v_da) in pred_map.items():
            up = u_da.isel(time=idx).values
            vp = v_da.isel(time=idx).values
            sp = np.sqrt(up ** 2 + vp ** 2)
            speed_maps[m] = sp
            diff = np.abs(sp - speed_hr)
            diff_mean = diff if diff_mean is None else diff_mean + diff
        diff_mean = diff_mean / max(len(speed_maps), 1)
        iy, ix = np.unravel_index(np.nanargmax(diff_mean), diff_mean.shape)
        patch = 64
        y0 = max(0, iy - patch // 2)
        x0 = max(0, ix - patch // 2)
        y1 = min(speed_hr.shape[0], y0 + patch)
        x1 = min(speed_hr.shape[1], x0 + patch)
        row = (y1 - y0) // 2

        # build transects
        x = np.arange(x0, x1)
        hr_line = speed_hr[y0 + row, x0:x1]
        # LR bilinear upsample to HR grid
        lr_u = u_lr.isel(time=idx).values
        lr_v = v_lr.isel(time=idx).values
        lr_speed = np.sqrt(lr_u ** 2 + lr_v ** 2)
        # bilinear upsample to HR size (pad if needed)
        import torch
        lr_t = torch.from_numpy(lr_speed[None, None, :, :].astype(np.float32))
        lr_up = torch.nn.functional.interpolate(lr_t, scale_factor=4, mode="bilinear", align_corners=False)[0, 0].numpy()
        hr_h, hr_w = speed_hr.shape
        if lr_up.shape[0] < hr_h or lr_up.shape[1] < hr_w:
            pad_h = hr_h - lr_up.shape[0]
            pad_w = hr_w - lr_up.shape[1]
            lr_up = np.pad(lr_up, ((0, max(pad_h, 0)), (0, max(pad_w, 0))), mode="edge")
        lr_up = lr_up[:hr_h, :hr_w]
        lr_line = lr_up[y0 + row, x0:x1]

        model_lines = {}
        diff_lines = {}
        for m, sp in speed_maps.items():
            a, b = calib.get(m, (1.0, 0.0))
            sp_use = a * sp + b if args.use_calibrated else sp
            line = sp_use[y0 + row, x0:x1]
            model_lines[m] = (x, line)
            diff = line - hr_line
            diff_lines[m] = (x, diff)
            stats = _calc_stats(diff)
            stats.update({"month": month, "t_idx": idx, "model": m, "calibrated": args.use_calibrated})
            summary_rows.append(stats)

        fig_path = out_dir / f"transect_{month}_t{idx}_{'cal' if args.use_calibrated else 'raw'}.png"
        _plot_transect((x, hr_line), (x, lr_line), model_lines, diff_lines,
                       fig_path, f"Transect t={idx} ({'calibrated' if args.use_calibrated else 'raw'})")

    # save summary stats
    pd.DataFrame(summary_rows).to_csv(out_dir / f"transect_stats_{month}_{'cal' if args.use_calibrated else 'raw'}.csv", index=False)

    # calibration success check on station subset
    cal_stations = Path(args.calib_dir) / f"calibration_stations_{month}.csv"
    if cal_stations.exists():
        station_ids = pd.read_csv(cal_stations)["station_id"].tolist()
        rows = _eval_station_calibration(args.eval_dir, month, station_ids, models)
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"calibration_station_check_{month}.csv", index=False)

    print(f"Saved transect figures to {out_dir}")


if __name__ == "__main__":
    main()
