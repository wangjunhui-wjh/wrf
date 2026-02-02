import argparse
from pathlib import Path
import re

import numpy as np
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


def _sample_indices(t_len, n_cases):
    if t_len <= n_cases:
        return list(range(t_len))
    return list(np.linspace(0, t_len - 1, n_cases, dtype=int))


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


def _plot_imshow(ax, arr, title, vmin=None, vmax=None, cmap="viridis"):
    im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _save_fig(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_fig1(speed_hr, err_maps, models, out_path):
    n = len(models)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    vmin, vmax = np.nanpercentile(speed_hr, [2, 98])
    _plot_imshow(axes[0], speed_hr, "HR Speed", vmin=vmin, vmax=vmax)
    err_all = np.concatenate([np.abs(err_maps[m]).ravel() for m in models])
    err_vmax = np.nanpercentile(err_all, 98)
    for i, m in enumerate(models):
        _plot_imshow(axes[i + 1], np.abs(err_maps[m]), f"{m} |Err|", vmin=0, vmax=err_vmax, cmap="magma")
    _save_fig(fig, out_path)


def _make_fig2(diff_maps, models, out_path):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    all_diff = np.concatenate([np.abs(diff_maps[m]).ravel() for m in models])
    vmax = np.nanpercentile(all_diff, 98)
    for i, m in enumerate(models):
        _plot_imshow(axes[i], diff_maps[m], f"{m} Pred-HR", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    _save_fig(fig, out_path)


def _make_fig3(speed_hr, speed_maps, models, out_path, patch=64):
    # pick patch around max inter-model diff
    diff_mean = None
    for m in models:
        diff = np.abs(speed_maps[m] - speed_hr)
        diff_mean = diff if diff_mean is None else diff_mean + diff
    diff_mean = diff_mean / max(len(models), 1)
    iy, ix = np.unravel_index(np.nanargmax(diff_mean), diff_mean.shape)
    y0 = max(0, iy - patch // 2)
    x0 = max(0, ix - patch // 2)
    y1 = min(speed_hr.shape[0], y0 + patch)
    x1 = min(speed_hr.shape[1], x0 + patch)

    hr_patch = speed_hr[y0:y1, x0:x1]
    row = (y1 - y0) // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin, vmax = np.nanpercentile(hr_patch, [2, 98])
    _plot_imshow(axes[0], hr_patch, "HR Zoom", vmin=vmin, vmax=vmax)

    x = np.arange(x0, x1)
    axes[1].axhline(0.0, color="k", linewidth=0.8)
    for m in models:
        diff_line = (speed_maps[m] - speed_hr)[y0 + row, x0:x1]
        axes[1].plot(x, diff_line, label=m)
    axes[1].set_title("Transect Error (Pred-HR)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("m/s")
    axes[1].legend()

    _save_fig(fig, out_path)


def _make_fig4(metrics_csv, out_path):
    import pandas as pd
    df = pd.read_csv(metrics_csv)
    df = df[df["var"] == "SPEED"]
    order = ["bicubic", "espcn", "unet", "physr"]
    df["model"] = pd.Categorical(df["model"], order)
    df = df.sort_values("model")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].bar(df["model"].astype(str), df["rmse"])
    axes[0].set_title("RMSE")
    axes[1].bar(df["model"].astype(str), df["mae"])
    axes[1].set_title("MAE")
    axes[2].bar(df["model"].astype(str), df["r"])
    axes[2].set_title("R")
    _save_fig(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures.")
    parser.add_argument("--month", required=True, help="Month (e.g., 2025-10)")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr dir")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="WRF HR zarr dir")
    parser.add_argument("--out-dir", default="processed_data/summary/figs", help="Output dir")
    parser.add_argument("--models", default="bicubic,espcn,unet,physr", help="Models to include")
    parser.add_argument("--cases", default="", help="Comma-separated time indices")
    parser.add_argument("--n-cases", type=int, default=3, help="Number of cases")
    parser.add_argument("--select", choices=["diff", "uniform"], default="diff", help="Case selection method")
    args = parser.parse_args()

    month = args.month
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_hr = xr.open_zarr(Path(args.hr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
    u_hr = _to_time_yx(ds_hr["U10"])
    v_hr = _to_time_yx(ds_hr["V10"])

    models = {m.strip() for m in args.models.split(",") if m.strip()}
    pred_map = {}
    for pred_path in sorted(Path(args.pred_dir).glob(f"pred_*_{month}.zarr")):
        model, _ = _parse_pred_name(pred_path)
        if models and model not in models:
            continue
        ds_pred = _open_pred_dataset(pred_path)
        pred_map[model] = (_to_time_yx(ds_pred["U10"]), _to_time_yx(ds_pred["V10"]))
    if not pred_map:
        raise ValueError("No prediction datasets found.")

    if args.cases:
        case_idxs = [int(x) for x in args.cases.split(",") if x.strip()]
    else:
        if args.select == "diff":
            case_idxs = _pick_case_times_by_diff(u_hr, v_hr, pred_map, n_cases=args.n_cases)
        else:
            case_idxs = _sample_indices(u_hr.sizes["time"], args.n_cases)

    for idx in case_idxs:
        uo = u_hr.isel(time=idx).values
        vo = v_hr.isel(time=idx).values
        speed_hr = np.sqrt(uo ** 2 + vo ** 2)
        speed_maps = {}
        err_maps = {}
        diff_maps = {}
        for m, (u_da, v_da) in pred_map.items():
            up = u_da.isel(time=idx).values
            vp = v_da.isel(time=idx).values
            speed = np.sqrt(up ** 2 + vp ** 2)
            speed_maps[m] = speed
            diff = speed - speed_hr
            diff_maps[m] = diff
            err_maps[m] = np.abs(diff)

        fig1 = out_dir / f"fig1_case_{month}_t{idx}.png"
        fig2 = out_dir / f"fig2_case_{month}_t{idx}.png"
        fig3 = out_dir / f"fig3_case_{month}_t{idx}.png"
        _make_fig1(speed_hr, err_maps, list(pred_map.keys()), fig1)
        _make_fig2(diff_maps, list(pred_map.keys()), fig2)
        _make_fig3(speed_hr, speed_maps, list(pred_map.keys()), fig3)

    metrics_csv = Path("processed_data/summary") / f"metrics_grid_overall_{month}.csv"
    if metrics_csv.exists():
        fig4 = out_dir / f"fig4_metrics_{month}.png"
        _make_fig4(metrics_csv, fig4)

    print(f"Saved paper figures to {out_dir}")


if __name__ == "__main__":
    main()
