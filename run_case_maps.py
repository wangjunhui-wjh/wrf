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


def _pick_case_times(ds_hr, n_cases=3):
    u = _to_time_yx(ds_hr["U10"])
    v = _to_time_yx(ds_hr["V10"])
    t_len = u.sizes["time"]
    # compute mean speed per time in batches
    means = []
    batch = 8
    for t0 in range(0, t_len, batch):
        t1 = min(t0 + batch, t_len)
        uo = u.isel(time=slice(t0, t1)).values
        vo = v.isel(time=slice(t0, t1)).values
        speed = np.sqrt(uo ** 2 + vo ** 2)
        means.extend(speed.reshape(speed.shape[0], -1).mean(axis=1).tolist())
    means = np.array(means)
    idx_sorted = np.argsort(means)
    # low, mid, high
    picks = [idx_sorted[0], idx_sorted[len(idx_sorted) // 2], idx_sorted[-1]]
    return picks[:n_cases]


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
    # pick top n with largest inter-model spread
    picks = list(idx_sorted[-n_cases:][::-1])
    return picks


def _plot_map(arr, title, out_path, vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(5, 4))
    plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate case maps for HR vs models.")
    parser.add_argument("--month", required=True, help="Month (e.g., 2025-10)")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr dir")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="WRF HR zarr dir")
    parser.add_argument("--out-dir", default="processed_data/summary/cases", help="Output dir")
    parser.add_argument("--cases", default="", help="Comma-separated time indices to plot")
    parser.add_argument("--n-cases", type=int, default=3, help="Number of cases to plot")
    parser.add_argument("--select", choices=["wind", "diff"], default="wind",
                        help="Case selection: wind (low/mid/high) or diff (max inter-model diff)")
    parser.add_argument("--models", default="bicubic,espcn,unet,physr", help="Models to include")
    parser.add_argument("--diff", action="store_true", help="Also output signed difference maps")
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

    if args.cases:
        case_idxs = [int(x) for x in args.cases.split(",") if x.strip()]
    else:
        if args.select == "diff":
            case_idxs = _pick_case_times_by_diff(u_hr, v_hr, pred_map, n_cases=args.n_cases)
        else:
            case_idxs = _pick_case_times(ds_hr, n_cases=args.n_cases)

    for idx in case_idxs:
        uo = u_hr.isel(time=idx).values
        vo = v_hr.isel(time=idx).values
        speed_hr = np.sqrt(uo ** 2 + vo ** 2)
        vmin, vmax = np.nanpercentile(speed_hr, [2, 98])

        _plot_map(speed_hr, f"HR Speed t={idx}", out_dir / f"case_{month}_t{idx}_hr_speed.png",
                  vmin=vmin, vmax=vmax)

        diffs = {}
        for model, (u_da, v_da) in pred_map.items():
            up = u_da.isel(time=idx).values
            vp = v_da.isel(time=idx).values
            speed = np.sqrt(up ** 2 + vp ** 2)
            _plot_map(speed, f"{model} Speed t={idx}",
                      out_dir / f"case_{month}_t{idx}_{model}_speed.png",
                      vmin=vmin, vmax=vmax)
            diff = speed - speed_hr
            diffs[model] = diff
            err = np.abs(diff)
            _plot_map(err, f"{model} |Err| t={idx}",
                      out_dir / f"case_{month}_t{idx}_{model}_error.png",
                      cmap="magma")
        if args.diff and diffs:
            all_diff = np.concatenate([np.abs(d).ravel() for d in diffs.values()])
            vmax_diff = np.nanpercentile(all_diff, 95)
            for model, diff in diffs.items():
                _plot_map(diff, f"{model} Diff t={idx}",
                          out_dir / f"case_{month}_t{idx}_{model}_diff.png",
                          vmin=-vmax_diff, vmax=vmax_diff, cmap="RdBu_r")

    print(f"Saved case maps to {out_dir}")


if __name__ == "__main__":
    main()
