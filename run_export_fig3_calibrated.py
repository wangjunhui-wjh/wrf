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


def _select_station_ids(station_matches_csv, k=12, seed=42):
    df = pd.read_csv(station_matches_csv)
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    stations = sorted(df["station_id"].unique())
    if len(stations) <= k:
        return stations
    return rng.choice(stations, size=k, replace=False).tolist()


def _fit_linear_calibration(station_matches_csv, station_ids):
    df = pd.read_csv(station_matches_csv)
    if df.empty:
        return 1.0, 0.0
    df = df[df["station_id"].isin(station_ids)]
    obs = df["ws_mean"].to_numpy()
    pred = df["pred_speed"].to_numpy()
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return 1.0, 0.0
    a, b = np.polyfit(pred[mask], obs[mask], 1)
    return float(a), float(b)


def _plot_diff_maps(diff_maps, out_path, title):
    models = list(diff_maps.keys())
    all_diff = np.concatenate([np.abs(d).ravel() for d in diff_maps.values()])
    vmax = np.nanpercentile(all_diff, 98)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, m in enumerate(models):
        ax = axes[i]
        im = ax.imshow(diff_maps[m], origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_title(m)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_transects(lines, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, (x, y) in lines.items():
        ax.plot(x, y, label=model)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Pred - HR (m/s)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Station-calibrated fig3 data and plots.")
    parser.add_argument("--month", required=True)
    parser.add_argument("--pred-dir", default="processed_data/pred")
    parser.add_argument("--hr-dir", default="processed_data/hr")
    parser.add_argument("--eval-dir", default="processed_data/eval")
    parser.add_argument("--out-dir", default="processed_data/summary/fig3_calibrated")
    parser.add_argument("--n-cases", type=int, default=3)
    parser.add_argument("--n-stations", type=int, default=12)
    args = parser.parse_args()

    month = args.month
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_hr = xr.open_zarr(Path(args.hr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
    u_hr = _to_time_yx(ds_hr["U10"])
    v_hr = _to_time_yx(ds_hr["V10"])

    pred_map = {}
    for pred_path in sorted(Path(args.pred_dir).glob(f"pred_*_{month}.zarr")):
        model, _ = _parse_pred_name(pred_path)
        ds_pred = _open_pred_dataset(pred_path)
        pred_map[model] = (_to_time_yx(ds_pred["U10"]), _to_time_yx(ds_pred["V10"]))

    if not pred_map:
        raise ValueError("No predictions found.")

    case_idxs = _pick_case_times_by_diff(u_hr, v_hr, pred_map, n_cases=args.n_cases)

    # pick station IDs from bicubic matches (stable set)
    station_csv = Path(args.eval_dir) / f"pred_bicubic_{month}" / "station_matches.csv"
    station_ids = _select_station_ids(station_csv, k=args.n_stations)
    if not station_ids:
        raise ValueError("No station IDs found for calibration.")
    pd.DataFrame({"station_id": station_ids}).to_csv(out_dir / f"calibration_stations_{month}.csv", index=False)

    # calibration params per model (fit on selected stations)
    calib = {}
    for model in pred_map:
        m_csv = Path(args.eval_dir) / f"pred_{model}_{month}" / "station_matches.csv"
        if m_csv.exists():
            a, b = _fit_linear_calibration(m_csv, station_ids)
        else:
            a, b = 1.0, 0.0
        calib[model] = (a, b)

    records = []

    for idx in case_idxs:
        uo = u_hr.isel(time=idx).values
        vo = v_hr.isel(time=idx).values
        speed_hr = np.sqrt(uo ** 2 + vo ** 2)

        # choose patch around max mean diff
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

        # build diff maps (raw & calibrated)
        diff_raw = {}
        diff_cal = {}
        tran_raw = {}
        tran_cal = {}

        for m, sp in speed_maps.items():
            diff = sp - speed_hr
            a, b = calib[m]
            sp_corr = a * sp + b
            diff_corr = sp_corr - speed_hr

            diff_raw[m] = diff[y0:y1, x0:x1]
            diff_cal[m] = diff_corr[y0:y1, x0:x1]

            x = np.arange(x0, x1)
            line = diff[y0 + row, x0:x1]
            line_corr = diff_corr[y0 + row, x0:x1]
            tran_raw[m] = (x, line)
            tran_cal[m] = (x, line_corr)

            # save patches
            np.savez_compressed(
                out_dir / f"case_{month}_t{idx}_{m}_patch.npz",
                pred=sp[y0:y1, x0:x1],
                diff=diff[y0:y1, x0:x1],
                pred_corrected=sp_corr[y0:y1, x0:x1],
                diff_corrected=diff_corr[y0:y1, x0:x1],
                a=a,
                b=b,
                y0=y0,
                x0=x0,
                y1=y1,
                x1=x1,
            )

            # save transect csv
            df_line = pd.DataFrame({
                "x": x,
                "diff": line,
                "diff_corrected": line_corr,
                "a": a,
                "b": b,
            })
            df_line.to_csv(out_dir / f"case_{month}_t{idx}_{m}_transect.csv", index=False)

            records.append({
                "month": month,
                "t_idx": idx,
                "model": m,
                "a": a,
                "b": b,
                "patch_y0": y0,
                "patch_x0": x0,
                "patch_y1": y1,
                "patch_x1": x1,
            })

        # save fig (raw & calibrated diff maps + transects)
        _plot_diff_maps(diff_raw, out_dir / f"case_{month}_t{idx}_diff_raw.png", f"Pred-HR (raw) t={idx}")
        _plot_diff_maps(diff_cal, out_dir / f"case_{month}_t{idx}_diff_calibrated.png", f"Pred-HR (calibrated) t={idx}")
        _plot_transects(tran_raw, out_dir / f"case_{month}_t{idx}_transect_raw.png", f"Transect raw t={idx}")
        _plot_transects(tran_cal, out_dir / f"case_{month}_t{idx}_transect_calibrated.png", f"Transect calibrated t={idx}")

    pd.DataFrame(records).to_csv(out_dir / f"cases_{month}_index.csv", index=False)
    print(f"Saved calibrated fig3 data to {out_dir}")


if __name__ == "__main__":
    main()
