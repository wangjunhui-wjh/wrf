import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr

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


def _fit_linear_calibration(station_matches_csv):
    df = pd.read_csv(station_matches_csv)
    if df.empty:
        return 1.0, 0.0
    obs = df["ws_mean"].to_numpy()
    pred = df["pred_speed"].to_numpy()
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return 1.0, 0.0
    # pred' = a*pred + b
    a, b = np.polyfit(pred[mask], obs[mask], 1)
    return float(a), float(b)


def main():
    parser = argparse.ArgumentParser(description="Export fig3 data and corrected versions.")
    parser.add_argument("--month", required=True)
    parser.add_argument("--pred-dir", default="processed_data/pred")
    parser.add_argument("--hr-dir", default="processed_data/hr")
    parser.add_argument("--eval-dir", default="processed_data/eval")
    parser.add_argument("--out-dir", default="processed_data/summary/fig3_data")
    parser.add_argument("--n-cases", type=int, default=3)
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

    # calibration params per model
    calib = {}
    for model in pred_map:
        station_csv = Path(args.eval_dir) / f"pred_{model}_{month}" / "station_matches.csv"
        if station_csv.exists():
            a, b = _fit_linear_calibration(station_csv)
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

        # save raw patch arrays
        patch_hr = speed_hr[y0:y1, x0:x1]
        np.savez_compressed(out_dir / f"case_{month}_t{idx}_patch.npz",
                            hr=patch_hr, y0=y0, x0=x0, y1=y1, x1=x1)

        # transect data
        x = np.arange(x0, x1)
        for m, sp in speed_maps.items():
            diff = sp - speed_hr
            line = diff[y0 + row, x0:x1]
            a, b = calib[m]
            sp_corr = a * sp + b
            diff_corr = sp_corr - speed_hr
            line_corr = diff_corr[y0 + row, x0:x1]

            df_line = pd.DataFrame({
                "x": x,
                "diff": line,
                "diff_corrected": line_corr,
                "a": a,
                "b": b,
            })
            df_line.to_csv(out_dir / f"case_{month}_t{idx}_{m}_transect.csv", index=False)

            # save patch arrays per model
            patch_pred = sp[y0:y1, x0:x1]
            patch_diff = diff[y0:y1, x0:x1]
            patch_pred_corr = sp_corr[y0:y1, x0:x1]
            patch_diff_corr = diff_corr[y0:y1, x0:x1]
            np.savez_compressed(
                out_dir / f"case_{month}_t{idx}_{m}_patch.npz",
                pred=patch_pred,
                diff=patch_diff,
                pred_corrected=patch_pred_corr,
                diff_corrected=patch_diff_corr,
                a=a,
                b=b,
                y0=y0,
                x0=x0,
                y1=y1,
                x1=x1,
            )

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

    pd.DataFrame(records).to_csv(out_dir / f"cases_{month}_index.csv", index=False)
    print(f"Saved fig3 data to {out_dir}")


if __name__ == "__main__":
    main()
