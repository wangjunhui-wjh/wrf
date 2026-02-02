import argparse
from pathlib import Path
import re

import numpy as np
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


def _align_time(ds, month, manifest_path):
    if manifest_path is None:
        return ds
    manifest = Path(manifest_path)
    if not manifest.exists():
        return ds
    import pandas as pd
    df = pd.read_csv(manifest)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    month_mask = df["time"].dt.strftime("%Y-%m") == month
    times = df.loc[month_mask, "time"].sort_values().reset_index(drop=True)
    if len(times) != ds.sizes.get("time", 0):
        return ds
    return ds.assign_coords(time=("time", times.to_numpy()))


class MetricsAccumulator:
    def __init__(self):
        self.count = 0
        self.sum_diff = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_obs = 0.0
        self.sum_pred = 0.0
        self.sum_obs2 = 0.0
        self.sum_pred2 = 0.0
        self.sum_obs_pred = 0.0

    def update(self, obs, pred):
        mask = np.isfinite(obs) & np.isfinite(pred)
        if not np.any(mask):
            return
        o = obs[mask].astype(np.float64)
        p = pred[mask].astype(np.float64)
        diff = p - o
        self.count += o.size
        self.sum_diff += diff.sum()
        self.sum_abs += np.abs(diff).sum()
        self.sum_sq += (diff ** 2).sum()
        self.sum_obs += o.sum()
        self.sum_pred += p.sum()
        self.sum_obs2 += (o ** 2).sum()
        self.sum_pred2 += (p ** 2).sum()
        self.sum_obs_pred += (o * p).sum()

    def finalize(self):
        if self.count == 0:
            return {"n": 0, "bias": np.nan, "rmse": np.nan, "mae": np.nan, "r": np.nan}
        n = self.count
        bias = self.sum_diff / n
        rmse = np.sqrt(self.sum_sq / n)
        mae = self.sum_abs / n
        denom = (self.sum_obs2 - (self.sum_obs ** 2) / n) * (self.sum_pred2 - (self.sum_pred ** 2) / n)
        if denom <= 0:
            r = np.nan
        else:
            r = (self.sum_obs_pred - (self.sum_obs * self.sum_pred) / n) / np.sqrt(denom)
        return {"n": int(n), "bias": float(bias), "rmse": float(rmse), "mae": float(mae), "r": float(r)}


def _to_time_yx(da):
    y_dim, x_dim = _find_spatial_dims(da)
    return da.transpose("time", y_dim, x_dim)


def compute_grid_metrics(ds_hr, ds_pred, batch=8):
    u_hr = _to_time_yx(ds_hr["U10"])
    v_hr = _to_time_yx(ds_hr["V10"])
    u_pr = _to_time_yx(ds_pred["U10"])
    v_pr = _to_time_yx(ds_pred["V10"])

    t_len = u_hr.sizes["time"]
    acc_u = MetricsAccumulator()
    acc_v = MetricsAccumulator()
    acc_s = MetricsAccumulator()

    for t0 in range(0, t_len, batch):
        t1 = min(t0 + batch, t_len)
        uo = u_hr.isel(time=slice(t0, t1)).values
        vo = v_hr.isel(time=slice(t0, t1)).values
        up = u_pr.isel(time=slice(t0, t1)).values
        vp = v_pr.isel(time=slice(t0, t1)).values

        acc_u.update(uo, up)
        acc_v.update(vo, vp)

        so = np.sqrt(uo ** 2 + vo ** 2)
        sp = np.sqrt(up ** 2 + vp ** 2)
        acc_s.update(so, sp)

    return acc_u.finalize(), acc_v.finalize(), acc_s.finalize()


def main():
    parser = argparse.ArgumentParser(description="Grid metrics: predictions vs WRF-HR truth.")
    parser.add_argument("--month", required=True, help="Month to evaluate (e.g., 2025-10)")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr directory")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="WRF HR zarr directory")
    parser.add_argument("--out", default="processed_data/summary/metrics_grid_overall_2025-10.csv",
                        help="Output CSV path")
    parser.add_argument("--manifest", default="raw_manifest/manifest_2026.csv",
                        help="Manifest CSV for time override")
    parser.add_argument("--batch", type=int, default=8, help="Time batch size")
    args = parser.parse_args()

    month = args.month
    hr_path = Path(args.hr_dir) / f"{month}.zarr"
    if not hr_path.exists():
        raise FileNotFoundError(f"Missing HR zarr: {hr_path}")

    ds_hr = xr.open_zarr(hr_path, consolidated=True, decode_times=False)
    ds_hr = _align_time(ds_hr, month, args.manifest)

    pred_datasets = []
    for pred_path in sorted(Path(args.pred_dir).glob(f"pred_*_{month}.zarr")):
        model, _ = _parse_pred_name(pred_path)
        ds_pred = _open_pred_dataset(pred_path)
        pred_datasets.append((model, _to_time_yx(ds_pred["U10"]), _to_time_yx(ds_pred["V10"])))

    # prepare accumulators per model
    acc = {}
    for model, _, _ in pred_datasets:
        acc[model] = {
            "U10": MetricsAccumulator(),
            "V10": MetricsAccumulator(),
            "SPEED": MetricsAccumulator(),
        }

    u_hr = _to_time_yx(ds_hr["U10"])
    v_hr = _to_time_yx(ds_hr["V10"])
    t_len = u_hr.sizes["time"]

    for t0 in range(0, t_len, args.batch):
        t1 = min(t0 + args.batch, t_len)
        uo = u_hr.isel(time=slice(t0, t1)).values
        vo = v_hr.isel(time=slice(t0, t1)).values
        so = np.sqrt(uo ** 2 + vo ** 2)

        for model, u_pr_da, v_pr_da in pred_datasets:
            u_pr = u_pr_da.isel(time=slice(t0, t1)).values
            v_pr = v_pr_da.isel(time=slice(t0, t1)).values
            acc[model]["U10"].update(uo, u_pr)
            acc[model]["V10"].update(vo, v_pr)
            sp = np.sqrt(u_pr ** 2 + v_pr ** 2)
            acc[model]["SPEED"].update(so, sp)

    rows = []
    for model in acc:
        for var in ["U10", "V10", "SPEED"]:
            row = {"model": model, "month": month, "var": var}
            row.update(acc[model][var].finalize())
            rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
