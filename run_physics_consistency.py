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


def _sample_indices(t_len, n_samples):
    if t_len <= n_samples:
        return list(range(t_len))
    return list(np.linspace(0, t_len - 1, n_samples, dtype=int))


def _compute_divergence(u, v):
    # u,v: 2D arrays [y,x]
    du_dx = (u[:, 2:] - u[:, :-2]) / 2.0
    dv_dy = (v[2:, :] - v[:-2, :]) / 2.0
    core = du_dx[1:-1, :] + dv_dy[:, 1:-1]
    return core


def divergence_pdf(ds_list, labels, n_samples=20, bins=200, out_path=None):
    # Use HR to define range
    u_hr, v_hr = ds_list[0]
    t_len = u_hr.sizes["time"]
    idxs = _sample_indices(t_len, n_samples)
    samples = []
    for t in idxs:
        div = _compute_divergence(u_hr.isel(time=t).values, v_hr.isel(time=t).values)
        samples.append(div.ravel())
    all_vals = np.concatenate(samples)
    lo, hi = np.percentile(all_vals, [1, 99])
    bins_edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=(6, 4))
    for (u_da, v_da), label in zip(ds_list, labels):
        vals = []
        for t in idxs:
            div = _compute_divergence(u_da.isel(time=t).values, v_da.isel(time=t).values)
            vals.append(div.ravel())
        vals = np.concatenate(vals)
        hist, edges = np.histogram(vals, bins=bins_edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.plot(centers, hist, label=label)

    plt.title("Divergence PDF")
    plt.xlabel("divergence (arbitrary units)")
    plt.ylabel("PDF")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()


def _radial_psd(field):
    # field: 2D
    f = field - np.nanmean(field)
    f = np.nan_to_num(f, nan=0.0)
    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    ny, nx = field.shape
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    kx2, ky2 = np.meshgrid(kx, ky)
    k = np.sqrt(kx2 ** 2 + ky2 ** 2)
    k_flat = k.ravel()
    P_flat = P.ravel()
    k_bins = np.linspace(0, k_flat.max(), min(ny, nx) // 2)
    inds = np.digitize(k_flat, k_bins)
    psd = np.zeros(len(k_bins))
    counts = np.zeros(len(k_bins))
    for i, p in zip(inds, P_flat):
        if 0 <= i - 1 < len(k_bins):
            psd[i - 1] += p
            counts[i - 1] += 1
    psd = np.where(counts > 0, psd / counts, np.nan)
    return k_bins, psd


def psd_plot(ds_list, labels, n_samples=10, out_path=None):
    plt.figure(figsize=(6, 4))
    for (u_da, v_da), label in zip(ds_list, labels):
        t_len = u_da.sizes["time"]
        idxs = _sample_indices(t_len, n_samples)
        k_acc = None
        psd_acc = None
        for t in idxs:
            speed = np.sqrt(u_da.isel(time=t).values ** 2 + v_da.isel(time=t).values ** 2)
            k, psd = _radial_psd(speed)
            if k_acc is None:
                k_acc = k
                psd_acc = np.nan_to_num(psd, nan=0.0)
            else:
                psd_acc += np.nan_to_num(psd, nan=0.0)
        psd_mean = psd_acc / len(idxs)
        plt.plot(k_acc, psd_mean, label=label)

    plt.yscale("log")
    plt.xscale("log")
    plt.title("Radial PSD of Wind Speed")
    plt.xlabel("Wavenumber (grid units)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.close()


def scale_consistency(pred_ds, lr_ds, scale=4, batch=8):
    u_lr = _to_time_yx(lr_ds["U10"])
    v_lr = _to_time_yx(lr_ds["V10"])
    u_pr = _to_time_yx(pred_ds["U10"])
    v_pr = _to_time_yx(pred_ds["V10"])

    t_len = u_lr.sizes["time"]
    h_lr, w_lr = u_lr.sizes[u_lr.dims[1]], u_lr.sizes[u_lr.dims[2]]
    h_hr = h_lr * scale
    w_hr = w_lr * scale

    def downsample(arr):
        arr = arr[:, :h_hr, :w_hr]
        arr = arr.reshape(arr.shape[0], h_lr, scale, w_lr, scale)
        return np.nanmean(arr, axis=(2, 4))

    from run_grid_metrics import MetricsAccumulator  # reuse
    acc_u = MetricsAccumulator()
    acc_v = MetricsAccumulator()
    acc_s = MetricsAccumulator()

    for t0 in range(0, t_len, batch):
        t1 = min(t0 + batch, t_len)
        uo = u_lr.isel(time=slice(t0, t1)).values
        vo = v_lr.isel(time=slice(t0, t1)).values
        up = u_pr.isel(time=slice(t0, t1)).values
        vp = v_pr.isel(time=slice(t0, t1)).values
        up_ds = downsample(up)
        vp_ds = downsample(vp)

        acc_u.update(uo, up_ds)
        acc_v.update(vo, vp_ds)
        so = np.sqrt(uo ** 2 + vo ** 2)
        sp = np.sqrt(up_ds ** 2 + vp_ds ** 2)
        acc_s.update(so, sp)

    return acc_u.finalize(), acc_v.finalize(), acc_s.finalize()


def main():
    parser = argparse.ArgumentParser(description="Physics consistency evaluation (divergence/PSD/scale).")
    parser.add_argument("--month", required=True, help="Month (e.g., 2025-10)")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr dir")
    parser.add_argument("--hr-dir", default="processed_data/hr", help="WRF HR zarr dir")
    parser.add_argument("--lr-dir", default="processed_data/lr", help="WRF LR zarr dir")
    parser.add_argument("--out-dir", default="processed_data/summary", help="Output dir")
    parser.add_argument("--samples", type=int, default=20, help="Samples for divergence/PSD")
    args = parser.parse_args()

    month = args.month
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_hr = xr.open_zarr(Path(args.hr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)
    ds_lr = xr.open_zarr(Path(args.lr_dir) / f"{month}.zarr", consolidated=True, decode_times=False)

    datasets = [("wrf_hr", ds_hr)]
    for pred_path in sorted(Path(args.pred_dir).glob(f"pred_*_{month}.zarr")):
        model, _ = _parse_pred_name(pred_path)
        if model not in {"bicubic", "unet", "physr"}:
            continue
        ds_pred = _open_pred_dataset(pred_path)
        datasets.append((model, ds_pred))

    # divergence PDF
    ds_list = [(_to_time_yx(ds["U10"]), _to_time_yx(ds["V10"])) for _, ds in datasets]
    labels = [name for name, _ in datasets]
    divergence_pdf(ds_list, labels, n_samples=args.samples,
                   out_path=out_dir / f"divergence_pdf_{month}.png")

    # PSD
    psd_plot(ds_list, labels, n_samples=max(5, args.samples // 2),
             out_path=out_dir / f"psd_{month}.png")

    # Scale consistency table
    rows = []
    for name, ds in datasets[1:]:
        u_m, v_m, s_m = scale_consistency(ds, ds_lr, scale=4, batch=8)
        for var_name, metrics in zip(["U10", "V10", "SPEED"], [u_m, v_m, s_m]):
            row = {"model": name, "month": month, "var": var_name}
            row.update(metrics)
            rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / f"scale_consistency_table_{month}.csv", index=False)
    print(f"Saved physics outputs to {out_dir}")


if __name__ == "__main__":
    main()
