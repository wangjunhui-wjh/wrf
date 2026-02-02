import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _open_zarr_manual(path):
    try:
        import zarr
    except Exception as exc:
        raise RuntimeError("zarr is required for manual zarr fallback.") from exc

    root = zarr.open_group(str(path), mode="r")
    array_keys = list(root.array_keys())
    if not array_keys:
        raise ValueError("No arrays found in zarr store.")

    time_arr = root["time"] if "time" in array_keys else None
    data_vars = {}
    for name in array_keys:
        if name == "time":
            continue
        arr = root[name]
        if arr.ndim == 3:
            dims = ("time", "y", "x")
        elif arr.ndim == 2:
            dims = ("y", "x")
        elif arr.ndim == 1 and time_arr is not None and arr.shape[0] == time_arr.shape[0]:
            dims = ("time",)
        else:
            dims = tuple(f"dim_{i}" for i in range(arr.ndim))
        data_vars[name] = (dims, arr)

    coords = {"time": ("time", time_arr)} if time_arr is not None else {}
    return xr.Dataset(data_vars=data_vars, coords=coords)


def _open_pred_dataset(path):
    path = Path(path)
    if path.is_dir() or path.suffix == ".zarr":
        last_err = None
        for consolidated in (True, False):
            try:
                return xr.open_zarr(path, consolidated=consolidated, decode_times=False)
            except Exception as exc:
                last_err = exc
        try:
            return xr.open_dataset(path, engine="zarr", decode_times=False)
        except Exception as exc:
            last_err = exc
        try:
            return _open_zarr_manual(path)
        except Exception:
            if last_err is not None:
                raise last_err
            raise
    return xr.open_dataset(path, decode_times=False)


def _resolve_uv_vars(ds, u_var=None, v_var=None):
    if u_var and v_var:
        if u_var not in ds or v_var not in ds:
            raise KeyError(f"U/V variables not found: {u_var}, {v_var}")
        return u_var, v_var

    candidates = [
        ("U10", "V10"),
        ("u10", "v10"),
        ("u", "v"),
        ("U", "V"),
    ]
    for u_name, v_name in candidates:
        if u_name in ds and v_name in ds:
            return u_name, v_name
    raise KeyError("Could not find U/V variables in dataset. Provide --u-var/--v-var.")


def _get_var(ds, names):
    for name in names:
        if name in ds:
            return ds[name]
        if name in ds.coords:
            return ds.coords[name]
    return None


def _get_latlon(ds, grid_static=None):
    lat = _get_var(ds, ["XLAT", "lat", "LAT", "latitude", "LATITUDE"])
    lon = _get_var(ds, ["XLONG", "lon", "LON", "longitude", "LONGITUDE"])
    if lat is not None and lon is not None:
        return lat.values, lon.values

    if grid_static:
        grid_static = Path(grid_static)
        if grid_static.exists():
            gs = xr.open_zarr(grid_static)
            lat = _get_var(gs, ["XLAT", "lat", "LAT", "latitude", "LATITUDE"])
            lon = _get_var(gs, ["XLONG", "lon", "LON", "longitude", "LONGITUDE"])
            if lat is not None and lon is not None:
                return lat.values, lon.values

    raise ValueError("Lat/Lon not found in dataset and no valid grid_static provided.")


def _find_spatial_dims(var):
    dims = list(var.dims)
    if "time" in dims:
        dims.remove("time")
    if len(dims) != 2:
        raise ValueError(f"Expected 2 spatial dims, got {dims}")
    return dims[0], dims[1]


def _nearest_index(lat_arr, lon_arr, lat, lon):
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        iy = int(np.nanargmin(np.abs(lat_arr - lat)))
        ix = int(np.nanargmin(np.abs(lon_arr - lon)))
        return iy, ix

    diff = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
    iy, ix = np.unravel_index(np.nanargmin(diff), diff.shape)
    return int(iy), int(ix)


def build_station_index(station_meta, lat_arr, lon_arr):
    indices = {}
    for _, row in station_meta.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        iy, ix = _nearest_index(lat_arr, lon_arr, row["lat"], row["lon"])
        indices[row["station_id"]] = (iy, ix)
    return indices


def extract_pred_at_stations(ds, u_name, v_name, station_meta, grid_static=None):
    lat_arr, lon_arr = _get_latlon(ds, grid_static=grid_static)
    y_dim, x_dim = _find_spatial_dims(ds[u_name])

    indices = build_station_index(station_meta, lat_arr, lon_arr)
    records = []

    for station_id, (iy, ix) in indices.items():
        u_series = ds[u_name].isel({y_dim: iy, x_dim: ix}).to_series()
        v_series = ds[v_name].isel({y_dim: iy, x_dim: ix}).to_series()
        df = pd.DataFrame({
            "time": u_series.index,
            "pred_u": u_series.values,
            "pred_v": v_series.values,
        })
        df["pred_speed"] = np.sqrt(df["pred_u"] ** 2 + df["pred_v"] ** 2)
        df["station_id"] = station_id
        records.append(df)

    if not records:
        return pd.DataFrame(columns=["time", "pred_u", "pred_v", "pred_speed", "station_id"])

    pred_df = pd.concat(records, ignore_index=True)
    pred_df["time"] = pd.to_datetime(pred_df["time"], errors="coerce")
    return pred_df


def _metrics(obs, pred):
    mask = np.isfinite(obs) & np.isfinite(pred)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "mae": np.nan, "r": np.nan}
    diff = pred[mask] - obs[mask]
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    if n >= 2:
        r = float(np.corrcoef(obs[mask], pred[mask])[0, 1])
    else:
        r = np.nan
    return {"n": n, "bias": bias, "rmse": rmse, "mae": mae, "r": r}


def _season_label(ts):
    month = ts.month
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def compute_station_metrics(obs_station_path, station_meta_path, pred_path, grid_static=None,
                            u_var=None, v_var=None, wind_bins=None):
    station_obs = pd.read_csv(obs_station_path)
    station_obs["time"] = pd.to_datetime(station_obs["time"], errors="coerce")
    station_meta = pd.read_csv(station_meta_path)

    if isinstance(pred_path, xr.Dataset):
        ds = pred_path
    else:
        ds = _open_pred_dataset(pred_path)
    u_name, v_name = _resolve_uv_vars(ds, u_var, v_var)

    pred_df = extract_pred_at_stations(ds, u_name, v_name, station_meta, grid_static=grid_static)
    pred_df = pred_df.dropna(subset=["time"])
    # Align to 30-min bins and remove duplicated timestamps from WRF outputs
    pred_df["time"] = pred_df["time"].dt.floor("30min")
    pred_df = pred_df.groupby(["station_id", "time"], as_index=False).mean(numeric_only=True)

    merged = station_obs.merge(pred_df, on=["station_id", "time"], how="inner")

    # metrics per station
    metrics_rows = []
    for station_id, group in merged.groupby("station_id"):
        m = _metrics(group["ws_mean"].to_numpy(), group["pred_speed"].to_numpy())
        metrics_rows.append({"station_id": station_id, **m})

    overall = _metrics(merged["ws_mean"].to_numpy(), merged["pred_speed"].to_numpy())
    metrics_rows.append({"station_id": "ALL", **overall})
    metrics_overall = pd.DataFrame(metrics_rows)

    # by season (all stations)
    merged["season"] = merged["time"].apply(_season_label)
    season_rows = []
    for season, group in merged.groupby("season"):
        m = _metrics(group["ws_mean"].to_numpy(), group["pred_speed"].to_numpy())
        season_rows.append({"season": season, **m})
    metrics_by_season = pd.DataFrame(season_rows)

    # by wind bins (all stations)
    if wind_bins is None:
        wind_bins = [0, 2, 4, 6, 8, 10, 12, 15, 20, np.inf]
    labels = []
    for i in range(len(wind_bins) - 1):
        right = "inf" if np.isinf(wind_bins[i + 1]) else str(wind_bins[i + 1])
        labels.append(f"{wind_bins[i]}-{right}")

    merged["wind_bin"] = pd.cut(merged["ws_mean"], bins=wind_bins, labels=labels, right=False)
    bin_rows = []
    for wind_bin, group in merged.groupby("wind_bin", observed=False):
        m = _metrics(group["ws_mean"].to_numpy(), group["pred_speed"].to_numpy())
        bin_rows.append({"wind_bin": wind_bin, **m})
    metrics_by_windbin = pd.DataFrame(bin_rows)

    return metrics_overall, metrics_by_season, metrics_by_windbin, merged


def _load_radar_meta(meta_path):
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_radar_timeseries(pred_ds, u_name, v_name, radar_csv_path, radar_meta_path,
                              grid_static, output_dir):
    radar_df = pd.read_csv(radar_csv_path)
    radar_df["time"] = pd.to_datetime(radar_df["time"], errors="coerce")
    radar_df = radar_df[radar_df["qc_flag"] == 0]

    meta = _load_radar_meta(radar_meta_path)
    if not meta.get("latitude") or not meta.get("longitude"):
        return None

    lat_arr, lon_arr = _get_latlon(pred_ds, grid_static=grid_static)
    y_dim, x_dim = _find_spatial_dims(pred_ds[u_name])
    iy, ix = _nearest_index(lat_arr, lon_arr, meta["latitude"], meta["longitude"])

    u_series = pred_ds[u_name].isel({y_dim: iy, x_dim: ix}).to_series()
    v_series = pred_ds[v_name].isel({y_dim: iy, x_dim: ix}).to_series()
    pred = pd.DataFrame({"time": u_series.index, "pred_u": u_series.values, "pred_v": v_series.values})
    pred["pred_speed"] = np.sqrt(pred["pred_u"] ** 2 + pred["pred_v"] ** 2)
    pred["time"] = pd.to_datetime(pred["time"], errors="coerce")

    merged = radar_df.merge(pred, on="time", how="inner")
    if merged.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(merged["time"], merged["speed"], label="Radar", linewidth=1.2)
    ax.plot(merged["time"], merged["pred_speed"], label="Pred", linewidth=1.2)
    ax.set_title(f"Radar vs Pred Speed: {radar_csv_path.stem}")
    ax.set_ylabel("m/s")
    ax.legend()
    fig.autofmt_xdate()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"{radar_csv_path.stem}_timeseries.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def divergence_stats(pred_ds, u_name, v_name, output_dir, max_times=200):
    y_dim, x_dim = _find_spatial_dims(pred_ds[u_name])
    if "time" in pred_ds[u_name].dims:
        u = pred_ds[u_name].isel(time=slice(0, max_times))
        v = pred_ds[v_name].isel(time=slice(0, max_times))
    else:
        u = pred_ds[u_name]
        v = pred_ds[v_name]

    u = u.transpose(..., y_dim, x_dim)
    v = v.transpose(..., y_dim, x_dim)
    u_vals = u.values
    v_vals = v.values

    # assume last two axes are y, x
    du_dx = np.gradient(u_vals, axis=-1)
    dv_dy = np.gradient(v_vals, axis=-2)
    div = du_dx + dv_dy

    flat = div.reshape(-1)
    flat = flat[np.isfinite(flat)]

    stats = {
        "count": int(flat.size),
        "mean": float(np.mean(flat)) if flat.size else np.nan,
        "std": float(np.std(flat)) if flat.size else np.nan,
        "p5": float(np.percentile(flat, 5)) if flat.size else np.nan,
        "p95": float(np.percentile(flat, 95)) if flat.size else np.nan,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "divergence_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(flat, bins=60, color="#4C78A8", alpha=0.8)
    ax.set_title("Divergence Histogram")
    ax.set_xlabel("divergence")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig_path = output_dir / "divergence_hist.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return stats_path, fig_path
