import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _read_csv_with_fallback(path, **kwargs):
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", **kwargs)


def _coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _mask_sentinel(series, sentinels):
    series = series.copy()
    for s in sentinels:
        series = series.mask(series == s)
    series = series.mask(series >= 999000)
    series = series.mask(series <= -999000)
    return series


def _std_pop(series):
    return series.std(ddof=0)


def _parse_header_info(first_line):
    info = {}
    parts = first_line.strip().split(",")
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def _parse_latlon(value):
    if not value:
        return None
    value = value.strip()
    sign = 1.0
    if value[0] in ("N", "E", "+"):
        value = value[1:]
    elif value[0] in ("S", "W", "-"):
        sign = -1.0
        value = value[1:]
    try:
        return sign * float(value)
    except ValueError:
        return None


def _load_station_base(base_path):
    base_path = Path(base_path)
    if not base_path.exists():
        return None

    df = _read_csv_with_fallback(base_path)
    if "站号" not in df.columns or "纬度" not in df.columns or "经度" not in df.columns:
        return None

    df = df.rename(columns={"站号": "station_id", "纬度": "lat", "经度": "lon", "站名": "station_name"})
    df["station_id"] = df["station_id"].astype(str).str.strip()
    df["lat"] = _coerce_numeric(df["lat"])
    df["lon"] = _coerce_numeric(df["lon"])
    if "station_name" in df.columns:
        df["station_name"] = df["station_name"].astype(str).str.strip()
    return df[["station_id", "lat", "lon", "station_name"]]


def process_station_30min(
    station_dir,
    output_dir,
    min_valid=20,
    freq="30min",
    sentinels=(999999, 99999, 9999, 999),
    station_base_path="data/base/自动站信息表.csv",
    drop_missing_coords=True,
):
    station_dir = Path(station_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    station_files = sorted(station_dir.glob("*.csv"))
    if not station_files:
        print(f"No station CSV files found in {station_dir}")
        return None, None

    outputs = []
    station_meta_rows = []

    for path in station_files:
        df = _read_csv_with_fallback(path)
        if "Datetime" not in df.columns:
            print(f"Skipping {path} (missing Datetime column)")
            continue

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime"])

        df["Station_Id_C"] = df["Station_Id_C"].astype(str).str.strip()
        if "Station_Name" in df.columns:
            df["Station_Name"] = df["Station_Name"].astype(str).str.strip()

        wind = _mask_sentinel(_coerce_numeric(df["WIN_S_Avg_2mi"]), sentinels)
        wind = wind.mask(wind < 0)
        df["WIN_S_Avg_2mi"] = wind

        if "TEM" in df.columns:
            df["TEM"] = _mask_sentinel(_coerce_numeric(df["TEM"]), sentinels)
        if "RHU" in df.columns:
            df["RHU"] = _mask_sentinel(_coerce_numeric(df["RHU"]), sentinels)
        if "PRE" in df.columns:
            df["PRE"] = _mask_sentinel(_coerce_numeric(df["PRE"]), sentinels)

        df["time_30min"] = df["Datetime"].dt.floor(freq)

        agg = {
            "WIN_S_Avg_2mi": ["mean", _std_pop, "count"],
        }
        if "TEM" in df.columns:
            agg["TEM"] = ["mean"]
        if "RHU" in df.columns:
            agg["RHU"] = ["mean"]
        if "PRE" in df.columns:
            agg["PRE"] = ["mean"]

        grouped = df.groupby(["Station_Id_C", "Station_Name", "time_30min"], dropna=False).agg(agg)
        grouped.columns = [
            "ws_mean" if c[0] == "WIN_S_Avg_2mi" and c[1] == "mean" else
            "ws_std" if c[0] == "WIN_S_Avg_2mi" and c[1] == "_std_pop" else
            "n_valid" if c[0] == "WIN_S_Avg_2mi" and c[1] == "count" else
            f"{c[0].lower()}_mean"
            for c in grouped.columns
        ]
        grouped = grouped.reset_index()
        grouped["qc_flag"] = np.where(grouped["n_valid"] >= min_valid, 0, 1)
        outputs.append(grouped)

        station_id = df["Station_Id_C"].iloc[0]
        station_name = df["Station_Name"].iloc[0] if "Station_Name" in df.columns else ""
        station_meta_rows.append(
            {"station_id": station_id, "station_name": station_name, "lat": np.nan, "lon": np.nan}
        )

    if not outputs:
        print("No station data processed.")
        return None, None

    combined = pd.concat(outputs, ignore_index=True)
    combined = combined.sort_values(["Station_Id_C", "time_30min"])
    combined = combined.rename(columns={"Station_Id_C": "station_id", "Station_Name": "station_name", "time_30min": "time"})

    station_meta = pd.DataFrame(station_meta_rows).drop_duplicates("station_id")
    base_df = _load_station_base(station_base_path)
    if base_df is not None:
        station_meta = station_meta.merge(base_df, on="station_id", how="left", suffixes=("", "_base"))
        station_meta["lat"] = station_meta["lat_base"].combine_first(station_meta["lat"])
        station_meta["lon"] = station_meta["lon_base"].combine_first(station_meta["lon"])
        if "station_name_base" in station_meta.columns:
            station_meta["station_name"] = station_meta["station_name_base"].combine_first(station_meta["station_name"])
        station_meta = station_meta.drop(columns=[c for c in station_meta.columns if c.endswith("_base")])

    if drop_missing_coords and base_df is not None:
        valid_meta = station_meta.dropna(subset=["lat", "lon"])
        valid_ids = set(valid_meta["station_id"])
        before = len(combined)
        combined = combined[combined["station_id"].isin(valid_ids)].copy()
        combined = combined.sort_values(["station_id", "time"])
        station_meta = valid_meta
        print(f"Dropped stations without coords: {before - len(combined)} rows removed")

    csv_path = output_dir / "station_30min.csv"
    combined.to_csv(csv_path, index=False)

    nc_path = output_dir / "station_30min.nc"
    ds = combined.set_index(["station_id", "time"]).to_xarray()
    ds.to_netcdf(nc_path)
    station_meta_path = output_dir / "station_meta.csv"
    station_meta.to_csv(station_meta_path, index=False)

    return csv_path, nc_path


def _find_height_columns(columns):
    heights = {}
    pattern = re.compile(r"^(?P<h>\d+)m\s+WindSpeed$")
    for col in columns:
        m = pattern.match(col.strip())
        if m:
            h = int(m.group("h"))
            heights[h] = col
    return heights


def _select_height(columns, target_height):
    heights = _find_height_columns(columns)
    if not heights:
        return None
    selected = min(heights.keys(), key=lambda h: abs(h - target_height))
    speed_col = heights[selected]
    snr_col = f"{selected}m SNR(dB)"
    return selected, speed_col, snr_col if snr_col in columns else None


def process_radar_30min(
    lidar_root,
    output_dir,
    target_height=90,
    min_beam_count=3,
    min_snr=10.0,
    freq="30min",
):
    lidar_root = Path(lidar_root)
    if not lidar_root.exists():
        print(f"Radar root not found, skipping: {lidar_root}")
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_dirs = [p for p in lidar_root.iterdir() if p.is_dir()]
    outputs = []

    for site in site_dirs:
        csv_files = sorted(site.rglob("level1/*.csv"))
        if not csv_files:
            continue

        site_frames = []
        site_meta = None

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
            header_info = _parse_header_info(first_line)
            if site_meta is None:
                site_meta = {
                    "site": site.name,
                    "latitude": _parse_latlon(header_info.get("latitude")),
                    "longitude": _parse_latlon(header_info.get("longtitude")),
                    "height_m": float(header_info.get("height", "nan")) if header_info.get("height") else np.nan,
                    "scan_mode": header_info.get("ScanMode"),
                }

            df = _read_csv_with_fallback(csv_path, skiprows=1)
            if "Date_time" not in df.columns:
                continue

            height_info = _select_height(df.columns, target_height)
            if not height_info:
                continue
            selected_height, speed_col, snr_col = height_info

            df = df.rename(columns={"Date_time": "time"})
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d %H:%M:%S", errors="coerce")
            df = df.dropna(subset=["time"])

            df["Direction"] = df["Direction"].astype(str).str.upper().str.strip()
            df["Pitch"] = _coerce_numeric(df["Pitch"])

            df["speed"] = _coerce_numeric(df[speed_col])
            df["speed"] = _mask_sentinel(df["speed"], (999, 9999, 99999, 999999))
            if snr_col:
                df["snr"] = _coerce_numeric(df[snr_col])
                df["speed"] = df["speed"].mask(df["snr"] < min_snr)

            df = df[df["Direction"].isin(["N", "E", "S", "W"])]
            df = df.dropna(subset=["speed", "Pitch"])
            if df.empty:
                continue

            df["time_30min"] = df["time"].dt.floor(freq)

            beam_stats = df.groupby(["time_30min", "Direction"]).agg(
                vr_mean=("speed", "mean"),
                n=("speed", "count"),
            )
            beam_stats = beam_stats.reset_index()
            pivot = beam_stats.pivot(index="time_30min", columns="Direction", values="vr_mean")
            counts = beam_stats.pivot(index="time_30min", columns="Direction", values="n")

            pitch_stats = df.groupby("time_30min").agg(alpha=("Pitch", lambda s: float(np.nanmean(np.abs(s - 90.0)))))

            merged = pivot.join(pitch_stats, how="outer")
            merged["n_n"] = counts.get("N", pd.Series(index=merged.index, dtype="float64"))
            merged["n_e"] = counts.get("E", pd.Series(index=merged.index, dtype="float64"))
            merged["n_s"] = counts.get("S", pd.Series(index=merged.index, dtype="float64"))
            merged["n_w"] = counts.get("W", pd.Series(index=merged.index, dtype="float64"))

            def calc_uv(row):
                alpha = row["alpha"]
                if pd.isna(alpha) or alpha <= 0.0:
                    return np.nan, np.nan
                sin_alpha = math.sin(math.radians(alpha))
                if sin_alpha <= 1e-6:
                    return np.nan, np.nan
                if pd.isna(row.get("E")) or pd.isna(row.get("W")) or pd.isna(row.get("N")) or pd.isna(row.get("S")):
                    return np.nan, np.nan
                u = (row["E"] - row["W"]) / (2.0 * sin_alpha)
                v = (row["N"] - row["S"]) / (2.0 * sin_alpha)
                return u, v

            uv = merged.apply(calc_uv, axis=1, result_type="expand")
            merged["u"] = uv[0]
            merged["v"] = uv[1]
            merged["speed"] = np.sqrt(merged["u"] ** 2 + merged["v"] ** 2)

            merged["n_valid"] = merged[["n_n", "n_e", "n_s", "n_w"]].min(axis=1).astype("float64")
            merged["qc_flag"] = np.where(
                (merged["n_valid"] >= min_beam_count) & merged["u"].notna() & merged["v"].notna(),
                0,
                1,
            )

            merged["height_m"] = selected_height
            merged = merged.reset_index().rename(columns={"time_30min": "time"})
            site_frames.append(merged)

        if not site_frames:
            continue

        site_df = pd.concat(site_frames, ignore_index=True)
        site_df = site_df.sort_values("time")
        if site_df.empty:
            continue

        base_name = f"radar_{site.name}_{target_height}m_30min"
        csv_path = output_dir / f"{base_name}.csv"
        site_df.to_csv(csv_path, index=False)

        nc_path = output_dir / f"{base_name}.nc"
        ds = site_df.set_index("time").to_xarray()
        ds.to_netcdf(nc_path)

        meta_path = output_dir / f"{base_name}_meta.json"
        meta_payload = site_meta or {"site": site.name}
        meta_payload["target_height_m"] = target_height
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)

        outputs.append((csv_path, nc_path, meta_path))

    return outputs
