import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def _parse_month_from_name(name):
    match = re.search(r"(\d{4}-\d{2})", name)
    return match.group(1) if match else None


def _load_month_times(manifest_path, month):
    df = pd.read_csv(manifest_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"].dt.strftime("%Y-%m") == month]
    if df.empty:
        return None
    return df.sort_values("time")["time"].reset_index(drop=True)


def _fix_one(pred_path, manifest_path, round_freq):
    month = _parse_month_from_name(pred_path.name)
    if not month:
        print(f"Skip {pred_path.name}: cannot parse month.")
        return False

    times = _load_month_times(manifest_path, month)
    if times is None:
        print(f"Skip {pred_path.name}: no manifest times for {month}.")
        return False

    if round_freq:
        times = times.dt.floor(round_freq)

    root = zarr.open_group(str(pred_path), mode="a")
    if "time" not in root:
        print(f"Skip {pred_path.name}: no time array.")
        return False

    tarr = root["time"]
    if len(times) != tarr.shape[0]:
        print(
            f"Skip {pred_path.name}: length mismatch (manifest {len(times)} != pred {tarr.shape[0]})."
        )
        return False

    time_ns = times.astype("datetime64[ns]").astype("int64").to_numpy()
    tarr[:] = time_ns
    tarr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    print(f"Fixed time for {pred_path.name} ({month}, n={len(times)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix pred zarr time axis using manifest times.")
    parser.add_argument("--pred-dir", default="processed_data/pred", help="Prediction zarr directory")
    parser.add_argument("--manifest", default="raw_manifest/manifest_2026.csv", help="Manifest CSV path")
    parser.add_argument("--round", default="30min", help="Floor time to this freq (e.g. 30min), empty to disable")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    manifest_path = Path(args.manifest)
    if not pred_dir.exists():
        raise FileNotFoundError(f"pred dir not found: {pred_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    round_freq = args.round.strip() if args.round else ""
    if not round_freq:
        round_freq = None

    for pred_path in sorted(pred_dir.glob("pred_*.zarr")):
        _fix_one(pred_path, manifest_path, round_freq)


if __name__ == "__main__":
    main()
