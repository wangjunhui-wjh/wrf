import argparse
from pathlib import Path

import pandas as pd
import xarray as xr
from netCDF4 import Dataset


def main():
    parser = argparse.ArgumentParser(description="Add HGT to grid_static.zarr from a WRF file.")
    parser.add_argument("--grid-static", default="processed_data/grid_static.zarr", help="grid_static zarr path")
    parser.add_argument("--wrf-file", default="", help="WRF file path (wrfout_*)")
    parser.add_argument("--manifest", default="raw_manifest/manifest_2026.csv", help="Manifest CSV to find a WRF file")
    args = parser.parse_args()

    grid_static = Path(args.grid_static)
    wrf_file = Path(args.wrf_file) if args.wrf_file else None

    if wrf_file is None or not wrf_file.exists():
        manifest = Path(args.manifest)
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        df = pd.read_csv(manifest)
        if "path" not in df.columns:
            raise ValueError("Manifest CSV missing 'path' column.")
        wrf_file = Path(df["path"].iloc[0])
    if not wrf_file.exists():
        raise FileNotFoundError(f"WRF file not found: {wrf_file}")

    nc = Dataset(str(wrf_file))
    try:
        xlat = nc.variables.get("XLAT")
        xlon = nc.variables.get("XLONG")
        hgt = nc.variables.get("HGT")
        if hgt is None:
            raise KeyError("HGT not found in WRF file.")
        xlat = xlat[0, :, :] if xlat is not None and xlat.ndim == 3 else (xlat[:] if xlat is not None else None)
        xlon = xlon[0, :, :] if xlon is not None and xlon.ndim == 3 else (xlon[:] if xlon is not None else None)
        hgt = hgt[0, :, :] if hgt.ndim == 3 else hgt[:]
    finally:
        nc.close()

    data_vars = {}
    if xlat is not None:
        data_vars["XLAT"] = (("south_north", "west_east"), xlat.astype("float32"))
    if xlon is not None:
        data_vars["XLONG"] = (("south_north", "west_east"), xlon.astype("float32"))
    data_vars["HGT"] = (("south_north", "west_east"), hgt.astype("float32"))

    ds_static = xr.Dataset(data_vars=data_vars)
    if grid_static.exists():
        ds_static.to_zarr(grid_static, mode="a")
    else:
        grid_static.parent.mkdir(parents=True, exist_ok=True)
        ds_static.to_zarr(grid_static, mode="w")

    print(f"HGT saved to {grid_static}")


if __name__ == "__main__":
    main()
