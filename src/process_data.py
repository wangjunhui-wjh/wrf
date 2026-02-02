import shutil
import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import os
import math
import zarr

# Try importing wrf-python
try:
    from wrf import getvar, latlon_coords
    HAS_WRF = True
except ImportError:
    HAS_WRF = False
    print("Warning: 'wrf-python' not found. Using native extraction with manual rotation fallback.")

from . import config

def extract_variables(wrf_file_path):
    """
    Extracts required variables from a single WRF file.
    Returns an xarray Dataset loaded in memory.
    """
    try:
        ncfile = Dataset(wrf_file_path)
    except Exception as e:
        print(f"Error opening {wrf_file_path}: {e}")
        return None

    ds_vars = {}
    coords = {}
    wind_coord = "grid"
    
    # --- Helper to get variable safely ---
    def get_nc_var(name):
        if name in ncfile.variables:
            data = ncfile.variables[name][:]
            if data.ndim == 3 and data.shape[0] == 1:
                return data[0, :, :]
            return data
        return None

    # --- Extract XLAT / XLONG ---
    xlat = get_nc_var("XLAT")
    xlong = get_nc_var("XLONG")
    
    if xlat is not None: coords["XLAT"] = xlat
    if xlong is not None: coords["XLONG"] = xlong

    # --- Extract Wind (U10, V10) ---
    extracted_wind = False
    
    if HAS_WRF and ("U10" in config.VARIABLES_TO_EXTRACT or "V10" in config.VARIABLES_TO_EXTRACT):
        try:
            uvmet10 = getvar(ncfile, "uvmet10", units="m s-1")
            ds_vars["U10"] = uvmet10.sel(u_v="u").values
            ds_vars["V10"] = uvmet10.sel(u_v="v").values
            extracted_wind = True
            wind_coord = "earth"
        except Exception as e:
            pass

    if not extracted_wind and ("U10" in config.VARIABLES_TO_EXTRACT or "V10" in config.VARIABLES_TO_EXTRACT):
        # Fallback
        u10 = get_nc_var("U10")
        v10 = get_nc_var("V10")
        sina = get_nc_var("SINALPHA")
        cosa = get_nc_var("COSALPHA")
        
        if u10 is not None and v10 is not None:
            if sina is not None and cosa is not None:
                u_earth = u10 * cosa - v10 * sina
                v_earth = v10 * cosa + u10 * sina
                ds_vars["U10"] = u_earth
                ds_vars["V10"] = v_earth
                wind_coord = "earth"
            else:
                ds_vars["U10"] = u10
                ds_vars["V10"] = v10

    # --- Extract Other Variables ---
    for var in config.VARIABLES_TO_EXTRACT:
        if var in ["U10", "V10", "XLAT", "XLONG"]: continue
        
        if HAS_WRF:
            try:
                val = getvar(ncfile, var)
                ds_vars[var] = val.values
                continue
            except:
                pass
        
        val = get_nc_var(var)
        if val is not None:
            ds_vars[var] = val

    # --- Extract Static Terrain (HGT) if available ---
    hgt = get_nc_var("HGT")
    if hgt is None and HAS_WRF:
        try:
            hgt_da = getvar(ncfile, "HGT")
            hgt = hgt_da.values
        except Exception:
            hgt = None
    if hgt is not None:
        coords["HGT"] = hgt
    
    ncfile.close()
    
    if not ds_vars:
        return None

    try:
        # Create dataset
        # Important: Cast to float32 to save space
        data_vars = {k: (('south_north', 'west_east'), v.astype(np.float32)) for k, v in ds_vars.items()}
        coords_vars = {k: (('south_north', 'west_east'), v.astype(np.float32)) for k, v in coords.items()}
        
        ds = xr.Dataset(data_vars, coords=coords_vars)
        ds.attrs["wind_coord"] = wind_coord
    except ValueError as e:
        print(f"Shape mismatch in {wrf_file_path}: {e}")
        return None
        
    return ds

def generate_lr_from_hr(ds_hr, factor=config.DOWNSAMPLE_FACTOR):
    ds_lr = ds_hr.coarsen(south_north=factor, west_east=factor, boundary='trim').mean()
    ds_lr.attrs.update(ds_hr.attrs)
    ds_lr.attrs["downsample_method"] = "coarsen_mean_trim"
    return ds_lr

def get_compression_encoding(ds):
    """
    Returns compression encoding for all variables.
    Uses Zstd with level 3 (good balance).
    """
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = {var: {'compressor': compressor} for var in ds.data_vars}
    return encoding

def process_month(year, month, manifest_df):
    """
    Processes all files for a given month.
    Checks for existing Zarr to support resume/skip.
    """
    output_path_hr = config.get_output_path(year, month, 'hr')
    output_path_lr = config.get_output_path(year, month, 'lr')
    static_path = config.OUTPUT_DIR / "grid_static.zarr"

    # Check if completed
    if output_path_hr.exists() and output_path_lr.exists():
        print(f"Output for {year}-{month:02d} already exists. Skipping.")
        return

    # Filter manifest
    manifest_df['time'] = pd.to_datetime(manifest_df['time'])
    monthly_files = manifest_df[
        (manifest_df['time'].dt.year == year) & 
        (manifest_df['time'].dt.month == month)
    ].sort_values('time')
    
    if monthly_files.empty:
        print(f"No files found for {year}-{month:02d}")
        return

    print(f"Processing {year}-{month:02d}: {len(monthly_files)} files to {output_path_hr}")
    
    if config.COPY_TO_LOCAL:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check output again and clean if needed
    if output_path_hr.exists(): shutil.rmtree(output_path_hr)
    if output_path_lr.exists(): shutil.rmtree(output_path_lr)

    initialized = False

    for idx, row in tqdm(monthly_files.iterrows(), total=len(monthly_files), desc=f"Proc {year}-{month:02d}"):
        src_path = Path(row['path'])
        process_path = src_path
        
        # 1. Copy (Optional)
        if config.COPY_TO_LOCAL:
            try:
                temp_path = config.TEMP_DIR / src_path.name
                shutil.copy2(src_path, temp_path)
                process_path = temp_path
            except Exception as e:
                print(f"Failed to copy {src_path}: {e}")
                continue
            
        # 2. Extract
        ds = extract_variables(str(process_path))
        
        # 3. Cleanup Temp (if copied)
        if config.COPY_TO_LOCAL and process_path.exists():
            try: os.remove(process_path)
            except: pass
            
        if ds is None:
            continue

        # Save static data once (if not exists)
        if not static_path.exists():
            # Extract static coords
            ds_static = ds[['XLAT', 'XLONG']]
            if 'HGT' in ds:
                ds_static['HGT'] = ds['HGT']
            ds_static.to_zarr(static_path, mode='w')
            print(f"Static grid data saved to {static_path}")

        # Remove static coords from dynamic dataset to save space
        # We keep them as coordinates but drop them from data_vars if they leaked in
        # Actually, to be safe, let's just drop them from the dataset we write to time-series
        # But we need to keep them as coordinates for alignment? 
        # Zarr handles coordinates well, but let's be explicit:
        # We only want time-varying variables in the monthly zarrs.
        ds_dynamic = ds.drop_vars(['XLAT', 'XLONG', 'HGT'], errors='ignore')
        ds_dynamic.attrs["wind_coord"] = ds.attrs.get("wind_coord", "grid")
        
        # Add time dimension
        ds_dynamic = ds_dynamic.expand_dims(time=[row['time']])
        ds_lr = generate_lr_from_hr(ds_dynamic)
        
        # Chunking
        ds_dynamic = ds_dynamic.chunk(config.CHUNKS)
        ds_lr = ds_lr.chunk(config.CHUNKS)
        
        # Compression
        encoding_hr = get_compression_encoding(ds_dynamic)
        encoding_lr = get_compression_encoding(ds_lr)

        # Write/Append
        if not initialized:
            ds_dynamic.to_zarr(output_path_hr, mode='w', consolidated=True, encoding=encoding_hr)
            ds_lr.to_zarr(output_path_lr, mode='w', consolidated=True, encoding=encoding_lr)
            initialized = True
        else:
            ds_dynamic.to_zarr(output_path_hr, mode='a', append_dim='time', consolidated=True)
            ds_lr.to_zarr(output_path_lr, mode='a', append_dim='time', consolidated=True)
            
    print(f"Completed {year}-{month:02d}")
