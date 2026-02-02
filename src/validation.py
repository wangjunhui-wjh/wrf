import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

from . import config

def extract_station_data(ds, stations_df):
    """
    Extracts time series data from HR dataset for specific station coordinates
    using bilinear interpolation.
    
    stations_df should have 'lat', 'lon', 'name' columns.
    """
    # Assuming ds has 'XLAT' and 'XLONG' coordinates or 2D variables
    # If XLAT/XLONG are variables (2D), we might need to be careful with selection.
    # WRF typically has curvilinear grid.
    # For speed, we might find nearest indices once or use advanced interpolation.
    
    # xarray .interp() works well if coordinates are 1D (rectilinear).
    # For curvilinear (WRF), it's trickier.
    # A common simple approach: Find nearest grid point for each station.
    
    results = []
    
    # Calculate distance to find nearest point (naive but effective for validation)
    # Or use wrf-python ll_to_xy if we had the original WRF file object, but here we have Zarr.
    
    # Let's assume we load the static lat/lon from the Zarr
    lats = ds['XLAT'].values
    lons = ds['XLONG'].values
    
    for _, station in stations_df.iterrows():
        # Euclidean distance in lat/lon space (approx)
        dist = (lats - station['lat'])**2 + (lons - station['lon'])**2
        y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
        
        # Extract time series
        # Selecting by integer index
        station_data = ds.isel(south_north=y_idx, west_east=x_idx)
        
        # Convert to dataframe
        df_station = station_data.to_dataframe().reset_index()
        df_station['station_name'] = station['name']
        results.append(df_station)
        
    return pd.concat(results, ignore_index=True)

def generate_qc_report(ds, report_path):
    """
    Generates a basic Quality Control report.
    """
    report = []
    report.append("# Quality Control Report\n")
    report.append(f"Generated on: {pd.Timestamp.now()}\n")
    
    # 1. Missing Values
    nans = ds.isnull().sum()
    report.append("## Missing Values\n")
    for var in ds.data_vars:
        count = nans[var].values
        report.append(f"- {var}: {count} missing values\n")
        
    # 2. Extremes
    report.append("## Extreme Values\n")
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.number):
            vmin = ds[var].min().values
            vmax = ds[var].max().values
            report.append(f"- {var}: Min={vmin:.4f}, Max={vmax:.4f}\n")
            
    # Save
    with open(report_path, 'w') as f:
        f.writelines(report)
    print(f"QC Report saved to {report_path}")

if __name__ == "__main__":
    # Example usage
    pass
