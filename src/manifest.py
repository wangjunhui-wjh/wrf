import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys
from . import config

def parse_wrf_filename(filename):
    """
    Parses WRF filename to extract domain and timestamp.
    Example: wrfout_d04_2025-01-01_00_00_00
    """
    match = re.search(r"wrfout_(d\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2})", filename)
    if match:
        domain = match.group(1)
        time_str = match.group(2)
        dt = datetime.strptime(time_str, "%Y-%m-%d_%H_%M_%S")
        return domain, dt
    return None, None

def scan_directory(root_dir):
    """
    Recursively scans the directory for wrfout files.
    Returns a list of dictionaries with file metadata.
    """
    root_path = Path(root_dir)
    data = []
    
    print(f"Scanning directory: {root_path} (This may take a while for large drives...)", flush=True)
    
    file_count = 0
    # Use tqdm to show activity, even if total is unknown
    with tqdm(desc="Scanning files", unit="file") as pbar:
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.startswith("wrfout"):
                    file_path = Path(root) / file
                    domain, dt = parse_wrf_filename(file)
                    
                    if domain and dt:
                        if domain == config.DOMAIN:
                            stats = file_path.stat()
                            data.append({
                                "domain": domain,
                                "time": dt,
                                "path": str(file_path),
                                "size_bytes": stats.st_size,
                                "mtime": datetime.fromtimestamp(stats.st_mtime)
                            })
                            file_count += 1
                            pbar.update(1)
                            
                            # Force update description periodically
                            if file_count % 100 == 0:
                                pbar.set_postfix(found=file_count)
    
    print(f"\nFound {len(data)} valid files.", flush=True)
    return data

def check_completeness(df):
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("time")
    df['date'] = df['time'].dt.date
    daily_counts = df.groupby('date').size()
    
    missing_report = []
    
    for date, count in daily_counts.items():
        if count != config.EXPECTED_FILES_PER_DAY:
            missing_report.append({
                "date": date,
                "count": count,
                "expected": config.EXPECTED_FILES_PER_DAY,
                "status": "Incomplete"
            })
    
    return pd.DataFrame(missing_report)

def generate_manifest(output_csv_path=None):
    if output_csv_path is None:
        output_csv_path = config.OUTPUT_DIR.parent / "raw_manifest" / f"manifest_{datetime.now().year}.csv"
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("Starting manifest generation...", flush=True)
    
    # Check if manifest already exists and ask/skip? 
    # For automation, we usually regenerate or check timestamp.
    # Here we regenerate to be safe, but scanning is slow.
    # Optimization: If manifest exists and is recent, load it?
    # Let's Stick to scanning but with progress bar.
    
    records = scan_directory(config.RAW_DATA_ROOT)
    
    if not records:
        print("No WRF files found.", flush=True)
        return None

    df = pd.DataFrame(records)
    
    duplicates = df[df.duplicated(subset=['time', 'domain'], keep=False)]
    if not duplicates.empty:
        print(f"Warning: Found {len(duplicates)} duplicate timestamps!", flush=True)
    
    missing_report = check_completeness(df)
    if not missing_report.empty:
        print("Found days with missing files:", flush=True)
        print(missing_report, flush=True)
        missing_csv = output_csv_path.parent / "missing_report.csv"
        missing_report.to_csv(missing_csv, index=False)
    else:
        print("All days appear complete.", flush=True)

    df.to_csv(output_csv_path, index=False)
    print(f"Manifest saved to {output_csv_path}", flush=True)
    
    return df

if __name__ == "__main__":
    generate_manifest()
