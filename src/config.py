from pathlib import Path

# --- Path Configurations ---
# Root directory of the raw data on the external drive (Adjust as needed)
# Assumes structure: E:\YYYY.M\YYYY-MM-DD_00\wrfout_d04_*
RAW_DATA_ROOT = Path("E:/") 

# Temporary directory on local SSD for faster IO during processing
# Set COPY_TO_LOCAL = True to enable copying before processing (recommended for HDDs with bad random seek)
TEMP_DIR = Path("D:/wrf_tmp")
COPY_TO_LOCAL = False  # Changed to False based on feedback to speed up pipeline

# Output directory for processed data
OUTPUT_DIR = Path("processed_data")

# --- Processing Configurations ---
DOMAIN = "d04"  # Target domain

# Variables to extract
VARIABLES_TO_EXTRACT = [
    "U10", "V10", 
    "XLAT", "XLONG", 
    "T2", "PSFC", "PBLH"
]

# Time interval in the filename (e.g., 30 minutes)
TIME_INTERVAL_MINUTES = 30
EXPECTED_FILES_PER_DAY = 48

# --- Super-Resolution Configurations ---
# Downsampling factor for LR generation
DOWNSAMPLE_FACTOR = 4

# --- Output Formats ---
OUTPUT_FORMAT = "zarr"

# --- Zarr Chunking Strategy ---
CHUNKS = {'time': 48, 'south_north': 100, 'west_east': 100}

def get_output_path(year, month, resolution="hr"):
    folder = OUTPUT_DIR / resolution
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{year}-{month:02d}.zarr"
