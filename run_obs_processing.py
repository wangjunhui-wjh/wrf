import argparse
from pathlib import Path

from src.obs_processing import process_station_30min, process_radar_30min


def main():
    parser = argparse.ArgumentParser(description="Process station and lidar observations into 30min products.")
    parser.add_argument("--station-dir", default="data/stationdata", help="Directory of station CSV files")
    parser.add_argument("--lidar-dir", default="data/lidar", help="Root directory of lidar CSV folders")
    parser.add_argument("--out-dir", default="processed_data/obs", help="Output directory for processed products")
    parser.add_argument("--target-height", type=int, default=90, help="Target height (m) for radar product")
    parser.add_argument("--min-valid", type=int, default=20, help="Minimum samples per 30min window for station QC")
    parser.add_argument("--station-base", default="data/base/自动站信息表.csv", help="Station base info CSV path")
    parser.add_argument("--keep-missing-coords", action="store_true", help="Keep stations without lat/lon in station outputs")
    parser.add_argument("--min-beam-count", type=int, default=3, help="Minimum samples per beam for radar QC")
    parser.add_argument("--min-snr", type=float, default=10.0, help="Minimum SNR threshold for radar QC")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Processing station data...")
    process_station_30min(
        args.station_dir,
        out_dir,
        min_valid=args.min_valid,
        station_base_path=args.station_base,
        drop_missing_coords=not args.keep_missing_coords,
    )

    print("Processing radar data...")
    process_radar_30min(
        args.lidar_dir,
        out_dir,
        target_height=args.target_height,
        min_beam_count=args.min_beam_count,
        min_snr=args.min_snr,
    )

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
