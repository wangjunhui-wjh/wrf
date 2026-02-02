import pandas as pd
import sys
from src import config, manifest, process_data

def main():
    # 1. Generate Manifest
    print("--- Step 1: Generating Manifest ---", flush=True)
    
    # Optional: Check if manifest exists and reuse to save time?
    # For now, generate fresh.
    manifest_df = manifest.generate_manifest()
    
    if manifest_df is None or manifest_df.empty:
        print("Manifest generation failed or no files found.", flush=True)
        return

    # 2. Process Data Month by Month
    print("\n--- Step 2: Processing Data ---", flush=True)
    
    # Ensure time column is datetime
    manifest_df['time'] = pd.to_datetime(manifest_df['time'])
    
    # Get unique Year-Month combinations sorted
    months = sorted(manifest_df['time'].dt.to_period('M').unique())
    
    print(f"Found data for {len(months)} months: {months}", flush=True)
    
    for period in months:
        year = period.year
        month = period.month
        
        print(f"\nProcessing Month: {year}-{month:02d}", flush=True)
        try:
            process_data.process_month(year, month, manifest_df)
        except Exception as e:
            print(f"Error processing {year}-{month:02d}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            # Continue to next month even if one fails
            continue
        
    print("\n--- Pipeline Execution Complete ---", flush=True)

if __name__ == "__main__":
    main()
