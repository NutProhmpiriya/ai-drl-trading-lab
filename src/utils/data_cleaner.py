import pandas as pd
import os
from pathlib import Path

def clean_forex_data(input_file: str, output_file: str = None) -> None:
    """
    Clean forex data to keep only required columns and format timestamp.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file. If None, will overwrite input file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Select and reorder columns
    required_columns = [
        'time', 'open', 'high', 'low', 'close',
        'tick_volume', 'spread', 'real_volume'
    ]
    
    # Ensure all required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only required columns
    df = df[required_columns]
    
    # Convert time to datetime if it isn't already
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time
    df = df.sort_values('time')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Save to output file
    output_file = output_file or input_file
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")

def clean_all_raw_data(raw_data_dir: str) -> None:
    """
    Clean all CSV files in the raw data directory.
    
    Args:
        raw_data_dir: Path to directory containing raw CSV files
    """
    raw_data_path = Path(raw_data_dir)
    
    if not raw_data_path.exists():
        raise ValueError(f"Directory not found: {raw_data_dir}")
    
    csv_files = list(raw_data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        return
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        try:
            clean_forex_data(str(csv_file))
            print(f"Successfully cleaned {csv_file.name}")
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    raw_data_dir = project_root / "data" / "raw"
    
    print(f"Cleaning data in {raw_data_dir}")
    clean_all_raw_data(str(raw_data_dir))
