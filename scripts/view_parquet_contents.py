import pandas as pd
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_parquet_contents.py <path_to_parquet_file>")
        sys.exit(1)
    
    parquet_file_path = sys.argv[1]
    
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Contents of {parquet_file_path}:")
        print(df.to_string())
    except Exception as e:
        print(f"Error reading or printing Parquet file: {e}")
        sys.exit(1)
