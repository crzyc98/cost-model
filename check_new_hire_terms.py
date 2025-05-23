import pandas as pd
import sys

def check_new_hire_terminations(event_log_path):
    try:
        # Read the parquet file
        df = pd.read_parquet(event_log_path)
        
        # Check if required columns exist
        required_cols = ['event_type', 'meta']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {', '.join(missing_cols)}")
            print("Available columns:", ', '.join(df.columns))
            return
        
        # Filter for new hire terminations
        new_hire_terms = df[
            (df['event_type'] == 'EVT_TERM') & 
            (df['meta'] == 'new_hire_term')
        ]
        
        print(f"Found {len(new_hire_terms)} new hire terminations in the event log.")
        
        if not new_hire_terms.empty:
            # Display relevant columns if they exist
            display_cols = ['employee_id', 'event_date', 'event_type', 'meta']
            display_cols = [col for col in display_cols if col in df.columns]
            print("\nSample of new hire terminations:")
            print(new_hire_terms[display_cols].head())
            
            # Check if there's a 'year' column to show distribution by year
            if 'year' in df.columns:
                print("\nTerminations by year:")
                print(new_hire_terms.groupby('year').size())
                
    except Exception as e:
        print(f"Error processing event log: {str(e)}")
        if 'df' in locals():
            print("\nAvailable columns in the event log:")
            print(df.columns.tolist())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_new_hire_terms.py <path_to_event_log.parquet>")
        sys.exit(1)
    
    event_log_path = sys.argv[1]
    check_new_hire_terminations(event_log_path)
