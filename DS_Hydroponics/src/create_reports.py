import os
import pandas as pd
import numpy as np
from datetime import datetime

# Paths
DATASET_PATH = 'data/hydroponics_dataset.csv'
OUTPUT_PATH = 'output'

def generate_dataset_summary(df, output_path):
    summary_rows = []
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        count = col_data.count()
        missing = col_data.isna().sum()
        unique = col_data.nunique()
        min_val = col_data.min() if pd.api.types.is_numeric_dtype(col_data) else ''
        max_val = col_data.max() if pd.api.types.is_numeric_dtype(col_data) else ''
        mean = col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else ''
        std = col_data.std() if pd.api.types.is_numeric_dtype(col_data) else ''
        samples = ', '.join([str(el) for el in col_data.dropna().unique()[:5]])
        summary_rows.append({
            'column': col,
            'type': dtype,
            'non_missing_count': count,
            'missing_count': missing,
            'unique_values': unique,
            'min': min_val,
            'max': max_val,
            'mean': mean,
            'std': std,
            'sample_values': samples
        })
    summary_df = pd.DataFrame(summary_rows)
    out_file = os.path.join(output_path, 'dataset_summary.csv')
    summary_df.to_csv(out_file, index=False)
    print(f"[+] dataset_summary.csv created at {out_file}")

def generate_simulation_report(df, output_path):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    out_file = os.path.join(output_path, f'simulation_report_{timestamp}.csv')
    
    # Example: Group by hour, compute sensor means and health/yield means (if timestamp exists)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        hourly_df = df.resample('H').mean().reset_index()
    else:
        # Just show means per column if no timestamp
        hourly_df = pd.DataFrame(df.mean()).T
    
    # Add summary stats for output
    summary_info = {
        'total_rows': len(df),
        'date_range_start': df.index.min() if 'timestamp' in df.columns else '',
        'date_range_end': df.index.max() if 'timestamp' in df.columns else '',
        'sensor_min_temperature': df['temperature'].min() if 'temperature' in df.columns else '',
        'sensor_max_temperature': df['temperature'].max() if 'temperature' in df.columns else '',
        'avg_growth_rate': df['growth_rate'].mean() if 'growth_rate' in df.columns else '',
        'avg_expected_yield': df['expected_yield'].mean() if 'expected_yield' in df.columns else '',
        'avg_health_score': df['health_score'].mean() if 'health_score' in df.columns else '',
        'avg_disease_risk': df['disease_risk'].mean() if 'disease_risk' in df.columns else ''
    }
    
    # Write simulation report
    with open(out_file, 'w', encoding='utf-8') as f:
        # Write header info
        for k, v in summary_info.items():
            f.write(f"{k},{v}\n")
        f.write('\n')
        # Write hourly averages or main stats
        hourly_df.to_csv(f, index=False)
    print(f"[+] simulation_report created at {out_file}")

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    generate_dataset_summary(df, OUTPUT_PATH)
    generate_simulation_report(df, OUTPUT_PATH)

if __name__ == "__main__":
    main()
