import pandas as pd
import numpy as np

try:
    df = pd.read_csv('final_dataset_safe.csv')
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nLULC counts:")
    print(df['LULC'].value_counts())
    print("\nSoil counts:")
    print(df['soil'].value_counts())
    
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Check ranges for specific columns
    for col in ['rainfall', 'slope', 'NDVI', 'SMI', 'elevation']:
        if col in df.columns:
            print(f"\n{col} range: {df[col].min()} to {df[col].max()}")
except Exception as e:
    print(f"Error: {e}")
