import pandas as pd
print("Original Dataset Columns:")
df_orig = pd.read_csv('landslide_dataset.csv')
print(df_orig.columns.tolist())
print(df_orig.head())

print("\nNew Dataset Columns:")
df_new = pd.read_csv('final_dataset_safe.csv')
print(df_new.columns.tolist())
print(df_new.head())
