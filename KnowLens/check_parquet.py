import pandas as pd

df = pd.read_parquet("data/chunks.parquet")

print("Total chunk:", len(df))
print("Total dokumen:", df["source"].nunique())
print("\nDistribusi per dokumen:\n")
print(df["source"].value_counts())