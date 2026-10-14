import pandas as pd
import os

INPUT = "data/processed/gnn_dataset_ngdr.csv"
OUTPUT = "data/processed/pilot_dataset.csv"

print("Loading NGDR-enriched dataset...")

df = pd.read_csv(INPUT)

# Keep only nodes with valid NGDR lithology
pilot = df[df["ngdr_lithology"].notna()].copy()

# Reset index
pilot.reset_index(drop=True, inplace=True)

# Save
os.makedirs("data/processed", exist_ok=True)
pilot.to_csv(OUTPUT, index=False)

print("\n==============================")
print("PILOT DATASET CREATED")
print("==============================")

print("Total pilot nodes:", len(pilot))
print("Features:", len(pilot.columns))

print("\nLithology distribution:")
print(pilot["ngdr_lithology"].value_counts())

print("\nSaved to:")
print(OUTPUT)