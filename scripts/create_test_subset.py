#!/usr/bin/env python3
"""Create a small test subset from ASAAS data."""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

# Paths
ASAAS_DIR = Path("J:/Phd Local/Gilles_paper2/Data/ASAAS/Data prep/98_Plot_Area")
OUT_DIR = Path("data/test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load headers
print("Loading header data...")
header = pd.read_csv(ASAAS_DIR / "ASAAS_HEADER_AREA_FULL.csv", low_memory=False)
print(f"Full header: {len(header)} rows")

# Filter to complete cases (no NA in required columns)
required_cols = ["Latitude", "Longitude", "Relevé area (m²)"]
header_complete = header.dropna(subset=required_cols)
print(f"Complete cases: {len(header_complete)} rows")

# Sample 1000 plots from complete cases
sample_ids = set(np.random.choice(header_complete["PlotObservationID"].values, size=1000, replace=False))
header_sample = header_complete[header_complete["PlotObservationID"].isin(sample_ids)]

# Load species (chunked for memory)
print("Loading species data...")
species_chunks = []
for chunk in pd.read_csv(ASAAS_DIR / "ASAAS_SPECIES_FULL.csv", chunksize=500000):
    filtered = chunk[chunk["PlotObservationID"].isin(sample_ids)]
    if len(filtered) > 0:
        species_chunks.append(filtered)

species_sample = pd.concat(species_chunks, ignore_index=True)
print(f"Sample: {len(header_sample)} plots, {len(species_sample)} species records")

# Summary stats
print(f"\nTarget stats:")
print(f"  Area: {header_sample['Relevé area (m²)'].describe()}")
print(f"  Altitude: {header_sample['Altitude (m)'].describe()}")
print(f"  EUNIS lvl1 classes: {header_sample['Eunis_lvl1'].nunique()}")

# Save
header_sample.to_csv(OUT_DIR / "asaas_header_sample.csv", index=False)
species_sample.to_csv(OUT_DIR / "asaas_species_sample.csv", index=False)
print(f"\nSaved to {OUT_DIR}/")
