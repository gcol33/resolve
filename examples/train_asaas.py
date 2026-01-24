"""Train RESOLVE model on ASAAS test subset."""
import pandas as pd
from resolve import ResolveDataset, Trainer

# Load data
header_path = "data/test/asaas_header_sample.csv"
species_path = "data/test/asaas_species_sample.csv"

header = pd.read_csv(header_path)
species = pd.read_csv(species_path)

print(f"Header shape: {header.shape}")
print(f"Species shape: {species.shape}")

# Prepare EUNIS target - encode to integers
eunis_mapping = {'M': 0, 'N': 1, 'P': 2, 'Q': 3, 'R': 4, 'S': 5, 'T': 6, 'U': 7, 'V': 8}
header['eunis_encoded'] = header['Eunis_lvl1'].map(eunis_mapping)

# Filter to valid plots
valid_mask = header['eunis_encoded'].notna() & header['Relevé area (m²)'].notna()
valid_ids = set(header.loc[valid_mask, 'PlotObservationID'])
header = header[header['PlotObservationID'].isin(valid_ids)].copy()
species = species[species['PlotObservationID'].isin(valid_ids)].copy()
header['eunis_encoded'] = header['eunis_encoded'].astype(int)

print(f"Filtered: {len(header)} plots, {len(species)} species records")

# Save filtered data to temp CSV and use from_csv (handles dict conversion)
header.to_csv("data/test/temp_header.csv", index=False)
species.to_csv("data/test/temp_species.csv", index=False)

dataset = ResolveDataset.from_csv(
    header="data/test/temp_header.csv",
    species="data/test/temp_species.csv",
    roles={
        "plot_id": "PlotObservationID",
        "species_id": "WFO_TAXON",
        "species_plot_id": "PlotObservationID",
        "coords_lat": "Latitude",
        "coords_lon": "Longitude",
        "taxonomy_genus": "WFO_GENUS",
        "taxonomy_family": "WFO_FAMILY",
        "abundance": "Cover %",
    },
    targets={
        "area": {"column": "Relevé area (m²)", "task": "regression", "transform": "log1p"},
        "eunis": {"column": "eunis_encoded", "task": "classification", "num_classes": 9},
    },
)

print(f"\nSchema: {dataset.schema.n_plots} plots, {dataset.schema.n_species} species")

# Train with new API
trainer = Trainer(dataset, max_epochs=200, patience=30)
print("\nTraining...")
result = trainer.fit()

print(f"\nBest epoch: {result.best_epoch}")
for target, metrics in result.final_metrics.items():
    print(f"  {target}: band_25={metrics.get('band_25', 'N/A'):.2%}" if 'band_25' in metrics else f"  {target}: acc={metrics.get('accuracy', 'N/A'):.2%}")

# Predict with confidence threshold
preds = trainer.predict(dataset, confidence_threshold=0.8)
print(f"\nPredictions (80% confidence): {sum(~pd.isna(preds['area']))} / {len(preds['area'])} non-NA")

trainer.save("data/test/asaas_model.pt")
print("Model saved.")
