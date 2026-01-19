"""Train Spacc model on ASAAS test subset."""
import pandas as pd
import torch
from resolve import SpaccDataset, Trainer
from resolve.data.roles import RoleMapping, TargetConfig

# Load data
header_path = "data/test/asaas_header_sample.csv"
species_path = "data/test/asaas_species_sample.csv"

header = pd.read_csv(header_path)
species = pd.read_csv(species_path)

print(f"Header shape: {header.shape}")
print(f"Species shape: {species.shape}")

# Prepare EUNIS target - encode to integers, handle missing/compound
eunis_mapping = {
    'M': 0, 'N': 1, 'P': 2, 'Q': 3, 'R': 4, 'S': 5, 'T': 6, 'U': 7, 'V': 8
}
header['eunis_encoded'] = header['Eunis_lvl1'].map(eunis_mapping)

# Filter to only rows with valid EUNIS and area
valid_mask = header['eunis_encoded'].notna() & header['Relevé area (m²)'].notna()
valid_ids = header.loc[valid_mask, 'PlotObservationID'].tolist()
print(f"Valid plots: {len(valid_ids)} / {len(header)}")

# Filter header and species
header = header[header['PlotObservationID'].isin(valid_ids)].copy()
species = species[species['PlotObservationID'].isin(valid_ids)].copy()

# Convert eunis to int
header['eunis_encoded'] = header['eunis_encoded'].astype(int)

print(f"Filtered header shape: {header.shape}")
print(f"Filtered species shape: {species.shape}")

# Define role mapping
roles = RoleMapping(
    plot_id="PlotObservationID",
    species_id="WFO_TAXON",
    species_plot_id="PlotObservationID",
    coords_lat="Latitude",
    coords_lon="Longitude",
    taxonomy_genus="WFO_GENUS",
    taxonomy_family="WFO_FAMILY",
    abundance="Cover %"
)

# Define targets as dict {name: TargetConfig}
targets = {
    "area": TargetConfig(
        column="Relevé area (m²)",
        task="regression",
        transform="log1p",
        weight=1.0
    ),
    "eunis": TargetConfig(
        column="eunis_encoded",
        task="classification",
        num_classes=9,  # 9 EUNIS L1 classes
        weight=1.0
    )
}

# Create dataset
dataset = SpaccDataset(
    header=header,
    species=species,
    roles=roles,
    targets=targets
)

print(f"\nDataset schema:")
print(f"  n_plots: {dataset.schema.n_plots}")
print(f"  n_species: {dataset.schema.n_species}")
print(f"  n_genera: {dataset.schema.n_genera}")
print(f"  n_families: {dataset.schema.n_families}")
print(f"  targets: {list(dataset.schema.targets.keys())}")

# Create model
from resolve import SpaccModel
model = SpaccModel(
    schema=dataset.schema,
    targets=targets,
    hidden_dims=[256, 128, 64]
)
print(f"  model latent_dim: {model.latent_dim}")

# Create trainer
trainer = Trainer(
    model=model,
    dataset=dataset,
    max_epochs=200,
    patience=30,
    batch_size=256,
    lr=1e-3,
    phase_boundaries=(50, 150)
)

# Train
print("\nTraining...")
result = trainer.fit()

print(f"\nBest epoch: {result.best_epoch}")
print("\nFinal metrics:")
for target, metrics in result.final_metrics.items():
    print(f"  {target}:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")

# Save model
save_path = "data/test/asaas_multitask_model.pt"
trainer.save(save_path)
print(f"\nModel saved to {save_path}")
