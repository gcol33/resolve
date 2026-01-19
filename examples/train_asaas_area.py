"""Train RESOLVE model on full ASAAS data for plot area prediction."""
import pandas as pd
from resolve import ResolveDataset, ResolveModel, Trainer
from resolve.data.roles import RoleMapping, TargetConfig
from resolve.train.loss import PhaseConfig

# Data paths - use existing test subset
header_path = "data/test/asaas_header_sample.csv"
species_path = "data/test/asaas_species_sample.csv"

print("Loading data...")
header = pd.read_csv(header_path, low_memory=False)
species = pd.read_csv(species_path)

print(f"Header shape: {header.shape}")
print(f"Species shape: {species.shape}")

# Covariate columns (environmental + traits)
covariate_cols = [
    "Cover total (%)",
    "Cover tree layer (%)",
    "Cover shrub layer (%)",
    "Cover herb layer (%)",
    "Cover moss layer (%)",
    "Aspect (°)",
    "Slope (°)",
    "Altitude (m)",
    "log_N_species",
    "mean_SLA",
    "mean_PlantHeight",
    "mean_SeedMass",
    "mean_LDMC",
    "pct_tree",
    "pct_shrub",
    "pct_herb",
    "pct_juvenile",
    "pct_seedling",
    "pct_moss",
    "pct_neophyte",
]

# Fill missing covariates with 0 (handle '-' as missing)
for col in covariate_cols:
    if col in header.columns:
        header[col] = pd.to_numeric(header[col], errors='coerce').fillna(0)

# Define role mapping
roles = RoleMapping(
    plot_id="PlotObservationID",
    species_id="WFO_TAXON",
    species_plot_id="PlotObservationID",
    coords_lat="Latitude",
    coords_lon="Longitude",
    taxonomy_genus="WFO_GENUS",
    taxonomy_family="WFO_FAMILY",
    abundance="Cover %",
    covariates=covariate_cols,
)

# Define target (area prediction with log1p transform)
targets = {
    "area": TargetConfig(
        column="Relevé area (m²)",
        task="regression",
        transform="log1p",
        weight=1.0,
    ),
}

# Create dataset
print("\nCreating dataset...")
dataset = ResolveDataset(
    header=header,
    species=species,
    roles=roles,
    targets=targets,
)

print(f"Dataset schema:")
print(f"  n_plots: {dataset.schema.n_plots:,}")
print(f"  n_species: {dataset.schema.n_species:,}")
print(f"  n_genera: {dataset.schema.n_genera:,}")
print(f"  n_families: {dataset.schema.n_families:,}")
print(f"  n_covariates: {len(dataset.schema.covariate_names)}")

# Smaller architecture for testing (full: [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32])
hidden_dims = [512, 256, 128, 64, 32]

model = ResolveModel(
    schema=dataset.schema,
    targets=targets,
    hash_dim=32,
    top_k=3,
    genus_emb_dim=8,
    family_emb_dim=8,
    hidden_dims=hidden_dims,
    dropout=0.3,
)

print(f"\nModel:")
print(f"  hidden_dims: {hidden_dims}")
print(f"  latent_dim: {model.latent_dim}")
print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

# Three-phase loss schedule (scaled down for test)
# Phase 1: MAE only
# Phase 2: MAE + SMAPE
# Phase 3: MAE + SMAPE + band
phases = {
    1: PhaseConfig(mae=1.0),
    2: PhaseConfig(mae=0.8, smape=0.2),
    3: PhaseConfig(mae=0.8, smape=0.15, band=0.05, band_threshold=0.25),
}
phase_boundaries = [30, 80]

# Create trainer
trainer = Trainer(
    model=model,
    dataset=dataset,
    batch_size=512,
    max_epochs=150,
    patience=30,
    lr=1e-3,
    weight_decay=1e-4,
    phases=phases,
    phase_boundaries=phase_boundaries,
    use_amp=True,
)

# Train
print("\nTraining...")
print(f"  batch_size: {trainer.batch_size:,}")
print(f"  max_epochs: {trainer.max_epochs}")
print(f"  use_amp: {trainer.use_amp}")
print()

result = trainer.fit()

print(f"\n{'='*60}")
print(f"Best epoch: {result.best_epoch}")
print(f"\nFinal metrics:")
for target, metrics in result.final_metrics.items():
    print(f"  {target}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"    {metric}: {value:.4f}")

# Save model
save_path = "data/test/resolve_area_model.pt"
trainer.save(save_path)
print(f"\nModel saved to {save_path}")
