"""Run 5-fold spatial block CV on full ASAAS habitats dataset."""

import sys
sys.path.insert(0, "src")

from resolve.data.dataset import ResolveDataset
from resolve.train.trainer import Trainer

# Data paths
HEADER = "J:/Phd Local/Gilles_paper_resolve/data/header_preprocessed_full.csv"
SPECIES = "J:/Phd Local/Gilles_paper_resolve/data/species_preprocessed_full.csv"

# Role mapping (matches bench scripts)
roles = {
    "plot_id": "PlotObservationID",
    "species_id": "WFO_TAXON",
    "species_plot_id": "PlotObservationID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover %",
    "taxonomy_genus": "WFO_GENUS",
    "taxonomy_family": "WFO_FAMILY",
}

# Targets: habitat classification (EUNIS L1 = 9 classes)
targets = {
    "habitat": {
        "column": "eunis_encoded",
        "task": "classification",
        "num_classes": 9,
    },
}

print("Loading dataset...")
dataset = ResolveDataset.from_csv(
    HEADER, SPECIES, roles, targets,
    species_normalization="norm",
)
print(f"Dataset: {dataset.n_plots:,} plots")

# Trainer config (matching v11.0 defaults)
trainer = Trainer(
    dataset,
    species_encoding="hash",
    hash_dim=32,
    hidden_dims=[2048, 1024, 512, 256, 128, 64],
    batch_size=32768,
    max_epochs=500,
    patience=50,
    lr=1e-3,
    loss_config="mae",
    verbose=1,
)

# Run 5-fold spatial block CV
result = trainer.cross_validate(
    n_splits=5,
    block_deg=1.0,
    seed=42,
)

print("\n" + str(result))
