#!/usr/bin/env python3
"""Test resolve with real ASAAS data subset."""

import resolve
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "test"

# Define roles mapping ASAAS columns to resolve roles
roles = {
    "plot_id": "PlotObservationID",
    "species_id": "WFO_TAXON",
    "species_plot_id": "PlotObservationID",
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
    "abundance": "Cover %",
    "taxonomy_genus": "WFO_GENUS",
    "taxonomy_family": "WFO_FAMILY",
    # Only use covariates with no/minimal NA values
    "covariates": [
        "log_N_species",
        "pct_tree",
        "pct_shrub",
        "pct_herb",
        "pct_neophyte",
    ],
}

# Define targets
# Note: For multi-target with different NA patterns, filter dataset first
targets = {
    "area": {
        "column": "Relevé area (m²)",
        "task": "regression",
        "transform": "log1p",
    },
}

def main():
    print("=" * 60)
    print("Testing RESOLVE with ASAAS data")
    print("=" * 60)

    # Load dataset
    print("\n1. Loading ASAAS subset...")
    dataset = resolve.ResolveDataset.from_csv(
        header=DATA_DIR / "asaas_header_sample.csv",
        species=DATA_DIR / "asaas_species_sample.csv",
        roles=roles,
        targets=targets,
    )

    schema = dataset.schema
    print(f"   Plots: {schema.n_plots}")
    print(f"   Species: {schema.n_species}")
    print(f"   Genera: {schema.n_genera}")
    print(f"   Families: {schema.n_families}")
    print(f"   Covariates: {len(schema.covariate_names)}")

    # Build model
    print("\n2. Building model...")
    model = resolve.ResolveModel(
        schema=schema,
        targets=dataset.targets,
        hash_dim=32,
        hidden_dims=[256, 128, 64],
    )
    print(f"   Latent dim: {model.latent_dim}")

    # Train
    print("\n3. Training (CPU, 30 epochs max)...")
    trainer = resolve.Trainer(
        model=model,
        dataset=dataset,
        batch_size=128,
        max_epochs=30,
        patience=10,
        phase_boundaries=(10, 20),
        device="cpu",
    )
    result = trainer.fit()

    print(f"\n   Best epoch: {result.best_epoch}")
    print("   Final metrics:")
    for target, metrics in result.final_metrics.items():
        print(f"     {target}:")
        for k, v in metrics.items():
            if "band" in k:
                print(f"       {k}: {v:.1%}")
            else:
                print(f"       {k}: {v:.2f}")

    # Save
    model_path = DATA_DIR / "asaas_test_model.pt"
    trainer.save(model_path)
    print(f"\n4. Model saved to {model_path}")

    # Predict
    print("\n5. Testing prediction...")
    predictor = resolve.Predictor.load(model_path, device="cpu")
    predictions = predictor.predict(dataset, return_latent=True)

    print(f"   Predictions: {len(predictions.plot_ids)} plots")
    print(f"   Latent shape: {predictions.latent.shape}")

    # Embeddings
    print("\n6. Learned embeddings:")
    genus_emb = predictor.get_genus_embeddings()
    family_emb = predictor.get_family_embeddings()
    print(f"   Genus: {genus_emb.shape}")
    print(f"   Family: {family_emb.shape}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
