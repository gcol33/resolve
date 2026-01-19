#!/usr/bin/env python3
"""
Minimal example of Spacc usage.

This example generates synthetic data and demonstrates the full workflow:
1. Create datasets
2. Build model
3. Train
4. Predict
"""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

import spacc


def generate_synthetic_data(n_plots: int = 500, seed: int = 42):
    """Generate synthetic ecological plot data."""
    np.random.seed(seed)

    # Generate plot header data
    # Latitude affects temperature and species composition
    latitudes = np.random.uniform(45, 55, n_plots)
    longitudes = np.random.uniform(5, 15, n_plots)

    # Area depends on habitat type (which depends on latitude)
    # Higher latitude = more forest = larger plots
    forest_prob = (latitudes - 45) / 10  # 0 at 45, 1 at 55
    is_forest = np.random.random(n_plots) < forest_prob

    areas = np.where(
        is_forest,
        np.random.exponential(400, n_plots),  # Forest: larger plots
        np.random.exponential(50, n_plots),   # Grassland: smaller plots
    )

    # Elevation correlates with latitude
    elevations = 500 + (latitudes - 45) * 100 + np.random.normal(0, 200, n_plots)
    elevations = np.clip(elevations, 100, 2500)

    # Temperature depends on latitude and elevation
    temperatures = 20 - (latitudes - 45) * 0.5 - elevations / 500 + np.random.normal(0, 1, n_plots)

    header = pd.DataFrame({
        "PlotID": [f"P{i:05d}" for i in range(n_plots)],
        "Latitude": latitudes,
        "Longitude": longitudes,
        "Area": areas,
        "Elevation": elevations,
        "Temperature": temperatures,
    })

    # Generate species data
    # Species composition depends on latitude (forest vs grassland species)
    forest_genera = ["Quercus", "Fagus", "Abies", "Picea", "Acer"]
    grassland_genera = ["Festuca", "Poa", "Carex", "Trifolium", "Plantago"]

    genus_to_family = {
        "Quercus": "Fagaceae",
        "Fagus": "Fagaceae",
        "Abies": "Pinaceae",
        "Picea": "Pinaceae",
        "Acer": "Sapindaceae",
        "Festuca": "Poaceae",
        "Poa": "Poaceae",
        "Carex": "Cyperaceae",
        "Trifolium": "Fabaceae",
        "Plantago": "Plantaginaceae",
    }

    species_rows = []
    for i in range(n_plots):
        # Number of species per plot
        n_species = np.random.randint(5, 25)

        # Choose genera based on forest probability
        if is_forest[i]:
            genera_pool = forest_genera + grassland_genera[:2]  # Mostly forest
            weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.05, 0.05]
        else:
            genera_pool = grassland_genera + forest_genera[:2]  # Mostly grassland
            weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.05, 0.05]

        for j in range(n_species):
            genus = np.random.choice(genera_pool, p=weights)
            species_rows.append({
                "PlotID": f"P{i:05d}",
                "Species": f"{genus}_sp{np.random.randint(1, 10)}",
                "Genus": genus,
                "Family": genus_to_family[genus],
                "Cover": np.random.exponential(10),
            })

    species = pd.DataFrame(species_rows)

    return header, species


def main():
    print("=" * 60)
    print("Spacc Minimal Example")
    print("=" * 60)

    # Generate data
    print("\n1. Generating synthetic data...")
    header, species = generate_synthetic_data(n_plots=500)
    print(f"   Created {len(header)} plots with {len(species)} species occurrences")

    # Save to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        header_path = Path(tmpdir) / "header.csv"
        species_path = Path(tmpdir) / "species.csv"
        model_path = Path(tmpdir) / "model.pt"

        header.to_csv(header_path, index=False)
        species.to_csv(species_path, index=False)

        # Define roles and targets
        roles = {
            "plot_id": "PlotID",
            "species_id": "Species",
            "species_plot_id": "PlotID",
            "coords_lat": "Latitude",
            "coords_lon": "Longitude",
            "abundance": "Cover",
            "taxonomy_genus": "Genus",
            "taxonomy_family": "Family",
            "covariates": ["Temperature"],
        }

        targets = {
            "area": {
                "column": "Area",
                "task": "regression",
                "transform": "log1p",
            },
            "elevation": {
                "column": "Elevation",
                "task": "regression",
            },
        }

        # Load dataset
        print("\n2. Loading dataset...")
        dataset = spacc.SpaccDataset.from_csv(
            header=header_path,
            species=species_path,
            roles=roles,
            targets=targets,
        )
        print(f"   Schema: {dataset.schema.n_plots} plots, {dataset.schema.n_species} species")
        print(f"   Taxonomy: {dataset.schema.n_genera} genera, {dataset.schema.n_families} families")

        # Build model
        print("\n3. Building model...")
        model = spacc.SpaccModel(
            schema=dataset.schema,
            targets=dataset.targets,
            hash_dim=32,
            hidden_dims=[256, 128, 64],
        )
        print(f"   Encoder latent dim: {model.latent_dim}")
        print(f"   Targets: {list(model.heads.keys())}")

        # Train
        print("\n4. Training...")
        trainer = spacc.Trainer(
            model=model,
            dataset=dataset,
            batch_size=64,
            max_epochs=50,
            patience=10,
            phase_boundaries=(15, 30),
            device="cpu",
        )
        result = trainer.fit()

        print(f"\n   Best epoch: {result.best_epoch}")
        print("   Final metrics:")
        for target, metrics in result.final_metrics.items():
            print(f"     {target}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if "band" in metric or "accuracy" in metric:
                        print(f"       {metric}: {value:.1%}")
                    else:
                        print(f"       {metric}: {value:.2f}")

        # Save model
        trainer.save(model_path)
        print(f"\n5. Model saved to {model_path}")

        # Load and predict
        print("\n6. Loading model and predicting...")
        predictor = spacc.Predictor.load(model_path, device="cpu")
        predictions = predictor.predict(dataset, return_latent=True)

        print(f"   Predictions shape: {len(predictions.plot_ids)} plots")
        print(f"   Latent embeddings shape: {predictions.latent.shape}")

        # Show sample predictions
        print("\n   Sample predictions vs actual:")
        for i in range(5):
            print(f"     Plot {predictions.plot_ids[i]}:")
            print(f"       Area: pred={predictions['area'][i]:.1f}, actual={header.iloc[i]['Area']:.1f}")
            print(f"       Elevation: pred={predictions['elevation'][i]:.1f}, actual={header.iloc[i]['Elevation']:.1f}")

        # Export predictions
        output_path = Path(tmpdir) / "predictions.csv"
        predictions.to_csv(output_path)
        print(f"\n7. Predictions exported to {output_path}")

        # Get learned embeddings
        print("\n8. Inspecting learned embeddings...")
        genus_emb = predictor.get_genus_embeddings()
        print(f"   Genus embedding shape: {genus_emb.shape}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
