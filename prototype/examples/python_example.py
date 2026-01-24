#!/usr/bin/env python3
"""
RESOLVE Python Example

Relational Encoding via Structured Observation Learning with Vector Embeddings

Demonstrates training a model using the data/obs pattern.
"""

import numpy as np
import pandas as pd
import torch

from resolve import PlotEncoder, DataSource, to_records


def create_sample_data(n_plots=100, n_species=50):
    """Create synthetic data for demonstration."""
    np.random.seed(42)

    genera = ['Quercus', 'Pinus', 'Acer', 'Betula', 'Fagus', 'Abies', 'Picea']
    families = ['Fagaceae', 'Pinaceae', 'Sapindaceae', 'Betulaceae']

    # Observation data (many per plot)
    obs_records = []
    for i in range(n_plots):
        plot_id = f'plot_{i:04d}'
        for _ in range(np.random.randint(3, 15)):
            genus = np.random.choice(genera)
            obs_records.append({
                'plot_id': plot_id,
                'species_id': f'species_{np.random.randint(0, n_species):03d}',
                'genus': genus,
                'family': families[genera.index(genus) % len(families)],
                'abundance': np.random.exponential(10)
            })

    # Plot data (one per plot)
    data_df = pd.DataFrame({
        'plot_id': [f'plot_{i:04d}' for i in range(n_plots)],
        'latitude': np.random.uniform(40, 60, n_plots),
        'longitude': np.random.uniform(-10, 30, n_plots),
        'elevation': np.random.uniform(0, 2000, n_plots),
        'ph': np.random.uniform(4, 8, n_plots),
        'nitrogen': np.random.exponential(0.2, n_plots),
        'carbon': np.random.exponential(3, n_plots),
    })

    return data_df, pd.DataFrame(obs_records)


def main():
    print("=" * 60)
    print("RESOLVE Python Example")
    print("=" * 60)

    # Create and split data
    print("\n1. Creating sample data...")
    data_df, obs_df = create_sample_data(n_plots=200, n_species=100)
    train_data, test_data = data_df.iloc[:160], data_df.iloc[160:]
    train_obs = obs_df[obs_df['plot_id'].isin(train_data['plot_id'])]
    test_obs = obs_df[obs_df['plot_id'].isin(test_data['plot_id'])]
    print(f"   Train: {len(train_data)} plots | Test: {len(test_data)} plots")

    # Create encoder
    print("\n2. Creating PlotEncoder...")
    encoder = PlotEncoder()
    encoder.add_numeric("coords", ["latitude", "longitude", "elevation"], source=DataSource.Plot)
    encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")
    encoder.add_embed("genus", ["genus"], dim=8, top_k=3, rank_by="abundance")

    # Encode using centralized to_records()
    print("\n3. Encoding data...")
    num_cols = ['latitude', 'longitude', 'elevation']
    encoded = encoder.fit_transform(
        to_records(train_data, numeric_cols=num_cols),
        to_records(train_obs, numeric_cols=['abundance'], cat_cols=['species_id', 'genus', 'family'], is_obs=True),
        train_data['plot_id'].tolist()
    )
    continuous = encoded.continuous_features()
    print(f"   Features shape: {continuous.shape}")

    # Train
    print("\n4. Training (5 epochs)...")
    model = torch.nn.Sequential(
        torch.nn.Linear(continuous.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 3)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    targets = torch.tensor(train_data[['ph', 'nitrogen', 'carbon']].values, dtype=torch.float32)

    for epoch in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(continuous), targets)
        loss.backward()
        optimizer.step()
        print(f"   Epoch {epoch + 1}: loss = {loss.item():.4f}")

    # Evaluate
    print("\n5. Evaluating...")
    model.eval()
    test_encoded = encoder.transform(
        to_records(test_data, numeric_cols=num_cols),
        to_records(test_obs, numeric_cols=['abundance'], cat_cols=['species_id', 'genus', 'family'], is_obs=True),
        test_data['plot_id'].tolist()
    )

    with torch.no_grad():
        pred = model(test_encoded.continuous_features()).numpy()

    for i, name in enumerate(['ph', 'nitrogen', 'carbon']):
        mae = np.mean(np.abs(pred[:, i] - test_data[name].values))
        print(f"   {name}: MAE = {mae:.4f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
