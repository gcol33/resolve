#!/usr/bin/env python3
"""
RESOLVE Inference Example

Demonstrates loading a fitted encoder and making predictions on new data.
"""

import numpy as np
import pandas as pd
import os

from resolve import PlotEncoder, DataSource, to_records


def create_new_data(n_plots=10):
    """Create new data for inference."""
    np.random.seed(123)
    genera = ['Quercus', 'Pinus', 'Acer', 'Betula', 'Fagus']
    families = ['Fagaceae', 'Pinaceae', 'Sapindaceae', 'Betulaceae']

    obs_records = []
    for i in range(n_plots):
        plot_id = f'new_plot_{i:03d}'
        for _ in range(np.random.randint(5, 12)):
            genus = np.random.choice(genera)
            obs_records.append({
                'plot_id': plot_id,
                'species_id': f'species_{np.random.randint(0, 100):03d}',
                'genus': genus,
                'family': families[genera.index(genus) % len(families)],
                'abundance': np.random.exponential(10)
            })

    data_df = pd.DataFrame({
        'plot_id': [f'new_plot_{i:03d}' for i in range(n_plots)],
        'latitude': np.random.uniform(45, 55, n_plots),
        'longitude': np.random.uniform(0, 20, n_plots),
        'elevation': np.random.uniform(200, 1500, n_plots),
    })

    return data_df, pd.DataFrame(obs_records)


def main():
    print("=" * 60)
    print("RESOLVE Inference Example")
    print("=" * 60)

    encoder_path = "plot_encoder.json"

    if os.path.exists(encoder_path):
        print(f"\n1. Loading encoder from {encoder_path}...")
        encoder = PlotEncoder.load(encoder_path)
    else:
        print(f"\n1. Creating new encoder (no saved encoder found)...")
        encoder = PlotEncoder()
        encoder.add_numeric("coords", ["latitude", "longitude", "elevation"], source=DataSource.Plot)
        encoder.add_hash("species", ["species_id"], dim=32, top_k=5, rank_by="abundance")

    print("\n2. Creating new data...")
    data_df, obs_df = create_new_data(n_plots=10)
    print(f"   {len(data_df)} plots, {len(obs_df)} observations")

    print("\n3. Encoding...")
    num_cols = ['latitude', 'longitude', 'elevation']
    plot_records = to_records(data_df, numeric_cols=num_cols)
    obs_records = to_records(obs_df, cat_cols=['species_id', 'genus', 'family'], numeric_cols=['abundance'], is_obs=True)

    if not encoder.is_fitted():
        encoded = encoder.fit_transform(plot_records, obs_records, data_df['plot_id'].tolist())
    else:
        encoded = encoder.transform(plot_records, obs_records, data_df['plot_id'].tolist())

    print(f"   Features shape: {encoded.continuous_features().shape}")

    print("\n4. Mock predictions:")
    for plot_id in data_df['plot_id'][:5]:
        print(f"   {plot_id}: pH={5.5 + np.random.normal(0, 0.5):.2f}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
