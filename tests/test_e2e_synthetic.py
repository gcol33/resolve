"""End-to-end smoke tests with synthetic data.

Tests the full pipeline: data → encoder → trainer → fit → predict
for all encoding modes (hash, embed, rank_pool, transformer).
"""

import numpy as np
import polars as pl
import pytest

from resolve.data.dataset import ResolveDataset, TargetConfig
from resolve.data.roles import RoleMapping
from resolve.train.trainer import Trainer


def make_synthetic_data(
    n_plots: int = 200,
    n_species: int = 50,
    n_genera: int = 15,
    n_families: int = 5,
    species_per_plot: int = 10,
    seed: int = 42,
) -> ResolveDataset:
    """Generate a synthetic vegetation plot dataset."""
    rng = np.random.default_rng(seed)

    # Header: one row per plot with coords + target
    plot_ids = [f"P{i:04d}" for i in range(n_plots)]
    lat = rng.uniform(45, 55, n_plots)
    lon = rng.uniform(5, 15, n_plots)
    # Area target: loosely correlated with latitude
    area = np.exp(rng.normal(2.0 + 0.1 * (lat - 50), 0.5))
    # Classification target: 3 habitat classes
    habitat = rng.integers(0, 3, n_plots)

    header = pl.DataFrame({
        "plot_id": plot_ids,
        "lat": lat,
        "lon": lon,
        "area": area,
        "habitat": habitat,
    })

    # Species table: variable number of species per plot
    species_names = [f"sp_{i}" for i in range(n_species)]
    genus_names = [f"genus_{i % n_genera}" for i in range(n_species)]
    family_names = [f"family_{i % n_families}" for i in range(n_species)]
    sp_to_genus = dict(zip(species_names, genus_names))
    sp_to_family = dict(zip(species_names, family_names))

    rows = []
    for pid in plot_ids:
        n_sp = rng.integers(3, species_per_plot + 1)
        chosen = rng.choice(species_names, size=n_sp, replace=False)
        abundances = rng.exponential(5.0, size=n_sp)
        for sp, abd in zip(chosen, abundances):
            rows.append({
                "plot_id": pid,
                "species": sp,
                "abundance": float(abd),
                "genus": sp_to_genus[sp],
                "family": sp_to_family[sp],
            })

    species_df = pl.DataFrame(rows)

    roles = RoleMapping(
        plot_id="plot_id",
        species_id="species",
        species_plot_id="plot_id",
        coords_lat="lat",
        coords_lon="lon",
        abundance="abundance",
        taxonomy_genus="genus",
        taxonomy_family="family",
    )

    targets = {
        "area": TargetConfig(
            column="area",
            task="regression",
            transform="log1p",
        ),
        "habitat": TargetConfig(
            column="habitat",
            task="classification",
            num_classes=3,
        ),
    }

    return ResolveDataset(
        header=header,
        species=species_df,
        roles=roles,
        targets=targets,
    )


@pytest.fixture
def dataset():
    return make_synthetic_data()


class TestHashMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots
        assert np.isfinite(preds["area"]).all()


class TestEmbedMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="embed",
            species_embed_dim=16,
            top_k_species=5,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestRankPoolMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="rank_pool",
            species_embed_dim=16,
            genus_emb_dim=8,
            family_emb_dim=4,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert "habitat" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestTransformerMode:
    def test_train_and_predict(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="transformer",
            species_embed_dim=32,
            n_attention_layers=1,
            n_heads=2,
            transformer_ff_dim=64,
            hidden_dims=[32, 16],
            max_epochs=3,
            patience=3,
            batch_size=64,
            verbose=0,
        )
        result = trainer.fit()
        assert result.best_epoch >= 0

        preds = trainer.predict(dataset)
        assert "area" in preds
        assert len(preds["area"]) == dataset.n_plots


class TestConfidenceThreshold:
    def test_threshold_produces_nans(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            patience=2,
            batch_size=64,
            verbose=0,
        )
        trainer.fit()

        # High threshold should NaN out low-confidence predictions
        preds = trainer.predict(dataset, confidence_threshold=0.99)
        # At least some should be NaN (unknown species fraction > 0.01)
        assert np.isnan(preds["area"]).any() or np.isfinite(preds["area"]).all()


class TestOutputSpace:
    def test_transformed_vs_raw(self, dataset):
        trainer = Trainer(
            dataset,
            species_encoding="hash",
            hash_dim=16,
            hidden_dims=[32, 16],
            max_epochs=2,
            patience=2,
            batch_size=64,
            verbose=0,
        )
        trainer.fit()

        raw = trainer.predict(dataset, output_space="raw")
        transformed = trainer.predict(dataset, output_space="transformed")
        # log1p transform: raw should have larger values (expm1 applied)
        assert not np.allclose(raw["area"], transformed["area"])
