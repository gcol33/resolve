"""The species graph a heterogeneous GNN passes messages on.

``HeterogeneousGNNConfig`` carries four fields describing how to build one --
``use_taxonomic_edges``, ``use_cooccurrence_edges``, ``k_cooccurrence`` and
``cooccurrence_threshold`` -- and no engine code read any of them; nor could a
graph be handed in from Python at all, so every ``heterogeneous_gnn`` forward
threw "Species graph not set." These cases cover the Python surface of the fix:
the builder, the dataset's taxonomy of its own vocabulary, and a model that
trains on the graph the trainer builds.
"""

from __future__ import annotations

import pytest

import resolve_core as rc

from conftest import make_dataset_config, make_roles, make_targets, make_train_config


def sparse_dataset(plot_csvs, *, use_taxonomy: bool = True) -> "rc.ResolveDataset":
    config = make_dataset_config(rc.SpeciesEncodingMode.Sparse,
                                 use_taxonomy=use_taxonomy)
    return rc.ResolveDataset.from_csv(
        plot_csvs.header, plot_csvs.species,
        make_roles(taxonomy=use_taxonomy), make_targets(), config)


def hetero_config(**overrides) -> "rc.HeterogeneousGNNConfig":
    config = rc.HeterogeneousGNNConfig()
    config.hidden_dim = 8
    config.output_dim = 6
    config.n_layers = 1
    config.n_heads = 2
    config.dropout = 0.0
    config.cooccurrence_threshold = 0.01
    config.k_cooccurrence = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_the_graph_carries_one_edge_type_per_relation(plot_csvs):
    dataset = sparse_dataset(plot_csvs)
    graph = rc.build_species_graph(dataset, hetero_config())

    assert graph.n_species == dataset.schema.n_species_vocab
    assert graph.n_edges > 0
    assert graph.edge_index.shape == (2, graph.n_edges)
    assert graph.edge_type.shape == (graph.n_edges,)

    types = set(int(t) for t in graph.edge_type.tolist())
    assert types <= {int(rc.SpeciesEdgeType.SameGenus),
                     int(rc.SpeciesEdgeType.SameFamily),
                     int(rc.SpeciesEdgeType.CoOccurrence)}
    # The corpus has several genera, several families and species that share
    # plots, so all three relations are present.
    assert types == {0, 1, 2}
    # <UNK> (code 0) is a node with no edges.
    assert int(graph.edge_index.min()) > 0


def test_each_relation_can_be_switched_off(plot_csvs):
    dataset = sparse_dataset(plot_csvs)

    taxonomy = rc.build_species_graph(
        dataset, hetero_config(use_cooccurrence_edges=False))
    shared = rc.build_species_graph(
        dataset, hetero_config(use_taxonomic_edges=False))
    both = rc.build_species_graph(dataset, hetero_config())

    assert set(int(t) for t in taxonomy.edge_type.tolist()) == {0, 1}
    assert set(int(t) for t in shared.edge_type.tolist()) == {2}
    assert both.n_edges == taxonomy.n_edges + shared.n_edges

    with pytest.raises(Exception):
        rc.build_species_graph(dataset, hetero_config(
            use_taxonomic_edges=False, use_cooccurrence_edges=False))


def test_the_cooccurrence_cut_is_the_configured_one(plot_csvs):
    dataset = sparse_dataset(plot_csvs)
    base = dict(use_taxonomic_edges=False)

    wide = rc.build_species_graph(
        dataset, hetero_config(**base, k_cooccurrence=8, cooccurrence_threshold=0.0))
    narrow = rc.build_species_graph(
        dataset, hetero_config(**base, k_cooccurrence=1, cooccurrence_threshold=0.0))
    assert narrow.n_edges < wide.n_edges

    # No pair shares every plot, so a threshold of 1 leaves nothing.
    none = rc.build_species_graph(
        dataset, hetero_config(**base, cooccurrence_threshold=1.0))
    assert none.n_edges == 0


def test_a_relation_the_dataset_cannot_supply_is_refused(plot_csvs, hash_dataset):
    # Hash encoding has no per-plot species vector to count co-occurrence in.
    with pytest.raises(Exception):
        rc.build_species_graph(hash_dataset,
                               hetero_config(use_taxonomic_edges=False))
    # Its taxonomy is still there.
    assert rc.build_species_graph(
        hash_dataset, hetero_config(use_cooccurrence_edges=False)).n_edges > 0

    # A dataset loaded without taxonomy cannot supply the taxonomic relation.
    plain = sparse_dataset(plot_csvs, use_taxonomy=False)
    with pytest.raises(Exception):
        rc.build_species_graph(plain, hetero_config(use_cooccurrence_edges=False))


def test_the_dataset_reports_the_taxonomy_of_its_vocabulary(plot_csvs):
    dataset = sparse_dataset(plot_csvs)
    genus = dataset.species_genus_ids
    family = dataset.species_family_ids

    assert genus is not None and family is not None
    assert genus.shape == (dataset.schema.n_species_vocab,)
    assert family.shape == (dataset.schema.n_species_vocab,)
    # <UNK> belongs to nothing; every real species does.
    assert int(genus[0]) == 0
    assert int(genus[1:].min()) > 0
    # The corpus assigns four genera and two families.
    assert len(set(int(g) for g in genus[1:].tolist())) == 4
    assert len(set(int(f) for f in family[1:].tolist())) == 2

    plain = sparse_dataset(plot_csvs, use_taxonomy=False)
    assert plain.species_genus_ids is None


def test_a_heterogeneous_gnn_trains_on_the_graph_the_trainer_builds(plot_csvs,
                                                                   tmp_path):
    dataset = sparse_dataset(plot_csvs)

    model_config = rc.ModelConfig()
    model_config.species_encoding = rc.SpeciesEncodingMode.Sparse
    model_config.uses_explicit_vector = True
    model_config.encoder_architecture = rc.EncoderArchitecture.HeterogeneousGNN
    model_config.hidden_dims = [12, 8]
    model_config.dropout = 0.0
    model_config.heterogeneous_gnn = hetero_config()

    model = rc.ResolveModel(dataset.schema, model_config)
    assert model.requires_species_graph
    assert not model.has_species_graph

    trainer = rc.Trainer(model, make_train_config(max_epochs=2, batch_size=32))
    trainer.prepare_data(dataset, test_size=0.25, seed=0)

    # The trainer is where the graph comes from: it has both the model and the
    # data.
    assert model.has_species_graph
    assert model.species_graph_edge_index.shape[0] == 2

    result = trainer.fit()
    assert result.test_loss_history
    assert result.test_loss_history[-1] == result.test_loss_history[-1]  # not NaN

    # The graph is part of the trained model, so scoring a checkpoint reads the
    # same one back.
    path = str(tmp_path / "hetero.pt")
    trainer.save(path)
    predictor = rc.Predictor.load(path, device="cpu")
    predictions = predictor.predict_dataset(dataset)
    assert "y" in predictions.predictions


def test_only_a_heterogeneous_gnn_accepts_a_species_graph(plot_csvs):
    dataset = sparse_dataset(plot_csvs)
    config = rc.ModelConfig()
    config.species_encoding = rc.SpeciesEncodingMode.Sparse
    config.uses_explicit_vector = True
    config.hidden_dims = [8]

    model = rc.ResolveModel(dataset.schema, config)
    assert not model.requires_species_graph
    graph = rc.build_species_graph(dataset, hetero_config())
    with pytest.raises(Exception):
        model.set_species_graph(graph.edge_index, graph.edge_type)
