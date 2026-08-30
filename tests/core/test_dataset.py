"""``ResolveDataset`` construction through the nanobind bindings.

Coverage kind: structure / plumbing. These assert that roles, targets, the
schema, and every accessor tensor come back with the right shape, dtype, and
definedness, and that the four loader entry points agree with each other. The
numerical behaviour they sit on top of is pinned by the Catch2 suite
(``src/core/tests/test_dataset.cpp``, ``test_dataframe.cpp``); what is tested
here is that the Python surface reaches it intact.
"""

from __future__ import annotations

import pytest
import torch

import resolve_core as rc

from conftest import (
    make_dataset_config,
    make_plot_csvs,
    make_roles,
    make_targets,
)


# ---------------------------------------------------------------------------
# Schema and role mapping
# ---------------------------------------------------------------------------

def test_from_csv_populates_schema(hash_dataset, plot_csvs):
    schema = hash_dataset.schema

    assert hash_dataset.n_plots == plot_csvs.n_plots
    assert len(hash_dataset.plot_ids) == plot_csvs.n_plots
    assert hash_dataset.plot_ids[0] == "P0"
    # +1 for the reserved <UNK> slot at id 0.
    assert schema.n_species_vocab == plot_csvs.n_species + 1
    assert schema.has_taxonomy
    assert schema.has_coordinates
    assert schema.has_abundance
    assert schema.covariate_names == ["elev"]
    assert {t.name for t in schema.targets} == {"y", "hab"}


def test_role_mapping_predicates():
    roles = make_roles(categoricals=["soil"])
    assert roles.has_taxonomy()
    assert roles.has_coordinates()
    assert roles.has_abundance()
    assert roles.has_categoricals()

    bare = make_roles(taxonomy=False, coordinates=False, covariates=False)
    assert not bare.has_taxonomy()
    assert not bare.has_coordinates()
    assert not bare.has_categoricals()


def test_target_specs_land_in_schema(hash_dataset):
    by_name = {t.name: t for t in hash_dataset.schema.targets}
    assert by_name["y"].task == rc.TaskType.Regression
    assert by_name["hab"].task == rc.TaskType.Classification
    assert by_name["hab"].num_classes == 3
    assert by_name["hab"].class_names == ["0", "1", "2"]


def test_target_tensors_have_expected_shape_and_dtype(hash_dataset):
    targets = hash_dataset.targets
    assert set(targets) == {"y", "hab"}
    assert targets["y"].shape == (hash_dataset.n_plots,)
    assert targets["y"].dtype == torch.float32
    assert targets["hab"].shape == (hash_dataset.n_plots,)
    assert targets["hab"].dtype == torch.int64
    assert int(targets["hab"].min()) >= 0
    assert int(targets["hab"].max()) < 3


# ---------------------------------------------------------------------------
# Accessor tensors
# ---------------------------------------------------------------------------

def test_hash_mode_accessors(hash_dataset):
    n = hash_dataset.n_plots
    assert hash_dataset.coordinates.shape == (n, 2)
    assert hash_dataset.covariates.shape == (n, 1)
    assert hash_dataset.hash_embedding.shape == (n, 16)
    assert hash_dataset.genus_ids.shape[0] == n
    assert hash_dataset.genus_ids.dtype == torch.int64
    assert hash_dataset.family_ids.dtype == torch.int64
    assert hash_dataset.unknown_fraction.shape[0] == n
    assert not hash_dataset.has_pool_data()


def test_taxonomy_ids_stay_inside_their_vocab(hash_dataset):
    schema = hash_dataset.schema
    assert int(hash_dataset.genus_ids.max()) < schema.n_genera
    assert int(hash_dataset.family_ids.max()) < schema.n_families
    assert int(hash_dataset.genus_ids.min()) >= 0


def test_pool_tensors_populated_for_rank_pool(pool_dataset, plot_csvs):
    n = pool_dataset.n_plots
    cap = pool_dataset.schema.pool_species_cap

    assert pool_dataset.has_pool_data()
    assert cap > 0
    assert pool_dataset.species_ids.shape == (n, cap)
    assert pool_dataset.species_ids.dtype == torch.int64
    assert pool_dataset.pool_genus_ids.shape == (n, cap)
    assert pool_dataset.pool_family_ids.shape == (n, cap)
    assert pool_dataset.pool_weights.shape == (n, cap)
    assert pool_dataset.pool_mask.shape == (n, cap)
    assert pool_dataset.pool_has_cover.shape[0] == n
    # An abundance column is mapped, so cover is flagged present for every plot.
    assert float(pool_dataset.pool_has_cover.min()) == 1.0
    # Real weights, not a placeholder of ones or zeros.
    assert float(pool_dataset.pool_weights.max()) > 0.0


def test_pool_weighting_and_cap_recorded_on_schema(pool_dataset):
    assert pool_dataset.schema.pool_weighting == rc.PoolWeighting.Log1p.value
    assert pool_dataset.schema.pool_species_cap == 3


def test_pool_weighting_changes_the_weights(plot_csvs):
    log1p = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(
            rc.SpeciesEncodingMode.RankPool, pool_weighting=rc.PoolWeighting.Log1p
        ),
    )
    binary = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(
            rc.SpeciesEncodingMode.RankPool, pool_weighting=rc.PoolWeighting.Binary
        ),
    )
    assert not torch.equal(log1p.pool_weights, binary.pool_weights)
    assert torch.equal(log1p.species_ids, binary.species_ids)


def test_sparse_mode_builds_a_species_vector(plot_csvs):
    dataset = rc.ResolveDataset.from_csv(
        plot_csvs.header,
        plot_csvs.species,
        make_roles(),
        make_targets(),
        make_dataset_config(rc.SpeciesEncodingMode.Sparse),
    )
    vector = dataset.species_vector
    assert vector is not None
    assert vector.shape == (dataset.n_plots, dataset.schema.n_species_vocab)
    # Three species per plot, so each row is sparse but non-empty.
    assert float(vector.sum()) > 0.0


# ---------------------------------------------------------------------------
# Loader equivalence
# ---------------------------------------------------------------------------

def test_from_pandas_matches_from_csv(plot_csvs):
    pd = pytest.importorskip("pandas")

    roles, targets, config = make_roles(), make_targets(), make_dataset_config()
    from_disk = rc.ResolveDataset.from_csv(
        plot_csvs.header, plot_csvs.species, roles, targets, config
    )
    in_memory = rc.ResolveDataset.from_pandas(
        pd.read_csv(plot_csvs.header),
        pd.read_csv(plot_csvs.species),
        roles,
        targets,
        config,
    )

    assert in_memory.n_plots == from_disk.n_plots
    assert list(in_memory.plot_ids) == list(from_disk.plot_ids)
    for name in ("coordinates", "covariates", "hash_embedding", "genus_ids", "family_ids"):
        assert torch.equal(getattr(in_memory, name), getattr(from_disk, name)), name
    for key in from_disk.targets:
        assert torch.equal(in_memory.targets[key], from_disk.targets[key]), key


def test_from_csv_with_schema_reuses_the_training_vocabularies(tmp_path, plot_csvs):
    """A second CSV encoded against a first must reuse its id assignment.

    This is the cross-split path (leave-one-dataset-out, transfer): the
    checkpoint's embedding rows are indexed by the training vocab, so a rebuild
    that renumbered species or taxonomy would silently look up the wrong rows.
    """
    roles, targets, config = make_roles(), make_targets(), make_dataset_config()
    train = rc.ResolveDataset.from_csv(
        plot_csvs.header, plot_csvs.species, roles, targets, config
    )

    # A held-out CSV pair whose rows are in a different order and whose species
    # set is a subset; its own vocab would number things differently.
    subset = make_plot_csvs(tmp_path, n_plots=20, n_species=6, prefix="holdout")
    scored = rc.ResolveDataset.from_csv_with_schema(
        subset.header, subset.species, roles, targets, train, config
    )

    assert scored.schema.n_species_vocab == train.schema.n_species_vocab
    assert scored.schema.n_genera == train.schema.n_genera
    assert scored.schema.n_families == train.schema.n_families
    assert int(scored.genus_ids.max()) < train.schema.n_genera


def test_single_table_loader(tmp_path):
    """``from_species_csv`` reads one long frame carrying the target inline."""
    rows = []
    for i in range(30):
        for j in range(3):
            rows.append([f"P{i}", f"sp{(i + j) % 6}", f"{1.0 + j:.2f}", float(i)])
    path = tmp_path / "long.csv"
    path.write_text(
        "plot_id,sp,cover,y\n" + "\n".join(",".join(str(c) for c in r) for r in rows) + "\n",
        encoding="utf-8",
    )

    roles = rc.RoleMapping()
    roles.plot_id = "plot_id"
    roles.species_id = "sp"
    roles.abundance = "cover"
    dataset = rc.ResolveDataset.from_species_csv(
        str(path), roles, [rc.TargetSpec.regression("y")], make_dataset_config()
    )

    assert dataset.n_plots == 30
    assert dataset.targets["y"].shape == (30,)


# ---------------------------------------------------------------------------
# Loader rejections
# ---------------------------------------------------------------------------

def test_unknown_role_column_raises(plot_csvs):
    roles = make_roles()
    roles.genus = "no_such_column"
    with pytest.raises(Exception):
        rc.ResolveDataset.from_csv(
            plot_csvs.header,
            plot_csvs.species,
            roles,
            make_targets(),
            make_dataset_config(),
        )


def test_column_claimed_by_two_roles_raises(plot_csvs):
    roles = make_roles()
    roles.covariates = ["elev"]
    roles.categoricals = ["elev"]
    with pytest.raises(Exception, match="both"):
        rc.ResolveDataset.from_csv(
            plot_csvs.header,
            plot_csvs.species,
            roles,
            make_targets(),
            make_dataset_config(),
        )


def test_missing_file_raises(tmp_path, plot_csvs):
    with pytest.raises(Exception):
        rc.ResolveDataset.from_csv(
            str(tmp_path / "does_not_exist.csv"),
            plot_csvs.species,
            make_roles(),
            make_targets(),
            make_dataset_config(),
        )


# ---------------------------------------------------------------------------
# Clearing an optional role (issue #111)
# ---------------------------------------------------------------------------

def test_optional_roles_accept_none():
    """``roles.latitude = None`` is the unset path, matching the getter.

    ``def_rw`` on an ``std::optional`` member gave a getter that read back
    ``None`` and a setter that refused it, so downstream code had no way to
    clear a role and reached for an empty-string sentinel instead.
    """
    roles = make_roles()
    assert roles.has_coordinates()

    roles.latitude = None
    roles.longitude = None
    assert roles.latitude is None
    assert roles.longitude is None
    assert not roles.has_coordinates()

    roles.genus = None
    roles.family = None
    assert not roles.has_taxonomy()

    roles.abundance = None
    assert not roles.has_abundance()

    # Still a string attribute for a real column name.
    roles.latitude = "lat"
    assert roles.latitude == "lat"


def test_empty_string_clears_a_role(plot_csvs):
    """The empty string is unset too, so the sentinel downstream already uses works."""
    cleared = make_roles()
    cleared.latitude = ""
    cleared.longitude = ""
    assert not cleared.has_coordinates()

    unset = make_roles()
    unset.latitude = None
    unset.longitude = None

    cfg = make_dataset_config()
    from_cleared = rc.ResolveDataset.from_csv(
        plot_csvs.header, plot_csvs.species, cleared, make_targets(), cfg
    )
    from_unset = rc.ResolveDataset.from_csv(
        plot_csvs.header, plot_csvs.species, unset, make_targets(), cfg
    )

    assert not from_cleared.schema.has_coordinates
    assert from_cleared.n_plots == from_unset.n_plots
    assert torch.equal(from_cleared.hash_embedding, from_unset.hash_embedding)


def test_a_misspelled_role_column_still_throws(plot_csvs):
    """Only EMPTY means unset: a typo is still the loud configuration error."""
    roles = make_roles()
    roles.latitude = "lattitude"
    with pytest.raises(Exception):
        rc.ResolveDataset.from_csv(
            plot_csvs.header, plot_csvs.species, roles, make_targets(),
            make_dataset_config(),
        )
