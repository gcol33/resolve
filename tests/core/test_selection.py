"""Species selection and the per-plot species budget through the bindings.

Coverage kind: structure / plumbing. ``DatasetConfig.selection`` used to be
applied only while the hashed representation was built, so a rank_pool /
transformer / sparse dataset reported the selection it was given and encoded
every species anyway, and embed hardcoded ``Top`` (issue #113). The numerical
contract is pinned by ``src/core/tests/test_species_selection.cpp``; what is
asserted here is that the Python surface -- the new ``species_budget`` knob,
the schema field, and ``effective_selection`` -- reaches it intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import resolve_core as rc

from conftest import make_roles, make_targets, write_csv


N_SPECIES = 8


@pytest.fixture
def rotated_csvs(tmp_path: Path) -> tuple[str, str]:
    """Plots that all record the same species with rotating covers.

    Plot ``i`` gives species ``j`` the cover ``((j + i) % N_SPECIES) + 1``, so
    every plot holds the covers 1..N with no ties and which species is most
    abundant moves with the plot. A selection that quietly falls back to "the
    first k records in CSV order" therefore disagrees with the ranking on every
    plot but the first.
    """
    n_plots = 16
    header_rows = []
    species_rows = []
    for i in range(n_plots):
        header_rows.append([f"P{i}", f"{1.0 + 0.5 * i:.3f}", i % 3])
        for j in range(N_SPECIES):
            cover = ((j + i) % N_SPECIES) + 1
            species_rows.append([f"P{i}", f"sp{j}", cover, f"g{j % 3}", f"f{j % 2}"])

    header = write_csv(tmp_path / "sel_header.csv", ["plot_id", "y", "hab"], header_rows)
    species = write_csv(
        tmp_path / "sel_species.csv",
        ["plot_id", "sp", "cover", "genus", "family"],
        species_rows,
    )
    return str(header), str(species)


def _roles():
    return make_roles(coordinates=False, covariates=False)


def _config(encoding, selection, budget):
    cfg = rc.DatasetConfig()
    cfg.species_encoding = encoding
    cfg.selection = selection
    cfg.species_budget = budget
    cfg.use_taxonomy = True
    return cfg


def _build(csvs, cfg):
    header, species = csvs
    return rc.ResolveDataset.from_csv(header, species, _roles(), make_targets(), cfg)


def _encoded_names(ds, row: int) -> set[str]:
    """The species a plot's row actually encodes, whichever tensor holds it."""
    vocab = ds.species_vocab

    def name_of(code: int) -> str | None:
        if code <= 0 or code >= len(vocab):
            return None
        return vocab[code]

    # An accessor whose tensor the encoding did not fill comes back as None.
    def filled(tensor):
        return tensor is not None and tensor.numel() > 0

    if filled(ds.pool_mask):
        ids = ds.species_ids[row]
        mask = ds.pool_mask[row]
        return {
            n
            for c in range(ids.size(0))
            if bool(mask[c].item()) and (n := name_of(int(ids[c].item()))) is not None
        }
    if filled(ds.species_vector):
        row_t = ds.species_vector[row]
        return {
            n
            for c in range(row_t.size(0))
            if float(row_t[c].item()) != 0.0 and (n := name_of(c)) is not None
        }
    ids = ds.species_ids[row]
    return {
        n
        for c in range(ids.size(0))
        if (n := name_of(int(ids[c].item()))) is not None
    }


def _expected_end(plot_index: int, k: int, *, top: bool) -> set[str]:
    ranked = sorted(
        range(N_SPECIES),
        key=lambda j: ((j + plot_index) % N_SPECIES) + 1,
        reverse=top,
    )
    return {f"sp{j}" for j in ranked[:k]}


# ---------------------------------------------------------------------------
# The knob exists and round-trips
# ---------------------------------------------------------------------------

def test_species_budget_defaults_to_no_budget():
    assert rc.DatasetConfig().species_budget == 0


def test_effective_selection_reports_what_a_load_applies():
    for encoding in (rc.SpeciesEncodingMode.Hash, rc.SpeciesEncodingMode.Embed):
        cfg = _config(encoding, rc.SelectionMode.Bottom, 0)
        assert rc.effective_selection(cfg) == rc.SelectionMode.Bottom

    for encoding in (
        rc.SpeciesEncodingMode.RankPool,
        rc.SpeciesEncodingMode.Transformer,
        rc.SpeciesEncodingMode.Sparse,
    ):
        cfg = _config(encoding, rc.SelectionMode.Bottom, 0)
        assert rc.effective_selection(cfg) == rc.SelectionMode.All
        cfg.species_budget = 3
        assert rc.effective_selection(cfg) == rc.SelectionMode.Bottom


# ---------------------------------------------------------------------------
# The pooled and sparse encodings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "encoding",
    [
        rc.SpeciesEncodingMode.RankPool,
        rc.SpeciesEncodingMode.Transformer,
        rc.SpeciesEncodingMode.Sparse,
    ],
)
def test_no_budget_encodes_every_species(rotated_csvs, encoding):
    for selection in (rc.SelectionMode.Top, rc.SelectionMode.Bottom, rc.SelectionMode.All):
        ds = _build(rotated_csvs, _config(encoding, selection, 0))
        for row in range(ds.n_plots):
            assert len(_encoded_names(ds, row)) == N_SPECIES
        # ... and the schema says so, rather than claiming a selection the run
        # did not make.
        assert ds.schema.selection == rc.SelectionMode.All
        assert ds.schema.species_budget == 0


@pytest.mark.parametrize(
    "encoding",
    [
        rc.SpeciesEncodingMode.RankPool,
        rc.SpeciesEncodingMode.Transformer,
        rc.SpeciesEncodingMode.Sparse,
    ],
)
@pytest.mark.parametrize(
    "selection,top",
    [(rc.SelectionMode.Top, True), (rc.SelectionMode.Bottom, False)],
)
def test_budget_keeps_the_ranked_end(rotated_csvs, encoding, selection, top):
    budget = 3
    ds = _build(rotated_csvs, _config(encoding, selection, budget))

    assert ds.schema.selection == selection
    assert ds.schema.species_budget == budget
    for row in range(ds.n_plots):
        assert _encoded_names(ds, row) == _expected_end(row, budget, top=top)


def test_the_two_arms_share_one_vocabulary(rotated_csvs):
    """An ablation's arms must be comparable: same codes, different assemblage."""
    top = _build(rotated_csvs, _config(rc.SpeciesEncodingMode.RankPool, rc.SelectionMode.Top, 3))
    bottom = _build(
        rotated_csvs, _config(rc.SpeciesEncodingMode.RankPool, rc.SelectionMode.Bottom, 3)
    )

    assert list(top.species_vocab) == list(bottom.species_vocab)
    assert any(
        _encoded_names(top, row) != _encoded_names(bottom, row)
        for row in range(top.n_plots)
    )


def test_budget_narrows_the_padded_width(rotated_csvs):
    narrow = _build(
        rotated_csvs, _config(rc.SpeciesEncodingMode.RankPool, rc.SelectionMode.Top, 2)
    )
    full = _build(
        rotated_csvs, _config(rc.SpeciesEncodingMode.RankPool, rc.SelectionMode.Top, 0)
    )
    assert narrow.species_ids.size(1) == 2
    assert full.species_ids.size(1) == N_SPECIES


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def test_embed_honours_bottom(rotated_csvs):
    cfg = _config(rc.SpeciesEncodingMode.Embed, rc.SelectionMode.Bottom, 0)
    cfg.top_k_species = 3
    ds = _build(rotated_csvs, cfg)
    for row in range(ds.n_plots):
        assert _encoded_names(ds, row) == _expected_end(row, 3, top=False)


def test_embed_top_bottom_fills_the_same_slot_count(rotated_csvs):
    cfg = _config(rc.SpeciesEncodingMode.Embed, rc.SelectionMode.TopBottom, 0)
    cfg.top_k_species = 4
    ds = _build(rotated_csvs, cfg)

    # The width is the slot count, not twice it: the model's embed encoder is
    # sized from top_k_species and has to keep matching.
    assert ds.species_ids.size(1) == 4
    for row in range(ds.n_plots):
        kept = _encoded_names(ds, row)
        assert kept >= _expected_end(row, 2, top=True)
        assert kept >= _expected_end(row, 2, top=False)


def test_embed_rejects_all(rotated_csvs):
    cfg = _config(rc.SpeciesEncodingMode.Embed, rc.SelectionMode.All, 0)
    with pytest.raises(Exception):
        _build(rotated_csvs, cfg)


# ---------------------------------------------------------------------------
# Hash is unchanged
# ---------------------------------------------------------------------------

def test_hash_ignores_the_budget(rotated_csvs):
    cfg = _config(rc.SpeciesEncodingMode.Hash, rc.SelectionMode.Top, 0)
    cfg.top_k = 2
    plain = _build(rotated_csvs, cfg)

    cfg.species_budget = 5  # not hash's knob
    with_budget = _build(rotated_csvs, cfg)

    import torch

    assert torch.equal(plain.hash_embedding, with_budget.hash_embedding)
    assert plain.schema.selection == rc.SelectionMode.Top


# ---------------------------------------------------------------------------
# Checkpoint round trip
# ---------------------------------------------------------------------------

def test_species_budget_survives_the_checkpoint(rotated_csvs, tmp_path):
    from conftest import make_model_config, make_train_config

    cfg = _config(rc.SpeciesEncodingMode.RankPool, rc.SelectionMode.Bottom, 3)
    ds = _build(rotated_csvs, cfg)

    model_config = make_model_config(rc.SpeciesEncodingMode.RankPool)
    model = rc.ResolveModel(ds.schema, model_config)
    trainer = rc.Trainer(model, make_train_config(max_epochs=1))
    trainer.prepare_data(ds)
    trainer.fit()

    path = str(tmp_path / "budget.pt")
    trainer.save(path)

    predictor = rc.Predictor.load(path, device="cpu")
    assert predictor.schema.species_budget == 3
    assert predictor.schema.selection == rc.SelectionMode.Bottom

    rebuilt = predictor.dataset_config
    assert rebuilt.species_budget == 3
    assert rebuilt.selection == rc.SelectionMode.Bottom
