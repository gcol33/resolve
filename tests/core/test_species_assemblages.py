"""Duplicate merging, representation and aggregation through the bindings.

The engine-side cases are in ``src/core/tests/test_species_encoders.cpp``.
"""

import pytest
import resolve_core as rc


def record(plot, species, abundance, genus="", family=""):
    r = rc.SpeciesRecord()
    r.plot_id, r.species_id, r.abundance = plot, species, abundance
    r.genus, r.family = genus, family
    return r


LAYERED = [
    record("P1", "known", 2.0),
    record("P1", "novel", 1.0),
    record("P1", "novel", 1.0),  # a second layer of the same species
]


def test_vocab_ranks_by_plot_count_and_round_trips():
    records = [
        record("P1", "zeta", 1.0),
        record("P2", "zeta", 1.0),
        record("P1", "alpha", 1.0),
        record("P1", "alpha", 1.0),
    ]
    vocab = rc.SpeciesVocab.from_records(records)
    assert vocab.names == ["<UNK>", "zeta", "alpha"]
    adopted = rc.SpeciesVocab.from_ordered(vocab.names)
    assert adopted.encode("alpha") == vocab.encode("alpha")


def test_unknown_statistics_count_species_once():
    vocab = rc.SpeciesVocab.from_ordered(["<UNK>", "known"])
    stats = rc.compute_unknown_species_stats(LAYERED, ["P1"], vocab)
    assert stats.count[0].item() == 1.0
    assert stats.fraction[0].item() == pytest.approx(0.5)
    presence = rc.compute_unknown_species_stats(
        LAYERED, ["P1"], vocab, rc.RepresentationMode.PresenceAbsence
    )
    assert presence.fraction[0].item() == pytest.approx(0.5)


def test_rank_pool_merges_layers_and_reads_presence_absence():
    encoder = rc.RankPoolEncoder(rc.PoolWeighting.Abundance)
    encoder.fit(LAYERED)
    merged = encoder.transform(LAYERED, ["P1"], has_abundance_column=True)
    assert merged.mask[0].sum().item() == 2
    assert merged.weights[0].tolist() == [2.0, 2.0]
    assert merged.has_cover[0].item() == 1.0

    presence = encoder.transform(
        LAYERED,
        ["P1"],
        has_abundance_column=True,
        representation=rc.RepresentationMode.PresenceAbsence,
    )
    assert presence.weights[0].tolist() == [1.0, 1.0]
    assert presence.has_cover[0].item() == 0.0

    scorer = rc.RankPoolEncoder(rc.PoolWeighting.Abundance)
    scorer.set_vocab(encoder.vocab)
    assert scorer.is_fitted()
    assert scorer.species_vocab.names == encoder.species_vocab.names


def test_embedding_taxonomy_slots_follow_the_aggregation():
    records = [
        record("P", "s1", 10.0, "gen_big"),
        record("P", "s2", 1.0, "gen_many"),
        record("P", "s3", 1.0, "gen_many"),
        record("P", "s4", 1.0, "gen_many"),
    ]
    for aggregation, leader in (
        (rc.AggregationMode.Abundance, "gen_big"),
        (rc.AggregationMode.Count, "gen_many"),
    ):
        encoder = rc.EmbeddingEncoder(4, 1, rc.SelectionMode.Top, aggregation)
        encoder.fit(records)
        encoded = encoder.transform(records, ["P"])
        assert encoded.genus_ids[0, 0].item() == encoder.taxonomy_vocab.encode_genus(
            leader
        )


def test_pooled_encodings_record_count_aggregation(tmp_path):
    header = tmp_path / "h.csv"
    species = tmp_path / "s.csv"
    header.write_text("plot_id,y\nP1,1\nP2,2\n")
    species.write_text("plot_id,sp,cover\nP1,a,1\nP2,b,2\n")
    roles = rc.RoleMapping()
    roles.plot_id, roles.species_id, roles.abundance = "plot_id", "sp", "cover"
    cfg = rc.DatasetConfig()
    cfg.aggregation = rc.AggregationMode.Count
    for encoding in (
        rc.SpeciesEncodingMode.RankPool,
        rc.SpeciesEncodingMode.Transformer,
    ):
        cfg.species_encoding = encoding
        dataset = rc.ResolveDataset.from_csv(
            str(header), str(species), roles, [rc.TargetSpec.regression("y")], cfg
        )
        assert dataset.schema.dataset_config.aggregation == rc.AggregationMode.Count
