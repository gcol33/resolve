"""Behaviour of ``resolve_core.fuzzy.FuzzyIndex`` through the bindings.

Coverage kind: behaviour. The index is a Damerau-Levenshtein top-N matcher
over a string vocabulary; these tests pin what a caller relies on: the edit
distances it reports, the (distance, id) ordering and ``top_n`` cap, the
distance cut-off, case folding, the bucket hint with its fall-back to the
whole vocabulary, and ``query_batch`` agreeing with ``query`` needle by
needle. The trie and automaton internals are covered by
``src/core/tests/test_fuzzy.cpp``.
"""

from __future__ import annotations

import pytest

import resolve_core as rc

ENTRIES = [
    "Quercus robur",
    "Quercus rubra",
    "Quercus petraea",
    "Fagus sylvatica",
    "Pinus sylvestris",
    "Pinus nigra",
    "Abies alba",
]


def genus(name: str) -> str:
    return name.split(" ", 1)[0]


def as_tuples(matches) -> list[tuple[str, int, int]]:
    return [(m.entry, m.id, m.distance) for m in matches]


@pytest.fixture(scope="module")
def index():
    return rc.fuzzy.FuzzyIndex.build(ENTRIES, max_edit_distance=3)


def test_build_reports_size_and_distance_cap(index):
    assert len(index) == len(ENTRIES)
    assert index.max_supported_distance == 3
    # Without a bucket function there is only the global trie.
    assert index.bucket_count == 1


def test_exact_match_is_distance_zero_with_its_entry_id(index):
    matches = index.query("Fagus sylvatica")
    assert as_tuples(matches)[0] == ("Fagus sylvatica", 3, 0)
    assert isinstance(matches[0], rc.fuzzy.Match)


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("Fagus silvatica", 1),  # substitution
        ("Fagus sylvatca", 1),  # deletion
        ("Fagus sylvatiica", 1),  # insertion
        ("Fagus sylvaitca", 1),  # adjacent transposition, one Damerau edit
    ],
)
def test_single_edits_are_distance_one(index, needle, expected):
    best = index.query(needle, max_edit_distance=2, top_n=1)
    assert as_tuples(best) == [("Fagus sylvatica", 3, expected)]


def test_transposition_costs_two_without_damerau():
    plain = rc.fuzzy.FuzzyIndex.build(ENTRIES, max_edit_distance=3, damerau=False)
    best = plain.query("Fagus sylvaitca", max_edit_distance=2, top_n=1)
    assert as_tuples(best) == [("Fagus sylvatica", 3, 2)]


def test_results_are_ordered_by_distance_then_id(index):
    matches = index.query("Quercus rob", max_edit_distance=3, top_n=10)
    keys = [(m.distance, m.id) for m in matches]
    assert keys == sorted(keys)
    # "Quercus robur" (2 insertions) beats "Quercus rubra" (3 edits).
    assert as_tuples(matches)[:2] == [("Quercus robur", 0, 2), ("Quercus rubra", 1, 3)]


def test_top_n_caps_the_result_list(index):
    everything = index.query("Quercus rob", max_edit_distance=3, top_n=10)
    assert len(everything) == 2
    assert as_tuples(
        index.query("Quercus rob", max_edit_distance=3, top_n=1)
    ) == as_tuples(everything[:1])
    assert index.query("Quercus rob", max_edit_distance=3, top_n=0) == []


def test_no_entry_within_the_distance_returns_nothing(index):
    assert index.query("Betula pendula", max_edit_distance=2) == []
    # "Quercus rubra" is two edits from the needle, so a cut-off of one keeps
    # only "Quercus robur".
    assert as_tuples(index.query("Quercus robru", max_edit_distance=1, top_n=10)) == [
        ("Quercus robur", 0, 1)
    ]
    assert len(index.query("Quercus robru", max_edit_distance=2, top_n=10)) == 2


def test_negative_distance_is_rejected(index):
    with pytest.raises(ValueError):
        index.query("Abies alba", max_edit_distance=-1)


def test_case_is_folded_by_default_and_the_entry_keeps_its_casing(index):
    best = index.query("QUERCUS PETRAEA", top_n=1)
    assert as_tuples(best) == [("Quercus petraea", 2, 0)]


def test_case_sensitive_index_counts_case_changes_as_edits():
    sensitive = rc.fuzzy.FuzzyIndex.build(
        ENTRIES, max_edit_distance=3, case_insensitive=False
    )
    assert as_tuples(sensitive.query("abies alba", top_n=1)) == [("Abies alba", 6, 1)]
    assert sensitive.query("ABIES ALBA", max_edit_distance=3) == []


def test_utf8_entries_match_by_codepoint():
    names = ["Rosa canina", "Rosa rubiginósa", "Sorbus aria"]
    utf8 = rc.fuzzy.FuzzyIndex.build(names, max_edit_distance=2)
    # One accented codepoint is one edit, not two bytes' worth.
    assert as_tuples(utf8.query("Rosa rubiginosa", top_n=1)) == [
        ("Rosa rubiginósa", 1, 1)
    ]


@pytest.fixture(scope="module")
def bucketed():
    return rc.fuzzy.FuzzyIndex.build(ENTRIES, max_edit_distance=5, bucket_fn=genus)


def test_bucket_function_partitions_the_vocabulary(bucketed):
    # Quercus, Fagus, Pinus, Abies, plus the global trie.
    assert bucketed.bucket_count == 5


def test_bucket_hint_restricts_matches_to_its_bucket(bucketed):
    needle = "Pinus sylvatica"
    unhinted = as_tuples(bucketed.query(needle, max_edit_distance=5, top_n=5))
    # Without a hint the closest entry is in the Fagus bucket (three edits on
    # the genus), ahead of the Pinus entry five edits away.
    assert unhinted == [("Fagus sylvatica", 3, 3), ("Pinus sylvestris", 4, 5)]
    # The hint returns the bucket-local matches without the closer entry from
    # another bucket.
    hinted = bucketed.query(needle, max_edit_distance=5, top_n=5, bucket_hint="Pinus")
    assert as_tuples(hinted) == [("Pinus sylvestris", 4, 5)]


def test_bucket_hint_is_case_folded(bucketed):
    assert as_tuples(bucketed.query("Pinus nigra", bucket_hint="PINUS", top_n=1)) == [
        ("Pinus nigra", 5, 0)
    ]


def test_bucket_hint_falls_back_to_the_whole_vocabulary(bucketed):
    needle = "Abies albba"
    expected = as_tuples(bucketed.query(needle))
    # An unknown bucket, and a bucket with nothing within the distance, both
    # search the global trie.
    assert as_tuples(bucketed.query(needle, bucket_hint="Larix")) == expected
    assert as_tuples(bucketed.query(needle, bucket_hint="Quercus")) == expected
    assert expected[0] == ("Abies alba", 6, 1)


NEEDLES = [
    "Quercus robru",
    "fagus sylvatica",
    "Pinus nigar",
    "Abies",
    "Betula pendula",
    "Quercus petrea",
]


def test_query_batch_matches_query_needle_by_needle(index):
    batch = index.query_batch(NEEDLES, max_edit_distance=2, top_n=3)
    assert len(batch) == len(NEEDLES)
    for needle, matches in zip(NEEDLES, batch):
        assert as_tuples(matches) == as_tuples(
            index.query(needle, max_edit_distance=2, top_n=3)
        )


def test_query_batch_with_one_bucket_hint(bucketed):
    batch = bucketed.query_batch(
        NEEDLES, max_edit_distance=3, top_n=2, bucket_hint="Quercus"
    )
    for needle, matches in zip(NEEDLES, batch):
        single = bucketed.query(
            needle, max_edit_distance=3, top_n=2, bucket_hint="Quercus"
        )
        assert as_tuples(matches) == as_tuples(single)


def test_query_batch_derives_a_hint_per_needle(bucketed):
    batch = bucketed.query_batch(NEEDLES, max_edit_distance=3, top_n=2, bucket_fn=genus)
    for needle, matches in zip(NEEDLES, batch):
        single = bucketed.query(
            needle, max_edit_distance=3, top_n=2, bucket_hint=genus(needle)
        )
        assert as_tuples(matches) == as_tuples(single)


def test_query_batch_on_an_empty_list(index):
    assert index.query_batch([]) == []


class BucketFailure(Exception):
    pass


def _failing_bucket(name: str) -> str:
    if name.startswith("Pinus"):
        raise BucketFailure(name)
    return genus(name)


def test_an_exception_in_the_bucket_function_propagates_from_build():
    with pytest.raises(BucketFailure, match="Pinus"):
        rc.fuzzy.FuzzyIndex.build(ENTRIES, bucket_fn=_failing_bucket)


def test_an_exception_in_the_bucket_function_propagates_from_query_batch(bucketed):
    # The callback runs inside the engine's parallel loop; its exception is
    # raised once the batch returns, and the index stays usable.
    with pytest.raises(BucketFailure):
        bucketed.query_batch(NEEDLES, bucket_fn=_failing_bucket)
    assert len(bucketed.query_batch(NEEDLES, bucket_fn=genus)) == len(NEEDLES)


def test_the_index_keeps_no_reference_to_the_bucket_function():
    import gc
    import weakref

    class Bucket:
        def __call__(self, name):
            return genus(name)

    bucket = Bucket()
    alive = weakref.ref(bucket)
    built = rc.fuzzy.FuzzyIndex.build(ENTRIES, bucket_fn=bucket)
    del bucket
    gc.collect()
    assert alive() is None
    assert built.bucket_count == 5
