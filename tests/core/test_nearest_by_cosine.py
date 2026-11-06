"""The engine's exact cosine nearest-neighbour search, ``resolve_core.nearest_by_cosine``.

Coverage kind: behavioural. Every result is checked against a brute-force
numpy search (all cosine similarities, sorted): the plain search over several
blocks of queries and reference rows, the leave-one-out search with duplicate
rows, presence rows (the Ochiai index), queries kept on the CPU while the
reference sits on the GPU, and the argument checks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import resolve_core as rc


def _cosine(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    r = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    return q.astype(np.float64) @ r.astype(np.float64).T


def _check_against_brute_force(indices, similarities, all_similarities, k):
    """Every returned row carries the similarity reported for it, and those are
    the k largest of its query, most similar first. Tied rows may come in any
    order, so the rows themselves are compared only through their values."""
    indices = indices.numpy()
    similarities = similarities.numpy()
    carried = np.take_along_axis(all_similarities, indices, axis=1)
    np.testing.assert_allclose(similarities, carried, rtol=1e-5, atol=1e-5)
    expected = -np.sort(-all_similarities, axis=1)[:, :k]
    np.testing.assert_allclose(similarities, expected, rtol=1e-5, atol=1e-5)


def test_exact_over_several_blocks_of_queries_and_reference():
    rng = np.random.default_rng(1)
    # 4500 queries make two blocks of at most 4096 rows, each searched against
    # 9000 reference rows in three blocks.
    queries = rng.standard_normal((4500, 12)).astype(np.float32)
    reference = rng.standard_normal((9000, 12)).astype(np.float32)
    k = 9

    indices, similarities = rc.nearest_by_cosine(
        torch.from_numpy(queries), torch.from_numpy(reference), k
    )
    assert indices.dtype == torch.int64
    assert similarities.dtype == torch.float32
    assert indices.shape == (4500, k)
    all_similarities = _cosine(queries, reference)
    _check_against_brute_force(indices, similarities, all_similarities, k)
    # Random continuous rows do not tie, so the rows themselves match too.
    expected = np.argsort(-all_similarities, axis=1, kind="stable")[:, :k]
    assert (indices.numpy() == expected).mean() > 0.999


def test_exclude_self_never_returns_the_query_but_keeps_its_duplicates():
    rng = np.random.default_rng(2)
    plots = (rng.random((5000, 40)) < 0.2).astype(np.float32)
    plots[plots.sum(axis=1) == 0, 0] = 1.0
    plots[1] = plots[0]
    plots[4600] = plots[7]
    k = 6

    rows = torch.from_numpy(plots)
    indices, similarities = rc.nearest_by_cosine(rows, rows, k, exclude_self=True)
    assert not (indices.numpy() == np.arange(5000)[:, None]).any()
    all_similarities = _cosine(plots, plots)
    np.fill_diagonal(all_similarities, -np.inf)
    _check_against_brute_force(indices, similarities, all_similarities, k)
    # Presence rows: the cosine of two species lists is their Ochiai index.
    shared = plots[0] @ plots[2]
    ochiai = shared / np.sqrt(plots[0].sum() * plots[2].sum())
    _, pair = rc.nearest_by_cosine(rows[0:1], rows[2:3], 1)
    assert pair.item() == pytest.approx(ochiai, abs=1e-6)
    # The identical plot is a neighbour like any other, at similarity 1.
    assert 1 in indices[0].tolist() and 0 in indices[1].tolist()
    assert 4600 in indices[7].tolist() and 7 in indices[4600].tolist()
    assert similarities[0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_k_is_clamped_and_float64_input_accepted():
    rows = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    indices, similarities = rc.nearest_by_cosine(rows, rows, 10)
    assert indices.shape == (3, 3)
    assert indices[:, 0].tolist() == [0, 1, 2]
    indices, _ = rc.nearest_by_cosine(rows, rows, 10, exclude_self=True)
    assert indices.shape == (3, 2)


def test_arguments_are_checked():
    rows = torch.ones(3, 2)
    with pytest.raises(ValueError, match="columns"):
        rc.nearest_by_cosine(rows, torch.ones(3, 4), 1)
    with pytest.raises(ValueError, match="at least 1"):
        rc.nearest_by_cosine(rows, rows, 0)
    with pytest.raises(ValueError, match="exclude_self"):
        rc.nearest_by_cosine(rows, torch.ones(4, 2), 1, exclude_self=True)
    with pytest.raises(ValueError, match="tensor"):
        rc.nearest_by_cosine(np.ones((3, 2), dtype=np.float32), rows, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_reference_on_the_gpu_with_queries_on_the_cpu():
    rng = np.random.default_rng(4)
    queries = rng.standard_normal((700, 32)).astype(np.float32)
    reference = rng.standard_normal((5000, 32)).astype(np.float32)
    indices, similarities = rc.nearest_by_cosine(
        torch.from_numpy(queries), torch.from_numpy(reference).cuda(), 5
    )
    assert indices.device.type == "cpu"
    _check_against_brute_force(indices, similarities, _cosine(queries, reference), 5)

    rows = torch.from_numpy(reference).cuda()
    indices, _ = rc.nearest_by_cosine(rows, rows, 5, exclude_self=True)
    assert not (indices.numpy() == np.arange(5000)[:, None]).any()
