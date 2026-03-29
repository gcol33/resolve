"""Pretrained species embeddings: load and inject external representations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def load_embeddings_from_csv(
    path: str | Path,
    species_column: str = "species",
    dim: int | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Load species embeddings from a CSV file.

    Expected format: species_column, dim_0, dim_1, ..., dim_N

    Args:
        path: Path to CSV file.
        species_column: Column containing species names.
        dim: Expected embedding dimension (inferred if None).

    Returns:
        (embeddings_dict, embedding_dim) where embeddings_dict maps
        species names to numpy arrays.
    """
    import polars as pl
    df = pl.read_csv(path)

    species_names = df[species_column].to_list()
    embed_cols = [c for c in df.columns if c != species_column]
    if dim is not None and len(embed_cols) != dim:
        raise ValueError(f"Expected {dim} embedding dims, got {len(embed_cols)}")

    embeddings = {}
    for i, name in enumerate(species_names):
        vec = df[i, embed_cols].to_numpy().flatten().astype(np.float32)
        embeddings[name] = vec

    return embeddings, len(embed_cols)


def load_embeddings_from_numpy(
    path: str | Path,
    species_names: list[str],
) -> tuple[dict[str, np.ndarray], int]:
    """Load species embeddings from a .npy or .npz file.

    Args:
        path: Path to .npy (matrix) or .npz (with 'embeddings' key).
        species_names: List of species names matching rows.

    Returns:
        (embeddings_dict, embedding_dim)
    """
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        matrix = data["embeddings"]
    else:
        matrix = np.load(path)

    if matrix.shape[0] != len(species_names):
        raise ValueError(
            f"Matrix has {matrix.shape[0]} rows but {len(species_names)} species names"
        )

    embeddings = {name: matrix[i].astype(np.float32) for i, name in enumerate(species_names)}
    return embeddings, matrix.shape[1]


def inject_pretrained_embeddings(
    model: nn.Module,
    embeddings: dict[str, np.ndarray],
    vocab: dict[str, int],
    embedding_layer: str = "species_embedding",
    freeze: bool = False,
) -> int:
    """Inject pretrained embeddings into a model's embedding layer.

    Matches species names between the pretrained embeddings and the model's
    vocabulary. Unmatched species keep their randomly initialized embeddings.

    Args:
        model: The model (or encoder) containing the embedding layer.
        embeddings: Dict mapping species names to embedding vectors.
        vocab: Dict mapping species names to integer IDs (from encoder).
        embedding_layer: Name of the embedding attribute on the encoder.
        freeze: If True, freeze the embedding layer after injection.

    Returns:
        Number of species that were matched and injected.
    """
    # Find the embedding layer
    encoder = model.encoder if hasattr(model, 'encoder') else model
    if not hasattr(encoder, embedding_layer):
        raise AttributeError(f"Encoder has no attribute '{embedding_layer}'")

    emb_module = getattr(encoder, embedding_layer)
    if not isinstance(emb_module, nn.Embedding):
        raise TypeError(f"{embedding_layer} is {type(emb_module)}, expected nn.Embedding")

    weight = emb_module.weight.data
    emb_dim = weight.shape[1]
    matched = 0

    for species, idx in vocab.items():
        if species in embeddings:
            vec = embeddings[species]
            if len(vec) != emb_dim:
                # Project if dimensions don't match
                if len(vec) > emb_dim:
                    vec = vec[:emb_dim]  # truncate
                else:
                    vec = np.pad(vec, (0, emb_dim - len(vec)))  # zero-pad
            weight[idx] = torch.from_numpy(vec)
            matched += 1

    if freeze:
        emb_module.weight.requires_grad_(False)

    return matched
