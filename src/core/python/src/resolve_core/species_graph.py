"""
Species graph construction for Heterogeneous GNN.

Builds a species graph from vegetation plot data with three edge types:
  - Type 0: Co-occurrence edges (species frequently co-occur in plots)
  - Type 1: Same-genus edges (species share genus)
  - Type 2: Same-family edges (species share family)

Usage:
    from resolve_core.species_graph import build_species_graph

    edge_index, edge_type = build_species_graph(
        species_data,            # DataFrame with plot_id, species_id, genus, family
        species_vocab,           # dict: species_name -> integer ID
        k_cooccurrence=20,       # top-k co-occurring species per species
        cooccurrence_threshold=0.01,
        use_taxonomic_edges=True,
        use_cooccurrence_edges=True,
    )
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Optional

__all__ = ["build_species_graph", "build_cooccurrence_matrix"]


def build_cooccurrence_matrix(
    plot_ids: np.ndarray,
    species_ids: np.ndarray,
    n_species: int,
) -> np.ndarray:
    """Build species co-occurrence matrix from plot-species data.

    Parameters
    ----------
    plot_ids : np.ndarray
        Plot identifier for each record.
    species_ids : np.ndarray
        Species integer ID for each record (0-indexed).
    n_species : int
        Total number of species in vocabulary.

    Returns
    -------
    np.ndarray
        (n_species, n_species) co-occurrence count matrix (symmetric).
    """
    # Group species by plot
    plot_species: dict[int | str, list[int]] = defaultdict(list)
    for pid, sid in zip(plot_ids, species_ids):
        plot_species[pid].append(int(sid))

    # Count co-occurrences
    cooccurrence = np.zeros((n_species, n_species), dtype=np.float32)
    for species_list in plot_species.values():
        unique_species = list(set(species_list))
        for i, s1 in enumerate(unique_species):
            for s2 in unique_species[i + 1 :]:
                if s1 < n_species and s2 < n_species:
                    cooccurrence[s1, s2] += 1.0
                    cooccurrence[s2, s1] += 1.0

    return cooccurrence


def build_species_graph(
    plot_ids: np.ndarray,
    species_ids: np.ndarray,
    n_species: int,
    genus_ids: Optional[np.ndarray] = None,
    family_ids: Optional[np.ndarray] = None,
    k_cooccurrence: int = 20,
    cooccurrence_threshold: float = 0.01,
    use_taxonomic_edges: bool = True,
    use_cooccurrence_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build heterogeneous species graph.

    Parameters
    ----------
    plot_ids : np.ndarray
        Plot identifier for each record.
    species_ids : np.ndarray
        Species integer ID for each record (0-indexed).
    n_species : int
        Total number of species in vocabulary.
    genus_ids : np.ndarray, optional
        Genus integer ID for each record (matching species_ids order).
    family_ids : np.ndarray, optional
        Family integer ID for each record (matching species_ids order).
    k_cooccurrence : int
        Top-k co-occurring species per species for edge construction.
    cooccurrence_threshold : float
        Minimum co-occurrence frequency (fraction of max) for an edge.
    use_taxonomic_edges : bool
        Whether to add same-genus and same-family edges.
    use_cooccurrence_edges : bool
        Whether to add co-occurrence edges.

    Returns
    -------
    edge_index : np.ndarray
        (2, n_edges) int64 array of [source, target] node indices.
    edge_type : np.ndarray
        (n_edges,) int64 array of edge type IDs.
    """
    src_list: list[int] = []
    tgt_list: list[int] = []
    type_list: list[int] = []

    # --- Edge type 0: Co-occurrence ---
    if use_cooccurrence_edges:
        cooc = build_cooccurrence_matrix(plot_ids, species_ids, n_species)

        # Normalize by max co-occurrence
        max_cooc = cooc.max()
        if max_cooc > 0:
            cooc_norm = cooc / max_cooc

            # For each species, keep top-k co-occurring species above threshold
            for s in range(n_species):
                row = cooc_norm[s]
                # Zero out self
                row[s] = 0.0
                # Find candidates above threshold
                candidates = np.where(row >= cooccurrence_threshold)[0]
                if len(candidates) == 0:
                    continue
                # Sort by co-occurrence, take top-k
                scores = row[candidates]
                top_k_idx = np.argsort(scores)[-k_cooccurrence:]
                neighbors = candidates[top_k_idx]
                for t in neighbors:
                    src_list.append(s)
                    tgt_list.append(int(t))
                    type_list.append(0)

    # --- Edge type 1: Same-genus ---
    if use_taxonomic_edges and genus_ids is not None:
        # Build species -> genus mapping (majority vote per species)
        species_genus: dict[int, int] = {}
        for sid, gid in zip(species_ids, genus_ids):
            sid_int, gid_int = int(sid), int(gid)
            if sid_int not in species_genus:
                species_genus[sid_int] = gid_int

        # Group species by genus
        genus_species: dict[int, list[int]] = defaultdict(list)
        for sid, gid in species_genus.items():
            if gid > 0:  # Skip UNK (0)
                genus_species[gid].append(sid)

        # Add edges between all species in same genus
        for species_in_genus in genus_species.values():
            for i, s1 in enumerate(species_in_genus):
                for s2 in species_in_genus[i + 1 :]:
                    src_list.extend([s1, s2])
                    tgt_list.extend([s2, s1])
                    type_list.extend([1, 1])

    # --- Edge type 2: Same-family ---
    if use_taxonomic_edges and family_ids is not None:
        # Build species -> family mapping
        species_family: dict[int, int] = {}
        for sid, fid in zip(species_ids, family_ids):
            sid_int, fid_int = int(sid), int(fid)
            if sid_int not in species_family:
                species_family[sid_int] = fid_int

        # Group species by family
        family_species: dict[int, list[int]] = defaultdict(list)
        for sid, fid in species_family.items():
            if fid > 0:  # Skip UNK (0)
                family_species[fid].append(sid)

        # Add edges (but skip pairs already connected by genus)
        existing_pairs: set[tuple[int, int]] = set()
        for i in range(len(src_list)):
            if type_list[i] == 1:
                existing_pairs.add((src_list[i], tgt_list[i]))

        for species_in_family in family_species.values():
            for i, s1 in enumerate(species_in_family):
                for s2 in species_in_family[i + 1 :]:
                    if (s1, s2) not in existing_pairs:
                        src_list.extend([s1, s2])
                        tgt_list.extend([s2, s1])
                        type_list.extend([2, 2])

    # Convert to numpy arrays
    if len(src_list) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type = np.zeros((0,), dtype=np.int64)
    else:
        edge_index = np.stack([
            np.array(src_list, dtype=np.int64),
            np.array(tgt_list, dtype=np.int64),
        ])
        edge_type = np.array(type_list, dtype=np.int64)

    return edge_index, edge_type
