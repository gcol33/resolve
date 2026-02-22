"""
TabR: Retrieval-Augmented Prediction for RESOLVE.

At inference, retrieves K nearest training plots in learned embedding space,
weights by similarity, and combines with model prediction. Wraps any trained
RESOLVE model without requiring C++ changes.

Usage:
    from resolve_core import Trainer, ResolveModel
    from resolve_core.tabr import TabRWrapper

    model, scalers = Trainer.load("model.pt", device="cuda")
    tabr = TabRWrapper(model, k_neighbors=96, alpha=0.3)
    tabr.build_index(train_continuous, train_genus_ids, train_family_ids,
                     train_species_ids, train_species_vector, train_targets)
    predictions = tabr.predict(test_continuous, test_genus_ids, test_family_ids,
                                test_species_ids, test_species_vector)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

__all__ = ["TabRWrapper"]


class TabRWrapper:
    """Retrieval-augmented prediction wrapper for any trained RESOLVE model.

    Uses faiss for efficient nearest-neighbor search in the learned latent space.
    Combines model predictions with similarity-weighted neighbor targets.

    Parameters
    ----------
    model : ResolveModel
        Trained RESOLVE model (must have get_latent method).
    k_neighbors : int
        Number of nearest neighbors to retrieve. Default 96.
    alpha : float
        Blending weight: final = (1-alpha)*model_pred + alpha*retrieval_pred.
        Default 0.3.
    temperature : float
        Temperature for softmax over similarity scores. Default 1.0.
    use_gpu_index : bool
        If True and faiss-gpu available, build GPU index. Default False.
    """

    def __init__(
        self,
        model,
        k_neighbors: int = 96,
        alpha: float = 0.3,
        temperature: float = 1.0,
        use_gpu_index: bool = False,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.model = model
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.temperature = temperature
        self.use_gpu_index = use_gpu_index

        self._index = None
        self._train_targets: dict[str, np.ndarray] = {}
        self._latent_dim: Optional[int] = None

    def build_index(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
        targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """Build faiss index from training data embeddings.

        Parameters
        ----------
        continuous : torch.Tensor
            (n_train, n_features) continuous features.
        genus_ids, family_ids, species_ids, species_vector : torch.Tensor, optional
            Auxiliary inputs passed to model.get_latent().
        targets : dict[str, torch.Tensor], optional
            Training targets per task. Required for retrieval-augmented prediction.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss is required for TabR. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            ) from e

        # Extract latent representations
        self.model.eval()
        with torch.no_grad():
            latent = self.model.get_latent(
                continuous,
                genus_ids if genus_ids is not None else torch.empty(0),
                family_ids if family_ids is not None else torch.empty(0),
                species_ids if species_ids is not None else torch.empty(0),
                species_vector if species_vector is not None else torch.empty(0),
            )

        latent_np = latent.cpu().numpy().astype(np.float32)
        self._latent_dim = latent_np.shape[1]

        # Normalize for cosine similarity
        faiss.normalize_L2(latent_np)

        # Build index
        self._index = faiss.IndexFlatIP(self._latent_dim)

        if self.use_gpu_index:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except (AttributeError, RuntimeError):
                pass  # Fall back to CPU index

        self._index.add(latent_np)

        # Store training targets
        self._train_targets = {}
        if targets is not None:
            for name, tensor in targets.items():
                self._train_targets[name] = tensor.cpu().numpy()

    def predict(
        self,
        continuous: torch.Tensor,
        genus_ids: Optional[torch.Tensor] = None,
        family_ids: Optional[torch.Tensor] = None,
        species_ids: Optional[torch.Tensor] = None,
        species_vector: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Predict with retrieval augmentation.

        Returns dict of target_name -> predictions, blending model output
        with similarity-weighted neighbor targets.
        """
        if self._index is None:
            raise RuntimeError("Must call build_index() before predict()")

        self.model.eval()
        with torch.no_grad():
            # Get model predictions
            model_preds = self.model.forward(
                continuous,
                genus_ids if genus_ids is not None else torch.empty(0),
                family_ids if family_ids is not None else torch.empty(0),
                species_ids if species_ids is not None else torch.empty(0),
                species_vector if species_vector is not None else torch.empty(0),
            )

            # Get latent representations for query
            latent = self.model.get_latent(
                continuous,
                genus_ids if genus_ids is not None else torch.empty(0),
                family_ids if family_ids is not None else torch.empty(0),
                species_ids if species_ids is not None else torch.empty(0),
                species_vector if species_vector is not None else torch.empty(0),
            )

        query_np = latent.cpu().numpy().astype(np.float32)

        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss is required for TabR prediction") from e

        faiss.normalize_L2(query_np)

        # Search for k nearest neighbors
        k = min(self.k_neighbors, self._index.ntotal)
        similarities, indices = self._index.search(query_np, k)

        # Convert similarities to weights via softmax
        sim_tensor = torch.from_numpy(similarities).float() / self.temperature
        weights = torch.softmax(sim_tensor, dim=1)  # (n_query, k)

        # Compute retrieval predictions for each target
        results = {}
        for name, pred_tensor in model_preds.items():
            if name not in self._train_targets:
                # No training targets available, use model prediction only
                results[name] = pred_tensor
                continue

            train_target = self._train_targets[name]

            # Gather neighbor targets
            neighbor_indices = indices  # (n_query, k)
            neighbor_targets = train_target[neighbor_indices]  # (n_query, k, ...)

            if neighbor_targets.ndim == 2:
                # Regression: weighted mean of neighbor targets
                neighbor_t = torch.from_numpy(neighbor_targets).float()
                retrieval_pred = (weights * neighbor_t).sum(dim=1, keepdim=True)
            elif neighbor_targets.ndim == 3:
                # Classification: weighted vote over neighbor logits/labels
                neighbor_t = torch.from_numpy(neighbor_targets).float()
                retrieval_pred = torch.einsum("bk,bkc->bc", weights, neighbor_t)
            else:
                results[name] = pred_tensor
                continue

            # Blend model prediction with retrieval prediction
            device = pred_tensor.device if hasattr(pred_tensor, "device") else "cpu"
            retrieval_pred = retrieval_pred.to(device)
            blended = (1.0 - self.alpha) * pred_tensor + self.alpha * retrieval_pred
            results[name] = blended

        return results
