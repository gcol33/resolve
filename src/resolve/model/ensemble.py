"""Ensemble: combine predictions from multiple encoding modes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from resolve.data.dataset import ResolveDataset
from resolve.inference.predictor import Predictor, ResolvePredictions
from resolve.train.trainer import Trainer


class EnsemblePredictor:
    """Ensemble of trained models with different encoding modes.

    Averages predictions from multiple independently-trained models.
    For regression: arithmetic mean. For classification: averaged probabilities.

    Usage:
        ensemble = EnsemblePredictor.train(
            dataset,
            encodings=["hash", "embed", "rank_pool"],
            max_epochs=100,
        )
        predictions = ensemble.predict(dataset)
    """

    def __init__(self, predictors: list[Predictor], weights: list[float] | None = None):
        self.predictors = predictors
        self.weights = weights or [1.0 / len(predictors)] * len(predictors)
        if len(self.weights) != len(self.predictors):
            raise ValueError("weights must match number of predictors")

    @classmethod
    def train(
        cls,
        dataset: ResolveDataset,
        encodings: list[str] = ("hash", "embed", "rank_pool"),
        save_dir: str | None = None,
        **trainer_kwargs,
    ) -> "EnsemblePredictor":
        """Train an ensemble of models with different encoding modes."""
        predictors = []
        for i, enc in enumerate(encodings):
            print(f"\n=== Training model {i+1}/{len(encodings)}: {enc} ===")
            trainer = Trainer(dataset, species_encoding=enc, **trainer_kwargs)
            result = trainer.fit()

            if save_dir:
                from pathlib import Path
                path = Path(save_dir) / f"model_{enc}.pt"
                trainer.save(str(path))
                predictors.append(Predictor.load(str(path)))
            else:
                # Build predictor from trainer state
                predictors.append(Predictor(
                    model=trainer.model,
                    scalers=trainer._scalers,
                    encoder=getattr(trainer, '_species_encoder', None)
                    or getattr(trainer, '_embedding_encoder', None)
                    or getattr(trainer, '_rank_pool_encoder', None),
                    schema=trainer._schema,
                    dataset=dataset,
                ))

        return cls(predictors)

    @classmethod
    def load(cls, paths: list[str], device: str = "cpu") -> "EnsemblePredictor":
        """Load an ensemble from multiple checkpoint files."""
        predictors = [Predictor.load(p, device=device) for p in paths]
        return cls(predictors)

    def predict(self, dataset: ResolveDataset) -> ResolvePredictions:
        """Average predictions from all models."""
        all_preds = [p.predict(dataset) for p in self.predictors]

        # Average predictions per target
        targets = list(all_preds[0].predictions.keys())
        averaged = {}
        for target in targets:
            stacked = np.stack([p.predictions[target] for p in all_preds])
            averaged[target] = np.average(stacked, axis=0, weights=self.weights)

        return ResolvePredictions(
            predictions=averaged,
            plot_ids=all_preds[0].plot_ids,
        )
