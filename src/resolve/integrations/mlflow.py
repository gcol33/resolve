"""MLflow integration for experiment tracking.

Usage:
    from resolve.integrations.mlflow import log_training_run

    trainer = Trainer(dataset, ...)
    result = trainer.fit()
    log_training_run(trainer, result, experiment_name="resolve-experiments")

Requires: pip install mlflow
"""

from __future__ import annotations

from typing import Any, Optional


def log_training_run(
    trainer: Any,
    result: Any,
    experiment_name: str = "resolve",
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Log a training run to MLflow.

    Args:
        trainer: Trained Trainer instance.
        result: TrainResult from trainer.fit().
        experiment_name: MLflow experiment name.
        run_name: Optional run name (auto-generated if None).
        tags: Optional tags dict.

    Returns:
        MLflow run ID.
    """
    try:
        import mlflow
    except ImportError:
        raise ImportError(
            "MLflow integration requires mlflow. "
            "Install with: pip install mlflow"
        )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_params({
            "species_encoding": trainer.species_encoding,
            "hidden_dims": str(list(trainer.hidden_dims)),
            "hash_dim": trainer.hash_dim,
            "lr": trainer.lr,
            "batch_size": trainer.batch_size,
            "max_epochs": trainer.max_epochs,
            "patience": trainer.patience,
            "dropout": trainer.dropout,
            "encoder_architecture": getattr(trainer, 'encoder_architecture', 'mlp'),
            "moe_routing": getattr(trainer, 'moe_routing', 'none'),
            "species_dropout": getattr(trainer, 'species_dropout', 0.0),
        })

        # Log metrics
        mlflow.log_metric("best_epoch", result.best_epoch)
        mlflow.log_metric("train_time_seconds", result.train_time)

        for target, metrics in result.final_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{target}/{metric_name}", value)

        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log model artifact
        if hasattr(trainer, '_checkpoint_path') and trainer._checkpoint_path:
            mlflow.log_artifact(trainer._checkpoint_path)

        return run.info.run_id
