"""Minimal REST API for RESOLVE model serving.

Usage:
    resolve serve --model model.pt --port 8000

    # Or programmatically:
    from resolve.serve import create_app
    app = create_app("model.pt")
    uvicorn.run(app, port=8000)

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def create_app(model_path: str, device: str = "cpu"):
    """Create a FastAPI app serving a trained RESOLVE model.

    Args:
        model_path: Path to saved checkpoint.
        device: "cpu" or "cuda".

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "REST serving requires fastapi and uvicorn. "
            "Install with: pip install fastapi uvicorn"
        )

    from resolve.inference.predictor import Predictor

    # Load model once at startup
    predictor = Predictor.load(model_path, device=device)

    app = FastAPI(
        title="RESOLVE Prediction API",
        description="Compositional data prediction via learned representations",
        version="0.5.0",
    )

    class PredictionRequest(BaseModel):
        """Single plot prediction request."""
        coordinates: list[float] | None = None  # [lat, lon]
        species: dict[str, float] = {}  # {species_name: abundance}
        covariates: dict[str, float] = {}  # {covariate_name: value}

    class BatchPredictionRequest(BaseModel):
        """Batch prediction request."""
        plots: list[PredictionRequest]

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": str(model_path), "device": device}

    @app.get("/info")
    async def info():
        schema = predictor.model.schema
        return {
            "encoding": predictor.model.species_encoding,
            "targets": [
                {"name": t.name, "task": t.task}
                for t in schema.targets
            ],
            "n_parameters": sum(p.numel() for p in predictor.model.parameters()),
            "device": device,
        }

    @app.post("/predict")
    async def predict(request: PredictionRequest):
        """Predict for a single plot.

        Note: This endpoint is for interactive use. For batch predictions,
        use /predict/batch or the Python API directly.
        """
        try:
            # For single-plot prediction, we need to encode species through
            # the same pipeline as training. This requires the dataset API.
            # For now, return a helpful error directing users to batch mode.
            raise HTTPException(
                status_code=501,
                detail="Single-plot prediction requires species encoding. "
                       "Use /predict/batch with a CSV file or the Python API."
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/dataset")
    async def predict_dataset(header_path: str, species_path: str):
        """Predict from CSV files on the server filesystem."""
        try:
            from resolve.data.dataset import ResolveDataset
            from resolve.data.roles import RoleMapping

            dataset = ResolveDataset.from_csv(
                header=header_path,
                species=species_path,
                roles=RoleMapping(),  # Use defaults
                targets={},
            )

            preds = predictor.predict(dataset)

            result = {
                "n_plots": len(preds.plot_ids),
                "predictions": {
                    target: values.tolist()
                    for target, values in preds.predictions.items()
                },
                "plot_ids": preds.plot_ids.tolist(),
            }
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8000, device: str = "cpu"):
    """Start the prediction server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Serving requires uvicorn. Install with: pip install uvicorn")

    app = create_app(model_path, device=device)
    uvicorn.run(app, host=host, port=port)
