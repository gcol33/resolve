"""A checkpoint with its weights stored in half precision.

A checkpoint is a TorchScript archive in which every entry is a tensor: the
weights under ``param_<name>``, and the model's buffers, configuration,
schema, scalers and vocabularies under their own keys. Loading copies each
stored weight into a model built in float32 (``load_weights_into``), which
casts it, so a checkpoint whose float32 weights are stored as float16 loads
into the same float32 model and runs the same float32 arithmetic; only the
rounding of the weights differs, and the file is about half the size. Every
other entry is kept as written: a scaler or a year held in float16 would
lose whole units.
"""

from __future__ import annotations

from pathlib import Path

import torch


def write_half_checkpoint(source: str | Path, destination: str | Path) -> int:
    """Write ``source`` to ``destination`` with its float32 weights in float16.

    Returns the number of weight tensors stored in half precision.
    """
    archive = torch.jit.load(str(source), map_location="cpu")
    halved = 0
    for name, tensor in list(archive.named_parameters()):
        if name.startswith("param_") and tensor.dtype == torch.float32:
            setattr(archive, name, torch.nn.Parameter(tensor.detach().half(), requires_grad=False))
            halved += 1
    torch.jit.save(archive, str(destination))
    return halved
