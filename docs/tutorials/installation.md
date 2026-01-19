# Installation

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- pandas >= 2.0
- scikit-learn >= 1.3

## From PyPI

```bash
pip install resolve
```

## From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve
pip install -e .
```

## Optional Dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- ruff for linting
- mypy for type checking

## Verifying Installation

```python
import resolve
print(resolve.__version__)

# Quick check
from resolve import ResolveDataset, ResolveModel, Trainer
print("RESOLVE installed successfully!")
```

## GPU Support

RESOLVE automatically detects CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

To use GPU for training, simply specify the device:

```python
trainer = resolve.Trainer(model, dataset, device="cuda")
```

Or let RESOLVE auto-detect:

```python
trainer = resolve.Trainer(model, dataset, device="auto")  # default
```
