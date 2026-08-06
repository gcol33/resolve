# Installation

RESOLVE is a C++ engine on libtorch. Python (`resolve_core`), R (`resolve`), and
the `resolve` command line are bindings over the same library, so a model
trained from one loads in any other.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (the extension links the libtorch that ships with it)
- CMake >= 3.18 and a C++17 compiler
- pandas and numpy for the in-memory (`from_pandas`) loaders

## Python

The engine builds from source against the PyTorch you already have installed:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve/src/core/python
pip install .
```

To iterate on the C++ sources, build in place instead:

```bash
pip install "scikit-build-core>=0.4.3" "nanobind>=2.0.0" cmake ninja
pip install . --no-build-isolation
```

## R

```r
install.packages("pak")
pak::pak("gcol33/resolve/r")
```

The R package loads the engine at runtime from a prebuilt backend library:

```r
library(resolve)
resolve.install_backend()          # CPU
resolve.install_backend(variant = "cuda")
```

## Command line

The CLI is built by the same CMake project:

```bash
cd resolve/src/core
cmake -B build -DBUILD_CLI=ON -DBUILD_TESTS=OFF -DBUILD_PYTHON=OFF
cmake --build build
./build/bin/resolve version
```

## Verifying the installation

```python
import resolve_core as rc

print(rc.__version__)
print(rc.TrainConfig().batch_size)
```

`resolve_core.__version__` is re-exported from the compiled engine constant, so
it reports the version of the binary actually loaded.

```r
library(resolve)
resolve.version()
```

## GPU support

CUDA is used when available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

Select the device on the training config:

```python
cfg = rc.TrainConfig()
cfg.device = "cuda"
cfg.vram_fraction = 0.80   # leave headroom when sharing the GPU with a desktop
```

Inference defaults to CPU. Pass `device="cuda"` to `Predictor.load` when the GPU
is idle and the test set is known to fit.
