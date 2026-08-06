# Contribution Guidelines

First of all, thank you very much for taking the time to contribute
to the **RESOLVE** project!

This document provides guidelines for contributing to RESOLVE—its codebase and documentation.
These guidelines are meant to guide you, not to restrict you.
If in doubt, use your best judgment and feel free to propose improvements through an issue or pull request.

#### Table Of Contents

- [Code of Conduct](#code-of-conduct)
- [What the repository contains](#what-the-repository-contains)
- [Building](#building)
  - [Obtaining the source](#obtaining-the-source)
  - [The engine and the Python bindings](#the-engine-and-the-python-bindings)
  - [The C++ tests and the CLI](#the-c-tests-and-the-cli)
  - [The R package](#the-r-package)
- [Testing](#testing)
- [Documentation](#documentation)
- [Project organization](#project-organization)
- [Contributing workflow](#contributing-workflow)
- [Style guidelines](#style-guidelines)
- [Versioning](#versioning)
- [Pull request checklist](#pull-request-checklist)
- [Reporting bugs](#reporting-bugs)

## Code of Conduct

This project and everyone participating in it is governed by our **Code of Conduct** (`CODE_OF_CONDUCT.md`).
By participating, you are expected to uphold this code and maintain a respectful, inclusive environment.

## What the repository contains

RESOLVE is a C++ engine on libtorch, in `src/core/`. Python (`resolve_core`), R
(`resolve`), and the `resolve` command line are bindings over that one library
and carry API translation only, so a feature lands in the engine first and then
in each binding, with tests on both sides.

The repository root is not itself a Python distribution. The installable Python
package is `resolve-core`, whose project root is `src/core/python`.

## Building

This section is focused on development. For regular installation, see the
[README](./README.md) or the
[installation guide](https://gillescolling.com/resolve/tutorials/installation/).

### Obtaining the source

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve
```

### The engine and the Python bindings

Requirements:

- Python >= 3.10
- PyTorch >= 2.0 — the extension links the libtorch that ships with it
- CMake >= 3.18, Ninja, and a C++17 compiler
- Git, and an editor or IDE

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install "torch>=2.0"
pip install "scikit-build-core>=0.4.3" "nanobind>=2.0.0" cmake ninja pandas numpy
pip install pytest pytest-cov

pip install ./src/core/python --no-build-isolation
```

`--no-build-isolation` reuses the torch you just installed rather than
downloading a second copy into an isolated build environment. Re-run the last
command after any change under `src/core/` to rebuild the extension.

Check that the extension you built is the one that loads:

```bash
python tools/version.py --check-built
```

### The C++ tests and the CLI

The Catch2 suite and the `resolve` binary come from the same CMake project,
built against a standalone libtorch:

```bash
cd src/core
cmake -B build -DBUILD_TESTS=ON -DBUILD_CLI=ON -DBUILD_PYTHON=OFF -DUSE_CUDA=OFF \
      -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
cmake --build build
cd build && ctest --output-on-failure
./bin/resolve version
```

torch-heavy translation units peak at several GB each, so cap the parallelism
(`cmake --build build --parallel 2`) on a machine with limited RAM.

`-DUSE_CUDA=ON` additionally needs the CUDA toolkit and a CUDA-enabled libtorch.

### The R package

The R package is a client over the `resolve_c` C ABI. It compiles against one
vendored pure-C header and loads the backend at runtime, so it installs and
checks with no backend present:

```bash
R CMD INSTALL r
R CMD check r
```

To exercise the engine from R, fetch a prebuilt backend for the installed
package version:

```r
library(resolve)
resolve.install_backend()                  # CPU
resolve.install_backend(variant = "cuda")
```

Or build `resolve_c` from this tree and point `RESOLVE_C_HOME` at the directory
holding the library:

```bash
cd src/core
cmake -B build-capi -DBUILD_R_CAPI=ON -DBUILD_TESTS=OFF -DBUILD_PYTHON=OFF \
      -DBUILD_CLI=OFF -DUSE_CUDA=OFF -DRESOLVE_USE_OPENMP=OFF \
      -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
cmake --build build-capi
```

`RESOLVE_USE_OPENMP=OFF` matters here: an OpenMP-linked backend stacks a third
OpenMP runtime on R's own, and the R bindings never reach the one function that
uses it.

The canonical `include/resolve/resolve_capi.h` is vendored into
`r/src/resolve/`; CI fails on drift between the two, so re-copy it whenever the
ABI changes.

## Testing

Three suites cover the engine and its bindings:

| Suite | Location | Run with |
|-------|----------|----------|
| C++ (Catch2) | `src/core/tests/` | `ctest` in the CMake build directory |
| Python bindings | `tests/core/` | `pytest tests/core` |
| R | `r/tests/testthat/` | `R CMD check r` |

```bash
pytest tests/core
pytest tests/core/test_dataset.py -k "from_pandas"
pytest tests/core --cov=resolve_core --cov-report=html
```

`tests/core` skips itself when `resolve_core` is not installed, so `pytest` at
the repository root works in an environment that has only the tooling.

Guidelines:

- Keep tests fast and reproducible, with fixed seeds.
- Cover edge cases and expected failures.
- Prefer small synthetic examples to large datasets.
- A model change needs a recovery test: simulate from a known generating
  function, fit to convergence, and assert held-out accuracy against a
  threshold. `tests/core/test_recovery.py` and `src/core/tests/test_recovery.cpp`
  are the pattern; shape and NaN checks alone say nothing about whether a
  fitter learns.

## Documentation

The site is MkDocs Material, configured in `mkdocs.yml`:

```bash
pip install mkdocs mkdocs-material
mkdocs serve      # preview at http://localhost:8000
mkdocs build      # writes site/
```

- Tutorials and guides: `docs/tutorials/`
- API reference: `docs/api/`
- Package overview: `README.md`
- Changelog: `NEWS.md`, surfaced on the site as `docs/changelog.md`

Every documented signature has to match the bindings it describes:
`src/core/python/src/bindings_*.cpp` for Python, `r/R/resolve.R` for R, and
`src/core/cli/main.cpp` for the command line.

## Project organization

```
resolve/
├── .github/                <- CI workflows and issue templates
├── VERSION                 <- single source of truth for the version
├── pyproject.toml          <- repo-level tool config (not a distribution)
├── CONTRIBUTING.md
├── NEWS.md
├── README.md
├── src/core/               <- the C++ engine
│   ├── include/resolve/    <- public headers, including the C ABI facade
│   ├── cpp_src/            <- implementation
│   ├── cuda/               <- CUDA kernels
│   ├── cli/                <- the resolve command line
│   ├── python/             <- nanobind bindings; root of the resolve-core wheel
│   └── tests/              <- Catch2 tests and fixtures
├── r/                      <- R package, an Rcpp client over the C ABI
├── tests/core/             <- pytest suite over resolve_core
├── benchmarks/             <- encoding and architecture comparison harness
├── docs/                   <- documentation website
└── examples/               <- example notebooks
```

## Contributing workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```
2. **Make focused commits** with clear messages.
3. **Run tests and checks** before committing:
   ```bash
   pytest tests/core
   ruff check benchmarks tests tools
   python tools/version.py --check
   ```
4. **Update documentation** and `NEWS.md`.
5. **Update examples** if user-facing behavior changes.
6. **Open a pull request** with a short description of your change.
7. **Respond to review feedback** constructively.

## Style guidelines

### C++

- C++17. Headers in `include/resolve/`, implementation in `cpp_src/`.
- Extract shared logic into one helper rather than copying it between
  specialized variants; the compiler inlines `static inline` helpers, so the
  single source of truth costs nothing at runtime.
- Comments describe what the code does and why, in domain terms.

### Python

- Follow PEP 8 conventions.
- Use type hints for function signatures.
- Prefer vectorized operations (NumPy/PyTorch) over loops.
- Validate inputs early with clear error messages.
- Document all public functions with docstrings.

### Tests

- Add or update tests when functionality changes.
- Keep tests minimal and reproducible.
- Use pytest fixtures for shared setup; `tests/core/conftest.py` holds the
  synthetic-data builders.

## Versioning

The repo-root `VERSION` file is the single source of truth. Bump it with:

```bash
python tools/version.py --set 0.7.4          # add --date YYYY-MM-DD on release day
```

That rewrites the engine constant (`resolve::VERSION`), the package manifests,
`r/DESCRIPTION`, `CITATION.cff`, and opens a `NEWS.md` section.
Everything else reads the version at build or run time, so it cannot drift:
the CLI, `resolve.version()` in R, `resolve_core.__version__`, and the
checkpoint metadata all report the compiled `resolve::VERSION`, and
`r/inst/CITATION` reads `r/DESCRIPTION`.

`python tools/version.py --check` reports drift and gates CI; the engine
constant is compiled in, so rebuild after a bump. The R package version doubles
as the release-tag version: `resolve.install_backend()` builds its download URL
from `packageVersion("resolve")`, so a bump is only usable once a matching
`v<VERSION>` release carries the backend assets.

## Pull request checklist

- [ ] Python binding tests pass (`pytest tests/core`)
- [ ] C++ tests pass (`ctest` in the CMake build directory)
- [ ] Code passes linting (`ruff check benchmarks tests tools`)
- [ ] Version declarations agree (`python tools/version.py --check`)
- [ ] Documentation updated (`NEWS.md`, and `docs/` for any API change)
- [ ] Examples updated if needed
- [ ] No unrelated formatting changes
- [ ] PR description clearly explains the change

## Reporting bugs

When reporting an issue, please include:
- A minimal reproducible example
- Output of `python -c "import resolve_core; print(resolve_core.__version__)"`
  (or `resolve version` for the CLI, `resolve.version()` in R)
- Your Python version
- Expected vs. actual results
- Operating system and PyTorch version
- GPU info if relevant (`torch.cuda.get_device_name()`)

---

By contributing to RESOLVE, you agree that your code is released under the same license as the package.
