# RESOLVE Release Plan

## Overview

Transform RESOLVE from research tool to polished software release.

**Priority order:**
1. C++ tests (Catch2)
2. Python tests (pytest)
3. R bindings update
4. R tests (testthat)
5. CLI implementation
6. PyPI/CRAN packaging
7. Documentation (last)

---

## Phase 1: C++ Tests (Catch2)

### Setup

```
src/core/
├── tests/
│   ├── CMakeLists.txt
│   ├── test_main.cpp          # Catch2 main
│   ├── test_loss.cpp          # Metrics tests
│   ├── test_model.cpp         # Model forward pass
│   ├── test_trainer.cpp       # Training loop
│   ├── test_dataset.cpp       # CSV loading, encoding
│   ├── test_encoding.cpp      # Hash, embed, sparse modes
│   └── fixtures/
│       ├── small_header.csv
│       └── small_species.csv
```

### CMake Integration

```cmake
# In src/core/CMakeLists.txt
option(BUILD_TESTS "Build test suite" OFF)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(resolve_tests
        tests/test_main.cpp
        tests/test_loss.cpp
        tests/test_model.cpp
        tests/test_trainer.cpp
        tests/test_dataset.cpp
        tests/test_encoding.cpp
    )
    target_link_libraries(resolve_tests PRIVATE resolve_core Catch2::Catch2WithMain)

    include(CTest)
    include(Catch)
    catch_discover_tests(resolve_tests)
endif()
```

### Test Categories

| File | Tests |
|------|-------|
| test_loss.cpp | MAE, RMSE, R², SMAPE, band_accuracy, classification_metrics, confusion_matrix |
| test_model.cpp | Forward pass shapes, latent extraction, device transfer |
| test_trainer.cpp | Data preparation, train/test split, LR scheduling, checkpointing |
| test_dataset.cpp | CSV loading, role mapping, schema inference |
| test_encoding.cpp | Hash encoding, embed mode, sparse mode, taxonomy vocab |

### Example Test Pattern

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/loss.hpp"

using namespace Catch::Matchers;

TEST_CASE("Metrics::r_squared", "[metrics][regression]") {
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto target = torch::tensor({1.1f, 2.0f, 2.9f, 4.1f, 5.0f});

    float r2 = resolve::Metrics::r_squared(pred, target);

    REQUIRE_THAT(r2, WithinAbs(0.99, 0.01));
}

TEST_CASE("Metrics::r_squared perfect fit", "[metrics][regression]") {
    auto vals = torch::tensor({1.0f, 2.0f, 3.0f});

    REQUIRE(resolve::Metrics::r_squared(vals, vals) == 1.0f);
}

TEST_CASE("Metrics::classification_metrics", "[metrics][classification]") {
    // 3 classes, batch of 5
    auto pred = torch::tensor({
        {0.9f, 0.05f, 0.05f},  // pred: 0
        {0.1f, 0.8f, 0.1f},    // pred: 1
        {0.1f, 0.1f, 0.8f},    // pred: 2
        {0.7f, 0.2f, 0.1f},    // pred: 0
        {0.1f, 0.7f, 0.2f}     // pred: 1
    });
    auto target = torch::tensor({0L, 1L, 2L, 1L, 1L});  // true: 0,1,2,1,1

    auto metrics = resolve::Metrics::classification_metrics(pred, target, 3);

    REQUIRE(metrics.accuracy == 0.8f);  // 4/5 correct
    REQUIRE(metrics.per_class_support[0] == 1);
    REQUIRE(metrics.per_class_support[1] == 3);
    REQUIRE(metrics.per_class_support[2] == 1);
}
```

---

## Phase 2: Python Tests (pytest)

### Setup

```
src/resolve/
├── tests/
│   ├── conftest.py            # Fixtures
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_trainer.py
│   ├── test_dataset.py
│   └── fixtures/
│       ├── small_header.csv
│       └── small_species.csv
```

### pyproject.toml additions

```toml
[project.optional-dependencies]
test = ["pytest>=7.0", "pytest-cov"]

[tool.pytest.ini_options]
testpaths = ["src/resolve/tests"]
addopts = "-v --tb=short"
```

### Example Test Pattern

```python
import pytest
import torch
from resolve import Trainer, ResolveModel, ResolveSchema, TargetConfig, TaskType

@pytest.fixture
def simple_schema():
    schema = ResolveSchema()
    schema.n_plots = 100
    schema.has_coordinates = True
    target = TargetConfig()
    target.name = "area"
    target.task = TaskType.Regression
    schema.targets = [target]
    return schema

@pytest.fixture
def sample_data(simple_schema):
    return {
        "coordinates": torch.randn(100, 2),
        "hash_embedding": torch.randn(100, 32),
        "targets": {"area": torch.randn(100)}
    }

def test_model_forward_shape(simple_schema):
    model = ResolveModel(simple_schema)
    x = torch.randn(16, 34)  # batch=16, coords(2) + hash(32)
    out = model.forward(x)

    assert "area" in out
    assert out["area"].shape == (16, 1)

def test_trainer_fit(simple_schema, sample_data):
    model = ResolveModel(simple_schema)
    trainer = Trainer(model)
    trainer.prepare_data_raw(
        sample_data["coordinates"],
        torch.Tensor(),  # no covariates
        sample_data["hash_embedding"],
        # ... other tensors
        sample_data["targets"]
    )
    result = trainer.fit()

    assert result.best_epoch >= 0
    assert "area" in result.final_metrics
```

---

## Phase 3: R Bindings Update

### Current State

R bindings are partial and need sync with current C++ API.

### Tasks

1. **Audit current R bindings** - Compare with Python bindings
2. **Update Rcpp exports** - Match Python API
3. **Add new features:**
   - `LRSchedulerType` enum
   - `class_weights` in TargetConfig
   - LR scheduler config fields
   - `r_squared` metric
   - `ResolveDataset` class
4. **Test R package builds**

### File Structure

```
r/
├── DESCRIPTION
├── NAMESPACE
├── R/
│   ├── resolve.R           # High-level API
│   ├── trainer.R           # Trainer wrapper
│   ├── dataset.R           # Dataset loading
│   └── metrics.R           # Metrics access
├── src/
│   ├── Makevars
│   ├── bindings.cpp        # Rcpp bindings
│   └── RcppExports.cpp
└── tests/
    └── testthat/
```

---

## Phase 4: R Tests (testthat)

### Setup

```
r/tests/
├── testthat.R
└── testthat/
    ├── test-metrics.R
    ├── test-model.R
    ├── test-trainer.R
    ├── test-dataset.R
    └── fixtures/
        ├── small_header.csv
        └── small_species.csv
```

### Example Test Pattern

```r
test_that("r_squared returns correct value", {
  pred <- c(1.0, 2.0, 3.0, 4.0, 5.0)
  target <- c(1.1, 2.0, 2.9, 4.1, 5.0)

  r2 <- resolve_r_squared(pred, target)

  expect_equal(r2, 0.99, tolerance = 0.01)
})

test_that("trainer completes training", {
  skip_on_cran()

  dataset <- resolve_load_dataset(
    header_path = test_path("fixtures/small_header.csv"),
    species_path = test_path("fixtures/small_species.csv"),
    roles = list(plot_id = "plot", species_id = "species"),
    targets = list(resolve_target_regression("area"))
  )

  result <- resolve_train(dataset, max_epochs = 10)

  expect_true(result$best_epoch >= 0)
  expect_true("area" %in% names(result$final_metrics))
})
```

---

## Phase 5: CLI Implementation

### Design

```bash
# Training
resolve train \
  --header data/header.csv \
  --species data/species.csv \
  --target area:regression:log1p \
  --target eunis:classification:9 \
  --output models/my_model.pt \
  --epochs 500 \
  --patience 50 \
  --lr 1e-3 \
  --lr-scheduler cosine

# Prediction
resolve predict \
  --model models/my_model.pt \
  --header data/new_header.csv \
  --species data/new_species.csv \
  --output predictions.csv

# Model info
resolve info models/my_model.pt
```

### Implementation

```
src/core/cli/
├── CMakeLists.txt
├── main.cpp              # CLI11 setup, command dispatch
├── train_cmd.cpp         # Train command
├── train_cmd.hpp
├── predict_cmd.cpp       # Predict command
├── predict_cmd.hpp
├── info_cmd.cpp          # Info command
└── info_cmd.hpp
```

### CMake

```cmake
# CLI executable
add_executable(resolve_cli
    cli/main.cpp
    cli/train_cmd.cpp
    cli/predict_cmd.cpp
    cli/info_cmd.cpp
)
target_link_libraries(resolve_cli PRIVATE resolve_core CLI11::CLI11)
set_target_properties(resolve_cli PROPERTIES OUTPUT_NAME "resolve")
```

---

## Phase 6: Packaging

### PyPI (Python)

1. Update `pyproject.toml` with metadata
2. Add `README.md` for PyPI
3. Configure scikit-build-core for wheel building
4. Set up GitHub Actions for wheel builds (cibuildwheel)
5. Publish to PyPI

### CRAN (R)

1. Complete R package structure
2. Add vignettes
3. Run `R CMD check --as-cran`
4. Submit to CRAN

---

## Phase 7: Documentation (Last)

### Components

1. **README.md** - Quick start, installation, basic usage
2. **API Reference** - Auto-generated from docstrings
3. **Tutorials** - Jupyter notebooks / R vignettes
4. **Architecture** - Design decisions, encoding modes

### Tools

- Python: mkdocs + mkdocstrings
- R: pkgdown
- Unified: Consider Quarto for both

---

## CI/CD (GitHub Actions)

### Workflows

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  cpp-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install libtorch
        run: # ...
      - name: Build and test
        run: |
          cmake -B build -DBUILD_TESTS=ON
          cmake --build build
          ctest --test-dir build --output-on-failure

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Install and test
        run: |
          pip install -e ".[test]"
          pytest

  r-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: r-lib/actions/setup-r@v2
      - name: Test
        run: |
          devtools::test()
        shell: Rscript {0}
```

---

## Timeline Estimate

| Phase | Effort |
|-------|--------|
| 1. C++ tests | Medium |
| 2. Python tests | Easy |
| 3. R bindings | Medium |
| 4. R tests | Easy |
| 5. CLI | Medium |
| 6. Packaging | Medium |
| 7. Documentation | Medium |

---

## Test Data Requirements

Create minimal test fixtures:
- `small_header.csv` - 50-100 plots, 2-3 covariates, 1 regression + 1 classification target
- `small_species.csv` - Corresponding species data with taxonomy

These should be small enough for fast tests but representative of real data structure.
