# RESOLVE Features

## Current Features

### Data Loading & Preprocessing

| Feature | Status | Notes |
|---------|--------|-------|
| CSV loading (header + species) | ✅ | `ResolveDataset::from_csv()` |
| Role-based column mapping | ✅ | Flexible schema via `RoleMapping` |
| Z-score normalization | ✅ | Computed on train set, applied to test |
| Log1p transform | ✅ | For skewed targets (e.g., area) |
| Train/test split | ✅ | Configurable ratio, shuffled |
| Species encoding (hash) | ✅ | MurmurHash3, configurable dim |
| Species encoding (embed) | ✅ | Learnable embeddings per position |
| Species encoding (sparse) | ✅ | Full vocabulary with linear projection |
| Taxonomy encoding | ✅ | Genus/family embeddings |
| Top-k species selection | ✅ | Top, bottom, top-bottom, all modes |
| Abundance normalization | ✅ | Raw, normalized, log1p |

### Model Architecture

| Feature | Status | Notes |
|---------|--------|-------|
| MLP backbone | ✅ | Configurable hidden dims |
| BatchNorm + GELU + Dropout | ✅ | Standard regularization |
| Multi-head output | ✅ | Regression + classification heads |
| Multi-task learning | ✅ | Shared encoder, task-specific heads |
| Configurable architecture | ✅ | `hidden_dims`, `dropout`, etc. |

### Training

| Feature | Status | Notes |
|---------|--------|-------|
| AdamW optimizer | ✅ | Configurable LR, weight decay |
| Gradient clipping | ✅ | Max norm = 1.0 |
| Early stopping | ✅ | Patience-based on test loss |
| Phased loss curriculum | ✅ | MAE → +SMAPE → +band penalty |
| Loss presets | ✅ | MAE, SMAPE, Combined modes |
| Checkpointing | ✅ | Best model, periodic, progress JSON |
| Resume training | ✅ | From checkpoint |
| Batch processing | ✅ | Configurable batch size |
| CUDA support | ✅ | Automatic device selection |
| Custom CUDA kernels | ✅ | Optimized hash embedding |
| Class weights | ✅ | Handle imbalanced classification |
| LR scheduling | ✅ | StepLR, CosineAnnealing |

### Metrics - Regression

| Feature | Status | Notes |
|---------|--------|-------|
| MAE | ✅ | Mean Absolute Error |
| RMSE | ✅ | Root Mean Squared Error |
| SMAPE | ✅ | Symmetric Mean Absolute Percentage Error |
| Band accuracy | ✅ | Configurable thresholds (25%, 50%, 75%) |
| R² score | ✅ | Coefficient of determination |

### Metrics - Classification

| Feature | Status | Notes |
|---------|--------|-------|
| Accuracy | ✅ | Overall accuracy |
| Confusion matrix | ✅ | Full NxN matrix |
| Per-class precision | ✅ | Per-class P |
| Per-class recall | ✅ | Per-class R |
| Per-class F1 | ✅ | Per-class F1 |
| Macro F1 | ✅ | Unweighted average |
| Weighted F1 | ✅ | Support-weighted average |
| Accuracy at threshold | ✅ | Confidence-filtered accuracy |
| Accuracy-coverage curve | ✅ | Multiple thresholds |
| ROC-AUC | ❌ | Planned |
| PR-AUC | ❌ | Planned |

### Inference

| Feature | Status | Notes |
|---------|--------|-------|
| Batch prediction | ✅ | `Predictor::predict()` |
| Latent extraction | ✅ | Get encoder embeddings |
| Automatic unscaling | ✅ | Inverse transform to original scale |
| Confidence values | ✅ | Via unknown_fraction |

### Language Bindings

| Feature | Status | Notes |
|---------|--------|-------|
| Python (nanobind) | ✅ | Full API exposure |
| R (Rcpp) | ✅ | Full API with testthat tests |
| CLI | ✅ | train, predict, info commands |

### Testing & CI

| Feature | Status | Notes |
|---------|--------|-------|
| C++ unit tests (Catch2) | ✅ | Model, dataset, metrics |
| Python tests (pytest) | ✅ | Full coverage |
| R tests (testthat) | ✅ | Metrics, encoder, trainer |
| GitHub Actions CI | ✅ | Cross-platform testing |
| PyPI release workflow | ✅ | cibuildwheel for wheels |
| R-CMD-check workflow | ✅ | CRAN compatibility |

---

## Planned Features

### Medium Priority (Quality of Life)

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| ROC-AUC metric | Medium | Medium | Area under ROC curve |
| PR-AUC metric | Medium | Medium | Area under precision-recall curve |
| Configurable gradient clipping | Medium | Easy | Expose max_norm parameter |
| Min delta for early stopping | Medium | Easy | Minimum improvement threshold |
| Warmup scheduler | Medium | Medium | Linear warmup for LR |

### Low Priority (Future)

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| Cross-validation | Low | High | K-fold, stratified K-fold |
| Model calibration | Low | Medium | Temperature scaling |
| Mixed precision (AMP) | Low | Medium | FP16/BF16 training |
| Gradient accumulation | Low | Medium | Effective larger batches |
| TorchScript export | Low | Medium | JIT compilation |
| ONNX export | Low | High | Cross-framework portability |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01 | Initial C++ engine with hash encoding |
| 0.2.0 | 2025-01 | Added embed/sparse encoding modes |
| 0.3.0 | 2025-01 | Per-class F1, confidence metrics |
| 0.4.0 | 2025-01 | R² score, class weights, LR scheduling |
