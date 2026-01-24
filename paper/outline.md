# RESOLVE: Species Composition as Biotic Context for Multi-Task Plot Attribute Prediction

## Abstract

Species composition data, routinely collected in ecological surveys, encodes rich information beyond species identities alone. We present RESOLVE (Representation Encoding of Species Outcomes via Linear Vector Embeddings), a framework that learns shared representations from species composition to predict plot-level attributes. Our approach uses linear compositional pooling—aggregating species contributions additively before nonlinear transformation—which enforces interpretability and prevents overfitting to spurious co-occurrence patterns. RESOLVE tracks species not seen during training, using unknown species fraction as a confidence proxy that enables selective gap-filling: users set a threshold to accept only confident predictions. We demonstrate that diverse plot characteristics can be predicted from species assemblages, validating the ecological intuition that species presence reflects integrated environmental and historical filtering.

## 1. Introduction

### 1.1 Species composition as biotic context
- Vegetation plots: standard unit in ecology, species lists + abundances
- Species presence/absence reflects integrated environmental filtering
- Current approaches treat species data as predictors for single outcomes
- Opportunity: multi-task learning to reveal shared ecological signal

### 1.2 The multi-task hypothesis
- Core claim: If species composition encodes a meaningful latent representation of "ecological context", this representation should simultaneously inform:
  - Physical plot characteristics (area, measured as relevé extent)
  - Climatic conditions (BIO1: mean annual temperature, BIO12: annual precipitation)
  - Habitat type (EUNIS classification)
- Success of multi-task prediction validates that species→context encoding exists
- Failure would suggest these attributes are independently determined

### 1.3 Contributions
1. Hybrid species encoding: feature hashing + learned taxonomic embeddings
2. Multi-task architecture with shared encoder and task-specific heads
3. Phased training strategy for stable multi-objective optimization
4. Open-source, cross-platform implementation (C++ core, Python/R bindings)

## 2. Methods

### 2.1 Problem formulation
- Input: Plot-level species composition {(species_i, genus_i, family_i, abundance_i)}
- Outputs: Multiple targets with heterogeneous types
  - Regression: continuous values (area, climate variables)
  - Classification: categorical labels (habitat type)
- Goal: Learn f: species_composition → (y_area, y_climate, y_habitat)

### 2.2 Species encoding

#### 2.2.1 Feature hashing for species identity
- High-dimensional, sparse species space
- Feature hashing with signed projections (Weinberger et al., 2009)
- Produces fixed-dimension dense embedding regardless of vocabulary size
- Formula: h(species) → d-dimensional vector via hash + sign

#### 2.2.2 Learned taxonomic embeddings
- Genus and family provide phylogenetic/functional groupings
- Top-k most abundant genera/families per plot
- Learned embedding matrices: E_genus ∈ R^{|G| × d_g}, E_family ∈ R^{|F| × d_f}
- Position-specific embeddings (rank 1 genus vs rank 2 genus)

#### 2.2.3 Combined encoding
- Concatenate: [coords || hash_emb || genus_emb_1..k || family_emb_1..k || covariates]
- Optional covariates: environmental layers, survey metadata

### 2.3 Model architecture

#### 2.3.1 Shared encoder
- Multi-layer perceptron with batch normalization
- GELU activations, dropout regularization
- Output: latent representation z ∈ R^{d_latent}

#### 2.3.2 Task-specific heads
- Regression heads: linear projection to scalar + optional inverse transform
- Classification heads: linear projection to logits

### 2.4 Training

#### 2.4.1 Multi-task loss
- Weighted sum: L = Σ_t w_t · L_t
- Regression: MAE (robust to outliers in area)
- Classification: cross-entropy

#### 2.4.2 Phased training
- Phase 1 (epochs 0-100): MAE only for regression, stable gradients
- Phase 2 (epochs 100-300): Add SMAPE for scale-invariant optimization
- Phase 3 (epochs 300+): Add band penalty for practical accuracy
- Motivation: curriculum learning for heterogeneous loss landscape

#### 2.4.3 Optimization
- AdamW optimizer with weight decay
- Early stopping on validation loss
- Standard train/test split (80/20)

### 2.5 Evaluation metrics
- Regression: MAE, RMSE, SMAPE, band accuracy (±25%)
- Classification: accuracy, per-class F1

## 3. Data

### 3.1 EVA/ASAAS database
- European vegetation plots
- Standardized species names (World Flora Online backbone)
- Plot-level metadata: coordinates, area, survey year

### 3.2 Target variables
- **Area**: Relevé area in m², log-transformed, wide range (1-10000+ m²)
- **Climate**: BIO1 (mean annual temp), BIO12 (annual precip) from WorldClim
- **Habitat**: EUNIS level 1 classification (9 classes: M, N, P, Q, R, S, T, U, V)

### 3.3 Data preparation
- Filter plots with complete coordinates and target values
- Taxonomic standardization to WFO backbone
- No environmental covariates (pure species-based prediction)

## 4. Results

### 4.1 Multi-task performance
- Report metrics for each target
- Compare with single-task baselines
- Key question: Does multi-task improve over single-task?

### 4.2 Learned representations
- t-SNE/UMAP visualization of latent space
- Clustering by habitat type
- Correlation of latent dimensions with environmental gradients

### 4.3 Taxonomic embeddings analysis
- Genus embedding space structure
- Do phylogenetically similar genera cluster?
- Family-level functional groupings

### 4.4 Ablation studies
- Hash-only vs taxonomy-only vs hybrid encoding
- Effect of top_k parameter
- Impact of phased training

## 5. Discussion

### 5.1 Species composition as ecological context
- Multi-task success suggests species encode integrated environmental signal
- Shared representation captures "ecological context" meaningfully
- Implications for trait-based ecology, functional biogeography

### 5.2 Practical applications
- Gap-filling: predict missing plot metadata
- Quality control: flag implausible area values
- Habitat classification: automated EUNIS assignment

### 5.3 Limitations
- Requires sufficient species coverage in training data
- Geographic transferability unknown
- Taxonomic resolution affects performance

### 5.4 Future directions
- Incorporate phylogenetic distances
- Attention mechanisms for species importance
- Transfer learning across regions

## 6. Conclusion

Species composition, when appropriately encoded, provides a rich biotic context that simultaneously informs multiple plot-level attributes. RESOLVE demonstrates that a shared latent representation learned from species assemblages can predict physical, climatic, and categorical habitat properties. The linear compositional pooling constraint provides interpretability while the unknown species tracking enables honest uncertainty quantification. We provide open-source implementations in Python and R to enable adoption in ecological research.

## Implementation

### Code availability
- Python package: `pip install resolve`
- R package: `devtools::install_github("gcol33/resolve")`
- Source code: https://github.com/gcol33/resolve

### System requirements
- Python ≥3.10, R ≥4.0
- PyTorch ≥2.0
- CPU or CUDA GPU

---

## Publication Strategy

### Option A: Methods Paper (Methods in Ecology and Evolution)

**Pitch:** Introduce the inverse-SDM framing. Show that linear pooling + unknown tracking outperforms standard approaches.

**Strengths:**
- Novel framing (species → plot attributes vs environment → species)
- Testable inductive bias (linear compositional pooling)
- Practical confidence mechanism

**Required:**
- Benchmark on 2–3 datasets
- Comparison with random forest, gradient boosting, MLP without linear constraint
- Ablation: linear pooling vs. attention-based pooling

### Option B: Software Paper (JOSS)

**Pitch:** Practical tool for gap-filling ecological data. Clean API, handles unknown species.

**Required:**
- Documented use case
- Installation and usage examples
- Statement of need

### Option C: Short Communication

**Pitch:** Focus on unknown species tracking + confidence threshold mechanism.

---

## Key Methodological Claims to Test

### 1. Linear Compositional Pooling

Species contributions are aggregated linearly before nonlinear transformation:

```
z_species = Σ w_i · h(species_i)
```

**Hypothesis:** Linear pooling outperforms unconstrained aggregation on small-to-medium ecological datasets.

**Test:** Compare RESOLVE vs. attention-based pooling vs. set transformers.

### 2. Unknown Species as Confidence Proxy

**Hypothesis:** Prediction error correlates with unknown_fraction.

**Test:** Hold out species, compute correlation between unknown_fraction and error.

### 3. Confidence Thresholding Works

**Hypothesis:** Filtering by confidence_threshold > 0.8 removes the worst predictions.

**Test:** Show error distribution for confident vs. uncertain predictions.

---

## Experiments to Run

| Experiment | Question | Comparison |
|------------|----------|------------|
| Linear pooling ablation | Does linear constraint help? | RESOLVE vs. attention vs. set transformer |
| Unknown calibration | Is unknown_fraction predictive of error? | Correlation plot |
| Benchmark | How does RESOLVE compare to baselines? | RF, XGBoost, MLP |
| Taxonomy value | Does genus/family help? | With vs. without taxonomy |
| Gap-filling | Can we selectively impute? | Confident vs. uncertain predictions |

---

## Target Journals

| Journal | Type | Fit |
|---------|------|-----|
| Methods in Ecology and Evolution | Methods | High |
| JOSS | Software | High (low barrier) |
| Ecological Informatics | Applied | Medium |
| Ecography | Methods | Medium |

---

## Next Steps

- [ ] Run ASAAS benchmark with current code
- [ ] Implement attention pooling variant for ablation
- [ ] Generate calibration plots (unknown_fraction vs. error)
- [ ] Draft Introduction section
- [ ] Request sPlot access for validation dataset

---

## References

[To be added: key citations on feature hashing, multi-task learning, vegetation databases, EUNIS classification, species-environment relationships]
