# Spacc: Species Composition as Biotic Context for Multi-Task Plot Attribute Prediction

## Abstract

Species composition data, routinely collected in vegetation surveys, encodes rich ecological information beyond species identities alone. We present Spacc, a multi-task neural network framework that learns a shared latent representation from species composition to simultaneously predict multiple plot-level attributes. Our approach demonstrates that diverse plot characteristics—including physical properties (plot area), environmental conditions (climate), and habitat classifications (EUNIS)—can be jointly predicted from a unified species-derived representation. This shared encoding supports the ecological intuition that species assemblages reflect integrated environmental and historical constraints. We provide cross-platform implementations in Python and R built on a portable C++ core, enabling broad adoption in ecological research.

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

Species composition, when appropriately encoded, provides a rich biotic context that simultaneously informs multiple plot-level attributes. The Spacc framework demonstrates that a shared latent representation learned from species assemblages can predict physical, climatic, and categorical habitat properties. This validates the ecological intuition that species presence reflects integrated environmental and historical filtering. We provide open-source implementations to enable adoption in vegetation science and related fields.

## Implementation

### Code availability
- Python package: `pip install spacc`
- R package: `install.packages("spacc")` (CRAN) or `devtools::install_github("gcol33/spacc")`
- Source code: https://github.com/gcol33/spacc-core

### System requirements
- Python ≥3.9, R ≥4.0
- PyTorch ≥2.0 (Python), torch for R
- CPU or CUDA GPU

## References

[To be added: key citations on feature hashing, multi-task learning, vegetation databases, EUNIS classification, species-environment relationships]
