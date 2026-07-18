#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <functional>
#include <iostream>
#include <set>

namespace resolve {

// =============================================================================
// Version and constants
// =============================================================================

inline constexpr const char* VERSION = "0.7.1";

// Training defaults
constexpr int kDefaultBatchSize = 4096;
constexpr int kDefaultMaxEpochs = 500;
constexpr int kDefaultPatience = 50;
constexpr float kDefaultLearningRate = 1e-3f;
constexpr float kDefaultWeightDecay = 1e-4f;
constexpr float kDefaultTestSize = 0.2f;
constexpr int kDefaultSeed = 42;

// Model architecture defaults
constexpr int kDefaultHashDim = 32;
constexpr float kDefaultDropout = 0.3f;
constexpr float kDefaultLeakyReLUSlope = 0.01f;
constexpr float kDefaultELUAlpha = 1.0f;
constexpr int kDefaultNormGroups = 32;

// Numerical stability constants
constexpr float kExpClampMin = -88.0f;
constexpr float kExpClampMax = 88.0f;
constexpr float kEpsilon = 1e-8f;

// Initialization constants
constexpr float kBertInitStd = 0.02f;
constexpr float kAttentionMaskFill = -1e9f;
constexpr float kGATInitStd = 0.01f;

// Phase boundaries for phased loss
constexpr int kDefaultPhase1Epoch = 100;
constexpr int kDefaultPhase2Epoch = 300;

// =============================================================================
// Logging
// =============================================================================

// Logging callback for training progress
using LogCallback = std::function<void(const std::string&)>;

// Default logging to stdout
inline void default_log(const std::string& msg) noexcept {
    std::cout << msg << std::endl;
}

// Null logger (discards all messages)
inline void null_log(const std::string&) noexcept {}

// Task type for prediction heads
enum class TaskType {
    Regression,
    Classification
};

// Target transform type
enum class TransformType {
    None,
    Log1p
};

// Species encoding mode
enum class SpeciesEncodingMode {
    Hash,        // Feature hashing (default)
    Embed,       // Learnable embeddings for top-k species
    Sparse,      // Explicit species abundance/presence vector
    RankPool,    // Rank-pooled species embeddings with weighted pooling
    Transformer  // Transformer species encoder with attention pooling
};

// Loss configuration presets
enum class LossConfigMode {
    MAE,       // Pure MAE loss (no SMAPE, no band penalty)
    SMAPE,     // SMAPE as primary loss
    Combined,  // Phased: MAE -> MAE+SMAPE -> MAE+SMAPE+band (default)
    NCA        // Neighborhood Component Analysis loss (classification only)
};

// Learning rate scheduler type
enum class LRSchedulerType {
    None,           // Constant learning rate
    StepLR,         // Step decay every N epochs
    CosineAnnealing // Cosine annealing to min_lr
};

// Species selection mode (which species to include)
enum class SelectionMode {
    Top,        // Top-k by abundance
    Bottom,     // Bottom-k by abundance
    TopBottom,  // Top-k and bottom-k
    All         // All species (explicit vector)
};

// How species are represented
enum class RepresentationMode {
    Abundance,        // Use abundance values
    PresenceAbsence   // Binary presence/absence
};

// Normalization for abundances
enum class NormalizationMode {
    Raw,    // Raw abundance values
    Norm,   // Normalized (sum to 1)
    Log1p   // Log1p transformed
};

// Aggregation mode for taxonomy
enum class AggregationMode {
    Abundance,  // Sum abundances
    Count       // Count species
};

// Activation function type for configurable architecture
enum class ActivationType {
    ReLU,
    LeakyReLU,
    GELU,       // Default (current behavior)
    SiLU,       // Also known as Swish
    Tanh,
    Mish,
    ELU,
    SELU,
    Softplus,
    PReLU       // Note: has learnable parameters
};

// Normalization layer type for configurable architecture
enum class NormLayerType {
    BatchNorm,  // Default (current behavior)
    LayerNorm,
    GroupNorm,
    RMSNorm,    // Custom implementation (not in libtorch)
    None        // No normalization
};

// Configuration for a prediction target
struct TargetConfig {
    std::string name;
    TaskType task;
    TransformType transform = TransformType::None;
    int num_classes = 0;  // For classification
    float weight = 1.0f;  // Loss weight in multi-task
    std::vector<float> class_weights;  // Optional class weights for imbalanced classification
    // Ordered class vocabulary for classification targets. class_names[i] is
    // the original CSV string that encodes to int code i. Empty for
    // regression. Populated by ResolveDataset::load_header_data when it
    // factorizes a string-coded classification column (e.g. EUNIS "M","N",
    // "P",...) into int64 codes. Persisted in the checkpoint so the
    // Predictor can map predicted codes back to human-readable class names.
    std::vector<std::string> class_names;
};

// Schema information for a dataset
struct ResolveSchema {
    int64_t n_plots = 0;
    int64_t n_species = 0;          // Number of unique species
    int64_t n_species_vocab = 0;    // Size of species vocabulary (for embed/sparse modes)
    bool has_coordinates = false;   // Default false, set true when coords found
    bool has_abundance = false;
    bool has_taxonomy = false;
    int64_t n_genera = 0;
    int64_t n_families = 0;
    int64_t n_genera_vocab = 0;     // Vocab size for embed mode
    int64_t n_families_vocab = 0;   // Vocab size for embed mode
    std::vector<std::string> covariate_names;
    std::vector<TargetConfig> targets;
    bool track_unknown_fraction = true;
    bool track_unknown_count = false;

    // Categorical covariates. Parallel layout: categorical_names[i] is the
    // CSV column name; categorical_vocab_sizes[i] is the size of that
    // column's embedding table (= K + 1 where K is the number of distinct
    // non-NA values, and the +1 is the reserved UNK slot at code 0).
    // Empty when the dataset has no categorical covariates.
    std::vector<std::string> categorical_names;
    std::vector<int64_t> categorical_vocab_sizes;
    // Embedding dimension shared across all categorical columns (matches the
    // value carried on ModelConfig). Stored on the schema so the model can
    // be reconstructed from a checkpoint without needing the original
    // ModelConfig — important for Predictor.load.
    int64_t categorical_embed_dim = 8;

    // Rank-pool / transformer pooling scheme + species cap used to build the
    // pool weight tensors at load time. Stored so a checkpoint can rebuild the
    // matching inference-side DatasetConfig instead of silently defaulting to
    // Log1p (which recomputes different per-species weights for the same model).
    // pool_weighting is the underlying PoolWeighting enum value
    // (Binary=0, Abundance=1, Log1p=2, Norm=3, Rank=4); kept as int because
    // types.hpp cannot include species_encoding.hpp. pool_species_cap mirrors
    // DatasetConfig::pool_species_cap (0 = auto p99).
    int pool_weighting = 2;    // PoolWeighting::Log1p
    int pool_species_cap = 0;  // 0 = auto

    // Helper: true if this schema has categorical covariates configured.
    [[nodiscard]] bool has_categoricals() const noexcept {
        return !categorical_names.empty();
    }
    // Helper: how many categorical columns.
    [[nodiscard]] int64_t n_categoricals() const noexcept {
        return static_cast<int64_t>(categorical_names.size());
    }
};

// Alias for backwards compatibility
using SpaccSchema = ResolveSchema;

// Mixture of Experts routing type
enum class MoERoutingType {
    None,       // No MoE (standard model)
    Soft,       // Soft gating - all experts contribute, weighted by gating scores
    TopK        // Sparse routing - only top-k experts activated per sample
};

// =============================================================================
// Advanced Architecture Types (v2.0)
// =============================================================================

// Encoder architecture type
enum class EncoderArchitecture {
    MLP,            // Default MLP-based encoder (current)
    FTTransformer,  // Feature Tokenizer + Transformer
    TabNet,         // Sequential attention with feature selection
    SAINT,          // Self-Attention + Inter-sample Attention
    TraitNet,       // Trait-based multi-species network
    GNN,            // Graph Neural Network
    ExcelFormer,    // Semi-permeable attention (FT-Transformer variant)
    HeterogeneousGNN // Heterogeneous GNN with typed message passing
};

// GNN type for graph-based encoder
enum class GNNType {
    GCN,        // Graph Convolutional Network
    GAT,        // Graph Attention Network
    GraphSAGE   // Sample and Aggregate
};

// Graph construction mode
enum class GraphConstructionMode {
    Spatial,        // k-NN based on coordinates
    Taxonomic,      // Based on taxonomic similarity
    CoOccurrence    // Based on species co-occurrence
};

// Trait-environment interaction mode
enum class TraitInteractionMode {
    Bilinear,   // Bilinear interaction
    MLP,        // MLP-based interaction
    Attention   // Attention-based interaction
};

// FT-Transformer configuration
struct FTTransformerConfig {
    int d_model = 192;              // Embedding dimension
    int n_heads = 8;                // Number of attention heads
    int n_layers = 3;               // Number of transformer layers
    float attention_dropout = 0.2f;
    float ffn_dropout = 0.1f;
    int ffn_multiplier = 4;         // FFN hidden = d_model * multiplier
    bool pre_norm = true;           // Pre-LN (stable) vs Post-LN
};

// TabNet configuration
struct TabNetConfig {
    int n_steps = 3;                // Number of decision steps
    int n_d = 64;                   // Decision embedding dimension
    int n_a = 64;                   // Attention embedding dimension
    float relaxation_factor = 1.5f; // Sparsity relaxation
    float sparsity_coefficient = 1e-3f;
    int virtual_batch_size = 128;   // Ghost batch norm size
    bool use_sparsemax = true;      // Use sparsemax vs entmax-1.5
};

// SAINT configuration
struct SAINTConfig {
    int d_model = 128;
    int n_heads = 8;
    int n_layers = 6;
    float attention_dropout = 0.1f;
    bool use_row_attention = true;  // Enable inter-sample attention
    bool use_contrastive_pretrain = false;
    float mixup_alpha = 0.4f;       // For MixUp augmentation
};

// GNN configuration
struct GNNConfig {
    GNNType gnn_type = GNNType::GAT;
    int n_layers = 3;
    int hidden_dim = 256;
    int n_heads = 4;                // For GAT
    int k_neighbors = 10;           // For graph construction
    GraphConstructionMode graph_mode = GraphConstructionMode::Spatial;
    float edge_dropout = 0.1f;
    bool use_edge_features = false;
};

// Trait-based network configuration
struct TraitNetConfig {
    int env_dim = 128;              // Environment encoding dimension
    int trait_dim = 64;             // Trait encoding dimension
    int interaction_dim = 256;      // Interaction layer dimension
    TraitInteractionMode interaction = TraitInteractionMode::Bilinear;
    bool shared_trait_encoder = true;
};

// ExcelFormer configuration
struct ExcelFormerConfig {
    int d_model = 192;
    int n_heads = 8;
    int n_layers = 3;
    float attention_dropout = 0.2f;
    int ffn_multiplier = 4;
    float importance_threshold = 0.5f;    // Features above this attend to all
    bool pre_norm = true;
};

// Heterogeneous GNN configuration
struct HeterogeneousGNNConfig {
    int hidden_dim = 128;               // Hidden dimension for message passing
    int output_dim = 64;                // Output species embedding dimension
    int n_layers = 3;                   // Number of message passing layers
    int n_edge_types = 3;               // co-occurrence, same-genus, same-family
    int n_heads = 4;                    // Attention heads for message aggregation
    float dropout = 0.1f;
    int k_cooccurrence = 20;            // Top-k co-occurring species per species
    float cooccurrence_threshold = 0.01f; // Min co-occurrence frequency for edges
    bool use_taxonomic_edges = true;    // Add same-genus/same-family edges
    bool use_cooccurrence_edges = true; // Add co-occurrence edges
};

// TabM (BatchEnsemble) configuration
struct TabMConfig {
    bool enabled = false;                    // Whether to use TabM instead of standard MLP
    int n_ensembles = 16;                    // Number of implicit ensemble members
    std::string aggregation = "mean";        // "mean" or "median" aggregation
};

// Parallel layer aggregation mode
enum class ParallelAggregation {
    Concat,     // Concatenate outputs (increases dim)
    Sum,        // Element-wise sum (requires same dims)
    Mean,       // Element-wise mean (requires same dims)
    Attention,  // Attention-weighted combination
    Gated       // Learned gating weights
};

// Single parallel branch configuration
struct ParallelBranchConfig {
    std::vector<int64_t> hidden_dims;           // MLP architecture for this branch
    ActivationType activation = ActivationType::GELU;
    NormLayerType normalization = NormLayerType::BatchNorm;
    float dropout = 0.3f;
    float branch_weight = 1.0f;                 // Weight for weighted sum (if applicable)
};

// Parallel layers configuration
struct ParallelLayersConfig {
    bool enabled = false;                       // Whether to use parallel layers
    std::vector<ParallelBranchConfig> branches; // Branch configurations
    ParallelAggregation aggregation = ParallelAggregation::Concat;
    int attention_heads = 4;                    // For Attention aggregation
    bool use_residual = true;                   // Add input to aggregated output
};

// Model configuration
struct ModelConfig {
    SpeciesEncodingMode species_encoding = SpeciesEncodingMode::Hash;
    bool uses_explicit_vector = false;  // For hash mode with selection="all"
    int hash_dim = kDefaultHashDim;
    int species_embed_dim = 32;
    int genus_emb_dim = 8;
    int family_emb_dim = 8;
    // Embedding dimension for each categorical-covariate column. Shared
    // across columns (one knob to tune). Must be > 0. Matches the value
    // stored on ResolveSchema after dataset construction.
    int categorical_embed_dim = 8;
    int top_k = 3;
    int top_k_species = 10;  // For embed mode
    int n_taxonomy_slots = 3;  // May be 2*top_k for top_bottom mode
    std::vector<int64_t> hidden_dims = {2048, 1024, 512, 256, 128, 64};
    float dropout = kDefaultDropout;

    // Mixture of Experts configuration
    MoERoutingType moe_routing = MoERoutingType::None;  // None = standard model
    int n_experts = 4;                                    // Number of expert networks
    std::vector<int64_t> expert_hidden_dims = {256, 128}; // Expert MLP architecture
    int moe_top_k = 2;                                    // For TopK routing: experts per sample
    float moe_noise_std = 0.1f;                          // Noise for load balancing in training
    float moe_aux_loss_weight = 0.01f;                   // Weight for auxiliary load balancing loss

    // Configurable architecture (activation, normalization, residuals)
    ActivationType activation = ActivationType::GELU;
    NormLayerType normalization = NormLayerType::BatchNorm;
    int norm_groups = kDefaultNormGroups;                 // For GroupNorm
    bool use_residual = false;                            // Enable residual connections
    float leaky_relu_slope = kDefaultLeakyReLUSlope;      // For LeakyReLU
    float elu_alpha = kDefaultELUAlpha;                   // For ELU

    // Multi-layer prediction heads
    std::vector<int64_t> head_hidden_dims = {};           // Empty = single linear (default)
    ActivationType head_activation = ActivationType::GELU;
    float head_dropout = 0.0f;                            // Dropout in prediction heads

    // Advanced architecture selection (v2.0)
    EncoderArchitecture encoder_architecture = EncoderArchitecture::MLP;

    // Architecture-specific configs
    FTTransformerConfig ft_transformer;
    TabNetConfig tabnet;
    SAINTConfig saint;
    GNNConfig gnn;
    TraitNetConfig trait_net;
    ExcelFormerConfig excelformer;
    HeterogeneousGNNConfig heterogeneous_gnn;

    // Parallel layers configuration
    ParallelLayersConfig parallel_layers;

    // TabM (BatchEnsemble) configuration
    TabMConfig tabm;

    // RankPool / Transformer shared
    float cover_dropout = 0.0f;

    // Transformer-specific
    int d_model = 128;
    int n_heads = 4;
    int n_attention_layers = 0;
    int transformer_ff_dim = 256;
    std::string transformer_pooling = "attention";
    float transformer_dropout = 0.1f;
};

// Training configuration
struct TrainConfig {
    int batch_size = kDefaultBatchSize;
    int max_epochs = kDefaultMaxEpochs;
    int patience = kDefaultPatience;
    float lr = kDefaultLearningRate;
    float weight_decay = kDefaultWeightDecay;
    std::pair<int, int> phase_boundaries = {kDefaultPhase1Epoch, kDefaultPhase2Epoch};
    LossConfigMode loss_config = LossConfigMode::Combined;
    torch::Device device = torch::kCPU;

    // Learning rate scheduling
    LRSchedulerType lr_scheduler = LRSchedulerType::None;
    int lr_step_size = 100;      // For StepLR: decay every N epochs
    float lr_gamma = 0.1f;       // For StepLR: multiply LR by gamma
    float lr_min = 1e-6f;        // For CosineAnnealing: minimum LR

    // Band accuracy thresholds for regression metrics
    std::vector<float> band_thresholds = {0.1f, 0.25f, 0.5f};

    // Checkpointing
    std::string checkpoint_dir;   // Directory for checkpoints (empty = disabled)
    int checkpoint_every = 0;     // Save checkpoint every N epochs (0 = only best)

    // Logging callback (defaults to stdout, use null_log to disable)
    LogCallback log = default_log;

    // Automatic Mixed Precision (AMP) - disabled by default for MLP models
    // AMP is most beneficial for transformers/convolutions; MLPs see minimal benefit
    bool use_amp = false;
    float amp_init_scale = 65536.0f;    // Initial gradient scale (2^16)
    float amp_growth_factor = 2.0f;     // Scale growth factor
    float amp_backoff_factor = 0.5f;    // Scale reduction on overflow
    int amp_growth_interval = 2000;     // Steps between scale increases

    // CUDA performance optimizations
    bool cudnn_benchmark = true;        // Auto-tune cuDNN algorithms (best for fixed input sizes)
    bool allow_tf32 = true;             // Allow TF32 on Ampere+ GPUs (faster matmuls, ~0.1% precision loss)

    // Fraction of GPU VRAM the PyTorch caching allocator may use on the
    // training device. 1.0 (default) lets dedicated training jobs on a solo
    // GPU use the full device. Pass an explicit lower value (e.g. 0.80) when
    // sharing the GPU with a desktop / GUI / other workloads — the Windows
    // WDDM driver spills overflowing VRAM allocations into shared system
    // memory, which hangs the whole desktop under load, so capping PyTorch's
    // allocator prevents that. Applied in Trainer::fit().
    float vram_fraction = 1.0f;

    // Smallest batch size the auto-halve-on-OOM loop in Trainer::fit() is
    // allowed to drop to. When training raises c10::OutOfMemoryError, the
    // trainer releases optimizer/scheduler/scaler state, empties the CUDA
    // caching allocator, halves batch_size, and restarts from epoch 0. If
    // halving would take batch_size below this floor, the original OOM is
    // rethrown as a std::runtime_error with the diagnostic context (original
    // bs, floor, point of failure). Default 1024 is conservative; lower
    // values are appropriate for memory-bound or sample-efficiency runs.
    int batch_size_floor = 1024;
};

// Batch of data for training/inference
struct ResolveBatch {
    torch::Tensor continuous;      // (batch, n_continuous)
    torch::Tensor genus_ids;       // (batch, n_taxonomy_slots) or empty
    torch::Tensor family_ids;      // (batch, n_taxonomy_slots) or empty
    torch::Tensor species_ids;     // (batch, top_k_species) for embed mode
    torch::Tensor species_vector;  // (batch, n_species) for sparse mode
    // Categorical covariates: (batch, n_categoricals) int64 with codes
    // produced by CategoricalVocab (0 = UNK). Empty when the schema
    // declares no categorical columns.
    torch::Tensor categorical_ids;
    // Pool-style encoder fields (rank_pool / transformer modes)
    torch::Tensor pool_genus_ids;  // (batch, max_species) or empty
    torch::Tensor pool_family_ids; // (batch, max_species) or empty
    torch::Tensor pool_weights;    // (batch, max_species) or empty
    torch::Tensor pool_mask;       // (batch, max_species) or empty
    torch::Tensor pool_has_cover;  // (batch,) or empty
    std::unordered_map<std::string, torch::Tensor> targets;  // target_name -> tensor

    ResolveBatch to(torch::Device device) const {
        ResolveBatch batch;
        batch.continuous = continuous.to(device);
        if (genus_ids.defined()) {
            batch.genus_ids = genus_ids.to(device);
        }
        if (family_ids.defined()) {
            batch.family_ids = family_ids.to(device);
        }
        if (species_ids.defined()) {
            batch.species_ids = species_ids.to(device);
        }
        if (species_vector.defined()) {
            batch.species_vector = species_vector.to(device);
        }
        if (categorical_ids.defined()) {
            batch.categorical_ids = categorical_ids.to(device);
        }
        if (pool_genus_ids.defined()) {
            batch.pool_genus_ids = pool_genus_ids.to(device);
        }
        if (pool_family_ids.defined()) {
            batch.pool_family_ids = pool_family_ids.to(device);
        }
        if (pool_weights.defined()) {
            batch.pool_weights = pool_weights.to(device);
        }
        if (pool_mask.defined()) {
            batch.pool_mask = pool_mask.to(device);
        }
        if (pool_has_cover.defined()) {
            batch.pool_has_cover = pool_has_cover.to(device);
        }
        for (const auto& [name, tensor] : targets) {
            batch.targets[name] = tensor.to(device);
        }
        return batch;
    }
};

// Alias for backwards compatibility
using SpaccBatch = ResolveBatch;

// Baseline comparison metrics for a single target
// Allows users to understand if the model is actually learning vs naive baselines
struct BaselineMetrics {
    // Regression baselines
    float baseline_mse = 0.0f;        // MSE if predicting global training mean
    float baseline_mae = 0.0f;        // MAE if predicting global training mean
    float model_mse = 0.0f;           // Model's test MSE
    float model_mae = 0.0f;           // Model's test MAE
    float skill_score = 0.0f;         // 1 - (model_mse / baseline_mse), higher is better
    float r_squared = 0.0f;           // Coefficient of determination

    // Classification baselines
    float baseline_accuracy = 0.0f;   // Accuracy if always predicting mode
    float model_accuracy = 0.0f;      // Model's test accuracy
    float accuracy_lift = 0.0f;       // model_accuracy - baseline_accuracy

    // Common
    float training_mean = 0.0f;       // Mean of training targets (for regression)
    int training_mode = -1;           // Mode class (for classification)
};

// Network diagnostics for detecting training issues
// These metrics help identify problems like dead neurons, gradient issues, etc.
struct LayerDiagnostics {
    std::string name;                 // Layer identifier (e.g., "hidden_0", "hidden_1")
    int64_t n_neurons = 0;            // Number of neurons in this layer
    int64_t n_dead = 0;               // Neurons with zero activation (dead ReLU)
    int64_t n_saturated = 0;          // Neurons always at max (saturated)
    float dead_fraction = 0.0f;       // Fraction of dead neurons
    float saturated_fraction = 0.0f;  // Fraction of saturated neurons
    float mean_activation = 0.0f;     // Mean activation across all neurons
    float std_activation = 0.0f;      // Std of activations (low = homogeneous = bad)
    float sparsity = 0.0f;            // Fraction of zero activations
};

struct NetworkDiagnostics {
    std::vector<LayerDiagnostics> layers;
    int64_t total_neurons = 0;
    int64_t total_dead = 0;
    int64_t total_saturated = 0;
    float overall_dead_fraction = 0.0f;
    float overall_saturated_fraction = 0.0f;
    bool has_issues = false;          // True if any layer has >10% dead/saturated
    std::string summary;              // Human-readable summary of issues
};

// Results from training
struct TrainResult {
    int best_epoch;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
    std::vector<float> train_loss_history;
    std::vector<float> test_loss_history;
    float train_time_seconds = 0.0f;
    int resumed_from_epoch = 0;

    // Baseline comparisons per target
    std::unordered_map<std::string, BaselineMetrics> baselines;

    // Network health diagnostics
    NetworkDiagnostics diagnostics;
};

// Results from cross-validation
struct CrossValidationResult {
    int n_folds = 0;

    // Aggregated metrics across folds (mean +/- std)
    std::unordered_map<std::string, std::unordered_map<std::string, float>> mean_metrics;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> std_metrics;

    // Per-fold results
    std::vector<TrainResult> fold_results;

    // Total time across all folds
    float total_time_seconds = 0.0f;
};

// Calibration data for classification targets
// Compares predicted probabilities vs actual frequencies
struct CalibrationBin {
    float bin_start = 0.0f;
    float bin_end = 0.0f;
    float mean_predicted_prob = 0.0f;  // Average predicted probability in bin
    float actual_frequency = 0.0f;      // Fraction of positives in bin
    int64_t count = 0;                  // Number of samples in bin
};

struct CalibrationResult {
    std::string target_name;
    int class_idx = -1;  // For multi-class: which class this is for (-1 = binary)
    std::vector<CalibrationBin> bins;
    float expected_calibration_error = 0.0f;  // ECE metric
    float max_calibration_error = 0.0f;       // MCE metric
};

// Residual analysis for regression targets
struct ResidualAnalysis {
    std::string target_name;
    std::vector<float> predictions;
    std::vector<float> actuals;
    std::vector<float> residuals;

    // Summary statistics
    float mean_residual = 0.0f;
    float std_residual = 0.0f;
    float skewness = 0.0f;
    float kurtosis = 0.0f;

    // Quantiles for residuals
    float q05 = 0.0f;  // 5th percentile
    float q25 = 0.0f;  // 25th percentile
    float q50 = 0.0f;  // Median
    float q75 = 0.0f;  // 75th percentile
    float q95 = 0.0f;  // 95th percentile
};

// Per-plot classification predictions on the held-out test fold. The
// regression-only ResidualAnalysis leaves classification targets without
// per-plot outputs (compute_residuals returns empty for them); this is the
// classification counterpart, so a saved checkpoint can be scored for
// per-class F1, confusion matrices, and top-k against the trainer's own test
// split. All tensors live on CPU with length n_test along dim 0.
struct ClassificationPredictions {
    std::string target_name;
    torch::Tensor predicted_classes;  // (n_test,) int64 argmax class codes
    torch::Tensor probabilities;      // (n_test, n_classes) float32 softmax rows
    torch::Tensor actuals;            // (n_test,) int64 ground-truth class codes
    // Ordered class vocabulary mirrored from the target's TargetConfig:
    // class_names[code] is the original CSV string for integer class `code`.
    // Empty when the classification column was already integer-coded.
    std::vector<std::string> class_names;
};

// Run metadata for reproducibility and provenance tracking
struct RunMetadata {
    std::string resolve_version = VERSION;
    std::string created_at;           // ISO 8601 timestamp
    std::string completed_at;         // ISO 8601 timestamp
    float train_time_seconds = 0.0f;
    int64_t n_plots_train = 0;
    int64_t n_plots_test = 0;
    int best_epoch = 0;
    int total_epochs = 0;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
};

// Predictions output
struct ResolvePredictions {
    std::unordered_map<std::string, torch::Tensor> predictions;
    std::unordered_map<std::string, torch::Tensor> targets;  // actual target values
    std::vector<std::string> plot_ids;
    torch::Tensor latent;    // optional latent representations
};

// Species record for encoding
struct SpeciesRecord {
    std::string species_id;
    std::string genus;
    std::string family;
    float abundance = 1.0f;
    std::string plot_id;
};

// Encoded species data (output of encoding process)
struct EncodedSpecies {
    torch::Tensor hash_embedding;   // (n_plots, hash_dim) for hash mode
    torch::Tensor genus_ids;        // (n_plots, n_taxonomy_slots)
    torch::Tensor family_ids;       // (n_plots, n_taxonomy_slots)
    torch::Tensor unknown_fraction; // (n_plots,)
    torch::Tensor unknown_count;    // (n_plots,)
    torch::Tensor species_vector;   // (n_plots, n_species) for sparse mode
    torch::Tensor species_ids;      // (n_plots, top_k_species) for embed mode
    std::vector<std::string> plot_ids;
};

// Taxonomy vocabulary for encoding genus/family names to IDs
class TaxonomyVocab {
public:
    TaxonomyVocab() = default;

    // Fit vocabulary from species records.
    //
    // IDs are assigned in sorted (alphabetical) order so the genus/family ->
    // ID mapping is a pure function of the SET of names, independent of the
    // order records arrive in. A first-appearance ordering made the IDs depend
    // on CSV row order: a checkpoint trained on one ordering and scored against
    // a differently-ordered rebuild (e.g. from_csv_with_schema in another
    // process) silently misaligned the genus/family embedding lookups. This
    // mirrors SpeciesVocab::from_records, which is already sorted. Index 0 is
    // reserved for unknown. See gcol33/resolve#5.
    void fit(const std::vector<SpeciesRecord>& records) {
        genus_to_idx_.clear();
        family_to_idx_.clear();

        genus_to_idx_["<UNK>"] = 0;
        family_to_idx_["<UNK>"] = 0;

        std::set<std::string> genera, families;
        for (const auto& rec : records) {
            if (!rec.genus.empty())  genera.insert(rec.genus);
            if (!rec.family.empty()) families.insert(rec.family);
        }
        for (const auto& g : genera) {
            genus_to_idx_[g] = static_cast<int64_t>(genus_to_idx_.size());
        }
        for (const auto& f : families) {
            family_to_idx_[f] = static_cast<int64_t>(family_to_idx_.size());
        }
    }

    // Encode genus name to ID
    [[nodiscard]] int64_t encode_genus(const std::string& genus) const noexcept {
        auto it = genus_to_idx_.find(genus);
        return it != genus_to_idx_.end() ? it->second : 0;
    }

    // Encode family name to ID
    [[nodiscard]] int64_t encode_family(const std::string& family) const noexcept {
        auto it = family_to_idx_.find(family);
        return it != family_to_idx_.end() ? it->second : 0;
    }

    [[nodiscard]] int64_t n_genera() const noexcept { return static_cast<int64_t>(genus_to_idx_.size()); }
    [[nodiscard]] int64_t n_families() const noexcept { return static_cast<int64_t>(family_to_idx_.size()); }

    // Accessors for serialization
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& genus_map() const noexcept { return genus_to_idx_; }
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& family_map() const noexcept { return family_to_idx_; }

    // Set from loaded data
    void set_genus_map(const std::unordered_map<std::string, int64_t>& m) { genus_to_idx_ = m; }
    void set_family_map(const std::unordered_map<std::string, int64_t>& m) { family_to_idx_ = m; }

    // Save vocabulary to archive (strings serialized as concatenated bytes with lengths)
    void save(torch::serialize::OutputArchive& archive, const std::string& prefix = "taxonomy_") const {
        // Build ordered lists from maps
        std::vector<std::string> genera(genus_to_idx_.size());
        for (const auto& [name, idx] : genus_to_idx_) {
            genera[idx] = name;
        }
        std::vector<std::string> families(family_to_idx_.size());
        for (const auto& [name, idx] : family_to_idx_) {
            families[idx] = name;
        }

        // Serialize genus vocab: lengths tensor + concatenated bytes tensor
        std::vector<int64_t> genus_lengths;
        std::vector<uint8_t> genus_bytes;
        for (const auto& s : genera) {
            genus_lengths.push_back(static_cast<int64_t>(s.size()));
            genus_bytes.insert(genus_bytes.end(), s.begin(), s.end());
        }
        archive.write(prefix + "genus_lengths", torch::tensor(genus_lengths));
        if (!genus_bytes.empty()) {
            archive.write(prefix + "genus_bytes", torch::from_blob(
                genus_bytes.data(), {static_cast<int64_t>(genus_bytes.size())}, torch::kUInt8).clone());
        } else {
            archive.write(prefix + "genus_bytes", torch::empty({0}, torch::kUInt8));
        }

        // Serialize family vocab
        std::vector<int64_t> family_lengths;
        std::vector<uint8_t> family_bytes;
        for (const auto& s : families) {
            family_lengths.push_back(static_cast<int64_t>(s.size()));
            family_bytes.insert(family_bytes.end(), s.begin(), s.end());
        }
        archive.write(prefix + "family_lengths", torch::tensor(family_lengths));
        if (!family_bytes.empty()) {
            archive.write(prefix + "family_bytes", torch::from_blob(
                family_bytes.data(), {static_cast<int64_t>(family_bytes.size())}, torch::kUInt8).clone());
        } else {
            archive.write(prefix + "family_bytes", torch::empty({0}, torch::kUInt8));
        }
    }

    // Load vocabulary from archive
    static TaxonomyVocab load(torch::serialize::InputArchive& archive, const std::string& prefix = "taxonomy_") {
        TaxonomyVocab vocab;

        // Load genus vocab
        torch::Tensor genus_lengths_t, genus_bytes_t;
        archive.read(prefix + "genus_lengths", genus_lengths_t);
        archive.read(prefix + "genus_bytes", genus_bytes_t);

        auto genus_lengths = genus_lengths_t.accessor<int64_t, 1>();
        auto genus_bytes_ptr = genus_bytes_t.data_ptr<uint8_t>();
        int64_t offset = 0;
        for (int64_t i = 0; i < genus_lengths_t.size(0); ++i) {
            std::string name(reinterpret_cast<const char*>(genus_bytes_ptr + offset), genus_lengths[i]);
            vocab.genus_to_idx_[name] = i;
            offset += genus_lengths[i];
        }

        // Load family vocab
        torch::Tensor family_lengths_t, family_bytes_t;
        archive.read(prefix + "family_lengths", family_lengths_t);
        archive.read(prefix + "family_bytes", family_bytes_t);

        auto family_lengths = family_lengths_t.accessor<int64_t, 1>();
        auto family_bytes_ptr = family_bytes_t.data_ptr<uint8_t>();
        offset = 0;
        for (int64_t i = 0; i < family_lengths_t.size(0); ++i) {
            std::string name(reinterpret_cast<const char*>(family_bytes_ptr + offset), family_lengths[i]);
            vocab.family_to_idx_[name] = i;
            offset += family_lengths[i];
        }

        return vocab;
    }

private:
    std::unordered_map<std::string, int64_t> genus_to_idx_;
    std::unordered_map<std::string, int64_t> family_to_idx_;
};

// Alias for backwards compatibility
using SpaccPredictions = ResolvePredictions;

} // namespace resolve
