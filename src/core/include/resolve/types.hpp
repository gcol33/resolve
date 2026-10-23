#pragma once

#include <torch/torch.h>
#include "resolve/string_list_archive.hpp"
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

inline constexpr const char* VERSION = "0.12.0";

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

// Gradient-norm clip threshold applied on every backward pass. Single source of
// truth for both training paths.
constexpr double kMaxGradNorm = 1.0;

// Initialization constants
constexpr float kBertInitStd = 0.02f;
constexpr float kAttentionMaskFill = -1e9f;
constexpr float kGATInitStd = 0.01f;

// Phase boundaries for phased loss
constexpr int kDefaultPhase1Epoch = 100;
constexpr int kDefaultPhase2Epoch = 300;

// Neighbourhood Components Analysis term (LossConfigMode::NCA). Shared by
// TrainConfig's nca_* knobs, NCATerm and NCALossImpl's constructor defaults, so
// the module and the loss preset cannot drift apart. See nca_objective in
// loss.hpp for what each one controls.
constexpr float kNCATemperature = 0.1f;
constexpr int   kNCANeighbors   = 32;
constexpr float kNCAWeight      = 0.1f;

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
    NCA        // Combined regression schedule + the Neighbourhood Components
               // Analysis term on classification targets (Goldberger et al.,
               // NIPS 2004; see nca_objective in loss.hpp)
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

// How a missing numeric covariate or coordinate enters the model.
enum class MissingValuePolicy {
    Zero,     // The value is read as 0.0 and nothing marks it as missing.
    Indicate  // The value is filled with the fitting fold's mean before
              // standardisation, and a 0/1 column beside the covariates marks
              // it: one per covariate, and one for the coordinate pair.
};

// What a training run does with a plot that carries no label for a target.
enum class TargetMissingPolicy {
    Drop,  // The plot is left out of the whole fit, for every target.
    Mask   // The plot is kept; it trains and is scored on the targets it is
           // labelled for and contributes nothing to this one. The dataset
           // holds the target_labels.hpp sentinel in its place. A plot labelled
           // for no target at all is still left out, since it trains nothing.
};

// How Trainer::prepare_data chooses the held-out fold that early stopping,
// the reported metrics and the baselines read.
enum class HoldOutMode {
    Random,   // A fraction of the plots, drawn by a seeded shuffle.
    Explicit  // The plots the caller names: a withheld spatial block or
              // database, or the fold a published protocol selects on.
};

// How a model suite combines the predictions of a target's members (suite.hpp).
enum class SuiteCombine {
    Mean,          // arithmetic mean on the target's own scale
    Vote,          // majority of the members' classes, ties to the higher mean probability
    CircularMean,  // mean of the members' bearings as unit vectors
    SphericalMean  // mean of the members' latitude / longitude as unit vectors on the sphere
};

// Whether a suite target is released for general use or with a stated limit.
enum class SuiteTargetStatus {
    Released,
    Experimental
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
    TaskType task = TaskType::Regression;
    TransformType transform = TransformType::None;
    int num_classes = 0;  // For classification
    float weight = 1.0f;  // Loss weight in multi-task
    std::vector<float> class_weights = {};  // Optional class weights for imbalanced classification
    // Ordered class vocabulary for classification targets. class_names[i] is
    // the original CSV string that encodes to int code i. Empty for
    // regression. Populated by ResolveDataset::load_header_data when it
    // factorizes a string-coded classification column (e.g. EUNIS "M","N",
    // "P",...) into int64 codes. Persisted in the checkpoint so the
    // Predictor can map predicted codes back to human-readable class names.
    // Both vector members carry an explicit default so a brace-initialization
    // that stops after `weight` -- the common form at the call sites -- is not
    // a missing-initializer under -Wextra.
    std::vector<std::string> class_names = {};
    // What the run did with a plot unlabelled for this target. Persisted in
    // the checkpoint; a checkpoint written before the field existed loads as
    // Drop, which is what its loader did.
    TargetMissingPolicy missing = TargetMissingPolicy::Drop;
};

// Schema information for a dataset
struct ResolveSchema {
    int64_t n_plots = 0;
    int64_t n_species = 0;          // Number of unique species
    int64_t n_species_vocab = 0;    // Size of species vocabulary (for embed/sparse modes)
    bool has_coordinates = false;   // Default false, set true when coords found
    bool has_abundance = false;
    bool has_taxonomy = false;
    // Genus / family vocabulary sizes, the reserved <UNK> code 0 included: the
    // heights of the taxonomy embedding tables. 0 without taxonomy.
    int64_t n_genera = 0;
    int64_t n_families = 0;
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
    // DatasetConfig::pool_species_cap, whose sentinels are 0 = no cap (pad to
    // the global per-plot max), -1 = auto p99, >0 = manual cap. After a
    // rank-pool / transformer load the dataset overwrites this with the
    // RESOLVED width, so a checkpoint always carries a concrete >0 cap.
    int pool_weighting = 2;    // PoolWeighting::Log1p
    int pool_species_cap = 0;  // 0 = no cap, -x = p(100 - x), >0 = manual cap

    // Remaining DatasetConfig knobs that shape the encoded tensors but are not
    // recoverable from ModelConfig. Persisted so an inference-side
    // DatasetConfig can be rebuilt in full from the checkpoint rather than
    // silently reverting to the struct defaults: top_k_species and
    // track_unknown_count change tensor widths (loud shape errors), while
    // selection / representation / normalization / aggregation / use_taxonomy
    // change values silently. Read back through
    // dataset_config_from_checkpoint() (dataset.hpp), the single place that
    // reassembles a DatasetConfig from a checkpoint.
    int top_k_species = 10;
    // The selection the run APPLIED, not the one it was configured with: a
    // rank_pool / transformer / sparse dataset with no species_budget encodes
    // every record, which is All. See effective_selection().
    SelectionMode selection = SelectionMode::Top;
    // Per-plot species budget for the rank_pool / transformer / sparse
    // encodings; 0 = no budget. Mirrors DatasetConfig::species_budget.
    int species_budget = 0;
    RepresentationMode representation = RepresentationMode::Abundance;
    NormalizationMode normalization = NormalizationMode::Raw;
    AggregationMode aggregation = AggregationMode::Abundance;
    bool use_taxonomy = true;

    // How missing covariates and coordinates enter the continuous block.
    // Mirrors DatasetConfig::missing_values, which a dataset load copies here.
    // Zero for a schema assembled by hand, whose caller lays out the block
    // itself, and for a checkpoint written before the field existed, which is
    // what those models were trained under.
    MissingValuePolicy missing_values = MissingValuePolicy::Zero;

    // The abundance a recorded 0 was read at. Mirrors
    // DatasetConfig::zero_abundance_as; 0 (a 0 stays 0) for a checkpoint
    // written before the field existed, which is what its loader did.
    float zero_abundance_as = 0.0f;

    // Ordered species / genus / family vocabularies fitted at training time.
    // Element i is the name that encodes to integer code i, and
    // index 0 is always the reserved "<UNK>" slot, so the vector length equals
    // n_species_vocab / n_genera / n_families respectively.
    //
    // Every non-hash encoder indexes an embedding table with a code that is a
    // function of the FILE the dataset was built from: the species vocab is
    // frequency-ranked and the taxonomy vocab is a function of the name set, so
    // a checkpoint carrying only the vocabulary SIZES lets an inference run
    // re-fit different codes and look up the wrong embedding rows -- wrong
    // predictions, no error. Carrying the names makes the mapping recoverable
    // from the checkpoint alone (ResolveDataset::from_csv_with_schema(const
    // ResolveSchema&) builds a dataset directly in this namespace) and makes a
    // mismatch detectable (Predictor::predict rejects a dataset whose vocab is
    // not the model's).
    //
    // Empty on a checkpoint that stores only the sizes; the size-only guard
    // applies then, and the loaders that need the names throw.
    std::vector<std::string> species_vocab;
    std::vector<std::string> genus_vocab;
    std::vector<std::string> family_vocab;

    // Helper: true if this schema has categorical covariates configured.
    [[nodiscard]] bool has_categoricals() const noexcept {
        return !categorical_names.empty();
    }
    // Helper: true if the fitted species vocabulary travelled with the schema.
    // False for a checkpoint that stores only the vocabulary sizes.
    [[nodiscard]] bool has_species_vocab() const noexcept {
        return !species_vocab.empty();
    }
    // Helper: true if either taxonomy vocabulary travelled with the schema.
    // A family-only dataset has an empty genus vocab and vice versa.
    [[nodiscard]] bool has_taxonomy_vocab() const noexcept {
        return !genus_vocab.empty() || !family_vocab.empty();
    }
    // Helper: how many categorical columns.
    [[nodiscard]] int64_t n_categoricals() const noexcept {
        return static_cast<int64_t>(categorical_names.size());
    }
    // Helper: total width the CategoricalEmbedder concatenates into `continuous`
    // (n_columns * embed_dim), matching CategoricalEmbedderImpl::output_dim().
    // Single source of truth for the fused categorical width; the model builds
    // the embedder and the adapter must size its input for the same width.
    [[nodiscard]] int64_t categorical_embed_width() const noexcept {
        return has_categoricals()
            ? static_cast<int64_t>(categorical_vocab_sizes.size()) * categorical_embed_dim
            : 0;
    }
    // Helper: how many missingness columns the continuous block carries -- one
    // for the coordinate pair and one per covariate under Indicate, none under
    // Zero. Single source of truth for every place that sizes the block.
    [[nodiscard]] int64_t missing_flag_width() const noexcept {
        if (missing_values != MissingValuePolicy::Indicate) return 0;
        return (has_coordinates ? 1 : 0) + static_cast<int64_t>(covariate_names.size());
    }
};

// Mixture of Experts routing type
enum class MoERoutingType {
    None,       // No MoE (standard model)
    Soft,       // Soft gating - all experts contribute, weighted by gating scores
    TopK        // Sparse routing - only top-k experts activated per sample
};

// Where the mixture of experts sits relative to the encoder.
//
// Tail - the experts ARE the encoder's last layers: the backbone MLP is built
//        over hidden_dims minus its final stage, and the mixture projects that
//        backbone output to hidden_dims.back(), which becomes the latent.
//        Capacity moves into the experts instead of being stacked on top.
//        Open to every species encoding (hash, embed, sparse, rank_pool,
//        transformer), all of which end in an MLP tail the mixture can replace.
// Post - the encoder produces its latent as usual and the mixture maps that
//        latent to one of the same width. This is the only placement open to an
//        encoder with no MLP tail to replace: the adapter architectures
//        (FT-Transformer, TabNet, SAINT, GNN, ExcelFormer, HeterogeneousGNN)
//        and TraitNet.
enum class MoEPlacement {
    Tail,
    Post
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

// What the plot-level GNN measures neighbourhood by. Every mode builds the same
// k-nearest-neighbour graph over the plots in the forward pass and differs only
// in the features it measures distance on. A mode whose features the dataset
// does not carry is refused at forward time rather than silently substituted.
enum class GraphConstructionMode {
    Spatial,        // k-NN by Euclidean distance between coordinates
    Taxonomic,      // k-NN by cosine similarity of genus/family composition
    CoOccurrence    // k-NN by cosine similarity of the species vector
};

// How TraitNet combines the environment encoding with a species' trait
// encoding into the per-species representation its head reads.
enum class TraitInteractionMode {
    Bilinear,   // A learned bilinear form of the two
    MLP,        // The two side by side, projected once
    Attention   // The environment queries the species, scaling its value vector
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
    // Simplex mapping every decision step's attentive transformer applies to
    // its feature logits. true = sparsemax (alpha = 2; Arik & Pfister's
    // default); false = 1.5-entmax (Peters et al. 2019, Eq. 13), which is still
    // sparse but keeps a strictly larger mask support on the same logits.
    // Reaches TabNetEncoder -> TabNetStep::attentive_forward via
    // ModelAdapter's TabNet branch, and round-trips through the checkpoint key
    // "tabnet_use_sparsemax".
    bool use_sparsemax = true;
};

// SAINT configuration
struct SAINTConfig {
    int d_model = 128;
    int n_heads = 8;
    int n_layers = 6;
    float attention_dropout = 0.1f;
    bool use_row_attention = true;  // Enable inter-sample attention
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
    // Carry each edge's similarity as its adjacency weight instead of a plain
    // 1, so message passing weighs a near neighbour above a far one. The
    // spatial graph weighs by a Gaussian kernel of the distance, the
    // composition graphs by the cosine similarity itself.
    bool use_edge_features = false;
};

// Trait-based network configuration
struct TraitNetConfig {
    int env_dim = 128;              // Environment encoding dimension
    int trait_dim = 64;             // Trait encoding dimension
    // Width of the environment-trait combination the head reads. 0 takes the
    // environment encoding's own width.
    int interaction_dim = 256;
    // How the environment and a species' traits are combined.
    TraitInteractionMode interaction = TraitInteractionMode::Bilinear;
    // One trait encoder applied to every species, or one set of weights per
    // species -- which costs n_species times the trait-encoder parameters, so
    // it suits a small species pool rather than a regional vocabulary.
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
    MoEPlacement moe_placement = MoEPlacement::Tail;     // Experts as the encoder tail, or after the latent

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

    // Keep the species, genus and family tables at their initialisation, so the
    // composition representation is fixed rather than learned
    // (ResolveModelImpl::composition_parameters). Encoder-architecture MLP only.
    bool freeze_composition = false;

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
    // Epochs to run of the max_epochs schedule. 0 trains under early stopping
    // on the held-out fold and returns the best epoch's weights. A positive
    // value runs exactly that many epochs with no early stopping and returns
    // the final weights; the learning-rate schedule is still laid out over
    // max_epochs, so the first fixed_epochs epochs follow the trajectory of a
    // full-length run. It is the one mode that trains without a held-out fold
    // (test_size = 0), which is how a model is refitted on every labelled plot
    // for a duration chosen beforehand.
    int fixed_epochs = 0;
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

    // Band accuracy thresholds for regression metrics. Reporting only: each
    // threshold t produces a "band_<100t>" metric (band_10, band_25, band_50).
    std::vector<float> band_thresholds = {0.1f, 0.25f, 0.5f};

    // Tolerance band the phase-3 PhasedLoss penalty optimizes toward -- the loss
    // penalizes predictions whose ratio to the target falls outside [1 - t, 1 + t].
    // Distinct from band_thresholds above (which only selects which
    // band accuracies are *reported*): the loss optimizes a single band, and it is
    // this knob, not the reporting vector, that changes what training minimizes.
    float band_threshold = 0.25f;

    // Neighbourhood Components Analysis term on classification targets. Active
    // only under loss_config == LossConfigMode::NCA, which is the one preset
    // that enables it; inert under MAE / SMAPE / Combined.
    //   nca_temperature sets the scale of the stochastic-neighbour softmax,
    //     i.e. the effective number of neighbours each sample spreads over.
    //   nca_neighbors truncates a sample's neighbour set to its most similar
    //     in-batch samples; <= 0 (or >= batch - 1) keeps the whole batch.
    //   nca_weight scales the term against the cross-entropy it is added to.
    float nca_temperature = kNCATemperature;
    int nca_neighbors = kNCANeighbors;
    float nca_weight = kNCAWeight;

    // Checkpointing. With a directory, fit() writes best.pt at every new best
    // epoch, checkpoint_<N>.pt every checkpoint_every epochs (0 = best only),
    // progress.json after every epoch and checkpoint.pt at the end. best.pt and
    // checkpoint_<N>.pt carry the run's progress, which Trainer::resume
    // continues from.
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

// The held-out fold scored against a naive baseline, for a single target:
// whether the model learns more than the training mean (regression) or the
// training mode (classification). Regression values are in the target's
// original units, and the model's scores are the ones final_metrics reports
// (model_mae == final_metrics[target]["mae"], r_squared == ["r2"],
// model_accuracy == ["accuracy"]).
struct BaselineMetrics {
    // Regression baselines
    float baseline_mse = 0.0f;        // MSE if predicting the training mean
    float baseline_mae = 0.0f;        // MAE if predicting the training mean
    float model_mse = 0.0f;           // Model's test MSE
    float model_mae = 0.0f;           // Model's test MAE
    float skill_score = 0.0f;         // 1 - (model_mse / baseline_mse), higher is better
    float r_squared = 0.0f;           // Coefficient of determination

    // Classification baselines
    float baseline_accuracy = 0.0f;   // Accuracy if always predicting mode
    float model_accuracy = 0.0f;      // Model's test accuracy
    float accuracy_lift = 0.0f;       // model_accuracy - baseline_accuracy

    // Common
    float training_mean = 0.0f;       // Mean of the training targets, original units (regression)
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
    int best_epoch = 0;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
    std::vector<float> train_loss_history;
    std::vector<float> test_loss_history;

    // The two histories above split by target. Per epoch, a target's entry is
    // the term it adds to that epoch's total: its weighted loss, averaged over
    // the epoch's batches the way the total is (a batch with no labelled row
    // for a masked target adds 0), and on the held-out fold its weighted loss
    // under the phase-invariant selection objective. Keyed by target name, one
    // value per epoch; the test map is empty when nothing is held out.
    //
    // The MoE load-balancing term belongs to no target and is recorded in
    // train_moe_aux_loss_history (empty without MoE; the held-out loss never
    // carries it). Per epoch, the train entries plus that term add up to
    // train_loss_history, and the test entries to test_loss_history, within
    // float rounding.
    std::unordered_map<std::string, std::vector<float>> train_loss_history_by_target;
    std::unordered_map<std::string, std::vector<float>> test_loss_history_by_target;
    std::vector<float> train_moe_aux_loss_history;

    // Training time of the whole run, a resumed run's earlier epochs included.
    float train_time_seconds = 0.0f;
    // The epoch a Trainer::resume continued the run from (its histories hold
    // the earlier epochs too); 0 for a run fit() trained from the start.
    int resumed_from_epoch = 0;

    // The batch size this run actually trained at, i.e. the value after any
    // CUDA auto-halve-on-OOM retries. Equal to the requested TrainConfig
    // batch_size on a clean run; smaller when the retry fired.
    //
    // fit() restores config_.batch_size to the requested value before
    // returning -- otherwise a later cross-validation fold would silently
    // train at the reduced size -- so the effective value is not readable off
    // the config afterwards. It is copied here (and onto the trainer's
    // effective_batch_size() accessor, from the same member) at fit() exit.
    int effective_batch_size = 0;

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
    // Position of each row in the test fold (Trainer::test_plot_ids()). The
    // rows are the plots labelled for the target, which is every test plot
    // unless the target's missing labels are masked.
    std::vector<int64_t> test_rows;

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
// split. All tensors live on CPU with one row per test plot labelled for the
// target (n_labelled; every test plot unless its missing labels are masked).
struct ClassificationPredictions {
    std::string target_name;
    torch::Tensor predicted_classes;  // (n_labelled,) int64 argmax class codes
    torch::Tensor probabilities;      // (n_labelled, n_classes) float32 softmax rows
    torch::Tensor actuals;            // (n_labelled,) int64 ground-truth class codes
    torch::Tensor test_rows;          // (n_labelled,) int64 position in the test fold
                                      // (Trainer::test_plot_ids()) of each row
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
    // How the held-out fold was chosen. Random: `test_size` of the plots,
    // drawn with `split_seed`. Explicit: the plots the caller named, and
    // test_size is 0. split_seed also seeds the per-epoch training shuffle.
    HoldOutMode hold_out = HoldOutMode::Random;
    float test_size = 0.0f;
    int split_seed = 0;
    int best_epoch = 0;
    int total_epochs = 0;
    std::unordered_map<std::string, std::unordered_map<std::string, float>> final_metrics;
};

// Predictions output
struct ResolvePredictions {
    // One entry per target. A regression target holds a float `(n)` tensor on
    // the original scale; a classification target holds the int64 `(n)` argmax
    // class code.
    std::unordered_map<std::string, torch::Tensor> predictions;
    // Softmax rows behind each classification target: float `(n, n_classes)`,
    // each row summing to one, column j = P(class code j). Regression targets
    // have no entry.
    std::unordered_map<std::string, torch::Tensor> probabilities;
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

// Taxonomy vocabulary for encoding genus/family names to IDs
class TaxonomyVocab {
public:
    TaxonomyVocab() = default;

    // Fit vocabulary from species records.
    //
    // IDs are assigned in sorted (alphabetical) order so the genus/family ->
    // ID mapping is a pure function of the SET of names, independent of the
    // order records arrive in, so a checkpoint scored against a
    // differently-ordered rebuild (e.g. from_csv_with_schema in another
    // process) reads the same genus/family embedding rows. Index 0 is
    // reserved for unknown.
    void fit(const std::vector<SpeciesRecord>& records) {
        std::set<std::string> genera, families;
        for (const auto& rec : records) {
            if (!rec.genus.empty())  genera.insert(rec.genus);
            if (!rec.family.empty()) families.insert(rec.family);
        }
        fit(genera, families);
    }

    // Fit from the sets of genus and family names (empty names excluded).
    void fit(const std::set<std::string>& genera, const std::set<std::string>& families) {
        genus_to_idx_.clear();
        family_to_idx_.clear();

        genus_to_idx_["<UNK>"] = 0;
        family_to_idx_["<UNK>"] = 0;

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

    [[nodiscard]] const std::unordered_map<std::string, int64_t>& genus_map() const noexcept { return genus_to_idx_; }
    [[nodiscard]] const std::unordered_map<std::string, int64_t>& family_map() const noexcept { return family_to_idx_; }

    // Ordered views of the two maps: element i is the name that encodes to
    // integer code i, with index 0 the reserved "<UNK>" slot. This is the
    // canonical serialization form -- save() below writes exactly these, and
    // ResolveSchema::genus_vocab / family_vocab carry them into the checkpoint
    // so an inference-side dataset can be rebuilt in the model's ID namespace.
    [[nodiscard]] std::vector<std::string> genus_names() const {
        return ordered_names(genus_to_idx_);
    }
    [[nodiscard]] std::vector<std::string> family_names() const {
        return ordered_names(family_to_idx_);
    }

    // Inverse of genus_names()/family_names(): rebuild the encoder from the two
    // ordered name lists (index = code). An empty list yields an empty map,
    // which encode_genus/encode_family already treat as "everything is UNK".
    [[nodiscard]] static TaxonomyVocab from_ordered(
        const std::vector<std::string>& genus_names,
        const std::vector<std::string>& family_names
    ) {
        TaxonomyVocab vocab;
        for (size_t i = 0; i < genus_names.size(); ++i) {
            vocab.genus_to_idx_[genus_names[i]] = static_cast<int64_t>(i);
        }
        for (size_t i = 0; i < family_names.size(); ++i) {
            vocab.family_to_idx_[family_names[i]] = static_cast<int64_t>(i);
        }
        return vocab;
    }

    // Writes the two ordered name lists (index = code) under
    // "<prefix>genus" / "<prefix>family" (string_list_archive.hpp).
    void save(torch::serialize::OutputArchive& archive, const std::string& prefix = "taxonomy_") const {
        write_string_list(archive, prefix + "genus", genus_names());
        write_string_list(archive, prefix + "family", family_names());
    }

    static TaxonomyVocab load(torch::serialize::InputArchive& archive, const std::string& prefix = "taxonomy_") {
        return from_ordered(read_string_list(archive, prefix + "genus"),
                            read_string_list(archive, prefix + "family"));
    }

private:
    // Invert a {name -> code} map into a code-indexed name list. Codes are
    // dense 0..N-1 by construction (fit / from_ordered / load all assign them
    // that way); a sparse map would leave holes as empty strings.
    [[nodiscard]] static std::vector<std::string> ordered_names(
        const std::unordered_map<std::string, int64_t>& to_idx
    ) {
        std::vector<std::string> names(to_idx.size());
        for (const auto& [name, idx] : to_idx) {
            if (idx >= 0 && idx < static_cast<int64_t>(names.size())) {
                names[static_cast<size_t>(idx)] = name;
            }
        }
        return names;
    }

    std::unordered_map<std::string, int64_t> genus_to_idx_;
    std::unordered_map<std::string, int64_t> family_to_idx_;
};

} // namespace resolve
