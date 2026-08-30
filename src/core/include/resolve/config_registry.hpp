#pragma once

// Field registries for the engine's configuration structs.
//
// A configuration field used to be spelled out in five places that shared no
// source of truth: the struct, the checkpoint writer, the checkpoint reader, the
// C-ABI value-tree parser/emitter, and the nanobind binding -- plus the JSON
// sidecar and `resolve info` for the ones those print. Adding a field meant
// finding all of them, and missing one was silent (issues #37, #38, #91, #102
// were each an instance of exactly that).
//
// Here each struct gets ONE list. A row names the member and the checkpoint key
// it serializes under; the C-ABI value-tree key, the Python attribute name, the
// JSON key and the `resolve info` label are the member name itself. Every
// consumer is a visitor over that list:
//
//   save_model_config / load_model_config, save_train_config /
//   load_train_config    -> archive write / read visitors (checkpoint.cpp)
//   parse_*_config, *_config_to_value
//                        -> value-tree read / write visitors (resolve_capi.cpp)
//   register_types       -> def_rw visitor (python/src/bindings_types.cpp)
//   write_metadata_json  -> JSON print visitor (checkpoint.cpp)
//   info_command         -> console print visitor (cli/info_cmd.cpp)
//
// Two things are then guaranteed at COMPILE time, not by review:
//
//   1. A member added to a struct without a row fails the build. Each
//      RESOLVE_DEFINE_FIELD_REGISTRY emits a static_assert comparing the row
//      count against the struct's aggregate arity.
//   2. A row whose type no visitor handles fails the build. Every visitor
//      dispatches with `if constexpr` over a closed set of field types and
//      static_asserts on the fallthrough.
//
// Adding a field is therefore one struct member plus one row, and the compiler
// finds every consumer that has to say something about it.
//
// The checkpoint key column is authoritative and must stay byte-identical to
// what earlier releases wrote -- a renamed key silently drops the field on load
// of an existing checkpoint. The registry supplies those spellings; it does not
// get to change them.

#include "resolve/types.hpp"
#include "resolve/dataset.hpp"
#include "resolve/enum_names.hpp"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace resolve {

// ============================================================================
// Compile-time arity probe
// ============================================================================

namespace registry_detail {

// Converts to any type, so a brace-init probe can supply one initializer per
// aggregate member without naming the member types. Declared and never defined:
// every use sits inside decltype, which is unevaluated.
struct AnyInit {
    template <typename T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, AnyInit>>>
    constexpr operator T() const;  // NOLINT(google-explicit-constructor)
};

template <std::size_t>
using AnyInitAt = AnyInit;

template <typename T, typename... A>
constexpr auto brace_init_ok(int) -> decltype(T{std::declval<A>()...}, true) {
    return true;
}
template <typename T, typename... A>
constexpr bool brace_init_ok(...) {
    return false;
}

template <typename T, std::size_t... I>
constexpr bool brace_init_with(std::index_sequence<I...>) {
    return brace_init_ok<T, AnyInitAt<I>...>(0);
}

// Number of direct members of an aggregate. Probes brace initialization with an
// increasing number of universal initializers; the first count that fails to
// compile is one past the member count. Each member accepts an AnyInit directly
// (it converts to anything), so brace elision never folds a nested aggregate's
// members into the outer count -- the number is the DIRECT member count.
template <typename T, std::size_t N = 0>
constexpr std::size_t aggregate_field_count() {
    if constexpr (brace_init_with<T>(std::make_index_sequence<N + 1>{})) {
        return aggregate_field_count<T, N + 1>();
    } else {
        return N;
    }
}

// Dependent false, so a visitor's `else static_assert(...)` fallthrough only
// fires for a type that actually reaches it.
template <typename>
inline constexpr bool always_false = false;

}  // namespace registry_detail

// True for a struct that has a field registry below, i.e. one a visitor can
// recurse into. Specialized by RESOLVE_DEFINE_FIELD_REGISTRY.
template <typename T>
inline constexpr bool is_registered_config_v = false;

// ============================================================================
// Registry definition machinery
// ============================================================================

// One row -> one visitor call. `visitor(member_name, checkpoint_key, member)`.
#define RESOLVE_FIELD_VISIT(member, key) visitor(#member, key, cfg.member);
#define RESOLVE_FIELD_COUNT(member, key) +1

// Emit the visitor entry points, the row count, and the arity guard for one
// struct. `LIST` is the struct's RESOLVE_*_FIELDS macro, passed by name.
#define RESOLVE_DEFINE_FIELD_REGISTRY(Struct, LIST)                            \
    template <typename V>                                                      \
    void for_each_field(Struct& cfg, V&& visitor) {                            \
        LIST(RESOLVE_FIELD_VISIT)                                              \
    }                                                                          \
    template <typename V>                                                      \
    void for_each_field(const Struct& cfg, V&& visitor) {                      \
        LIST(RESOLVE_FIELD_VISIT)                                              \
    }                                                                          \
    inline constexpr std::size_t field_registry_size(const Struct*) {          \
        return 0 LIST(RESOLVE_FIELD_COUNT);                                    \
    }                                                                          \
    template <>                                                                \
    inline constexpr bool is_registered_config_v<Struct> = true;               \
    static_assert(                                                             \
        (0 LIST(RESOLVE_FIELD_COUNT)) ==                                       \
            registry_detail::aggregate_field_count<Struct>(),                  \
        #Struct " has a member with no row in its field registry (or a row "   \
        "with no member). Every field appears exactly once in the "            \
        "RESOLVE_*_FIELDS list for " #Struct "; that list drives checkpoint "  \
        "save/load, the C-ABI value tree, the Python bindings, the JSON "      \
        "sidecar and `resolve info`.");

// ============================================================================
// Architecture sub-config registries
// ============================================================================
//
// The checkpoint namespace is flat, so each sub-config's keys carry a prefix
// ("ft_", "tabnet_", "excel_", ...). Those spellings predate this file and are
// reproduced verbatim.

#define RESOLVE_FT_TRANSFORMER_CONFIG_FIELDS(F) \
    F(d_model,           "ft_d_model")          \
    F(n_heads,           "ft_n_heads")          \
    F(n_layers,          "ft_n_layers")         \
    F(attention_dropout, "ft_attention_dropout") \
    F(ffn_dropout,       "ft_ffn_dropout")      \
    F(ffn_multiplier,    "ft_ffn_multiplier")   \
    F(pre_norm,          "ft_pre_norm")

RESOLVE_DEFINE_FIELD_REGISTRY(FTTransformerConfig, RESOLVE_FT_TRANSFORMER_CONFIG_FIELDS)

#define RESOLVE_TABNET_CONFIG_FIELDS(F)                    \
    F(n_steps,              "tabnet_n_steps")              \
    F(n_d,                  "tabnet_n_d")                  \
    F(n_a,                  "tabnet_n_a")                  \
    F(relaxation_factor,    "tabnet_relaxation_factor")    \
    F(sparsity_coefficient, "tabnet_sparsity_coefficient") \
    F(virtual_batch_size,   "tabnet_virtual_batch_size")   \
    F(use_sparsemax,        "tabnet_use_sparsemax")

RESOLVE_DEFINE_FIELD_REGISTRY(TabNetConfig, RESOLVE_TABNET_CONFIG_FIELDS)

#define RESOLVE_SAINT_CONFIG_FIELDS(F)                             \
    F(d_model,                  "saint_d_model")                   \
    F(n_heads,                  "saint_n_heads")                   \
    F(n_layers,                 "saint_n_layers")                  \
    F(attention_dropout,        "saint_attention_dropout")         \
    F(use_row_attention,        "saint_use_row_attention")         \
    F(use_contrastive_pretrain, "saint_use_contrastive_pretrain")  \
    F(mixup_alpha,              "saint_mixup_alpha")

RESOLVE_DEFINE_FIELD_REGISTRY(SAINTConfig, RESOLVE_SAINT_CONFIG_FIELDS)

// gnn_type keeps the un-prefixed spelling "gnn_type" it has always had; the
// member name already carries the family.
#define RESOLVE_GNN_CONFIG_FIELDS(F)             \
    F(gnn_type,          "gnn_type")             \
    F(n_layers,          "gnn_n_layers")         \
    F(hidden_dim,        "gnn_hidden_dim")       \
    F(n_heads,           "gnn_n_heads")          \
    F(k_neighbors,       "gnn_k_neighbors")      \
    F(graph_mode,        "gnn_graph_mode")       \
    F(edge_dropout,      "gnn_edge_dropout")     \
    F(use_edge_features, "gnn_use_edge_features")

RESOLVE_DEFINE_FIELD_REGISTRY(GNNConfig, RESOLVE_GNN_CONFIG_FIELDS)

#define RESOLVE_TRAIT_NET_CONFIG_FIELDS(F)                  \
    F(env_dim,              "trait_env_dim")                \
    F(trait_dim,            "trait_trait_dim")              \
    F(interaction_dim,      "trait_interaction_dim")        \
    F(interaction,          "trait_interaction")            \
    F(shared_trait_encoder, "trait_shared_trait_encoder")

RESOLVE_DEFINE_FIELD_REGISTRY(TraitNetConfig, RESOLVE_TRAIT_NET_CONFIG_FIELDS)

#define RESOLVE_EXCELFORMER_CONFIG_FIELDS(F)                 \
    F(d_model,              "excel_d_model")                 \
    F(n_heads,              "excel_n_heads")                 \
    F(n_layers,             "excel_n_layers")                \
    F(attention_dropout,    "excel_attention_dropout")       \
    F(ffn_multiplier,       "excel_ffn_multiplier")          \
    F(importance_threshold, "excel_importance_threshold")    \
    F(pre_norm,             "excel_pre_norm")

RESOLVE_DEFINE_FIELD_REGISTRY(ExcelFormerConfig, RESOLVE_EXCELFORMER_CONFIG_FIELDS)

#define RESOLVE_HETEROGENEOUS_GNN_CONFIG_FIELDS(F)                    \
    F(hidden_dim,             "hgnn_hidden_dim")                      \
    F(output_dim,             "hgnn_output_dim")                      \
    F(n_layers,               "hgnn_n_layers")                        \
    F(n_edge_types,           "hgnn_n_edge_types")                    \
    F(n_heads,                "hgnn_n_heads")                         \
    F(dropout,                "hgnn_dropout")                         \
    F(k_cooccurrence,         "hgnn_k_cooccurrence")                  \
    F(cooccurrence_threshold, "hgnn_cooccurrence_threshold")          \
    F(use_taxonomic_edges,    "hgnn_use_taxonomic_edges")             \
    F(use_cooccurrence_edges, "hgnn_use_cooccurrence_edges")

RESOLVE_DEFINE_FIELD_REGISTRY(HeterogeneousGNNConfig,
                              RESOLVE_HETEROGENEOUS_GNN_CONFIG_FIELDS)

#define RESOLVE_TABM_CONFIG_FIELDS(F)      \
    F(enabled,     "tabm_enabled")         \
    F(n_ensembles, "tabm_n_ensembles")     \
    F(aggregation, "tabm_aggregation")

RESOLVE_DEFINE_FIELD_REGISTRY(TabMConfig, RESOLVE_TABM_CONFIG_FIELDS)

// A branch is only ever serialized inside a numbered block, so its keys are the
// bare member names and the block prefix is supplied at write/read time (see
// parallel_branch_prefix below).
#define RESOLVE_PARALLEL_BRANCH_CONFIG_FIELDS(F) \
    F(hidden_dims,   "hidden_dims")              \
    F(activation,    "activation")               \
    F(normalization, "normalization")            \
    F(dropout,       "dropout")                  \
    F(branch_weight, "branch_weight")

RESOLVE_DEFINE_FIELD_REGISTRY(ParallelBranchConfig,
                              RESOLVE_PARALLEL_BRANCH_CONFIG_FIELDS)

// `branches` is variable-length: the key below holds the branch COUNT, and each
// branch's fields follow under parallel_branch_<i>_.
#define RESOLVE_PARALLEL_LAYERS_CONFIG_FIELDS(F)     \
    F(enabled,         "parallel_enabled")           \
    F(branches,        "parallel_n_branches")        \
    F(aggregation,     "parallel_aggregation")       \
    F(attention_heads, "parallel_attention_heads")   \
    F(use_residual,    "parallel_use_residual")

RESOLVE_DEFINE_FIELD_REGISTRY(ParallelLayersConfig,
                              RESOLVE_PARALLEL_LAYERS_CONFIG_FIELDS)

// Checkpoint key prefix for branch i. Paired with the "parallel_n_branches"
// count key above; both sides of the round-trip build the prefix here.
inline std::string parallel_branch_prefix(std::size_t i) {
    return "parallel_branch_" + std::to_string(i) + "_";
}

// ============================================================================
// ModelConfig
// ============================================================================
//
// ModelConfig's own fields serialize under their bare member names; the nine
// sub-config members recurse into the registries above, which carry their own
// prefixed keys (so the key column here is unused for those rows and repeats the
// member name).

#define RESOLVE_MODEL_CONFIG_FIELDS(F)                      \
    F(species_encoding,      "species_encoding")            \
    F(uses_explicit_vector,  "uses_explicit_vector")        \
    F(hash_dim,              "hash_dim")                    \
    F(species_embed_dim,     "species_embed_dim")           \
    F(genus_emb_dim,         "genus_emb_dim")               \
    F(family_emb_dim,        "family_emb_dim")              \
    F(categorical_embed_dim, "categorical_embed_dim")       \
    F(top_k,                 "top_k")                       \
    F(top_k_species,         "top_k_species")               \
    F(n_taxonomy_slots,      "n_taxonomy_slots")            \
    F(hidden_dims,           "hidden_dims")                 \
    F(dropout,               "dropout")                     \
    F(moe_routing,           "moe_routing")                 \
    F(n_experts,             "n_experts")                   \
    F(expert_hidden_dims,    "expert_hidden_dims")          \
    F(moe_top_k,             "moe_top_k")                   \
    F(moe_noise_std,         "moe_noise_std")               \
    F(moe_aux_loss_weight,   "moe_aux_loss_weight")         \
    F(activation,            "activation")                  \
    F(normalization,         "normalization")               \
    F(norm_groups,           "norm_groups")                 \
    F(use_residual,          "use_residual")                \
    F(leaky_relu_slope,      "leaky_relu_slope")            \
    F(elu_alpha,             "elu_alpha")                   \
    F(head_hidden_dims,      "head_hidden_dims")            \
    F(head_activation,       "head_activation")             \
    F(head_dropout,          "head_dropout")                \
    F(encoder_architecture,  "encoder_architecture")        \
    F(ft_transformer,        "ft_transformer")              \
    F(tabnet,                "tabnet")                      \
    F(saint,                 "saint")                       \
    F(gnn,                   "gnn")                         \
    F(trait_net,             "trait_net")                   \
    F(excelformer,           "excelformer")                 \
    F(heterogeneous_gnn,     "heterogeneous_gnn")           \
    F(parallel_layers,       "parallel_layers")             \
    F(tabm,                  "tabm")                        \
    F(cover_dropout,         "cover_dropout")               \
    F(d_model,               "d_model")                     \
    F(n_heads,               "n_heads")                     \
    F(n_attention_layers,    "n_attention_layers")          \
    F(transformer_ff_dim,    "transformer_ff_dim")          \
    F(transformer_pooling,   "transformer_pooling")         \
    F(transformer_dropout,   "transformer_dropout")

RESOLVE_DEFINE_FIELD_REGISTRY(ModelConfig, RESOLVE_MODEL_CONFIG_FIELDS)

// The keys load_model_config demands rather than defaulting. They have been in
// every checkpoint since the first release, so their absence means a truncated
// or foreign archive, and quietly substituting a default would build a
// differently-shaped model whose weights then fail to load with a shape error
// far from the cause.
inline constexpr const char* kRequiredModelConfigKeys[] = {
    "species_encoding", "uses_explicit_vector", "hash_dim", "species_embed_dim",
    "genus_emb_dim", "family_emb_dim", "top_k", "top_k_species",
    "n_taxonomy_slots", "dropout", "hidden_dims",
};

// ============================================================================
// DatasetConfig
// ============================================================================
//
// The loader config is not serialized under its own keys: a checkpoint carries
// these knobs on ResolveSchema (schema_* keys, issue #102) and
// dataset_config_from_checkpoint reassembles them. The key column is therefore
// empty, and archive visitors skip empty keys.

#define RESOLVE_DATASET_CONFIG_FIELDS(F) \
    F(species_encoding,       "")        \
    F(hash_dim,               "")        \
    F(top_k,                  "")        \
    F(top_k_species,          "")        \
    F(selection,              "")        \
    F(representation,         "")        \
    F(normalization,          "")        \
    F(aggregation,            "")        \
    F(track_unknown_fraction, "")        \
    F(track_unknown_count,    "")        \
    F(use_taxonomy,           "")        \
    F(use_cuda_hash,          "")        \
    F(pool_weighting,         "")        \
    F(pool_species_cap,       "")        \
    F(species_budget,         "")

RESOLVE_DEFINE_FIELD_REGISTRY(DatasetConfig, RESOLVE_DATASET_CONFIG_FIELDS)

// ============================================================================
// TrainConfig
// ============================================================================
//
// save_train_config persists the hyperparameters that define the recipe; the
// device, the checkpoint destination, the AMP / cuDNN switches and the log
// callback describe the machine the run happened on rather than the recipe, so
// they carry an empty key and stay out of the archive (they still cross the
// C-ABI value tree, which is a live config, and the Python bindings). The
// callback is not marshalable anywhere and every visitor skips it by type.
//
// phase_boundaries is a pair and occupies two keys: the row's key is the stem,
// with "1" and "2" appended.

#define RESOLVE_TRAIN_CONFIG_FIELDS(F)                  \
    F(batch_size,          "train_batch_size")          \
    F(max_epochs,          "train_max_epochs")          \
    F(patience,            "train_patience")            \
    F(lr,                  "train_lr")                  \
    F(weight_decay,        "train_weight_decay")        \
    F(phase_boundaries,    "train_phase_boundary_")     \
    F(loss_config,         "train_loss_config")         \
    F(device,              "")                          \
    F(lr_scheduler,        "train_lr_scheduler")        \
    F(lr_step_size,        "train_lr_step_size")        \
    F(lr_gamma,            "train_lr_gamma")            \
    F(lr_min,              "train_lr_min")              \
    F(band_thresholds,     "train_band_thresholds")     \
    F(band_threshold,      "train_band_threshold")      \
    F(nca_temperature,     "train_nca_temperature")     \
    F(nca_neighbors,       "train_nca_neighbors")       \
    F(nca_weight,          "train_nca_weight")          \
    F(checkpoint_dir,      "")                          \
    F(checkpoint_every,    "")                          \
    F(log,                 "")                          \
    F(use_amp,             "")                          \
    F(amp_init_scale,      "")                          \
    F(amp_growth_factor,   "")                          \
    F(amp_backoff_factor,  "")                          \
    F(amp_growth_interval, "")                          \
    F(cudnn_benchmark,     "")                          \
    F(allow_tf32,          "")                          \
    F(vram_fraction,       "train_vram_fraction")       \
    F(batch_size_floor,    "train_batch_size_floor")

RESOLVE_DEFINE_FIELD_REGISTRY(TrainConfig, RESOLVE_TRAIN_CONFIG_FIELDS)

// The batch size that actually trained the model, which is config_.batch_size
// after any CUDA auto-halve-on-OOM retry. Not a TrainConfig field (the config
// carries the REQUESTED value under train_batch_size); written and read beside
// the registry pass so a fallback run stays detectable (issue #86).
inline constexpr const char* kEffectiveBatchSizeKey = "train_effective_batch_size";

// ============================================================================
// Visitor helpers
// ============================================================================

// A row with an empty key is not persisted in the checkpoint.
inline bool has_checkpoint_key(const char* key) noexcept {
    return key != nullptr && key[0] != '\0';
}

// True for the field types a config visitor must handle. Used only to give the
// nanobind binder a clean skip for the two members Python reaches another way
// (device, through a string property) or not at all (the log callback).
template <typename T>
inline constexpr bool is_python_bindable_field_v =
    !std::is_same_v<T, torch::Device> && !std::is_same_v<T, LogCallback>;

}  // namespace resolve
