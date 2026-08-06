#pragma once

// String <-> enum names for every RESOLVE enum that crosses a text boundary:
// the C ABI value tree (R config lists), the CLI's flag values, and the
// checkpoint's human-readable JSON sidecar.
//
// One table per enum. The parser and the emitter both read that table, so a
// spelling cannot drift between the site that produces a name and the site
// that consumes it, and adding a variant is a single row rather than an edit
// in every switch statement that mentions the enum.

#include "resolve/types.hpp"
#include "resolve/species_encoding.hpp"  // PoolWeighting

#include <cstddef>
#include <stdexcept>
#include <string>

namespace resolve {

// One (name, value) row of an enum's name table.
template <typename EnumT>
struct EnumName {
    const char* name;
    EnumT value;
};

// Comma-separated list of every accepted spelling, for error messages and CLI
// help text.
template <typename EnumT, std::size_t N>
std::string enum_value_list(const EnumName<EnumT> (&table)[N]) {
    std::string out;
    for (std::size_t i = 0; i < N; ++i) {
        if (i > 0) out += ", ";
        out += table[i].name;
    }
    return out;
}

// Name -> enum. Throws std::runtime_error naming the offending value and every
// accepted spelling.
template <typename EnumT, std::size_t N>
EnumT parse_enum_name(const std::string& s,
                      const EnumName<EnumT> (&table)[N],
                      const char* type_name) {
    for (const auto& entry : table) {
        if (s == entry.name) return entry.value;
    }
    throw std::runtime_error(std::string("Invalid ") + type_name + ": " + s +
                             ". Valid values: " + enum_value_list(table));
}

// Enum -> name. `fallback` covers a value outside the table (only reachable via
// a corrupt cast), so the emitters stay total.
template <typename EnumT, std::size_t N>
const char* enum_name_of(EnumT value,
                         const EnumName<EnumT> (&table)[N],
                         const char* fallback) {
    for (const auto& entry : table) {
        if (entry.value == value) return entry.name;
    }
    return fallback;
}

// ============================================================================
// Tables
// ============================================================================

inline constexpr EnumName<SpeciesEncodingMode> kSpeciesEncodingNames[] = {
    {"hash", SpeciesEncodingMode::Hash},
    {"embed", SpeciesEncodingMode::Embed},
    {"sparse", SpeciesEncodingMode::Sparse},
    {"rank_pool", SpeciesEncodingMode::RankPool},
    {"transformer", SpeciesEncodingMode::Transformer},
};

inline constexpr EnumName<SelectionMode> kSelectionModeNames[] = {
    {"top", SelectionMode::Top},
    {"bottom", SelectionMode::Bottom},
    {"top_bottom", SelectionMode::TopBottom},
    {"all", SelectionMode::All},
};

inline constexpr EnumName<RepresentationMode> kRepresentationModeNames[] = {
    {"abundance", RepresentationMode::Abundance},
    {"presence_absence", RepresentationMode::PresenceAbsence},
};

inline constexpr EnumName<NormalizationMode> kNormalizationModeNames[] = {
    {"raw", NormalizationMode::Raw},
    {"norm", NormalizationMode::Norm},
    {"log1p", NormalizationMode::Log1p},
};

inline constexpr EnumName<AggregationMode> kAggregationModeNames[] = {
    {"abundance", AggregationMode::Abundance},
    {"count", AggregationMode::Count},
};

inline constexpr EnumName<PoolWeighting> kPoolWeightingNames[] = {
    {"binary", PoolWeighting::Binary},
    {"abundance", PoolWeighting::Abundance},
    {"log1p", PoolWeighting::Log1p},
    {"norm", PoolWeighting::Norm},
    {"rank", PoolWeighting::Rank},
};

inline constexpr EnumName<TaskType> kTaskTypeNames[] = {
    {"regression", TaskType::Regression},
    {"classification", TaskType::Classification},
};

inline constexpr EnumName<TransformType> kTransformTypeNames[] = {
    {"none", TransformType::None},
    {"log1p", TransformType::Log1p},
};

inline constexpr EnumName<LossConfigMode> kLossConfigModeNames[] = {
    {"mae", LossConfigMode::MAE},
    {"smape", LossConfigMode::SMAPE},
    {"combined", LossConfigMode::Combined},
    {"nca", LossConfigMode::NCA},
};

inline constexpr EnumName<LRSchedulerType> kLRSchedulerTypeNames[] = {
    {"none", LRSchedulerType::None},
    {"step", LRSchedulerType::StepLR},
    {"cosine", LRSchedulerType::CosineAnnealing},
};

inline constexpr EnumName<MoERoutingType> kMoERoutingTypeNames[] = {
    {"none", MoERoutingType::None},
    {"soft", MoERoutingType::Soft},
    {"topk", MoERoutingType::TopK},
};

inline constexpr EnumName<ActivationType> kActivationTypeNames[] = {
    {"relu", ActivationType::ReLU},
    {"leaky_relu", ActivationType::LeakyReLU},
    {"gelu", ActivationType::GELU},
    {"silu", ActivationType::SiLU},
    {"tanh", ActivationType::Tanh},
    {"mish", ActivationType::Mish},
    {"elu", ActivationType::ELU},
    {"selu", ActivationType::SELU},
    {"softplus", ActivationType::Softplus},
    {"prelu", ActivationType::PReLU},
};

inline constexpr EnumName<NormLayerType> kNormLayerTypeNames[] = {
    {"batch_norm", NormLayerType::BatchNorm},
    {"layer_norm", NormLayerType::LayerNorm},
    {"group_norm", NormLayerType::GroupNorm},
    {"rms_norm", NormLayerType::RMSNorm},
    {"none", NormLayerType::None},
};

inline constexpr EnumName<EncoderArchitecture> kEncoderArchitectureNames[] = {
    {"mlp", EncoderArchitecture::MLP},
    {"ft_transformer", EncoderArchitecture::FTTransformer},
    {"tabnet", EncoderArchitecture::TabNet},
    {"saint", EncoderArchitecture::SAINT},
    {"trait_net", EncoderArchitecture::TraitNet},
    {"gnn", EncoderArchitecture::GNN},
    {"excelformer", EncoderArchitecture::ExcelFormer},
    {"heterogeneous_gnn", EncoderArchitecture::HeterogeneousGNN},
};

inline constexpr EnumName<GNNType> kGNNTypeNames[] = {
    {"gcn", GNNType::GCN},
    {"gat", GNNType::GAT},
    {"graphsage", GNNType::GraphSAGE},
};

inline constexpr EnumName<GraphConstructionMode> kGraphConstructionModeNames[] = {
    {"spatial", GraphConstructionMode::Spatial},
    {"taxonomic", GraphConstructionMode::Taxonomic},
    {"cooccurrence", GraphConstructionMode::CoOccurrence},
};

inline constexpr EnumName<TraitInteractionMode> kTraitInteractionModeNames[] = {
    {"bilinear", TraitInteractionMode::Bilinear},
    {"mlp", TraitInteractionMode::MLP},
    {"attention", TraitInteractionMode::Attention},
};

inline constexpr EnumName<ParallelAggregation> kParallelAggregationNames[] = {
    {"concat", ParallelAggregation::Concat},
    {"sum", ParallelAggregation::Sum},
    {"mean", ParallelAggregation::Mean},
    {"attention", ParallelAggregation::Attention},
    {"gated", ParallelAggregation::Gated},
};

// ============================================================================
// Parsers and emitters
// ============================================================================

inline SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    return parse_enum_name(s, kSpeciesEncodingNames, "species encoding mode");
}
inline const char* species_encoding_to_string(SpeciesEncodingMode m) {
    return enum_name_of(m, kSpeciesEncodingNames, "unknown");
}

inline SelectionMode parse_selection_mode(const std::string& s) {
    return parse_enum_name(s, kSelectionModeNames, "selection mode");
}
inline const char* selection_mode_to_string(SelectionMode m) {
    return enum_name_of(m, kSelectionModeNames, "top");
}

inline RepresentationMode parse_representation_mode(const std::string& s) {
    return parse_enum_name(s, kRepresentationModeNames, "representation mode");
}
inline const char* representation_mode_to_string(RepresentationMode m) {
    return enum_name_of(m, kRepresentationModeNames, "abundance");
}

inline NormalizationMode parse_normalization_mode(const std::string& s) {
    return parse_enum_name(s, kNormalizationModeNames, "normalization mode");
}
inline const char* normalization_mode_to_string(NormalizationMode m) {
    return enum_name_of(m, kNormalizationModeNames, "raw");
}

inline AggregationMode parse_aggregation_mode(const std::string& s) {
    return parse_enum_name(s, kAggregationModeNames, "aggregation mode");
}
inline const char* aggregation_mode_to_string(AggregationMode m) {
    return enum_name_of(m, kAggregationModeNames, "abundance");
}

inline PoolWeighting parse_pool_weighting(const std::string& s) {
    return parse_enum_name(s, kPoolWeightingNames, "pool weighting");
}
inline const char* pool_weighting_to_string(PoolWeighting m) {
    return enum_name_of(m, kPoolWeightingNames, "log1p");
}

inline TaskType parse_task_type(const std::string& s) {
    return parse_enum_name(s, kTaskTypeNames, "task type");
}
inline const char* task_type_to_string(TaskType m) {
    return enum_name_of(m, kTaskTypeNames, "regression");
}

inline TransformType parse_transform_type(const std::string& s) {
    return parse_enum_name(s, kTransformTypeNames, "transform type");
}
inline const char* transform_type_to_string(TransformType m) {
    return enum_name_of(m, kTransformTypeNames, "none");
}

inline LossConfigMode parse_loss_config_mode(const std::string& s) {
    return parse_enum_name(s, kLossConfigModeNames, "loss config mode");
}
inline const char* loss_config_mode_to_string(LossConfigMode m) {
    return enum_name_of(m, kLossConfigModeNames, "combined");
}

inline LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    return parse_enum_name(s, kLRSchedulerTypeNames, "LR scheduler type");
}
inline const char* lr_scheduler_type_to_string(LRSchedulerType m) {
    return enum_name_of(m, kLRSchedulerTypeNames, "none");
}

inline MoERoutingType parse_moe_routing_type(const std::string& s) {
    return parse_enum_name(s, kMoERoutingTypeNames, "MoE routing type");
}
inline const char* moe_routing_type_to_string(MoERoutingType m) {
    return enum_name_of(m, kMoERoutingTypeNames, "none");
}

inline ActivationType parse_activation_type(const std::string& s) {
    return parse_enum_name(s, kActivationTypeNames, "activation type");
}
inline const char* activation_type_to_string(ActivationType m) {
    return enum_name_of(m, kActivationTypeNames, "gelu");
}

inline NormLayerType parse_norm_layer_type(const std::string& s) {
    return parse_enum_name(s, kNormLayerTypeNames, "normalization layer type");
}
inline const char* norm_layer_type_to_string(NormLayerType m) {
    return enum_name_of(m, kNormLayerTypeNames, "batch_norm");
}

inline EncoderArchitecture parse_encoder_architecture(const std::string& s) {
    return parse_enum_name(s, kEncoderArchitectureNames, "encoder architecture");
}
inline const char* encoder_architecture_to_string(EncoderArchitecture m) {
    return enum_name_of(m, kEncoderArchitectureNames, "mlp");
}

inline GNNType parse_gnn_type(const std::string& s) {
    return parse_enum_name(s, kGNNTypeNames, "GNN type");
}
inline const char* gnn_type_to_string(GNNType m) {
    return enum_name_of(m, kGNNTypeNames, "gat");
}

inline GraphConstructionMode parse_graph_construction_mode(const std::string& s) {
    return parse_enum_name(s, kGraphConstructionModeNames, "graph construction mode");
}
inline const char* graph_construction_mode_to_string(GraphConstructionMode m) {
    return enum_name_of(m, kGraphConstructionModeNames, "spatial");
}

inline TraitInteractionMode parse_trait_interaction_mode(const std::string& s) {
    return parse_enum_name(s, kTraitInteractionModeNames, "trait interaction mode");
}
inline const char* trait_interaction_mode_to_string(TraitInteractionMode m) {
    return enum_name_of(m, kTraitInteractionModeNames, "bilinear");
}

inline ParallelAggregation parse_parallel_aggregation(const std::string& s) {
    return parse_enum_name(s, kParallelAggregationNames, "parallel aggregation");
}
inline const char* parallel_aggregation_to_string(ParallelAggregation m) {
    return enum_name_of(m, kParallelAggregationNames, "concat");
}

// ============================================================================
// Generic table access
// ============================================================================
//
// The named helpers above are the call-site form: the caller knows which enum it
// holds. A table-driven consumer does not -- the config field registry
// (config_registry.hpp) visits an enum-typed field generically and has only the
// type. EnumNames<E> makes the table reachable from the type, pointing at the
// SAME rows, so a spelling is still written in exactly one place.

template <typename EnumT>
struct EnumNames;

#define RESOLVE_DECLARE_ENUM_NAMES(EnumT, Table, Label, Fallback) \
    template <>                                                   \
    struct EnumNames<EnumT> {                                     \
        static constexpr auto& table = Table;                     \
        static constexpr const char* label = Label;               \
        static constexpr const char* fallback = Fallback;         \
    };

RESOLVE_DECLARE_ENUM_NAMES(SpeciesEncodingMode, kSpeciesEncodingNames,
                           "species encoding mode", "unknown")
RESOLVE_DECLARE_ENUM_NAMES(SelectionMode, kSelectionModeNames,
                           "selection mode", "top")
RESOLVE_DECLARE_ENUM_NAMES(RepresentationMode, kRepresentationModeNames,
                           "representation mode", "abundance")
RESOLVE_DECLARE_ENUM_NAMES(NormalizationMode, kNormalizationModeNames,
                           "normalization mode", "raw")
RESOLVE_DECLARE_ENUM_NAMES(AggregationMode, kAggregationModeNames,
                           "aggregation mode", "abundance")
RESOLVE_DECLARE_ENUM_NAMES(PoolWeighting, kPoolWeightingNames,
                           "pool weighting", "log1p")
RESOLVE_DECLARE_ENUM_NAMES(TaskType, kTaskTypeNames,
                           "task type", "regression")
RESOLVE_DECLARE_ENUM_NAMES(TransformType, kTransformTypeNames,
                           "transform type", "none")
RESOLVE_DECLARE_ENUM_NAMES(LossConfigMode, kLossConfigModeNames,
                           "loss config mode", "combined")
RESOLVE_DECLARE_ENUM_NAMES(LRSchedulerType, kLRSchedulerTypeNames,
                           "LR scheduler type", "none")
RESOLVE_DECLARE_ENUM_NAMES(MoERoutingType, kMoERoutingTypeNames,
                           "MoE routing type", "none")
RESOLVE_DECLARE_ENUM_NAMES(ActivationType, kActivationTypeNames,
                           "activation type", "gelu")
RESOLVE_DECLARE_ENUM_NAMES(NormLayerType, kNormLayerTypeNames,
                           "normalization layer type", "batch_norm")
RESOLVE_DECLARE_ENUM_NAMES(EncoderArchitecture, kEncoderArchitectureNames,
                           "encoder architecture", "mlp")
RESOLVE_DECLARE_ENUM_NAMES(GNNType, kGNNTypeNames,
                           "GNN type", "gat")
RESOLVE_DECLARE_ENUM_NAMES(GraphConstructionMode, kGraphConstructionModeNames,
                           "graph construction mode", "spatial")
RESOLVE_DECLARE_ENUM_NAMES(TraitInteractionMode, kTraitInteractionModeNames,
                           "trait interaction mode", "bilinear")
RESOLVE_DECLARE_ENUM_NAMES(ParallelAggregation, kParallelAggregationNames,
                           "parallel aggregation", "concat")

#undef RESOLVE_DECLARE_ENUM_NAMES

// Enum -> name / name -> enum for a caller that knows only the type. Both route
// through the same enum_name_of / parse_enum_name the named helpers use.
template <typename EnumT>
const char* enum_to_name(EnumT value) {
    return enum_name_of(value, EnumNames<EnumT>::table, EnumNames<EnumT>::fallback);
}

template <typename EnumT>
EnumT enum_from_name(const std::string& s) {
    return parse_enum_name(s, EnumNames<EnumT>::table, EnumNames<EnumT>::label);
}

}  // namespace resolve
