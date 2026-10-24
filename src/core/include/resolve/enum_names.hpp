#pragma once

// String <-> enum names for every RESOLVE enum that crosses a text boundary:
// the C ABI value tree (R config lists), the CLI's flag values, the checkpoint's
// JSON sidecar, `resolve info` and the suite manifest.
//
// One table per enum, declared once below and reachable from the type through
// EnumNames<E>. enum_to_name and enum_from_name both read that table, so a
// spelling cannot drift between the site that produces a name and the site that
// consumes it, and adding a variant is a single row.

#include "resolve/types.hpp"

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

inline constexpr EnumName<MissingValuePolicy> kMissingValuePolicyNames[] = {
    {"zero", MissingValuePolicy::Zero},
    {"indicate", MissingValuePolicy::Indicate},
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

inline constexpr EnumName<TargetMissingPolicy> kTargetMissingPolicyNames[] = {
    {"drop", TargetMissingPolicy::Drop},
    {"mask", TargetMissingPolicy::Mask},
};

inline constexpr EnumName<HoldOutMode> kHoldOutModeNames[] = {
    {"random", HoldOutMode::Random},
    {"explicit", HoldOutMode::Explicit},
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
    {"plateau", LRSchedulerType::ReduceOnPlateau},
    {"linear", LRSchedulerType::LinearDecay},
};

inline constexpr EnumName<SelectionMetric> kSelectionMetricNames[] = {
    {"loss", SelectionMetric::Loss},
    {"accuracy", SelectionMetric::Accuracy},
};

inline constexpr EnumName<MoERoutingType> kMoERoutingTypeNames[] = {
    {"none", MoERoutingType::None},
    {"soft", MoERoutingType::Soft},
    {"topk", MoERoutingType::TopK},
};

inline constexpr EnumName<MoEPlacement> kMoEPlacementNames[] = {
    {"tail", MoEPlacement::Tail},
    {"post", MoEPlacement::Post},
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

inline constexpr EnumName<TransformerPooling> kTransformerPoolingNames[] = {
    {"attention", TransformerPooling::Attention},
    {"cls", TransformerPooling::Cls},
};

inline constexpr EnumName<TabMAggregation> kTabMAggregationNames[] = {
    {"mean", TabMAggregation::Mean},
    {"median", TabMAggregation::Median},
};

inline constexpr EnumName<MaskStrategy> kMaskStrategyNames[] = {
    {"random", MaskStrategy::Random},
    {"block", MaskStrategy::Block},
    {"structured", MaskStrategy::Structured},
};

inline constexpr EnumName<SuiteCombine> kSuiteCombineNames[] = {
    {"mean", SuiteCombine::Mean},
    {"vote", SuiteCombine::Vote},
    {"circular_mean", SuiteCombine::CircularMean},
    {"spherical_mean", SuiteCombine::SphericalMean},
};

inline constexpr EnumName<SuiteTargetStatus> kSuiteTargetStatusNames[] = {
    {"released", SuiteTargetStatus::Released},
    {"experimental", SuiteTargetStatus::Experimental},
};

// ============================================================================
// Access by type
// ============================================================================
//
// EnumNames<E> makes an enum's table reachable from its type, which is all a
// table-driven consumer such as the config field registry (config_registry.hpp)
// has: `label` names the enum in an error, `fallback` is the name of a value
// outside the table (reachable only through a corrupt cast).

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
RESOLVE_DECLARE_ENUM_NAMES(MissingValuePolicy, kMissingValuePolicyNames,
                           "missing value policy", "indicate")
RESOLVE_DECLARE_ENUM_NAMES(PoolWeighting, kPoolWeightingNames,
                           "pool weighting", "log1p")
RESOLVE_DECLARE_ENUM_NAMES(TaskType, kTaskTypeNames,
                           "task type", "regression")
RESOLVE_DECLARE_ENUM_NAMES(TransformType, kTransformTypeNames,
                           "transform type", "none")
RESOLVE_DECLARE_ENUM_NAMES(TargetMissingPolicy, kTargetMissingPolicyNames,
                           "target missing-label policy", "drop")
RESOLVE_DECLARE_ENUM_NAMES(HoldOutMode, kHoldOutModeNames,
                           "hold-out mode", "random")
RESOLVE_DECLARE_ENUM_NAMES(LossConfigMode, kLossConfigModeNames,
                           "loss config mode", "combined")
RESOLVE_DECLARE_ENUM_NAMES(LRSchedulerType, kLRSchedulerTypeNames,
                           "LR scheduler type", "none")
RESOLVE_DECLARE_ENUM_NAMES(SelectionMetric, kSelectionMetricNames,
                           "selection metric", "loss")
RESOLVE_DECLARE_ENUM_NAMES(MoERoutingType, kMoERoutingTypeNames,
                           "MoE routing type", "none")
RESOLVE_DECLARE_ENUM_NAMES(MoEPlacement, kMoEPlacementNames,
                           "MoE placement", "tail")
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
RESOLVE_DECLARE_ENUM_NAMES(TransformerPooling, kTransformerPoolingNames,
                           "transformer pooling", "attention")
RESOLVE_DECLARE_ENUM_NAMES(TabMAggregation, kTabMAggregationNames,
                           "TabM aggregation", "mean")
RESOLVE_DECLARE_ENUM_NAMES(MaskStrategy, kMaskStrategyNames,
                           "mask strategy", "random")
RESOLVE_DECLARE_ENUM_NAMES(SuiteCombine, kSuiteCombineNames,
                           "suite combine rule", "mean")
RESOLVE_DECLARE_ENUM_NAMES(SuiteTargetStatus, kSuiteTargetStatusNames,
                           "suite target status", "released")

#undef RESOLVE_DECLARE_ENUM_NAMES

// Enum -> name.
template <typename EnumT>
const char* enum_to_name(EnumT value) {
    return enum_name_of(value, EnumNames<EnumT>::table, EnumNames<EnumT>::fallback);
}

// Name -> enum. Throws std::runtime_error naming the value and every accepted
// spelling.
template <typename EnumT>
EnumT enum_from_name(const std::string& s) {
    return parse_enum_name(s, EnumNames<EnumT>::table, EnumNames<EnumT>::label);
}

// The accepted spellings of an enum, comma-separated, for help text.
template <typename EnumT>
std::string enum_spellings() {
    return enum_value_list(EnumNames<EnumT>::table);
}

// The two names the model's error messages print.
inline const char* species_encoding_to_string(SpeciesEncodingMode m) { return enum_to_name(m); }
inline const char* encoder_architecture_to_string(EncoderArchitecture m) {
    return enum_to_name(m);
}

}  // namespace resolve
