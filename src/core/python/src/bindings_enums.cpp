#include "bindings_common.hpp"
#include "resolve/species_encoding.hpp"

void register_enums(nb::module_& m) {
    nb::enum_<resolve::TaskType>(m, "TaskType")
        .value("Regression", resolve::TaskType::Regression)
        .value("Classification", resolve::TaskType::Classification)
        .export_values();

    nb::enum_<resolve::TransformType>(m, "TransformType")
        .value("None_", resolve::TransformType::None)
        .value("Log1p", resolve::TransformType::Log1p)
        .export_values();

    nb::enum_<resolve::SpeciesEncodingMode>(m, "SpeciesEncodingMode")
        .value("Hash", resolve::SpeciesEncodingMode::Hash)
        .value("Embed", resolve::SpeciesEncodingMode::Embed)
        .value("Sparse", resolve::SpeciesEncodingMode::Sparse)
        .value("RankPool", resolve::SpeciesEncodingMode::RankPool)
        .value("Transformer", resolve::SpeciesEncodingMode::Transformer)
        .export_values();

    nb::enum_<resolve::LossConfigMode>(m, "LossConfigMode")
        .value("MAE", resolve::LossConfigMode::MAE)
        .value("SMAPE", resolve::LossConfigMode::SMAPE)
        .value("Combined", resolve::LossConfigMode::Combined)
        .value("NCA", resolve::LossConfigMode::NCA)
        .export_values();

    nb::enum_<resolve::SelectionMode>(m, "SelectionMode")
        .value("Top", resolve::SelectionMode::Top)
        .value("Bottom", resolve::SelectionMode::Bottom)
        .value("TopBottom", resolve::SelectionMode::TopBottom)
        .value("All", resolve::SelectionMode::All)
        .export_values();

    nb::enum_<resolve::RepresentationMode>(m, "RepresentationMode")
        .value("Abundance", resolve::RepresentationMode::Abundance)
        .value("PresenceAbsence", resolve::RepresentationMode::PresenceAbsence)
        .export_values();

    nb::enum_<resolve::NormalizationMode>(m, "NormalizationMode")
        .value("Raw", resolve::NormalizationMode::Raw)
        .value("Norm", resolve::NormalizationMode::Norm)
        .value("Log1p", resolve::NormalizationMode::Log1p)
        .export_values();

    nb::enum_<resolve::AggregationMode>(m, "AggregationMode")
        .value("Abundance", resolve::AggregationMode::Abundance)
        .value("Count", resolve::AggregationMode::Count)
        .export_values();

    nb::enum_<resolve::LRSchedulerType>(m, "LRSchedulerType")
        .value("None_", resolve::LRSchedulerType::None)
        .value("StepLR", resolve::LRSchedulerType::StepLR)
        .value("CosineAnnealing", resolve::LRSchedulerType::CosineAnnealing)
        .export_values();

    nb::enum_<resolve::MoERoutingType>(m, "MoERoutingType")
        .value("None_", resolve::MoERoutingType::None)
        .value("Soft", resolve::MoERoutingType::Soft)
        .value("TopK", resolve::MoERoutingType::TopK)
        .export_values();

    nb::enum_<resolve::ActivationType>(m, "ActivationType")
        .value("ReLU", resolve::ActivationType::ReLU)
        .value("LeakyReLU", resolve::ActivationType::LeakyReLU)
        .value("GELU", resolve::ActivationType::GELU)
        .value("SiLU", resolve::ActivationType::SiLU)
        .value("Tanh", resolve::ActivationType::Tanh)
        .value("Mish", resolve::ActivationType::Mish)
        .value("ELU", resolve::ActivationType::ELU)
        .value("SELU", resolve::ActivationType::SELU)
        .value("Softplus", resolve::ActivationType::Softplus)
        .value("PReLU", resolve::ActivationType::PReLU)
        .export_values();

    nb::enum_<resolve::NormLayerType>(m, "NormLayerType")
        .value("BatchNorm", resolve::NormLayerType::BatchNorm)
        .value("LayerNorm", resolve::NormLayerType::LayerNorm)
        .value("GroupNorm", resolve::NormLayerType::GroupNorm)
        .value("RMSNorm", resolve::NormLayerType::RMSNorm)
        .value("None_", resolve::NormLayerType::None)
        .export_values();

    // Architecture enums
    nb::enum_<resolve::EncoderArchitecture>(m, "EncoderArchitecture")
        .value("MLP", resolve::EncoderArchitecture::MLP)
        .value("FTTransformer", resolve::EncoderArchitecture::FTTransformer)
        .value("TabNet", resolve::EncoderArchitecture::TabNet)
        .value("SAINT", resolve::EncoderArchitecture::SAINT)
        .value("TraitNet", resolve::EncoderArchitecture::TraitNet)
        .value("GNN", resolve::EncoderArchitecture::GNN)
        .value("ExcelFormer", resolve::EncoderArchitecture::ExcelFormer)
        .value("HeterogeneousGNN", resolve::EncoderArchitecture::HeterogeneousGNN)
        .export_values();

    nb::enum_<resolve::GNNType>(m, "GNNType")
        .value("GCN", resolve::GNNType::GCN)
        .value("GAT", resolve::GNNType::GAT)
        .value("GraphSAGE", resolve::GNNType::GraphSAGE)
        .export_values();

    nb::enum_<resolve::GraphConstructionMode>(m, "GraphConstructionMode")
        .value("Spatial", resolve::GraphConstructionMode::Spatial)
        .value("Taxonomic", resolve::GraphConstructionMode::Taxonomic)
        .value("CoOccurrence", resolve::GraphConstructionMode::CoOccurrence)
        .export_values();

    nb::enum_<resolve::TraitInteractionMode>(m, "TraitInteractionMode")
        .value("Bilinear", resolve::TraitInteractionMode::Bilinear)
        .value("MLP", resolve::TraitInteractionMode::MLP)
        .value("Attention", resolve::TraitInteractionMode::Attention)
        .export_values();

    nb::enum_<resolve::ParallelAggregation>(m, "ParallelAggregation")
        .value("Concat", resolve::ParallelAggregation::Concat)
        .value("Sum", resolve::ParallelAggregation::Sum)
        .value("Mean", resolve::ParallelAggregation::Mean)
        .value("Attention", resolve::ParallelAggregation::Attention)
        .value("Gated", resolve::ParallelAggregation::Gated)
        .export_values();

    nb::enum_<resolve::PoolWeighting>(m, "PoolWeighting")
        .value("Binary", resolve::PoolWeighting::Binary)
        .value("Abundance", resolve::PoolWeighting::Abundance)
        .value("Log1p", resolve::PoolWeighting::Log1p)
        .value("Norm", resolve::PoolWeighting::Norm)
        .value("Rank", resolve::PoolWeighting::Rank)
        .export_values();
}
