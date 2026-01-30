#include "bindings_common.hpp"

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
        .export_values();

    nb::enum_<resolve::LossConfigMode>(m, "LossConfigMode")
        .value("MAE", resolve::LossConfigMode::MAE)
        .value("SMAPE", resolve::LossConfigMode::SMAPE)
        .value("Combined", resolve::LossConfigMode::Combined)
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
}
