#pragma once

// The encoder architectures that carry a ModelConfig sub-config, one row each.
//
// A row names the architecture, the ModelConfig member holding its sub-config,
// the label the CLI help and `resolve info` print for it, and whether the
// encoder builds a plot's representation from the other plots of the same
// forward pass. The plot GNN does, in every graph_mode: its kNN graph (over
// coordinates, taxonomy features or composition) joins the plots of one
// forward, so a plot's prediction depends on which plots share it, and the
// model trains and predicts on the whole set at once. SAINT's inference rows
// attend to the fixed reference rows it captured in training, not to each
// other, and the heterogeneous GNN passes messages on the species graph of the
// checkpoint.
//
// The CLI flag tables (cli/config_flags.hpp), `resolve info`
// (cli/config_report.hpp) and ResolveModel::requires_full_batch_training read
// this table, so an architecture is added in one row here plus one member
// pointer below.

#include "resolve/types.hpp"

#include <cstddef>
#include <tuple>
#include <utility>

namespace resolve {

struct ArchitectureEntry {
    EncoderArchitecture architecture;
    const char* member;  // the ModelConfig member holding its sub-config
    const char* label;
    bool batch_dependent;
};

inline constexpr ArchitectureEntry kArchitectures[] = {
    {EncoderArchitecture::FTTransformer, "ft_transformer", "FT-Transformer", false},
    {EncoderArchitecture::TabNet, "tabnet", "TabNet", false},
    {EncoderArchitecture::SAINT, "saint", "SAINT", false},
    {EncoderArchitecture::GNN, "gnn", "GNN", true},
    {EncoderArchitecture::TraitNet, "trait_net", "TraitNet", false},
    {EncoderArchitecture::ExcelFormer, "excelformer", "ExcelFormer", false},
    {EncoderArchitecture::HeterogeneousGNN, "heterogeneous_gnn", "HeterogeneousGNN", false},
};

// The sub-config members, in kArchitectures order.
inline constexpr auto kArchitectureConfigs = std::make_tuple(
    &ModelConfig::ft_transformer, &ModelConfig::tabnet, &ModelConfig::saint, &ModelConfig::gnn,
    &ModelConfig::trait_net, &ModelConfig::excelformer, &ModelConfig::heterogeneous_gnn);

static_assert(std::tuple_size_v<decltype(kArchitectureConfigs)> ==
                  sizeof(kArchitectures) / sizeof(kArchitectures[0]),
              "every kArchitectures row needs its member pointer in kArchitectureConfigs");

// The row of `architecture`, or null for one without a sub-config (the MLP).
constexpr const ArchitectureEntry* find_architecture(EncoderArchitecture architecture) {
    for (const auto& entry : kArchitectures) {
        if (entry.architecture == architecture) return &entry;
    }
    return nullptr;
}

constexpr bool is_batch_dependent(EncoderArchitecture architecture) {
    const ArchitectureEntry* entry = find_architecture(architecture);
    return entry != nullptr && entry->batch_dependent;
}

// Calls f(entry, sub_config) for every row, sub_config a reference into
// `config` (const when `config` is).
template <typename Config, typename F>
void for_each_architecture(Config& config, F&& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f(kArchitectures[I], config.*std::get<I>(kArchitectureConfigs)), ...);
    }(std::make_index_sequence<std::tuple_size_v<decltype(kArchitectureConfigs)>>{});
}

}  // namespace resolve
