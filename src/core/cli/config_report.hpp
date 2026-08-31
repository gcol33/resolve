// RESOLVE CLI - config reporting
//
// `resolve info` prints three config blocks -- the model, the loading-side
// dataset config, and the training recipe -- and all three are driven by the
// shared field registry (config_registry.hpp) rather than by a hand-written
// line per field, so a hyperparameter added to ModelConfig, DatasetConfig or
// TrainConfig is reported in the edit that adds it.
//
// The writer takes its ostream so a test can render a block into a
// stringstream, and the rule deciding which ModelConfig rows a given
// checkpoint shows lives here rather than in info_cmd.cpp, which is linked
// only into the CLI binary.

#ifndef RESOLVE_CLI_CONFIG_REPORT_HPP
#define RESOLVE_CLI_CONFIG_REPORT_HPP

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "resolve/config_registry.hpp"
#include "resolve/types.hpp"

namespace resolve_cli {

// Decides whether one top-level row is printed, by field name. An empty filter
// prints every row.
using FieldFilter = std::function<bool(const char*)>;

// Prints one field-registry row as "<indent><label> <field>: <value>". A whole
// config block therefore follows the struct. Enum values print as the names the
// CLI accepts on its flags, from the shared tables in enum_names.hpp.
struct ConsoleFieldWriter {
    std::ostream& out;
    std::string label;
    std::string indent;
    // When true, a row whose registry key is empty is skipped. Those fields are
    // not in the archive, so they read back as struct defaults, and printing
    // them would report a recipe the run never used.
    bool persisted_only = false;
    // Applied to this writer's own rows only; a nested sub-config is visited by
    // a fresh writer carrying none. Field names repeat across structs --
    // ModelConfig and FTTransformerConfig both have n_heads -- so a name-keyed
    // rule written for one struct must not reach another.
    FieldFilter filter{};

    std::string prefix() const { return label.empty() ? indent : indent + label + " "; }

    std::ostream& open_line(const char* name) const {
        out << prefix() << name << ": ";
        return out;
    }

    // A sub-config's rows are placed by indentation alone: repeating the outer
    // label under it would read as though the nested field belonged to the
    // outer struct.
    ConsoleFieldWriter nested(const std::string& step) const {
        return ConsoleFieldWriter{out, "", indent + step, persisted_only, FieldFilter{}};
    }

    template <typename Seq>
    void print_values(const Seq& values) const {
        out << "[";
        bool first = true;
        for (const auto& v : values) {
            if (!first) out << ", ";
            first = false;
            out << v;
        }
        out << "]" << std::endl;
    }

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        if (persisted_only && !resolve::has_checkpoint_key(key)) return;
        if (filter && !filter(name)) return;
        if constexpr (std::is_same_v<T, resolve::LogCallback>) {
            (void)name;  // a callback has nothing to report
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            open_line(name) << (value.is_cuda() ? "cuda" : "cpu") << std::endl;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            out << prefix() << name << ":" << std::endl;
            resolve::for_each_field(value, nested("  "));
        } else if constexpr (std::is_same_v<T, std::vector<resolve::ParallelBranchConfig>>) {
            open_line(name) << value.size() << std::endl;
            for (std::size_t i = 0; i < value.size(); ++i) {
                out << indent << "  branch " << i << ":" << std::endl;
                resolve::for_each_field(value[i], nested("    "));
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            open_line(name) << (value ? "yes" : "no") << std::endl;
        } else if constexpr (std::is_enum_v<T>) {
            open_line(name) << resolve::enum_to_name(value) << std::endl;
        } else if constexpr (std::is_same_v<T, std::string>) {
            open_line(name) << value << std::endl;
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
            open_line(name) << value << std::endl;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            open_line(name);
            print_values(value);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            open_line(name) << value.first << ", " << value.second << std::endl;
        } else {
            static_assert(resolve::registry_detail::always_false<T>,
                          "config field type has no console representation; "
                          "add a branch to ConsoleFieldWriter");
        }
    }
};

// Print every field of one config struct under a common label.
template <typename Cfg>
void print_config_block(std::ostream& out, const char* label, const Cfg& config,
                        bool persisted_only = false, FieldFilter filter = {}) {
    resolve::for_each_field(
        config, ConsoleFieldWriter{out, label, "  ", persisted_only, std::move(filter)});
}

// The ModelConfig rows model_field_shown can hide. Kept as a list so a test can
// assert every name is still a registry row: a renamed field would otherwise
// leave a rule here matching nothing, and the row it used to gate would start
// printing unconditionally with nothing to say so.
inline constexpr const char* kConditionalModelFields[] = {
    "ft_transformer", "tabnet", "saint", "gnn", "trait_net", "excelformer",
    "heterogeneous_gnn", "tabm", "parallel_layers", "moe_placement", "n_experts",
    "expert_hidden_dims", "moe_top_k", "moe_noise_std", "moe_aux_loss_weight",
    "head_activation", "head_dropout",
};

// A ModelConfig row is hidden only when another field of the same struct
// switches its feature off: an architecture sub-config that is not the selected
// architecture, the mixture's hyperparameters when routing is none, TabM and
// the parallel branches when disabled, and the head's shape when the head has
// no hidden layers. A field the checkpoint carries but this encoder does not
// read -- hash_dim on a rank_pool model, d_model on a hash one -- still prints,
// as hash_dim always has: `info` answers what the checkpoint holds.
inline bool model_field_shown(const resolve::ModelConfig& config, std::string_view name) {
    using resolve::EncoderArchitecture;
    using resolve::MoERoutingType;

    const auto arch = config.encoder_architecture;
    if (name == "ft_transformer") return arch == EncoderArchitecture::FTTransformer;
    if (name == "tabnet") return arch == EncoderArchitecture::TabNet;
    if (name == "saint") return arch == EncoderArchitecture::SAINT;
    if (name == "gnn") return arch == EncoderArchitecture::GNN;
    if (name == "trait_net") return arch == EncoderArchitecture::TraitNet;
    if (name == "excelformer") return arch == EncoderArchitecture::ExcelFormer;
    if (name == "heterogeneous_gnn") return arch == EncoderArchitecture::HeterogeneousGNN;

    if (name == "tabm") return config.tabm.enabled;
    if (name == "parallel_layers") return config.parallel_layers.enabled;

    // moe_routing itself always prints, so a model with no mixture says so
    // rather than going silent on the subject.
    if (name == "moe_placement" || name == "n_experts" || name == "expert_hidden_dims" ||
        name == "moe_top_k" || name == "moe_noise_std" || name == "moe_aux_loss_weight") {
        return config.moe_routing != MoERoutingType::None;
    }

    // head_hidden_dims prints either way -- an empty list is the answer.
    if (name == "head_activation" || name == "head_dropout") {
        return !config.head_hidden_dims.empty();
    }

    return true;
}

// The filter `resolve info` hands print_config_block for a checkpoint's model.
inline FieldFilter model_field_filter(const resolve::ModelConfig& config) {
    return [&config](const char* name) { return model_field_shown(config, name); };
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_CONFIG_REPORT_HPP
