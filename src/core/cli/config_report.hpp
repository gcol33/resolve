// RESOLVE CLI - checkpoint reporting
//
// `resolve info` prints a checkpoint's description (checkpoint_description,
// the document its JSON sidecar holds) as indented text: one block per section,
// every row of a config block under that block's label ("Model hash_dim: 32"),
// a nested sub-config indented under its row, a list of objects as "- " items.
// The rows come from the field registry through the description, so a field
// added to ModelConfig, DatasetConfig or TrainConfig is reported in the edit
// that adds it.
//
// The renderer takes its ostream so a test can render a block into a
// stringstream, and the rule deciding which ModelConfig rows a checkpoint
// shows lives here rather than in info_cmd.cpp, which is linked only into the
// CLI binary.

#ifndef RESOLVE_CLI_CONFIG_REPORT_HPP
#define RESOLVE_CLI_CONFIG_REPORT_HPP

#include <functional>
#include <ostream>
#include <string>
#include <string_view>

#include "resolve/architectures.hpp"
#include "resolve/json.hpp"
#include "resolve/types.hpp"

namespace resolve_cli {

// Decides whether one top-level row is printed, by field name. An empty filter
// prints every row.
using FieldFilter = std::function<bool(const char*)>;

namespace report_detail {

// A scalar as info prints it: booleans as yes / no, a string without quotes,
// a number in its shortest exact spelling, null (a value the run never set,
// such as the best loss of a run that scored no epoch) as NA.
inline std::string scalar_text(const resolve::json::Value& v) {
    using Kind = resolve::json::Value::Kind;
    switch (v.kind()) {
        case Kind::Null: return "NA";
        case Kind::Bool: return v.as_bool() ? "yes" : "no";
        case Kind::String: return v.as_string();
        default: return resolve::json::dump(v, 0);
    }
}

inline bool is_scalar_list(const resolve::json::Value& v) {
    for (const auto& item : v.items()) {
        if (item.is_object() || item.is_array()) return false;
    }
    return true;
}

inline void print_rows(std::ostream& out, const resolve::json::Value& object,
                       const std::string& label, const std::string& indent,
                       const FieldFilter& filter);

// One row: "<indent><label> <name>: <value>", the label only on a block's own
// rows.
inline void print_row(std::ostream& out, const std::string& name, const resolve::json::Value& v,
                      const std::string& label, const std::string& indent,
                      const std::string& first_indent) {
    const std::string head = first_indent + (label.empty() ? "" : label + " ") + name + ":";
    if (v.is_object()) {
        out << head << "\n";
        print_rows(out, v, "", indent + "  ", FieldFilter{});
    } else if (v.is_array() && is_scalar_list(v)) {
        out << head << " [";
        bool first = true;
        for (const auto& item : v.items()) {
            out << (first ? "" : ", ") << scalar_text(item);
            first = false;
        }
        out << "]\n";
    } else if (v.is_array()) {
        out << head << "\n";
        for (const auto& item : v.items()) {
            const std::string item_indent = indent + "    ";
            bool first = true;
            for (const auto& [key, value] : item.members()) {
                print_row(out, key, value, "", item_indent,
                          first ? indent + "  - " : item_indent);
                first = false;
            }
        }
    } else {
        out << head << " " << scalar_text(v) << "\n";
    }
}

inline void print_rows(std::ostream& out, const resolve::json::Value& object,
                       const std::string& label, const std::string& indent,
                       const FieldFilter& filter) {
    for (const auto& [name, value] : object.members()) {
        if (filter && !filter(name.c_str())) continue;
        print_row(out, name, value, label, indent, indent);
    }
}

}  // namespace report_detail

// Print every member of a description object under a common label. The filter
// applies to the object's own rows only; a nested sub-config is printed whole.
// Field names repeat across structs -- ModelConfig and FTTransformerConfig
// both have n_heads -- so a name-keyed rule written for one struct must not
// reach another.
inline void print_block(std::ostream& out, const resolve::json::Value& object, const char* label,
                        const FieldFilter& filter = {}) {
    report_detail::print_rows(out, object, label, "  ", filter);
}

// The ModelConfig rows model_field_shown can hide. Kept as a list so a test can
// assert every name is still a registry row: a renamed field would otherwise
// leave a rule here matching nothing, and the row it used to gate would start
// printing unconditionally with nothing to say so. The architecture
// sub-configs are the rows of kArchitectures.
inline constexpr const char* kConditionalModelFields[] = {
    "tabm", "parallel_layers", "moe_placement", "n_experts", "expert_hidden_dims",
    "moe_top_k", "moe_noise_std", "moe_aux_loss_weight", "head_activation", "head_dropout",
};

// A ModelConfig row is hidden only when another field of the same struct
// switches its feature off: an architecture sub-config that is not the selected
// architecture, the mixture's hyperparameters when routing is none, TabM and
// the parallel branches when disabled, and the head's shape when the head has
// no hidden layers. A field the checkpoint carries but this encoder does not
// read -- hash_dim on a rank_pool model, d_model on a hash one -- still prints:
// `info` answers what the checkpoint holds.
inline bool model_field_shown(const resolve::ModelConfig& config, std::string_view name) {
    for (const auto& entry : resolve::kArchitectures) {
        if (name == entry.member) return config.encoder_architecture == entry.architecture;
    }

    if (name == "tabm") return config.tabm.enabled;
    if (name == "parallel_layers") return config.parallel_layers.enabled;

    // moe_routing itself always prints, so a model with no mixture says so
    // rather than going silent on the subject.
    if (name == "moe_placement" || name == "n_experts" || name == "expert_hidden_dims" ||
        name == "moe_top_k" || name == "moe_noise_std" || name == "moe_aux_loss_weight") {
        return config.moe_routing != resolve::MoERoutingType::None;
    }

    // head_hidden_dims prints either way -- an empty list is the answer.
    if (name == "head_activation" || name == "head_dropout") {
        return !config.head_hidden_dims.empty();
    }

    return true;
}

// The filter `resolve info` applies to a checkpoint's model block.
inline FieldFilter model_field_filter(const resolve::ModelConfig& config) {
    return [&config](const char* name) { return model_field_shown(config, name); };
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_CONFIG_REPORT_HPP
