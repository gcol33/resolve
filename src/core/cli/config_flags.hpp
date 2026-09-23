// RESOLVE CLI - configuration flags from the field registry
//
// The CLI could reach no architecture hyperparameter at all: `resolve train`
// selected an encoder architecture but every field of every sub-config kept its
// default, so a standalone run could not train what the bindings could -- the
// gap issue #104 closed for covariates and the mixture, one level down. Writing
// fifty rows into cli_spec.hpp by hand would close it once and leave the next
// field out again.
//
// So the flags come from the same field registry (config_registry.hpp) that
// drives the checkpoint, the C-ABI value tree, nanobind and `resolve info`:
// `append_config_flags` emits one row per field and `read_config_flags` puts
// the parsed value back. A field added to any sub-config gets its flag, its
// help line and its read in the edit that adds the field.
//
// The flag name is the field's CHECKPOINT KEY with dashes for underscores --
// `ft_d_model` becomes `--ft-d-model` -- because that key already carries the
// struct prefix, and nine sub-configs repeat member names (`n_heads` is in
// four of them). A field whose key is empty is not persisted and gets no flag.
//
// Like config_report.hpp, this is CLI-side: linked into the CLI binary and
// reachable from a test by relative path.

#ifndef RESOLVE_CLI_CONFIG_FLAGS_HPP
#define RESOLVE_CLI_CONFIG_FLAGS_HPP

#include <cstdint>
#include <deque>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "resolve/config_registry.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/types.hpp"

#include "arg_parser.hpp"

namespace resolve_cli {

namespace config_flags_detail {

// FlagSpec holds const char*, and the tables it lands in are static, so a
// generated string has to outlive the call that made it. A deque never moves
// what it already holds.
inline const char* own(std::string text) {
    static std::deque<std::string> storage;
    storage.push_back(std::move(text));
    return storage.back().c_str();
}

inline std::string dashed(const char* key) {
    std::string out = "--";
    for (const char* c = key; *c != '\0'; ++c) {
        out.push_back(*c == '_' ? '-' : *c);
    }
    return out;
}

// The accepted spellings of an enum field, read from the same tables the
// parser and `resolve info` use, so help cannot drift from what is accepted.
template <typename EnumT>
std::string enum_spellings() {
    std::string out;
    for (const auto& entry : resolve::EnumNames<EnumT>::table) {
        if (!out.empty()) out += ", ";
        out += entry.name;
    }
    return out;
}

// std::to_string gives a float six decimals ("0.100000"), which reads badly in
// a help column; an ostringstream prints the shortest form that round-trips
// through the same parser the flag is read with.
inline std::string float_text(float value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

template <typename Seq>
std::string join_values(const Seq& values) {
    std::ostringstream out;
    bool first = true;
    for (const auto& value : values) {
        if (!first) out << ",";
        first = false;
        out << value;
    }
    return out.str();
}

}  // namespace config_flags_detail

// Emits the rows for one configuration struct. `label` names the struct in the
// help text ("FT-Transformer"), and the default column shows the value the
// struct carries, so `resolve help` states the real defaults.
struct ConfigFlagWriter {
    std::vector<FlagSpec>& flags;
    std::string label;

    void row(const char* key, Arity arity, const char* value_label,
             std::string default_value, std::string help) const {
        flags.push_back({config_flags_detail::own(config_flags_detail::dashed(key)),
                         arity, value_label,
                         config_flags_detail::own(std::move(default_value)),
                         config_flags_detail::own(std::move(help))});
    }

    std::string help_for(const char* name) const { return label + " " + name; }

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        // A field the checkpoint does not carry has no key to name a flag
        // after, and would not survive the run anyway.
        if (!resolve::has_checkpoint_key(key)) return;

        if constexpr (std::is_same_v<T, bool>) {
            // The CLI's boolean convention: a pair of presence flags, the
            // later one on the command line winning.
            const std::string on = config_flags_detail::dashed(key);
            const std::string off = "--no-" + on.substr(2);
            flags.push_back({config_flags_detail::own(on), Arity::Flag, "", "",
                             config_flags_detail::own(
                                 help_for(name) + " on (default " +
                                 (value ? "on" : "off") + ")")});
            flags.push_back({config_flags_detail::own(off), Arity::Flag, "", "",
                             config_flags_detail::own(help_for(name) + " off")});
        } else if constexpr (std::is_enum_v<T>) {
            row(key, Arity::Value, "V", resolve::enum_to_name(value),
                help_for(name) + ": " + config_flags_detail::enum_spellings<T>());
        } else if constexpr (std::is_same_v<T, int>) {
            row(key, Arity::Value, "N", std::to_string(value), help_for(name));
        } else if constexpr (std::is_same_v<T, float>) {
            row(key, Arity::Value, "FLOAT", config_flags_detail::float_text(value),
                help_for(name));
        } else if constexpr (std::is_same_v<T, std::string>) {
            row(key, Arity::Value, "S", value, help_for(name));
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            row(key, Arity::Value, "LIST",
                config_flags_detail::join_values(value),
                help_for(name) + ", comma-separated");
        } else if constexpr (std::is_same_v<T,
                                            std::vector<resolve::ParallelBranchConfig>>) {
            // Variable-length, so it has its own repeatable grammar in
            // cli_spec.hpp rather than a value flag here.
            (void)name;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            // Only the sub-configs themselves are offered on the CLI, each
            // under its own label; a nested one would need a second prefix.
            (void)name;
        } else {
            (void)name;  // nothing a command line can carry
        }
    }
};

// Reads the rows back onto the struct. Every field is assigned, because an
// omitted flag still reports a value: the default the writer captured from a
// freshly constructed config. So a field nobody mentioned comes back at its
// struct default -- which is why this runs before anything sets a sub-config
// field in code, and after it would overwrite such a value.
struct ConfigFlagReader {
    const ParsedArgs& args;

    template <typename T>
    void operator()(const char* name, const char* key, T& value) const {
        if (!resolve::has_checkpoint_key(key)) return;
        const std::string flag = config_flags_detail::dashed(key);

        if constexpr (std::is_same_v<T, bool>) {
            value = args.get_switch(flag, "--no-" + flag.substr(2), value);
        } else if constexpr (std::is_enum_v<T>) {
            const std::string text = args.get(flag);
            try {
                value = resolve::enum_from_name<T>(text);
            } catch (const std::exception&) {
                throw ArgError(flag + ": unknown value '" + text +
                               "'; accepted: " +
                               config_flags_detail::enum_spellings<T>());
            }
        } else if constexpr (std::is_same_v<T, int>) {
            value = args.get_int(flag);
        } else if constexpr (std::is_same_v<T, float>) {
            value = args.get_float(flag);
        } else if constexpr (std::is_same_v<T, std::string>) {
            value = args.get(flag);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            std::vector<int64_t> parsed;
            for (const auto& item : args.get_list(flag)) {
                try {
                    parsed.push_back(std::stoll(item));
                } catch (const std::exception&) {
                    throw ArgError(flag + " expects whole numbers, got '" + item + "'");
                }
            }
            value = std::move(parsed);
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            std::vector<float> parsed;
            for (const auto& item : args.get_list(flag)) {
                try {
                    parsed.push_back(std::stof(item));
                } catch (const std::exception&) {
                    throw ArgError(flag + " expects numbers, got '" + item + "'");
                }
            }
            value = std::move(parsed);
        } else {
            (void)name;  // no flag was emitted for this field either
        }
    }
};

// One sub-config's flags, under a label that names it in the help text.
template <typename Cfg>
void append_config_flags(std::vector<FlagSpec>& flags, const char* label,
                         const Cfg& defaults) {
    resolve::for_each_field(defaults, ConfigFlagWriter{flags, label});
}

// Read one sub-config's flags back. Raises ArgError on an unusable value.
template <typename Cfg>
void read_config_flags(Cfg& config, const ParsedArgs& args) {
    resolve::for_each_field(config, ConfigFlagReader{args});
}

// Every architecture sub-config, in one place so the table and the reader
// cannot cover different sets. The labels are the ones `resolve info` prints.
inline void append_architecture_flags(std::vector<FlagSpec>& flags) {
    const resolve::ModelConfig defaults;
    append_config_flags(flags, "FT-Transformer", defaults.ft_transformer);
    append_config_flags(flags, "TabNet", defaults.tabnet);
    append_config_flags(flags, "SAINT", defaults.saint);
    append_config_flags(flags, "GNN", defaults.gnn);
    append_config_flags(flags, "TraitNet", defaults.trait_net);
    append_config_flags(flags, "ExcelFormer", defaults.excelformer);
    append_config_flags(flags, "HeterogeneousGNN", defaults.heterogeneous_gnn);
    append_config_flags(flags, "TabM", defaults.tabm);
    append_config_flags(flags, "Parallel layers", defaults.parallel_layers);
}

// ... and the matching reads.
inline void read_architecture_flags(resolve::ModelConfig& config,
                                    const ParsedArgs& args) {
    read_config_flags(config.ft_transformer, args);
    read_config_flags(config.tabnet, args);
    read_config_flags(config.saint, args);
    read_config_flags(config.gnn, args);
    read_config_flags(config.trait_net, args);
    read_config_flags(config.excelformer, args);
    read_config_flags(config.heterogeneous_gnn, args);
    read_config_flags(config.tabm, args);
    read_config_flags(config.parallel_layers, args);
}

// One branch of a parallel block: widths, then the optional fields in order.
//
//   --parallel-branch 256,128
//   --parallel-branch 256,128:gelu:layer_norm:0.2:0.5
//
// Repeat the flag per branch; the order is the branch order. Colons separate
// the fields, commas the widths, exactly as `--target COL:TYPE:N` reads.
inline resolve::ParallelBranchConfig parse_parallel_branch(const std::string& spec) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(spec);
    while (std::getline(stream, part, ':')) parts.push_back(part);
    if (parts.empty() || parts[0].empty()) {
        throw ArgError("--parallel-branch needs at least the hidden widths, "
                       "e.g. 256,128");
    }
    if (parts.size() > 5) {
        throw ArgError("--parallel-branch takes at most "
                       "DIMS:ACTIVATION:NORMALIZATION:DROPOUT:WEIGHT, got '" +
                       spec + "'");
    }

    resolve::ParallelBranchConfig branch;
    branch.hidden_dims.clear();
    std::istringstream widths(parts[0]);
    std::string width;
    while (std::getline(widths, width, ',')) {
        if (width.empty()) continue;
        try {
            branch.hidden_dims.push_back(std::stoll(width));
        } catch (const std::exception&) {
            throw ArgError("--parallel-branch expects whole widths, got '" +
                           width + "'");
        }
    }
    if (branch.hidden_dims.empty()) {
        throw ArgError("--parallel-branch needs at least one hidden width, "
                       "e.g. 256,128");
    }

    auto enum_field = [&spec](const std::string& text, auto& field,
                              const char* what) {
        using FieldT = std::decay_t<decltype(field)>;
        try {
            field = resolve::enum_from_name<FieldT>(text);
        } catch (const std::exception&) {
            throw ArgError(std::string("--parallel-branch ") + what +
                           ": unknown value '" + text + "' in '" + spec +
                           "'; accepted: " +
                           config_flags_detail::enum_spellings<FieldT>());
        }
    };
    if (parts.size() > 1 && !parts[1].empty()) {
        enum_field(parts[1], branch.activation, "activation");
    }
    if (parts.size() > 2 && !parts[2].empty()) {
        enum_field(parts[2], branch.normalization, "normalization");
    }
    auto float_field = [&spec](const std::string& text, float& field,
                               const char* what) {
        try {
            field = std::stof(text);
        } catch (const std::exception&) {
            throw ArgError(std::string("--parallel-branch ") + what +
                           " expects a number, got '" + text + "' in '" +
                           spec + "'");
        }
    };
    if (parts.size() > 3 && !parts[3].empty()) {
        float_field(parts[3], branch.dropout, "dropout");
    }
    if (parts.size() > 4 && !parts[4].empty()) {
        float_field(parts[4], branch.branch_weight, "weight");
    }
    return branch;
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_CONFIG_FLAGS_HPP
