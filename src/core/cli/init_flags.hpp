// RESOLVE CLI - fine-tuning for `resolve train`: --init-from and the
// InitFromConfig flags that say which suite member to start from and what to
// carry over.

#ifndef RESOLVE_CLI_INIT_FLAGS_HPP
#define RESOLVE_CLI_INIT_FLAGS_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "resolve/fine_tune.hpp"

#include "arg_parser.hpp"
#include "config_flags.hpp"

namespace resolve_cli {

// InitFromConfig's flags.
inline constexpr FieldFlag kInitFromConfigFlags[] = {
    {"reuse", "--init-reuse",
     "With --init-from: what is carried over besides the\nencoder: encoder (every head starts fresh) or\nencoder_heads (also the head of each target the\nparent shares)"},
    {"suite_target", "--init-target",
     "With --init-from a suite directory: the target\nwhose member to start from"},
    {"suite_seed", "--init-seed",
     "With --init-from a suite directory: that member's\nseed (default: the target's first member)"},
};

// The fine-tuning rows of the `train` flag table.
inline void append_init_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--init-from", Arity::Value, "PATH", "",
                     "Fine-tune: start from a trained model, a checkpoint\n"
                     "or a suite directory. The data is read with its\n"
                     "vocabularies and loader settings and the model\n"
                     "takes its configuration; only the head, dropout and\n"
                     "freeze flags may differ (--freeze-encoder trains the\n"
                     "heads alone)."});
    append_config_flags(flags, "Fine-tuning", resolve::InitFromConfig{},
                        FlagNaming::members(kInitFromConfigFlags));
}

// --init-from and its settings, or nothing without --init-from. An
// InitFromConfig flag without --init-from is an error.
inline std::optional<std::pair<std::string, resolve::InitFromConfig>> read_init_from(
    const ParsedArgs& args) {
    resolve::InitFromConfig config;
    read_config_flags(config, args, FlagNaming::members(kInitFromConfigFlags));
    std::string source;
    if (!args.get_if_present("--init-from", source)) {
        for (const auto& flag : kInitFromConfigFlags) {
            if (args.has(flag.flag)) {
                throw ArgError(std::string(flag.flag) + " needs --init-from");
            }
        }
        return std::nullopt;
    }
    return std::make_pair(std::move(source), config);
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_INIT_FLAGS_HPP
