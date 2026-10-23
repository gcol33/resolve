// RESOLVE CLI - configuration flags from the field registry
//
// `resolve train` reaches every DatasetConfig, ModelConfig and TrainConfig
// field and every architecture sub-config field through flags generated from
// the field registry (config_registry.hpp) that drives the checkpoint, the
// C-ABI value tree, nanobind and `resolve info`: ConfigFlagWriter emits one row
// per field, with the struct's own default as the row's default, and
// ConfigFlagReader puts the parsed value back. A field added to any of these
// structs gets its flag, its help line and its read in the edit that adds it.
//
// Two naming schemes:
//
//   * a top-level struct's field is named after its member with dashes
//     (`batch_size` -> --batch-size); a table per struct overrides a name, a
//     help text, or leaves a field without a flag (one derived from the data or
//     from another struct), and may state when the field applies;
//   * a sub-config's field is named after its CHECKPOINT KEY with dashes
//     (`ft_d_model` -> --ft-d-model), because that key already carries the
//     struct prefix and nine sub-configs repeat member names (`n_heads` is in
//     four of them). A sub-config field without a key is not persisted and gets
//     no flag.
//
// A boolean is a pair of presence flags, --x and --no-x, the later one on the
// command line winning.
//
// Like config_report.hpp, this is CLI-side: linked into the CLI binary and
// reachable from a test by relative path.

#ifndef RESOLVE_CLI_CONFIG_FLAGS_HPP
#define RESOLVE_CLI_CONFIG_FLAGS_HPP

#include <cstdint>
#include <cstring>
#include <deque>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "resolve/architectures.hpp"
#include "resolve/config_registry.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/number_text.hpp"
#include "resolve/types.hpp"

#include "arg_parser.hpp"

namespace resolve_cli {

// The configs a gate reads: the values parsed so far for the command line.
struct FlagContext {
    const resolve::DatasetConfig& dataset;
    const resolve::ModelConfig& model;
    const resolve::TrainConfig& train;
};

// When a field applies. Returns null when it does, otherwise what it needs
// ("--encoding transformer"). A flag given for a field that does not apply is
// refused rather than ignored.
using FlagGate = const char* (*)(const FlagContext&);

// How one field of a top-level struct appears on the command line. A field
// without a row is named after its member, with the struct label as its help.
struct FieldFlag {
    const char* field;        // the registry member name
    const char* flag;         // "--encoding"; for a bool the flag that sets it;
                              // null: named after the member; kNoFlag: none
    const char* help;         // null: "<label> <field>"
    FlagGate gate = nullptr;  // null: always applies
};

// A field the command line does not set: it is derived from the data or from
// another struct's flag, or reached through a flag of its own (--cuda).
inline constexpr const char* kNoFlag = "";

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

inline std::string off_flag(const std::string& on) { return "--no-" + on.substr(2); }

template <typename T>
std::string value_text(const T& value) {
    if constexpr (std::is_same_v<T, float>) {
        return resolve::number_to_decimal(value);
    } else {
        std::ostringstream out;
        out << value;
        return out.str();
    }
}

template <typename Seq>
std::string join_values(const Seq& values) {
    std::string out;
    for (const auto& value : values) {
        if (!out.empty()) out += ",";
        out += value_text(value);
    }
    return out;
}

// The field types a command line cannot carry, each reached another way: the
// device through --cuda, the parallel branches through --parallel-branch, a
// sub-config through its own block, the log callback not at all.
template <typename T>
inline constexpr bool reached_elsewhere_v =
    std::is_same_v<T, torch::Device> || std::is_same_v<T, resolve::LogCallback> ||
    std::is_same_v<T, std::vector<resolve::ParallelBranchConfig>> ||
    resolve::is_registered_config_v<T>;

}  // namespace config_flags_detail

// Where a struct's flags take their names from, and the struct's override
// table (empty for a sub-config).
struct FlagNaming {
    bool by_key = false;
    const FieldFlag* overrides = nullptr;
    std::size_t n_overrides = 0;

    template <std::size_t N>
    static FlagNaming members(const FieldFlag (&table)[N]) {
        return FlagNaming{false, table, N};
    }
    static FlagNaming keys() { return FlagNaming{true, nullptr, 0}; }

    [[nodiscard]] const FieldFlag* find(const char* name) const {
        for (std::size_t i = 0; i < n_overrides; ++i) {
            if (std::strcmp(overrides[i].field, name) == 0) return &overrides[i];
        }
        return nullptr;
    }

    // The flag of one field, or "" when it has none.
    [[nodiscard]] std::string flag_of(const char* name, const char* key) const {
        if (by_key) return resolve::has_checkpoint_key(key) ? config_flags_detail::dashed(key) : "";
        const FieldFlag* row = find(name);
        if (row == nullptr || row->flag == nullptr) return config_flags_detail::dashed(name);
        return row->flag;
    }
};

// Emits the rows for one configuration struct. `label` names the struct in the
// generic help text ("FT-Transformer"), and the default column shows the value
// the struct carries, so `resolve help` states the real defaults.
struct ConfigFlagWriter {
    std::vector<FlagSpec>& flags;
    std::string label;
    FlagNaming naming;

    void row(const std::string& flag, Arity arity, const char* value_label,
             std::string default_value, std::string help) const {
        flags.push_back({config_flags_detail::own(flag), arity, value_label,
                         config_flags_detail::own(std::move(default_value)),
                         config_flags_detail::own(std::move(help))});
    }

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        if constexpr (config_flags_detail::reached_elsewhere_v<T>) {
            (void)name;
            (void)key;
            (void)value;
        } else {
            const std::string flag = naming.flag_of(name, key);
            if (flag.empty()) return;
            const FieldFlag* override_row = naming.find(name);
            std::string help = override_row != nullptr && override_row->help != nullptr
                                   ? std::string(override_row->help)
                                   : label + " " + name;

            if constexpr (std::is_same_v<T, bool>) {
                row(flag, Arity::Flag, "", "", help + " (default " + (value ? "on" : "off") + ")");
                row(config_flags_detail::off_flag(flag), Arity::Flag, "", "",
                    "Turns " + flag + " off");
            } else if constexpr (std::is_enum_v<T>) {
                row(flag, Arity::Value, "V", resolve::enum_to_name(value),
                    help + ": " + resolve::enum_spellings<T>());
            } else if constexpr (std::is_same_v<T, int>) {
                row(flag, Arity::Value, "N", std::to_string(value), help);
            } else if constexpr (std::is_same_v<T, float>) {
                row(flag, Arity::Value, "FLOAT", config_flags_detail::value_text(value), help);
            } else if constexpr (std::is_same_v<T, std::string>) {
                row(flag, Arity::Value, "S", value, help);
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                                 std::is_same_v<T, std::vector<float>>) {
                row(flag, Arity::Value, "LIST", config_flags_detail::join_values(value),
                    help + ", comma-separated");
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                row(flag, Arity::Value, "N,N",
                    std::to_string(value.first) + "," + std::to_string(value.second), help);
            } else {
                static_assert(resolve::registry_detail::always_false<T>,
                              "config field type has no command-line form; "
                              "add a branch here and to ConfigFlagReader");
            }
        }
    }
};

// Reads the rows back onto the struct. An omitted flag reports the default the
// writer captured from a freshly constructed config, so a field nobody
// mentioned comes back at its struct default -- which is why this runs before
// anything sets a field in code.
struct ConfigFlagReader {
    const ParsedArgs& args;
    FlagNaming naming;

    template <typename T>
    void operator()(const char* name, const char* key, T& value) const {
        if constexpr (config_flags_detail::reached_elsewhere_v<T>) {
            (void)name;
            (void)key;
            (void)value;
        } else {
            const std::string flag = naming.flag_of(name, key);
            if (flag.empty()) return;

            if constexpr (std::is_same_v<T, bool>) {
                value = args.get_switch(flag, config_flags_detail::off_flag(flag), value);
            } else if constexpr (std::is_enum_v<T>) {
                const std::string text = args.get(flag);
                try {
                    value = resolve::enum_from_name<T>(text);
                } catch (const std::exception&) {
                    throw ArgError(flag + ": unknown value '" + text + "'; accepted: " +
                                   resolve::enum_spellings<T>());
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
                    parsed.push_back(parse_int64_value(flag, item));
                }
                value = std::move(parsed);
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                std::vector<float> parsed;
                for (const auto& item : args.get_list(flag)) {
                    parsed.push_back(parse_float_value(flag, item));
                }
                value = std::move(parsed);
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                const auto items = args.get_list(flag);
                if (items.size() != 2) {
                    throw ArgError(flag + " expects two integers N,N, got '" + args.get(flag) +
                                   "'");
                }
                value = {parse_int_value(flag, items[0]), parse_int_value(flag, items[1])};
            } else {
                static_assert(resolve::registry_detail::always_false<T>,
                              "config field type has no command-line form; "
                              "add a branch here and to ConfigFlagWriter");
            }
        }
    }
};

// ============================================================================
// Top-level structs
// ============================================================================

namespace flag_gates {

inline const char* transformer(const FlagContext& c) {
    return c.dataset.species_encoding == resolve::SpeciesEncodingMode::Transformer
               ? nullptr
               : "--encoding transformer";
}
inline const char* pooled(const FlagContext& c) {
    return c.dataset.species_encoding == resolve::SpeciesEncodingMode::RankPool ||
                   c.dataset.species_encoding == resolve::SpeciesEncodingMode::Transformer
               ? nullptr
               : "--encoding rank_pool or transformer";
}
inline const char* hash(const FlagContext& c) {
    return c.dataset.species_encoding == resolve::SpeciesEncodingMode::Hash ? nullptr
                                                                             : "--encoding hash";
}
inline const char* mixture(const FlagContext& c) {
    return c.model.moe_routing != resolve::MoERoutingType::None ? nullptr
                                                                 : "--moe-routing soft or topk";
}
inline const char* head_layers(const FlagContext& c) {
    return c.model.head_hidden_dims.empty() ? "--head-hidden-dims" : nullptr;
}
inline const char* step_schedule(const FlagContext& c) {
    return c.train.lr_scheduler == resolve::LRSchedulerType::StepLR ? nullptr
                                                                     : "--lr-scheduler step";
}
inline const char* plateau_schedule(const FlagContext& c) {
    return c.train.lr_scheduler == resolve::LRSchedulerType::ReduceOnPlateau
               ? nullptr
               : "--lr-scheduler plateau";
}
inline const char* floored_schedule(const FlagContext& c) {
    return c.train.lr_scheduler == resolve::LRSchedulerType::CosineAnnealing ||
                   c.train.lr_scheduler == resolve::LRSchedulerType::ReduceOnPlateau
               ? nullptr
               : "--lr-scheduler cosine or plateau";
}
inline const char* nca_loss(const FlagContext& c) {
    return c.train.loss_config == resolve::LossConfigMode::NCA ? nullptr : "--loss-config nca";
}
inline const char* amp(const FlagContext& c) { return c.train.use_amp ? nullptr : "--amp"; }
inline const char* checkpoints(const FlagContext& c) {
    return c.train.checkpoint_dir.empty() ? "--checkpoint-dir" : nullptr;
}

}  // namespace flag_gates

inline constexpr FieldFlag kDatasetConfigFlags[] = {
    {"species_encoding", "--encoding", "Species encoding"},
    {"hash_dim", nullptr, "Hash dimension"},
    {"top_k", nullptr, "Top-k species per taxonomy slot"},
    {"top_k_species", nullptr, "Species slots kept per plot in embed mode"},
    {"selection", nullptr, "Which species to keep"},
    {"representation", nullptr,
     "The value each species carries into every\nencoding (presence_absence: 1.0)"},
    {"normalization", nullptr, "Abundance normalization of the hash weights"},
    {"aggregation", nullptr,
     "How a plot's taxa are weighed: the fixed genus/family\nslots rank them by summed abundance or by species\nheld; rank_pool / transformer pool them by species\nweight or once per species"},
    {"track_unknown_fraction", "--unknown-fraction", "The unknown-species-fraction feature"},
    {"track_unknown_count", "--unknown-count", "The unknown-species-count feature"},
    {"use_taxonomy", "--taxonomy", "Read the genus/family columns when present"},
    {"use_cuda_hash", nullptr,
     "Hash species on the GPU each batch instead of\nprecomputing the embedding",
     &flag_gates::hash},
    {"pool_weighting", nullptr,
     "Per-species pooling weight for the rank_pool /\ntransformer encoders"},
    {"pool_species_cap", nullptr,
     "Species-per-plot cap for the pool encoders:\n0 = no cap, -x = the (100 - x)th percentile of\n"
     "species per plot (-1 = p99, -5 = p95; x in 1..99),\n>0 = manual cap"},
    {"species_budget", nullptr,
     "Species kept per plot by --selection for the\nrank_pool / transformer / sparse encodings:\n"
     "0 = no budget (encode every species). hash uses\n--top-k and embed uses --top-k-species instead."},
    {"missing_values", nullptr,
     "How a missing covariate or coordinate enters the\nmodel: indicate (fill with the fitting fold's mean\n"
     "and add a 0/1 missingness column) or zero"},
    {"zero_abundance_as", nullptr,
     "Abundance a recorded 0 is read at. Archives that\nwrite a presence-only record as 0% cover need a\n"
     "positive value here, or a log1p weighting gives\nthose species no weight. 0 = leave a 0 alone."},
};

inline constexpr FieldFlag kModelConfigFlags[] = {
    // Read off the dataset: the DatasetConfig flags above set them, and the
    // taxonomy slot count is the width the loader built.
    {"species_encoding", kNoFlag, nullptr},
    {"uses_explicit_vector", kNoFlag, nullptr},
    {"hash_dim", kNoFlag, nullptr},
    {"top_k", kNoFlag, nullptr},
    {"top_k_species", kNoFlag, nullptr},
    {"n_taxonomy_slots", kNoFlag, nullptr},
    {"species_embed_dim", nullptr, "Species embedding width (embed, rank_pool)"},
    {"genus_emb_dim", nullptr, "Genus embedding width"},
    {"family_emb_dim", nullptr, "Family embedding width"},
    {"categorical_embed_dim", nullptr, "Embedding width of each categorical column"},
    {"hidden_dims", nullptr, "MLP hidden layer widths"},
    {"dropout", nullptr, "Dropout rate in the encoder MLP"},
    {"moe_routing", nullptr, "Mixture of experts routing"},
    {"n_experts", nullptr, "Expert networks in the mixture (>= 2)", &flag_gates::mixture},
    {"expert_hidden_dims", nullptr, "Hidden widths inside each expert", &flag_gates::mixture},
    {"moe_top_k", nullptr, "Experts activated per sample under topk routing",
     &flag_gates::mixture},
    {"moe_noise_std", nullptr, "Gate-logit noise during training, for load balancing",
     &flag_gates::mixture},
    {"moe_aux_loss_weight", nullptr, "Weight on the mixture's load-balancing loss",
     &flag_gates::mixture},
    {"moe_placement", nullptr,
     "Where the mixture sits: tail (it replaces the\nencoder's final MLP stage) or post (it runs over\n"
     "the finished latent). Only post is available for a\nnon-MLP --encoder-architecture"},
    {"activation", nullptr, "Activation of the encoder MLP"},
    {"normalization", "--norm-layer", "Normalization layer of the encoder MLP"},
    {"norm_groups", nullptr, "Groups of a group_norm layer"},
    {"use_residual", "--residual", "Residual connections in the encoder MLP"},
    {"leaky_relu_slope", nullptr, "Negative slope of leaky_relu"},
    {"elu_alpha", nullptr, "Alpha of elu"},
    {"head_hidden_dims", nullptr, "Hidden widths of each prediction head (empty: linear)"},
    {"head_activation", nullptr, "Activation of the prediction heads",
     &flag_gates::head_layers},
    {"head_dropout", nullptr, "Dropout in the prediction heads", &flag_gates::head_layers},
    {"encoder_architecture", nullptr, "Encoder architecture"},
    {"cover_dropout", nullptr,
     "Probability of dropping a plot's cover values in\nthe rank_pool / transformer encoders",
     &flag_gates::pooled},
    {"freeze_composition", nullptr,
     "Keep the species, genus and family tables at their\ninitialisation instead of learning them"},
    {"d_model", nullptr, "Transformer token dimension", &flag_gates::transformer},
    {"n_heads", nullptr, "Transformer attention heads", &flag_gates::transformer},
    {"n_attention_layers", nullptr,
     "Transformer self-attention layers. Required >= 1\nwhen --transformer-pooling cls",
     &flag_gates::transformer},
    {"transformer_ff_dim", nullptr, "Feed-forward width inside each transformer block",
     &flag_gates::transformer},
    {"transformer_pooling", nullptr, "Transformer pooling: attention or cls",
     &flag_gates::transformer},
    {"transformer_dropout", nullptr, "Dropout inside transformer blocks",
     &flag_gates::transformer},
};

inline constexpr FieldFlag kTrainConfigFlags[] = {
    {"batch_size", nullptr, "Batch size"},
    {"max_epochs", nullptr, "Maximum epochs"},
    {"patience", nullptr, "Early stopping patience"},
    {"fixed_epochs", nullptr,
     "Train exactly N epochs of the --max-epochs\nschedule, with no early stopping, and keep the\n"
     "final weights (0 = early stopping). The only mode\nthat accepts --test-size 0, for a refit on every\n"
     "plot."},
    {"lr", nullptr, "Learning rate"},
    {"weight_decay", nullptr, "AdamW weight decay"},
    {"phase_boundaries", nullptr,
     "Epochs at which the combined loss adds SMAPE,\nthen the band penalty"},
    {"loss_config", nullptr, "Loss recipe"},
    {"device", kNoFlag, nullptr},
    {"lr_scheduler", nullptr,
     "Learning rate schedule (plateau decays when the\nheld-out loss stops improving)"},
    {"lr_step_size", nullptr, "Epochs between decays", &flag_gates::step_schedule},
    {"lr_gamma", nullptr, "Decay factor", &flag_gates::step_schedule},
    {"lr_min", nullptr,
     "Floor learning rate: where the cosine schedule ends,\nbelow which a plateau never decays",
     &flag_gates::floored_schedule},
    {"lr_plateau_factor", nullptr,
     "Factor a plateau multiplies the learning rate by,\nin (0, 1)",
     &flag_gates::plateau_schedule},
    {"lr_plateau_patience", nullptr,
     "Epochs without improvement of the held-out loss\na plateau allows; the next one decays the rate",
     &flag_gates::plateau_schedule},
    {"lr_plateau_threshold", nullptr,
     "Relative improvement on the best held-out loss an\nepoch must make to count as one (0 = any\n"
     "decrease, as early stopping counts; PyTorch's\ndefault is 1e-4)",
     &flag_gates::plateau_schedule},
    {"lr_plateau_cooldown", nullptr,
     "Epochs after a decay during which no bad epoch\ncounts",
     &flag_gates::plateau_schedule},
    {"band_thresholds", nullptr,
     "Band accuracies to REPORT. Does not change what\ntraining optimizes"},
    {"band_threshold", nullptr,
     "Tolerance band the phase-3 loss penalty optimizes\ntoward (predictions outside [1-t, 1+t] times the\n"
     "target are penalized)"},
    {"nca_temperature", nullptr,
     "Scale of the stochastic-neighbour softmax in the\nNCA term, i.e. the effective number of neighbours\n"
     "each sample spreads over",
     &flag_gates::nca_loss},
    {"nca_neighbors", nullptr,
     "Neighbours per sample the NCA term sums over, its\nmost similar in-batch samples. 0 or less keeps the\n"
     "whole batch",
     &flag_gates::nca_loss},
    {"nca_weight", nullptr, "Weight of the NCA term against the cross-entropy\nit is added to",
     &flag_gates::nca_loss},
    {"checkpoint_dir", nullptr,
     "Directory for per-epoch checkpoints and the\nprogress file (empty = disabled)"},
    {"checkpoint_every", nullptr, "Also checkpoint every N epochs (0 = best only)",
     &flag_gates::checkpoints},
    {"log", kNoFlag, nullptr},
    {"use_amp", "--amp",
     "CUDA automatic mixed precision (helps transformer\nencoders, rarely MLPs)"},
    {"amp_init_scale", nullptr, "Initial AMP gradient scale", &flag_gates::amp},
    {"amp_growth_factor", nullptr, "AMP gradient-scale growth factor", &flag_gates::amp},
    {"amp_backoff_factor", nullptr, "AMP gradient-scale reduction on overflow",
     &flag_gates::amp},
    {"amp_growth_interval", nullptr, "Steps between AMP gradient-scale increases",
     &flag_gates::amp},
    {"cudnn_benchmark", nullptr, "The cuDNN autotuner (off for run-to-run determinism)"},
    {"allow_tf32", "--tf32", "TF32 matmuls on Ampere+ GPUs"},
    {"vram_fraction", nullptr,
     "Fraction of GPU VRAM the PyTorch caching allocator\nmay use. Pass a lower value (e.g. 0.80) when\n"
     "sharing the GPU with a desktop to leave headroom."},
    {"batch_size_floor", nullptr,
     "Smallest batch size the out-of-memory retry may\nhalve to: on a CUDA out-of-memory error the trainer\n"
     "halves the batch size and restarts from epoch 0;\nbelow this floor the error is rethrown."},
};

// The override table of each top-level struct.
inline FlagNaming flag_naming(const resolve::DatasetConfig*) {
    return FlagNaming::members(kDatasetConfigFlags);
}
inline FlagNaming flag_naming(const resolve::ModelConfig*) {
    return FlagNaming::members(kModelConfigFlags);
}
inline FlagNaming flag_naming(const resolve::TrainConfig*) {
    return FlagNaming::members(kTrainConfigFlags);
}

// One struct's flags, under a label that names it in the generic help text.
template <typename Cfg>
void append_config_flags(std::vector<FlagSpec>& flags, const char* label, const Cfg& defaults,
                         FlagNaming naming) {
    resolve::for_each_field(defaults, ConfigFlagWriter{flags, label, naming});
}

// Read one struct's flags back. Raises ArgError on an unusable value.
template <typename Cfg>
void read_config_flags(Cfg& config, const ParsedArgs& args, FlagNaming naming) {
    resolve::for_each_field(config, ConfigFlagReader{args, naming});
}

// The DatasetConfig, ModelConfig and TrainConfig flags of `resolve train`, in
// that order.
inline void append_top_level_flags(std::vector<FlagSpec>& flags) {
    append_config_flags(flags, "Dataset", resolve::DatasetConfig{},
                        flag_naming(static_cast<const resolve::DatasetConfig*>(nullptr)));
    append_config_flags(flags, "Model", resolve::ModelConfig{},
                        flag_naming(static_cast<const resolve::ModelConfig*>(nullptr)));
    append_config_flags(flags, "Train", resolve::TrainConfig{},
                        flag_naming(static_cast<const resolve::TrainConfig*>(nullptr)));
}

// Read a top-level struct's flags.
template <typename Cfg>
void read_top_level_flags(Cfg& config, const ParsedArgs& args) {
    read_config_flags(config, args, flag_naming(static_cast<const Cfg*>(nullptr)));
}

// Refuse a flag given for a field that does not apply to the configs read.
template <typename Cfg>
void check_flag_gates(const Cfg& config, const ParsedArgs& args, const FlagContext& context) {
    const FlagNaming naming = flag_naming(static_cast<const Cfg*>(nullptr));
    resolve::for_each_field(config, [&](const char* name, const char* key, const auto&) {
        const FieldFlag* row = naming.find(name);
        if (row == nullptr || row->gate == nullptr) return;
        const char* needs = row->gate(context);
        if (needs == nullptr) return;
        const std::string flag = naming.flag_of(name, key);
        if (args.has(flag)) throw ArgError(flag + " applies only with " + needs);
    });
}

// ============================================================================
// Sub-configs
// ============================================================================

// Every architecture sub-config, then TabM and the parallel block, under the
// labels `resolve info` prints.
inline void append_architecture_flags(std::vector<FlagSpec>& flags) {
    const resolve::ModelConfig defaults;
    resolve::for_each_architecture(defaults, [&flags](const resolve::ArchitectureEntry& entry,
                                                      const auto& sub_config) {
        append_config_flags(flags, entry.label, sub_config, FlagNaming::keys());
    });
    append_config_flags(flags, "TabM", defaults.tabm, FlagNaming::keys());
    append_config_flags(flags, "Parallel layers", defaults.parallel_layers, FlagNaming::keys());
}

// ... and the matching reads.
inline void read_architecture_flags(resolve::ModelConfig& config, const ParsedArgs& args) {
    resolve::for_each_architecture(
        config, [&args](const resolve::ArchitectureEntry&, auto& sub_config) {
            read_config_flags(sub_config, args, FlagNaming::keys());
        });
    read_config_flags(config.tabm, args, FlagNaming::keys());
    read_config_flags(config.parallel_layers, args, FlagNaming::keys());
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
        branch.hidden_dims.push_back(parse_int64_value("--parallel-branch width", width));
    }
    if (branch.hidden_dims.empty()) {
        throw ArgError("--parallel-branch needs at least one hidden width, "
                       "e.g. 256,128");
    }

    auto enum_field = [&spec](const std::string& text, auto& field, const char* what) {
        using FieldT = std::decay_t<decltype(field)>;
        try {
            field = resolve::enum_from_name<FieldT>(text);
        } catch (const std::exception&) {
            throw ArgError(std::string("--parallel-branch ") + what + ": unknown value '" +
                           text + "' in '" + spec + "'; accepted: " +
                           resolve::enum_spellings<FieldT>());
        }
    };
    if (parts.size() > 1 && !parts[1].empty()) {
        enum_field(parts[1], branch.activation, "activation");
    }
    if (parts.size() > 2 && !parts[2].empty()) {
        enum_field(parts[2], branch.normalization, "normalization");
    }
    if (parts.size() > 3 && !parts[3].empty()) {
        branch.dropout = parse_float_value("--parallel-branch dropout", parts[3]);
    }
    if (parts.size() > 4 && !parts[4].empty()) {
        branch.branch_weight = parse_float_value("--parallel-branch weight", parts[4]);
    }
    return branch;
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_CONFIG_FLAGS_HPP
