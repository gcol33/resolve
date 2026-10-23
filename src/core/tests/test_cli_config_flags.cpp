#include <catch2/catch_test_macros.hpp>

#include "resolve/config_registry.hpp"
#include "resolve/types.hpp"

// CLI-side headers, reached the way test_effective_batch.cpp reaches the flag
// table: these are compiled into the CLI binary, not into the engine library.
#include "../cli/config_flags.hpp"
#include "../cli/cli_spec.hpp"

#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace resolve;
using namespace resolve_cli;

// ============================================================================
// The CLI reaches every configuration field
//
// The DatasetConfig, ModelConfig and TrainConfig flags of `resolve train` and
// the flags of every architecture sub-config are generated from the field
// registry that drives the checkpoint, the C ABI, nanobind and `resolve info`.
// These cases pin that the generated table covers the registry, that a value
// lands on the field it names with the struct's own default, that an unusable
// value or a flag for a field that does not apply is refused, and that the
// override tables name real fields.
// ============================================================================

namespace {

std::set<std::string> flag_names(const std::vector<FlagSpec>& flags) {
    std::set<std::string> names;
    for (const auto& flag : flags) names.insert(flag.name);
    return names;
}

// Every checkpoint key of a config struct, which is what the flag names are
// generated from.
struct KeyCollector {
    std::vector<std::string>& keys;
    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        (void)name;
        (void)value;
        if (resolve::has_checkpoint_key(key)) keys.push_back(key);
    }
};

template <typename Cfg>
std::vector<std::string> keys_of(const Cfg& config) {
    std::vector<std::string> keys;
    for_each_field(config, KeyCollector{keys});
    return keys;
}

std::string dashed(const std::string& key) {
    std::string out = "--";
    for (char c : key) out.push_back(c == '_' ? '-' : c);
    return out;
}

ParsedArgs args_with(const std::vector<std::pair<std::string, std::string>>& given) {
    ParsedArgs args(&train_spec());
    for (const auto& [flag, value] : given) args.add(flag, value);
    return args;
}

}  // namespace

TEST_CASE("Every architecture sub-config field has a train flag",
          "[cli][config_flags]") {
    const auto names = flag_names(train_spec().flags());
    const ModelConfig defaults;

    auto covered = [&](const auto& config) {
        for (const auto& key : keys_of(config)) {
            const std::string flag = dashed(key);
            INFO("field key " << key << " -> " << flag);
            // A bool is a pair of presence flags, everything else a value
            // flag; either way the positive spelling is declared.
            CHECK(names.count(flag) == 1);
        }
    };
    covered(defaults.ft_transformer);
    covered(defaults.tabnet);
    covered(defaults.saint);
    covered(defaults.gnn);
    covered(defaults.trait_net);
    covered(defaults.excelformer);
    covered(defaults.heterogeneous_gnn);
    covered(defaults.tabm);

    // The branches are variable-length and carry their own grammar.
    CHECK(names.count("--parallel-branch") == 1);
    CHECK(names.count("--parallel-enabled") == 1);
    CHECK(names.count("--no-parallel-enabled") == 1);
}

TEST_CASE("The train table declares no flag twice", "[cli][config_flags]") {
    std::map<std::string, int> seen;
    for (const auto& flag : train_spec().flags()) seen[flag.name] += 1;
    for (const auto& [name, count] : seen) {
        INFO("flag " << name);
        CHECK(count == 1);
    }
}

TEST_CASE("A generated flag lands on the field it names", "[cli][config_flags]") {
    ModelConfig config;
    auto args = args_with({
        {"--ft-d-model", "64"},
        {"--ft-ffn-dropout", "0.25"},
        {"--tabnet-n-steps", "7"},
        {"--tabnet-virtual-batch-size", "256"},
        {"--no-tabnet-use-sparsemax", ""},
        {"--gnn-graph-mode", "taxonomic"},
        {"--gnn-use-edge-features", ""},
        {"--trait-interaction", "attention"},
        {"--trait-interaction-dim", "48"},
        {"--no-trait-shared-trait-encoder", ""},
        {"--hgnn-k-cooccurrence", "12"},
        {"--hgnn-cooccurrence-threshold", "0.05"},
        {"--no-hgnn-use-taxonomic-edges", ""},
        {"--tabm-n-ensembles", "5"},
        {"--excel-pre-norm", ""},
    });
    read_architecture_flags(config, args);

    CHECK(config.ft_transformer.d_model == 64);
    CHECK(config.ft_transformer.ffn_dropout == 0.25f);
    CHECK(config.tabnet.n_steps == 7);
    CHECK(config.tabnet.virtual_batch_size == 256);
    CHECK_FALSE(config.tabnet.use_sparsemax);
    CHECK(config.gnn.graph_mode == GraphConstructionMode::Taxonomic);
    CHECK(config.gnn.use_edge_features);
    CHECK(config.trait_net.interaction == TraitInteractionMode::Attention);
    CHECK(config.trait_net.interaction_dim == 48);
    CHECK_FALSE(config.trait_net.shared_trait_encoder);
    CHECK(config.heterogeneous_gnn.k_cooccurrence == 12);
    CHECK(config.heterogeneous_gnn.cooccurrence_threshold == 0.05f);
    CHECK_FALSE(config.heterogeneous_gnn.use_taxonomic_edges);
    CHECK(config.tabm.n_ensembles == 5);
    CHECK(config.excelformer.pre_norm);

    // A field nobody mentioned keeps the value it had, because the flag
    // reports the default the writer took from the same struct.
    const ModelConfig untouched;
    CHECK(config.saint.d_model == untouched.saint.d_model);
    CHECK(config.gnn.k_neighbors == untouched.gnn.k_neighbors);
    CHECK(config.tabm.enabled == untouched.tabm.enabled);
}

TEST_CASE("An unusable value for a generated flag is refused by name",
          "[cli][config_flags]") {
    ModelConfig config;

    auto bad_enum = args_with({{"--gnn-graph-mode", "sideways"}});
    CHECK_THROWS_AS(read_architecture_flags(config, bad_enum), ArgError);

    auto bad_number = args_with({{"--tabnet-n-steps", "many"}});
    CHECK_THROWS_AS(read_architecture_flags(config, bad_number), ArgError);

    // The parser itself rejects a flag the table does not declare, so a typo
    // in a sub-config name never reaches the reader.
    CHECK_THROWS_AS(parse_args(train_spec(), {"--tabnet-nsteps", "7"}), ArgError);
}

TEST_CASE("A parallel branch reads its widths and its optional fields",
          "[cli][config_flags]") {
    const std::vector<int64_t> two_widths{256, 128};
    const std::vector<int64_t> one_width{64};
    auto plain = parse_parallel_branch("256,128");
    CHECK(plain.hidden_dims == two_widths);
    // Everything after the widths keeps the struct default.
    const ParallelBranchConfig defaults;
    CHECK(plain.activation == defaults.activation);
    CHECK(plain.normalization == defaults.normalization);
    CHECK(plain.dropout == defaults.dropout);
    CHECK(plain.branch_weight == defaults.branch_weight);

    auto full = parse_parallel_branch("64:relu:layer_norm:0.15:0.5");
    CHECK(full.hidden_dims == one_width);
    CHECK(full.activation == ActivationType::ReLU);
    CHECK(full.normalization == NormLayerType::LayerNorm);
    CHECK(full.dropout == 0.15f);
    CHECK(full.branch_weight == 0.5f);

    // A skipped field keeps its default.
    auto partial = parse_parallel_branch("32::layer_norm");
    CHECK(partial.activation == defaults.activation);
    CHECK(partial.normalization == NormLayerType::LayerNorm);

    CHECK_THROWS_AS(parse_parallel_branch(""), ArgError);
    CHECK_THROWS_AS(parse_parallel_branch("wide"), ArgError);
    CHECK_THROWS_AS(parse_parallel_branch("64:sideways"), ArgError);
    CHECK_THROWS_AS(parse_parallel_branch("64:relu:layer_norm:soggy"), ArgError);
    CHECK_THROWS_AS(parse_parallel_branch("64:a:b:c:d:e"), ArgError);
}

TEST_CASE("A generated flag reports the default the struct carries",
          "[cli][config_flags]") {
    const ModelConfig defaults;
    const auto& flags = train_spec().flags();
    auto find = [&](const std::string& name) -> const FlagSpec* {
        for (const auto& flag : flags) {
            if (name == flag.name) return &flag;
        }
        return nullptr;
    };

    const FlagSpec* steps = find("--tabnet-n-steps");
    REQUIRE(steps != nullptr);
    CHECK(std::string(steps->default_value) ==
          std::to_string(defaults.tabnet.n_steps));

    const FlagSpec* mode = find("--gnn-graph-mode");
    REQUIRE(mode != nullptr);
    CHECK(std::string(mode->default_value) ==
          std::string(enum_to_name(defaults.gnn.graph_mode)));
    // The help lists what the parser accepts, from the same enum table.
    CHECK(std::string(mode->help).find("cooccurrence") != std::string::npos);
}

// ============================================================================
// Top-level structs
// ============================================================================

namespace {

std::vector<std::string> field_names_of(const auto& config) {
    std::vector<std::string> names;
    for_each_field(config, [&names](const char* name, const char*, const auto&) {
        names.emplace_back(name);
    });
    return names;
}

template <typename Cfg, std::size_t N>
void check_overrides_name_fields(const FieldFlag (&table)[N]) {
    const auto names = field_names_of(Cfg{});
    for (const auto& row : table) {
        INFO("override row " << row.field);
        CHECK(std::find(names.begin(), names.end(), std::string(row.field)) != names.end());
    }
}

template <typename Cfg>
void check_every_field_has_its_flag(const std::set<std::string>& declared) {
    const FlagNaming naming = flag_naming(static_cast<const Cfg*>(nullptr));
    for_each_field(Cfg{}, [&](const char* name, const char* key, const auto& value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (!config_flags_detail::reached_elsewhere_v<T>) {
            const std::string flag = naming.flag_of(name, key);
            INFO("field " << name << " -> " << (flag.empty() ? "(no flag)" : flag));
            const FieldFlag* row = naming.find(name);
            if (row != nullptr && row->flag == kNoFlag) {
                CHECK(flag.empty());
            } else {
                CHECK(declared.count(flag) == 1);
            }
        }
    });
}

const FlagSpec* find_flag(const std::string& name) {
    for (const auto& flag : train_spec().flags()) {
        if (name == flag.name) return &flag;
    }
    return nullptr;
}

}  // namespace

TEST_CASE("Every top-level config field has a train flag unless another source sets it",
          "[cli][config_flags]") {
    const auto declared = flag_names(train_spec().flags());
    check_every_field_has_its_flag<DatasetConfig>(declared);
    check_every_field_has_its_flag<ModelConfig>(declared);
    check_every_field_has_its_flag<TrainConfig>(declared);

    // The names a run has always been started with.
    for (const char* name : {"--encoding", "--no-taxonomy", "--no-unknown-fraction",
                             "--unknown-count", "--use-cuda-hash", "--amp", "--no-amp",
                             "--cudnn-benchmark", "--no-cudnn-benchmark", "--no-tf32",
                             "--hidden-dims", "--expert-hidden-dims", "--band-thresholds",
                             "--norm-layer", "--phase-boundaries"}) {
        INFO("flag " << name);
        CHECK(declared.count(name) == 1);
    }
    // Derived from the data or from the DatasetConfig flags.
    CHECK(declared.count("--n-taxonomy-slots") == 0);
    CHECK(declared.count("--uses-explicit-vector") == 0);
    CHECK(declared.count("--device") == 0);
}

TEST_CASE("Every override row names a real registry field", "[cli][config_flags]") {
    check_overrides_name_fields<DatasetConfig>(kDatasetConfigFlags);
    check_overrides_name_fields<ModelConfig>(kModelConfigFlags);
    check_overrides_name_fields<TrainConfig>(kTrainConfigFlags);
}

TEST_CASE("A top-level flag's default is the struct's own", "[cli][config_flags]") {
    // The transformer's attention layers: the hand-written row said 2 while the
    // struct says 0.
    const FlagSpec* layers = find_flag("--n-attention-layers");
    REQUIRE(layers != nullptr);
    CHECK(std::string(layers->default_value) == std::to_string(ModelConfig{}.n_attention_layers));

    const FlagSpec* hidden = find_flag("--hidden-dims");
    REQUIRE(hidden != nullptr);
    CHECK(std::string(hidden->default_value) == "2048,1024,512,256,128,64");

    const FlagSpec* lr_min = find_flag("--lr-min");
    REQUIRE(lr_min != nullptr);
    CHECK(std::string(lr_min->default_value) == "1e-06");

    const FlagSpec* phases = find_flag("--phase-boundaries");
    REQUIRE(phases != nullptr);
    const TrainConfig train;
    CHECK(std::string(phases->default_value) == std::to_string(train.phase_boundaries.first) +
                                                    "," +
                                                    std::to_string(train.phase_boundaries.second));
}

TEST_CASE("Top-level flags land on their fields", "[cli][config_flags]") {
    auto args = args_with({
        {"--encoding", "transformer"},
        {"--no-taxonomy", ""},
        {"--missing-values", "zero"},
        {"--zero-abundance-as", "1.5"},
        {"--hidden-dims", "64, 32"},
        {"--norm-layer", "layer_norm"},
        {"--d-model", "48"},
        {"--band-thresholds", "0.1,0.2"},
        {"--phase-boundaries", "5,9"},
        {"--amp", ""},
        {"--no-tf32", ""},
    });
    DatasetConfig dataset;
    read_top_level_flags(dataset, args);
    ModelConfig model;
    read_top_level_flags(model, args);
    TrainConfig train;
    read_top_level_flags(train, args);

    CHECK(dataset.species_encoding == SpeciesEncodingMode::Transformer);
    CHECK_FALSE(dataset.use_taxonomy);
    CHECK(dataset.missing_values == MissingValuePolicy::Zero);
    CHECK(dataset.zero_abundance_as == 1.5f);
    CHECK(model.hidden_dims == std::vector<int64_t>{64, 32});
    CHECK(model.normalization == NormLayerType::LayerNorm);
    CHECK(model.d_model == 48);
    CHECK(train.band_thresholds == std::vector<float>{0.1f, 0.2f});
    CHECK(train.phase_boundaries == std::pair<int, int>{5, 9});
    CHECK(train.use_amp);
    CHECK_FALSE(train.allow_tf32);
    // The dataset's normalization keeps its own default: the model's
    // normalization layer has its own flag.
    CHECK(dataset.normalization == DatasetConfig{}.normalization);

    const FlagContext context{dataset, model, train};
    CHECK_NOTHROW(check_flag_gates(model, args, context));
}

TEST_CASE("A flag for a field that does not apply is refused", "[cli][config_flags]") {
    auto check_refused = [](std::vector<std::pair<std::string, std::string>> given,
                            const std::string& flag) {
        auto args = args_with(given);
        DatasetConfig dataset;
        read_top_level_flags(dataset, args);
        ModelConfig model;
        read_top_level_flags(model, args);
        TrainConfig train;
        read_top_level_flags(train, args);
        const FlagContext context{dataset, model, train};
        try {
            check_flag_gates(dataset, args, context);
            check_flag_gates(model, args, context);
            check_flag_gates(train, args, context);
            FAIL("no error for " << flag);
        } catch (const ArgError& e) {
            CHECK_THAT(std::string(e.what()), Catch::Matchers::ContainsSubstring(flag));
        }
    };
    check_refused({{"--d-model", "64"}}, "--d-model");
    check_refused({{"--encoding", "embed"}, {"--use-cuda-hash", ""}}, "--use-cuda-hash");
    check_refused({{"--n-experts", "3"}}, "--n-experts");
    check_refused({{"--cover-dropout", "0.2"}}, "--cover-dropout");
    check_refused({{"--nca-weight", "0.5"}}, "--nca-weight");
    check_refused({{"--lr-gamma", "0.5"}}, "--lr-gamma");
    check_refused({{"--checkpoint-every", "5"}}, "--checkpoint-every");
}

TEST_CASE("A number with trailing characters, a non-number or an int out of range is refused",
          "[cli][config_flags]") {
    CHECK_THROWS_AS(parse_int64_value("--n", "256x"), ArgError);
    CHECK_THROWS_AS(parse_int64_value("--n", " 256"), ArgError);
    CHECK_THROWS_AS(parse_int64_value("--n", "2.5"), ArgError);
    CHECK_THROWS_AS(parse_int64_value("--n", ""), ArgError);
    CHECK(parse_int64_value("--n", "-7") == -7);
    CHECK_THROWS_AS(parse_int_value("--n", "3000000000"), ArgError);
    CHECK_THROWS_AS(parse_float_value("--x", "0.5x"), ArgError);
    CHECK_THROWS_AS(parse_float_value("--x", "1,5"), ArgError);
    CHECK_THROWS_AS(parse_float_value("--x", "inf"), ArgError);
    CHECK(parse_float_value("--x", "0.25") == 0.25f);

    ModelConfig model;
    auto bad_list = args_with({{"--hidden-dims", "64,32x"}});
    CHECK_THROWS_AS(read_top_level_flags(model, bad_list), ArgError);
    auto bad_int = args_with({{"--batch-size", "4096.0"}});
    TrainConfig train;
    CHECK_THROWS_AS(read_top_level_flags(train, bad_int), ArgError);
}

TEST_CASE("A table that declares a flag twice is refused", "[cli][config_flags]") {
    std::vector<FlagSpec> flags = {{"--x", Arity::Flag, "", "", "one"},
                                   {"--x", Arity::Flag, "", "", "two"}};
    CHECK_THROWS_AS(CommandSpec("toy", "Toy:", flags), std::logic_error);
}

TEST_CASE("The architecture table covers every encoder with a sub-config", "[cli][config_flags]") {
    // Every architecture but the MLP has a row, each sub-config has its flags,
    // and the batch-dependent one is the plot GNN, in every graph mode.
    for (const auto& entry : kEncoderArchitectureNames) {
        INFO("architecture " << entry.name);
        CHECK((find_architecture(entry.value) != nullptr) ==
              (entry.value != EncoderArchitecture::MLP));
    }
    CHECK(is_batch_dependent(EncoderArchitecture::GNN));
    CHECK_FALSE(is_batch_dependent(EncoderArchitecture::SAINT));
    CHECK_FALSE(is_batch_dependent(EncoderArchitecture::HeterogeneousGNN));
    CHECK_FALSE(is_batch_dependent(EncoderArchitecture::MLP));

    // for_each_architecture hands each row the sub-config its member names.
    ModelConfig config;
    std::vector<std::string> members;
    for_each_architecture(config, [&](const ArchitectureEntry& entry, auto& sub_config) {
        members.emplace_back(entry.member);
        for_each_field(config, [&](const char* name, const char*, const auto& value) {
            if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                         std::decay_t<decltype(sub_config)>>) {
                if (std::string(name) == entry.member) CHECK(&value == &sub_config);
            }
        });
    });
    CHECK(members.size() == std::size(kArchitectures));
}
