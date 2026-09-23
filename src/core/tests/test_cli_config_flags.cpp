#include <catch2/catch_test_macros.hpp>

#include "resolve/config_registry.hpp"
#include "resolve/types.hpp"

// CLI-side headers, reached the way test_effective_batch.cpp reaches the flag
// table: these are compiled into the CLI binary, not into the engine library.
#include "../cli/config_flags.hpp"
#include "../cli/cli_spec.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve_cli;

// ============================================================================
// The CLI reaches every architecture hyperparameter
//
// `resolve train` selected an encoder architecture and then left every field of
// every sub-config at its default: no flag existed for any of them, so a
// standalone run could not train what the bindings could -- the gap issue #104
// closed for covariates and the mixture, one level down. Writing fifty rows by
// hand would have closed it once and left the next field out again, so the
// rows and the reads come from the same field registry that drives the
// checkpoint, the C ABI, nanobind and `resolve info`.
//
// These cases pin that the generated table covers the registry, that a value
// lands on the field it names, and that an unusable value is refused.
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
