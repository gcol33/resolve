// Tests for what `resolve info` prints about a checkpoint's model.
//
// The Model Configuration block used to be a hand-written std::cout line per
// field plus a switch over the architectures, while the Data Encoding and
// Training Configuration blocks below it were already driven by the shared
// field registry. A ModelConfig field therefore did NOT appear in `info` for
// free -- moe_placement needed a line added by hand -- which is the same
// missing-consumer shape as the defects issue #108 closed everywhere else.
//
// The block is now one registry pass with a filter, and these cases pin both
// halves of that: every field prints unless another field of the same struct
// switches its feature off, and the filter's name-keyed rules still match real
// registry rows. The writer takes an ostream so the block can be rendered into
// a stringstream here; info_cmd.cpp passes std::cout.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "resolve/types.hpp"

#include "../cli/config_report.hpp"

using resolve::ModelConfig;

namespace {

std::string render_model_block(const ModelConfig& config) {
    std::ostringstream out;
    resolve_cli::print_config_block(out, "Model", config, /*persisted_only=*/false,
                                    resolve_cli::model_field_filter(config));
    return out.str();
}

// A top-level row is "  Model <name>: <value>" for a scalar and
// "  Model <name>:" for a sub-config, so the label and the colon together
// identify one without matching a field whose name it prefixes (top_k does not
// match top_k_species).
bool has_row(const std::string& text, const std::string& name) {
    return text.find("  Model " + name + ":") != std::string::npos;
}

std::vector<std::string> model_field_names() {
    std::vector<std::string> names;
    ModelConfig config;
    resolve::for_each_field(config, [&names](const char* name, const char*, const auto&) {
        names.emplace_back(name);
    });
    return names;
}

// Everything the filter can gate, switched on, so only the sub-configs of the
// architectures this config did NOT select are legitimately absent.
ModelConfig fully_enabled_config() {
    ModelConfig config;
    config.encoder_architecture = resolve::EncoderArchitecture::FTTransformer;
    config.moe_routing = resolve::MoERoutingType::Soft;
    config.moe_placement = resolve::MoEPlacement::Post;
    config.tabm.enabled = true;
    config.parallel_layers.enabled = true;
    config.parallel_layers.branches.resize(2);
    config.head_hidden_dims = {64, 32};
    return config;
}

}  // namespace

TEST_CASE("info reports every ModelConfig field it is not told to hide", "[info]") {
    const ModelConfig config = fully_enabled_config();
    const std::string text = render_model_block(config);

    // The sub-configs of the six architectures this checkpoint did not select.
    // Reporting them would read as configuration the model uses.
    const std::vector<std::string> unselected = {"tabnet",      "saint",       "gnn",
                                                 "trait_net",   "excelformer", "heterogeneous_gnn"};

    for (const auto& name : model_field_names()) {
        const bool is_unselected =
            std::find(unselected.begin(), unselected.end(), name) != unselected.end();
        INFO("field: " << name);
        REQUIRE(has_row(text, name) == !is_unselected);
    }
}

TEST_CASE("every conditional field name is a real registry row", "[info]") {
    // A renamed field would otherwise leave a rule in model_field_shown matching
    // nothing, and the row it used to gate would start printing unconditionally
    // with nothing to say so.
    const auto names = model_field_names();
    for (const char* conditional : resolve_cli::kConditionalModelFields) {
        INFO("conditional field: " << conditional);
        REQUIRE(std::find(names.begin(), names.end(), std::string(conditional)) != names.end());
    }
}

TEST_CASE("only the selected architecture's sub-config is printed", "[info]") {
    SECTION("MLP selects none of them") {
        ModelConfig config;
        config.encoder_architecture = resolve::EncoderArchitecture::MLP;
        const std::string text = render_model_block(config);

        REQUIRE(has_row(text, "encoder_architecture"));
        REQUIRE_FALSE(has_row(text, "ft_transformer"));
        REQUIRE_FALSE(has_row(text, "tabnet"));
        REQUIRE_FALSE(has_row(text, "saint"));
        REQUIRE_FALSE(has_row(text, "gnn"));
        REQUIRE_FALSE(has_row(text, "trait_net"));
        REQUIRE_FALSE(has_row(text, "excelformer"));
        REQUIRE_FALSE(has_row(text, "heterogeneous_gnn"));
    }

    SECTION("TabNet selects its own and no other") {
        ModelConfig config;
        config.encoder_architecture = resolve::EncoderArchitecture::TabNet;
        const std::string text = render_model_block(config);

        REQUIRE(has_row(text, "tabnet"));
        REQUIRE_FALSE(has_row(text, "ft_transformer"));
        REQUIRE_FALSE(has_row(text, "saint"));

        // The sub-config's own rows are placed by indentation, without the outer
        // label: "Model n_steps" would read as a ModelConfig field.
        REQUIRE(text.find("\n    n_steps: ") != std::string::npos);
        REQUIRE(text.find("  Model n_steps:") == std::string::npos);
    }
}

TEST_CASE("a nested row is not gated by the outer struct's rules", "[info]") {
    // ModelConfig and FTTransformerConfig both carry n_heads. The filter is
    // name-keyed, so it must reach the outer row only -- a rule written for one
    // struct silencing a same-named field in another would be invisible.
    ModelConfig config;
    config.encoder_architecture = resolve::EncoderArchitecture::FTTransformer;
    config.n_heads = 7;
    config.ft_transformer.n_heads = 3;
    const std::string text = render_model_block(config);

    REQUIRE(text.find("  Model n_heads: 7") != std::string::npos);
    REQUIRE(text.find("\n    n_heads: 3") != std::string::npos);
}

TEST_CASE("the mixture's hyperparameters follow moe_routing", "[info]") {
    SECTION("routing off says so and hides the rest") {
        ModelConfig config;
        config.moe_routing = resolve::MoERoutingType::None;
        const std::string text = render_model_block(config);

        REQUIRE(text.find("  Model moe_routing: none") != std::string::npos);
        REQUIRE_FALSE(has_row(text, "moe_placement"));
        REQUIRE_FALSE(has_row(text, "n_experts"));
        REQUIRE_FALSE(has_row(text, "expert_hidden_dims"));
        REQUIRE_FALSE(has_row(text, "moe_top_k"));
        REQUIRE_FALSE(has_row(text, "moe_noise_std"));
        REQUIRE_FALSE(has_row(text, "moe_aux_loss_weight"));
    }

    SECTION("routing on reports the placement") {
        ModelConfig config;
        config.moe_routing = resolve::MoERoutingType::TopK;
        config.moe_placement = resolve::MoEPlacement::Tail;
        const std::string text = render_model_block(config);

        REQUIRE(text.find("  Model moe_routing: topk") != std::string::npos);
        REQUIRE(text.find("  Model moe_placement: tail") != std::string::npos);
        REQUIRE(has_row(text, "n_experts"));
        REQUIRE(has_row(text, "moe_aux_loss_weight"));
    }
}

TEST_CASE("TabM, the parallel branches and the head follow their own switches", "[info]") {
    SECTION("all three off") {
        ModelConfig config;
        const std::string text = render_model_block(config);

        REQUIRE_FALSE(config.tabm.enabled);
        REQUIRE_FALSE(config.parallel_layers.enabled);
        REQUIRE(config.head_hidden_dims.empty());
        REQUIRE_FALSE(has_row(text, "tabm"));
        REQUIRE_FALSE(has_row(text, "parallel_layers"));
        REQUIRE_FALSE(has_row(text, "head_activation"));
        REQUIRE_FALSE(has_row(text, "head_dropout"));

        // An empty head is reported as empty rather than by going silent.
        REQUIRE(text.find("  Model head_hidden_dims: []") != std::string::npos);
    }

    SECTION("all three on") {
        ModelConfig config = fully_enabled_config();
        const std::string text = render_model_block(config);

        REQUIRE(has_row(text, "tabm"));
        REQUIRE(has_row(text, "parallel_layers"));
        REQUIRE(has_row(text, "head_activation"));
        REQUIRE(text.find("  Model head_hidden_dims: [64, 32]") != std::string::npos);

        // Each branch is numbered under the parallel block.
        REQUIRE(text.find("\n    branches: 2") != std::string::npos);
        REQUIRE(text.find("\n      branch 0:") != std::string::npos);
        REQUIRE(text.find("\n      branch 1:") != std::string::npos);
        REQUIRE(text.find("\n        branch_weight: ") != std::string::npos);
    }
}

TEST_CASE("persisted_only drops the rows a checkpoint does not carry", "[info]") {
    // DatasetConfig travels on ResolveSchema rather than under its own keys, so
    // every row's registry key is empty: asking for the persisted subset asks
    // for nothing, while the unfiltered block is what `info` prints.
    resolve::DatasetConfig config;

    std::ostringstream persisted;
    resolve_cli::print_config_block(persisted, "Dataset", config, /*persisted_only=*/true);
    REQUIRE(persisted.str().empty());

    std::ostringstream full;
    resolve_cli::print_config_block(full, "Dataset", config);
    REQUIRE(full.str().find("  Dataset species_budget: ") != std::string::npos);
}
