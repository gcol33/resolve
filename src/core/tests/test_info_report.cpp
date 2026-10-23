// Tests for what `resolve info` prints about a checkpoint.
//
// `info` prints the checkpoint description (checkpoint_description, the
// document the JSON sidecar holds), so every config row comes from the field
// registry. These cases pin the model block's filter -- every field prints
// unless another field of the same struct switches its feature off -- that the
// filter's name-keyed rules still match real registry rows, and the layout of
// nested sub-configs and lists. The renderer takes an ostream so a block can be
// rendered into a stringstream here; info_cmd.cpp passes std::cout.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "resolve/architectures.hpp"
#include "resolve/config_json.hpp"
#include "resolve/types.hpp"

#include "../cli/config_report.hpp"

using resolve::ModelConfig;

namespace {

std::string render_model_block(const ModelConfig& config) {
    std::ostringstream out;
    resolve_cli::print_block(out, resolve::config_to_json(config), "Model",
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

std::size_t count(const std::string& text, const std::string& what) {
    std::size_t n = 0;
    for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
    return n;
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

    // The sub-configs of the architectures this checkpoint did not select.
    // Reporting them would read as configuration the model uses.
    for (const auto& name : model_field_names()) {
        const auto* entry = std::find_if(
            std::begin(resolve::kArchitectures), std::end(resolve::kArchitectures),
            [&name](const resolve::ArchitectureEntry& e) { return name == e.member; });
        const bool is_unselected = entry != std::end(resolve::kArchitectures) &&
                                   entry->architecture != config.encoder_architecture;
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
    for (const auto& entry : resolve::kArchitectures) {
        INFO("architecture member: " << entry.member);
        REQUIRE(std::find(names.begin(), names.end(), std::string(entry.member)) != names.end());
    }
}

TEST_CASE("only the selected architecture's sub-config is printed", "[info]") {
    SECTION("MLP selects none of them") {
        ModelConfig config;
        config.encoder_architecture = resolve::EncoderArchitecture::MLP;
        const std::string text = render_model_block(config);

        REQUIRE(has_row(text, "encoder_architecture"));
        for (const auto& entry : resolve::kArchitectures) {
            INFO("architecture member: " << entry.member);
            REQUIRE_FALSE(has_row(text, entry.member));
        }
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
        config.parallel_layers.branches[1].branch_weight = 0.5f;
        const std::string text = render_model_block(config);

        REQUIRE(has_row(text, "tabm"));
        REQUIRE(has_row(text, "parallel_layers"));
        REQUIRE(has_row(text, "head_activation"));
        REQUIRE(text.find("  Model head_hidden_dims: [64, 32]") != std::string::npos);

        // Each branch is one "- " item under the parallel block, its first
        // field on the item line and the rest indented under it.
        REQUIRE(text.find("\n    branches:\n") != std::string::npos);
        REQUIRE(count(text, "\n      - hidden_dims: ") == 2);
        REQUIRE(text.find("\n        branch_weight: 0.5\n") != std::string::npos);
    }
}

TEST_CASE("a description holds only the rows a checkpoint carries", "[info]") {
    // The device, the AMP / cuDNN switches, the checkpoint destination and the
    // log callback carry no key: a checkpoint reads them back as defaults, so
    // describing them would report what the run never recorded.
    const auto train = resolve::config_to_json(resolve::TrainConfig{});
    CHECK(train.contains("batch_size"));
    CHECK(train.contains("loss_config"));
    CHECK_FALSE(train.contains("device"));
    CHECK_FALSE(train.contains("use_amp"));
    CHECK_FALSE(train.contains("checkpoint_dir"));
    CHECK_FALSE(train.contains("log"));

    // Every loader setting is carried.
    const auto dataset = resolve::config_to_json(resolve::DatasetConfig{});
    std::size_t rows = 0;
    resolve::for_each_field(resolve::DatasetConfig{},
                            [&](const char* name, const char*, const auto&) {
                                INFO("field: " << name);
                                CHECK(dataset.contains(name));
                                ++rows;
                            });
    CHECK(dataset.size() == rows);

    std::ostringstream out;
    resolve_cli::print_block(out, dataset, "Dataset");
    CHECK(out.str().find("  Dataset species_budget: 0\n") != std::string::npos);
    CHECK(out.str().find("  Dataset use_taxonomy: yes\n") != std::string::npos);
}

TEST_CASE("info prints a null as NA and a list of objects as items", "[info]") {
    resolve::json::Value object = resolve::json::Value::object();
    object.set("best_loss", std::numeric_limits<float>::infinity());
    object.set("names", resolve::json::Value::array_of(std::vector<std::string>{"a", "b"}));
    resolve::json::Value targets = resolve::json::Value::array();
    resolve::json::Value target = resolve::json::Value::object();
    target.set("name", "area");
    target.set("task", "regression");
    targets.push_back(target);
    object.set("targets", targets);

    std::ostringstream out;
    resolve_cli::print_block(out, object, "");
    CHECK(out.str() ==
          "  best_loss: NA\n"
          "  names: [a, b]\n"
          "  targets:\n"
          "    - name: area\n"
          "      task: regression\n");
}
