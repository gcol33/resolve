// The PlotInputs bundle: every row operation reaches every field through the
// one field list, undefined fields stay undefined, and the field list agrees
// with the struct and with the model's positional order.

#include <catch2/catch_test_macros.hpp>

#include "resolve/continuous_block.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_inputs.hpp"

#include "test_support.hpp"

#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// Rows i of every defined field hold the value i (times a per-field offset), so
// a row operation can be checked on each field independently.
PlotInputs numbered_inputs(int64_t n) {
    const auto rows = torch::arange(n, torch::kInt64);
    PlotInputs in;
    in.continuous = rows.to(torch::kFloat32).unsqueeze(1).repeat({1, 3});
    in.genus_ids = (rows + 100).unsqueeze(1).repeat({1, 2});
    in.species_ids = (rows + 200).unsqueeze(1).repeat({1, 4});
    in.pool_mask = (rows % 2 == 0).unsqueeze(1).repeat({1, 4});
    in.pool_has_cover = rows.to(torch::kFloat32) + 0.5f;
    in.categorical_ids = (rows + 300).unsqueeze(1);
    return in;
}

std::vector<std::string> defined_fields(const PlotInputs& in) {
    std::vector<std::string> names;
    in.for_each([&](const PlotInputField& field, const torch::Tensor& t) {
        if (t.defined()) names.emplace_back(field.name);
    });
    return names;
}

}  // namespace

TEST_CASE("The field list covers every PlotInputs member in model order", "[plot_inputs]") {
    std::vector<std::string> names;
    PlotInputs{}.for_each([&](const PlotInputField& field, const torch::Tensor&) {
        names.emplace_back(field.name);
    });
    REQUIRE(names == std::vector<std::string>{
        "continuous", "genus_ids", "family_ids", "species_ids", "species_vector",
        "pool_genus_ids", "pool_family_ids", "pool_weights", "pool_mask",
        "pool_has_cover", "categorical_ids"});
    REQUIRE(kPlotInputFields.size() == names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        CHECK(names[i] == kPlotInputFields[i].name);
    }
    CHECK(kPlotInputFields[0].dtype == torch::kFloat32);
    CHECK(kPlotInputFields[8].dtype == torch::kBool);
}

TEST_CASE("index_select and slice take the same rows of every defined field",
          "[plot_inputs]") {
    const auto in = numbered_inputs(10);
    const auto rows = torch::tensor(std::vector<int64_t>{7, 2, 5});

    const auto picked = in.index_select(rows);
    REQUIRE(defined_fields(picked) == defined_fields(in));
    CHECK(picked.n_plots() == 3);
    CHECK(torch::equal(picked.continuous.select(1, 0),
                       torch::tensor(std::vector<float>{7, 2, 5})));
    CHECK(torch::equal(picked.genus_ids.select(1, 1),
                       torch::tensor(std::vector<int64_t>{107, 102, 105})));
    CHECK(torch::equal(picked.categorical_ids.squeeze(1),
                       torch::tensor(std::vector<int64_t>{307, 302, 305})));
    CHECK(torch::equal(picked.pool_has_cover, torch::tensor(std::vector<float>{7.5f, 2.5f, 5.5f})));
    CHECK_FALSE(picked.family_ids.defined());
    CHECK_FALSE(picked.species_vector.defined());

    const auto sliced = in.slice(4, 7);
    CHECK(sliced.n_plots() == 3);
    CHECK(torch::equal(sliced.species_ids.select(1, 0),
                       torch::tensor(std::vector<int64_t>{204, 205, 206})));
    CHECK(torch::equal(sliced.pool_mask, in.pool_mask.slice(0, 4, 7)));
    CHECK_FALSE(sliced.pool_weights.defined());
}

TEST_CASE("cat stacks the parts and inverts a split", "[plot_inputs]") {
    const auto in = numbered_inputs(9);
    const auto joined = PlotInputs::cat({in.slice(0, 4), in.slice(4, 9)});
    REQUIRE(defined_fields(joined) == defined_fields(in));
    in.for_each([&](const PlotInputField& field, const torch::Tensor& t) {
        if (!t.defined()) return;
        INFO(field.name);
        torch::Tensor other;
        joined.for_each([&](const PlotInputField& f, const torch::Tensor& u) {
            if (std::string(f.name) == field.name) other = u;
        });
        CHECK(torch::equal(t, other));
    });

    SECTION("a field defined in only some parts is refused") {
        auto partial = in.slice(4, 9);
        partial.genus_ids = torch::Tensor();
        REQUIRE_THROWS_AS((PlotInputs::cat({in.slice(0, 4), partial})), std::invalid_argument);
    }
}

TEST_CASE("to and without_empty carry undefined fields through", "[plot_inputs]") {
    auto in = numbered_inputs(5);
    in.family_ids = torch::empty({5, 0}, torch::kInt64);  // the loaders' "absent"

    const auto moved = in.to(torch::kCPU);
    CHECK(defined_fields(moved) == defined_fields(in));

    const auto cleaned = in.without_empty();
    CHECK_FALSE(cleaned.family_ids.defined());
    CHECK(cleaned.genus_ids.defined());
    CHECK(defined_fields(cleaned).size() + 1 == defined_fields(in).size());

    CHECK(PlotInputs{}.n_plots() == 0);
}

TEST_CASE("The target-map row operations match PlotInputs", "[plot_inputs]") {
    const TensorMap targets{{"y", torch::arange(6, torch::kFloat32)},
                            {"c", torch::arange(6, torch::kInt64) % 3}};
    const auto rows = torch::tensor(std::vector<int64_t>{5, 0});
    const auto picked = index_select_rows(targets, rows);
    CHECK(torch::equal(picked.at("y"), torch::tensor(std::vector<float>{5, 0})));
    CHECK(torch::equal(picked.at("c"), torch::tensor(std::vector<int64_t>{2, 0})));

    const auto joined = cat_rows({index_select_rows(targets, torch::arange(0, 2)),
                                  index_select_rows(targets, torch::arange(2, 6))});
    CHECK(torch::equal(joined.at("y"), targets.at("y")));
    CHECK(torch::equal(joined.at("c"), targets.at("c")));
    REQUIRE_THROWS_AS((cat_rows({targets, TensorMap{{"y", targets.at("y")}}})),
                      std::invalid_argument);
}

TEST_CASE("ContinuousInputs slices and counts rows of its present parts",
          "[plot_inputs][continuous]") {
    ContinuousInputs parts;
    parts.coordinates = torch::arange(12, torch::kFloat32).reshape({6, 2});
    parts.unknown_fraction = torch::linspace(0.0, 1.0, 6);
    CHECK(parts.n_rows() == 6);

    const auto sliced = parts.slice(2, 5);
    CHECK(sliced.n_rows() == 3);
    CHECK(torch::equal(sliced.coordinates, parts.coordinates.slice(0, 2, 5)));
    CHECK_FALSE(sliced.covariates.defined());

    std::vector<std::string> names;
    ContinuousInputs{}.for_each([&](const char* name, const torch::Tensor&) {
        names.emplace_back(name);
    });
    CHECK(names == std::vector<std::string>{"coordinates", "covariates", "unknown_fraction",
                                            "unknown_count", "hash_embedding"});
    CHECK(ContinuousInputs{}.n_rows() == 0);
}

TEST_CASE("forward_single reads the pool inputs a rank_pool model needs",
          "[plot_inputs][model]") {
    const SmallCorpus corpus;
    auto ds = corpus.load(SpeciesEncodingMode::RankPool);
    torch::manual_seed(3);
    ResolveModel model(ds.schema(), small_encoder_config(SpeciesEncodingMode::RankPool, {16, 8}));
    model->eval();
    torch::NoGradGuard no_grad;

    PlotInputs inputs = ds.plot_inputs().without_empty();
    inputs.continuous = torch::randn({ds.n_plots(), ds.continuous_block(false).size(1)});
    const auto single = model->forward_single("y", inputs);
    const auto full = model->forward(inputs).at("y");
    CHECK(torch::allclose(single, full));
}
