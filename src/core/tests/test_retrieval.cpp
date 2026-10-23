// Retrieval-augmented prediction (retrieval.hpp): the exact blocked cosine
// search, the blend of regression and classification labels against values
// worked out by hand, unlabelled neighbours, and the Predictor feature end to
// end (k = 1, alpha = 1 on the reference plots returns their own labels).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/retrieval.hpp"
#include "resolve/target_labels.hpp"
#include "resolve/trainer.hpp"

#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <sstream>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::WithinAbs;

namespace {

// The four unit vectors of the plane as reference latents: plot j lies at
// angle j * 90 degrees.
torch::Tensor compass_latents() {
    return torch::tensor({1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f}).reshape({4, 2});
}

TargetConfig regression_target(const std::string& name) {
    TargetConfig target;
    target.name = name;
    target.task = TaskType::Regression;
    return target;
}

TargetConfig classification_target(const std::string& name, int n_classes) {
    TargetConfig target;
    target.name = name;
    target.task = TaskType::Classification;
    target.num_classes = n_classes;
    return target;
}

// The softmax weights of two similarities at temperature 1.
std::pair<double, double> softmax2(double a, double b) {
    const double ea = std::exp(a);
    const double eb = std::exp(b);
    return {ea / (ea + eb), eb / (ea + eb)};
}

}  // namespace

TEST_CASE("nearest_by_cosine is the exact top k over several reference blocks", "[retrieval]") {
    torch::manual_seed(7);
    // 5000 queries make a block of 3355 reference rows, so 10000 reference
    // rows are searched in three blocks.
    const torch::Tensor queries = torch::randn({5000, 8});
    const torch::Tensor reference = torch::randn({10000, 8});
    const int64_t k = 7;

    const Neighbours found = nearest_by_cosine(queries, reference, k);
    const torch::Tensor all = torch::nn::functional::normalize(
                                  queries, torch::nn::functional::NormalizeFuncOptions().dim(1))
                                  .matmul(torch::nn::functional::normalize(
                                              reference,
                                              torch::nn::functional::NormalizeFuncOptions().dim(1))
                                              .t());
    const auto [expected_similarities, expected_indices] = all.topk(k, 1);

    REQUIRE(found.indices.sizes() == expected_indices.sizes());
    CHECK(torch::equal(found.indices, expected_indices));
    CHECK(torch::allclose(found.similarities, expected_similarities, 1e-5, 1e-6));
}

TEST_CASE("nearest_by_cosine clamps k to the reference and refuses bad shapes", "[retrieval]") {
    const torch::Tensor query = torch::tensor({1.0f, 0.2f}).reshape({1, 2});
    const Neighbours found = nearest_by_cosine(query, compass_latents(), 10);
    REQUIRE(found.indices.size(1) == 4);
    CHECK(torch::equal(found.indices[0], torch::tensor({int64_t{0}, int64_t{1}, int64_t{3},
                                                        int64_t{2}})));
    CHECK_THROWS_AS(nearest_by_cosine(torch::ones({3, 3}), compass_latents(), 2),
                    std::invalid_argument);
    CHECK_THROWS_AS(nearest_by_cosine(torch::ones({3, 2}), compass_latents(), 0),
                    std::invalid_argument);
}

TEST_CASE("RetrievalConfig validates its fields", "[retrieval]") {
    CHECK_NOTHROW(RetrievalConfig{}.validate());
    CHECK_THROWS_AS((RetrievalConfig{0, 0.3f, 1.0f}.validate()), std::invalid_argument);
    CHECK_THROWS_AS((RetrievalConfig{5, -0.1f, 1.0f}.validate()), std::invalid_argument);
    CHECK_THROWS_AS((RetrievalConfig{5, 1.5f, 1.0f}.validate()), std::invalid_argument);
    CHECK_THROWS_AS((RetrievalConfig{5, 0.3f, 0.0f}.validate()), std::invalid_argument);
    CHECK_THROWS_AS((RetrievalConfig{5, std::nanf(""), 1.0f}.validate()), std::invalid_argument);
}

TEST_CASE("A regression prediction is blended with its neighbours' labels in original units",
          "[retrieval]") {
    // Query (1, 0.1): cosine 1/sqrt(1.01) with plot 0 and 0.1/sqrt(1.01) with
    // plot 1, its two nearest.
    const torch::Tensor query = torch::tensor({1.0f, 0.1f}).reshape({1, 2});
    const double s0 = 1.0 / std::sqrt(1.01);
    const double s1 = 0.1 / std::sqrt(1.01);
    const RetrievalConfig config{2, 0.5f, 1.0f};

    SECTION("every neighbour labelled") {
        const RetrievalIndex index(compass_latents(), {regression_target("y")},
                                   {{"y", torch::tensor({10.0f, 20.0f, 30.0f, 40.0f})}}, config);
        TensorMap predictions{{"y", torch::tensor({100.0f})}};
        TensorMap probabilities;
        index.blend(query, predictions, probabilities);
        const auto [w0, w1] = softmax2(s0, s1);
        const double expected = 0.5 * 100.0 + 0.5 * (w0 * 10.0 + w1 * 20.0);
        CHECK_THAT(predictions.at("y").item<double>(), WithinAbs(expected, 1e-4));
    }
    SECTION("an unlabelled neighbour gives its weight to the labelled one") {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const RetrievalIndex index(compass_latents(), {regression_target("y")},
                                   {{"y", torch::tensor({nan, 20.0f, 30.0f, 40.0f})}}, config);
        TensorMap predictions{{"y", torch::tensor({100.0f})}};
        TensorMap probabilities;
        index.blend(query, predictions, probabilities);
        CHECK_THAT(predictions.at("y").item<double>(), WithinAbs(0.5 * 100.0 + 0.5 * 20.0, 1e-4));
    }
    SECTION("no labelled neighbour keeps the model's prediction") {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const RetrievalIndex index(compass_latents(), {regression_target("y")},
                                   {{"y", torch::tensor({nan, nan, 30.0f, 40.0f})}}, config);
        TensorMap predictions{{"y", torch::tensor({100.0f})}};
        TensorMap probabilities;
        index.blend(query, predictions, probabilities);
        CHECK(predictions.at("y").item<float>() == 100.0f);
    }
    SECTION("alpha 0 is the model alone, alpha 1 the neighbours alone") {
        const TensorMap labels{{"y", torch::tensor({10.0f, 20.0f, 30.0f, 40.0f})}};
        TensorMap model_only{{"y", torch::tensor({100.0f})}};
        TensorMap probabilities;
        RetrievalIndex(compass_latents(), {regression_target("y")}, labels, {2, 0.0f, 1.0f})
            .blend(query, model_only, probabilities);
        CHECK(model_only.at("y").item<float>() == 100.0f);

        TensorMap neighbours_only{{"y", torch::tensor({100.0f})}};
        RetrievalIndex(compass_latents(), {regression_target("y")}, labels, {2, 1.0f, 1.0f})
            .blend(query, neighbours_only, probabilities);
        const auto [w0, w1] = softmax2(s0, s1);
        CHECK_THAT(neighbours_only.at("y").item<double>(), WithinAbs(w0 * 10.0 + w1 * 20.0, 1e-4));
    }
}

TEST_CASE("A classification prediction blends probabilities with the neighbours' label "
          "distribution",
          "[retrieval]") {
    // Query (1, 0.1) with k = 3 reads plots 0, 1 and 3; plot 3 is unlabelled,
    // so plots 0 (class 0) and 1 (class 1) share the weight.
    const torch::Tensor query = torch::tensor({1.0f, 0.1f}).reshape({1, 2});
    const double s0 = 1.0 / std::sqrt(1.01);
    const double s1 = 0.1 / std::sqrt(1.01);
    const RetrievalIndex index(
        compass_latents(), {classification_target("hab", 3)},
        {{"hab", torch::tensor({int64_t{0}, int64_t{1}, int64_t{2}, kUnlabelledClass})}},
        {3, 0.5f, 1.0f});

    TensorMap predictions{{"hab", torch::tensor({int64_t{2}})}};
    TensorMap probabilities{{"hab", torch::tensor({0.2f, 0.3f, 0.5f}).reshape({1, 3})}};
    index.blend(query, predictions, probabilities);

    const auto [w0, w1] = softmax2(s0, s1);
    const torch::Tensor blended = probabilities.at("hab");
    CHECK_THAT(blended.index({0, 0}).item<double>(), WithinAbs(0.5 * 0.2 + 0.5 * w0, 1e-5));
    CHECK_THAT(blended.index({0, 1}).item<double>(), WithinAbs(0.5 * 0.3 + 0.5 * w1, 1e-5));
    CHECK_THAT(blended.index({0, 2}).item<double>(), WithinAbs(0.5 * 0.5, 1e-5));
    CHECK_THAT(blended.sum().item<double>(), WithinAbs(1.0, 1e-5));
    CHECK(predictions.at("hab").item<int64_t>() == blended.argmax(1).item<int64_t>());
    CHECK(predictions.at("hab").item<int64_t>() == 0);
}

TEST_CASE("RetrievalIndex refuses a reference without a target's labels or with foreign codes",
          "[retrieval]") {
    CHECK_THROWS_AS(RetrievalIndex(compass_latents(), {regression_target("y")}, {}, {}),
                    std::invalid_argument);
    CHECK_THROWS_AS(RetrievalIndex(compass_latents(), {regression_target("y")},
                                   {{"y", torch::ones({3})}}, {}),
                    std::invalid_argument);
    CHECK_THROWS_AS(RetrievalIndex(compass_latents(), {classification_target("hab", 2)},
                                   {{"hab", torch::tensor({int64_t{0}, int64_t{1}, int64_t{2},
                                                           int64_t{0}})}},
                                   {}),
                    std::invalid_argument);
    CHECK_THROWS_AS(RetrievalIndex(torch::empty({0, 2}), {}, {}, {}), std::invalid_argument);
}

namespace {

const std::vector<std::string> kHabitats = {"forest", "grass", "scrub"};

// 60 plots spread over two covariates, a regression target y and a habitat
// class; every fifth plot has no habitat label (masked).
ResolveDataset retrieval_dataset() {
    std::ostringstream header;
    header << "plot_id,cov1,cov2,y,hab\n";
    std::ostringstream species;
    species << "plot_id,sp,cover\n";
    for (int i = 0; i < 60; ++i) {
        const double c1 = (i % 10) * 0.7;
        const double c2 = (i / 10) * 1.3;
        header << "P" << i << "," << c1 << "," << c2 << "," << (3.0 * c1 - c2) << ","
               << (i % 5 == 4 ? "" : kHabitats[static_cast<size_t>(i % 3)]) << "\n";
        species << "P" << i << ",sp" << (i % 9) << ",1.0\n";
    }
    const TempFile header_csv(header.str());
    const TempFile species_csv(species.str());
    TargetSpec habitat = TargetSpec::classification("hab", 3);
    habitat.missing = TargetMissingPolicy::Mask;
    return ResolveDataset::from_csv(header_csv.path(), species_csv.path(),
                                    Roles().covariates({"cov1", "cov2"}),
                                    {TargetSpec::regression("y"), habitat}, hash_dataset_config(8));
}

Predictor trained_predictor(const ResolveDataset& dataset) {
    torch::manual_seed(3);
    ResolveModel model(dataset.schema(),
                       model_config_for_dataset(dataset, hash_mlp_config({32, 16})));
    Trainer trainer(model, quiet_train_config(3));
    trainer.prepare_data(dataset, 0.2f, 1);
    trainer.fit();
    return Predictor(trainer.model(), trainer.scalers());
}

}  // namespace

TEST_CASE("Predictor retrieval with k = 1 and alpha = 1 returns the reference plots' own labels",
          "[retrieval]") {
    const ResolveDataset dataset = retrieval_dataset();
    Predictor predictor = trained_predictor(dataset);
    const ResolvePredictions plain = predictor.predict(dataset, /*return_latent=*/true);

    predictor.set_retrieval(dataset, {1, 1.0f, 1.0f}, /*batch_size=*/16);
    REQUIRE(predictor.retrieval() != nullptr);
    CHECK(predictor.retrieval()->n_reference() == 60);
    const ResolvePredictions retrieved = predictor.predict(dataset, false, /*batch_size=*/7);

    const torch::Tensor y = dataset.targets().at("y");
    CHECK(torch::allclose(retrieved.predictions.at("y"), y, 1e-5, 1e-5));

    const torch::Tensor hab = dataset.targets().at("hab");
    const torch::Tensor labelled = hab >= 0;
    const torch::Tensor predicted = retrieved.predictions.at("hab");
    CHECK(torch::equal(predicted.index({labelled}), hab.index({labelled})));
    // An unlabelled reference plot finds only itself, so it keeps the model's
    // prediction and probabilities.
    const torch::Tensor unlabelled = labelled.logical_not();
    CHECK(torch::equal(predicted.index({unlabelled}),
                       plain.predictions.at("hab").index({unlabelled})));
    CHECK(torch::allclose(retrieved.probabilities.at("hab").sum(1), torch::ones({60}), 1e-5,
                          1e-5));

    SECTION("alpha 0 changes nothing") {
        predictor.set_retrieval(dataset, {5, 0.0f, 1.0f});
        const ResolvePredictions same = predictor.predict(dataset);
        CHECK(torch::allclose(same.predictions.at("y"), plain.predictions.at("y")));
        CHECK(torch::equal(same.predictions.at("hab"), plain.predictions.at("hab")));
    }
    SECTION("clear_retrieval restores the model's own predictions") {
        predictor.clear_retrieval();
        CHECK(predictor.retrieval() == nullptr);
        const ResolvePredictions again = predictor.predict(dataset);
        CHECK(torch::equal(again.predictions.at("y"), plain.predictions.at("y")));
    }
    SECTION("the latent is not blended") {
        CHECK(torch::allclose(predictor.predict(dataset, /*return_latent=*/true).latent,
                              plain.latent));
    }
}

TEST_CASE("Predictor retrieval refuses a reference without the model's targets", "[retrieval]") {
    const ResolveDataset dataset = retrieval_dataset();
    Predictor predictor = trained_predictor(dataset);

    std::ostringstream header;
    header << "plot_id,cov1,cov2\n";
    std::ostringstream species;
    species << "plot_id,sp,cover\n";
    for (int i = 0; i < 5; ++i) {
        header << "Q" << i << "," << i << "," << i << "\n";
        species << "Q" << i << ",sp" << i << ",1.0\n";
    }
    const TempFile header_csv(header.str());
    const TempFile species_csv(species.str());
    const ResolveDataset unlabelled = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), Roles().covariates({"cov1", "cov2"}), {},
        hash_dataset_config(8));
    CHECK_THROWS_AS(predictor.set_retrieval(unlabelled), std::invalid_argument);
    CHECK(predictor.retrieval() == nullptr);
    CHECK_THROWS_AS(predictor.set_retrieval(dataset, {0, 0.3f, 1.0f}), std::invalid_argument);
}
