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
#include "resolve/sparse_rows.hpp"
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
    // 5000 queries are searched in two blocks of at most 4096 rows, each
    // against 10000 reference rows in three blocks of at most 4096.
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
    CHECK_THROWS_AS(nearest_by_cosine(torch::ones({3, 2}, torch::kInt64), compass_latents(), 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(nearest_by_cosine(torch::ones({3, 2}).to_sparse(), compass_latents(), 1),
                    std::invalid_argument);
}

TEST_CASE("nearest_by_cosine leaves each plot out of its own neighbours, never its duplicates",
          "[retrieval]") {
    torch::manual_seed(11);
    // 5000 plots: two blocks of queries and two of reference rows, so the
    // pairs left out lie in two different blocks. Plot 1 repeats plot 0, and
    // plot 4500 (second block) repeats plot 3 (first block).
    torch::Tensor plots = torch::randn({5000, 6}, torch::kFloat64);
    plots[1].copy_(plots[0]);
    plots[4500].copy_(plots[3]);
    const int64_t k = 5;

    const Neighbours found = nearest_by_cosine(plots, plots, k, /*exclude_self=*/true);
    REQUIRE(found.indices.sizes() == std::vector<int64_t>{5000, k});

    const torch::Tensor unit = torch::nn::functional::normalize(
        plots.to(torch::kFloat32), torch::nn::functional::NormalizeFuncOptions().dim(1));
    torch::Tensor all = unit.matmul(unit.t());
    all.fill_diagonal_(-std::numeric_limits<float>::infinity());
    const auto [expected_similarities, expected_indices] = all.topk(k, 1);

    const torch::Tensor own_row = torch::arange(5000, torch::kInt64).unsqueeze(1);
    CHECK_FALSE(found.indices.eq(own_row).any().item<bool>());
    // A duplicate pair ties for some queries, so the check is that every
    // returned row has the similarity reported for it and that those are the
    // k largest, rather than one order of tied rows.
    CHECK(torch::allclose(all.gather(1, found.indices), found.similarities, 1e-5, 1e-6));
    CHECK(torch::allclose(found.similarities, expected_similarities, 1e-5, 1e-6));

    CHECK(found.indices[0][0].item<int64_t>() == 1);
    CHECK(found.indices[1][0].item<int64_t>() == 0);
    CHECK(found.indices[3][0].item<int64_t>() == 4500);
    CHECK(found.indices[4500][0].item<int64_t>() == 3);
    CHECK_THAT(found.similarities[4500][0].item<double>(), WithinAbs(1.0, 1e-5));
}

TEST_CASE("nearest_by_cosine with exclude_self clamps k below the plot count", "[retrieval]") {
    const Neighbours found = nearest_by_cosine(compass_latents(), compass_latents(), 10, true);
    REQUIRE(found.indices.size(1) == 3);
    // Plot 0 at 0 degrees: its neighbours at 90 and 270 degrees tie, the lower
    // row first, then 180.
    CHECK(torch::equal(found.indices[0], torch::tensor({int64_t{1}, int64_t{3}, int64_t{2}})));
    CHECK_FALSE(found.indices.eq(torch::arange(4, torch::kInt64).unsqueeze(1)).any().item<bool>());

    const torch::Tensor single = torch::tensor({1.0f, 0.0f}).reshape({1, 2});
    CHECK(nearest_by_cosine(single, single, 3, true).indices.sizes() ==
          std::vector<int64_t>{1, 0});
    CHECK_THROWS_AS(nearest_by_cosine(single, compass_latents(), 1, true), std::invalid_argument);
}

TEST_CASE("A tie between equal scores goes to the lower reference row, across blocks",
          "[retrieval]") {
    // 10000 reference rows in three blocks: every third row holds the query's
    // species, so 3334 rows tie at similarity 1 across all blocks, the rest at 0.
    torch::Tensor reference = torch::zeros({10000, 4});
    for (int64_t r = 0; r < 10000; ++r) reference[r][r % 3 == 0 ? 0 : 1] = 1.0f;
    const torch::Tensor query = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({1, 4});
    const Neighbours found = nearest_by_cosine(query, reference, 5);
    CHECK(torch::equal(found.indices[0],
                       torch::tensor({int64_t{0}, int64_t{3}, int64_t{6}, int64_t{9}, int64_t{12}})));
    // The same over the sparse layout, and below the ties: rows at 0 in row order.
    const Neighbours sparse = nearest_by_cosine(query, to_sparse_rows(reference), 5);
    CHECK(torch::equal(sparse.indices, found.indices));
    const torch::Tensor other = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f}).reshape({1, 4});
    CHECK(torch::equal(nearest_by_cosine(other, reference, 3).indices[0],
                       torch::tensor({int64_t{0}, int64_t{1}, int64_t{2}})));

    // Plots at one position: the nearest are the lowest rows among them.
    torch::Tensor places = torch::zeros({9000, 2}, torch::kFloat64);
    places.select(1, 0).fill_(48.0);
    places.select(1, 1).fill_(16.0);
    places[2][0] = 50.0;
    const torch::Tensor here = torch::tensor({48.0, 16.0}, torch::kFloat64).reshape({1, 2});
    CHECK(torch::equal(nearest_by_great_circle(here, places, 4).indices[0],
                       torch::tensor({int64_t{0}, int64_t{1}, int64_t{3}, int64_t{4}})));
}

TEST_CASE("nearest_by_cosine runs on the reference's device with queries in host memory",
          "[retrieval]") {
    if (!torch::cuda::is_available()) SKIP("needs a CUDA device");
    torch::manual_seed(3);
    const torch::Tensor queries = torch::randn({300, 16});
    const torch::Tensor reference = torch::randn({2000, 16});
    const Neighbours on_host = nearest_by_cosine(queries, reference, 8);
    // A trained model leaves TF32 on for cuBLAS; the search stays float32.
    const bool tf32 = at::globalContext().allowTF32CuBLAS();
    at::globalContext().setAllowTF32CuBLAS(true);
    const Neighbours on_device = nearest_by_cosine(queries, reference.to(torch::kCUDA), 8);
    CHECK(at::globalContext().allowTF32CuBLAS());
    at::globalContext().setAllowTF32CuBLAS(tf32);
    REQUIRE(on_device.indices.device().is_cuda());
    CHECK(torch::allclose(on_device.similarities.cpu(), on_host.similarities, 1e-5, 1e-6));
    CHECK(torch::equal(on_device.indices.cpu(), on_host.indices));
    const Neighbours left_out =
        nearest_by_cosine(reference.to(torch::kCUDA), reference.to(torch::kCUDA), 8, true);
    CHECK_FALSE(left_out.indices.cpu()
                    .eq(torch::arange(2000, torch::kInt64).unsqueeze(1))
                    .any()
                    .item<bool>());
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

TEST_CASE("nearest_by_cosine over sparse rows is the dense search", "[retrieval]") {
    torch::manual_seed(13);
    const torch::Tensor dense = (torch::rand({600, 40}) < 0.15).to(torch::kFloat32);
    const torch::Tensor queries = dense.slice(0, 0, 200);
    const Neighbours expected = nearest_by_cosine(queries, dense, 6);
    const Neighbours found = nearest_by_cosine(to_sparse_rows(queries), to_sparse_rows(dense), 6);
    CHECK(torch::allclose(found.similarities, expected.similarities, 1e-5, 1e-6));
    CHECK(torch::allclose(dense.index({found.indices}).mul(queries.unsqueeze(1)).sum(2),
                          dense.index({expected.indices}).mul(queries.unsqueeze(1)).sum(2)));
}

TEST_CASE("nearest_by_great_circle is exact and skips positions without coordinates",
          "[retrieval]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const torch::Tensor reference = torch::tensor({47.0, 14.0, 47.0, 14.0, 47.0, 14.0, 50.0, 14.0, nan, nan},
                                                  torch::kFloat64)
                                        .reshape({5, 2});
    const PlaceNeighbours left_out = nearest_by_great_circle(reference, reference, 2, true);
    CHECK_FALSE(left_out.indices.eq(torch::arange(5).unsqueeze(1)).any().item<bool>());
    CHECK(left_out.indices[3][0].item<int64_t>() <= 2);
    CHECK((left_out.indices[4] == -1).all().item<bool>());
    CHECK((left_out.indices.slice(0, 0, 4) != 4).all().item<bool>());
    CHECK_THAT(left_out.km[0][0].item<double>(), WithinAbs(0.0, 1e-9));
    CHECK_THROWS_AS(nearest_by_great_circle(torch::ones({2, 3}, torch::kFloat64), reference, 1),
                    std::invalid_argument);
}
