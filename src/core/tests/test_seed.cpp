#include <catch2/catch_test_macros.hpp>

#include "resolve/model.hpp"
#include "resolve/process.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

using namespace resolve;
using namespace resolve::test;

namespace {

// One short fit of the SmallCorpus after manual_seed(seed): the trained
// weights and the held-out loss of each epoch.
struct SeededFit {
    std::vector<torch::Tensor> parameters;
    std::vector<float> test_loss;
};

SeededFit fit_after_seed(uint64_t seed) {
    const SmallCorpus corpus;
    const auto ds = corpus.load(SpeciesEncodingMode::Hash);
    manual_seed(seed);
    ResolveModel model(ds.schema(), small_encoder_config(SpeciesEncodingMode::Hash, {16, 8}));
    Trainer trainer(model, quiet_train_config(3));
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/0);
    SeededFit out;
    out.test_loss = trainer.fit().test_loss_history;
    for (const auto& p : trainer.model()->parameters()) out.parameters.push_back(p.detach().clone());
    return out;
}

bool same_parameters(const SeededFit& a, const SeededFit& b) {
    if (a.parameters.size() != b.parameters.size()) return false;
    for (size_t i = 0; i < a.parameters.size(); ++i) {
        if (!torch::equal(a.parameters[i], b.parameters[i])) return false;
    }
    return true;
}

}  // namespace

TEST_CASE("manual_seed fixes torch's draws", "[seed]") {
    manual_seed(11);
    const auto a = torch::rand({64});
    manual_seed(11);
    const auto b = torch::rand({64});
    manual_seed(12);
    const auto c = torch::rand({64});
    REQUIRE(torch::equal(a, b));
    REQUIRE_FALSE(torch::equal(a, c));
}

TEST_CASE("A fit after the same seed gives the same model", "[seed]") {
    const auto first = fit_after_seed(5);
    const auto again = fit_after_seed(5);
    const auto other = fit_after_seed(6);
    REQUIRE(first.test_loss == again.test_loss);
    REQUIRE(same_parameters(first, again));
    REQUIRE_FALSE(same_parameters(first, other));
}
