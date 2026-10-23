// Unknown-species tracking, and the NCA loss preset.
//
// Part 1 -- unknown_fraction / unknown_count
//
//   DatasetConfig::track_unknown_fraction defaults to true, so every model
//   trained with the defaults is handed an unknown-mass column, which a dataset
//   built from a training checkpoint's vocabulary fills with the share of each
//   plot's species absent from that vocabulary. The tests below build a scoring
//   dataset in which a KNOWN share of each plot's abundance comes from species
//   the training file never contained, and assert the reported values, for
//   every encoding mode.
//
//   The definition (compute_unknown_species_stats, species_encoding.hpp):
//     fraction[p] = unknown abundance in p / total abundance in p
//     count[p]    = number of unknown species in p
//   over the plot's full assemblage, before top-k selection or a pool cap.
//
// Part 2 -- LossConfigMode::NCA
//
//   loss_config = NCA enables the NCA neighbourhood term (nca_objective) rather
//   than falling through to Combined, and these tests pin that it is selected,
//   that it changes the computed loss, that it carries gradient, that it
//   trains, and that it survives a checkpoint round-trip.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/dataset.hpp"
#include "resolve/loss.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace resolve;
using namespace resolve::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// ---------------------------------------------------------------------------
// Synthetic corpus
// ---------------------------------------------------------------------------
//
// Training file: 32 plots over three species (sp_a, sp_b, sp_c), so the fitted
// vocabulary is exactly {sp_a, sp_b, sp_c}.
//
// Scoring file: 12 plots. Every plot carries the two known species sp_a (cover
// 3.0) and sp_b (cover 1.0), plus `i % 3` novel species (nov_0, nov_1) at cover
// 1.0 each. So for plot i, with k = i % 3:
//
//     expected count    = k
//     expected fraction = k / (4 + k)     -> 0, 0.2, 1/3
//
// The abundance-weighted fraction and the species count therefore disagree for
// every k > 0, which a test that only checked "non-zero" would not catch.

struct SpeciesEntry {
    std::string name;
    double cover;
};

std::string genus_of(const std::string& sp) { return "gen_" + sp.substr(sp.size() - 1); }
std::string family_of(const std::string& sp) {
    return sp.rfind("nov", 0) == 0 ? "fam_novel" : "fam_known";
}

int novel_count(int plot_index) { return plot_index % 3; }

std::vector<SpeciesEntry> scoring_species(int plot_index) {
    std::vector<SpeciesEntry> entries{{"sp_a", 3.0}, {"sp_b", 1.0}};
    for (int k = 0; k < novel_count(plot_index); ++k) {
        entries.push_back({"nov_" + std::to_string(k), 1.0});
    }
    return entries;
}

double expected_fraction(int plot_index) {
    const double k = novel_count(plot_index);
    return k / (4.0 + k);
}

std::string header_csv(const std::string& prefix, int n_plots) {
    std::ostringstream out;
    out << "plot_id,lat,lon,cov1,y,cls\n";
    for (int i = 0; i < n_plots; ++i) {
        out << prefix << i << "," << (40.0 + 0.1 * i) << "," << (-5.0 + 0.1 * i)
            << "," << (0.5 * i) << "," << (1.0 + 0.25 * i) << "," << (i % 3) << "\n";
    }
    return out.str();
}

void write_species_rows(std::ostringstream& out, const std::string& plot_id,
                        const std::vector<SpeciesEntry>& entries) {
    for (const auto& e : entries) {
        out << plot_id << "," << e.name << "," << e.cover << ","
            << genus_of(e.name) << "," << family_of(e.name) << "\n";
    }
}

std::string training_species_csv(int n_plots) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < n_plots; ++i) {
        // Cycle which of the three species carries the heavy cover so the
        // frequency-ranked vocabulary has a stable, non-degenerate order.
        std::vector<SpeciesEntry> entries{
            {"sp_a", 1.0 + (i % 3 == 0 ? 2.0 : 0.0)},
            {"sp_b", 1.0 + (i % 3 == 1 ? 2.0 : 0.0)},
            {"sp_c", 1.0 + (i % 3 == 2 ? 2.0 : 0.0)},
        };
        write_species_rows(out, "T" + std::to_string(i), entries);
    }
    return out.str();
}

std::string scoring_species_csv(int n_plots) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < n_plots; ++i) {
        write_species_rows(out, "S" + std::to_string(i), scoring_species(i));
    }
    return out.str();
}

RoleMapping test_roles() { return Roles().taxonomy().coordinates().covariates({"cov1"}); }

std::vector<TargetSpec> test_targets() {
    return {TargetSpec::regression("y"), TargetSpec::classification("cls", 3)};
}

DatasetConfig tracking_config(SpeciesEncodingMode mode) {
    return vocab_dataset_config(mode, /*track_unknown=*/true);
}

constexpr int kTrainPlots = 32;
constexpr int kScorePlots = 12;

// Training + scoring datasets sharing one vocabulary namespace: the scoring set
// is built from the training set's SCHEMA (the checkpoint-only path), which is
// the construction under test.
struct Pair {
    ResolveDataset train;
    ResolveDataset score;
};

Pair build_pair(const DatasetConfig& cfg) {
    TempFile train_header(header_csv("T", kTrainPlots));
    TempFile train_species(training_species_csv(kTrainPlots));
    TempFile score_header(header_csv("S", kScorePlots));
    TempFile score_species(scoring_species_csv(kScorePlots));

    auto roles = test_roles();
    auto targets = test_targets();

    Pair p;
    p.train = ResolveDataset::from_csv(train_header.path(), train_species.path(),
                                       roles, targets, cfg);
    p.score = ResolveDataset::from_csv_with_schema(
        score_header.path(), score_species.path(), roles, targets,
        p.train.schema(), cfg);
    return p;
}

std::vector<float> to_vector(const torch::Tensor& t) {
    REQUIRE(t.defined());
    auto cpu = t.to(torch::kCPU).contiguous().to(torch::kFloat32);
    auto acc = cpu.accessor<float, 1>();
    std::vector<float> out;
    out.reserve(static_cast<size_t>(cpu.size(0)));
    for (int64_t i = 0; i < cpu.size(0); ++i) out.push_back(acc[i]);
    return out;
}

}  // namespace

// ===========================================================================
// Part 1: unknown-species tracking
// ===========================================================================

TEST_CASE("Novel species are reported per plot, in every encoding mode",
          "[dataset][unknown]") {
    for (auto mode : all_species_encodings()) {
        INFO("encoding mode: " << species_encoding_to_string(mode));
        auto cfg = tracking_config(mode);
        auto pair = build_pair(cfg);

        const auto fraction = to_vector(pair.score.unknown_fraction());
        const auto count = to_vector(pair.score.unknown_count());

        REQUIRE(fraction.size() == static_cast<size_t>(kScorePlots));
        REQUIRE(count.size() == static_cast<size_t>(kScorePlots));

        for (int i = 0; i < kScorePlots; ++i) {
            INFO("plot S" << i << " expects " << novel_count(i) << " novel species");
            REQUIRE_THAT(fraction[static_cast<size_t>(i)],
                         WithinAbs(static_cast<float>(expected_fraction(i)), 1e-6f));
            REQUIRE_THAT(count[static_cast<size_t>(i)],
                         WithinAbs(static_cast<float>(novel_count(i)), 1e-6f));
        }

        // At least one plot is genuinely novel, so a still-zeroed column fails
        // the loop above rather than passing vacuously.
        REQUIRE(*std::max_element(fraction.begin(), fraction.end()) > 0.0f);
    }
}

TEST_CASE("A vocabulary fitted on the same records reports no novelty",
          "[dataset][unknown]") {
    // The training dataset fits its vocabulary from its own species, so every
    // name is known by construction and both columns are legitimately zero.
    for (auto mode : all_species_encodings()) {
        INFO("encoding mode: " << species_encoding_to_string(mode));
        auto cfg = tracking_config(mode);
        auto pair = build_pair(cfg);

        for (float v : to_vector(pair.train.unknown_fraction())) {
            REQUIRE_THAT(v, WithinAbs(0.0f, 1e-6f));
        }
        for (float v : to_vector(pair.train.unknown_count())) {
            REQUIRE_THAT(v, WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Tracking flags gate the feature columns", "[dataset][unknown]") {
    auto cfg = tracking_config(SpeciesEncodingMode::Hash);

    SECTION("fraction only") {
        cfg.track_unknown_count = false;
        auto pair = build_pair(cfg);
        REQUIRE(pair.score.unknown_fraction().defined());
        REQUIRE_FALSE(pair.score.unknown_count().defined());
        REQUIRE(pair.score.schema().dataset_config.track_unknown_fraction);
        REQUIRE_FALSE(pair.score.schema().dataset_config.track_unknown_count);
    }

    SECTION("neither") {
        cfg.track_unknown_fraction = false;
        cfg.track_unknown_count = false;
        auto pair = build_pair(cfg);
        REQUIRE_FALSE(pair.score.unknown_fraction().defined());
        REQUIRE_FALSE(pair.score.unknown_count().defined());
    }

    SECTION("count only") {
        cfg.track_unknown_fraction = false;
        auto pair = build_pair(cfg);
        REQUIRE_FALSE(pair.score.unknown_fraction().defined());
        const auto count = to_vector(pair.score.unknown_count());
        for (int i = 0; i < kScorePlots; ++i) {
            REQUIRE_THAT(count[static_cast<size_t>(i)],
                         WithinAbs(static_cast<float>(novel_count(i)), 1e-6f));
        }
    }
}

TEST_CASE("Novelty is measured before the rank-pool species cap truncates",
          "[dataset][unknown]") {
    // The cap slices each plot's list down to `cap` entries. The novelty of the
    // assemblage is a property of the plot, not of the slice the encoder kept,
    // so the reported values must match the uncapped run exactly.
    auto uncapped_cfg = tracking_config(SpeciesEncodingMode::RankPool);
    auto capped_cfg = uncapped_cfg;
    capped_cfg.pool_species_cap = 2;  // scoring plots carry up to 4 species

    auto uncapped = build_pair(uncapped_cfg);
    auto capped = build_pair(capped_cfg);

    REQUIRE(capped.score.species_ids().size(1) == 2);
    REQUIRE(uncapped.score.species_ids().size(1) > 2);

    const auto a = to_vector(uncapped.score.unknown_fraction());
    const auto b = to_vector(capped.score.unknown_fraction());
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        REQUIRE_THAT(b[i], WithinAbs(a[i], 1e-6f));
    }
}

TEST_CASE("The dataset columns match the standalone helper", "[dataset][unknown]") {
    // One definition of "unknown species" in the engine: the dataset, the
    // rank-pool encoder and the embedding encoder must not be able to drift.
    auto cfg = tracking_config(SpeciesEncodingMode::RankPool);
    auto pair = build_pair(cfg);

    // Rebuild the records and the training vocabulary the scoring dataset used.
    std::vector<SpeciesRecord> records;
    std::vector<std::string> plot_ids;
    for (int i = 0; i < kScorePlots; ++i) {
        const std::string pid = "S" + std::to_string(i);
        plot_ids.push_back(pid);
        for (const auto& e : scoring_species(i)) {
            SpeciesRecord r;
            r.plot_id = pid;
            r.species_id = e.name;
            r.abundance = static_cast<float>(e.cover);
            r.genus = genus_of(e.name);
            r.family = family_of(e.name);
            records.push_back(std::move(r));
        }
    }

    const auto stats = compute_unknown_species_stats(
        PlotAssemblages::build(records, plot_ids),
        SpeciesVocab::from_ordered(pair.train.species_vocab()));

    const auto dataset_fraction = to_vector(pair.score.unknown_fraction());
    const auto helper_fraction = to_vector(stats.fraction);
    const auto dataset_count = to_vector(pair.score.unknown_count());
    const auto helper_count = to_vector(stats.count);

    for (size_t i = 0; i < dataset_fraction.size(); ++i) {
        REQUIRE_THAT(helper_fraction[i], WithinAbs(dataset_fraction[i], 1e-6f));
        REQUIRE_THAT(helper_count[i], WithinAbs(dataset_count[i], 1e-6f));
    }
}

TEST_CASE("An encoder's adopted vocabulary reports the same novelty",
          "[species_encoders][unknown]") {
    std::vector<SpeciesRecord> train_records;
    for (int i = 0; i < 6; ++i) {
        for (const char* sp : {"sp_a", "sp_b", "sp_c"}) {
            SpeciesRecord r;
            r.plot_id = "T" + std::to_string(i);
            r.species_id = sp;
            r.abundance = 1.0f;
            r.genus = genus_of(sp);
            r.family = family_of(sp);
            train_records.push_back(std::move(r));
        }
    }

    std::vector<SpeciesRecord> score_records;
    std::vector<std::string> score_plots;
    for (int i = 0; i < kScorePlots; ++i) {
        const std::string pid = "S" + std::to_string(i);
        score_plots.push_back(pid);
        for (const auto& e : scoring_species(i)) {
            SpeciesRecord r;
            r.plot_id = pid;
            r.species_id = e.name;
            r.abundance = static_cast<float>(e.cover);
            r.genus = genus_of(e.name);
            r.family = family_of(e.name);
            score_records.push_back(std::move(r));
        }
    }

    const auto scored = PlotAssemblages::build(score_records, score_plots);
    const auto require_expected_novelty = [&](const SpeciesVocab& vocab) {
        const auto stats = compute_unknown_species_stats(scored, vocab);
        const auto fraction = to_vector(stats.fraction);
        const auto count = to_vector(stats.count);
        for (int i = 0; i < kScorePlots; ++i) {
            REQUIRE_THAT(fraction[static_cast<size_t>(i)],
                         WithinAbs(static_cast<float>(expected_fraction(i)), 1e-6f));
            REQUIRE_THAT(count[static_cast<size_t>(i)],
                         WithinAbs(static_cast<float>(novel_count(i)), 1e-6f));
        }
    };

    SECTION("rank pool") {
        RankPoolEncoder encoder(PoolWeighting::Log1p, /*min_frequency=*/1);
        encoder.fit(train_records);
        RankPoolEncoder scorer(PoolWeighting::Log1p, /*min_frequency=*/1);
        scorer.set_vocab(encoder.vocab());
        require_expected_novelty(scorer.species_vocab());

        // A novel species occupies a slot under the unknown code 0.
        auto encoded = scorer.transform(scored);
        auto ids = encoded.species_ids.accessor<int64_t, 2>();
        auto mask = encoded.mask.accessor<bool, 2>();
        for (int i = 0; i < kScorePlots; ++i) {
            int unknown_slots = 0;
            for (int64_t j = 0; j < encoded.species_ids.size(1); ++j) {
                if (mask[i][j] && ids[i][j] == 0) ++unknown_slots;
            }
            REQUIRE(unknown_slots == novel_count(i));
        }
    }

    SECTION("embedding") {
        EmbeddingEncoder encoder(/*top_k_species=*/3, /*top_k_taxonomy=*/2);
        encoder.fit(train_records);
        EmbeddingEncoder scorer(/*top_k_species=*/3, /*top_k_taxonomy=*/2);
        scorer.set_vocab(encoder.vocab());
        require_expected_novelty(scorer.species_vocab());
    }
}

TEST_CASE("Both unknown columns stay consistent with the model's input width",
          "[dataset][unknown][model]") {
    // n_continuous counts the tracking flags (model.cpp / adapter.cpp) while the
    // loader allocates the tensors from the same flags. Fitting on the training
    // set and scoring the (novel-species) set through a saved checkpoint runs
    // both continuous-block concatenation sites -- Trainer::prepare_data and
    // Predictor::predict -- so a width disagreement surfaces here as a shape
    // error rather than at a user's fit.
    auto cfg = tracking_config(SpeciesEncodingMode::Hash);
    auto pair = build_pair(cfg);

    const TempPath path_file(".pt");
    const std::string path = path_file.path();

    torch::manual_seed(7);
    ResolveModel model(pair.train.schema(), vocab_model_config(SpeciesEncodingMode::Hash));
    Trainer trainer(model, quiet_train_config(/*epochs=*/2));
    trainer.prepare_data(pair.train, /*test_size=*/0.25f, /*seed=*/0);
    auto result = trainer.fit();
    REQUIRE(result.train_loss_history.size() >= 1);
    trainer.save(path);

    auto predictor = Predictor::load(path);
    auto out = predictor.predict(pair.score);
    REQUIRE(out.predictions.count("y") == 1);
    REQUIRE(out.predictions.at("y").size(0) == kScorePlots);
}

// ===========================================================================
// Part 2: LossConfigMode::NCA
// ===========================================================================

namespace {

// Paper reference implementation, written as explicit loops over the sums in
// Goldberger et al. (2004) eqs. (1), (2) and (6), independent of the masked
// log-softmax formulation nca_objective uses.
double reference_nca(const torch::Tensor& embeddings, const torch::Tensor& labels,
                     double temperature) {
    auto unit = torch::nn::functional::normalize(
                    embeddings, torch::nn::functional::NormalizeFuncOptions().dim(1))
                    .to(torch::kCPU)
                    .to(torch::kFloat64)
                    .contiguous();
    auto lab = labels.to(torch::kCPU).contiguous();
    const int64_t n = unit.size(0);
    const int64_t d = unit.size(1);
    auto u = unit.accessor<double, 2>();
    auto c = lab.accessor<int64_t, 1>();

    double total = 0.0;
    int64_t contributing = 0;
    for (int64_t i = 0; i < n; ++i) {
        double denominator = 0.0;
        double numerator = 0.0;
        int64_t same_class_neighbours = 0;
        for (int64_t k = 0; k < n; ++k) {
            if (k == i) continue;  // p_ii = 0
            double dot = 0.0;
            for (int64_t f = 0; f < d; ++f) dot += u[i][f] * u[k][f];
            const double w = std::exp(dot / temperature);
            denominator += w;
            if (c[k] == c[i]) {
                numerator += w;
                ++same_class_neighbours;
            }
        }
        if (same_class_neighbours == 0) continue;  // contributes 0
        total += -std::log(numerator / denominator);
        ++contributing;
    }
    return contributing > 0 ? total / static_cast<double>(contributing) : 0.0;
}

std::vector<TargetConfig> classification_target_configs() {
    TargetConfig cfg;
    cfg.name = "cls";
    cfg.task = TaskType::Classification;
    cfg.num_classes = 3;
    cfg.weight = 1.0f;
    return {cfg};
}

}  // namespace

TEST_CASE("nca_objective matches the paper's eq. (1)/(2)/(6) definition",
          "[loss][nca]") {
    torch::manual_seed(11);
    auto embeddings = torch::randn({16, 5});
    auto labels = torch::tensor({0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
                                torch::kInt64);

    for (double temperature : {0.1, 0.5, 1.0}) {
        INFO("temperature " << temperature);
        // n_neighbors = 0 keeps the untruncated in-batch neighbour set, which is
        // the form the reference implements.
        const double actual = nca_objective(embeddings, labels,
                                            static_cast<float>(temperature),
                                            /*n_neighbors=*/0)
                                  .item<double>();
        REQUIRE_THAT(actual, WithinRel(reference_nca(embeddings, labels, temperature), 1e-4));
    }
}

TEST_CASE("nca_objective excludes samples with no same-class neighbour",
          "[loss][nca]") {
    torch::manual_seed(12);
    auto embeddings = torch::randn({4, 3});
    // Class 2 appears once, so that sample has no same-class neighbour and must
    // not enter the average (its own term is defined as 0).
    auto labels = torch::tensor({0, 0, 1, 2}, torch::kInt64);

    const double actual = nca_objective(embeddings, labels, 0.1f, 0).item<double>();
    REQUIRE_THAT(actual, WithinRel(reference_nca(embeddings, labels, 0.1), 1e-4));
    REQUIRE(std::isfinite(actual));
}

namespace {

// Whether `loss` adds the NCA term to its classification loss: on a fixed
// batch, the loss differs from plain cross-entropy.
bool adds_nca_term(const PhasedLoss& loss) {
    torch::manual_seed(13);
    const auto logits = torch::randn({12, 3});
    const auto labels =
        torch::tensor({0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2}, torch::kInt64);
    const double ce =
        torch::nn::functional::cross_entropy(logits, labels).item<double>();
    return std::abs(loss.classification_loss(logits, labels).item<double>() - ce) > 1e-6;
}

// The same through the multi-task combiner the trainer holds.
bool adds_nca_term(const MultiTaskLoss& loss) {
    torch::manual_seed(13);
    const auto configs = classification_target_configs();
    std::unordered_map<std::string, torch::Tensor> predictions, targets;
    for (const auto& cfg : configs) {
        predictions[cfg.name] = torch::randn({12, cfg.num_classes});
        targets[cfg.name] = torch::randint(0, cfg.num_classes, {12}, torch::kInt64);
    }
    double ce = 0.0;
    for (const auto& cfg : configs) {
        ce += cfg.weight * torch::nn::functional::cross_entropy(
                               predictions[cfg.name], targets[cfg.name]).item<double>();
    }
    return std::abs(loss.compute(predictions, targets, 0).first.item<double>() - ce) > 1e-6;
}

}  // namespace

TEST_CASE("LossConfigMode::NCA selects the NCA term", "[loss][nca]") {
    REQUIRE(adds_nca_term(PhasedLoss::from_config(LossConfigMode::NCA)));
    REQUIRE_FALSE(adds_nca_term(PhasedLoss::from_config(LossConfigMode::Combined)));
    REQUIRE_FALSE(adds_nca_term(PhasedLoss::from_config(LossConfigMode::MAE)));
    REQUIRE_FALSE(adds_nca_term(PhasedLoss::from_config(LossConfigMode::SMAPE)));

    // And it reaches the multi-task combiner the trainer actually holds.
    REQUIRE(adds_nca_term(MultiTaskLoss(classification_target_configs(), {100, 300},
                                        LossConfigMode::NCA)));
    REQUIRE_FALSE(adds_nca_term(MultiTaskLoss(classification_target_configs(), {100, 300},
                                              LossConfigMode::Combined)));
}

TEST_CASE("The NCA knob changes the computed classification loss", "[loss][nca]") {
    torch::manual_seed(14);
    auto logits = torch::randn({24, 3});
    auto labels = torch::randint(0, 3, {24}, torch::kInt64);

    auto combined = PhasedLoss::from_config(LossConfigMode::Combined);
    auto nca = PhasedLoss::from_config(LossConfigMode::NCA);

    const double ce = combined.classification_loss(logits, labels).item<double>();
    const double with_nca = nca.classification_loss(logits, labels).item<double>();

    REQUIRE(std::abs(with_nca - ce) > 1e-4);

    // Exactly cross-entropy plus the weighted neighbourhood term.
    const double term = nca_objective(logits, labels, kNCATemperature, kNCANeighbors)
                            .item<double>();
    REQUIRE_THAT(with_nca, WithinRel(ce + kNCAWeight * term, 1e-5));
}

TEST_CASE("The NCA preset leaves the regression schedule alone", "[loss][nca]") {
    torch::manual_seed(15);
    auto pred = torch::randn({20});
    auto target = torch::randn({20}) + 3.0f;

    auto combined = PhasedLoss::from_config(LossConfigMode::Combined, {2, 4});
    auto nca = PhasedLoss::from_config(LossConfigMode::NCA, {2, 4});

    for (int epoch : {0, 3, 9}) {
        INFO("epoch " << epoch);
        REQUIRE(nca.get_phase(epoch) == combined.get_phase(epoch));
        REQUIRE_THAT(nca.regression_loss(pred, target, epoch).item<double>(),
                     WithinRel(combined.regression_loss(pred, target, epoch).item<double>(),
                               1e-9));
    }
}

TEST_CASE("The NCA term carries gradient", "[loss][nca]") {
    torch::manual_seed(16);
    auto labels = torch::randint(0, 3, {24}, torch::kInt64);
    auto base = torch::randn({24, 3});

    auto with_nca_logits = base.clone().set_requires_grad(true);
    PhasedLoss::from_config(LossConfigMode::NCA)
        .classification_loss(with_nca_logits, labels)
        .backward();
    REQUIRE(with_nca_logits.grad().defined());
    REQUIRE(with_nca_logits.grad().abs().sum().item<double>() > 0.0);

    auto ce_logits = base.clone().set_requires_grad(true);
    PhasedLoss::from_config(LossConfigMode::Combined)
        .classification_loss(ce_logits, labels)
        .backward();

    // A different objective must produce a different descent direction, not just
    // a different scalar.
    const double delta = (with_nca_logits.grad() - ce_logits.grad())
                             .abs()
                             .max()
                             .item<double>();
    REQUIRE(delta > 1e-6);
}

TEST_CASE("A model trains under the NCA loss config", "[loss][nca][trainer]") {
    auto cfg = tracking_config(SpeciesEncodingMode::Hash);
    auto pair = build_pair(cfg);

    torch::manual_seed(17);
    ResolveModel model(pair.train.schema(),
                       vocab_model_config(SpeciesEncodingMode::Hash));
    auto train_config = quiet_train_config(/*epochs=*/25);
    train_config.loss_config = LossConfigMode::NCA;
    train_config.lr = 1e-2f;

    Trainer trainer(model, train_config);
    trainer.prepare_data(pair.train, /*test_size=*/0.25f, /*seed=*/0);
    auto result = trainer.fit();

    REQUIRE(result.train_loss_history.size() >= 5);
    const float first = result.train_loss_history.front();
    const float last = result.train_loss_history.back();
    REQUIRE(std::isfinite(first));
    REQUIRE(std::isfinite(last));
    REQUIRE(last < first);
}

TEST_CASE("loss_config = NCA survives the checkpoint round-trip",
          "[loss][nca][checkpoint]") {
    auto cfg = tracking_config(SpeciesEncodingMode::Hash);
    auto pair = build_pair(cfg);

    const TempPath path_file(".pt");
    const std::string path = path_file.path();

    torch::manual_seed(18);
    ResolveModel model(pair.train.schema(),
                       vocab_model_config(SpeciesEncodingMode::Hash));
    auto train_config = quiet_train_config();
    train_config.loss_config = LossConfigMode::NCA;

    Trainer trainer(model, train_config);
    trainer.prepare_data(pair.train, /*test_size=*/0.25f, /*seed=*/0);
    trainer.save(path);

    auto loaded = Trainer::load_train_config(path);
    REQUIRE(loaded.loss_config == LossConfigMode::NCA);
    REQUIRE(adds_nca_term(PhasedLoss::from_config(loaded.loss_config)));
}

TEST_CASE("The NCA hyperparameters are configurable and round-trip",
          "[loss][nca][checkpoint][config]") {
    // temperature / n_neighbors / weight are TrainConfig fields, so a run's own
    // recipe carries them to the loss, into the checkpoint and back out again.
    TrainConfig tuned;
    tuned.loss_config = LossConfigMode::NCA;
    tuned.nca_temperature = 0.45f;   // default 0.1
    tuned.nca_neighbors = 7;         // default 32
    tuned.nca_weight = 0.6f;         // default 0.1

    SECTION("the values reach the loss and change what it computes") {
        torch::manual_seed(19);
        auto logits = torch::randn({24, 3});
        auto labels = torch::randint(0, 3, {24}, torch::kInt64);

        const auto defaults = PhasedLoss::from_config(
            LossConfigMode::NCA, {100, 300}, 0.25f, nca_term_of(TrainConfig{}));
        const auto custom = PhasedLoss::from_config(
            LossConfigMode::NCA, {100, 300}, tuned.band_threshold, nca_term_of(tuned));

        const double with_defaults = defaults.classification_loss(logits, labels).item<double>();
        const double with_custom = custom.classification_loss(logits, labels).item<double>();
        REQUIRE(std::abs(with_custom - with_defaults) > 1e-4);

        // Exactly cross-entropy plus the tuned term, i.e. all three knobs act.
        const double ce = PhasedLoss::from_config(LossConfigMode::Combined)
                              .classification_loss(logits, labels)
                              .item<double>();
        const double term = nca_objective(logits, labels, tuned.nca_temperature,
                                          tuned.nca_neighbors)
                                .item<double>();
        REQUIRE_THAT(with_custom, WithinRel(ce + tuned.nca_weight * term, 1e-5));

        // Each knob on its own moves the number, so none of the three is inert.
        for (int knob = 0; knob < 3; ++knob) {
            INFO("knob " << knob);
            TrainConfig one_off;
            if (knob == 0) one_off.nca_temperature = tuned.nca_temperature;
            if (knob == 1) one_off.nca_neighbors = tuned.nca_neighbors;
            if (knob == 2) one_off.nca_weight = tuned.nca_weight;
            const double value = PhasedLoss::from_config(LossConfigMode::NCA, {100, 300},
                                                         0.25f, nca_term_of(one_off))
                                     .classification_loss(logits, labels)
                                     .item<double>();
            REQUIRE(std::abs(value - with_defaults) > 1e-5);
        }

        // And through the combiner the Trainer actually holds.
        const std::vector<TargetConfig> cls = classification_target_configs();
        std::unordered_map<std::string, torch::Tensor> preds{{"cls", logits}};
        std::unordered_map<std::string, torch::Tensor> actuals{{"cls", labels}};
        const double multi_custom =
            MultiTaskLoss(cls, {100, 300}, LossConfigMode::NCA, tuned.band_threshold,
                          nca_term_of(tuned))
                .compute(preds, actuals, /*epoch=*/0)
                .first.item<double>();
        REQUIRE_THAT(multi_custom, WithinRel(with_custom, 1e-5));
    }

    SECTION("they survive the checkpoint") {
        auto cfg = tracking_config(SpeciesEncodingMode::Hash);
        auto pair = build_pair(cfg);

        const TempPath path_file(".pt");
        const std::string path = path_file.path();

        torch::manual_seed(20);
        ResolveModel model(pair.train.schema(),
                           vocab_model_config(SpeciesEncodingMode::Hash));
        auto train_config = quiet_train_config();
        train_config.loss_config = tuned.loss_config;
        train_config.nca_temperature = tuned.nca_temperature;
        train_config.nca_neighbors = tuned.nca_neighbors;
        train_config.nca_weight = tuned.nca_weight;

        Trainer trainer(model, train_config);
        trainer.prepare_data(pair.train, /*test_size=*/0.25f, /*seed=*/0);
        trainer.save(path);

        const TrainConfig loaded = Trainer::load_train_config(path);
        REQUIRE_THAT(loaded.nca_temperature, WithinRel(tuned.nca_temperature, 1e-6f));
        REQUIRE(loaded.nca_neighbors == tuned.nca_neighbors);
        REQUIRE_THAT(loaded.nca_weight, WithinRel(tuned.nca_weight, 1e-6f));
    }

    SECTION("a checkpoint written before the keys existed keeps the defaults") {
        // try_read semantics: an archive with none of the train_nca_* keys reads
        // back as the struct defaults rather than as zeros.
        const TempPath path_file(".pt");
        const std::string path = path_file.path();
        {
            torch::serialize::OutputArchive archive;
            archive.write("train_batch_size", torch::tensor(2048));
            archive.save_to(path);
        }

        const TrainConfig loaded = Trainer::load_train_config(path);
        const TrainConfig defaults;
        REQUIRE(loaded.batch_size == 2048);
        REQUIRE_THAT(loaded.nca_temperature, WithinRel(defaults.nca_temperature, 1e-6f));
        REQUIRE(loaded.nca_neighbors == defaults.nca_neighbors);
        REQUIRE_THAT(loaded.nca_weight, WithinRel(defaults.nca_weight, 1e-6f));
    }
}
