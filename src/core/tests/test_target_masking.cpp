// Per-target missing-label masking (TargetMissingPolicy::Mask).
//
// Under Drop (the default) a plot missing any target's label leaves the fit.
// Under Mask the plot stays with a sentinel in that target's tensor (class code
// -1, regression NaN) and trains and is scored on its other targets only. The
// cases below pin the two properties the design rests on -- Mask reduces to
// Drop exactly where the two must agree, and an unlabelled row never reaches its
// target's loss -- and then the data, checkpoint and recovery behaviour.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/dataset.hpp"
#include "resolve/loss.hpp"
#include "resolve/model.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/target_labels.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

// One synthetic plot: two covariates, a regression label and a class label,
// either of which may be missing ("" in the CSV).
struct Plot {
    double c1 = 0.0;
    double c2 = 0.0;
    std::optional<double> y;
    std::optional<std::string> cls;
};

// y = 2 c1 - 1.5 c2 + 3; cls = "pos" when c1 + c2 > 0, else "neg".
std::vector<Plot> make_plots(int64_t n, double p_y, double p_cls, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<Plot> plots(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        auto& p = plots[static_cast<size_t>(i)];
        p.c1 = std::sin(static_cast<double>(i) * 0.13);
        p.c2 = std::cos(static_cast<double>(i) * 0.07);
        if (unif(gen) < p_y) p.y = 2.0 * p.c1 - 1.5 * p.c2 + 3.0;
        if (unif(gen) < p_cls) p.cls = (p.c1 + p.c2 > 0.0) ? "pos" : "neg";
    }
    return plots;
}

struct Files {
    TempFile header;
    TempFile species;
};

Files write_csvs(const std::vector<Plot>& plots) {
    std::ostringstream hdr, spc;
    hdr << "plot_id,lat,lon,cov1,cov2,y,cls\n";
    spc << "plot_id,sp,cover\n";
    for (size_t i = 0; i < plots.size(); ++i) {
        const auto& p = plots[i];
        hdr << "P" << i << "," << (40.0 + static_cast<double>(i) * 0.001) << ","
            << (-5.0 + static_cast<double>(i) * 0.001) << "," << p.c1 << "," << p.c2 << ",";
        if (p.y) hdr << *p.y;
        hdr << ",";
        if (p.cls) hdr << *p.cls;
        hdr << "\n";
        spc << "P" << i << ",sp" << (i % 8) << ",1.0\n";
    }
    return Files{TempFile(hdr.str(), ".csv", WriteMode::Binary),
                 TempFile(spc.str(), ".csv", WriteMode::Binary)};
}

TargetSpec y_spec(TargetMissingPolicy missing) {
    auto s = TargetSpec::regression("y");
    s.missing = missing;
    return s;
}

TargetSpec cls_spec(TargetMissingPolicy missing) {
    TargetSpec s;
    s.column_name = "cls";
    s.target_name = "cls";
    s.task = TaskType::Classification;
    s.missing = missing;
    return s;
}

ResolveDataset load(const Files& f, const std::vector<TargetSpec>& targets) {
    return ResolveDataset::from_csv(f.header.path(), f.species.path(),
                                    Roles().coordinates().covariates({"cov1", "cov2"}), targets,
                                    hash_dataset_config());
}

ModelConfig model_config() { return hash_mlp_config({32, 16}); }

TrainConfig train_config(int fixed_epochs = 0, int max_epochs = 300) {
    TrainConfig t = recovery_train_config(max_epochs);
    t.fixed_epochs = fixed_epochs;
    return t;
}

// A seeded fit: the seed fixes the weight initialisation (the global torch
// RNG, drawn at model construction) and prepare_data's seed fixes the split
// and the shuffle.
struct Fit {
    std::unique_ptr<Trainer> trainer;
    TrainResult result;
};

Fit seeded_fit(const ResolveDataset& ds, const TrainConfig& tcfg, float test_size = 0.25f) {
    torch::manual_seed(20261012);
    ResolveModel model(ds.schema(), model_config());
    auto trainer = std::make_unique<Trainer>(model, tcfg);
    trainer->prepare_data(ds, test_size, /*seed=*/11);
    TrainResult result = trainer->fit();
    return Fit{std::move(trainer), std::move(result)};
}

// Every parameter and buffer of the two models is bit-identical.
void require_same_weights(Trainer& a, Trainer& b) {
    auto pa = a.model()->named_parameters();
    auto pb = b.model()->named_parameters();
    REQUIRE(pa.size() == pb.size());
    for (const auto& item : pa) {
        const auto* other = pb.find(item.key());
        REQUIRE(other != nullptr);
        INFO(item.key());
        REQUIRE(torch::equal(item.value(), *other));
    }
    auto ba = a.model()->named_buffers();
    auto bb = b.model()->named_buffers();
    REQUIRE(ba.size() == bb.size());
    for (const auto& item : ba) {
        const auto* other = bb.find(item.key());
        REQUIRE(other != nullptr);
        INFO(item.key());
        REQUIRE(torch::equal(item.value(), *other));
    }
}

int64_t count_labelled(const std::vector<Plot>& plots, bool y, bool cls) {
    int64_t n = 0;
    for (const auto& p : plots) {
        if ((!y || p.y) && (!cls || p.cls)) ++n;
    }
    return n;
}

TargetConfig target_config(const std::string& name, TaskType task,
                           TargetMissingPolicy missing, int num_classes = 0) {
    TargetConfig cfg{name, task, TransformType::None, num_classes, 1.0f};
    cfg.missing = missing;
    return cfg;
}

}  // namespace

// =============================================================================
// Mask reduces to Drop exactly where the two must agree
// =============================================================================

TEST_CASE("Mask on fully labelled targets fits bit-for-bit like Drop",
          "[masking][equivalence]") {
    const auto plots = make_plots(240, 1.0, 1.0, 1);
    const auto files = write_csvs(plots);

    const auto drop = load(files, {y_spec(TargetMissingPolicy::Drop),
                                   cls_spec(TargetMissingPolicy::Drop)});
    const auto mask = load(files, {y_spec(TargetMissingPolicy::Mask),
                                   cls_spec(TargetMissingPolicy::Mask)});
    REQUIRE(drop.plot_ids() == mask.plot_ids());
    REQUIRE(torch::equal(drop.targets().at("y"), mask.targets().at("y")));
    REQUIRE(torch::equal(drop.targets().at("cls"), mask.targets().at("cls")));

    auto a = seeded_fit(drop, train_config(/*fixed_epochs=*/8));
    auto b = seeded_fit(mask, train_config(/*fixed_epochs=*/8));
    REQUIRE(a.result.train_loss_history == b.result.train_loss_history);
    REQUIRE(a.result.test_loss_history == b.result.test_loss_history);
    require_same_weights(*a.trainer, *b.trainer);
}

TEST_CASE("A single masked target drops the same plots and fits bit-for-bit like Drop",
          "[masking][equivalence]") {
    const auto plots = make_plots(300, 0.7, 0.6, 2);
    const auto files = write_csvs(plots);

    SECTION("regression") {
        const auto drop = load(files, {y_spec(TargetMissingPolicy::Drop)});
        const auto mask = load(files, {y_spec(TargetMissingPolicy::Mask)});
        REQUIRE(drop.n_plots() == count_labelled(plots, true, false));
        REQUIRE(drop.plot_ids() == mask.plot_ids());
        REQUIRE(torch::equal(drop.targets().at("y"), mask.targets().at("y")));

        auto a = seeded_fit(drop, train_config(/*fixed_epochs=*/8));
        auto b = seeded_fit(mask, train_config(/*fixed_epochs=*/8));
        REQUIRE(a.result.train_loss_history == b.result.train_loss_history);
        require_same_weights(*a.trainer, *b.trainer);
    }
    SECTION("classification") {
        const auto drop = load(files, {cls_spec(TargetMissingPolicy::Drop)});
        const auto mask = load(files, {cls_spec(TargetMissingPolicy::Mask)});
        REQUIRE(drop.n_plots() == count_labelled(plots, false, true));
        REQUIRE(drop.plot_ids() == mask.plot_ids());
        REQUIRE(drop.schema().targets[0].class_names == mask.schema().targets[0].class_names);
        REQUIRE(torch::equal(drop.targets().at("cls"), mask.targets().at("cls")));

        auto a = seeded_fit(drop, train_config(/*fixed_epochs=*/8));
        auto b = seeded_fit(mask, train_config(/*fixed_epochs=*/8));
        REQUIRE(a.result.train_loss_history == b.result.train_loss_history);
        require_same_weights(*a.trainer, *b.trainer);
    }
}

// =============================================================================
// The loss never reads an unlabelled row
// =============================================================================

TEST_CASE("Rows missing a masked target give its head zero gradient",
          "[masking][loss]") {
    torch::manual_seed(3);
    const int64_t batch = 12;
    const int64_t latent_dim = 5;
    const int n_classes = 3;

    // A shared latent and one linear head per target, standing in for the
    // encoder and the TaskHeads.
    auto latent = torch::randn({batch, latent_dim}, torch::requires_grad());
    torch::nn::Linear head_y(latent_dim, 1);
    torch::nn::Linear head_c(latent_dim, n_classes);

    const std::vector<TargetConfig> cfgs = {
        target_config("y", TaskType::Regression, TargetMissingPolicy::Mask),
        target_config("c", TaskType::Classification, TargetMissingPolicy::Mask, n_classes),
    };
    MultiTaskLoss loss_fn(cfgs, {100, 300}, LossConfigMode::Combined);

    auto y = torch::randn({batch});
    auto c = torch::randint(0, n_classes, {batch}, torch::kLong);
    const std::vector<int64_t> unlabelled_y = {1, 4, 5, 9};
    const std::vector<int64_t> unlabelled_c = {0, 4, 7};
    for (auto i : unlabelled_y) y[i] = std::numeric_limits<float>::quiet_NaN();
    for (auto i : unlabelled_c) c[i] = kUnlabelledClass;

    SECTION("per row: the prediction gradient is exactly zero where the label is missing") {
        auto pred_y = head_y(latent);
        auto pred_c = head_c(latent);
        pred_y.retain_grad();
        pred_c.retain_grad();
        auto [total, parts] = loss_fn.compute({{"y", pred_y}, {"c", pred_c}},
                                              {{"y", y}, {"c", c}}, /*epoch=*/0);
        REQUIRE(std::isfinite(total.item<float>()));
        total.backward();

        const auto gy = pred_y.grad().reshape({-1});
        const auto gc = pred_c.grad();
        const std::set<int64_t> uy(unlabelled_y.begin(), unlabelled_y.end());
        const std::set<int64_t> uc(unlabelled_c.begin(), unlabelled_c.end());
        for (int64_t i = 0; i < batch; ++i) {
            INFO("row " << i);
            if (uy.count(i)) REQUIRE(gy[i].item<float>() == 0.0f);
            else REQUIRE(gy[i].item<float>() != 0.0f);
            if (uc.count(i)) REQUIRE(torch::all(gc[i] == 0).item<bool>());
            else REQUIRE(torch::any(gc[i] != 0).item<bool>());
        }
    }

    SECTION("per head: the gradient equals the Drop loss on the labelled rows") {
        auto [total, parts] = loss_fn.compute({{"y", head_y(latent)}, {"c", head_c(latent)}},
                                              {{"y", y}, {"c", c}}, /*epoch=*/0);
        total.backward();
        const auto masked_wy = head_y->weight.grad().clone();
        const auto masked_wc = head_c->weight.grad().clone();
        head_y->zero_grad();
        head_c->zero_grad();

        // The Drop loss over each target's labelled rows only.
        auto keep_y = labelled_mask(y, TaskType::Regression).nonzero().reshape({-1});
        auto keep_c = labelled_mask(c, TaskType::Classification).nonzero().reshape({-1});
        const std::vector<TargetConfig> drop_y = {
            target_config("y", TaskType::Regression, TargetMissingPolicy::Drop)};
        const std::vector<TargetConfig> drop_c = {
            target_config("c", TaskType::Classification, TargetMissingPolicy::Drop, n_classes)};
        auto ly = MultiTaskLoss(drop_y).compute(
            {{"y", head_y(latent.index_select(0, keep_y))}},
            {{"y", y.index_select(0, keep_y)}}, 0).first;
        auto lc = MultiTaskLoss(drop_c).compute(
            {{"c", head_c(latent.index_select(0, keep_c))}},
            {{"c", c.index_select(0, keep_c)}}, 0).first;
        (ly + lc).backward();
        REQUIRE(torch::allclose(masked_wy, head_y->weight.grad(), 1e-6, 1e-7));
        REQUIRE(torch::allclose(masked_wc, head_c->weight.grad(), 1e-6, 1e-7));
    }

    SECTION("a batch with no labelled row leaves the head without gradient") {
        auto y_none = torch::full({batch}, std::numeric_limits<float>::quiet_NaN());
        auto [total, parts] = loss_fn.compute({{"y", head_y(latent)}, {"c", head_c(latent)}},
                                              {{"y", y_none}, {"c", c}}, /*epoch=*/0);
        total.backward();
        const auto g = head_y->weight.grad();
        REQUIRE((!g.defined() || torch::all(g == 0).item<bool>()));
        REQUIRE(torch::any(head_c->weight.grad() != 0).item<bool>());
    }
}

TEST_CASE("A batch with no labelled row for a masked target adds exactly zero",
          "[masking][loss]") {
    const int64_t batch = 8;
    const std::vector<TargetConfig> cfgs = {
        target_config("y", TaskType::Regression, TargetMissingPolicy::Mask),
        target_config("c", TaskType::Classification, TargetMissingPolicy::Mask, 4),
    };
    torch::manual_seed(5);
    auto pred_y = torch::randn({batch, 1});
    auto pred_c = torch::randn({batch, 4});
    auto y = torch::randn({batch});
    auto c_none = torch::full({batch}, kUnlabelledClass, torch::kLong);

    for (int epoch : {0, 150, 400}) {  // every phase of the combined schedule
        INFO("epoch " << epoch);
        MultiTaskLoss loss_fn(cfgs, {100, 300}, LossConfigMode::Combined);
        auto [total, parts] = loss_fn.compute({{"y", pred_y}, {"c", pred_c}},
                                              {{"y", y}, {"c", c_none}}, epoch);
        REQUIRE(parts.at("c").item<float>() == 0.0f);
        REQUIRE(std::isfinite(total.item<float>()));
        REQUIRE(total.item<float>() == parts.at("y").item<float>());

        auto y_none = torch::full({batch}, std::numeric_limits<float>::quiet_NaN());
        auto both = loss_fn.compute({{"y", pred_y}, {"c", pred_c}},
                                    {{"y", y_none}, {"c", c_none}}, epoch);
        REQUIRE(both.first.item<float>() == 0.0f);
    }

    SECTION("the NCA preset too") {
        MultiTaskLoss loss_fn(cfgs, {100, 300}, LossConfigMode::NCA);
        auto [total, parts] = loss_fn.compute({{"y", pred_y}, {"c", pred_c}},
                                              {{"y", y}, {"c", c_none}}, 0);
        REQUIRE(parts.at("c").item<float>() == 0.0f);
        REQUIRE(std::isfinite(total.item<float>()));
    }
}

// =============================================================================
// Data
// =============================================================================

TEST_CASE("A masked target keeps unlabelled plots with the sentinel",
          "[masking][dataset]") {
    std::vector<Plot> plots(6);
    plots[0] = {0.1, 0.2, 1.0, std::string("A")};
    plots[1] = {0.2, 0.1, std::nullopt, std::string("B")};
    plots[2] = {0.3, 0.4, 3.0, std::nullopt};
    plots[3] = {0.4, 0.3, std::nullopt, std::nullopt};   // labelled for nothing
    plots[4] = {0.5, 0.6, 5.0, std::string("A")};
    plots[5] = {0.6, 0.5, 6.0, std::string("B")};
    const auto files = write_csvs(plots);

    const auto ds = load(files, {y_spec(TargetMissingPolicy::Mask),
                                 cls_spec(TargetMissingPolicy::Mask)});
    // P3 carries no label at all and is the only plot dropped.
    REQUIRE(ds.plot_ids() == std::vector<std::string>{"P0", "P1", "P2", "P4", "P5"});
    const auto y = ds.targets().at("y");
    const auto c = ds.targets().at("cls");
    REQUIRE(std::isnan(y[1].item<float>()));
    REQUIRE(y[2].item<float>() == 3.0f);
    REQUIRE(c[2].item<int64_t>() == kUnlabelledClass);
    REQUIRE(c[1].item<int64_t>() == 1);
    REQUIRE(ds.schema().targets[0].missing == TargetMissingPolicy::Mask);
    REQUIRE(ds.schema().targets[1].missing == TargetMissingPolicy::Mask);

    SECTION("the same file under Drop keeps only the fully labelled plots") {
        const auto drop = load(files, {y_spec(TargetMissingPolicy::Drop),
                                       cls_spec(TargetMissingPolicy::Drop)});
        REQUIRE(drop.plot_ids() == std::vector<std::string>{"P0", "P4", "P5"});
        REQUIRE(drop.schema().targets[0].missing == TargetMissingPolicy::Drop);
    }
    SECTION("Drop on one target and Mask on the other") {
        const auto mixed = load(files, {y_spec(TargetMissingPolicy::Drop),
                                        cls_spec(TargetMissingPolicy::Mask)});
        REQUIRE(mixed.plot_ids() == std::vector<std::string>{"P0", "P2", "P4", "P5"});
        REQUIRE(mixed.targets().at("cls")[1].item<int64_t>() == kUnlabelledClass);
    }
}

TEST_CASE("A masked class list is learned from the plots labelled for it",
          "[masking][dataset]") {
    std::vector<Plot> plots(6);
    plots[0] = {0.1, 0.2, 1.0, std::string("beta")};
    plots[1] = {0.2, 0.1, 2.0, std::nullopt};
    plots[2] = {0.3, 0.4, std::nullopt, std::string("alpha")};  // dropped by y (Drop)
    plots[3] = {0.4, 0.3, 4.0, std::string("gamma")};
    plots[4] = {0.5, 0.6, 5.0, std::nullopt};
    plots[5] = {0.6, 0.5, 6.0, std::string("beta")};
    const auto files = write_csvs(plots);

    const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop),
                                 cls_spec(TargetMissingPolicy::Mask)});
    // "alpha" occurs only on a plot the Drop target removes, so it is no
    // class: the list is fit on the kept plots that carry a label.
    const auto& cfg = ds.schema().targets[1];
    REQUIRE(cfg.class_names == std::vector<std::string>{"beta", "gamma"});
    REQUIRE(cfg.num_classes == 2);
    REQUIRE(ds.plot_ids() == std::vector<std::string>{"P0", "P1", "P3", "P4", "P5"});
    const auto c = ds.targets().at("cls");
    REQUIRE(c[0].item<int64_t>() == 0);
    REQUIRE(c[1].item<int64_t>() == kUnlabelledClass);
    REQUIRE(c[2].item<int64_t>() == 1);
    REQUIRE(c[3].item<int64_t>() == kUnlabelledClass);

    SECTION("a label outside an explicit mapping is unlabelled under Mask") {
        auto spec = cls_spec(TargetMissingPolicy::Mask);
        spec.class_mapping = {{"beta", 0}};
        const auto explicit_ds = load(files, {y_spec(TargetMissingPolicy::Mask), spec});
        // P3's "gamma" is outside the mapping, so P3 trains y only. P2 has no
        // y and an unmapped "alpha": labelled for nothing, it is dropped.
        REQUIRE(explicit_ds.plot_ids() ==
                std::vector<std::string>{"P0", "P1", "P3", "P4", "P5"});
        const auto codes = explicit_ds.targets().at("cls");
        REQUIRE(codes[0].item<int64_t>() == 0);
        REQUIRE(codes[2].item<int64_t>() == kUnlabelledClass);
        REQUIRE(codes[4].item<int64_t>() == 0);
    }
    SECTION("a masked target labelled on no plot is refused") {
        std::vector<Plot> none(3);
        for (size_t i = 0; i < none.size(); ++i) none[i] = {0.1 * double(i), 0.2, 1.0, std::nullopt};
        const auto none_files = write_csvs(none);
        REQUIRE_THROWS_AS(load(none_files, {y_spec(TargetMissingPolicy::Drop),
                                            cls_spec(TargetMissingPolicy::Mask)}),
                          std::invalid_argument);
    }
}

TEST_CASE("The target scaler is fit on the labelled training plots",
          "[masking][trainer]") {
    const auto plots = make_plots(200, 0.5, 1.0, 4);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Mask),
                                 cls_spec(TargetMissingPolicy::Drop)});

    torch::manual_seed(1);
    ResolveModel model(ds.schema(), model_config());
    Trainer trainer(model, train_config(/*fixed_epochs=*/1));
    trainer.prepare_data(ds, 0.25f, 11);

    // Reproduce the scaler from the raw labelled values of the training plots.
    const auto y = ds.targets().at("y").index_select(0, trainer.train_indices());
    const auto labelled = y.index_select(0, torch::isfinite(y).nonzero().reshape({-1}));
    REQUIRE(labelled.numel() > 0);
    REQUIRE(labelled.numel() < y.numel());
    const auto& [mean, scale] = trainer.scalers().target_scalers.at("y");
    REQUIRE(mean.item<float>() == Catch::Approx(labelled.mean().item<float>()).margin(1e-5));
    REQUIRE(scale.item<float>() ==
            Catch::Approx(labelled.std().item<float>()).epsilon(1e-3));

    trainer.fit();
    const auto res = trainer.compute_residuals("y");
    // The residuals are on the labelled test plots, in original units.
    const auto y_test = ds.targets().at("y").index_select(0, trainer.test_indices());
    const int64_t n_test_labelled = torch::isfinite(y_test).sum().item<int64_t>();
    REQUIRE(static_cast<int64_t>(res.actuals.size()) == n_test_labelled);
    REQUIRE(res.test_rows.size() == res.actuals.size());
    for (size_t i = 0; i < res.test_rows.size(); ++i) {
        REQUIRE(res.actuals[i] == Catch::Approx(y_test[res.test_rows[i]].item<float>()).margin(1e-4));
    }
}

// =============================================================================
// Checkpoint
// =============================================================================

TEST_CASE("The missing-label policy round-trips through the checkpoint",
          "[masking][checkpoint]") {
    const auto plots = make_plots(160, 0.6, 0.6, 5);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Mask),
                                 cls_spec(TargetMissingPolicy::Drop)});

    torch::manual_seed(2);
    ResolveModel model(ds.schema(), model_config());
    Trainer trainer(model, train_config(/*fixed_epochs=*/2));
    trainer.prepare_data(ds, 0.25f, 11);
    trainer.fit();

    const TempPath ckpt(".pt");
    RunMetadata meta;
    trainer.save(ckpt.path(), &meta);
    auto [loaded, scalers, vocab] = Trainer::load(ckpt.path(), torch::kCPU);
    REQUIRE(loaded->schema().targets.size() == 2);
    for (const auto& t : loaded->schema().targets) {
        INFO(t.name);
        REQUIRE(t.missing == (t.name == "y" ? TargetMissingPolicy::Mask
                                            : TargetMissingPolicy::Drop));
    }

    // The JSON sidecar names the policy of each target.
    std::string json_path = ckpt.path();
    json_path = json_path.substr(0, json_path.size() - 3) + ".json";
    {
        std::ifstream in(json_path);
        std::stringstream buf;
        buf << in.rdbuf();
        const auto text = buf.str();
        REQUIRE(text.find("\"name\": \"y\"") != std::string::npos);
        REQUIRE(text.find("\"missing\": \"mask\"") != std::string::npos);
        REQUIRE(text.find("\"missing\": \"drop\"") != std::string::npos);
    }
    std::filesystem::remove(json_path);

    SECTION("a checkpoint written before the field existed loads as Drop") {
        ResolveSchema schema = ds.schema();
        torch::serialize::OutputArchive out;
        save_schema(out, schema);
        std::ostringstream bytes;
        out.save_to(bytes);

        // Copy every key except the per-target policy into a fresh archive.
        std::istringstream src_stream(bytes.str());
        torch::serialize::InputArchive src;
        src.load_from(src_stream);
        torch::serialize::OutputArchive legacy;
        int n_dropped = 0;
        for (const auto& key : src.keys()) {
            const std::string suffix = "_missing";
            if (key.size() > suffix.size() &&
                key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0 &&
                key.rfind("schema_target_", 0) == 0) {
                ++n_dropped;
                continue;
            }
            torch::Tensor t;
            REQUIRE(src.try_read(key, t));
            legacy.write(key, t);
        }
        REQUIRE(n_dropped == 2);
        std::ostringstream legacy_bytes;
        legacy.save_to(legacy_bytes);
        std::istringstream legacy_stream(legacy_bytes.str());
        torch::serialize::InputArchive in;
        in.load_from(legacy_stream);
        const auto reloaded = load_schema(in);
        REQUIRE(reloaded.targets.size() == 2);
        for (const auto& t : reloaded.targets) {
            REQUIRE(t.missing == TargetMissingPolicy::Drop);
        }
    }
}

// =============================================================================
// Recovery: two partly labelled targets on one shared encoder
// =============================================================================

TEST_CASE("A masked two-target fit recovers both targets from partly labelled plots",
          "[masking][recovery]") {
    // Each target is labelled on an independent random half of the plots, so
    // about a quarter carry both labels, half carry one and a quarter none.
    const int64_t n = 1000;
    const auto plots = make_plots(n, 0.5, 0.5, 7);
    const auto files = write_csvs(plots);

    const auto drop = load(files, {y_spec(TargetMissingPolicy::Drop),
                                   cls_spec(TargetMissingPolicy::Drop)});
    const auto mask = load(files, {y_spec(TargetMissingPolicy::Mask),
                                   cls_spec(TargetMissingPolicy::Mask)});
    const int64_t n_both = count_labelled(plots, true, true);
    const int64_t n_y = count_labelled(plots, true, false);
    const int64_t n_cls = count_labelled(plots, false, true);
    int64_t n_any = 0;
    for (const auto& p : plots) if (p.y || p.cls) ++n_any;
    REQUIRE(drop.n_plots() == n_both);
    REQUIRE(mask.n_plots() == n_any);
    REQUIRE(mask.n_plots() > 2 * drop.n_plots());
    REQUIRE(torch::isfinite(mask.targets().at("y")).sum().item<int64_t>() == n_y);
    REQUIRE((mask.targets().at("cls") >= 0).sum().item<int64_t>() == n_cls);

    auto fit = seeded_fit(mask, train_config());
    auto& trainer = *fit.trainer;

    // Per-target n is reported beside the metrics, and matches the labelled
    // test plots.
    const auto y_test = mask.targets().at("y").index_select(0, trainer.test_indices());
    const auto c_test = mask.targets().at("cls").index_select(0, trainer.test_indices());
    const int64_t n_y_test = torch::isfinite(y_test).sum().item<int64_t>();
    const int64_t n_c_test = (c_test >= 0).sum().item<int64_t>();
    REQUIRE(fit.result.final_metrics.at("y").at(kLabelledMetric) == static_cast<float>(n_y_test));
    REQUIRE(fit.result.final_metrics.at("cls").at(kLabelledMetric) == static_cast<float>(n_c_test));
    REQUIRE(std::isfinite(fit.result.final_metrics.at("y").at("mae")));
    REQUIRE(std::isfinite(fit.result.final_metrics.at("cls").at("accuracy")));

    // Regression: the held-out labelled plots are recovered.
    const auto res = trainer.compute_residuals("y");
    REQUIRE(static_cast<int64_t>(res.predictions.size()) == n_y_test);
    const double r = pearson(res.predictions, res.actuals);
    INFO("held-out correlation " << r);
    REQUIRE(r > 0.9);

    // Classification: scored on the labelled test plots, each row placed in
    // the fold by test_rows.
    const auto cp = trainer.compute_classification_predictions("cls");
    REQUIRE(cp.actuals.size(0) == n_c_test);
    REQUIRE(torch::all(cp.actuals >= 0).item<bool>());
    REQUIRE(torch::equal(cp.actuals, c_test.index_select(0, cp.test_rows)));
    const float acc = (cp.predicted_classes == cp.actuals).to(torch::kFloat32).mean().item<float>();
    INFO("held-out accuracy " << acc);
    REQUIRE(acc > 0.8f);

    // Baselines exist for both and are scored on the labelled plots.
    REQUIRE(fit.result.baselines.count("y") == 1);
    REQUIRE(fit.result.baselines.count("cls") == 1);
    REQUIRE(std::isfinite(fit.result.baselines.at("y").model_mse));
    REQUIRE(fit.result.baselines.at("cls").model_accuracy > 0.8f);

    // Calibration reads the labelled plots only.
    const auto cal = trainer.compute_calibration("cls", 10);
    int64_t n_binned = 0;
    for (const auto& b : cal.bins) n_binned += b.count;
    REQUIRE(n_binned == n_c_test);
}

TEST_CASE("Cross-validation keeps masked targets consistent across folds",
          "[masking][cv]") {
    const auto plots = make_plots(300, 0.6, 0.6, 9);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Mask),
                                 cls_spec(TargetMissingPolicy::Mask)});

    torch::manual_seed(4);
    ResolveModel model(ds.schema(), model_config());
    Trainer trainer(model, train_config(/*fixed_epochs=*/3));
    trainer.prepare_data(ds, 0.2f, 11);
    const auto before = trainer.compute_classification_predictions("cls").actuals.clone();

    const auto cv = trainer.cross_validate(3, 13);
    REQUIRE(cv.fold_results.size() == 3);
    float n_y_total = 0.0f;
    for (const auto& fold : cv.fold_results) {
        const auto& m = fold.final_metrics.at("y");
        REQUIRE(std::isfinite(m.at("mae")));
        REQUIRE(std::isfinite(fold.final_metrics.at("cls").at("accuracy")));
        n_y_total += m.at(kLabelledMetric);
    }
    // The folds partition the plots, so the labelled counts add up.
    REQUIRE(n_y_total == static_cast<float>(torch::isfinite(ds.targets().at("y")).sum().item<int64_t>()));
    // The pre-CV split is restored, sentinels included.
    REQUIRE(torch::equal(trainer.compute_classification_predictions("cls").actuals, before));
}

// =============================================================================
// The loss history split by target
// =============================================================================

namespace {

// Per epoch, the per-target terms of `names` -- plus `moe_aux`, the term that
// belongs to no target, when given -- add up to `total`.
void require_decomposes(const std::unordered_map<std::string, std::vector<float>>& by_target,
                        const std::vector<float>& total,
                        const std::vector<std::string>& names,
                        const std::vector<float>* moe_aux = nullptr) {
    REQUIRE(by_target.size() == names.size());
    for (const auto& name : names) {
        INFO("target " << name);
        REQUIRE(by_target.count(name) == 1);
        REQUIRE(by_target.at(name).size() == total.size());
    }
    if (moe_aux) REQUIRE(moe_aux->size() == total.size());
    for (size_t e = 0; e < total.size(); ++e) {
        double sum = moe_aux ? static_cast<double>((*moe_aux)[e]) : 0.0;
        for (const auto& name : names) sum += by_target.at(name)[e];
        INFO("epoch " << e << ": terms " << sum << ", total " << total[e]);
        REQUIRE(std::isfinite(total[e]));
        REQUIRE(sum == Catch::Approx(total[e]).epsilon(1e-5).margin(1e-6));
    }
}

}  // namespace

TEST_CASE("The per-target loss history adds up to the total, batches without a label included",
          "[masking][loss_history]") {
    // One plot in ten carries a class label and a batch holds four plots, so
    // the masked target has fewer labelled training plots than the epoch has
    // batches: some batches hold none of its labels and add 0 for it.
    const auto plots = make_plots(240, 1.0, 0.1, 21);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop),
                                 cls_spec(TargetMissingPolicy::Mask)});
    auto tcfg = train_config(/*fixed_epochs=*/5);
    tcfg.batch_size = 4;
    auto fit = seeded_fit(ds, tcfg);
    const auto& r = fit.result;

    const auto cls_train = ds.targets().at("cls").index_select(0, fit.trainer->train_indices());
    const int64_t n_cls_labelled = (cls_train >= 0).sum().item<int64_t>();
    const int64_t n_batches = (cls_train.size(0) + tcfg.batch_size - 1) / tcfg.batch_size;
    REQUIRE(n_cls_labelled > 0);
    REQUIRE(n_cls_labelled < n_batches);

    REQUIRE(r.train_loss_history.size() == 5);
    REQUIRE(r.test_loss_history.size() == 5);
    require_decomposes(r.train_loss_history_by_target, r.train_loss_history, {"y", "cls"});
    require_decomposes(r.test_loss_history_by_target, r.test_loss_history, {"y", "cls"});
    REQUIRE(r.train_moe_aux_loss_history.empty());

    for (size_t e = 0; e < r.train_loss_history.size(); ++e) {
        INFO("epoch " << e);
        CHECK(r.train_loss_history_by_target.at("cls")[e] > 0.0f);
        CHECK(r.train_loss_history_by_target.at("y")[e] > 0.0f);
        CHECK(r.test_loss_history_by_target.at("cls")[e] >= 0.0f);
    }
}

TEST_CASE("A target's weight scales its term of the loss history",
          "[masking][loss_history]") {
    const auto plots = make_plots(200, 1.0, 1.0, 22);
    const auto files = write_csvs(plots);
    auto y = y_spec(TargetMissingPolicy::Drop);
    y.weight = 3.0f;
    const auto ds = load(files, {y, cls_spec(TargetMissingPolicy::Drop)});
    auto fit = seeded_fit(ds, train_config(/*fixed_epochs=*/3));
    require_decomposes(fit.result.train_loss_history_by_target,
                       fit.result.train_loss_history, {"y", "cls"});
    require_decomposes(fit.result.test_loss_history_by_target,
                       fit.result.test_loss_history, {"y", "cls"});
}

TEST_CASE("A single target's loss history is the total history",
          "[masking][loss_history]") {
    const auto plots = make_plots(200, 0.8, 0.8, 23);
    const auto files = write_csvs(plots);

    SECTION("regression") {
        const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop)});
        auto fit = seeded_fit(ds, train_config(/*fixed_epochs=*/4));
        REQUIRE(fit.result.train_loss_history_by_target.size() == 1);
        REQUIRE(fit.result.train_loss_history_by_target.at("y") == fit.result.train_loss_history);
        REQUIRE(fit.result.test_loss_history_by_target.at("y") == fit.result.test_loss_history);
    }
    SECTION("masked classification") {
        const auto ds = load(files, {cls_spec(TargetMissingPolicy::Mask)});
        auto fit = seeded_fit(ds, train_config(/*fixed_epochs=*/4));
        REQUIRE(fit.result.train_loss_history_by_target.size() == 1);
        REQUIRE(fit.result.train_loss_history_by_target.at("cls") == fit.result.train_loss_history);
        REQUIRE(fit.result.test_loss_history_by_target.at("cls") == fit.result.test_loss_history);
    }
    SECTION("early stopping") {
        const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop)});
        auto fit = seeded_fit(ds, train_config(/*fixed_epochs=*/0, /*max_epochs=*/6));
        REQUIRE_FALSE(fit.result.train_loss_history.empty());
        REQUIRE(fit.result.train_loss_history_by_target.at("y") == fit.result.train_loss_history);
        REQUIRE(fit.result.test_loss_history_by_target.at("y") == fit.result.test_loss_history);
    }
}

TEST_CASE("Nothing held out leaves the per-target test history empty",
          "[masking][loss_history]") {
    const auto plots = make_plots(160, 1.0, 0.5, 24);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop),
                                 cls_spec(TargetMissingPolicy::Mask)});
    auto fit = seeded_fit(ds, train_config(/*fixed_epochs=*/3), /*test_size=*/0.0f);
    REQUIRE(fit.result.train_loss_history.size() == 3);
    REQUIRE(fit.result.test_loss_history.empty());
    REQUIRE(fit.result.test_loss_history_by_target.empty());
    require_decomposes(fit.result.train_loss_history_by_target,
                       fit.result.train_loss_history, {"y", "cls"});
}

TEST_CASE("The MoE load-balancing term is recorded beside the targets' terms",
          "[masking][loss_history][moe]") {
    const auto plots = make_plots(200, 1.0, 0.6, 25);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Drop),
                                 cls_spec(TargetMissingPolicy::Mask)});

    torch::manual_seed(20261012);
    auto mcfg = model_config();
    mcfg.moe_routing = MoERoutingType::Soft;
    mcfg.moe_aux_loss_weight = 0.5f;
    ResolveModel model(ds.schema(), mcfg);
    REQUIRE(model->uses_moe());
    Trainer trainer(model, train_config(/*fixed_epochs=*/3));
    trainer.prepare_data(ds, 0.25f, 11);
    const auto r = trainer.fit();

    REQUIRE(r.train_moe_aux_loss_history.size() == 3);
    for (float aux : r.train_moe_aux_loss_history) CHECK(aux > 0.0f);
    require_decomposes(r.train_loss_history_by_target, r.train_loss_history, {"y", "cls"},
                       &r.train_moe_aux_loss_history);
    // The held-out selection loss carries no load-balancing term.
    require_decomposes(r.test_loss_history_by_target, r.test_loss_history, {"y", "cls"});
}

TEST_CASE("Each cross-validation fold carries its own per-target loss history",
          "[masking][loss_history][cv]") {
    const auto plots = make_plots(240, 0.7, 0.7, 26);
    const auto files = write_csvs(plots);
    const auto ds = load(files, {y_spec(TargetMissingPolicy::Mask),
                                 cls_spec(TargetMissingPolicy::Mask)});

    torch::manual_seed(6);
    ResolveModel model(ds.schema(), model_config());
    Trainer trainer(model, train_config(/*fixed_epochs=*/3));
    trainer.prepare_data(ds, 0.2f, 11);
    const auto cv = trainer.cross_validate(3, 13);
    REQUIRE(cv.fold_results.size() == 3);
    for (const auto& fold : cv.fold_results) {
        REQUIRE(fold.train_loss_history.size() == 3);
        require_decomposes(fold.train_loss_history_by_target, fold.train_loss_history,
                           {"y", "cls"});
        require_decomposes(fold.test_loss_history_by_target, fold.test_loss_history,
                           {"y", "cls"});
    }
}

TEST_CASE("MultiTaskLoss::contributions are the weighted terms compute() sums",
          "[masking][loss_history][loss]") {
    torch::manual_seed(7);
    const int64_t batch = 10;
    auto y_cfg = target_config("y", TaskType::Regression, TargetMissingPolicy::Mask);
    y_cfg.weight = 2.5f;
    const auto c_cfg = target_config("c", TaskType::Classification, TargetMissingPolicy::Mask, 3);
    const MultiTaskLoss loss_fn({y_cfg, c_cfg}, {100, 300}, LossConfigMode::Combined);

    const auto pred_y = torch::randn({batch, 1});
    const auto pred_c = torch::randn({batch, 3});
    const auto y = torch::randn({batch});
    const auto c_none = torch::full({batch}, kUnlabelledClass, torch::kInt64);

    for (int epoch : {0, 150, 400}) {
        INFO("epoch " << epoch);
        auto [total, parts] = loss_fn.compute({{"y", pred_y}, {"c", pred_c}},
                                              {{"y", y}, {"c", c_none}}, epoch);
        const auto terms = loss_fn.contributions(parts);
        REQUIRE(terms.sizes() == torch::IntArrayRef{2});
        REQUIRE(terms.dtype() == torch::kFloat32);
        REQUIRE_FALSE(terms.requires_grad());
        // targets() order: y, then c with no labelled row, which adds exactly 0.
        REQUIRE(terms[0].item<float>() == (2.5f * parts.at("y")).item<float>());
        REQUIRE(terms[1].item<float>() == 0.0f);
        REQUIRE(terms.sum().item<float>() == total.item<float>());
    }

    // A target compute() could not score holds 0.
    auto [total, parts] = loss_fn.compute({{"y", pred_y}}, {{"y", y}}, 0);
    const auto terms = loss_fn.contributions(parts);
    REQUIRE(terms[1].item<float>() == 0.0f);
    REQUIRE(terms[0].item<float>() == total.item<float>());
}
