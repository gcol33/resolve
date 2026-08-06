// Tests for issue #105 (the effective batch size is unreadable after fit()),
// the #110 trainer cleanups, and the CLI's declarative flag table (#104).
//
// Driving the OOM auto-halve without a GPU
// ---------------------------------------
// Trainer::fit wraps its epoch loop in a catch for c10::OutOfMemoryError. The
// progress line it prints at the end of every tenth epoch goes through the
// caller-supplied TrainConfig::log callback, which is invoked INSIDE that try
// block -- so a callback that throws an OutOfMemoryError once puts the real
// retry path (release_training_state, decide_oom_retry, halve, restore the
// initial weights, restart from epoch 0) under test on a CPU build, with no
// fault-injection seam added to production code.
//
// What that does NOT cover: the allocator failure itself. Whether a genuine
// CUDA OOM surfaces as c10::OutOfMemoryError or as an AcceleratorError carrying
// cudaErrorMemoryAllocation is decided before the catch; from the catch onward
// (which is everything this file asserts on) the path is identical.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"

#include "../cli/arg_parser.hpp"
#include "../cli/cli_spec.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;

namespace {

class TempPath {
public:
    explicit TempPath(const std::string& suffix) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_effbatch_" + std::to_string(counter_++) + suffix);
    }
    ~TempPath() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }

    void write(const std::string& content) const {
        std::ofstream file(path_);
        file << content;
    }

private:
    std::filesystem::path path_;
    static int counter_;
};
int TempPath::counter_ = 0;

// Small regression dataset: one hash-encoded species per plot, two covariates,
// a linear target. Fits in one 64-row batch, so the halved run is genuinely a
// different loop shape (two batches).
ResolveDataset make_dataset(int64_t n_plots) {
    std::ostringstream hdr;
    hdr << "plot_id,cov1,cov2,y\n";
    std::ostringstream spc;
    spc << "plot_id,sp,cover\n";
    for (int64_t i = 0; i < n_plots; ++i) {
        const double c1 = static_cast<double>(i % 7) * 1.1;
        const double c2 = static_cast<double>(i % 11) * 0.9;
        hdr << "P" << i << "," << c1 << "," << c2 << "," << (c1 + c2) << "\n";
        spc << "P" << i << ",sp" << (i % 13) << ",1.0\n";
    }

    TempPath header_csv(".csv");
    TempPath species_csv(".csv");
    header_csv.write(hdr.str());
    species_csv.write(spc.str());

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.covariates = {"cov1", "cov2"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4;
    dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false;
    dcfg.track_unknown_count = false;

    return ResolveDataset::from_csv(header_csv.path(), species_csv.path(), roles,
                                    {TargetSpec::regression("y")}, dcfg);
}

ModelConfig make_model_config() {
    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4;
    mcfg.hidden_dims = {8, 4};
    return mcfg;
}

// A TrainConfig::log callback that raises an out-of-memory error the first
// `n_failures` times fit() prints an epoch progress line. Every other message
// (including the retry notice fit() emits after catching) passes through
// silently.
class OomInjector {
public:
    explicit OomInjector(int n_failures)
        : remaining_(std::make_shared<int>(n_failures)) {}

    [[nodiscard]] LogCallback callback() const {
        auto remaining = remaining_;
        return [remaining](const std::string& msg) {
            if (msg.rfind("Epoch ", 0) == 0 && *remaining > 0) {
                --(*remaining);
                C10_THROW_ERROR(OutOfMemoryError,
                                "CUDA out of memory (injected by OomInjector)");
            }
        };
    }

private:
    std::shared_ptr<int> remaining_;
};

int read_checkpoint_int(const std::string& path, const char* key) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::Tensor value;
    REQUIRE(archive.try_read(key, value));
    return value.item<int>();
}

}  // namespace

// =============================================================================
// Issue #105: the effective batch size after fit()
// =============================================================================

TEST_CASE("A clean fit reports the requested batch size as effective",
          "[effective_batch]") {
    auto dataset = make_dataset(64);
    ResolveModel model(dataset.schema(), make_model_config());

    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.batch_size_floor = 8;
    tcfg.max_epochs = 2;
    tcfg.patience = 2;
    tcfg.device = torch::kCPU;
    tcfg.log = null_log;

    Trainer trainer(model, tcfg);
    REQUIRE(trainer.effective_batch_size() == 0);  // no fit yet

    trainer.prepare_data(dataset, 0.25f, 42);
    const TrainResult result = trainer.fit();

    CHECK(result.effective_batch_size == 64);
    CHECK(trainer.effective_batch_size() == 64);
    CHECK(trainer.config().batch_size == 64);
}

TEST_CASE("An OOM-halved fit reports the halved batch size, config the requested one",
          "[effective_batch]") {
    auto dataset = make_dataset(64);
    ResolveModel model(dataset.schema(), make_model_config());

    OomInjector injector(1);

    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.batch_size_floor = 8;
    tcfg.max_epochs = 2;
    tcfg.patience = 2;
    tcfg.device = torch::kCPU;
    tcfg.log = injector.callback();

    Trainer trainer(model, tcfg);
    trainer.prepare_data(dataset, 0.25f, 42);
    const TrainResult result = trainer.fit();

    // The retry fired once: 64 -> 32, which is above the floor of 8.
    CHECK(result.effective_batch_size == 32);
    CHECK(trainer.effective_batch_size() == 32);

    // config() still reports what the caller asked for, so a later
    // cross_validate does not silently inherit the shrink. Before this issue
    // that made the fallback undetectable in memory: the CLI compared these two
    // and always found them equal.
    CHECK(trainer.config().batch_size == 64);
    CHECK(result.effective_batch_size != trainer.config().batch_size);

    // The restart re-ran the whole schedule from epoch 0, so the histories
    // belong to the halved attempt alone.
    CHECK(result.train_loss_history.size() == 2);
    CHECK(result.test_loss_history.size() == 2);
}

TEST_CASE("A checkpoint saved after an OOM-halved fit records both batch sizes",
          "[effective_batch]") {
    auto dataset = make_dataset(64);
    ResolveModel model(dataset.schema(), make_model_config());

    OomInjector injector(1);

    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.batch_size_floor = 8;
    tcfg.max_epochs = 2;
    tcfg.patience = 2;
    tcfg.device = torch::kCPU;
    tcfg.log = injector.callback();

    Trainer trainer(model, tcfg);
    trainer.prepare_data(dataset, 0.25f, 42);
    const TrainResult result = trainer.fit();
    REQUIRE(result.effective_batch_size == 32);

    // save() runs AFTER fit() restored config_.batch_size, so the effective
    // value has to come from the trainer's tracked member, not the config.
    TempPath checkpoint(".pt");
    trainer.save(checkpoint.path());

    CHECK(read_checkpoint_int(checkpoint.path(), "train_batch_size") == 64);
    CHECK(read_checkpoint_int(checkpoint.path(), "train_effective_batch_size") == 32);
    CHECK(read_checkpoint_int(checkpoint.path(), "train_batch_size_floor") == 8);

    // load_train_config recovers the REQUESTED recipe, so re-running it asks
    // for the same batch size the operator originally chose.
    CHECK(Trainer::load_train_config(checkpoint.path()).batch_size == 64);
}

TEST_CASE("An OOM that would halve below the floor is rethrown with context",
          "[effective_batch]") {
    auto dataset = make_dataset(64);
    ResolveModel model(dataset.schema(), make_model_config());

    OomInjector injector(8);  // fails every attempt

    TrainConfig tcfg;
    tcfg.batch_size = 64;
    tcfg.batch_size_floor = 64;  // halving to 32 is already below the floor
    tcfg.max_epochs = 2;
    tcfg.patience = 2;
    tcfg.device = torch::kCPU;
    tcfg.log = injector.callback();

    Trainer trainer(model, tcfg);
    trainer.prepare_data(dataset, 0.25f, 42);

    bool threw = false;
    try {
        trainer.fit();
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string what = e.what();
        CHECK(what.find("batch_size_floor=64") != std::string::npos);
        CHECK(what.find("injected by OomInjector") != std::string::npos);
    }
    CHECK(threw);
}

// =============================================================================
// Issue #110 item 2: train_epoch's scaler source
// =============================================================================

TEST_CASE("CPU training runs the no-scaler-cache branch of train_epoch",
          "[effective_batch]") {
    // On CPU there is no GPU scaler cache, so train_epoch takes the branch
    // whose alternative used to be a materialized temporary (the conditional
    // yielded a prvalue, so the reference bound to a lifetime-extended copy of
    // gpu_scalers_ instead of to the member). It is now a static empty map.
    // A completed CPU fit with finite losses is the behavioural guard.
    auto dataset = make_dataset(64);
    ResolveModel model(dataset.schema(), make_model_config());

    TrainConfig tcfg;
    tcfg.batch_size = 16;
    tcfg.max_epochs = 3;
    tcfg.patience = 3;
    tcfg.device = torch::kCPU;
    tcfg.log = null_log;

    Trainer trainer(model, tcfg);
    trainer.prepare_data(dataset, 0.25f, 42);
    const TrainResult result = trainer.fit();

    REQUIRE(result.train_loss_history.size() == 3);
    for (float loss : result.train_loss_history) {
        CHECK(std::isfinite(loss));
    }
}

// =============================================================================
// Issue #104: the CLI flag table
// =============================================================================

using resolve_cli::Arity;
using resolve_cli::ArgError;
using resolve_cli::CommandSpec;
using resolve_cli::FlagSpec;
using resolve_cli::ParsedArgs;
using resolve_cli::parse_args;
using resolve_cli::render_usage;

namespace {

const CommandSpec& toy_spec() {
    static const CommandSpec spec(
        "toy", "Toy Options:",
        {
            {"--max-epochs", Arity::Value, "N", "500", "Maximum epochs"},
            {"--test-size", Arity::Value, "FLOAT", "0.2", "Test split ratio"},
            {"--covariate", Arity::Repeatable, "COL", "", "A covariate column"},
            {"--cuda", Arity::Flag, "", "", "Use CUDA"},
            {"--amp", Arity::Flag, "", "", "Enable AMP"},
            {"--no-amp", Arity::Flag, "", "", "Disable AMP"},
            {"--hidden-dims", Arity::Value, "LIST", "8,4", "Hidden widths"},
        });
    return spec;
}

}  // namespace

TEST_CASE("The flag table parses values, flags, repeats and defaults", "[cli_args]") {
    const ParsedArgs args = parse_args(
        toy_spec(),
        {"--max-epochs", "10", "--covariate", "elev", "--covariate", "slope", "--cuda"});

    const std::vector<std::string> expected_covariates{"elev", "slope"};
    const std::vector<std::string> expected_dims{"8", "4"};

    CHECK(args.get_int("--max-epochs") == 10);
    CHECK(args.has("--cuda"));
    CHECK(args.get_all("--covariate") == expected_covariates);

    // Untouched flags fall back to the table's declared default.
    CHECK(args.get_float("--test-size") == 0.2f);
    CHECK_FALSE(args.has("--test-size"));
    CHECK(args.get_list("--hidden-dims") == expected_dims);
}

TEST_CASE("An unknown flag is rejected and the near-miss is named", "[cli_args]") {
    // The two spellings that used to be silently ignored: a dropped hyphen and
    // an underscore. Both normalize onto a declared flag.
    CHECK_THROWS_AS(parse_args(toy_spec(), {"--maxepochs", "10"}), ArgError);
    CHECK_THROWS_AS(parse_args(toy_spec(), {"--test_size", "0.3"}), ArgError);

    try {
        parse_args(toy_spec(), {"--maxepochs", "10"});
    } catch (const ArgError& e) {
        const std::string what = e.what();
        CHECK(what.find("--maxepochs") != std::string::npos);
        CHECK(what.find("--max-epochs") != std::string::npos);
    }

    // No near-miss: the message points at the help instead.
    try {
        parse_args(toy_spec(), {"--nonsense", "1"});
    } catch (const ArgError& e) {
        CHECK(std::string(e.what()).find("resolve help") != std::string::npos);
    }
}

TEST_CASE("A missing value, a stray positional, and a bad number are rejected",
          "[cli_args]") {
    CHECK_THROWS_AS(parse_args(toy_spec(), {"--max-epochs"}), ArgError);
    CHECK_THROWS_AS(parse_args(toy_spec(), {"--max-epochs", "--cuda"}), ArgError);
    CHECK_THROWS_AS(parse_args(toy_spec(), {"train.csv"}), ArgError);

    const ParsedArgs args = parse_args(toy_spec(), {"--max-epochs", "ten"});
    CHECK_THROWS_AS(args.get_int("--max-epochs"), ArgError);
}

TEST_CASE("A negative value is a value, not a flag", "[cli_args]") {
    const ParsedArgs args = parse_args(toy_spec(), {"--max-epochs", "-1"});
    CHECK(args.get_int("--max-epochs") == -1);
}

TEST_CASE("Opposing presence flags resolve last-wins", "[cli_args]") {
    CHECK(parse_args(toy_spec(), {}).get_switch("--amp", "--no-amp", false) == false);
    CHECK(parse_args(toy_spec(), {"--amp"}).get_switch("--amp", "--no-amp", false) == true);
    CHECK(parse_args(toy_spec(), {"--amp", "--no-amp"})
              .get_switch("--amp", "--no-amp", false) == false);
    CHECK(parse_args(toy_spec(), {"--no-amp", "--amp"})
              .get_switch("--amp", "--no-amp", false) == true);
}

TEST_CASE("Reading a flag the table does not declare is an error, not a default",
          "[cli_args]") {
    const ParsedArgs args = parse_args(toy_spec(), {});
    CHECK_THROWS_AS(args.get("--not-declared"), ArgError);
    CHECK_THROWS_AS(args.has("--not-declared"), ArgError);
}

TEST_CASE("Usage is generated from the table", "[cli_args]") {
    const std::string usage = render_usage(toy_spec());
    for (const auto& flag : toy_spec().flags()) {
        CHECK(usage.find(flag.name) != std::string::npos);
    }
    // Defaults are rendered, so help cannot disagree with what the parser uses.
    CHECK(usage.find("(default: 500)") != std::string::npos);
    CHECK(usage.find("Toy Options:") != std::string::npos);
}

TEST_CASE("The shipped command tables declare the flags their commands read",
          "[cli_args]") {
    const CommandSpec& train = resolve_cli::train_spec();
    for (const char* name : {"--header", "--species", "--output", "--covariate",
                             "--categorical", "--target", "--seed", "--encoding",
                             "--batch-size", "--batch-size-floor", "--max-epochs",
                             "--weight-decay", "--lr-scheduler", "--lr-min",
                             "--lr-gamma", "--checkpoint-dir", "--checkpoint-every",
                             "--no-amp", "--cudnn-benchmark", "--band-threshold",
                             "--nca-temperature", "--nca-neighbors", "--nca-weight",
                             "--cover-dropout", "--transformer-ff-dim",
                             "--transformer-dropout", "--encoder-architecture",
                             "--top-k-species", "--selection", "--representation",
                             "--normalization", "--use-cuda-hash", "--cv-folds",
                             "--cv-spatial"}) {
        INFO("train flag " << name);
        CHECK(train.find(name) != nullptr);
    }

    const CommandSpec& predict = resolve_cli::predict_spec();
    for (const char* name : {"--model", "--header", "--species", "--output",
                             "--covariate", "--categorical", "--predict-batch-size"}) {
        INFO("predict flag " << name);
        CHECK(predict.find(name) != nullptr);
    }

    CHECK(resolve_cli::info_spec().find("--model") != nullptr);

    // Every declared flag carries help text, since the help block is generated
    // from these rows.
    for (const CommandSpec* spec : {&train, &predict, &resolve_cli::info_spec()}) {
        for (const auto& flag : spec->flags()) {
            INFO(spec->name() << " " << flag.name);
            CHECK(std::string(flag.name).rfind("--", 0) == 0);
            CHECK(std::string(flag.help).size() > 0);
            if (flag.arity != Arity::Flag) {
                CHECK(std::string(flag.value_label).size() > 0);
            }
        }
    }
}
