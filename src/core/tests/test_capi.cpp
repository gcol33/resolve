// The flat C ABI (resolve_capi.h), called as a C client calls it: strict
// configuration maps at every level of nesting, device names, the value-tree
// builders' failure contract, default configs, the one dataset loader, the
// model sized for a dataset, retrieval, pretraining and the fuzzy index.

#include <catch2/catch_test_macros.hpp>

#include "resolve/resolve_capi.h"

#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve::test;

namespace {

// Owns a value tree for the length of a test.
struct Value {
    resolve_value_t* p;
    explicit Value(resolve_value_t* v) : p(v) {}
    ~Value() { resolve_value_free(p); }
    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;
    resolve_value_t* get() const { return p; }
};

using Dataset = std::unique_ptr<resolve_dataset_t, decltype(&resolve_dataset_free)>;
using Model = std::unique_ptr<resolve_model_t, decltype(&resolve_model_free)>;
using TrainerHandle = std::unique_ptr<resolve_trainer_t, decltype(&resolve_trainer_free)>;
using PredictorHandle = std::unique_ptr<resolve_predictor_t, decltype(&resolve_predictor_free)>;
using Fuzzy = std::unique_ptr<resolve_fuzzy_t, decltype(&resolve_fuzzy_free)>;

std::string last_error() { return resolve_last_error(); }

resolve_value_t* string_array(const std::vector<std::string>& values) {
    std::vector<const char*> pointers;
    for (const auto& v : values) pointers.push_back(v.c_str());
    return resolve_value_new_string_array(pointers.data(), static_cast<int64_t>(pointers.size()));
}

// A 30-plot hash corpus with a covariate, genus / family and a regression
// target "y"; the header and species tables on disk.
struct Corpus {
    TempFile header;
    TempFile species;

    Corpus() : header(make_header()), species(make_species()) {}

    static std::string make_header() {
        std::ostringstream out;
        out << "plot_id,elev,y\n";
        for (int i = 0; i < 30; ++i) out << "p" << i << "," << (100 + 7 * i) << "," << (0.5 * i) << "\n";
        return out.str();
    }
    static std::string make_species() {
        std::ostringstream out;
        out << "plot_id,sp,cover,genus,family\n";
        for (int i = 0; i < 30; ++i) {
            for (int j = 0; j < 4; ++j) {
                out << "p" << i << ",sp_" << ((i + j) % 6) << "," << (j + 1) << ",gen_"
                    << ((i + j) % 3) << ",fam_" << ((i + j) % 2) << "\n";
            }
        }
        return out.str();
    }

    // roles / targets / config maps for this corpus.
    static resolve_value_t* roles() {
        resolve_value_t* r = resolve_value_new_map();
        resolve_map_set_string(r, "plot_id", "plot_id");
        resolve_map_set_string(r, "species_id", "sp");
        resolve_map_set_string(r, "abundance", "cover");
        resolve_map_set_string(r, "genus", "genus");
        resolve_map_set_string(r, "family", "family");
        resolve_map_set_value(r, "covariates", string_array({"elev"}));
        return r;
    }
    static resolve_value_t* targets() {
        resolve_value_t* t = resolve_value_new_map();
        resolve_value_t* y = resolve_value_new_map();
        resolve_map_set_string(y, "task", "regression");
        resolve_map_set_value(t, "y", y);
        return t;
    }

    Dataset load(resolve_value_t* config = nullptr) const {
        const Value header_path(resolve_value_new_string(header.path().c_str()));
        const Value species_path(resolve_value_new_string(species.path().c_str()));
        const Value r(roles());
        const Value t(targets());
        const Value c(config != nullptr ? config : resolve_value_new_map());
        return Dataset(resolve_dataset_load(header_path.get(), species_path.get(), r.get(), t.get(),
                                            c.get(), nullptr, nullptr, nullptr),
                       &resolve_dataset_free);
    }
};

resolve_value_t* small_model_config() {
    resolve_value_t* c = resolve_value_new_map();
    const int64_t hidden[] = {16, 8};
    resolve_map_set_int_array(c, "hidden_dims", hidden, 2);
    return c;
}

}  // namespace

TEST_CASE("C ABI: a config key no field names is refused, at every level of nesting",
          "[capi]") {
    const Corpus corpus;
    const Dataset ds = corpus.load();
    REQUIRE(ds != nullptr);

    SECTION("top level") {
        const Value config(resolve_value_new_map());
        resolve_map_set_string(config.get(), "species_encodng", "embed");
        CHECK(resolve_model_create_for_dataset(ds.get(), config.get()) == nullptr);
        CHECK(last_error().find("unknown key 'species_encodng'") != std::string::npos);
        CHECK(last_error().find("species_encoding") != std::string::npos);
    }
    SECTION("a nested sub-config") {
        const Value config(resolve_value_new_map());
        resolve_value_t* ft = resolve_value_new_map();
        resolve_map_set_int(ft, "d_modle", 16);
        resolve_map_set_value(config.get(), "ft_transformer", ft);
        CHECK(resolve_model_create_for_dataset(ds.get(), config.get()) == nullptr);
        CHECK(last_error().find("model config.ft_transformer") != std::string::npos);
        CHECK(last_error().find("'d_modle'") != std::string::npos);
    }
    SECTION("a parallel branch inside a list") {
        const Value config(resolve_value_new_map());
        resolve_value_t* parallel = resolve_value_new_map();
        resolve_value_t* branches = resolve_value_new_list();
        resolve_value_t* branch = resolve_value_new_map();
        resolve_map_set_double(branch, "dropuot", 0.1);
        resolve_list_append_value(branches, branch);
        resolve_map_set_value(parallel, "branches", branches);
        resolve_map_set_value(config.get(), "parallel_layers", parallel);
        CHECK(resolve_model_create_for_dataset(ds.get(), config.get()) == nullptr);
        CHECK(last_error().find("parallel_layers.branches[0]") != std::string::npos);
    }
    SECTION("a dataset config and a train config") {
        resolve_value_t* config = resolve_value_new_map();
        resolve_map_set_int(config, "hash_dimm", 8);
        CHECK(corpus.load(config) == nullptr);
        CHECK(last_error().find("dataset config") != std::string::npos);

        const Model model(resolve_model_create_for_dataset(ds.get(), nullptr), &resolve_model_free);
        REQUIRE(model != nullptr);
        const Value train(resolve_value_new_map());
        resolve_map_set_int(train.get(), "max_epoch", 3);
        CHECK(resolve_trainer_create(model.get(), train.get()) == nullptr);
        CHECK(last_error().find("'max_epoch'") != std::string::npos);
    }
    SECTION("a key given twice") {
        const Value config(resolve_value_new_map());
        resolve_map_set_double(config.get(), "dropout", 0.1);
        resolve_map_set_double(config.get(), "dropout", 0.2);
        CHECK(resolve_model_create_for_dataset(ds.get(), config.get()) == nullptr);
        CHECK(last_error().find("'dropout' is given twice") != std::string::npos);
    }
    SECTION("a value of the wrong kind") {
        const Value config(resolve_value_new_map());
        resolve_map_set_string(config.get(), "dropout", "0.2");
        CHECK(resolve_model_create_for_dataset(ds.get(), config.get()) == nullptr);
        CHECK(last_error().find("model config.dropout") != std::string::npos);
        CHECK(last_error().find("number") != std::string::npos);
    }
}

TEST_CASE("C ABI: devices are parsed, not guessed", "[capi]") {
    const Corpus corpus;
    const Dataset ds = corpus.load();
    const Model model(resolve_model_create_for_dataset(ds.get(), nullptr), &resolve_model_free);
    REQUIRE(model != nullptr);
    CHECK(resolve_model_to_device(model.get(), "cpu") == 0);
    for (const char* bad : {"gpu", "CUDA", "cuda:x", ""}) {
        INFO(bad);
        CHECK(resolve_model_to_device(model.get(), bad) == -1);
        CHECK(last_error().find("cuda:N") != std::string::npos);
    }
    CHECK(resolve_model_to_device(model.get(), nullptr) == -1);

    const Value train(resolve_value_new_map());
    resolve_map_set_string(train.get(), "device", "gpu");
    CHECK(resolve_trainer_create(model.get(), train.get()) == nullptr);
}

TEST_CASE("C ABI: value builders report failures instead of terminating", "[capi]") {
    const Value list(resolve_value_new_list());
    CHECK(resolve_map_set_int(nullptr, "k", 1) == -1);
    CHECK(resolve_map_set_int(list.get(), "k", 1) == -1);
    const Value map(resolve_value_new_map());
    CHECK(resolve_map_set_int(map.get(), nullptr, 1) == -1);
    CHECK(resolve_map_set_double_array(map.get(), "x", nullptr, 3) == -1);
    CHECK(resolve_map_set_double_array(map.get(), "empty", nullptr, 0) == 0);
    CHECK(resolve_map_set_int_matrix(map.get(), "m", nullptr, -1, 2) == -1);
    // A child handed to a failing setter is freed, not leaked or left to the
    // caller.
    CHECK(resolve_map_set_value(nullptr, "k", resolve_value_new_int(1)) == -1);
    CHECK(resolve_list_append_value(map.get(), resolve_value_new_int(1)) == -1);
    CHECK(resolve_map_set_value(map.get(), "null_child", nullptr) == -1);
    CHECK(resolve_value_new_double_array(nullptr, 2) == nullptr);
    CHECK(resolve_map_size(map.get()) == 1);

    double out = 0.0;
    const double values[] = {1.0, 2.0};
    CHECK(resolve_metric_mae(values, values, 2, nullptr) == -1);
    CHECK(resolve_metric_mae(nullptr, values, 2, &out) == -1);
    CHECK(resolve_metric_mae(values, values, 2, &out) == 0);
    CHECK(out == 0.0);
}

TEST_CASE("C ABI: default configs are the engine's struct defaults", "[capi]") {
    const Value model(resolve_default_config("model"));
    REQUIRE(model.get() != nullptr);
    CHECK(resolve_value_as_int(resolve_map_get(model.get(), "hash_dim")) == resolve::ModelConfig{}.hash_dim);
    CHECK(std::string(resolve_value_as_string(resolve_map_get(model.get(), "species_encoding"))) == "hash");
    const Value train(resolve_default_config("train"));
    CHECK(resolve_value_as_int(resolve_map_get(train.get(), "max_epochs")) == resolve::TrainConfig{}.max_epochs);
    CHECK(std::string(resolve_value_as_string(resolve_map_get(train.get(), "device"))) == "cpu");
    for (const char* name : {"dataset", "spatial_block", "retrieval", "pretrain", "masked_species", "vae"}) {
        INFO(name);
        const Value defaults(resolve_default_config(name));
        CHECK(resolve_value_kind(defaults.get()) == RESOLVE_VALUE_MAP);
    }
    CHECK(resolve_default_config("modle") == nullptr);
    CHECK(last_error().find("model") != std::string::npos);
}

TEST_CASE("C ABI: one loader reads paths and in-memory tables", "[capi]") {
    const Corpus corpus;
    const Dataset from_csv = corpus.load();
    REQUIRE(from_csv != nullptr);

    // The species table in memory, the header from its CSV.
    std::vector<std::string> plot, sp, cover, genus, family;
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 4; ++j) {
            plot.push_back("p" + std::to_string(i));
            sp.push_back("sp_" + std::to_string((i + j) % 6));
            cover.push_back(std::to_string(j + 1));
            genus.push_back("gen_" + std::to_string((i + j) % 3));
            family.push_back("fam_" + std::to_string((i + j) % 2));
        }
    }
    const Value species(resolve_value_new_map());
    resolve_map_set_value(species.get(), "plot_id", string_array(plot));
    resolve_map_set_value(species.get(), "sp", string_array(sp));
    resolve_map_set_value(species.get(), "cover", string_array(cover));
    resolve_map_set_value(species.get(), "genus", string_array(genus));
    resolve_map_set_value(species.get(), "family", string_array(family));
    const Value header(resolve_value_new_string(corpus.header.path().c_str()));
    const Value roles(Corpus::roles());
    const Value targets(Corpus::targets());
    const Dataset mixed(resolve_dataset_load(header.get(), species.get(), roles.get(), targets.get(),
                                             nullptr, nullptr, nullptr, nullptr),
                        &resolve_dataset_free);
    REQUIRE(mixed != nullptr);
    const Value csv_hash(resolve_dataset_get(from_csv.get(), "hash_embedding"));
    const Value mixed_hash(resolve_dataset_get(mixed.get(), "hash_embedding"));
    int64_t nr = 0, nc = 0, nr2 = 0, nc2 = 0;
    const double* a = resolve_value_as_double_matrix(csv_hash.get(), &nr, &nc);
    const double* b = resolve_value_as_double_matrix(mixed_hash.get(), &nr2, &nc2);
    REQUIRE(nr == 30);
    REQUIRE((nr == nr2 && nc == nc2));
    CHECK(std::equal(a, a + nr * nc, b));

    // Its own vocabularies, by dataset or by tree; never both.
    const Value vocabs(resolve_dataset_get(from_csv.get(), "vocabs"));
    const Dataset by_vocabs(resolve_dataset_load(header.get(), species.get(), roles.get(), targets.get(),
                                                 nullptr, nullptr, vocabs.get(), nullptr),
                            &resolve_dataset_free);
    CHECK(by_vocabs != nullptr);

    // With names, an entered name reads as its token: sp_0 written otherwise
    // hashes as sp_0 again, and the report counts its records.
    {
        std::vector<std::string> written = sp;
        for (auto& name : written) {
            if (name == "sp_0") name = "Sp zero entered";
        }
        const Value renamed(resolve_value_new_map());
        resolve_map_set_value(renamed.get(), "plot_id", string_array(plot));
        resolve_map_set_value(renamed.get(), "sp", string_array(written));
        resolve_map_set_value(renamed.get(), "cover", string_array(cover));
        resolve_map_set_value(renamed.get(), "genus", string_array(genus));
        resolve_map_set_value(renamed.get(), "family", string_array(family));
        const Value names(resolve_value_new_map());
        resolve_value_t* table = resolve_value_new_map();
        resolve_map_set_value(table, "name", string_array({"Sp zero entered"}));
        resolve_map_set_value(table, "token", string_array({"sp_0"}));
        resolve_map_set_value(table, "source", string_array({"eva"}));
        resolve_map_set_value(names.get(), "table", table);
        const Dataset resolved(resolve_dataset_load(header.get(), renamed.get(), roles.get(),
                                                    targets.get(), nullptr, nullptr, vocabs.get(),
                                                    names.get()),
                               &resolve_dataset_free);
        REQUIRE(resolved != nullptr);
        const Value resolved_hash(resolve_dataset_get(resolved.get(), "hash_embedding"));
        const double* r = resolve_value_as_double_matrix(resolved_hash.get(), &nr2, &nc2);
        CHECK(std::equal(a, a + nr * nc, r));
        const Value report(resolve_dataset_get(resolved.get(), "species_names"));
        REQUIRE(resolve_value_string_array_size(resolve_map_get(report.get(), "name")) == 1);
        CHECK(std::string(resolve_value_string_at(resolve_map_get(report.get(), "token"), 0)) == "sp_0");
        CHECK(std::string(resolve_value_string_at(resolve_map_get(report.get(), "route"), 0)) == "alias");
        int64_t n = 0;
        CHECK(resolve_value_as_int_array(resolve_map_get(report.get(), "n_records"), &n)[0] == 20);

        // Names need a vocabulary to resolve against.
        CHECK(resolve_dataset_load(header.get(), renamed.get(), roles.get(), targets.get(), nullptr,
                                   nullptr, nullptr, names.get()) == nullptr);
        CHECK(last_error().find("vocab") != std::string::npos);
        const Value bad(resolve_value_new_map());
        resolve_map_set_int(bad.get(), "edits", 1);
        CHECK(resolve_dataset_load(header.get(), renamed.get(), roles.get(), targets.get(), nullptr,
                                   nullptr, vocabs.get(), bad.get()) == nullptr);
    }
    CHECK(resolve_dataset_load(header.get(), species.get(), roles.get(), targets.get(), nullptr,
                               from_csv.get(), vocabs.get(), nullptr) == nullptr);
    CHECK(last_error().find("not both") != std::string::npos);
    CHECK(resolve_dataset_load(header.get(), nullptr, roles.get(), targets.get(), nullptr, nullptr,
                               nullptr, nullptr) == nullptr);

    SECTION("a target named twice is refused") {
        const Value twice(resolve_value_new_map());
        resolve_value_t* first = resolve_value_new_map();
        resolve_map_set_string(first, "column", "y");
        resolve_value_t* second = resolve_value_new_map();
        resolve_map_set_string(second, "column", "elev");
        resolve_map_set_value(twice.get(), "y", first);
        resolve_map_set_value(twice.get(), "y", second);
        CHECK(resolve_dataset_load(header.get(), species.get(), roles.get(), twice.get(), nullptr,
                                   nullptr, nullptr, nullptr) == nullptr);
        CHECK(last_error().find("given twice") != std::string::npos);
    }
}

TEST_CASE("C ABI: a model created for a dataset takes its sizing fields from it", "[capi]") {
    const Corpus corpus;
    resolve_value_t* dataset_config = resolve_value_new_map();
    resolve_map_set_int(dataset_config, "top_k", 2);
    resolve_map_set_int(dataset_config, "hash_dim", 12);
    resolve_map_set_string(dataset_config, "selection", "top_bottom");
    const Dataset ds = corpus.load(dataset_config);
    REQUIRE(ds != nullptr);

    const Value model_config(small_model_config());
    resolve_map_set_int(model_config.get(), "hash_dim", 99);
    const Model model(resolve_model_create_for_dataset(ds.get(), model_config.get()), &resolve_model_free);
    REQUIRE(model != nullptr);
    const Value config(resolve_model_get(model.get(), "config"));
    CHECK(resolve_value_as_int(resolve_map_get(config.get(), "hash_dim")) == 12);
    CHECK(resolve_value_as_int(resolve_map_get(config.get(), "top_k")) == 2);
    CHECK(resolve_value_as_int(resolve_map_get(config.get(), "n_taxonomy_slots")) == 4);
}

TEST_CASE("C ABI: retrieval, pretraining and the VAE run through the boundary", "[capi]") {
    const Corpus corpus;
    const Dataset ds = corpus.load();
    REQUIRE(ds != nullptr);
    const Value model_config(small_model_config());
    const Model model(resolve_model_create_for_dataset(ds.get(), model_config.get()), &resolve_model_free);
    REQUIRE(model != nullptr);

    SECTION("pretraining") {
        const Value config(resolve_value_new_map());
        resolve_map_set_int(config.get(), "pretrain_epochs", 2);
        resolve_map_set_int(config.get(), "batch_size", 8);
        const Value result(resolve_model_pretrain(model.get(), "scarf", ds.get(), config.get()));
        REQUIRE(result.get() != nullptr);
        int64_t n = 0;
        resolve_value_as_double_array(resolve_map_get(result.get(), "loss_history"), &n);
        CHECK(n == 2);
        CHECK(resolve_model_pretrain(model.get(), "bert", ds.get(), nullptr) == nullptr);
        CHECK(last_error().find("jepa") != std::string::npos);
        resolve_map_set_string(config.get(), "mask_strategy", "blocks");
        CHECK(resolve_model_pretrain(model.get(), "jepa", ds.get(), config.get()) == nullptr);
    }
    SECTION("retrieval") {
        const Value train(resolve_value_new_map());
        resolve_map_set_int(train.get(), "max_epochs", 1);
        resolve_map_set_int(train.get(), "batch_size", 8);
        const TrainerHandle trainer(resolve_trainer_create(model.get(), train.get()),
                                    &resolve_trainer_free);
        REQUIRE(trainer != nullptr);
        REQUIRE(resolve_trainer_prepare_data_from_dataset(trainer.get(), ds.get(), 0.2, 1) == 0);
        const TempPath checkpoint(".pt");
        REQUIRE(resolve_trainer_save(trainer.get(), checkpoint.path().c_str(), nullptr) == 0);
        const PredictorHandle predictor(resolve_predictor_load(checkpoint.path().c_str(), "cpu", 1.0),
                                        &resolve_predictor_free);
        REQUIRE(predictor != nullptr);

        const Value none(resolve_predictor_get(predictor.get(), "retrieval"));
        CHECK(resolve_value_kind(none.get()) == RESOLVE_VALUE_NULL);
        const Value config(resolve_value_new_map());
        resolve_map_set_int(config.get(), "k_neighbors", 1);
        resolve_map_set_double(config.get(), "alpha", 1.0);
        REQUIRE(resolve_predictor_set_retrieval(predictor.get(), ds.get(), config.get(), 7) == 0);
        const Value set(resolve_predictor_get(predictor.get(), "retrieval"));
        CHECK(resolve_value_as_int(resolve_map_get(set.get(), "n_reference")) == 30);
        CHECK(resolve_value_as_int(resolve_map_get(set.get(), "k_neighbors")) == 1);

        const Value preds(resolve_predictor_predict_dataset(predictor.get(), ds.get(), 0, 4096));
        INFO(last_error());
        REQUIRE(preds.get() != nullptr);
        int64_t n = 0;
        const double* y = resolve_value_as_double_array(
            resolve_map_get(resolve_map_get(preds.get(), "predictions"), "y"), &n);
        REQUIRE(n == 30);
        for (int64_t i = 0; i < n; ++i) CHECK(std::abs(y[i] - 0.5 * static_cast<double>(i)) < 1e-4);

        CHECK(resolve_predictor_clear_retrieval(predictor.get()) == 0);
        resolve_map_set_int(config.get(), "neighbours", 3);
        CHECK(resolve_predictor_set_retrieval(predictor.get(), ds.get(), config.get(), 7) == -1);
    }
    SECTION("VAE pretraining needs the sparse species vector") {
        CHECK(resolve_vae_pretrain(ds.get(), nullptr) == nullptr);
        resolve_value_t* sparse = resolve_value_new_map();
        resolve_map_set_string(sparse, "species_encoding", "sparse");
        const Dataset sparse_ds = corpus.load(sparse);
        REQUIRE(sparse_ds != nullptr);
        const Value config(resolve_value_new_map());
        resolve_map_set_int(config.get(), "pretrain_epochs", 1);
        resolve_map_set_int(config.get(), "latent_dim", 4);
        const int64_t dims[] = {8};
        resolve_map_set_int_array(config.get(), "encoder_dims", dims, 1);
        resolve_map_set_int_array(config.get(), "decoder_dims", dims, 1);
        const Value result(resolve_vae_pretrain(sparse_ds.get(), config.get()));
        REQUIRE(result.get() != nullptr);
        CHECK(resolve_value_as_int(resolve_map_get(result.get(), "latent_dim")) == 4);
        int64_t rows = 0, cols = 0;
        resolve_value_as_double_matrix(resolve_map_get(result.get(), "projection_weights"), &rows, &cols);
        CHECK(rows == 8);
    }
}

TEST_CASE("C ABI: the fuzzy index with buckets and per-needle hints", "[capi]") {
    const Value entries(string_array({"Quercus robur", "Quercus petraea", "Fagus sylvatica"}));
    const Value options(resolve_value_new_map());
    resolve_map_set_value(options.get(), "buckets", string_array({"Quercus", "Quercus", "Fagus"}));
    const Fuzzy index(resolve_fuzzy_build(entries.get(), options.get()), &resolve_fuzzy_free);
    INFO(last_error());
    REQUIRE(index != nullptr);
    const Value size(resolve_fuzzy_get(index.get(), "size"));
    CHECK(resolve_value_as_int(size.get()) == 3);
    const Value buckets(resolve_fuzzy_get(index.get(), "bucket_count"));
    CHECK(resolve_value_as_int(buckets.get()) >= 2);

    const Value needles(string_array({"Quercus robr", "Fagus silvatica"}));
    const Value query_options(resolve_value_new_map());
    resolve_map_set_int(query_options.get(), "top_n", 1);
    resolve_map_set_value(query_options.get(), "bucket_hints", string_array({"Quercus", "Fagus"}));
    const Value matches(resolve_fuzzy_query(index.get(), needles.get(), query_options.get()));
    REQUIRE(matches.get() != nullptr);
    int64_t n = 0;
    const int64_t* ids = resolve_value_as_int_array(resolve_map_get(matches.get(), "id"), &n);
    REQUIRE(n == 2);
    CHECK(ids[0] == 0);
    CHECK(ids[1] == 2);
    const int64_t* distance = resolve_value_as_int_array(resolve_map_get(matches.get(), "distance"), &n);
    CHECK(distance[0] == 1);

    const Value short_options(resolve_value_new_map());
    resolve_map_set_value(short_options.get(), "buckets", string_array({"a"}));
    CHECK(resolve_fuzzy_build(entries.get(), short_options.get()) == nullptr);
    const Value clash(string_array({"x", "x"}));
    const Value clash_options(resolve_value_new_map());
    resolve_map_set_value(clash_options.get(), "buckets", string_array({"a", "b"}));
    CHECK(resolve_fuzzy_build(clash.get(), clash_options.get()) == nullptr);
    CHECK(last_error().find("two different values") != std::string::npos);
}

TEST_CASE("C ABI: gap fitting fits a target from vectors and reports its choice", "[capi][gap_fitting]") {
    const int64_t n = 200;
    std::vector<double> raw(n), recorded(n), species(n * 4, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        raw[i] = 10.0 + 0.1 * i;
        recorded[i] = i % 5 == 0 ? std::numeric_limits<double>::quiet_NaN() : raw[i] + 3.0;
        species[i * 4 + i % 4] = 1.0;
    }
    const Value target(resolve_value_new_map());
    resolve_map_set_string(target.get(), "name", "y");
    resolve_map_set_string(target.get(), "scale", "linear");
    const Value raw_v(resolve_value_new_double_array(raw.data(), n));
    const Value recorded_v(resolve_value_new_double_array(recorded.data(), n));
    const Value species_v(resolve_value_new_double_matrix(species.data(), n, 4));
    const Value fit(resolve_fit_gaps(target.get(), raw_v.get(), recorded_v.get(), species_v.get(),
                                     nullptr, nullptr, nullptr, nullptr));
    REQUIRE(fit.get() != nullptr);
    int64_t len = 0;
    const double* fitted = resolve_value_as_double_array(resolve_map_get(fit.get(), "fitted"), &len);
    REQUIRE(len == n);
    // A constant offset: the calibrated gaps land on the recorded relation.
    CHECK(std::abs(fitted[0] - (raw[0] + 3.0)) < 1e-6);
    CHECK(std::string(resolve_value_string_at(resolve_map_get(fit.get(), "method"), 1)) == "recorded");
    const resolve_value_t* groups = resolve_map_get(resolve_map_get(fit.get(), "report"), "groups");
    REQUIRE(resolve_map_get(groups, "unlocated") != nullptr);

    const Value bad(resolve_value_new_map());
    resolve_map_set_string(bad.get(), "scale", "sideways");
    CHECK(resolve_fit_gaps(bad.get(), raw_v.get(), recorded_v.get(), species_v.get(), nullptr, nullptr,
                           nullptr, nullptr) == nullptr);
    CHECK(last_error().find("sideways") != std::string::npos);
    const Value defaults(resolve_default_config("gap_fitting"));
    CHECK(resolve_value_kind(resolve_map_get(defaults.get(), "candidates")) == RESOLVE_VALUE_STRING_ARRAY);
}

TEST_CASE("C ABI: names resolve against a vocabulary with their routes", "[capi][names]") {
    const Value vocabulary(string_array({"<UNK>", "Quercus robur", "Anemone altaica"}));
    const Value names(string_array({"Quercus robur", "Quercus robru", "Anemone nemorosa", "Poa annua"}));
    const Value options(resolve_value_new_map());
    resolve_value_t* table = resolve_value_new_map();
    resolve_map_set_value(table, "name", string_array({"Anemone nemorosa"}));
    resolve_map_set_value(table, "token", string_array({"Anemone altaica"}));
    resolve_map_set_value(table, "source", string_array({"eva"}));
    resolve_map_set_value(options.get(), "table", table);
    resolve_map_set_int(options.get(), "max_edit_distance", 1);

    const Value out(resolve_names_resolve(vocabulary.get(), names.get(), options.get()));
    REQUIRE(out.get() != nullptr);
    const resolve_value_t* route = resolve_map_get(out.get(), "route");
    const resolve_value_t* token = resolve_map_get(out.get(), "token");
    REQUIRE(resolve_value_string_array_size(route) == 4);
    CHECK(std::string(resolve_value_string_at(route, 0)) == "exact");
    CHECK(std::string(resolve_value_string_at(route, 1)) == "spelling");
    CHECK(std::string(resolve_value_string_at(token, 1)) == "Quercus robur");
    CHECK(std::string(resolve_value_string_at(route, 2)) == "alias");
    CHECK(std::string(resolve_value_string_at(token, 2)) == "Anemone altaica");
    CHECK(std::string(resolve_value_string_at(resolve_map_get(out.get(), "source"), 2)) == "eva");
    CHECK(std::string(resolve_value_string_at(route, 3)) == "unresolved");
    CHECK(resolve_map_get(out.get(), "n_records") == nullptr);

    // No options: no table, one edit.
    const Value plain(resolve_names_resolve(vocabulary.get(), names.get(), nullptr));
    CHECK(std::string(resolve_value_string_at(resolve_map_get(plain.get(), "route"), 2)) == "unresolved");

    // A table by path is the release file.
    const TempFile file(R"({"source": "t", "aliases": {"Anemone nemorosa": ["Anemone altaica", "eva"]}})",
                        ".json", WriteMode::Binary);
    const Value by_path(resolve_value_new_map());
    resolve_map_set_string(by_path.get(), "table", file.path().c_str());
    const Value read(resolve_names_resolve(vocabulary.get(), names.get(), by_path.get()));
    CHECK(std::string(resolve_value_string_at(resolve_map_get(read.get(), "route"), 2)) == "alias");

    resolve_map_set_int(options.get(), "max_edit_distance", 9);
    CHECK(resolve_names_resolve(vocabulary.get(), names.get(), options.get()) == nullptr);
    CHECK(last_error().find("max_edit_distance") != std::string::npos);
}

TEST_CASE("C ABI: terrain descriptors, a terrain correction and a coded field through the boundary",
          "[capi][terrain]") {
    TempDir dir;
    write_plane_terrain(dir, {{46, 9}, {45, 9}}, 30);
    const std::vector<double> lat = {46.31, 46.72};
    const std::vector<double> lon = {9.42, 9.6};
    const Value lat_v(resolve_value_new_double_array(lat.data(), 2));
    const Value lon_v(resolve_value_new_double_array(lon.data(), 2));
    const Value grids(resolve_value_new_map());
    resolve_map_set_string(grids.get(), "grids", dir.path().c_str());
    const Value d(resolve_terrain_descriptors(grids.get(), lat_v.get(), lon_v.get()));
    REQUIRE(d.get() != nullptr);
    int64_t len = 0;
    const double* elevation = resolve_value_as_double_array(resolve_map_get(d.get(), "elevation"), &len);
    REQUIRE(len == 2);
    CHECK(std::abs(elevation[0] - terrain_plane(46.31, 9.42)) < 1e-3);

    const Value codec(resolve_value_new_map());
    resolve_map_set_double(codec.get(), "max_error", 0.5);
    const std::string coded = dir.file("coded.bin");
    const Value metrics(resolve_encode_terrain_grids(dir.path().c_str(), codec.get(), coded.c_str(), nullptr));
    REQUIRE(metrics.get() != nullptr);
    CHECK(resolve_map_get(metrics.get(), "T46_9") != nullptr);
    const Value field(resolve_value_new_map());
    resolve_map_set_string(field.get(), "field", coded.c_str());
    const Value f(resolve_terrain_descriptors(field.get(), lat_v.get(), lon_v.get()));
    REQUIRE(f.get() != nullptr);
    const double* field_elevation = resolve_value_as_double_array(resolve_map_get(f.get(), "elevation"), &len);
    CHECK(std::abs(field_elevation[1] - elevation[1]) <= 0.5 + 1e-3);

    // One tile whole, from the grids and from the coded field, by name.
    const Value grid_tile(resolve_terrain_grid(grids.get(), "T46_9"));
    const Value field_tile(resolve_terrain_grid(field.get(), "T46_9"));
    REQUIRE(grid_tile.get() != nullptr);
    REQUIRE(field_tile.get() != nullptr);
    const resolve_value_t* geometry = resolve_map_get(grid_tile.get(), "tile");
    CHECK(std::string(resolve_value_as_string(resolve_map_get(geometry, "name"))) == "T46_9");
    CHECK(resolve_value_as_double(resolve_map_get(geometry, "rows")) == 30.0);
    int64_t cells = 0;
    int64_t field_cells = 0;
    const double* stored = resolve_value_as_double_array(resolve_map_get(grid_tile.get(), "elevation"), &cells);
    const double* decoded = resolve_value_as_double_array(resolve_map_get(field_tile.get(), "elevation"), &field_cells);
    REQUIRE(cells == 30 * 30);
    REQUIRE(field_cells == cells);
    const double lat_first = resolve_value_as_double(resolve_map_get(geometry, "lat_first"));
    const double dlat = resolve_value_as_double(resolve_map_get(geometry, "dlat"));
    const double lon_first = resolve_value_as_double(resolve_map_get(geometry, "lon_first"));
    const double dlon = resolve_value_as_double(resolve_map_get(geometry, "dlon"));
    // Row-major from north to south: cell (2, 5).
    CHECK(std::abs(stored[2 * 30 + 5] - terrain_plane(lat_first - 2 * dlat, lon_first + 5 * dlon)) < 1e-2);
    double largest = 0.0;
    for (int64_t i = 0; i < cells; ++i) largest = std::max(largest, std::abs(decoded[i] - stored[i]));
    CHECK(largest <= 0.5 + 1e-3);
    CHECK(resolve_terrain_grid(grids.get(), "T10_10") == nullptr);
    CHECK(last_error().find("T10_10") != std::string::npos);

    const int64_t n = 300;
    std::vector<double> truth(n), prediction(n), dem(n);
    for (int64_t i = 0; i < n; ++i) {
        truth[i] = 100.0 + 3.0 * static_cast<double>(i);
        prediction[i] = truth[i] + (i % 2 ? 40.0 : -40.0);
        dem[i] = truth[i] + (i % 7 == 0 ? 300.0 : 0.5);
    }
    const Value inputs(resolve_value_new_map());
    resolve_map_set_double_array(inputs.get(), "prediction", prediction.data(), n);
    resolve_map_set_double_array(inputs.get(), "dem", dem.data(), n);
    const Value truth_v(resolve_value_new_double_array(truth.data(), n));
    const Value config(resolve_value_new_map());
    resolve_map_set_int(config.get(), "epochs", 2);
    resolve_map_set_int(config.get(), "batch_size", 64);
    const std::string path = dir.file("altitude.terrain");
    const Value report(resolve_fit_terrain_correction("altitude", "elevation", inputs.get(), truth_v.get(),
                                                      config.get(), path.c_str()));
    REQUIRE(report.get() != nullptr);
    const Value applied(resolve_terrain_correction_apply(path.c_str(), inputs.get(), nullptr));
    REQUIRE(applied.get() != nullptr);
    CHECK(std::string(resolve_value_as_string(resolve_map_get(applied.get(), "descriptor"))) == "elevation");
    resolve_value_as_double_array(resolve_map_get(applied.get(), "corrected"), &len);
    CHECK(len == n);

    CHECK(resolve_fit_terrain_correction("altitude", "aspect", inputs.get(), truth_v.get(), nullptr, path.c_str()) ==
          nullptr);
    CHECK(last_error().find("aspect") != std::string::npos);
    const Value bad_key(resolve_value_new_map());
    resolve_map_set_int(bad_key.get(), "epoch", 2);
    CHECK(resolve_fit_terrain_correction("altitude", "elevation", inputs.get(), truth_v.get(), bad_key.get(),
                                         path.c_str()) == nullptr);
    const Value defaults(resolve_default_config("terrain_codec"));
    CHECK(resolve_value_kind(resolve_map_get(defaults.get(), "max_error")) == RESOLVE_VALUE_DOUBLE);
}
