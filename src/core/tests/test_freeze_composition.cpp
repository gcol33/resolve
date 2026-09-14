// ModelConfig::freeze_composition keeps the species, genus and family tables at
// their initialisation.
//
// Contract:
//   * every species encoding with such a table exposes it through
//     ResolveModel::composition_parameters(), as the parameters themselves;
//   * with the knob on, a fit under non-zero weight decay leaves each table
//     bit-identical while the rest of the network trains;
//   * with it off, the same fit moves the tables;
//   * the knob survives a checkpoint, so a reloaded model is frozen too;
//   * a model with no composition table (an adapter architecture, or hash
//     without taxonomy) refuses the knob instead of ignoring it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/trainer.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_freeze_" + std::to_string(counter_++) + suffix);
        std::ofstream file(path_);
        file << content;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    [[nodiscard]] std::string path() const { return path_.string(); }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

constexpr int kSpecies = 6;
constexpr int kPlots = 40;

std::string header_csv() {
    std::ostringstream out;
    out << "plot_id,lon,lat,elev,y\n";
    for (int i = 0; i < kPlots; ++i) {
        out << "p" << i << "," << (10.0 + 0.1 * i) << "," << (47.0 + 0.1 * i)
            << "," << (100 + 5 * i) << "," << (1.0 + 0.25 * i) << "\n";
    }
    return out.str();
}

std::string species_csv() {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < kPlots; ++i) {
        for (int j = 0; j < kSpecies; ++j) {
            out << "p" << i << ",sp_" << j << "," << (((j + i) % kSpecies) + 1)
                << ",gen_" << (j % 3) << ",fam_" << (j % 2) << "\n";
        }
    }
    return out.str();
}

ResolveDataset build(const std::string& header_path, const std::string& species_path,
                     SpeciesEncodingMode mode, bool taxonomy = true) {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    if (taxonomy) {
        roles.genus = "genus";
        roles.family = "family";
    }
    roles.longitude = "lon";
    roles.latitude = "lat";
    roles.covariates = {"elev"};
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.use_taxonomy = taxonomy;
    return ResolveDataset::from_csv(header_path, species_path, roles,
                                    {TargetSpec::regression("y")}, cfg);
}

ModelConfig model_config(SpeciesEncodingMode mode, bool freeze) {
    ModelConfig cfg;
    cfg.species_encoding = mode;
    cfg.hidden_dims = {16, 12};
    cfg.species_embed_dim = 8;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hash_dim = 32;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.n_attention_layers = 1;
    cfg.transformer_ff_dim = 16;
    cfg.freeze_composition = freeze;
    return cfg;
}

TrainConfig train_config() {
    TrainConfig cfg;
    cfg.max_epochs = 3;
    cfg.batch_size = 16;
    cfg.lr = 1e-2f;
    cfg.weight_decay = 0.1f;
    cfg.device = torch::kCPU;
    return cfg;
}

std::vector<torch::Tensor> snapshot(const std::vector<torch::Tensor>& tensors) {
    std::vector<torch::Tensor> out;
    for (const auto& t : tensors) out.push_back(t.detach().clone());
    return out;
}

const SpeciesEncodingMode kEncodings[] = {
    SpeciesEncodingMode::Hash,
    SpeciesEncodingMode::Embed,
    SpeciesEncodingMode::Sparse,
    SpeciesEncodingMode::RankPool,
    SpeciesEncodingMode::Transformer,
};

const char* name_of(SpeciesEncodingMode mode) {
    switch (mode) {
        case SpeciesEncodingMode::Hash: return "hash";
        case SpeciesEncodingMode::Embed: return "embed";
        case SpeciesEncodingMode::Sparse: return "sparse";
        case SpeciesEncodingMode::RankPool: return "rank_pool";
        case SpeciesEncodingMode::Transformer: return "transformer";
    }
    return "?";
}

}  // namespace

TEST_CASE("Frozen composition tables keep their initialisation through a fit",
          "[freeze_composition]") {
    TempFile header(header_csv());
    TempFile species(species_csv());

    for (auto mode : kEncodings) {
        INFO("encoding " << name_of(mode));
        torch::manual_seed(5);
        auto ds = build(header.path(), species.path(), mode);
        ResolveModel model(ds.schema(), model_config(mode, /*freeze=*/true));

        const auto tables = model->composition_parameters();
        REQUIRE_FALSE(tables.empty());
        for (const auto& t : tables) REQUIRE_FALSE(t.requires_grad());

        const auto tables_before = snapshot(tables);
        std::vector<torch::Tensor> rest;
        for (const auto& p : model->parameters()) {
            if (p.requires_grad()) rest.push_back(p);
        }
        REQUIRE_FALSE(rest.empty());
        const auto rest_before = snapshot(rest);

        Trainer trainer(model, train_config());
        trainer.prepare_data(ds);
        trainer.fit();

        const auto tables_after = model->composition_parameters();
        REQUIRE(tables_after.size() == tables_before.size());
        for (size_t i = 0; i < tables_after.size(); ++i) {
            CHECK(torch::equal(tables_after[i], tables_before[i]));
        }
        bool any_moved = false;
        for (size_t i = 0; i < rest.size(); ++i) {
            any_moved = any_moved || !torch::equal(rest[i], rest_before[i]);
        }
        CHECK(any_moved);
    }
}

TEST_CASE("Unfrozen composition tables train", "[freeze_composition]") {
    TempFile header(header_csv());
    TempFile species(species_csv());

    for (auto mode : kEncodings) {
        INFO("encoding " << name_of(mode));
        torch::manual_seed(5);
        auto ds = build(header.path(), species.path(), mode);
        ResolveModel model(ds.schema(), model_config(mode, /*freeze=*/false));

        const auto before = snapshot(model->composition_parameters());
        REQUIRE_FALSE(before.empty());
        for (const auto& t : model->composition_parameters()) CHECK(t.requires_grad());

        Trainer trainer(model, train_config());
        trainer.prepare_data(ds);
        trainer.fit();

        const auto after = model->composition_parameters();
        bool any_moved = false;
        for (size_t i = 0; i < after.size(); ++i) {
            any_moved = any_moved || !torch::equal(after[i], before[i]);
        }
        CHECK(any_moved);
    }
}

TEST_CASE("A frozen model reloads frozen", "[freeze_composition][checkpoint]") {
    TempFile header(header_csv());
    TempFile species(species_csv());
    torch::manual_seed(9);

    auto ds = build(header.path(), species.path(), SpeciesEncodingMode::RankPool);
    ResolveModel model(ds.schema(),
                       model_config(SpeciesEncodingMode::RankPool, /*freeze=*/true));
    Trainer trainer(model, train_config());
    trainer.prepare_data(ds);
    trainer.fit();

    TempFile checkpoint("", ".pt");
    trainer.save(checkpoint.path());
    auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);

    CHECK(predictor.model()->config().freeze_composition);
    const auto reloaded = predictor.model()->composition_parameters();
    const auto trained = model->composition_parameters();
    REQUIRE(reloaded.size() == trained.size());
    for (size_t i = 0; i < reloaded.size(); ++i) {
        CHECK_FALSE(reloaded[i].requires_grad());
        CHECK(torch::equal(reloaded[i], trained[i]));
    }
}

TEST_CASE("A model with no composition table refuses freeze_composition",
          "[freeze_composition]") {
    TempFile header(header_csv());
    TempFile species(species_csv());

    SECTION("hash encoding without taxonomy") {
        auto ds = build(header.path(), species.path(), SpeciesEncodingMode::Hash,
                        /*taxonomy=*/false);
        CHECK(ResolveModel(ds.schema(), model_config(SpeciesEncodingMode::Hash, false))
                  ->composition_parameters().empty());
        CHECK_THROWS_WITH(
            ResolveModel(ds.schema(), model_config(SpeciesEncodingMode::Hash, true)),
            Catch::Matchers::ContainsSubstring("freeze_composition"));
    }

    SECTION("an adapter architecture") {
        auto ds = build(header.path(), species.path(), SpeciesEncodingMode::Embed);
        auto cfg = model_config(SpeciesEncodingMode::Embed, /*freeze=*/true);
        cfg.encoder_architecture = EncoderArchitecture::FTTransformer;
        CHECK_THROWS_WITH(ResolveModel(ds.schema(), cfg),
                          Catch::Matchers::ContainsSubstring("encoder_architecture"));
    }
}
