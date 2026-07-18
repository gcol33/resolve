// Tests for CategoricalVocab + CategoricalEmbedder + categorical-aware
// dataset/model/checkpoint integration.
//
// These tests exercise the categorical-covariate port end-to-end:
//   1. CategoricalVocab fit/encode roundtrip + NA handling + UNK code 0
//   2. CategoricalVocab save/load roundtrip through an InputArchive
//   3. CategoricalEmbedder forward shape + per-column embedding lookup
//   4. ResolveDataset.from_csv loads a categorical column, factorizes it,
//      and populates schema.categorical_names + categorical_vocab_sizes
//   5. ResolveModel constructed with categoricals on the schema concatenates
//      categorical embeddings into the continuous vector before the encoder
//   6. Checkpoint save/load roundtrip preserves the vocab so Predictor.load
//      can decode new CSVs with training-consistent codes

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/categorical.hpp"
#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/predictor.hpp"

#include <filesystem>
#include <fstream>

using namespace resolve;

namespace {

// Minimal TempFile helper (duplicated from test_dataset.cpp to keep tests
// independent — these test files are independently buildable).
class TempFile {
public:
    explicit TempFile(const std::string& content,
                      const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_cat_test_" + std::to_string(counter_++) + suffix);
        std::ofstream file(path_);
        file << content;
    }
    ~TempFile() { std::filesystem::remove(path_); }
    [[nodiscard]] std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

}  // namespace

// =============================================================================
// 1. CategoricalVocab — fit/encode/NA handling
// =============================================================================

TEST_CASE("CategoricalVocab factorizes sorted-unique non-NA values "
          "to codes 1..K", "[categorical][vocab]") {
    CategoricalVocab vocab;
    std::vector<std::string> raw = {"red", "blue", "green", "blue", "red"};
    vocab.fit_column("color", raw);

    SECTION("vocab_size includes UNK slot") {
        // K=3 unique non-NA values + 1 UNK
        REQUIRE(vocab.vocab_size("color") == 4);
    }

    SECTION("codes are sorted lexicographically starting at 1") {
        // sorted unique: blue=1, green=2, red=3 (UNK=0)
        REQUIRE(vocab.encode("color", "blue") == 1);
        REQUIRE(vocab.encode("color", "green") == 2);
        REQUIRE(vocab.encode("color", "red") == 3);
    }

    SECTION("unknown value returns 0 (UNK)") {
        REQUIRE(vocab.encode("color", "purple") == 0);
    }

    SECTION("unknown column returns 0 (UNK)") {
        REQUIRE(vocab.encode("nonexistent", "anything") == 0);
    }

    SECTION("column_names() preserves fit order") {
        vocab.fit_column("size", {"small", "large"});
        const auto& names = vocab.column_names();
        REQUIRE(names.size() == 2);
        REQUIRE(names[0] == "color");
        REQUIRE(names[1] == "size");
    }
}

TEST_CASE("CategoricalVocab treats NA-like strings as code 0",
          "[categorical][vocab]") {
    CategoricalVocab vocab;
    // Match Python's _NA_STRINGS exactly.
    std::vector<std::string> raw = {
        "x", "NA", "y", "", "NaN", "x", "NULL", ".", "n/a", "y", "-"
    };
    vocab.fit_column("c", raw);

    // Only "x" and "y" are non-NA; vocab has 2 + 1 UNK = 3.
    REQUIRE(vocab.vocab_size("c") == 3);
    REQUIRE(vocab.encode("c", "x") == 1);
    REQUIRE(vocab.encode("c", "y") == 2);

    SECTION("every NA-like string maps to 0") {
        for (const std::string& s : {"", "NA", "na", "N/A", "n/a", "NaN",
                                      "nan", "NULL", "null", "None",
                                      "none", ".", "-"}) {
            REQUIRE(vocab.encode("c", s) == 0);
        }
    }
}

TEST_CASE("CategoricalVocab::encode_batch produces the right shape and "
          "codes", "[categorical][vocab]") {
    CategoricalVocab vocab;
    vocab.fit_column("c1", {"a", "b", "a", "c"});
    vocab.fit_column("c2", {"x", "y", "x", ""});

    auto t = vocab.encode_batch(
        {"c1", "c2"},
        {{"a", "b", "a", "c"}, {"x", "y", "x", ""}});

    REQUIRE(t.dim() == 2);
    REQUIRE(t.size(0) == 4);
    REQUIRE(t.size(1) == 2);
    REQUIRE(t.scalar_type() == torch::kInt64);

    auto a = t.accessor<int64_t, 2>();
    // c1 codes: a=1, b=2, c=3
    REQUIRE(a[0][0] == 1);
    REQUIRE(a[1][0] == 2);
    REQUIRE(a[2][0] == 1);
    REQUIRE(a[3][0] == 3);
    // c2 codes: x=1, y=2, ""=0 (NA -> UNK)
    REQUIRE(a[0][1] == 1);
    REQUIRE(a[1][1] == 2);
    REQUIRE(a[2][1] == 1);
    REQUIRE(a[3][1] == 0);
}

// =============================================================================
// 2. CategoricalVocab save/load roundtrip
// =============================================================================

TEST_CASE("CategoricalVocab save/load roundtrip preserves all maps",
          "[categorical][vocab][serialize]") {
    CategoricalVocab original;
    original.fit_column("c1", {"a", "b", "a", "c"});
    original.fit_column("c2", {"x", "", "y"});

    auto tmp_path = std::filesystem::temp_directory_path() /
                    "cat_vocab_roundtrip.pt";

    {
        torch::serialize::OutputArchive out;
        original.save(out, "tcat_");
        out.save_to(tmp_path.string());
    }

    CategoricalVocab loaded;
    {
        torch::serialize::InputArchive in;
        in.load_from(tmp_path.string());
        loaded = CategoricalVocab::load(in, "tcat_");
    }

    REQUIRE(loaded.column_names() == original.column_names());
    REQUIRE(loaded.vocab_size("c1") == original.vocab_size("c1"));
    REQUIRE(loaded.vocab_size("c2") == original.vocab_size("c2"));

    // Code assignments survive the roundtrip
    REQUIRE(loaded.encode("c1", "a") == original.encode("c1", "a"));
    REQUIRE(loaded.encode("c1", "b") == original.encode("c1", "b"));
    REQUIRE(loaded.encode("c1", "c") == original.encode("c1", "c"));
    REQUIRE(loaded.encode("c2", "x") == original.encode("c2", "x"));
    REQUIRE(loaded.encode("c2", "y") == original.encode("c2", "y"));
    REQUIRE(loaded.encode("c2", "") == 0);  // NA always 0

    std::filesystem::remove(tmp_path);
}

TEST_CASE("CategoricalVocab::load returns empty vocab for archives without "
          "the categorical section (back-compat)",
          "[categorical][vocab][serialize][backcompat]") {
    auto tmp_path = std::filesystem::temp_directory_path() /
                    "cat_vocab_backcompat.pt";

    // Write an archive that has no categorical_ keys at all.
    {
        torch::serialize::OutputArchive out;
        out.write("unrelated_key", torch::tensor(42));
        out.save_to(tmp_path.string());
    }

    torch::serialize::InputArchive in;
    in.load_from(tmp_path.string());
    auto loaded = CategoricalVocab::load(in, "trainer_categorical_");

    REQUIRE(loaded.column_names().empty());
    REQUIRE(loaded.vocab_sizes().empty());

    std::filesystem::remove(tmp_path);
}

// =============================================================================
// 3. CategoricalEmbedder — forward + output shape
// =============================================================================

TEST_CASE("CategoricalEmbedder forward returns (B, N*D) and uses one "
          "embedding table per column",
          "[categorical][embedder]") {
    const std::vector<int64_t> vocab_sizes = {5, 3};  // two columns
    const int64_t embed_dim = 4;
    CategoricalEmbedder embedder(vocab_sizes, embed_dim);

    SECTION("output_dim matches contract") {
        REQUIRE(embedder->output_dim() == 2 * embed_dim);
        REQUIRE(embedder->n_columns() == 2);
        REQUIRE(embedder->embed_dim() == embed_dim);
    }

    SECTION("forward produces (B, N*D) for valid ids") {
        // 2-D nested-initializer-list literals don't bind to torch::tensor
        // overloads in MSVC; build the tensor element-wise instead.
        auto ids = torch::zeros({3, 2}, torch::kInt64);
        auto a = ids.accessor<int64_t, 2>();
        a[0][0] = 0; a[0][1] = 0;
        a[1][0] = 1; a[1][1] = 2;
        a[2][0] = 4; a[2][1] = 1;
        auto out = embedder->forward(ids);
        REQUIRE(out.dim() == 2);
        REQUIRE(out.size(0) == 3);
        REQUIRE(out.size(1) == 2 * embed_dim);
        REQUIRE(out.scalar_type() == torch::kFloat32);
    }

    SECTION("forward throws on wrong number of columns") {
        auto bad_ids = torch::zeros({1, 3}, torch::kInt64);
        REQUIRE_THROWS_AS(embedder->forward(bad_ids), std::invalid_argument);
    }

    SECTION("get_table_weights returns each table's matrix") {
        auto w0 = embedder->get_table_weights(0);
        REQUIRE(w0.dim() == 2);
        REQUIRE(w0.size(0) == 5);
        REQUIRE(w0.size(1) == embed_dim);
        auto w1 = embedder->get_table_weights(1);
        REQUIRE(w1.size(0) == 3);
        REQUIRE(w1.size(1) == embed_dim);
        REQUIRE_THROWS_AS(embedder->get_table_weights(2), std::out_of_range);
    }
}

TEST_CASE("CategoricalEmbedder with empty vocab returns (B, 0) tensor",
          "[categorical][embedder]") {
    // Explicit std::vector to avoid ambiguity with the ModuleHolder's
    // default constructor when passing `{}` as the first argument.
    CategoricalEmbedder embedder(std::vector<int64_t>{}, /*embed_dim=*/8);
    REQUIRE(embedder->output_dim() == 0);
    REQUIRE(embedder->n_columns() == 0);

    auto ids = torch::empty({5, 0}, torch::kInt64);
    auto out = embedder->forward(ids);
    REQUIRE(out.size(0) == 5);
    REQUIRE(out.size(1) == 0);
}

// =============================================================================
// 4. ResolveDataset CSV load with categoricals
// =============================================================================

TEST_CASE("ResolveDataset.from_csv loads a categorical column and "
          "populates schema correctly",
          "[categorical][dataset]") {
    // Header: 5 plots, 2 numeric covariates, 1 categorical (region) with
    // values {north, south, east, NA, north}. Plus 1 target.
    TempFile header_csv(
        "plot_id,lat,lon,cov1,region,target_y\n"
        "P0,40.0,-5.0,1.0,north,10.0\n"
        "P1,41.0,-4.0,2.0,south,12.0\n"
        "P2,42.0,-3.0,3.0,east,8.0\n"
        "P3,43.0,-2.0,4.0,NA,9.5\n"
        "P4,44.0,-1.0,5.0,north,11.0\n"
    );
    TempFile species_csv(
        "plot_id,sp,cover\n"
        "P0,a,0.5\nP0,b,0.3\n"
        "P1,a,0.7\n"
        "P2,c,0.9\n"
        "P3,a,0.4\nP3,d,0.6\n"
        "P4,b,1.0\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.latitude = "lat";
    roles.longitude = "lon";
    roles.covariates = {"cov1"};
    roles.categoricals = {"region"};

    auto ts = TargetSpec::regression("target_y");
    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 8;

    auto ds = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles, {ts}, dcfg);

    SECTION("schema reports the categorical column + vocab size") {
        REQUIRE(ds.schema().categorical_names.size() == 1);
        REQUIRE(ds.schema().categorical_names[0] == "region");
        REQUIRE(ds.schema().has_categoricals());
        // 3 unique non-NA values (north/south/east) + 1 UNK
        REQUIRE(ds.schema().categorical_vocab_sizes[0] == 4);
        REQUIRE(ds.schema().n_categoricals() == 1);
    }

    SECTION("categorical_ids has shape (n_plots, 1) and the NA row is 0") {
        const auto& ids = ds.categorical_ids();
        REQUIRE(ids.defined());
        REQUIRE(ids.dim() == 2);
        REQUIRE(ids.size(0) == 5);
        REQUIRE(ids.size(1) == 1);
        REQUIRE(ids.scalar_type() == torch::kInt64);
        auto a = ids.accessor<int64_t, 2>();
        // sorted-unique codes: east=1, north=2, south=3
        REQUIRE(a[0][0] == 2);  // north
        REQUIRE(a[1][0] == 3);  // south
        REQUIRE(a[2][0] == 1);  // east
        REQUIRE(a[3][0] == 0);  // NA -> UNK
        REQUIRE(a[4][0] == 2);  // north
    }

    SECTION("loader rejects a column listed as both covariate + categorical") {
        roles.covariates = {"cov1", "region"};
        roles.categoricals = {"region"};
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header_csv.path(), species_csv.path(), roles, {ts}, dcfg));
    }

    SECTION("loader rejects a categorical column not in the header CSV") {
        roles.categoricals = {"does_not_exist"};
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header_csv.path(), species_csv.path(), roles, {ts}, dcfg));
    }
}

// =============================================================================
// 5. ResolveModel — fuse_categoricals_ behavior end-to-end
// =============================================================================

TEST_CASE("ResolveModel with categoricals widens n_continuous and passes "
          "fused tensor to the encoder",
          "[categorical][model]") {
    // Build a schema by hand to avoid loading a CSV.
    ResolveSchema schema;
    schema.n_plots = 8;
    schema.n_species = 3;
    schema.has_coordinates = true;
    schema.covariate_names = {"cov_a", "cov_b"};
    schema.track_unknown_fraction = false;
    schema.track_unknown_count = false;
    schema.has_taxonomy = false;
    // Two categorical columns; vocab sizes include the UNK slot.
    schema.categorical_names = {"col1", "col2"};
    schema.categorical_vocab_sizes = {5, 4};
    schema.categorical_embed_dim = 6;

    TargetConfig tgt;
    tgt.name = "y";
    tgt.task = TaskType::Regression;
    schema.targets = {tgt};

    ModelConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.encoder_architecture = EncoderArchitecture::MLP;
    cfg.hash_dim = 4;
    cfg.hidden_dims = {8, 4};
    cfg.categorical_embed_dim = 6;

    ResolveModel model(schema, cfg);

    SECTION("schema embed_dim is synced from ModelConfig") {
        REQUIRE(model->schema().categorical_embed_dim == 6);
    }

    SECTION("forward accepts an explicit categorical_ids tensor") {
        const int64_t B = 4;
        const int64_t n_cont_input =
            /*coords*/ 2 + /*covariates*/ 2 + /*hash*/ 4;
        auto continuous = torch::randn({B, n_cont_input});
        // Build cat_ids element-wise (2-D nested-initializer literals
        // don't bind to torch::tensor overloads on MSVC).
        auto cat_ids = torch::zeros({B, 2}, torch::kInt64);
        auto a = cat_ids.accessor<int64_t, 2>();
        a[0][0] = 0; a[0][1] = 1;
        a[1][0] = 1; a[1][1] = 2;
        a[2][0] = 4; a[2][1] = 3;
        a[3][0] = 2; a[3][1] = 0;
        auto out = model->forward(continuous, {}, {}, {}, {},
                                  {}, {}, {}, {}, {}, cat_ids);
        REQUIRE(out.count("y") == 1);
        REQUIRE(out["y"].size(0) == B);
    }

    SECTION("forward without categorical_ids pads with zeros (back-compat)") {
        const int64_t B = 3;
        const int64_t n_cont_input = 2 + 2 + 4;
        auto continuous = torch::randn({B, n_cont_input});
        // No cat_ids supplied — model should fall back to zero padding.
        auto out = model->forward(continuous);
        REQUIRE(out.count("y") == 1);
        REQUIRE(out["y"].size(0) == B);
    }

    SECTION("forward rejects categorical_ids with the wrong column count") {
        const int64_t B = 4;
        const int64_t n_cont_input = 2 + 2 + 4;
        auto continuous = torch::randn({B, n_cont_input});
        auto bad = torch::zeros({B, 3}, torch::kInt64);  // expected 2
        REQUIRE_THROWS(model->forward(continuous, {}, {}, {}, {},
                                      {}, {}, {}, {}, {}, bad));
    }
}

TEST_CASE("ResolveModel adapter + categoricals sizes the numerical block correctly",
          "[categorical][adapter]") {
    // The model fuses the CategoricalEmbedder output into `continuous` before
    // the adapter runs. The adapter must size n_numerical_ to include that
    // width; otherwise TabNet/GNN crash on the shape mismatch and the
    // transformer adapters silently read only the leading columns.
    ResolveSchema schema;
    schema.n_plots = 8;
    schema.n_species = 10;
    schema.has_coordinates = true;
    schema.covariate_names = {"cov_a", "cov_b"};
    schema.track_unknown_fraction = false;
    schema.track_unknown_count = false;
    schema.has_taxonomy = false;
    schema.categorical_names = {"col1", "col2"};
    schema.categorical_vocab_sizes = {5, 4};
    schema.categorical_embed_dim = 6;

    TargetConfig tgt;
    tgt.name = "y";
    tgt.task = TaskType::Regression;
    schema.targets = {tgt};

    const int64_t B = 4;
    const int64_t hash_dim = 4;
    const int64_t n_cont_input = 2 /*coords*/ + 2 /*cov*/ + hash_dim;  // hash already in continuous

    auto continuous = torch::randn({B, n_cont_input});
    auto cat_ids = torch::zeros({B, 2}, torch::kInt64);
    {
        auto a = cat_ids.accessor<int64_t, 2>();
        a[0][0] = 0; a[0][1] = 1;
        a[1][0] = 1; a[1][1] = 2;
        a[2][0] = 4; a[2][1] = 3;
        a[3][0] = 2; a[3][1] = 0;
    }

    auto make_cfg = [&](EncoderArchitecture arch) {
        ModelConfig cfg;
        cfg.species_encoding = SpeciesEncodingMode::Hash;
        cfg.encoder_architecture = arch;
        cfg.hash_dim = hash_dim;
        cfg.hidden_dims = {16, 8};
        cfg.categorical_embed_dim = 6;
        return cfg;
    };

    SECTION("TabNet forward runs (previously crashed on the shape mismatch)") {
        auto cfg = make_cfg(EncoderArchitecture::TabNet);
        cfg.tabnet.n_d = 8;
        cfg.tabnet.n_a = 8;
        cfg.tabnet.n_steps = 2;
        ResolveModel model(schema, cfg);
        auto out = model->forward(continuous, {}, {}, {}, {},
                                  {}, {}, {}, {}, {}, cat_ids);
        REQUIRE(out.count("y") == 1);
        REQUIRE(out["y"].size(0) == B);
    }

    SECTION("ExcelFormer forward runs with all fused columns visible") {
        auto cfg = make_cfg(EncoderArchitecture::ExcelFormer);
        cfg.excelformer.d_model = 32;
        cfg.excelformer.n_heads = 4;
        cfg.excelformer.n_layers = 2;
        ResolveModel model(schema, cfg);
        auto out = model->forward(continuous, {}, {}, {}, {},
                                  {}, {}, {}, {}, {}, cat_ids);
        REQUIRE(out.count("y") == 1);
        REQUIRE(out["y"].size(0) == B);
    }
}

// =============================================================================
// 6. End-to-end checkpoint roundtrip with categoricals
// =============================================================================

TEST_CASE("Trainer save/load roundtrip preserves categorical vocab + "
          "embedder weights",
          "[categorical][checkpoint][trainer]") {
    // Mini synthetic dataset with a categorical column.
    TempFile header_csv(
        "plot_id,lat,lon,cov1,region,y\n"
        "P0,40.0,-5.0,1.0,north,10.0\n"
        "P1,41.0,-4.0,2.0,south,12.0\n"
        "P2,42.0,-3.0,3.0,east,8.0\n"
        "P3,43.0,-2.0,4.0,north,9.5\n"
        "P4,44.0,-1.0,5.0,south,11.0\n"
        "P5,45.0,0.0,6.0,east,7.5\n"
        "P6,46.0,1.0,7.0,north,13.0\n"
        "P7,47.0,2.0,8.0,south,9.0\n"
    );
    TempFile species_csv(
        "plot_id,sp,cover\n"
        "P0,a,0.5\nP1,b,0.7\nP2,c,0.9\nP3,a,0.4\n"
        "P4,b,1.0\nP5,c,0.6\nP6,a,0.3\nP7,b,0.8\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.latitude = "lat";
    roles.longitude = "lon";
    roles.covariates = {"cov1"};
    roles.categoricals = {"region"};

    DatasetConfig dcfg;
    dcfg.species_encoding = SpeciesEncodingMode::Hash;
    dcfg.hash_dim = 4;
    dcfg.use_taxonomy = false;
    dcfg.track_unknown_fraction = false;
    dcfg.track_unknown_count = false;

    auto ds = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles,
        {TargetSpec::regression("y")}, dcfg);

    ModelConfig mcfg;
    mcfg.species_encoding = SpeciesEncodingMode::Hash;
    mcfg.encoder_architecture = EncoderArchitecture::MLP;
    mcfg.hash_dim = 4;
    mcfg.hidden_dims = {8, 4};
    mcfg.categorical_embed_dim = 4;

    ResolveModel model(ds.schema(), mcfg);

    TrainConfig tcfg;
    tcfg.batch_size = 4;
    tcfg.max_epochs = 1;
    tcfg.patience = 1;
    tcfg.lr = 1e-3f;

    Trainer trainer(model, tcfg);
    trainer.prepare_data(ds, /*test_size=*/0.25f, /*seed=*/0);

    SECTION("Trainer captured the dataset's categorical vocab") {
        REQUIRE(trainer.categorical_vocab().column_names().size() == 1);
        REQUIRE(trainer.categorical_vocab().column_names()[0] == "region");
        // 3 unique values + UNK
        REQUIRE(trainer.categorical_vocab().vocab_size("region") == 4);
    }

    // Skip fit() — the test focuses on save/load mechanics, not training.
    // We instead invoke save() after prepare_data so the random-init
    // embedding weights + the schema + the vocab are all persisted.
    auto ckpt_path =
        (std::filesystem::temp_directory_path() / "cat_ckpt_test.pt").string();
    trainer.save(ckpt_path);

    SECTION("Predictor::load restores schema, vocab, and model") {
        auto predictor = Predictor::load(ckpt_path);
        const auto& loaded_schema = predictor.model()->schema();

        REQUIRE(loaded_schema.categorical_names.size() == 1);
        REQUIRE(loaded_schema.categorical_names[0] == "region");
        REQUIRE(loaded_schema.categorical_vocab_sizes[0] == 4);
        REQUIRE(loaded_schema.categorical_embed_dim == 4);

        const auto& loaded_vocab = predictor.categorical_vocab();
        REQUIRE(loaded_vocab.column_names().size() == 1);
        REQUIRE(loaded_vocab.encode("region", "north") ==
                trainer.categorical_vocab().encode("region", "north"));
        REQUIRE(loaded_vocab.encode("region", "south") ==
                trainer.categorical_vocab().encode("region", "south"));
        REQUIRE(loaded_vocab.encode("region", "east") ==
                trainer.categorical_vocab().encode("region", "east"));
        REQUIRE(loaded_vocab.encode("region", "missing-at-train") == 0);
    }

    std::filesystem::remove(ckpt_path);
}
