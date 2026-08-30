#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/csv_reader.hpp"
#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include <fstream>
#include <filesystem>

using namespace resolve;

// Helper to create temporary CSV files for testing
class TempFile {
public:
    TempFile(const std::string& content, const std::string& suffix = ".csv") {
        // Create temp file path
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_test_" + std::to_string(counter_++) + suffix);

        // Write content
        std::ofstream file(path_);
        file << content;
        file.close();
    }

    ~TempFile() {
        std::filesystem::remove(path_);
    }

    std::string path() const { return path_.string(); }

private:
    std::filesystem::path path_;
    static int counter_;
};

int TempFile::counter_ = 0;

// ============================================================================
// CSV Reader Tests
// ============================================================================

TEST_CASE("CSVReader basic parsing", "[csv]") {
    TempFile csv(
        "col1,col2,col3\n"
        "a,1,2.5\n"
        "b,2,3.5\n"
        "c,3,4.5\n"
    );

    CSVReader reader(csv.path());

    SECTION("columns are parsed correctly") {
        REQUIRE(reader.columns().size() == 3);
        REQUIRE(reader.columns()[0] == "col1");
        REQUIRE(reader.columns()[1] == "col2");
        REQUIRE(reader.columns()[2] == "col3");
    }

    SECTION("column index lookup works") {
        REQUIRE(reader.column_index("col1") == 0);
        REQUIRE(reader.column_index("col2") == 1);
        REQUIRE(reader.column_index("col3") == 2);
        REQUIRE(reader.column_index("nonexistent") == -1);
    }

    SECTION("row count is correct") {
        REQUIRE(reader.count_rows() == 3);
    }

    SECTION("read_all returns all rows") {
        auto rows = reader.read_all();
        REQUIRE(rows.size() == 3);
        REQUIRE(rows[0][0] == "a");
        REQUIRE(rows[1][1] == "2");
        REQUIRE(rows[2][2] == "4.5");
    }
}

TEST_CASE("CSVReader handles quoted fields", "[csv]") {
    TempFile csv(
        "name,description\n"
        "item1,\"contains, comma\"\n"
        "item2,\"contains \"\"quotes\"\"\"\n"
    );

    CSVReader reader(csv.path());
    auto rows = reader.read_all();

    REQUIRE(rows.size() == 2);
    REQUIRE(rows[0][1] == "contains, comma");
    REQUIRE(rows[1][1] == "contains \"quotes\"");
}

TEST_CASE("CSVReader handles empty fields", "[csv]") {
    TempFile csv(
        "a,b,c\n"
        "1,,3\n"
        ",2,\n"
    );

    CSVReader reader(csv.path());
    auto rows = reader.read_all();

    REQUIRE(rows.size() == 2);
    REQUIRE(rows[0][1] == "");
    REQUIRE(rows[1][0] == "");
    REQUIRE(rows[1][2] == "");
}

// ============================================================================
// RoleMapping Tests
// ============================================================================

TEST_CASE("RoleMapping helper methods", "[role_mapping]") {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";

    SECTION("without optional columns") {
        REQUIRE_FALSE(roles.has_coordinates());
        REQUIRE_FALSE(roles.has_taxonomy());
        REQUIRE_FALSE(roles.has_abundance());
    }

    SECTION("with coordinates") {
        roles.longitude = "lon";
        roles.latitude = "lat";
        REQUIRE(roles.has_coordinates());
    }

    SECTION("with taxonomy") {
        roles.genus = "genus";
        REQUIRE(roles.has_taxonomy());

        roles.family = "family";
        REQUIRE(roles.has_taxonomy());
    }

    SECTION("with abundance") {
        roles.abundance = "cover";
        REQUIRE(roles.has_abundance());
    }
}

TEST_CASE("TargetSpec convenience constructors", "[role_mapping]") {
    SECTION("regression target") {
        auto spec = TargetSpec::regression("area", TransformType::Log1p);
        REQUIRE(spec.column_name == "area");
        REQUIRE(spec.target_name == "area");
        REQUIRE(spec.task == TaskType::Regression);
        REQUIRE(spec.transform == TransformType::Log1p);
    }

    SECTION("classification target") {
        auto spec = TargetSpec::classification("habitat", 9);
        REQUIRE(spec.column_name == "habitat");
        REQUIRE(spec.target_name == "habitat");
        REQUIRE(spec.task == TaskType::Classification);
        REQUIRE(spec.num_classes == 9);
    }
}

// ============================================================================
// TaxonomyVocab Tests
// ============================================================================

TEST_CASE("TaxonomyVocab encoding", "[vocab]") {
    std::vector<SpeciesRecord> records = {
        {"sp1", "Quercus", "Fagaceae", 1.0f, "p1"},
        {"sp2", "Fagus", "Fagaceae", 1.0f, "p1"},
        {"sp3", "Pinus", "Pinaceae", 1.0f, "p2"},
        {"sp4", "Abies", "Pinaceae", 1.0f, "p2"},
    };

    TaxonomyVocab vocab;
    vocab.fit(records);

    SECTION("vocabulary sizes are correct") {
        // +1 for unknown
        REQUIRE(vocab.n_genera() == 5);   // <UNK>, Quercus, Fagus, Pinus, Abies
        REQUIRE(vocab.n_families() == 3); // <UNK>, Fagaceae, Pinaceae
    }

    SECTION("known taxa encode to positive IDs") {
        REQUIRE(vocab.encode_genus("Quercus") > 0);
        REQUIRE(vocab.encode_genus("Pinus") > 0);
        REQUIRE(vocab.encode_family("Fagaceae") > 0);
    }

    SECTION("unknown taxa encode to 0") {
        REQUIRE(vocab.encode_genus("Unknown") == 0);
        REQUIRE(vocab.encode_family("Unknown") == 0);
    }
}

// ============================================================================
// ResolveDataset Tests
// ============================================================================

TEST_CASE("ResolveDataset from_species_csv basic loading", "[dataset]") {
    // Create test data
    TempFile csv(
        "plot_id,species,cover,lon,lat,genus,family,area\n"
        "p1,sp1,0.5,10.0,50.0,Quercus,Fagaceae,100\n"
        "p1,sp2,0.3,10.0,50.0,Fagus,Fagaceae,100\n"
        "p2,sp1,0.8,11.0,51.0,Quercus,Fagaceae,200\n"
        "p2,sp3,0.2,11.0,51.0,Pinus,Pinaceae,200\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";
    roles.longitude = "lon";
    roles.latitude = "lat";
    roles.genus = "genus";
    roles.family = "family";

    std::vector<TargetSpec> targets = {
        TargetSpec::regression("area")
    };

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.top_k = 2;

    auto dataset = ResolveDataset::from_species_csv(
        csv.path(), roles, targets, config
    );

    SECTION("correct number of plots loaded") {
        REQUIRE(dataset.n_plots() == 2);
    }

    SECTION("plot IDs are correct") {
        auto& ids = dataset.plot_ids();
        REQUIRE(ids.size() == 2);
        REQUIRE((ids[0] == "p1" || ids[0] == "p2"));
    }

    SECTION("schema is populated") {
        auto& schema = dataset.schema();
        REQUIRE(schema.n_plots == 2);
        REQUIRE(schema.has_coordinates == true);
        REQUIRE(schema.has_taxonomy == true);
        REQUIRE(schema.targets.size() == 1);
    }

    SECTION("coordinates are loaded") {
        auto& coords = dataset.coordinates();
        REQUIRE(coords.defined());
        REQUIRE(coords.size(0) == 2);
        REQUIRE(coords.size(1) == 2);
    }

    SECTION("hash embedding is generated") {
        auto& hash_emb = dataset.hash_embedding();
        REQUIRE(hash_emb.defined());
        REQUIRE(hash_emb.size(0) == 2);
        REQUIRE(hash_emb.size(1) == 16);
    }

    SECTION("taxonomy IDs are generated") {
        auto& genus_ids = dataset.genus_ids();
        auto& family_ids = dataset.family_ids();
        REQUIRE(genus_ids.defined());
        REQUIRE(family_ids.defined());
        REQUIRE(genus_ids.size(0) == 2);
        REQUIRE(genus_ids.size(1) == 2);  // top_k
    }

    SECTION("targets are loaded") {
        auto& tgt = dataset.targets();
        REQUIRE(tgt.count("area") > 0);
        REQUIRE(tgt.at("area").size(0) == 2);
    }
}

TEST_CASE("ResolveDataset from_species_csv factorizes string classification targets", "[dataset]") {
    // Regression guard for the bug where from_species_csv pushed every target
    // cell through safe_stof and cast to int64, collapsing string-coded classes
    // (e.g. "Forest"/"Grassland") to class 0 with no class_names and no NA drop.
    TempFile csv(
        "plot_id,species,cover,habitat\n"
        "p1,sp1,0.5,Forest\n"
        "p1,sp2,0.3,Forest\n"
        "p2,sp1,0.8,Grassland\n"
        "p3,sp3,0.2,Forest\n"
        "p4,sp1,0.1,NA\n"   // missing target -> plot dropped
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    std::vector<TargetSpec> targets = {
        TargetSpec::classification("habitat", /*num_classes=*/0)  // auto-fit
    };

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.top_k = 2;

    auto dataset = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);

    SECTION("NA-target plot dropped") {
        REQUIRE(dataset.n_plots() == 3);  // p4 dropped, not 4
    }

    SECTION("string classes factorized to distinct codes") {
        const auto& tgt = dataset.targets().at("habitat");
        REQUIRE(tgt.dtype() == torch::kLong);
        REQUIRE(tgt.size(0) == 3);
        // Two distinct classes -> codes {0, 1}, not all 0 (the bug). Two Forest
        // (0) + one Grassland (1) over the kept plots -> min 0, max 1, sum 1.
        REQUIRE(tgt.min().item<int64_t>() == 0);
        REQUIRE(tgt.max().item<int64_t>() == 1);
        REQUIRE(tgt.sum().item<int64_t>() == 1);
    }

    SECTION("class_names and num_classes populated on schema") {
        const auto& schema_targets = dataset.schema().targets;
        bool found = false;
        for (const auto& tc : schema_targets) {
            if (tc.name == "habitat") {
                found = true;
                REQUIRE(tc.num_classes == 2);
                REQUIRE(tc.class_names.size() == 2);
                // Auto-fit sorts unique non-NA values: Forest < Grassland.
                REQUIRE(tc.class_names[0] == "Forest");
                REQUIRE(tc.class_names[1] == "Grassland");
            }
        }
        REQUIRE(found);
    }
}

TEST_CASE("Classification num_classes smaller than emitted codes throws", "[dataset][classification]") {
    // Direct integer labels are used verbatim as class codes, so a 1-indexed
    // column 1..3 emits class_names of size 4 (index 0 reserved). Passing
    // num_classes=3 would let a target code 3 index a 3-wide head out of bounds
    // (issue #79). The loader must reject the too-small explicit value.
    TempFile csv(
        "plot_id,species,cover,habitat\n"
        "p1,sp1,0.5,1\n"
        "p2,sp2,0.5,2\n"
        "p3,sp3,0.5,3\n"
    );
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.top_k = 2;

    SECTION("too-small num_classes rejected") {
        std::vector<TargetSpec> targets = { TargetSpec::classification("habitat", 3) };
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("auto-size (num_classes=0) succeeds and sizes the head to max_code+1") {
        std::vector<TargetSpec> targets = { TargetSpec::classification("habitat", 0) };
        auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);
        for (const auto& tc : ds.schema().targets) {
            if (tc.name == "habitat") REQUIRE(tc.num_classes == 4);  // codes 1..3 -> size 4
        }
    }
    SECTION("num_classes >= max_code+1 is kept as-is") {
        std::vector<TargetSpec> targets = { TargetSpec::classification("habitat", 10) };
        auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);
        for (const auto& tc : ds.schema().targets) {
            if (tc.name == "habitat") REQUIRE(tc.num_classes == 10);
        }
    }
}

TEST_CASE("Classification vocab is fit from surviving rows only", "[dataset][classification]") {
    // A class that occurs only in a plot dropped because ANOTHER target (area) is
    // missing must not enter class_names -- it would be an untrainable zero-example
    // class and (lexicographically) shift every other code (issue #84).
    TempFile csv(
        "plot_id,species,cover,habitat,area\n"
        "p1,sp1,0.5,Forest,100\n"
        "p2,sp2,0.5,Grassland,200\n"
        "p3,sp3,0.5,Xerophyte,NA\n"  // area NA -> p3 dropped; Xerophyte lives only here
    );
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    std::vector<TargetSpec> targets = {
        TargetSpec::classification("habitat", 0),
        TargetSpec::regression("area")
    };
    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.top_k = 2;

    auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);

    REQUIRE(ds.n_plots() == 2);  // p3 dropped
    for (const auto& tc : ds.schema().targets) {
        if (tc.name == "habitat") {
            REQUIRE(tc.num_classes == 2);          // not 3
            REQUIRE(tc.class_names.size() == 2);
            for (const auto& n : tc.class_names) REQUIRE(n != "Xerophyte");
        }
    }
}

TEST_CASE("ResolveDataset embed mode", "[dataset]") {
    TempFile csv(
        "plot_id,species,cover\n"
        "p1,sp1,0.5\n"
        "p1,sp2,0.3\n"
        "p1,sp3,0.1\n"
        "p1,sp4,0.1\n"
        "p2,sp1,0.8\n"
        "p2,sp5,0.2\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    std::vector<TargetSpec> targets;  // No targets for this test

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Embed;
    config.top_k_species = 3;

    auto dataset = ResolveDataset::from_species_csv(
        csv.path(), roles, targets, config
    );

    SECTION("species IDs are generated") {
        auto& species_ids = dataset.species_ids();
        REQUIRE(species_ids.defined());
        REQUIRE(species_ids.size(0) == 2);
        REQUIRE(species_ids.size(1) == 3);  // top_k_species
    }

    SECTION("species vocabulary is built") {
        auto& vocab = dataset.species_vocab();
        REQUIRE(vocab.size() >= 5);  // At least 5 unique species + <UNK>
    }
}

TEST_CASE("ResolveDataset sparse mode", "[dataset]") {
    TempFile csv(
        "plot_id,species,cover\n"
        "p1,sp1,0.5\n"
        "p1,sp2,0.3\n"
        "p2,sp1,0.8\n"
        "p2,sp3,0.2\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    std::vector<TargetSpec> targets;

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;

    auto dataset = ResolveDataset::from_species_csv(
        csv.path(), roles, targets, config
    );

    SECTION("species vector is generated") {
        auto& species_vec = dataset.species_vector();
        REQUIRE(species_vec.defined());
        REQUIRE(species_vec.size(0) == 2);
        // Size(1) should be vocab size
    }
}

// =============================================================================
// ResolveDataset::from_csv_with_schema — cross-split vocab reuse
// =============================================================================
//
// Recovery test for the cross-split factory. Builds a train CSV pair and a
// test CSV pair where the test set shares some species/categorical/taxonomy/
// class values with train and introduces some new ones. Loads the train set
// with the regular factory, then loads the test set against the train's
// fitted vocabularies via from_csv_with_schema. Asserts:
//
//   - Shared species: test species_ids match the train IDs for the same name
//     (vocab namespace alignment — the whole point of the feature).
//   - New (test-only) species: encoded as 0 (UNK).
//   - Shared categorical values: same code as train.
//   - New categorical values: encoded as 0 (UNK).
//   - Test-set species with genus/family in the train vocab: nonzero
//     genus_id / family_id matching train.
//   - Test-set species with genus/family unseen in training: 0.
//   - Inherited classification class mapping: a test row with a shared label
//     gets the same class index; a test row with an unseen label is dropped
//     (existing missing-target row-drop path; documented as the contract).
//
// Uses Embed mode + top_k_species=1 so species_ids[i][0] is the (only)
// species encoded for plot i and we can compare it directly against the
// train mapping by name.

TEST_CASE("ResolveDataset::from_csv_with_schema reuses train vocab + class "
          "mapping on a held-out set",
          "[dataset][cross_split]") {
    // Train fixture: 4 plots, each with exactly one species so species_ids
    // is unambiguous. region in {north,south,east}. habitat in {forest,grass}.
    TempFile train_header(
        "plot_id,region,habitat\n"
        "T0,north,forest\n"
        "T1,south,grass\n"
        "T2,east,forest\n"
        "T3,north,grass\n"
    );
    TempFile train_species(
        "plot_id,species,cover,genus,family\n"
        "T0,sp_a,1.0,gen_x,fam_x\n"
        "T1,sp_b,1.0,gen_y,fam_y\n"
        "T2,sp_c,1.0,gen_x,fam_x\n"
        "T3,sp_train_only,1.0,gen_y,fam_y\n"
    );

    // Test fixture: 5 plots.
    //   E0: sp_a (shared)         region=north (shared)   habitat=forest (shared)
    //   E1: sp_b (shared)         region=west  (new -> 0) habitat=grass  (shared)
    //   E2: sp_test_only (new)    region=north (shared)   habitat=grass  (shared)
    //                             species genus/family also test-only.
    //   E3: sp_a (shared)         region=north            habitat=desert (unseen
    //                             -> row dropped by missing-target filter)
    //   E4: sp_c (shared, train genus/family) region=south (shared) habitat=forest
    TempFile test_header(
        "plot_id,region,habitat\n"
        "E0,north,forest\n"
        "E1,west,grass\n"
        "E2,north,grass\n"
        "E3,north,desert\n"
        "E4,south,forest\n"
    );
    TempFile test_species(
        "plot_id,species,cover,genus,family\n"
        "E0,sp_a,1.0,gen_x,fam_x\n"
        "E1,sp_b,1.0,gen_y,fam_y\n"
        "E2,sp_test_only,1.0,gen_new,fam_new\n"
        "E3,sp_a,1.0,gen_x,fam_x\n"
        "E4,sp_c,1.0,gen_x,fam_x\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    roles.categoricals = {"region"};

    std::vector<TargetSpec> targets = {
        TargetSpec::classification("habitat", /*num_classes=*/0)
    };

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Embed;
    config.top_k_species = 1;  // one species per plot -> species_ids[i][0] is it
    config.top_k = 1;
    config.use_taxonomy = true;
    config.track_unknown_fraction = false;
    config.track_unknown_count = false;

    auto train_ds = ResolveDataset::from_csv(
        train_header.path(), train_species.path(), roles, targets, config);

    // Train sanity: 4 plots loaded.
    REQUIRE(train_ds.n_plots() == 4);
    REQUIRE(train_ds.schema().categorical_names.size() == 1);
    REQUIRE(train_ds.schema().categorical_names[0] == "region");

    // Extract train IDs by name so we can later compare against test IDs.
    auto train_species_id = [&](const std::string& name) -> int64_t {
        const auto& v = train_ds.species_vocab();
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] == name) return static_cast<int64_t>(i);
        }
        return -1;  // sentinel: not found
    };
    const int64_t id_sp_a = train_species_id("sp_a");
    const int64_t id_sp_b = train_species_id("sp_b");
    const int64_t id_sp_c = train_species_id("sp_c");
    REQUIRE(id_sp_a > 0);
    REQUIRE(id_sp_b > 0);
    REQUIRE(id_sp_c > 0);

    const auto& train_cat_vocab = train_ds.categorical_vocab();
    const int64_t code_north = train_cat_vocab.encode("region", "north");
    const int64_t code_south = train_cat_vocab.encode("region", "south");
    const int64_t code_east  = train_cat_vocab.encode("region", "east");
    REQUIRE(code_north > 0);
    REQUIRE(code_south > 0);
    REQUIRE(code_east  > 0);

    const int64_t gen_x = train_ds.taxonomy_vocab().encode_genus("gen_x");
    const int64_t gen_y = train_ds.taxonomy_vocab().encode_genus("gen_y");
    const int64_t fam_x = train_ds.taxonomy_vocab().encode_family("fam_x");
    const int64_t fam_y = train_ds.taxonomy_vocab().encode_family("fam_y");
    REQUIRE(gen_x > 0);
    REQUIRE(gen_y > 0);
    REQUIRE(fam_x > 0);
    REQUIRE(fam_y > 0);

    // Load test set against the train schema.
    auto test_ds = ResolveDataset::from_csv_with_schema(
        test_header.path(), test_species.path(), roles, targets,
        train_ds, config);

    // E3 (habitat=desert) is the only row with a label unseen in train. The
    // explicit-mapping branch drops unmapped rows the same way it drops NA
    // targets, so the test set keeps 4 of 5 plots.
    SECTION("rows with unseen classification labels are dropped") {
        REQUIRE(test_ds.n_plots() == 4);
        const auto& pids = test_ds.plot_ids();
        for (const auto& pid : pids) {
            REQUIRE(pid != "E3");
        }
    }

    // Map plot_id -> row index for direct lookups (header CSV order is the
    // order of plot_ids_, but row-dropping may shift indices; build a map).
    auto pid_to_row = [&]() {
        std::unordered_map<std::string, int64_t> m;
        const auto& pids = test_ds.plot_ids();
        for (size_t i = 0; i < pids.size(); ++i) {
            m[pids[i]] = static_cast<int64_t>(i);
        }
        return m;
    }();

    SECTION("shared species encode to the same IDs as in train") {
        const auto& sp_ids = test_ds.species_ids();
        REQUIRE(sp_ids.defined());
        REQUIRE(sp_ids.size(0) == test_ds.n_plots());
        REQUIRE(sp_ids.size(1) == 1);
        auto acc = sp_ids.accessor<int64_t, 2>();

        REQUIRE(acc[pid_to_row.at("E0")][0] == id_sp_a);
        REQUIRE(acc[pid_to_row.at("E1")][0] == id_sp_b);
        REQUIRE(acc[pid_to_row.at("E4")][0] == id_sp_c);
    }

    SECTION("test-only species encode to UNK=0") {
        const auto& sp_ids = test_ds.species_ids();
        auto acc = sp_ids.accessor<int64_t, 2>();
        // E2 carries sp_test_only — never seen by the train fit.
        REQUIRE(acc[pid_to_row.at("E2")][0] == 0);
    }

    SECTION("shared categorical values match train codes; new -> UNK=0") {
        const auto& cat_ids = test_ds.categorical_ids();
        REQUIRE(cat_ids.defined());
        REQUIRE(cat_ids.size(0) == test_ds.n_plots());
        REQUIRE(cat_ids.size(1) == 1);
        auto acc = cat_ids.accessor<int64_t, 2>();

        // E0 (north), E2 (north), E4 (south) share with train.
        REQUIRE(acc[pid_to_row.at("E0")][0] == code_north);
        REQUIRE(acc[pid_to_row.at("E2")][0] == code_north);
        REQUIRE(acc[pid_to_row.at("E4")][0] == code_south);
        // E1 (west) is unseen in train -> UNK.
        REQUIRE(acc[pid_to_row.at("E1")][0] == 0);

        // Vocab itself is unchanged on the test dataset (still the train fit).
        REQUIRE(test_ds.categorical_vocab().vocab_size("region") ==
                train_cat_vocab.vocab_size("region"));
    }

    SECTION("shared taxonomy resolves through the train vocab; new -> 0") {
        const auto& g_ids = test_ds.genus_ids();
        const auto& f_ids = test_ds.family_ids();
        REQUIRE(g_ids.defined());
        REQUIRE(f_ids.defined());
        REQUIRE(g_ids.size(0) == test_ds.n_plots());
        REQUIRE(g_ids.size(1) == 1);  // top_k = 1
        auto ga = g_ids.accessor<int64_t, 2>();
        auto fa = f_ids.accessor<int64_t, 2>();

        // E0 sp_a -> gen_x/fam_x (shared with train)
        REQUIRE(ga[pid_to_row.at("E0")][0] == gen_x);
        REQUIRE(fa[pid_to_row.at("E0")][0] == fam_x);
        // E1 sp_b -> gen_y/fam_y (shared)
        REQUIRE(ga[pid_to_row.at("E1")][0] == gen_y);
        REQUIRE(fa[pid_to_row.at("E1")][0] == fam_y);
        // E4 sp_c -> gen_x/fam_x (shared via train sp_c row)
        REQUIRE(ga[pid_to_row.at("E4")][0] == gen_x);
        REQUIRE(fa[pid_to_row.at("E4")][0] == fam_x);
        // E2 sp_test_only -> gen_new/fam_new (NOT in train vocab)
        REQUIRE(ga[pid_to_row.at("E2")][0] == 0);
        REQUIRE(fa[pid_to_row.at("E2")][0] == 0);
    }

    SECTION("inherited classification class mapping aligns with train") {
        // Both train and test see "forest" and "grass". Class indices must be
        // identical across the two datasets so the trained head's softmax is
        // indexed correctly when predicting the test set.
        const auto& train_tgt = train_ds.targets().at("habitat");
        const auto& test_tgt  = test_ds.targets().at("habitat");
        REQUIRE(train_tgt.dtype() == torch::kLong);
        REQUIRE(test_tgt.dtype()  == torch::kLong);

        // Pick a known "forest" plot in each split and verify same code.
        // Train T0 = forest. Test E0 = forest.
        auto train_acc = train_tgt.accessor<int64_t, 1>();
        auto test_acc  = test_tgt.accessor<int64_t, 1>();

        auto train_row_of = [&](const std::string& pid) -> int64_t {
            const auto& pids = train_ds.plot_ids();
            for (size_t i = 0; i < pids.size(); ++i) {
                if (pids[i] == pid) return static_cast<int64_t>(i);
            }
            return -1;
        };

        const int64_t forest_code = train_acc[train_row_of("T0")];
        const int64_t grass_code  = train_acc[train_row_of("T1")];
        REQUIRE(forest_code != grass_code);

        REQUIRE(test_acc[pid_to_row.at("E0")] == forest_code);  // forest
        REQUIRE(test_acc[pid_to_row.at("E1")] == grass_code);   // grass
        REQUIRE(test_acc[pid_to_row.at("E2")] == grass_code);   // grass
        REQUIRE(test_acc[pid_to_row.at("E4")] == forest_code);  // forest

        // The TargetConfig on the schema carries the train class_names so
        // num_classes round-trips identically.
        REQUIRE(test_ds.schema().targets.size() == 1);
        REQUIRE(test_ds.schema().targets[0].num_classes ==
                train_ds.schema().targets[0].num_classes);
        REQUIRE(test_ds.schema().targets[0].class_names ==
                train_ds.schema().targets[0].class_names);
    }

    SECTION("species vocab size unchanged — no test-only species added") {
        // Reused vocab must report identical species counts on both splits.
        // (Catches regressions where from_csv_with_schema accidentally extends
        // the vocab with test-only species.)
        REQUIRE(test_ds.schema().n_species == train_ds.schema().n_species);
        REQUIRE(test_ds.schema().n_species_vocab ==
                train_ds.schema().n_species_vocab);
    }
}

TEST_CASE("ResolveDataset classification target", "[dataset]") {
    TempFile csv(
        "plot_id,species,habitat\n"
        "p1,sp1,2\n"
        "p1,sp2,2\n"
        "p2,sp1,5\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";

    std::vector<TargetSpec> targets = {
        TargetSpec::classification("habitat", 10)
    };

    DatasetConfig config;

    auto dataset = ResolveDataset::from_species_csv(
        csv.path(), roles, targets, config
    );

    SECTION("classification target is loaded as long tensor") {
        auto& tgt = dataset.targets();
        REQUIRE(tgt.count("habitat") > 0);
        auto& habitat = tgt.at("habitat");
        REQUIRE(habitat.dtype() == torch::kLong);
    }

    SECTION("schema contains classification config") {
        auto& schema = dataset.schema();
        REQUIRE(schema.targets.size() == 1);
        REQUIRE(schema.targets[0].task == TaskType::Classification);
        REQUIRE(schema.targets[0].num_classes == 10);
    }
}

TEST_CASE("from_csv fails loudly on missing role columns", "[dataset][validation]") {
    TempFile header(
        "plot_id,lat,lon,bio1,bio2,area\n"
        "p1,50.0,10.0,1.0,2.0,100\n"
        "p2,51.0,11.0,1.5,2.5,200\n"
    );
    TempFile species(
        "plot_id,species,cover\n"
        "p1,sp1,1.0\n"
        "p2,sp2,1.0\n"
    );

    auto base_roles = [] {
        RoleMapping r;
        r.plot_id = "plot_id"; r.species_id = "species"; r.abundance = "cover";
        r.latitude = "lat"; r.longitude = "lon";
        return r;
    };
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.hash_dim = 8;

    SECTION("typo'd covariate column throws (not silently dropped)") {
        RoleMapping roles = base_roles();
        roles.covariates = {"bio1", "boi2"};  // typo
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("area")}, cfg));
    }

    SECTION("missing target column throws (not a silent 0-row dataset)") {
        RoleMapping roles = base_roles();
        roles.covariates = {"bio1", "bio2"};
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("areea")}, cfg));
    }

    SECTION("missing coordinate column throws") {
        RoleMapping roles = base_roles();
        roles.latitude = "lattitude";  // typo
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("area")}, cfg));
    }

    SECTION("all columns present loads cleanly") {
        RoleMapping roles = base_roles();
        roles.covariates = {"bio1", "bio2"};
        auto ds = ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("area")}, cfg);
        REQUIRE(ds.n_plots() == 2);
        REQUIRE(ds.schema().covariate_names.size() == 2);
    }
}

// Issue #94: the single-table species loaders must throw loudly when a role
// names a column that cannot be resolved (typo / absent), instead of silently
// dropping the feature and baking the wrong feature count into the checkpoint --
// matching the header loader's guard tested above.
TEST_CASE("ResolveDataset from_species_csv throws on unresolvable role column",
          "[dataset]") {
    TempFile csv(
        "plot_id,species,cover,lon,lat,genus,family,area\n"
        "p1,sp1,0.5,10.0,50.0,Quercus,Fagaceae,100\n"
        "p2,sp1,0.8,11.0,51.0,Quercus,Fagaceae,200\n"
    );

    auto base_roles = [] {
        RoleMapping r;
        r.plot_id = "plot_id"; r.species_id = "species"; r.abundance = "cover";
        r.longitude = "lon"; r.latitude = "lat";
        r.genus = "genus"; r.family = "family";
        return r;
    };
    std::vector<TargetSpec> targets = {TargetSpec::regression("area")};
    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 8;

    SECTION("typo'd genus column throws") {
        RoleMapping roles = base_roles();
        roles.genus = "genuss";  // typo
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("typo'd family column throws") {
        RoleMapping roles = base_roles();
        roles.family = "fam";  // typo
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("typo'd abundance column throws") {
        RoleMapping roles = base_roles();
        roles.abundance = "coverr";  // typo
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("typo'd longitude column throws") {
        RoleMapping roles = base_roles();
        roles.longitude = "long";  // typo
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("typo'd latitude column throws") {
        RoleMapping roles = base_roles();
        roles.latitude = "lattitude";  // typo
        REQUIRE_THROWS(ResolveDataset::from_species_csv(csv.path(), roles, targets, config));
    }
    SECTION("all role columns present loads with taxonomy + coordinates") {
        RoleMapping roles = base_roles();
        auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);
        REQUIRE(ds.n_plots() == 2);
        REQUIRE(ds.schema().has_taxonomy == true);
        REQUIRE(ds.schema().has_coordinates == true);
    }
    SECTION("unset optional roles do not throw (only named-but-absent do)") {
        RoleMapping roles;
        roles.plot_id = "plot_id"; roles.species_id = "species";
        // no abundance / coords / taxonomy roles set at all
        auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);
        REQUIRE(ds.n_plots() == 2);
        REQUIRE(ds.schema().has_taxonomy == false);
        REQUIRE(ds.schema().has_coordinates == false);
    }
}

// Issue #94 (related): an abundance column that is present but holds a
// missing/NA/unparseable cell defaults the weight to 1.0 and warns, rather than
// silently conflating missing cover with a real presence. Load must still
// succeed (warning, not error).
TEST_CASE("ResolveDataset from_species_csv coerces unparseable abundance to 1.0",
          "[dataset]") {
    TempFile csv(
        "plot_id,species,cover,area\n"
        "p1,sp1,0.5,100\n"
        "p1,sp2,NA,100\n"       // NA cover -> 1.0 + warn
        "p2,sp1,,200\n"         // empty cover -> 1.0 + warn
        "p2,sp3,2.5abc,200\n"   // garbage cover -> 1.0 + warn
    );

    RoleMapping roles;
    roles.plot_id = "plot_id"; roles.species_id = "species"; roles.abundance = "cover";
    std::vector<TargetSpec> targets = {TargetSpec::regression("area")};
    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;

    auto ds = ResolveDataset::from_species_csv(csv.path(), roles, targets, config);
    REQUIRE(ds.n_plots() == 2);
    // sp2 (NA cover) and sp3 (garbage cover) still loaded as presences.
    REQUIRE(ds.schema().n_species_vocab == 4);  // <UNK> + sp1 + sp2 + sp3
}

// Issue #68: a species that occurs only in a plot dropped for a missing target
// must NOT inflate the vocabulary (its embedding row would never be referenced).
TEST_CASE("ResolveDataset vocab excludes species only in dropped plots",
          "[dataset][vocab]") {
    // p3 has a missing (NA) target -> dropped; "sp_only" appears only in p3.
    TempFile csv(
        "plot_id,species,cover,y\n"
        "p1,sp1,1.0,10\n"
        "p1,sp2,1.0,10\n"
        "p2,sp1,1.0,20\n"
        "p2,sp3,1.0,20\n"
        "p3,sp_only,1.0,NA\n"
    );

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";

    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;

    auto ds = ResolveDataset::from_species_csv(
        csv.path(), roles, {TargetSpec::regression("y")}, config);

    REQUIRE(ds.n_plots() == 2);  // p3 dropped

    const auto& vocab = ds.species_vocab();
    REQUIRE(std::find(vocab.begin(), vocab.end(), "sp_only") == vocab.end());
    REQUIRE(std::find(vocab.begin(), vocab.end(), "sp1") != vocab.end());
    REQUIRE(std::find(vocab.begin(), vocab.end(), "sp3") != vocab.end());
    // <UNK> + sp1 + sp2 + sp3 == 4 (sp_only excluded).
    REQUIRE(ds.schema().n_species_vocab == 4);
}

// ============================================================================
// Clearing an optional role (issue #111)
// ============================================================================
//
// RoleMapping's optional columns are std::optional<std::string>, but a caller
// whose language binding offers no "unset" spelling clears a role by assigning
// the empty string. Read as a NAME that produced `column not found: ""`, so a
// deliberately cleared coordinate role became a load failure. Empty means
// unset -- and only empty: a non-empty name the file does not carry is still
// the loud configuration error issue #94 added.

TEST_CASE("an optional role cleared with an empty string is unset",
          "[dataset][roles][issue111]") {
    TempFile header(
        "plot_id,lat,lon,bio1,area\n"
        "p1,50.0,10.0,1.0,100\n"
        "p2,51.0,11.0,1.5,200\n"
    );
    TempFile species(
        "plot_id,species,cover,genus,family\n"
        "p1,sp1,1.0,Quercus,Fagaceae\n"
        "p2,sp2,1.0,Pinus,Pinaceae\n"
    );

    auto base_roles = [] {
        RoleMapping r;
        r.plot_id = "plot_id"; r.species_id = "species"; r.abundance = "cover";
        r.latitude = "lat"; r.longitude = "lon";
        r.genus = "genus"; r.family = "family";
        return r;
    };
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.hash_dim = 8;

    SECTION("RoleMapping reports a cleared role as unset") {
        RoleMapping roles = base_roles();
        roles.latitude = "";
        roles.longitude = "";
        REQUIRE_FALSE(roles.has_coordinates());
        REQUIRE_FALSE(roles.latitude_column().has_value());
        REQUIRE_FALSE(roles.longitude_column().has_value());

        roles.genus = "";
        REQUIRE(roles.has_taxonomy());  // family still mapped
        roles.family = "";
        REQUIRE_FALSE(roles.has_taxonomy());

        roles.abundance = "";
        REQUIRE_FALSE(roles.has_abundance());
    }

    SECTION("a nullopt role and an empty-string role load identically") {
        RoleMapping cleared = base_roles();
        cleared.latitude = "";
        cleared.longitude = "";

        RoleMapping unset = base_roles();
        unset.latitude = std::nullopt;
        unset.longitude = std::nullopt;

        auto from_cleared = ResolveDataset::from_csv(
            header.path(), species.path(), cleared, {TargetSpec::regression("area")}, cfg);
        auto from_unset = ResolveDataset::from_csv(
            header.path(), species.path(), unset, {TargetSpec::regression("area")}, cfg);

        REQUIRE(from_cleared.n_plots() == from_unset.n_plots());
        REQUIRE(from_cleared.schema().has_coordinates == from_unset.schema().has_coordinates);
        REQUIRE_FALSE(from_cleared.schema().has_coordinates);
        REQUIRE(torch::equal(from_cleared.hash_embedding(), from_unset.hash_embedding()));
    }

    SECTION("clearing a species-table role is unset, not a missing column") {
        RoleMapping roles = base_roles();
        roles.genus = "";
        roles.family = "";
        auto ds = ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("area")}, cfg);
        REQUIRE(ds.n_plots() == 2);
        REQUIRE_FALSE(ds.schema().has_taxonomy);
    }

    SECTION("a non-empty name the file lacks still throws") {
        RoleMapping roles = base_roles();
        roles.latitude = "lattitude";
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header.path(), species.path(), roles, {TargetSpec::regression("area")}, cfg));

        RoleMapping typo_genus = base_roles();
        typo_genus.genus = "genuss";
        REQUIRE_THROWS(ResolveDataset::from_csv(
            header.path(), species.path(), typo_genus, {TargetSpec::regression("area")}, cfg));
    }
}
