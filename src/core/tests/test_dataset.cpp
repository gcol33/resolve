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
