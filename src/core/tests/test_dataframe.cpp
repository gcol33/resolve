#include <catch2/catch_test_macros.hpp>
#include "resolve/dataset.hpp"
#include "resolve/row_source.hpp"
#include "resolve/role_mapping.hpp"
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

using namespace resolve;

// ============================================================================
// In-memory (DataFrame) dataset loaders (issue #22).
//
// The contract: a dataset built from an in-memory ColumnTable is byte-identical
// to one built from the equivalent CSV, because both paths run the exact same
// loader body through the RowSource seam. These tests build the same data both
// ways and assert tensor-level equality across every encoded field.
// ============================================================================

namespace {

int g_counter = 0;

class TempCSV {
public:
    explicit TempCSV(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_df_test_" + std::to_string(g_counter++) + ".csv");
        std::ofstream f(path_);
        f << content;
    }
    ~TempCSV() { std::filesystem::remove(path_); }
    std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
};

std::string join_csv(const std::vector<std::string>& names,
                     const std::vector<std::vector<std::string>>& rows) {
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i) out += ",";
        out += names[i];
    }
    out += "\n";
    for (const auto& r : rows) {
        for (size_t i = 0; i < r.size(); ++i) {
            if (i) out += ",";
            out += r[i];
        }
        out += "\n";
    }
    return out;
}

// Row-major rows -> column-major ColumnTable.
ColumnTable to_table(const std::vector<std::string>& names,
                     const std::vector<std::vector<std::string>>& rows) {
    std::vector<std::vector<std::string>> cols(names.size());
    for (auto& c : cols) c.reserve(rows.size());
    for (const auto& r : rows) {
        for (size_t c = 0; c < names.size(); ++c) cols[c].push_back(r[c]);
    }
    return ColumnTable(names, std::move(cols));
}

void require_tensor_eq(const torch::Tensor& a, const torch::Tensor& b) {
    REQUIRE(a.defined() == b.defined());
    if (!a.defined()) return;
    REQUIRE(a.sizes() == b.sizes());
    REQUIRE(a.dtype() == b.dtype());
    REQUIRE(torch::equal(a, b));
}

void require_dataset_eq(const ResolveDataset& a, const ResolveDataset& b) {
    REQUIRE(a.n_plots() == b.n_plots());
    REQUIRE(a.plot_ids() == b.plot_ids());
    require_tensor_eq(a.coordinates(), b.coordinates());
    require_tensor_eq(a.covariates(), b.covariates());
    require_tensor_eq(a.species_ids(), b.species_ids());
    require_tensor_eq(a.species_vector(), b.species_vector());
    require_tensor_eq(a.hash_embedding(), b.hash_embedding());
    require_tensor_eq(a.genus_ids(), b.genus_ids());
    require_tensor_eq(a.family_ids(), b.family_ids());
    require_tensor_eq(a.categorical_ids(), b.categorical_ids());
    require_tensor_eq(a.pool_genus_ids(), b.pool_genus_ids());
    require_tensor_eq(a.pool_family_ids(), b.pool_family_ids());
    require_tensor_eq(a.pool_weights(), b.pool_weights());
    require_tensor_eq(a.pool_mask(), b.pool_mask());
    require_tensor_eq(a.pool_has_cover(), b.pool_has_cover());
    REQUIRE(a.targets().size() == b.targets().size());
    for (const auto& [k, v] : a.targets()) {
        REQUIRE(b.targets().count(k) == 1);
        require_tensor_eq(v, b.targets().at(k));
    }
}

// Header: one row per plot (targets, covariate, categorical).
const std::vector<std::string> kHeaderNames =
    {"plot_id", "area", "habitat", "elevation", "soil"};
const std::vector<std::vector<std::string>> kHeaderRows = {
    {"p1", "100", "forest", "12.5", "sand"},
    {"p2", "200", "grass",  "8.0",  "clay"},
    {"p3", "150", "forest", "20.0", "sand"},
};

// Species: long format (multiple rows per plot), with taxonomy + cover.
const std::vector<std::string> kSpeciesNames =
    {"plot_id", "species", "cover", "genus", "family"};
const std::vector<std::vector<std::string>> kSpeciesRows = {
    {"p1", "sp1", "0.5", "Quercus", "Fagaceae"},
    {"p1", "sp2", "0.3", "Fagus",   "Fagaceae"},
    {"p2", "sp1", "0.8", "Quercus", "Fagaceae"},
    {"p2", "sp3", "0.2", "Pinus",   "Pinaceae"},
    {"p3", "sp2", "0.4", "Fagus",   "Fagaceae"},
    {"p3", "sp3", "0.6", "Pinus",   "Pinaceae"},
};

RoleMapping make_roles() {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    roles.covariates = {"elevation"};
    roles.categoricals = {"soil"};
    return roles;
}

std::vector<TargetSpec> make_targets() {
    return {
        TargetSpec::regression("area"),
        TargetSpec::classification("habitat", 2),
    };
}

DatasetConfig make_config() {
    DatasetConfig c;
    c.species_encoding = SpeciesEncodingMode::RankPool;
    c.pool_weighting = PoolWeighting::Log1p;
    c.use_taxonomy = true;
    return c;
}

}  // namespace

TEST_CASE("from_dataframe matches from_csv (two-file)", "[dataset][dataframe]") {
    TempCSV header_csv(join_csv(kHeaderNames, kHeaderRows));
    TempCSV species_csv(join_csv(kSpeciesNames, kSpeciesRows));

    auto roles = make_roles();
    auto targets = make_targets();
    auto config = make_config();

    auto from_disk = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles, targets, config);

    auto header_tbl = to_table(kHeaderNames, kHeaderRows);
    auto species_tbl = to_table(kSpeciesNames, kSpeciesRows);
    auto from_mem = ResolveDataset::from_dataframe(
        header_tbl, species_tbl, roles, targets, config);

    require_dataset_eq(from_disk, from_mem);
}

TEST_CASE("from_species_dataframe matches from_species_csv", "[dataset][dataframe]") {
    // Single long table carrying the target inline (one value per plot).
    const std::vector<std::string> names =
        {"plot_id", "species", "cover", "genus", "family", "area"};
    const std::vector<std::vector<std::string>> rows = {
        {"p1", "sp1", "0.5", "Quercus", "Fagaceae", "100"},
        {"p1", "sp2", "0.3", "Fagus",   "Fagaceae", "100"},
        {"p2", "sp1", "0.8", "Quercus", "Fagaceae", "200"},
        {"p2", "sp3", "0.2", "Pinus",   "Pinaceae", "200"},
    };
    TempCSV csv(join_csv(names, rows));

    auto roles = make_roles();
    std::vector<TargetSpec> targets = {TargetSpec::regression("area")};
    auto config = make_config();

    auto from_disk = ResolveDataset::from_species_csv(
        csv.path(), roles, targets, config);
    auto from_mem = ResolveDataset::from_species_dataframe(
        to_table(names, rows), roles, targets, config);

    require_dataset_eq(from_disk, from_mem);
}

TEST_CASE("from_dataframe_header matches from_csv", "[dataset][dataframe]") {
    TempCSV header_csv(join_csv(kHeaderNames, kHeaderRows));
    TempCSV species_csv(join_csv(kSpeciesNames, kSpeciesRows));

    auto roles = make_roles();
    auto targets = make_targets();
    auto config = make_config();

    auto from_disk = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles, targets, config);

    // Header in memory, species streamed from the same CSV path.
    auto from_mixed = ResolveDataset::from_dataframe_header(
        to_table(kHeaderNames, kHeaderRows), species_csv.path(),
        roles, targets, config);

    require_dataset_eq(from_disk, from_mixed);
}

TEST_CASE("from_dataframe_with_schema matches from_csv_with_schema",
          "[dataset][dataframe]") {
    TempCSV header_csv(join_csv(kHeaderNames, kHeaderRows));
    TempCSV species_csv(join_csv(kSpeciesNames, kSpeciesRows));

    auto roles = make_roles();
    auto targets = make_targets();
    auto config = make_config();

    // Fit a source dataset to provide the vocabularies / class mappings.
    auto source = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles, targets, config);

    auto from_disk = ResolveDataset::from_csv_with_schema(
        header_csv.path(), species_csv.path(), roles, targets, source, config);
    auto from_mem = ResolveDataset::from_dataframe_with_schema(
        to_table(kHeaderNames, kHeaderRows), to_table(kSpeciesNames, kSpeciesRows),
        roles, targets, source, config);

    require_dataset_eq(from_disk, from_mem);
}

TEST_CASE("from_dataframe matches from_csv (hash encoding)", "[dataset][dataframe]") {
    TempCSV header_csv(join_csv(kHeaderNames, kHeaderRows));
    TempCSV species_csv(join_csv(kSpeciesNames, kSpeciesRows));

    auto roles = make_roles();
    auto targets = make_targets();
    DatasetConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 16;
    config.use_taxonomy = true;

    auto from_disk = ResolveDataset::from_csv(
        header_csv.path(), species_csv.path(), roles, targets, config);
    auto from_mem = ResolveDataset::from_dataframe(
        to_table(kHeaderNames, kHeaderRows), to_table(kSpeciesNames, kSpeciesRows),
        roles, targets, config);

    require_dataset_eq(from_disk, from_mem);
}

TEST_CASE("ColumnTable validates shape and names", "[dataset][dataframe]") {
    SECTION("unequal column lengths throw") {
        std::vector<std::string> names = {"a", "b"};
        std::vector<std::vector<std::string>> cols = {{"1", "2"}, {"3"}};
        REQUIRE_THROWS_AS(ColumnTable(names, cols), std::runtime_error);
    }
    SECTION("duplicate column names throw") {
        std::vector<std::string> names = {"a", "a"};
        std::vector<std::vector<std::string>> cols = {{"1"}, {"2"}};
        REQUIRE_THROWS_AS(ColumnTable(names, cols), std::runtime_error);
    }
    SECTION("names/columns count mismatch throws") {
        std::vector<std::string> names = {"a"};
        std::vector<std::vector<std::string>> cols = {{"1"}, {"2"}};
        REQUIRE_THROWS_AS(ColumnTable(names, cols), std::runtime_error);
    }
}

TEST_CASE("InMemoryRowSource yields rows in order", "[dataset][dataframe]") {
    auto tbl = to_table(kSpeciesNames, kSpeciesRows);
    InMemoryRowSource src(tbl);
    REQUIRE(src.num_rows() == kSpeciesRows.size());
    REQUIRE(src.columns() == kSpeciesNames);
    REQUIRE(src.column_index("cover") == 2);
    REQUIRE(src.column_index("absent") == -1);

    std::vector<std::vector<std::string>> seen;
    src.read_rows([&](size_t i, const std::vector<std::string>& row) {
        REQUIRE(i == seen.size());
        seen.push_back(row);
    });
    REQUIRE(seen == kSpeciesRows);
}
