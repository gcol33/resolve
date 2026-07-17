// Regression tests for the 2026-07-17 review fixes (issues #23-#36). Each test
// pins a specific defect so it cannot silently return.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/encoder.hpp"
#include "resolve/categorical.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace resolve;

namespace {
class TempFile {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_reviewfix_" + std::to_string(counter_++) + suffix);
        std::ofstream f(path_);
        f << content;
    }
    ~TempFile() { std::filesystem::remove(path_); }
    [[nodiscard]] std::string path() const { return path_.string(); }
private:
    std::filesystem::path path_;
    static int counter_;
};
int TempFile::counter_ = 0;

RoleMapping basic_roles() {
    RoleMapping r;
    r.plot_id = "plot_id"; r.species_id = "sp"; r.abundance = "cover";
    return r;
}
DatasetConfig hash_cfg() {
    DatasetConfig c;
    c.species_encoding = SpeciesEncodingMode::Hash;
    c.hash_dim = 4; c.use_taxonomy = false;
    c.track_unknown_fraction = false; c.track_unknown_count = false;
    return c;
}
}  // namespace

// #25 -- CLS pooling with zero attention layers used to silently emit a constant
// (the CLS parameter), dropping every species. It must now be rejected.
TEST_CASE("Transformer CLS pooling requires >=1 attention layer", "[reviewfix][transformer]") {
    REQUIRE_THROWS_AS(
        PlotEncoderTransformer(
            /*n_continuous=*/5, /*n_species=*/50, /*n_genera=*/0, /*n_families=*/0,
            /*d_model=*/32, /*n_heads=*/2, /*n_attention_layers=*/0,
            /*transformer_ff_dim=*/64, /*transformer_pooling=*/"cls"),
        std::invalid_argument);

    // attention pooling with 0 layers stays valid.
    REQUIRE_NOTHROW(
        PlotEncoderTransformer(
            /*n_continuous=*/5, /*n_species=*/50, /*n_genera=*/0, /*n_families=*/0,
            /*d_model=*/32, /*n_heads=*/2, /*n_attention_layers=*/0,
            /*transformer_ff_dim=*/64, /*transformer_pooling=*/"attention"));
}

// #25 -- with an attention layer, CLS pooling must depend on the species input
// (a constant pooled vector would make every plot identical).
TEST_CASE("Transformer CLS pooling depends on species", "[reviewfix][transformer]") {
    PlotEncoderTransformer encoder(
        /*n_continuous=*/4, /*n_species=*/40, /*n_genera=*/0, /*n_families=*/0,
        /*d_model=*/16, /*n_heads=*/2, /*n_attention_layers=*/1,
        /*transformer_ff_dim=*/32, /*transformer_pooling=*/"cls");
    encoder->eval();

    auto continuous = torch::zeros({2, 4});
    auto sp_a = torch::randint(1, 40, {2, 6}, torch::kInt64);
    auto sp_b = torch::randint(1, 40, {2, 6}, torch::kInt64);
    auto out_a = encoder->forward(continuous, sp_a);
    auto out_b = encoder->forward(continuous, sp_b);
    REQUIRE(out_a.isfinite().all().item<bool>());
    REQUIRE_FALSE(torch::allclose(out_a, out_b));
}

// #7 -- NA detection is one canonical, case-insensitive function shared by the
// target and categorical paths.
TEST_CASE("is_na_string is canonical and case-insensitive", "[reviewfix][na]") {
    for (const auto* s : {"", "NA", "na", "Na", "N/A", "n/a", "NaN", "NAN",
                          "nan", "NULL", "null", "None", "none", ".", "-"}) {
        REQUIRE(is_na_string(s));
    }
    for (const auto* s : {"0", "NAN1", "species", "N", "na ", " NA"}) {
        REQUIRE_FALSE(is_na_string(s));
    }
}

// #11 -- duplicate plot_id in the header must be rejected, not silently create
// two plot slots sharing one species record.
TEST_CASE("Duplicate plot_id in header is rejected", "[reviewfix][dataset]") {
    TempFile hdr("plot_id,y\nP0,1.0\nP1,2.0\nP0,3.0\n");
    TempFile spc("plot_id,sp,cover\nP0,sp1,1.0\nP1,sp2,1.0\n");
    REQUIRE_THROWS(ResolveDataset::from_csv(
        hdr.path(), spc.path(), basic_roles(),
        {TargetSpec::regression("y")}, hash_cfg()));
}

// #9 -- a quoted field containing an embedded newline is one logical row, not
// two physical rows.
TEST_CASE("CSV quoted field with embedded newline is one row", "[reviewfix][csv]") {
    TempFile hdr("plot_id,note,y\nP0,\"line one\nline two\",10\nP1,simple,20\n");
    TempFile spc("plot_id,sp,cover\nP0,sp1,1.0\nP1,sp2,1.0\n");
    RoleMapping roles = basic_roles();
    auto ds = ResolveDataset::from_csv(
        hdr.path(), spc.path(), roles, {TargetSpec::regression("y")}, hash_cfg());
    REQUIRE(ds.n_plots() == 2);
}

// #24 -- a classification column with a negative integer label used to write out
// of bounds. It must load (falling back to a compact factorization).
TEST_CASE("Negative integer class label loads without crashing", "[reviewfix][dataset]") {
    std::ostringstream hdr, spc;
    hdr << "plot_id,hab\n";
    spc << "plot_id,sp,cover\n";
    for (int i = 0; i < 30; ++i) {
        int hab = (i % 3) - 1;  // -1, 0, 1
        hdr << "P" << i << "," << hab << "\n";
        spc << "P" << i << ",sp" << (i % 4) << ",1.0\n";
    }
    TempFile hdr_f(hdr.str()), spc_f(spc.str());
    ResolveDataset ds;
    REQUIRE_NOTHROW(ds = ResolveDataset::from_csv(
        hdr_f.path(), spc_f.path(), basic_roles(),
        {TargetSpec::classification("hab", 3)}, hash_cfg()));
    REQUIRE(ds.n_plots() == 30);
}
