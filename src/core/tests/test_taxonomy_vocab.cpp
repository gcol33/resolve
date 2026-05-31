// Tests for the issue #5 fix: TaxonomyVocab must assign genus/family IDs
// deterministically (sorted), independent of the order records arrive in.
//
// The bug: TaxonomyVocab::fit assigned IDs by first-appearance order, so the
// same underlying data in a different CSV row order produced a different
// genus/family -> ID mapping. A checkpoint trained on one ordering and scored
// against a differently-ordered rebuild (from_csv_with_schema in another
// process) silently misaligned the genus/family embedding lookups (~5pp EUNIS
// accuracy drop downstream). SpeciesVocab was already sorted; TaxonomyVocab is
// now too.
//
// Coverage:
//   1. Unit: fit() yields the same map for any record ordering, IDs are in
//      sorted order, and <UNK> stays at 0.
//   2. Unit: save/load round-trips the (now sorted) maps.
//   3. Regression: two datasets whose taxonomy vocab is fit from the same
//      species set in different row orders, reused via from_csv_with_schema,
//      produce identical genus_ids / family_ids tensors (the downstream
//      symptom in the issue).

#include <catch2/catch_test_macros.hpp>

#include "resolve/types.hpp"
#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace resolve;

namespace {

SpeciesRecord mk(const std::string& sp, const std::string& genus,
                 const std::string& family) {
    SpeciesRecord r;
    r.species_id = sp;
    r.genus = genus;
    r.family = family;
    r.abundance = 1.0f;
    r.plot_id = "P0";
    return r;
}

class TempFile {
public:
    explicit TempFile(const std::string& content,
                      const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_taxvocab_" + std::to_string(counter_++) + suffix);
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

}  // namespace

// =============================================================================
// 1. fit() is order-independent and sorted
// =============================================================================

TEST_CASE("TaxonomyVocab::fit assigns IDs in sorted, order-independent way",
          "[taxonomy][vocab]") {
    // Genera deliberately NOT in alphabetical first-appearance order.
    std::vector<SpeciesRecord> recs = {
        mk("s1", "Quercus", "Fagaceae"),
        mk("s2", "Abies",   "Pinaceae"),
        mk("s3", "Zelkova", "Ulmaceae"),
        mk("s4", "Betula",  "Betulaceae"),
        mk("s2", "Abies",   "Pinaceae"),   // duplicate genus/family
        mk("s5", "",        ""),           // empty -> ignored
    };

    auto reversed = recs;
    std::reverse(reversed.begin(), reversed.end());

    TaxonomyVocab va;
    va.fit(recs);
    TaxonomyVocab vb;
    vb.fit(reversed);

    // Order-independence: identical name -> ID maps for both orderings.
    REQUIRE(va.genus_map() == vb.genus_map());
    REQUIRE(va.family_map() == vb.family_map());

    // <UNK> reserved at 0.
    REQUIRE(va.encode_genus("<UNK>") == 0);
    REQUIRE(va.encode_family("<UNK>") == 0);

    // Sorted assignment: Abies < Betula < Quercus < Zelkova -> 1..4.
    REQUIRE(va.encode_genus("Abies") == 1);
    REQUIRE(va.encode_genus("Betula") == 2);
    REQUIRE(va.encode_genus("Quercus") == 3);
    REQUIRE(va.encode_genus("Zelkova") == 4);

    // Sorted families: Betulaceae < Fagaceae < Pinaceae < Ulmaceae -> 1..4.
    REQUIRE(va.encode_family("Betulaceae") == 1);
    REQUIRE(va.encode_family("Fagaceae") == 2);
    REQUIRE(va.encode_family("Pinaceae") == 3);
    REQUIRE(va.encode_family("Ulmaceae") == 4);

    // 4 genera + 4 families + the reserved <UNK> slot each.
    REQUIRE(va.n_genera() == 5);
    REQUIRE(va.n_families() == 5);

    // Unknown names fall back to 0.
    REQUIRE(va.encode_genus("Nothofagus") == 0);
}

// =============================================================================
// 2. save / load round-trip preserves the sorted maps
// =============================================================================

TEST_CASE("TaxonomyVocab save/load round-trips the maps", "[taxonomy][vocab]") {
    std::vector<SpeciesRecord> recs = {
        mk("s1", "Quercus", "Fagaceae"),
        mk("s2", "Abies",   "Pinaceae"),
        mk("s3", "Betula",  "Betulaceae"),
    };
    TaxonomyVocab va;
    va.fit(recs);

    auto path = (std::filesystem::temp_directory_path() /
                 "resolve_taxvocab_roundtrip.pt").string();
    {
        torch::serialize::OutputArchive oa;
        va.save(oa, "tax_");
        oa.save_to(path);
    }
    TaxonomyVocab vb;
    {
        torch::serialize::InputArchive ia;
        ia.load_from(path);
        vb = TaxonomyVocab::load(ia, "tax_");
    }
    std::filesystem::remove(path);

    REQUIRE(va.genus_map() == vb.genus_map());
    REQUIRE(va.family_map() == vb.family_map());
}

// =============================================================================
// 3. Regression: row-order of the schema source no longer changes the
//    genus/family ID tensors of a dataset built via from_csv_with_schema
// =============================================================================

namespace {

// Header: one row per plot, a single regression target.
std::string make_header(int n_plots) {
    std::ostringstream h;
    h << "plot_id,y\n";
    for (int i = 0; i < n_plots; ++i) {
        h << "P" << i << "," << (1.0 + i) << "\n";
    }
    return h.str();
}

// Species rows for n_plots, each plot carrying the same 3 species drawn from a
// pool whose genera span the alphabet. `reversed` writes the global row order
// backwards, which flips first-appearance order of the genera while keeping the
// per-plot species set (and within-plot order) identical.
std::string make_species(int n_plots, bool reversed) {
    static const char* genera[]  = {"Zelkova", "Quercus", "Abies",
                                    "Betula", "Larix", "Fagus"};
    static const char* families[] = {"Ulmaceae", "Fagaceae", "Pinaceae",
                                     "Betulaceae", "Pinaceae", "Fagaceae"};
    std::vector<std::string> rows;
    for (int i = 0; i < n_plots; ++i) {
        for (int k = 0; k < 3; ++k) {
            int g = (i + k) % 6;
            std::ostringstream r;
            r << "P" << i << ",sp" << g << "," << genera[g] << ","
              << families[g] << ",1.0";
            rows.push_back(r.str());
        }
    }
    if (reversed) std::reverse(rows.begin(), rows.end());

    std::ostringstream s;
    s << "plot_id,sp,genus,family,cover\n";
    for (const auto& r : rows) s << r << "\n";
    return s.str();
}

RoleMapping tax_roles() {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.genus = "genus";
    roles.family = "family";
    roles.abundance = "cover";
    return roles;
}

}  // namespace

TEST_CASE("from_csv_with_schema genus/family IDs are schema-order invariant",
          "[taxonomy][dataset]") {
    const int n_plots = 12;

    // Two schema sources over the same species set, different row orders.
    TempFile hdr_a(make_header(n_plots)),  spc_a(make_species(n_plots, false));
    TempFile hdr_b(make_header(n_plots)),  spc_b(make_species(n_plots, true));
    // A fixed evaluation set (single fixed row order).
    TempFile hdr_t(make_header(n_plots)),  spc_t(make_species(n_plots, false));

    auto roles = tax_roles();
    std::vector<TargetSpec> targets = {TargetSpec::regression("y")};
    DatasetConfig cfg;  // hash encoding + use_taxonomy default true

    auto src_a = ResolveDataset::from_csv(hdr_a.path(), spc_a.path(), roles, targets, cfg);
    auto src_b = ResolveDataset::from_csv(hdr_b.path(), spc_b.path(), roles, targets, cfg);

    // Sanity: the two sources carry the same genus/family vocabulary content
    // even though they were built from different row orders.
    REQUIRE(src_a.taxonomy_vocab().genus_map() == src_b.taxonomy_vocab().genus_map());
    REQUIRE(src_a.taxonomy_vocab().family_map() == src_b.taxonomy_vocab().family_map());

    auto test_a = ResolveDataset::from_csv_with_schema(
        hdr_t.path(), spc_t.path(), roles, targets, src_a, cfg);
    auto test_b = ResolveDataset::from_csv_with_schema(
        hdr_t.path(), spc_t.path(), roles, targets, src_b, cfg);

    REQUIRE(test_a.genus_ids().defined());
    REQUIRE(test_a.genus_ids().numel() > 0);
    REQUIRE(test_a.genus_ids().sizes() == test_b.genus_ids().sizes());

    // The crux: identical genus/family ID tensors regardless of which schema
    // source's row order was used. Under the first-appearance bug these
    // differed in a large fraction of elements.
    REQUIRE(torch::equal(test_a.genus_ids(), test_b.genus_ids()));
    REQUIRE(torch::equal(test_a.family_ids(), test_b.family_ids()));
}
