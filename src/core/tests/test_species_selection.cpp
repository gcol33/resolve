// DatasetConfig::selection outside the hash encoding (issue #113).
//
// apply_selection was called only inside the hash branch of encode_species, so
// a rank_pool / transformer / sparse dataset reported the selection it was
// given on its schema and then encoded every species the plot recorded, and the
// embed branch hardcoded SelectionMode::Top whatever it was asked for. The
// value was accepted, persisted to the checkpoint, and had no effect.
//
// The contract these tests pin:
//
//   * Each encoding takes its per-plot budget from the knob that also fixes its
//     width -- top_k for hash, top_k_species for embed, and the new
//     DatasetConfig::species_budget for rank_pool / transformer / sparse.
//   * species_budget defaults to 0 = no budget, so a pooled or sparse dataset
//     built with any existing configuration encodes exactly what it did before.
//   * With a budget, Top and Bottom select different species, and exactly the
//     ones the abundance ranking names.
//   * The schema records the selection the run APPLIED (effective_selection),
//     so a checkpoint cannot claim a selection that never happened.
//   * The species vocabulary is fitted over every record whatever the budget,
//     so the integer codes are identical across the arms of an ablation.

#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/model.hpp"
#include "resolve/predictor.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/trainer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;

namespace {

class TempFile {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv") {
        path_ = std::filesystem::temp_directory_path() /
                ("resolve_selection_" + std::to_string(counter_++) + suffix);
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

// ---------------------------------------------------------------------------
// Synthetic corpus
// ---------------------------------------------------------------------------
//
// n_plots plots, each recording all kSpecies species exactly once. Plot i gives
// species j the cover ((j + i) % kSpecies) + 1, so every plot holds the covers
// 1..kSpecies with no ties, and WHICH species is most abundant rotates with the
// plot. A selection that quietly falls back to "the first k records in CSV
// order" therefore disagrees with the ranking on every plot but the first.

constexpr int kSpecies = 8;

std::string species_name(int j) { return "sp_" + std::to_string(j); }

int cover_of(int plot_index, int j) { return ((j + plot_index) % kSpecies) + 1; }

// The species names a top-k / bottom-k selection must keep for one plot.
std::set<std::string> expected_end(int plot_index, int k, bool top) {
    std::vector<std::pair<int, int>> by_cover;  // (cover, species index)
    for (int j = 0; j < kSpecies; ++j) by_cover.emplace_back(cover_of(plot_index, j), j);
    std::sort(by_cover.begin(), by_cover.end(),
              [top](const auto& a, const auto& b) {
                  return top ? a.first > b.first : a.first < b.first;
              });
    std::set<std::string> out;
    for (int i = 0; i < k && i < static_cast<int>(by_cover.size()); ++i) {
        out.insert(species_name(by_cover[static_cast<size_t>(i)].second));
    }
    return out;
}

std::string header_csv(int n_plots) {
    std::ostringstream out;
    out << "plot_id,y\n";
    for (int i = 0; i < n_plots; ++i) {
        out << "p" << i << "," << (1.0 + 0.5 * i) << "\n";
    }
    return out.str();
}

std::string species_csv(int n_plots) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < n_plots; ++i) {
        for (int j = 0; j < kSpecies; ++j) {
            out << "p" << i << "," << species_name(j) << "," << cover_of(i, j)
                << ",gen_" << (j % 3) << ",fam_" << (j % 2) << "\n";
        }
    }
    return out.str();
}

RoleMapping species_roles() {
    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "sp";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    return roles;
}

std::vector<TargetSpec> targets() {
    return {TargetSpec::regression("y")};
}

DatasetConfig base_config(SpeciesEncodingMode mode) {
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.use_taxonomy = true;
    return cfg;
}

// The species names a plot's row actually encodes, read back out of whichever
// tensor the encoding fills.
std::set<std::string> encoded_species(const ResolveDataset& ds, int64_t plot_row) {
    const auto& vocab = ds.species_vocab();
    std::set<std::string> out;

    const auto name_of = [&](int64_t code) -> std::string {
        if (code <= 0 || code >= static_cast<int64_t>(vocab.size())) return {};
        return vocab[static_cast<size_t>(code)];
    };

    if (ds.pool_mask().defined() && ds.pool_mask().numel() > 0) {
        auto ids = ds.species_ids()[plot_row];
        auto mask = ds.pool_mask()[plot_row];
        for (int64_t c = 0; c < ids.size(0); ++c) {
            if (!mask[c].item<bool>()) continue;
            auto name = name_of(ids[c].item<int64_t>());
            if (!name.empty()) out.insert(name);
        }
        return out;
    }
    if (ds.species_vector().defined() && ds.species_vector().numel() > 0) {
        auto row = ds.species_vector()[plot_row];
        for (int64_t c = 0; c < row.size(0); ++c) {
            if (row[c].item<float>() == 0.0f) continue;
            auto name = name_of(c);
            if (!name.empty()) out.insert(name);
        }
        return out;
    }
    // Embed: fixed slots, 0 = padding.
    auto ids = ds.species_ids()[plot_row];
    for (int64_t c = 0; c < ids.size(0); ++c) {
        auto name = name_of(ids[c].item<int64_t>());
        if (!name.empty()) out.insert(name);
    }
    return out;
}

ResolveDataset build(const std::string& header_path, const std::string& species_path,
                     const DatasetConfig& cfg) {
    return ResolveDataset::from_csv(header_path, species_path, species_roles(),
                                    targets(), cfg);
}

}  // namespace

// ===========================================================================
// The shared selection core
// ===========================================================================

TEST_CASE("selection_indices ranks and truncates", "[selection]") {
    const std::vector<std::pair<std::string, float>> species = {
        {"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}};

    SECTION("Top returns the k most abundant, most abundant first") {
        const auto idx = selection_indices(species, SelectionMode::Top, 2);
        REQUIRE(idx.size() == 2);
        REQUIRE(species[idx[0]].first == "c");  // 5.0
        REQUIRE(species[idx[1]].first == "a");  // 3.0
    }

    SECTION("Bottom returns the k least abundant, least abundant first") {
        const auto idx = selection_indices(species, SelectionMode::Bottom, 2);
        REQUIRE(idx.size() == 2);
        REQUIRE(species[idx[0]].first == "b");  // 1.0
        REQUIRE(species[idx[1]].first == "d");  // 2.0
    }

    SECTION("TopBottom concatenates both ends without repeating a species") {
        const auto idx = selection_indices(species, SelectionMode::TopBottom, 2);
        REQUIRE(idx.size() == 4);
        std::set<std::string> names;
        for (size_t i : idx) names.insert(species[i].first);
        REQUIRE(names.size() == 4);  // every entry distinct
    }

    SECTION("TopBottom over a short list keeps each species once") {
        // 4 species, k = 3 per end: the two ends overlap and the dedup has to
        // fire, or a species occupies two slots.
        const auto idx = selection_indices(species, SelectionMode::TopBottom, 3);
        std::set<size_t> unique(idx.begin(), idx.end());
        REQUIRE(unique.size() == idx.size());
        REQUIRE(idx.size() == species.size());
    }

    SECTION("All ignores k and keeps original order") {
        const auto idx = selection_indices(species, SelectionMode::All, 2);
        REQUIRE(idx.size() == species.size());
        for (size_t i = 0; i < idx.size(); ++i) REQUIRE(idx[i] == i);
    }
}

TEST_CASE("apply_selection still gathers what it always did", "[selection]") {
    // Regression guard on the refactor onto selection_indices: the pair-
    // returning form is what the hash branch and the embed encoder consume, and
    // its ORDER carries meaning (slot 0 of an embed row is the top species).
    const std::vector<std::pair<std::string, float>> species = {
        {"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}};

    auto top = apply_selection(species, SelectionMode::Top, 2);
    REQUIRE(top.size() == 2);
    REQUIRE(top[0].first == "c");
    REQUIRE(top[0].second == 5.0f);
    REQUIRE(top[1].first == "a");

    auto bottom = apply_selection(species, SelectionMode::Bottom, 2);
    REQUIRE(bottom[0].first == "b");
    REQUIRE(bottom[1].first == "d");

    auto all = apply_selection(species, SelectionMode::All, 2);
    REQUIRE(all == species);

    // Fewer entries than k: everything survives, untruncated.
    auto short_top = apply_selection(species, SelectionMode::Top, 10);
    REQUIRE(short_top.size() == species.size());
}

TEST_CASE("species_budget_indices is a no-op without a budget", "[selection][budget]") {
    const std::vector<std::pair<std::string, float>> species = {
        {"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}};

    for (auto mode : {SelectionMode::Top, SelectionMode::Bottom,
                      SelectionMode::TopBottom, SelectionMode::All}) {
        for (int budget : {0, -1, -7}) {
            const auto idx = species_budget_indices(species, mode, budget);
            REQUIRE(idx.size() == species.size());
            for (size_t i = 0; i < idx.size(); ++i) REQUIRE(idx[i] == i);
        }
    }

    // All names no rule for picking a subset, so a budget does not narrow it.
    REQUIRE(species_budget_indices(species, SelectionMode::All, 2).size() == species.size());
}

TEST_CASE("species_budget_indices returns CSV order", "[selection][budget]") {
    // The rank-pool species cap documents itself as a slice of CSV row order,
    // so the budget must hand back the survivors in that order rather than in
    // the abundance ranking.
    const std::vector<std::pair<std::string, float>> species = {
        {"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}};

    const auto idx = species_budget_indices(species, SelectionMode::Top, 2);
    REQUIRE(idx.size() == 2);
    REQUIRE(idx[0] == 0);  // "a", CSV position 0
    REQUIRE(idx[1] == 2);  // "c", CSV position 2
    REQUIRE(std::is_sorted(idx.begin(), idx.end()));
}

// ===========================================================================
// effective_selection
// ===========================================================================

TEST_CASE("effective_selection reports what a load applies", "[selection][schema]") {
    SECTION("hash and embed always select") {
        for (auto mode : {SpeciesEncodingMode::Hash, SpeciesEncodingMode::Embed}) {
            auto cfg = base_config(mode);
            cfg.selection = SelectionMode::Bottom;
            cfg.species_budget = 0;
            REQUIRE(effective_selection(cfg) == SelectionMode::Bottom);
        }
    }

    SECTION("pooled and sparse report All without a budget") {
        for (auto mode : {SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer,
                          SpeciesEncodingMode::Sparse}) {
            auto cfg = base_config(mode);
            cfg.selection = SelectionMode::Bottom;
            cfg.species_budget = 0;
            REQUIRE(effective_selection(cfg) == SelectionMode::All);

            cfg.species_budget = 3;
            REQUIRE(effective_selection(cfg) == SelectionMode::Bottom);
        }
    }
}

// ===========================================================================
// The pooled encodings
// ===========================================================================

TEST_CASE("rank_pool honours the species budget", "[selection][rank_pool]") {
    constexpr int kPlots = 6;
    constexpr int kBudget = 3;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    SECTION("no budget encodes every species, whatever the selection says") {
        for (auto sel : {SelectionMode::Top, SelectionMode::Bottom,
                         SelectionMode::TopBottom, SelectionMode::All}) {
            auto cfg = base_config(SpeciesEncodingMode::RankPool);
            cfg.selection = sel;
            cfg.species_budget = 0;
            auto ds = build(header.path(), species.path(), cfg);

            for (int64_t p = 0; p < kPlots; ++p) {
                REQUIRE(encoded_species(ds, p).size() == static_cast<size_t>(kSpecies));
            }
            // ... and says so, rather than reporting a selection it did not make.
            REQUIRE(ds.schema().selection == SelectionMode::All);
        }
    }

    SECTION("Top keeps exactly the most abundant species") {
        auto cfg = base_config(SpeciesEncodingMode::RankPool);
        cfg.selection = SelectionMode::Top;
        cfg.species_budget = kBudget;
        auto ds = build(header.path(), species.path(), cfg);

        REQUIRE(ds.schema().selection == SelectionMode::Top);
        REQUIRE(ds.schema().species_budget == kBudget);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) ==
                    expected_end(static_cast<int>(p), kBudget, /*top=*/true));
        }
    }

    SECTION("Bottom keeps exactly the least abundant species") {
        auto cfg = base_config(SpeciesEncodingMode::RankPool);
        cfg.selection = SelectionMode::Bottom;
        cfg.species_budget = kBudget;
        auto ds = build(header.path(), species.path(), cfg);

        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) ==
                    expected_end(static_cast<int>(p), kBudget, /*top=*/false));
        }
    }

    SECTION("Top and Bottom encode different assemblages") {
        auto top_cfg = base_config(SpeciesEncodingMode::RankPool);
        top_cfg.selection = SelectionMode::Top;
        top_cfg.species_budget = kBudget;
        auto bottom_cfg = top_cfg;
        bottom_cfg.selection = SelectionMode::Bottom;

        auto top = build(header.path(), species.path(), top_cfg);
        auto bottom = build(header.path(), species.path(), bottom_cfg);

        bool any_difference = false;
        for (int64_t p = 0; p < kPlots; ++p) {
            if (encoded_species(top, p) != encoded_species(bottom, p)) any_difference = true;
        }
        REQUIRE(any_difference);

        // The vocabulary is fitted over every record either way, so the two arms
        // of an ablation share one integer-code namespace and stay comparable.
        REQUIRE(top.species_vocab() == bottom.species_vocab());
        REQUIRE(top.schema().n_species_vocab == bottom.schema().n_species_vocab);
    }

    SECTION("the padded width follows the budget") {
        auto cfg = base_config(SpeciesEncodingMode::RankPool);
        cfg.selection = SelectionMode::Top;
        cfg.species_budget = kBudget;
        auto ds = build(header.path(), species.path(), cfg);
        REQUIRE(ds.species_ids().size(1) == kBudget);

        cfg.species_budget = 0;
        auto full = build(header.path(), species.path(), cfg);
        REQUIRE(full.species_ids().size(1) == kSpecies);
    }
}

TEST_CASE("transformer honours the species budget", "[selection][transformer]") {
    constexpr int kPlots = 5;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    auto cfg = base_config(SpeciesEncodingMode::Transformer);
    cfg.selection = SelectionMode::Bottom;
    cfg.species_budget = 2;
    auto ds = build(header.path(), species.path(), cfg);

    for (int64_t p = 0; p < kPlots; ++p) {
        REQUIRE(encoded_species(ds, p) == expected_end(static_cast<int>(p), 2, /*top=*/false));
    }
}

TEST_CASE("the species cap slices what the budget left", "[selection][rank_pool]") {
    // Selection narrows to the budget; the cap then slices the survivors in CSV
    // row order, which is what pool_species_cap documents. With budget 4 and cap
    // 2 the row holds two of the four most abundant species, and both are drawn
    // from that set rather than from the plot at large.
    constexpr int kPlots = 4;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    auto cfg = base_config(SpeciesEncodingMode::RankPool);
    cfg.selection = SelectionMode::Top;
    cfg.species_budget = 4;
    cfg.pool_species_cap = 2;
    auto ds = build(header.path(), species.path(), cfg);

    REQUIRE(ds.species_ids().size(1) == 2);
    for (int64_t p = 0; p < kPlots; ++p) {
        const auto kept = encoded_species(ds, p);
        REQUIRE(kept.size() == 2);
        const auto budgeted = expected_end(static_cast<int>(p), 4, /*top=*/true);
        for (const auto& name : kept) {
            REQUIRE(budgeted.count(name) == 1);
        }
    }
}

// ===========================================================================
// Sparse
// ===========================================================================

TEST_CASE("sparse honours the species budget", "[selection][sparse]") {
    constexpr int kPlots = 5;
    constexpr int kBudget = 3;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    SECTION("no budget writes every species") {
        auto cfg = base_config(SpeciesEncodingMode::Sparse);
        cfg.selection = SelectionMode::Bottom;
        cfg.species_budget = 0;
        auto ds = build(header.path(), species.path(), cfg);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p).size() == static_cast<size_t>(kSpecies));
        }
        REQUIRE(ds.schema().selection == SelectionMode::All);
    }

    SECTION("a budget leaves the unselected columns at zero") {
        auto cfg = base_config(SpeciesEncodingMode::Sparse);
        cfg.selection = SelectionMode::Top;
        cfg.species_budget = kBudget;
        auto ds = build(header.path(), species.path(), cfg);

        // The vector is still vocabulary-wide: only which columns carry a value
        // moves, so the column meaning is unchanged.
        REQUIRE(ds.species_vector().size(1) == ds.schema().n_species_vocab);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) ==
                    expected_end(static_cast<int>(p), kBudget, /*top=*/true));
        }
    }
}

// ===========================================================================
// Embed
// ===========================================================================

TEST_CASE("embed honours the configured selection", "[selection][embed]") {
    constexpr int kPlots = 5;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    SECTION("Top fills the slots with the most abundant species") {
        auto cfg = base_config(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::Top;
        cfg.top_k_species = 3;
        auto ds = build(header.path(), species.path(), cfg);
        REQUIRE(ds.species_ids().size(1) == 3);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) == expected_end(static_cast<int>(p), 3, true));
        }
    }

    SECTION("Bottom is no longer silently treated as Top") {
        auto cfg = base_config(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::Bottom;
        cfg.top_k_species = 3;
        auto ds = build(header.path(), species.path(), cfg);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) == expected_end(static_cast<int>(p), 3, false));
        }
    }

    SECTION("TopBottom draws from both ends into the same slot count") {
        auto cfg = base_config(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::TopBottom;
        cfg.top_k_species = 4;
        auto ds = build(header.path(), species.path(), cfg);

        // The width is the slot count, not twice it: the model's embed encoder
        // is sized from top_k_species and must keep matching.
        REQUIRE(ds.species_ids().size(1) == 4);
        for (int64_t p = 0; p < kPlots; ++p) {
            const auto kept = encoded_species(ds, p);
            REQUIRE(kept.size() == 4);
            const auto top2 = expected_end(static_cast<int>(p), 2, true);
            const auto bottom2 = expected_end(static_cast<int>(p), 2, false);
            for (const auto& name : top2) REQUIRE(kept.count(name) == 1);
            for (const auto& name : bottom2) REQUIRE(kept.count(name) == 1);
        }
    }

    SECTION("All is rejected rather than quietly meaning Top") {
        auto cfg = base_config(SpeciesEncodingMode::Embed);
        cfg.selection = SelectionMode::All;
        REQUIRE_THROWS_AS(build(header.path(), species.path(), cfg), std::invalid_argument);
    }
}

// ===========================================================================
// Hash is unchanged
// ===========================================================================

TEST_CASE("hash still selects on top_k and ignores the budget", "[selection][hash]") {
    constexpr int kPlots = 5;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    auto cfg = base_config(SpeciesEncodingMode::Hash);
    cfg.selection = SelectionMode::Top;
    cfg.top_k = 2;

    auto without_budget = build(header.path(), species.path(), cfg);
    cfg.species_budget = 5;  // not hash's knob
    auto with_budget = build(header.path(), species.path(), cfg);

    REQUIRE(torch::allclose(without_budget.hash_embedding(), with_budget.hash_embedding()));
    REQUIRE(without_budget.schema().selection == SelectionMode::Top);

    // Different top_k really does change the hashed content, so the comparison
    // above is not comparing two constant tensors.
    cfg.species_budget = 0;
    cfg.top_k = 5;
    auto wider = build(header.path(), species.path(), cfg);
    REQUIRE(!torch::allclose(without_budget.hash_embedding(), wider.hash_embedding()));
}

// ===========================================================================
// Round trip
// ===========================================================================

TEST_CASE("species_budget survives the checkpoint", "[selection][checkpoint]") {
    constexpr int kPlots = 24;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    auto cfg = base_config(SpeciesEncodingMode::RankPool);
    cfg.selection = SelectionMode::Bottom;
    cfg.species_budget = 3;
    auto ds = build(header.path(), species.path(), cfg);

    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::RankPool;
    model_config.hidden_dims = {16};
    model_config.species_embed_dim = 8;

    TrainConfig train_config;
    train_config.max_epochs = 1;
    train_config.batch_size = 8;
    train_config.device = torch::kCPU;

    ResolveModel model(ds.schema(), model_config);
    Trainer trainer(model, train_config);
    trainer.prepare_data(ds);
    trainer.fit();

    TempFile checkpoint("", ".pt");
    trainer.save(checkpoint.path());

    auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);
    const auto& loaded_schema = predictor.schema();
    REQUIRE(loaded_schema.species_budget == 3);
    REQUIRE(loaded_schema.selection == SelectionMode::Bottom);

    auto rebuilt = dataset_config_from_checkpoint(loaded_schema, model_config);
    REQUIRE(rebuilt.species_budget == 3);
    REQUIRE(rebuilt.selection == SelectionMode::Bottom);

    // Rebuilding the loader config from the checkpoint reproduces the same
    // encoded assemblage, which is the point of persisting the knob.
    rebuilt.pool_species_cap = 0;
    auto rescored = build(header.path(), species.path(), rebuilt);
    for (int64_t p = 0; p < kPlots; ++p) {
        REQUIRE(encoded_species(rescored, p) == encoded_species(ds, p));
    }
}

TEST_CASE("a checkpoint without the budget key reads back as no budget",
          "[selection][checkpoint]") {
    // Back-compat: species_budget is absent from every checkpoint written before
    // issue #113, and the schema default 0 is exactly what those runs did.
    ResolveSchema schema;
    REQUIRE(schema.species_budget == 0);

    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::RankPool;
    auto cfg = dataset_config_from_checkpoint(schema, model_config);
    REQUIRE(cfg.species_budget == 0);
}
