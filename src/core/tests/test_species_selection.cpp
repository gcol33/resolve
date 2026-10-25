// DatasetConfig::selection in every species encoding.
//
// The contract these tests pin:
//
//   * Each encoding takes its per-plot budget from the knob that also fixes its
//     width -- top_k for hash, top_k_species for embed, and
//     DatasetConfig::species_budget for rank_pool / transformer / sparse.
//   * species_budget defaults to 0 = no budget, so a pooled or sparse dataset
//     encodes every species the plot records.
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
#include "resolve/sparse_rows.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/trainer.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace resolve;
using namespace resolve::test;

namespace {

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
        auto row = sparse_rows_to_dense(ds.species_vector())[plot_row];
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
    return ResolveDataset::from_csv(header_path, species_path, Roles().taxonomy(),
                                    targets(), cfg);
}

}  // namespace

// ===========================================================================
// The shared selection core
// ===========================================================================

namespace {

// One plot holding `species`, in the order given.
PlotAssemblages one_plot(const std::vector<std::pair<std::string, float>>& species) {
    std::vector<SpeciesRecord> records;
    for (const auto& [name, cover] : species) records.push_back({name, "", "", cover, "p"});
    return PlotAssemblages::build(records, {"p"});
}

std::vector<std::string> names_at(const PlotAssemblages& plots,
                                  const std::vector<size_t>& idx) {
    std::vector<std::string> out;
    for (size_t i : idx) out.push_back(plots.species_name(plots.species(0)[i]));
    return out;
}

using Names = std::vector<std::string>;

}  // namespace

TEST_CASE("selection_indices ranks and truncates", "[selection]") {
    const auto plots = one_plot({{"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}});

    SECTION("Top returns the k most abundant, most abundant first") {
        REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Top, 2)) ==
                Names{"c", "a"});
    }

    SECTION("Bottom returns the k least abundant, least abundant first") {
        REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Bottom, 2)) ==
                Names{"b", "d"});
    }

    SECTION("TopBottom lists the top end, then the bottom end") {
        REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::TopBottom, 2)) ==
                Names{"c", "a", "b", "d"});
    }

    SECTION("TopBottom over a short list keeps each species once") {
        // 4 species, k = 3 per end: the two ends overlap and a species must
        // not occupy two slots.
        REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::TopBottom, 3)) ==
                Names{"c", "a", "d", "b"});
    }

    SECTION("All ignores k and keeps plot order") {
        REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::All, 2)) ==
                Names{"a", "b", "c", "d"});
    }
}

TEST_CASE("selection_indices ranks a plot holding no more than k species",
          "[selection]") {
    // A plot shorter than the budget is still ranked: the slot order of the
    // fixed-width encodings means "most abundant first" in every plot.
    const auto plots = one_plot({{"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}});
    REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Top, 10)) ==
            Names{"c", "a", "b"});
    REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Top, 3)) ==
            Names{"c", "a", "b"});
    REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Bottom, 10)) ==
            Names{"b", "a", "c"});
}

TEST_CASE("species_budget_indices is a no-op without a budget", "[selection][budget]") {
    const auto plots = one_plot({{"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}});

    for (auto mode : {SelectionMode::Top, SelectionMode::Bottom,
                      SelectionMode::TopBottom, SelectionMode::All}) {
        for (int budget : {0, -1, -7}) {
            const auto idx = species_budget_indices(plots, 0, mode, budget);
            REQUIRE(idx == std::vector<size_t>{0, 1, 2, 3});
        }
    }

    // All names no rule for picking a subset, so a budget does not narrow it.
    REQUIRE(species_budget_indices(plots, 0, SelectionMode::All, 2).size() == 4);
}

TEST_CASE("species_budget_indices returns plot order", "[selection][budget]") {
    // The rank-pool species cap documents itself as a slice of plot order, so
    // the budget hands back the survivors in that order rather than in the
    // abundance ranking.
    const auto plots = one_plot({{"a", 3.0f}, {"b", 1.0f}, {"c", 5.0f}, {"d", 2.0f}});
    REQUIRE(species_budget_indices(plots, 0, SelectionMode::Top, 2) ==
            std::vector<size_t>{0, 2});
}

TEST_CASE("selection ranks by recorded abundance under presence/absence",
          "[selection][representation]") {
    std::vector<SpeciesRecord> records = {
        {"a", "", "", 3.0f, "p"}, {"b", "", "", 1.0f, "p"}, {"c", "", "", 5.0f, "p"}};
    const auto plots =
        PlotAssemblages::build(records, {"p"}, RepresentationMode::PresenceAbsence);
    for (float ranking : plots.ranking(0)) REQUIRE(plots.value(ranking) == 1.0f);
    REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Top, 2)) ==
            Names{"c", "a"});
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
            REQUIRE(ds.schema().dataset_config.selection == SelectionMode::All);
        }
    }

    SECTION("Top keeps exactly the most abundant species") {
        auto cfg = base_config(SpeciesEncodingMode::RankPool);
        cfg.selection = SelectionMode::Top;
        cfg.species_budget = kBudget;
        auto ds = build(header.path(), species.path(), cfg);

        REQUIRE(ds.schema().dataset_config.selection == SelectionMode::Top);
        REQUIRE(ds.schema().dataset_config.species_budget == kBudget);
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

TEST_CASE("the species cap keeps the most abundant of what the budget left",
          "[selection][rank_pool]") {
    // Selection narrows to the budget; the cap then keeps the most abundant
    // survivors. With budget 4 and cap 2 the row holds the two most abundant
    // species of the plot.
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
        REQUIRE(encoded_species(ds, p) == expected_end(static_cast<int>(p), 2, /*top=*/true));
    }
}

TEST_CASE("the species cap keeps the most abundant species whatever the record order",
          "[selection][rank_pool]") {
    // Every plot records its species in the same CSV order while the most
    // abundant one rotates, so a cap that kept the first records would match
    // the ranking on the first plot only.
    constexpr int kPlots = 4;
    TempFile header(header_csv(kPlots));
    TempFile species(species_csv(kPlots));

    for (auto mode : {SpeciesEncodingMode::RankPool, SpeciesEncodingMode::Transformer}) {
        auto cfg = base_config(mode);
        cfg.pool_species_cap = 3;
        auto ds = build(header.path(), species.path(), cfg);

        REQUIRE(ds.species_ids().size(1) == 3);
        for (int64_t p = 0; p < kPlots; ++p) {
            REQUIRE(encoded_species(ds, p) == expected_end(static_cast<int>(p), 3, /*top=*/true));
        }
    }
}

TEST_CASE("a record with an empty species name encodes as unknown", "[selection][vocab]") {
    const auto plots = PlotAssemblages::build(
        {{"sp_a", "", "", 5.0f, "p0"}, {"", "", "", 9.0f, "p0"}, {"", "", "", 2.0f, "p1"},
         {"sp_b", "", "", 1.0f, "p1"}},
        {"p0", "p1"});
    const auto vocab = SpeciesVocab::fit(plots, /*min_count=*/1);

    REQUIRE(vocab.size() == 3);
    REQUIRE(vocab.encode("") == 0);
    REQUIRE(vocab.encode("sp_a") > 0);
    REQUIRE(vocab.encode("sp_b") > 0);
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
        REQUIRE(ds.schema().dataset_config.selection == SelectionMode::All);
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

    SECTION("Bottom is not treated as Top") {
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
    REQUIRE(without_budget.schema().dataset_config.selection == SelectionMode::Top);

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

    const TempPath checkpoint(".pt");
    trainer.save(checkpoint.path());

    auto predictor = Predictor::load(checkpoint.path(), torch::kCPU);
    const auto& loaded_schema = predictor.schema();
    REQUIRE(loaded_schema.dataset_config.species_budget == 3);
    REQUIRE(loaded_schema.dataset_config.selection == SelectionMode::Bottom);

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
    // Back-compat: a checkpoint without the species_budget key reads back as
    // the schema default 0, i.e. no budget.
    ResolveSchema schema;
    REQUIRE(schema.dataset_config.species_budget == 0);

    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::RankPool;
    auto cfg = dataset_config_from_checkpoint(schema, model_config);
    REQUIRE(cfg.species_budget == 0);
}

// ============================================================================
// Tie-breaking
// ============================================================================

// Abundance ties break by name, so the selected set does not depend on input
// (CSV row) order.
TEST_CASE("selection is order-independent on abundance ties", "[species_selection][determinism]") {
    // Five species, all equal abundance (presence/absence data): the top-3 must
    // be the three alphabetically-first names regardless of input order.
    const auto a = one_plot(
        {{"zeta", 1.0f}, {"alpha", 1.0f}, {"gamma", 1.0f}, {"beta", 1.0f}, {"delta", 1.0f}});
    const auto b = one_plot(
        {{"delta", 1.0f}, {"gamma", 1.0f}, {"beta", 1.0f}, {"alpha", 1.0f}, {"zeta", 1.0f}});

    const Names expected = {"alpha", "beta", "delta"};
    REQUIRE(names_at(a, selection_indices(a, 0, SelectionMode::Top, 3)) == expected);
    REQUIRE(names_at(b, selection_indices(b, 0, SelectionMode::Top, 3)) == expected);
}

// Mixed abundances still rank by abundance first; only genuine ties use the name.
TEST_CASE("selection ranks by abundance, name only on ties", "[species_selection][determinism]") {
    const auto plots = one_plot({{"low", 0.1f}, {"tieB", 2.0f}, {"tieA", 2.0f}, {"high", 5.0f}});
    REQUIRE(names_at(plots, selection_indices(plots, 0, SelectionMode::Top, 3)) ==
            Names{"high", "tieA", "tieB"});
}
