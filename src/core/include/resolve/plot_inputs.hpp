#pragma once

// The per-plot tensors one model forward reads, as one value.
//
// Every consumer that moves model inputs around -- the trainer's split, batches,
// device mirrors and cross-validation folds, the predictor's chunks, the C ABI
// and the Python bindings -- holds a PlotInputs and goes through the operations
// below, which visit the field list RESOLVE_PLOT_INPUT_FIELDS. A new model input
// is one member plus one row; the static_assert after the struct fails the build
// when the two disagree.

#include "resolve/aggregate_arity.hpp"
#include "resolve/sparse_rows.hpp"

#include <torch/torch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

// How a field stores its rows: a dense tensor, or sparse rows
// (sparse_rows.hpp), the layout of a matrix that is almost entirely zeros.
enum class FieldLayout { Dense, SparseRows };

// One row of the field list: the member name, which is also the keyword the
// bindings and the C ABI value map use, and the dtype and layout the model
// expects.
struct PlotInputField {
    const char* name;
    torch::ScalarType dtype;
    FieldLayout layout;
};

// Order is the positional order of the Python model methods.
#define RESOLVE_PLOT_INPUT_FIELDS(F)                \
    F(continuous,      torch::kFloat32, Dense)      \
    F(genus_ids,       torch::kInt64,   Dense)      \
    F(family_ids,      torch::kInt64,   Dense)      \
    F(species_ids,     torch::kInt64,   Dense)      \
    F(species_vector,  torch::kFloat32, SparseRows) \
    F(pool_genus_ids,  torch::kInt64,   Dense)      \
    F(pool_family_ids, torch::kInt64,   Dense)      \
    F(pool_weights,    torch::kFloat32, Dense)      \
    F(pool_mask,       torch::kBool,    Dense)      \
    F(pool_has_cover,  torch::kFloat32, Dense)      \
    F(categorical_ids, torch::kInt64,   Dense)

// `t` in the dtype and layout of `field`: a dense matrix given for a
// sparse-rows field converts to sparse rows. An undefined tensor, and a dense
// one with no elements (the loaders' and bindings' "absent"), keep their
// layout.
[[nodiscard]] inline torch::Tensor conform_to_field(const PlotInputField& field,
                                                    const torch::Tensor& t) {
    if (!t.defined()) return t;
    torch::Tensor out = t;
    if (field.layout == FieldLayout::SparseRows && !is_sparse_rows(out) && out.numel() > 0) {
        out = to_sparse_rows(out);
    }
    return out.scalar_type() == field.dtype ? out : out.to(field.dtype);
}

// Row i of every defined field belongs to plot i. A field the model does not
// read for its encoding stays undefined, and every operation below carries an
// undefined field through as undefined.
struct PlotInputs {
    // (n, n_continuous): the standardised continuous block
    // (continuous_block.hpp). In hash mode the hash embedding is its tail.
    torch::Tensor continuous{};
    // (n, n_taxonomy_slots) genus / family codes of the fixed taxonomy slots.
    torch::Tensor genus_ids{};
    torch::Tensor family_ids{};
    // (n, k) species codes, 0 = padding: the top_k_species slots of embed
    // mode, or one entry per pooled species for rank_pool / transformer.
    torch::Tensor species_ids{};
    // (n, n_species_vocab) explicit species vector of the sparse encoding, as
    // sparse rows: each plot's species codes and values.
    torch::Tensor species_vector{};
    // rank_pool / transformer, aligned with species_ids: (n, max_species)
    // genus and family code, weight and validity of each pooled species, and
    // (n,) whether the plot records cover at all.
    torch::Tensor pool_genus_ids{};
    torch::Tensor pool_family_ids{};
    torch::Tensor pool_weights{};
    torch::Tensor pool_mask{};
    torch::Tensor pool_has_cover{};
    // (n, n_categoricals) codes from CategoricalVocab, 0 = UNK.
    torch::Tensor categorical_ids{};

    // Calls f(const PlotInputField&, tensor) for every field in list order.
    template <typename F>
    void for_each(F&& f) {
        visit(*this, f);
    }
    template <typename F>
    void for_each(F&& f) const {
        visit(*this, f);
    }

    // A copy with fn(tensor) applied to every defined field.
    template <typename F>
    [[nodiscard]] PlotInputs map(F&& fn) const {
        PlotInputs out = *this;
        out.for_each([&](const PlotInputField&, torch::Tensor& t) {
            if (t.defined()) t = fn(t);
        });
        return out;
    }

    // The rows `rows` (int64 indices, on any device) of every field.
    [[nodiscard]] PlotInputs index_select(const torch::Tensor& rows) const {
        return map([&](const torch::Tensor& t) { return select_rows(t, rows); });
    }

    // Rows [start, end) of every field.
    [[nodiscard]] PlotInputs slice(int64_t start, int64_t end) const {
        return map([&](const torch::Tensor& t) {
            return t.dim() == 0 ? t : slice_rows(t, start, end);
        });
    }

    [[nodiscard]] PlotInputs to(const torch::Device& device) const {
        return map([&](const torch::Tensor& t) { return t.to(device); });
    }

    // The rows of every part, stacked in order. A field must be defined in all
    // parts or in none: the parts are pieces of one set of plots.
    [[nodiscard]] static PlotInputs cat(const std::vector<PlotInputs>& parts) {
        std::vector<std::vector<const torch::Tensor*>> fields(parts.size());
        for (std::size_t p = 0; p < parts.size(); ++p) {
            parts[p].for_each([&](const PlotInputField&, const torch::Tensor& t) {
                fields[p].push_back(&t);
            });
        }
        PlotInputs out;
        std::size_t i = 0;
        out.for_each([&](const PlotInputField& field, torch::Tensor& dst) {
            std::vector<torch::Tensor> pieces;
            pieces.reserve(parts.size());
            for (const auto& part_fields : fields) {
                if (part_fields[i]->defined()) pieces.push_back(*part_fields[i]);
            }
            ++i;
            if (pieces.empty()) return;
            if (pieces.size() != parts.size()) {
                throw std::invalid_argument(
                    std::string("PlotInputs::cat: field '") + field.name +
                    "' is defined in " + std::to_string(pieces.size()) + " of " +
                    std::to_string(parts.size()) + " parts");
            }
            dst = concat_rows(pieces);
        });
        return out;
    }

    // A copy with every defined field in its field's dtype and layout
    // (conform_to_field): the bundle a binding builds from caller tensors.
    [[nodiscard]] PlotInputs conformed() const {
        PlotInputs out = *this;
        out.for_each([](const PlotInputField& field, torch::Tensor& t) {
            t = conform_to_field(field, t);
        });
        return out;
    }

    // The number of plots: the rows of the first defined field, 0 when none is.
    [[nodiscard]] int64_t n_plots() const {
        int64_t n = -1;
        for_each([&](const PlotInputField&, const torch::Tensor& t) {
            if (n < 0 && t.defined() && t.dim() > 0) n = t.size(0);
        });
        return n < 0 ? 0 : n;
    }

    // A copy in which every defined field with no elements is undefined: an
    // empty tensor is how the loaders and the bindings spell "no such input".
    [[nodiscard]] PlotInputs without_empty() const {
        PlotInputs out = *this;
        out.for_each([](const PlotInputField&, torch::Tensor& t) {
            if (t.defined() && t.numel() == 0) t = torch::Tensor();
        });
        return out;
    }

private:
    template <typename Self, typename F>
    static void visit(Self& self, F& f) {
#define RESOLVE_PLOT_INPUT_VISIT(member, dtype, layout) \
        f(PlotInputField{#member, dtype, FieldLayout::layout}, self.member);
        RESOLVE_PLOT_INPUT_FIELDS(RESOLVE_PLOT_INPUT_VISIT)
#undef RESOLVE_PLOT_INPUT_VISIT
    }
};

#define RESOLVE_PLOT_INPUT_ROW(member, dtype, layout) \
    PlotInputField{#member, dtype, FieldLayout::layout},
inline constexpr std::array kPlotInputFields{
    RESOLVE_PLOT_INPUT_FIELDS(RESOLVE_PLOT_INPUT_ROW)
};
#undef RESOLVE_PLOT_INPUT_ROW

static_assert(kPlotInputFields.size() ==
                  registry_detail::aggregate_field_count<PlotInputs>(),
              "PlotInputs has a member with no row in RESOLVE_PLOT_INPUT_FIELDS "
              "(or a row with no member). The list drives every copy, slice, "
              "device move and binding of the model inputs.");

// A per-target tensor map (targets, or a forward's outputs) keyed by name, with
// the row operations PlotInputs has.
using TensorMap = std::unordered_map<std::string, torch::Tensor>;

[[nodiscard]] inline TensorMap index_select_rows(const TensorMap& m,
                                                 const torch::Tensor& rows) {
    TensorMap out;
    for (const auto& [name, t] : m) out[name] = t.index_select(0, rows.to(t.device()));
    return out;
}

[[nodiscard]] inline TensorMap to_device(const TensorMap& m, const torch::Device& device) {
    TensorMap out;
    for (const auto& [name, t] : m) out[name] = t.to(device);
    return out;
}

// The rows of every part stacked in order, for the keys of the first part; a
// key missing from a later part is an error.
[[nodiscard]] inline TensorMap cat_rows(const std::vector<TensorMap>& parts) {
    TensorMap out;
    if (parts.empty()) return out;
    for (const auto& [name, first] : parts.front()) {
        std::vector<torch::Tensor> pieces{first};
        for (std::size_t p = 1; p < parts.size(); ++p) {
            auto it = parts[p].find(name);
            if (it == parts[p].end()) {
                throw std::invalid_argument("cat_rows: '" + name +
                                            "' is missing from part " + std::to_string(p));
            }
            pieces.push_back(it->second);
        }
        out[name] = torch::cat(pieces, /*dim=*/0);
    }
    return out;
}

}  // namespace resolve
