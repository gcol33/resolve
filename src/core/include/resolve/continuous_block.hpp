#pragma once

#include "resolve/aggregate_arity.hpp"
#include "resolve/scalers.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <utility>
#include <vector>

namespace resolve {

// The per-plot numeric inputs the model reads as one continuous block, before
// the categorical embeddings are fused onto it. A missing coordinate or
// covariate is NaN; every tensor is optional (undefined or empty when the data
// carries no such input). The member name is also the keyword the bindings and
// the C ABI value map use.
#define RESOLVE_CONTINUOUS_INPUT_FIELDS(F) \
    F(coordinates)                         \
    F(covariates)                          \
    F(unknown_fraction)                    \
    F(unknown_count)                       \
    F(hash_embedding)

struct ContinuousInputs {
    torch::Tensor coordinates;       // (n, 2)
    torch::Tensor covariates;        // (n, n_covariates)
    torch::Tensor unknown_fraction;  // (n,)
    torch::Tensor unknown_count;     // (n,)
    torch::Tensor hash_embedding;    // (n, hash_dim); appended last when present

    // Calls f(name, tensor) for every field in list order.
    template <typename F>
    void for_each(F&& f) {
        visit(*this, f);
    }
    template <typename F>
    void for_each(F&& f) const {
        visit(*this, f);
    }

    // Rows [start, end) of every defined field.
    [[nodiscard]] ContinuousInputs slice(int64_t start, int64_t end) const {
        ContinuousInputs out = *this;
        out.for_each([&](const char*, torch::Tensor& t) {
            if (t.defined() && t.dim() > 0) t = t.slice(/*dim=*/0, start, end);
        });
        return out;
    }

    // The rows of the first non-empty field, 0 when every field is empty.
    [[nodiscard]] int64_t n_rows() const {
        int64_t n = -1;
        for_each([&](const char*, const torch::Tensor& t) {
            if (n < 0 && t.defined() && t.numel() > 0) n = t.size(0);
        });
        return n < 0 ? 0 : n;
    }

private:
    template <typename Self, typename F>
    static void visit(Self& self, F& f) {
#define RESOLVE_CONTINUOUS_INPUT_VISIT(member) f(#member, self.member);
        RESOLVE_CONTINUOUS_INPUT_FIELDS(RESOLVE_CONTINUOUS_INPUT_VISIT)
#undef RESOLVE_CONTINUOUS_INPUT_VISIT
    }
};

#define RESOLVE_CONTINUOUS_INPUT_COUNT(member) +1
static_assert((0 RESOLVE_CONTINUOUS_INPUT_FIELDS(RESOLVE_CONTINUOUS_INPUT_COUNT)) ==
                  registry_detail::aggregate_field_count<ContinuousInputs>(),
              "ContinuousInputs has a member with no row in "
              "RESOLVE_CONTINUOUS_INPUT_FIELDS (or a row with no member).");
#undef RESOLVE_CONTINUOUS_INPUT_COUNT

// Assemble the continuous block, column order:
//
//   coordinates (2) | coordinate flag (1) | covariates (k) | covariate flags (k)
//   | unknown fraction (1) | unknown count (1) | hash embedding (hash_dim)
//
// Under MissingValuePolicy::Indicate the value columns keep their NaN, which
// fit_continuous_scalers / standardize_continuous fill, and each flag column is
// 1.0 where its value is missing (the coordinate flag covers the pair). Under
// Zero there are no flag columns and a NaN is read as 0.0. The flag columns are
// present exactly when their value columns are, so the width agrees with
// ResolveSchema::missing_flag_width() for data built from that schema.
[[nodiscard]] torch::Tensor assemble_continuous(const ContinuousInputs& inputs,
                                                MissingValuePolicy policy,
                                                int64_t n_rows);

// The value each column of a block fills its missing cells with: the mean of
// that column's recorded (non-NaN) values, and 0.0 for a column with none.
[[nodiscard]] torch::Tensor continuous_column_fill(const torch::Tensor& rows);

// The scale to standardize by: the sample standard deviation, offset so a
// constant column does not divide by zero, and 1.0 where it cannot be
// estimated at all. A single row has no sample standard deviation -- torch
// returns NaN -- and dividing by that turns every standardized value, and from
// there every prediction, into NaN in silence. Used for the continuous block
// and for each regression target.
[[nodiscard]] torch::Tensor standardization_scale(const torch::Tensor& values,
                                                  int64_t dim = -1);

// Replace every missing cell with its column's fill, leaving recorded values
// untouched. This is the fill fit_continuous_scalers computes, without the
// standardization -- what a consumer with no fitted scalers needs in order to
// read a block whose missing cells are NaN, a pretext task above all.
[[nodiscard]] torch::Tensor fill_missing_continuous(const torch::Tensor& block);

// Fit the fill, mean and scale of every column on `fitting_rows`, a slice of an
// assembled block. The fill is the mean of a column's recorded (non-NaN) values,
// 0.0 when it has none; the mean and scale are taken after filling.
void fit_continuous_scalers(Scalers& scalers, const torch::Tensor& fitting_rows);

// Fill missing cells and standardise with fitted scalers. A scaler without a
// fill (a checkpoint written before missing values were filled) reads a NaN as
// 0.0, which is what such a model was trained on. A column that did not vary in
// the fitting rows (the unknown-species fraction, whose vocabulary is built on
// those rows; a flag whose covariate was never missing there) standardises to
// 0.0 whatever its value, since the model never saw it take another.
[[nodiscard]] torch::Tensor standardize_continuous(const torch::Tensor& block,
                                                   const Scalers& scalers);

// Undo standardize_continuous: rescale, then mark every cell whose flag says it
// was missing as NaN again, so a refit on a different set of rows (a
// cross-validation fold) fills from that fold's recorded values alone. A column
// that did not vary in the fitting rows comes back at its fitted mean.
[[nodiscard]] torch::Tensor unstandardize_continuous(const torch::Tensor& block,
                                                     const Scalers& scalers,
                                                     const ResolveSchema& schema);

// For each flag column of a schema's block, the value columns it marks and its
// own index. Empty under Zero.
[[nodiscard]] std::vector<std::pair<std::vector<int64_t>, int64_t>>
missing_flag_columns(const ResolveSchema& schema);

}  // namespace resolve
