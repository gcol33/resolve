#pragma once

// Gap fitting: what a dataset's own complete plots add to the predictions for
// the plots that lack a value.
//
// A dataset usually records a variable (survey year, coordinates, altitude)
// on most of its plots and lacks it on some. The plots that carry it are
// labelled examples from the same surveys, recorded by the same people, as the
// plots whose value is missing (the gaps). Gap fitting runs after prediction:
// on the complete plots it chooses how to combine the model's raw prediction
// with the dataset's own neighbours, and applies that choice to the gaps. The
// raw prediction is never replaced; the fitted value is returned beside it,
// with a report of what was chosen and why.
//
// The candidates (GapMethod) are a nested chain plus two neighbour medians:
//
//   local_constant         the complete plots' centre: their mean (median on a
//                          log scale; mean direction of a bearing; geographic
//                          median of a position);
//   resolve                the raw prediction;
//   offset_slope           y = mean(y) + b (raw - mean(raw)), b by least squares;
//   knn_species            the median recorded value of the k complete plots
//                          most similar in species composition (Ochiai
//                          similarity of presence);
//   knn_spatial            the same over the k nearest complete plots by
//                          great-circle distance;
//   stack_species,         raw and the neighbour median(s) combined by least
//   stack_species_spatial  squares on the complete plots.
//
// k comes from a grid (1, 3, 5, 10, 20, 50), chosen by the out-of-fold error
// of the neighbour median on the complete plots.
//
// The choice is judged by cross-validation over the complete plots, with folds
// laid out to hold plots out the way the gaps are held out (k-fold
// nearest-neighbour distance matching, Linnenbrink, Mila, Ludwig & Meyer 2024,
// "kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for
// map accuracy estimation", Geoscientific Model Development 17, 5897-5912).
// Each gap's distance to its nearest complete plot is measured on every axis it
// can be measured on: species dissimilarity (one minus the similarity), km
// (when the gaps carry coordinates; never for a position target, whose gaps
// have none), survey years (when the gaps carry a year; a year target's gaps
// have none). The candidate layouts group the complete plots into squares of
// the equal-area plane, into survey-year periods, into both, or plot by plot,
// at a range of sizes, and deal the groups out to n_folds folds; a layout in
// which a fold holds more than max_fold_share of the complete plots is passed
// over (the plot-by-plot layout never is). Gaps farther from the complete
// plots than any fold of them lies from the rest are met by a buffer: plot by
// plot, a held-out plot also loses every plot nearer than a quartile of the
// gaps' distances in km, in years, or in both (buffered cross-validation;
// Roberts et al. 2017, "Cross-validation strategies for data with temporal,
// spatial, hierarchical, or phylogenetic structure", Ecography 40, 913-929).
// The layout chosen is the one whose held-out plots lie from the plots they
// may read as the gaps lie from the complete plots: the smallest sum over the
// axes of the Wasserstein distance between the two distributions of
// nearest-neighbour distances, each taken on the rank scale of the two pooled
// (the mean over the pooled distances of the absolute difference between the
// two empirical distribution functions), so every axis lies in [0, 1] and an
// axis on which the gaps all sit at one distance costs a layout only the share
// of its held-out plots moved off it.
//
// Every fitted part is fitted without the plot it is judged on: a held-out
// plot's neighbours come from the other folds, the least-squares coefficients
// are fitted on the other folds (on their out-of-fold neighbour terms), the
// local constant is the other folds' centre, and k is chosen by the
// out-of-fold error. A complete plot left with no neighbour in the other folds
// is judged as predicted by the local constant, and a gap with no neighbour
// gets it. Every candidate is scored by the target's headline error -- mean
// absolute error; on a log scale the mean absolute error of log1p; the mean
// circular error of a bearing; the median great-circle error in km of a
// position -- over the out-of-fold predictions (cv_error); the lowest is
// chosen, refitted on every complete plot and predicts the gaps. Gaps with and
// without coordinates choose apart, the second among the candidates that read
// no position.
//
// The lowest of several cross-validated errors reads lower than the error of
// the method it picks, so the error the whole procedure is reported with
// (estimated_error) repeats it with each fold held out in turn -- layout kept,
// k, candidate and coefficients chosen and fitted on the other folds -- and
// scores it on that fold (nested cross-validation; Varma & Simon 2006, "Bias in
// error estimation when using cross-validation for model selection", BMC
// Bioinformatics 7, 91). It needs at least three folds.
//
// Values are fitted in a plane in which an offset, a slope and a median mean
// something (GapScale): the value; log1p of it; a bearing as its (sine,
// cosine) unit vector, whose least-squares coefficients are complex numbers
// (a turn and a shrink); a position in km of a Lambert azimuthal equal-area
// projection centred on the complete plots' geographic median, one
// coefficient for both axes.
//
// Regression targets only. A classification has one fixed definition, so a
// dataset's own labelled plots must not shift what a plot is called.

#include "resolve/json.hpp"
#include "resolve/row_source.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

// The seed the fold layout deals its groups and draws its matching sample with
// by default.
inline constexpr int kGapMatchingSeed = 20261106;

struct GapFittingConfig {
    // The methods the choice is made among; empty is every method. A method
    // that reads spatial neighbours is a candidate only where the gaps carry
    // coordinates, so gaps without coordinates need at least one that reads
    // no position (fit_gaps throws otherwise).
    std::vector<GapMethod> candidates = {
        GapMethod::LocalConstant, GapMethod::Resolve,     GapMethod::OffsetSlope,
        GapMethod::KnnSpecies,    GapMethod::StackSpecies, GapMethod::KnnSpatial,
        GapMethod::StackSpeciesSpatial};
    // The neighbour counts k is chosen from (each >= 1); a k above the
    // complete plots available is left out.
    std::vector<int64_t> k_grid = {1, 3, 5, 10, 20, 50};
    // Spatial neighbours as candidates. Off reads composition only.
    bool spatial = true;
    // Seed of the fold layout: the order groups are dealt to folds in and the
    // complete plots its distances are measured on.
    int matching_seed = kGapMatchingSeed;
    // Folds of the cross-validation (at least 2; at most one per complete
    // plot). The error estimate of the whole procedure needs 3.
    int64_t n_folds = 10;
    // The largest share of the complete plots one fold of a candidate layout
    // may hold, in (1 / n_folds, 1].
    float max_fold_share = 0.5f;
    // Complete plots whose out-of-fold distances are compared with the gaps'
    // for each candidate layout (all of them when there are fewer).
    int64_t matching_sample = 1000;

    // Throws std::invalid_argument naming the first problem.
    void validate() const;
};

// The target a fit is for.
struct GapTarget {
    std::string name;
    GapScale scale = GapScale::Linear;
    // A bearing's full turn, in the target's units (Circular only).
    double period = 360.0;
};

// What gap fitting reads about the plots, one row per plot in the order of
// the prediction it fits.
struct GapPlots {
    // (n, n_species) species composition, dense or sparse rows; an entry
    // other than 0 marks the species present. Required.
    torch::Tensor species;
    // (n, 2) latitude and longitude in degrees, NaN where unknown. Undefined:
    // no plot has a position.
    torch::Tensor coordinates;
    // (n,) survey year, NaN where unknown. Undefined: no plot has one.
    torch::Tensor years;
};

// The gaps' distances to their nearest complete plot on one axis.
struct GapAxisSummary {
    std::string axis;       // "similarity", "km" or "years"
    bool matched = false;   // whether the fold layout was matched on it
    int64_t n = 0;          // gaps with a distance on this axis
    // Quantiles 0, 0.25, 0.5, 0.75 and 1 of those distances; NaN when n = 0.
    std::vector<double> quantiles;
};

// One least-squares coefficient: real on a linear, log or geographic scale;
// for a bearing a complex number, reported as its modulus and its angle.
struct GapCoefficient {
    std::string term;  // "resolve", "species" or "spatial"
    bool complex = false;
    double value = 0.0;      // real
    double modulus = 0.0;    // complex
    double angle_deg = 0.0;  // complex
};

// One neighbour kind's choice of k by the out-of-fold error.
struct GapNeighbourChoice {
    std::string kind;  // "species" or "spatial"
    int64_t k = 0;
    std::vector<std::pair<int64_t, double>> cv_error;  // per k of the grid
};

// How the complete plots were laid out into folds.
struct GapFoldLayout {
    int64_t n_folds = 0;
    // The side of the squares (km) and the length of the survey-year periods
    // the plots were grouped by; NaN where the layout does not group by it
    // (both NaN: plot by plot). Injected folds report both NaN.
    double block_km = std::numeric_limits<double>::quiet_NaN();
    double block_years = std::numeric_limits<double>::quiet_NaN();
    // The buffer within which a held-out plot loses the other plots, in km
    // and in years; NaN for none.
    double buffer_km = std::numeric_limits<double>::quiet_NaN();
    double buffer_years = std::numeric_limits<double>::quiet_NaN();
    bool injected = false;
    // Per matched axis, the rank-scale Wasserstein distance between the held-out
    // plots' and the gaps' nearest-neighbour distances, and their sum.
    std::vector<std::pair<std::string, double>> mismatch;
    double total_mismatch = std::numeric_limits<double>::quiet_NaN();
    int64_t n_layouts = 0;         // candidate layouts compared
    int64_t n_matching_plots = 0;  // complete plots their distances were measured on
};

// The choice made for one group of gaps.
struct GapGroupChoice {
    std::string group;  // "located" (gaps with coordinates) or "unlocated"
    int64_t n_complete = 0;  // complete plots the choice was judged on
    int64_t n_gaps = 0;
    GapMethod method = GapMethod::Resolve;
    // Every candidate's headline error over the out-of-fold predictions of
    // those complete plots, in candidate order.
    std::vector<std::pair<GapMethod, double>> cv_error;
    // The headline error of the whole procedure (layout kept; k, candidate
    // and coefficients chosen on the other folds), over the complete plots it
    // was scored on; NaN with fewer than three folds.
    double estimated_error = std::numeric_limits<double>::quiet_NaN();
    int64_t n_estimated = 0;
    // The same error per held-out fold, over that fold's complete plots:
    // fold_error[f] (NaN where the fold scored none) over fold_n_estimated[f]
    // plots; empty with fewer than three folds. Their spread is how far the
    // error of one fold's plots, a region of a blocked layout, falls from the
    // pooled estimate.
    std::vector<double> fold_error = {};
    std::vector<int64_t> fold_n_estimated = {};
    // What the chosen method fitted on every complete plot: its coefficients
    // (least squares), the complete plots they were fitted on, and the k of
    // each neighbour term.
    std::vector<GapCoefficient> coefficients;
    int64_t n_fitted = 0;
    std::vector<std::pair<std::string, int64_t>> k;
};

struct GapFitReport {
    std::string target;
    GapScale scale = GapScale::Linear;
    int64_t n_complete = 0;
    int64_t n_gaps = 0;
    // Plots lacking the value whose raw prediction is not finite (a plot the
    // model did not impute): neither a gap nor a complete plot.
    int64_t n_not_imputed = 0;
    // Why nothing was chosen (fewer than two complete plots), empty otherwise.
    std::string skipped;
    std::vector<GapAxisSummary> axes;
    GapFoldLayout folds;
    std::vector<GapNeighbourChoice> neighbours;
    std::vector<GapGroupChoice> groups;
};

// The method code a complete plot carries in GapFit::method: it keeps its
// recorded value.
inline constexpr int64_t kGapRecorded = -1;
// The method code of a plot lacking the value whose raw prediction is not
// finite: it has no fitted value.
inline constexpr int64_t kGapNotImputed = -2;

// The plots that record a value: (n,) bool, true where every value of the
// plot's row of `recorded` ((n,), or (n, 2) latitude / longitude) is finite.
// The rest are the gaps.
[[nodiscard]] torch::Tensor recorded_plots(const torch::Tensor& recorded);

struct GapFit {
    // Per plot in the target's units, like the raw prediction: (n,) float64,
    // or (n, 2) latitude / longitude for a position. A plot recording the
    // value carries it, a gap the chosen method's value, a plot that is not
    // imputed NaN.
    torch::Tensor fitted;
    // (n,) int64: kGapRecorded for a plot recording the value, kGapNotImputed
    // for one lacking it without a raw prediction, the GapMethod (as its
    // integer) that predicted each gap otherwise.
    torch::Tensor method;
    GapFitReport report;
};

// "recorded" for kGapRecorded, "not_imputed" for kGapNotImputed, the
// GapMethod's name otherwise.
[[nodiscard]] std::string gap_method_label(int64_t code);

// The report as JSON, every number locale-free (number_text.hpp).
[[nodiscard]] json::Value gap_fit_report_to_json(const GapFitReport& report);

// Fit the gaps of one target. `raw` is the model's prediction and `recorded`
// the dataset's value per plot, both in the target's units: (n,) for a linear,
// log or circular scale, (n, 2) latitude / longitude in degrees for a
// geographic one. A plot whose recorded value is NaN (either coordinate of a
// position) is a gap, every other plot a complete one. A plot whose raw
// prediction is not finite (either coordinate of a position) was not imputed:
// it is neither a gap nor a complete plot, so no method reads it and it keeps
// its recorded value, or NaN with kGapNotImputed.
//
// `folds`, when given, replaces the matched layout: (n_complete,) int64, the
// fold of each complete plot (recorded value and raw prediction finite) in
// plot order, numbered from 0 with every fold
// from 0 to its largest number holding a plot, so a run reproduces another
// implementation's layout.
//
// With no gap, the fit returns the recorded values. With fewer than two
// complete plots nothing can be chosen: the gaps keep the raw prediction and
// the report says why. Throws std::invalid_argument for shapes that do not
// match or an invalid config.
[[nodiscard]] GapFit fit_gaps(const GapTarget& target,
                              const torch::Tensor& raw,
                              const torch::Tensor& recorded,
                              const GapPlots& plots,
                              const GapFittingConfig& config = {},
                              const torch::Tensor* folds = nullptr);

// The fold layout fit_gaps matches to the gaps, for inspecting what a fit is
// judged on.
struct GapFolds {
    torch::Tensor complete;  // (n_complete,) int64 plot rows of the complete plots
    torch::Tensor gaps;      // (n_gaps,) int64 plot rows of the gaps
    torch::Tensor folds;     // (n_complete,) int64 fold of each complete plot
    GapFoldLayout layout;
};

// The layout fit_gaps reads for `recorded` (as in fit_gaps). Throws
// std::invalid_argument with fewer than two complete plots or no gap.
[[nodiscard]] GapFolds gap_folds(const GapTarget& target,
                                 const torch::Tensor& recorded,
                                 const GapPlots& plots,
                                 const GapFittingConfig& config = {});

// ---------------------------------------------------------------------------
// Reading what gap fitting needs from a dataset's own tables
// ---------------------------------------------------------------------------

// The columns of a dataset's tables gap fitting reads. Plot-level columns
// come from the header table, or without one from each plot's first species
// row.
struct GapColumns {
    std::string plot_id;
    std::string species_id;
    std::optional<std::string> latitude;     // with longitude: the spatial axis
    std::optional<std::string> longitude;
    std::optional<std::string> survey_year;  // the survey-year axis
    // Per target, the column(s) holding its recorded value: one, or latitude
    // and longitude for a position.
    std::vector<std::pair<std::string, std::vector<std::string>>> recorded;
};

struct GapTables {
    GapPlots plots;
    // Per target of GapColumns::recorded, its recorded values: (n,) or (n, 2),
    // NaN where missing.
    std::unordered_map<std::string, torch::Tensor> recorded;
};

// Read the species composition (every named species, whatever a model's
// vocabulary holds), the positions, the survey years and the recorded values
// of the plots `plot_ids`, in that order. Throws std::runtime_error for a
// named column the table lacks or a plot the tables do not hold.
[[nodiscard]] GapTables read_gap_tables(const std::optional<TableSource>& header,
                                        const TableSource& species,
                                        const GapColumns& columns,
                                        const std::vector<std::string>& plot_ids);

// One target of a single model to fit: its scale and the prediction head(s)
// it reads -- one; for a bearing one (in the target's units) or two (its sine,
// then its cosine); for a position two (latitude, then longitude).
struct GapFitRequest {
    GapTarget target;
    std::vector<std::string> outputs;
};

// Fit the gaps of every request from one model's predictions and the tables
// read for its plots (read_gap_tables over predictions.plot_ids), the tables'
// recorded values named by the request's target name.
[[nodiscard]] std::vector<GapFit> fit_gaps(const ResolvePredictions& predictions,
                                           const GapTables& tables,
                                           const std::vector<GapFitRequest>& requests,
                                           const GapFittingConfig& config = {});

}  // namespace resolve
