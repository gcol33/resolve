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
// k comes from a grid (1, 3, 5, 10, 20, 50), chosen by the error of the
// leave-one-out prediction of the complete plots.
//
// The choice is made at the distance the gaps actually are. Each gap's
// distance to its nearest complete plot is measured on every axis it can be
// measured on: species similarity, km (when the gaps carry coordinates; never
// for a position target, whose gaps have none), survey years (when the gaps
// carry a year; a year target's gaps have none). One distance per axis is
// drawn per complete plot from the gaps' distances, and each complete plot is
// predicted with the complete plots nearer to it than that hidden
// (nearest-neighbour distance matching, Mila et al. 2022, "Nearest neighbour
// distance matching Leave-One-Out Cross-Validation for map validation",
// Methods in Ecology and Evolution). A complete plot left with no
// neighbour at that distance is judged as predicted by the local constant.
// Every candidate is scored on the complete plots by the target's headline
// error -- mean absolute error; on a log scale the mean absolute error of
// log1p; the mean circular error of a bearing; the median great-circle error
// in km of a position -- and the best predicts the gaps. Gaps with and
// without coordinates choose apart, the second among the candidates that read
// no position.
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
#include "resolve/retrieval.hpp"
#include "resolve/row_source.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

// The seed the matched test draws its hiding distances with by default.
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
    // Seed of the draw of one hiding distance per complete plot and axis.
    int matching_seed = kGapMatchingSeed;

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
    bool matched = false;   // whether the matched test hid complete plots by it
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

// One neighbour kind's leave-one-out choice of k, at the gaps' distance.
struct GapNeighbourChoice {
    std::string kind;  // "species" or "spatial"
    int64_t k = 0;
    std::vector<std::pair<int64_t, double>> matched_error;  // per k of the grid
};

// The choice made for one group of gaps.
struct GapGroupChoice {
    std::string group;  // "located" (gaps with coordinates) or "unlocated"
    int64_t n_complete = 0;  // complete plots the choice was judged on
    int64_t n_gaps = 0;
    GapMethod method = GapMethod::Resolve;
    // Every candidate's headline error on those complete plots, in candidate
    // order, at the gaps' distance.
    std::vector<std::pair<GapMethod, double>> matched_error;
    // What the chosen method fitted: its coefficients (least squares), the
    // complete plots they were fitted on, and the k of each neighbour term.
    std::vector<GapCoefficient> coefficients;
    int64_t n_fitted = 0;
    std::vector<std::pair<std::string, int64_t>> k;
};

struct GapFitReport {
    std::string target;
    GapScale scale = GapScale::Linear;
    int64_t n_complete = 0;
    int64_t n_gaps = 0;
    // Why nothing was chosen (fewer than two complete plots), empty otherwise.
    std::string skipped;
    std::vector<GapAxisSummary> axes;
    std::vector<GapNeighbourChoice> neighbours;
    std::vector<GapGroupChoice> groups;
};

// The method code a complete plot carries in GapFit::method: it keeps its
// recorded value.
inline constexpr int64_t kGapRecorded = -1;

struct GapFit {
    // Per plot in the target's units, like the raw prediction: (n,) float64,
    // or (n, 2) latitude / longitude for a position. A complete plot carries
    // its recorded value, a gap the chosen method's value.
    torch::Tensor fitted;
    // (n,) int64: kGapRecorded for a complete plot, the GapMethod (as its
    // integer) that predicted each gap otherwise.
    torch::Tensor method;
    GapFitReport report;
};

// "recorded" for kGapRecorded, the GapMethod's name otherwise.
[[nodiscard]] std::string gap_method_label(int64_t code);

// The report as JSON, every number locale-free (number_text.hpp).
[[nodiscard]] json::Value gap_fit_report_to_json(const GapFitReport& report);

// Fit the gaps of one target. `raw` is the model's prediction and `recorded`
// the dataset's value per plot, both in the target's units: (n,) for a linear,
// log or circular scale, (n, 2) latitude / longitude in degrees for a
// geographic one. A plot whose recorded value is NaN (either coordinate of a
// position) is a gap, every other plot a complete one.
//
// `radii`, when given, replaces the drawn hiding distances: one per complete
// plot (in plot order) on each axis the matched test reads (undefined
// tensors for the others), so a run reproduces another implementation's draw.
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
                              const HidingRadii* radii = nullptr);

// The neighbours the matched test reads, for inspecting what a fit is judged
// on. Indices point into `complete`; a row lists its neighbours first and -1
// after them.
struct MatchedGapNeighbours {
    torch::Tensor complete;  // (n_complete,) int64 plot rows of the complete plots
    torch::Tensor gaps;      // (n_gaps,) int64 plot rows of the gaps
    HidingRadii radii;       // the distances drawn, one per complete plot and axis
    torch::Tensor species;   // (n_complete, k) by species similarity
    torch::Tensor spatial;   // (n_complete, k) by great-circle distance; undefined
                             // without the spatial axis
};

// The matched neighbours fit_gaps reads for `recorded` (as in fit_gaps), with
// the distances drawn with the config's seed or, given `radii`, those. Throws
// std::invalid_argument with fewer than two complete plots or no gap.
[[nodiscard]] MatchedGapNeighbours matched_gap_neighbours(const GapTarget& target,
                                                          const torch::Tensor& recorded,
                                                          const GapPlots& plots,
                                                          const GapFittingConfig& config = {},
                                                          const HidingRadii* radii = nullptr);

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
