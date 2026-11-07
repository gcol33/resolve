#include "resolve/gap_fitting.hpp"

#include "resolve/enum_names.hpp"
#include "resolve/geo.hpp"
#include "resolve/row_cells.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/suite.hpp"

#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kTwoPi = 2.0 * std::numbers::pi;

const auto kDouble = torch::TensorOptions().dtype(torch::kFloat64);
const auto kLong = torch::TensorOptions().dtype(torch::kInt64);

torch::Tensor as_double(const torch::Tensor& t) { return t.to(torch::kCPU, torch::kFloat64); }

// Rows of a (n, d) matrix where every value is finite.
torch::Tensor finite_rows(const torch::Tensor& rows) { return torch::isfinite(rows).all(1); }

// numpy's median: the middle value, or the mean of the two middle values of an
// even count; NaN for none.
double median_of(const torch::Tensor& values) {
    const torch::Tensor v = std::get<0>(as_double(values).reshape({-1}).sort());
    const int64_t n = v.size(0);
    if (n == 0) return kNaN;
    const auto a = v.accessor<double, 1>();
    return n % 2 == 1 ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

// ---------------------------------------------------------------------------
// The plane values are fitted in
// ---------------------------------------------------------------------------

// A target's values as rows of a plane in which an offset, a slope and a
// median mean something (GapScale).
class Plane {
public:
    // `complete` holds the complete plots' recorded values, from which a
    // geographic plane takes its centre.
    Plane(const GapTarget& target, const torch::Tensor& complete)
        : scale_(target.scale), period_(target.period) {
        if (scale_ == GapScale::Geographic) {
            const auto [lat, lon] = geographic_median(complete.select(1, 0), complete.select(1, 1));
            projection_.emplace(lat, lon);
        }
    }

    [[nodiscard]] GapScale scale() const noexcept { return scale_; }

    // Values in the target's units, (n,) or (n, 2), to (n, d) plane rows.
    [[nodiscard]] torch::Tensor to(const torch::Tensor& values) const {
        const torch::Tensor v = as_double(values);
        switch (scale_) {
            case GapScale::Linear: return v.unsqueeze(1);
            case GapScale::Log: return torch::log1p(torch::clamp_min(v, 0.0)).unsqueeze(1);
            case GapScale::Circular: {
                const torch::Tensor angle = v * (kTwoPi / period_);
                return torch::stack({torch::sin(angle), torch::cos(angle)}, 1);
            }
            case GapScale::Geographic: {
                const auto [east, north] = projection_->forward(v.select(1, 0), v.select(1, 1));
                return torch::stack({east, north}, 1) / 1000.0;
            }
        }
        throw std::logic_error("gap fitting: unknown scale");
    }

    // (n, d) plane rows back to values in the target's units.
    [[nodiscard]] torch::Tensor back(const torch::Tensor& rows) const {
        switch (scale_) {
            case GapScale::Linear: return rows.select(1, 0).clone();
            case GapScale::Log: return torch::expm1(rows.select(1, 0));
            case GapScale::Circular: {
                const torch::Tensor angle = torch::atan2(rows.select(1, 0), rows.select(1, 1));
                return torch::remainder(angle * (period_ / kTwoPi), period_);
            }
            case GapScale::Geographic: {
                const auto [lat, lon] =
                    projection_->inverse(rows.select(1, 0) * 1000.0, rows.select(1, 1) * 1000.0);
                return torch::stack({lat, lon}, 1);
            }
        }
        throw std::logic_error("gap fitting: unknown scale");
    }

    // Each plot's error in the quantity the target's headline aggregates:
    // the absolute error, of log1p on a log scale (a prediction floored at
    // zero), the shorter arc of a bearing, the great-circle km of a position.
    [[nodiscard]] torch::Tensor errors(const torch::Tensor& predicted, const torch::Tensor& truth) const {
        const torch::Tensor p = back(predicted);
        const torch::Tensor t = back(truth);
        switch (scale_) {
            case GapScale::Linear: return (p - t).abs();
            case GapScale::Log:
                return (torch::log1p(torch::clamp_min(p, 0.0)) - torch::log1p(t)).abs();
            case GapScale::Circular: {
                const torch::Tensor d = torch::remainder((p - t).abs(), period_);
                return torch::minimum(d, period_ - d);
            }
            case GapScale::Geographic:
                return great_circle_km(p.select(1, 0), p.select(1, 1), t.select(1, 0), t.select(1, 1));
        }
        throw std::logic_error("gap fitting: unknown scale");
    }

    // The headline over plots: the median km of a position, the mean otherwise.
    [[nodiscard]] double aggregate(const torch::Tensor& errors) const {
        if (scale_ == GapScale::Geographic) return median_of(errors);
        return errors.numel() == 0 ? kNaN : errors.mean().item<double>();
    }

    // The local constant in the plane: the geographic median (the plane's
    // origin), the mean direction of a bearing, the median on a log scale
    // (the constant with the lowest absolute error there), the mean otherwise.
    [[nodiscard]] torch::Tensor centre(const torch::Tensor& truth) const {
        switch (scale_) {
            case GapScale::Geographic: return torch::zeros({truth.size(1)}, kDouble);
            case GapScale::Log: return torch::full({1}, median_of(truth), kDouble);
            case GapScale::Linear:
            case GapScale::Circular: return truth.mean(0);
        }
        throw std::logic_error("gap fitting: unknown scale");
    }

private:
    GapScale scale_;
    double period_;
    std::optional<LambertAzimuthalEqualArea> projection_;
};

// ---------------------------------------------------------------------------
// Neighbour medians
// ---------------------------------------------------------------------------

// The first k neighbours' plane values summarised per row: the coordinate-wise
// median, or for a bearing their mean direction as a unit vector. A row lists
// its neighbours first and -1 after them; it is summarised over the ones it
// has, and is NaN without any.
torch::Tensor neighbour_median(const torch::Tensor& values, const torch::Tensor& neighbours,
                               int64_t k, bool circular) {
    const int64_t n = neighbours.size(0);
    const int64_t d = values.size(1);
    const torch::Tensor nb = neighbours.slice(1, 0, k);
    const torch::Tensor found = nb >= 0;
    const torch::Tensor n_found = found.sum(1);
    const torch::Tensor picked = values.index({nb.clamp_min(0)});  // (n, k, d)
    torch::Tensor out;
    if (circular) {
        const torch::Tensor sum = (picked * found.unsqueeze(2)).sum(1);
        const torch::Tensor mean = sum / n_found.clamp_min(1).unsqueeze(1);
        out = mean / mean.norm(2, {1}, /*keepdim=*/true).clamp_min(1e-12);
    } else {
        const torch::Tensor padded = torch::where(
            found.unsqueeze(2), picked, torch::full_like(picked, std::numeric_limits<double>::infinity()));
        const torch::Tensor sorted = std::get<0>(padded.sort(1));
        const torch::Tensor lo =
            torch::div((n_found - 1).clamp_min(0), 2, "floor").view({n, 1, 1}).expand({n, 1, d});
        const torch::Tensor hi =
            torch::div(n_found, 2, "floor").clamp_max(k - 1).view({n, 1, 1}).expand({n, 1, d});
        out = 0.5 * (sorted.gather(1, lo) + sorted.gather(1, hi)).squeeze(1);
    }
    return torch::where((n_found > 0).unsqueeze(1), out, torch::full_like(out, kNaN));
}

// ---------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------

struct LeastSquares {
    torch::Tensor cal;
    torch::Tensor test;
    torch::Tensor coefficients;  // (n_terms,) float64, or complex128 for a bearing
    int64_t n_fitted = 0;
};

// y = mean(y) + sum_j c_j (x_j - mean(x_j)) by least squares on the complete
// plots where every term is defined, applied to the complete plots and the
// gaps; NaN where a term is undefined. One coefficient per term for every axis
// of the plane: real for a linear, log or geographic plane, complex for a
// bearing, so it turns and shrinks the term's direction instead of moving its
// sine and cosine apart.
LeastSquares least_squares(const torch::Tensor& truth, const std::vector<torch::Tensor>& cal,
                           const std::vector<torch::Tensor>& test, bool circular) {
    auto defined = [](const std::vector<torch::Tensor>& xs) {
        torch::Tensor ok = finite_rows(xs.front());
        for (std::size_t j = 1; j < xs.size(); ++j) ok = ok & finite_rows(xs[j]);
        return ok;
    };
    auto lift = [&](const torch::Tensor& x) {
        if (!circular) return x;
        const torch::Tensor z = torch::nan_to_num(x, 0.0);
        return torch::complex(z.select(1, 1), z.select(1, 0)).unsqueeze(1);
    };

    const torch::Tensor rows = defined(cal);
    LeastSquares out;
    out.n_fitted = rows.sum().item<int64_t>();
    const auto n_terms = static_cast<int64_t>(cal.size());
    auto undefined_like = [&](const torch::Tensor& like) {
        return torch::full({like.size(0), truth.size(1)}, kNaN, kDouble);
    };
    if (out.n_fitted == 0) {
        out.cal = undefined_like(cal.front());
        out.test = undefined_like(test.front());
        out.coefficients = torch::full({n_terms}, kNaN, kDouble);
        return out;
    }
    const torch::Tensor y = lift(truth).index({rows});
    const torch::Tensor y_bar = y.mean(0);
    std::vector<torch::Tensor> x_bar;
    std::vector<torch::Tensor> columns;
    for (const torch::Tensor& x : cal) {
        const torch::Tensor xr = lift(x).index({rows});
        x_bar.push_back(xr.mean(0));
        columns.push_back((xr - x_bar.back()).reshape({-1}));
    }
    const torch::Tensor design = torch::stack(columns, 1);
    const torch::Tensor target = (y - y_bar).reshape({-1, 1});
    out.coefficients =
        std::get<0>(torch::linalg_lstsq(design, target, std::nullopt, "gelsd")).squeeze(1);

    auto apply = [&](const std::vector<torch::Tensor>& xs) {
        torch::Tensor value = y_bar.unsqueeze(0).expand({xs.front().size(0), y_bar.size(0)}).clone();
        for (int64_t j = 0; j < n_terms; ++j) {
            value = value + out.coefficients[j] * (lift(xs[static_cast<std::size_t>(j)]) - x_bar[static_cast<std::size_t>(j)]);
        }
        torch::Tensor rows_out =
            circular ? torch::stack({torch::imag(value.select(1, 0)), torch::real(value.select(1, 0))}, 1)
                     : value;
        rows_out = rows_out.to(torch::kFloat64).contiguous();
        return torch::where(defined(xs).unsqueeze(1), rows_out, torch::full_like(rows_out, kNaN));
    };
    out.cal = apply(cal);
    out.test = apply(test);
    return out;
}

GapCoefficient describe_coefficient(const std::string& term, const torch::Tensor& c) {
    GapCoefficient out;
    out.term = term;
    if (c.is_complex()) {
        out.complex = true;
        out.modulus = torch::abs(c).item<double>();
        out.angle_deg = torch::angle(c).item<double>() * 180.0 / std::numbers::pi;
    } else {
        out.value = c.item<double>();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------

// How a candidate turns its inputs into a value.
enum class Combination { Constant, Raw, Neighbours, LeastSquares };

// The neighbour kinds a term can read.
enum NeighbourKind : std::size_t { kSpecies = 0, kSpatial = 1, kNeighbourKinds = 2 };
constexpr const char* kKindNames[kNeighbourKinds] = {"species", "spatial"};

struct CandidateSpec {
    GapMethod method;
    Combination combination;
    bool reads[kNeighbourKinds];  // species, spatial
};

// Every candidate, in GapMethod order (the order ties are broken in).
constexpr CandidateSpec kCandidates[] = {
    {GapMethod::LocalConstant, Combination::Constant, {false, false}},
    {GapMethod::Resolve, Combination::Raw, {false, false}},
    {GapMethod::OffsetSlope, Combination::LeastSquares, {false, false}},
    {GapMethod::KnnSpecies, Combination::Neighbours, {true, false}},
    {GapMethod::StackSpecies, Combination::LeastSquares, {true, false}},
    {GapMethod::KnnSpatial, Combination::Neighbours, {false, true}},
    {GapMethod::StackSpeciesSpatial, Combination::LeastSquares, {true, true}},
};

const CandidateSpec& candidate_spec(GapMethod method) {
    for (const auto& spec : kCandidates) {
        if (spec.method == method) return spec;
    }
    throw std::invalid_argument("gap fitting: no candidate for method value " +
                                std::to_string(static_cast<int>(method)));
}

// One neighbour kind's median at the k the matched leave-one-out chose: on
// the complete plots (out of sample, at the gaps' distance) and on the gaps.
struct NeighbourTerm {
    torch::Tensor cal;
    torch::Tensor test;
    GapNeighbourChoice choice;
};

// A candidate's values in the plane and what it fitted.
struct CandidateFit {
    torch::Tensor cal;
    torch::Tensor test;
    std::vector<GapCoefficient> coefficients;
    int64_t n_fitted = 0;
    std::vector<std::pair<std::string, int64_t>> k;
};

// ---------------------------------------------------------------------------
// One target divided into complete plots and gaps, in the plane
// ---------------------------------------------------------------------------

struct Fold {
    Plane plane;
    torch::Tensor truth_cal;
    torch::Tensor raw_cal;
    torch::Tensor raw_test;
    torch::Tensor species_cal;
    torch::Tensor species_test;
    torch::Tensor coords_cal;  // (n_cal, 2), undefined without positions
    torch::Tensor coords_test;
    torch::Tensor years_cal;   // undefined without years
    torch::Tensor years_test;
    bool spatial_axis = false;
    bool time_axis = false;

    [[nodiscard]] int64_t n_cal() const { return truth_cal.size(0); }
    [[nodiscard]] int64_t n_test() const { return raw_test.size(0); }
    [[nodiscard]] bool circular() const { return plane.scale() == GapScale::Circular; }

    [[nodiscard]] int64_t k_max(NeighbourKind kind, const std::vector<int64_t>& grid) const {
        const int64_t usable = kind == kSpecies ? n_cal() : finite_rows(coords_cal).sum().item<int64_t>();
        return std::min(*std::max_element(grid.begin(), grid.end()), usable - 1);
    }
};

// Each gap's distance to its nearest complete plot on every axis.
struct GapDistances {
    torch::Tensor similarity;  // (n_test,)
    torch::Tensor km;          // undefined without the spatial axis
    torch::Tensor years;       // undefined without the time axis
};

// Each year's distance to the nearest reference year; NaN where a year is
// missing.
torch::Tensor nearest_year_gap(const torch::Tensor& years, const torch::Tensor& reference) {
    const torch::Tensor ref = std::get<0>(reference.index({torch::isfinite(reference)}).sort());
    torch::Tensor out = torch::full({years.size(0)}, kNaN, kDouble);
    const torch::Tensor ok = torch::isfinite(years);
    if (ref.size(0) == 0 || !ok.any().item<bool>()) return out;
    const torch::Tensor y = years.index({ok});
    const torch::Tensor pos = torch::searchsorted(ref, y);
    const int64_t last = ref.size(0) - 1;
    const torch::Tensor below = ref.index({(pos - 1).clamp(0, last)});
    const torch::Tensor above = ref.index({pos.clamp(0, last)});
    out.index_put_({ok}, torch::minimum((y - below).abs(), (y - above).abs()));
    return out;
}

GapDistances gap_distances(const Fold& fold) {
    GapDistances out;
    out.similarity = as_double(
        nearest_by_cosine(fold.species_test, fold.species_cal, 1).similarities.select(1, 0));
    if (fold.spatial_axis) {
        out.km = nearest_by_great_circle(fold.coords_test, fold.coords_cal, 1).km.select(1, 0);
    }
    if (fold.time_axis) out.years = nearest_year_gap(fold.years_test, fold.years_cal);
    return out;
}

// One distance per complete plot drawn from the gaps' known distances, zeros
// when none is known.
torch::Tensor drawn(const torch::Tensor& distances, int64_t n, at::Generator& generator) {
    const torch::Tensor known = distances.index({torch::isfinite(distances)});
    if (known.size(0) == 0) return torch::zeros({n}, kDouble);
    return known.index({torch::randint(0, known.size(0), {n}, generator, kLong)});
}

GapAxisSummary summarise_axis(const char* axis, const torch::Tensor& distances, bool matched) {
    GapAxisSummary out;
    out.axis = axis;
    out.matched = matched;
    if (!distances.defined()) {
        out.quantiles.assign(5, kNaN);
        return out;
    }
    const torch::Tensor known = distances.index({torch::isfinite(distances)});
    out.n = known.size(0);
    if (out.n == 0) {
        out.quantiles.assign(5, kNaN);
        return out;
    }
    const torch::Tensor q =
        torch::quantile(known, torch::tensor({0.0, 0.25, 0.5, 0.75, 1.0}, kDouble));
    const auto a = q.accessor<double, 1>();
    for (int64_t i = 0; i < 5; ++i) out.quantiles.push_back(a[i]);
    return out;
}

// Each complete plot's neighbours among the complete plots no nearer to it
// than its drawn gap distance on every axis: by species (always) and by
// great-circle distance (with the spatial axis).
std::vector<torch::Tensor> matched_neighbours(const Fold& fold, const HidingRadii& radii,
                                              const std::vector<int64_t>& grid) {
    NeighbourHiding hiding;
    hiding.queries = PlotAxes{fold.species_cal, fold.coords_cal, fold.years_cal};
    hiding.reference = hiding.queries;
    hiding.radii.similarity = radii.similarity;
    if (fold.spatial_axis) hiding.radii.km = radii.km;
    if (fold.time_axis) hiding.radii.years = radii.years;

    std::vector<torch::Tensor> out(kNeighbourKinds);
    const int64_t k_species = fold.k_max(kSpecies, grid);
    if (k_species >= 1) {
        out[kSpecies] = nearest_by_cosine(fold.species_cal, fold.species_cal, k_species,
                                          /*exclude_self=*/true, &hiding)
                            .indices;
    }
    if (fold.spatial_axis) {
        const int64_t k_spatial = fold.k_max(kSpatial, grid);
        if (k_spatial >= 1) {
            out[kSpatial] = nearest_by_great_circle(fold.coords_cal, fold.coords_cal, k_spatial,
                                                    /*exclude_self=*/true, &hiding)
                                .indices;
        }
    }
    return out;
}

// The gaps' neighbours among the complete plots, of one kind.
torch::Tensor gap_neighbours(const Fold& fold, NeighbourKind kind, int64_t k) {
    if (kind == kSpecies) return nearest_by_cosine(fold.species_test, fold.species_cal, k).indices;
    return nearest_by_great_circle(fold.coords_test, fold.coords_cal, k).indices;
}

// Choose k by the headline error on the complete plots that have matched
// neighbours; nullopt when the grid holds no k the neighbours allow.
std::optional<NeighbourTerm> neighbour_term(const Fold& fold, NeighbourKind kind,
                                            const torch::Tensor& matched,
                                            const std::vector<int64_t>& grid) {
    if (!matched.defined()) return std::nullopt;
    const int64_t width = matched.size(1);
    const bool circular = fold.circular();
    const torch::Tensor has = matched.select(1, 0) >= 0;
    const torch::Tensor matched_has = matched.index({has});
    const torch::Tensor truth_has = fold.truth_cal.index({has});
    NeighbourTerm term;
    term.choice.kind = kKindNames[kind];
    double best_error = kNaN;
    bool any = false;
    for (const int64_t k : grid) {
        if (k > width) continue;
        const double error =
            fold.plane.aggregate(fold.plane.errors(neighbour_median(fold.truth_cal, matched_has, k, circular),
                                                   truth_has));
        term.choice.matched_error.emplace_back(k, error);
        // The first k of the grid wins a tie, and stands when no complete
        // plot has a matched neighbour (every error NaN).
        if (!any || error < best_error) {
            best_error = error;
            term.choice.k = k;
        }
        any = true;
    }
    if (!any) return std::nullopt;
    term.cal = neighbour_median(fold.truth_cal, matched, term.choice.k, circular);
    term.test = neighbour_median(fold.truth_cal, gap_neighbours(fold, kind, width), term.choice.k,
                                 circular);
    return term;
}

CandidateFit fit_candidate(const Fold& fold, const CandidateSpec& spec, const torch::Tensor& centre,
                           const std::vector<std::optional<NeighbourTerm>>& terms) {
    CandidateFit fit;
    switch (spec.combination) {
        case Combination::Constant:
            fit.cal = centre.unsqueeze(0).expand({fold.n_cal(), centre.size(0)}).clone();
            fit.test = centre.unsqueeze(0).expand({fold.n_test(), centre.size(0)}).clone();
            return fit;
        case Combination::Raw:
            fit.cal = fold.raw_cal;
            fit.test = fold.raw_test;
            return fit;
        case Combination::Neighbours: {
            const std::size_t kind = spec.reads[kSpecies] ? kSpecies : kSpatial;
            const NeighbourTerm& term = *terms[kind];
            fit.cal = term.cal;
            fit.test = term.test;
            fit.k.emplace_back(kKindNames[kind], term.choice.k);
            return fit;
        }
        case Combination::LeastSquares: {
            std::vector<torch::Tensor> cal{fold.raw_cal};
            std::vector<torch::Tensor> test{fold.raw_test};
            std::vector<std::string> names{"resolve"};
            for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
                if (!spec.reads[kind]) continue;
                cal.push_back(terms[kind]->cal);
                test.push_back(terms[kind]->test);
                names.emplace_back(kKindNames[kind]);
                fit.k.emplace_back(kKindNames[kind], terms[kind]->choice.k);
            }
            const LeastSquares ls = least_squares(fold.truth_cal, cal, test, fold.circular());
            fit.cal = ls.cal;
            fit.test = ls.test;
            fit.n_fitted = ls.n_fitted;
            for (std::size_t j = 0; j < names.size(); ++j) {
                fit.coefficients.push_back(
                    describe_coefficient(names[j], ls.coefficients[static_cast<int64_t>(j)]));
            }
            return fit;
        }
    }
    throw std::logic_error("gap fitting: unknown combination");
}

int64_t rows_of(const torch::Tensor& t) { return t.defined() ? t.size(0) : -1; }

// The raw and recorded values as the scale lays them out: (n,) or (n, 2).
void require_values(const GapTarget& target, const torch::Tensor& t, const char* what, int64_t n) {
    const bool two = target.scale == GapScale::Geographic;
    const bool ok = t.defined() && t.size(0) == n &&
                    (two ? (t.dim() == 2 && t.size(1) == 2) : t.dim() == 1);
    if (!ok) {
        throw std::invalid_argument(
            "gap fitting: " + std::string(what) + " of target '" + target.name + "' must be " +
            (two ? "a (n_plots, 2) latitude / longitude matrix" : "a (n_plots,) vector") +
            " over the " + std::to_string(n) + " plots");
    }
}

// Presence of each species: an entry other than 0 counts once.
torch::Tensor presence_of(const torch::Tensor& species) {
    if (is_sparse_rows(species)) {
        return sparse_rows_with_values(species, (species.values() != 0).to(torch::kFloat32));
    }
    return (species != 0).to(torch::kFloat32);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

void GapFittingConfig::validate() const {
    if (k_grid.empty()) throw std::invalid_argument("gap fitting: k_grid holds no k");
    for (const int64_t k : k_grid) {
        if (k < 1) {
            throw std::invalid_argument("gap fitting: every k of k_grid must be at least 1, got " +
                                        std::to_string(k));
        }
    }
    for (const GapMethod method : candidates) (void)candidate_spec(method);
}

std::string gap_method_label(int64_t code) {
    if (code == kGapRecorded) return "recorded";
    return enum_to_name(static_cast<GapMethod>(code));
}

json::Value gap_fit_report_to_json(const GapFitReport& report) {
    json::Value out = json::Value::object();
    out.set("target", report.target);
    out.set("scale", enum_to_name(report.scale));
    out.set("n_complete", report.n_complete);
    out.set("n_gaps", report.n_gaps);
    if (!report.skipped.empty()) out.set("skipped", report.skipped);
    json::Value axes = json::Value::object();
    for (const auto& axis : report.axes) {
        json::Value a = json::Value::object();
        a.set("matched", axis.matched);
        a.set("n", axis.n);
        json::Value q = json::Value::object();
        const char* labels[] = {"min", "q25", "median", "q75", "max"};
        for (std::size_t i = 0; i < axis.quantiles.size() && i < 5; ++i) q.set(labels[i], axis.quantiles[i]);
        a.set("gap_distance", std::move(q));
        axes.set(axis.axis, std::move(a));
    }
    out.set("axes", std::move(axes));
    json::Value neighbours = json::Value::object();
    for (const auto& choice : report.neighbours) {
        json::Value c = json::Value::object();
        c.set("k", choice.k);
        json::Value errors = json::Value::object();
        for (const auto& [k, error] : choice.matched_error) errors.set(std::to_string(k), error);
        c.set("matched_error", std::move(errors));
        neighbours.set(choice.kind, std::move(c));
    }
    out.set("neighbours", std::move(neighbours));
    json::Value groups = json::Value::object();
    for (const auto& group : report.groups) {
        json::Value g = json::Value::object();
        g.set("method", enum_to_name(group.method));
        g.set("n_complete", group.n_complete);
        g.set("n_gaps", group.n_gaps);
        json::Value errors = json::Value::object();
        for (const auto& [method, error] : group.matched_error) errors.set(enum_to_name(method), error);
        g.set("matched_error", std::move(errors));
        if (!group.coefficients.empty()) {
            json::Value coefficients = json::Value::object();
            for (const auto& c : group.coefficients) {
                if (c.complex) {
                    json::Value z = json::Value::object();
                    z.set("modulus", c.modulus);
                    z.set("angle_deg", c.angle_deg);
                    coefficients.set(c.term, std::move(z));
                } else {
                    coefficients.set(c.term, c.value);
                }
            }
            g.set("coefficients", std::move(coefficients));
            g.set("n_fitted", group.n_fitted);
        }
        if (!group.k.empty()) {
            json::Value k = json::Value::object();
            for (const auto& [kind, value] : group.k) k.set(kind, value);
            g.set("k", std::move(k));
        }
        groups.set(group.group, std::move(g));
    }
    out.set("groups", std::move(groups));
    return out;
}

namespace {

void validate_inputs(const GapTarget& target, const torch::Tensor& raw, const torch::Tensor& recorded,
                     const GapPlots& plots) {
    if (target.scale == GapScale::Circular && !(std::isfinite(target.period) && target.period > 0.0)) {
        throw std::invalid_argument("gap fitting: target '" + target.name +
                                    "' is circular and needs a positive period");
    }
    if (!plots.species.defined() || plots.species.dim() != 2) {
        throw std::invalid_argument("gap fitting: the species composition must be a (n_plots, "
                                    "n_species) matrix");
    }
    const int64_t n = plots.species.size(0);
    if (raw.defined()) require_values(target, raw, "the raw prediction", n);
    require_values(target, recorded, "the recorded values", n);
    if (plots.coordinates.defined() &&
        (plots.coordinates.dim() != 2 || plots.coordinates.size(1) != 2 || rows_of(plots.coordinates) != n)) {
        throw std::invalid_argument("gap fitting: coordinates must be a (n_plots, 2) matrix");
    }
    if (plots.years.defined() && (plots.years.dim() != 1 || rows_of(plots.years) != n)) {
        throw std::invalid_argument("gap fitting: years must be a (n_plots,) vector");
    }
}

// The complete plots (every recorded value finite) and the gaps, as rows.
std::pair<torch::Tensor, torch::Tensor> divide(const torch::Tensor& recorded_values) {
    const torch::Tensor rows = recorded_values.dim() == 1 ? recorded_values.unsqueeze(1) : recorded_values;
    const torch::Tensor complete = finite_rows(rows);
    return {complete.nonzero().select(1, 0), complete.logical_not().nonzero().select(1, 0)};
}

// One target's plots in the plane, with the matched test's hiding radii and
// each complete plot's matched neighbours.
struct Prepared {
    Fold fold;
    GapDistances distances;
    HidingRadii radii;
    std::vector<torch::Tensor> matched;
};

Prepared prepare(const GapTarget& target, const torch::Tensor& raw_values,
                 const torch::Tensor& recorded_values, const torch::Tensor& cal_index,
                 const torch::Tensor& test_index, const GapPlots& plots, const GapFittingConfig& config,
                 const HidingRadii* injected) {
    const torch::Tensor recorded_cal = recorded_values.index({cal_index});
    const Plane plane(target, recorded_cal);
    const torch::Tensor species = presence_of(plots.species.to(torch::kCPU));
    const torch::Tensor coords = plots.coordinates.defined() ? as_double(plots.coordinates) : torch::Tensor();
    const torch::Tensor years = plots.years.defined() ? as_double(plots.years) : torch::Tensor();
    auto rows = [](const torch::Tensor& t, const torch::Tensor& index) {
        return t.defined() ? t.index({index}) : torch::Tensor();
    };

    Prepared out{Fold{plane,
                      plane.to(recorded_cal),
                      raw_values.defined() ? plane.to(raw_values.index({cal_index})) : torch::Tensor(),
                      raw_values.defined() ? plane.to(raw_values.index({test_index})) : torch::Tensor(),
                      select_rows(species, cal_index),
                      select_rows(species, test_index),
                      rows(coords, cal_index),
                      rows(coords, test_index),
                      rows(years, cal_index),
                      rows(years, test_index)},
                 {},
                 {},
                 {}};
    Fold& fold = out.fold;
    // Plots are near or far in space unless the target is the position itself
    // (whose gaps have none), and near or far in time where both sides carry
    // survey years (a year target's gaps carry none).
    fold.spatial_axis = config.spatial && target.scale != GapScale::Geographic && coords.defined() &&
                        finite_rows(fold.coords_cal).sum().item<int64_t>() >= 2;
    fold.time_axis = years.defined() && torch::isfinite(fold.years_cal).any().item<bool>() &&
                     torch::isfinite(fold.years_test).any().item<bool>();

    out.distances = gap_distances(fold);
    if (injected != nullptr) {
        auto take = [&](const torch::Tensor& t, const char* axis) {
            if (!t.defined()) return torch::Tensor();
            if (t.dim() != 1 || t.size(0) != fold.n_cal()) {
                throw std::invalid_argument(std::string("gap fitting: the injected ") + axis +
                                            " radii must hold one value per complete plot (" +
                                            std::to_string(fold.n_cal()) + ")");
            }
            return as_double(t);
        };
        out.radii.similarity = take(injected->similarity, "similarity");
        out.radii.km = take(injected->km, "km");
        out.radii.years = take(injected->years, "years");
        if (!out.radii.similarity.defined()) {
            throw std::invalid_argument("gap fitting: injected radii need the similarity axis");
        }
    } else {
        at::Generator generator =
            at::detail::createCPUGenerator(static_cast<uint64_t>(config.matching_seed));
        out.radii.similarity = drawn(out.distances.similarity, fold.n_cal(), generator);
        if (fold.spatial_axis) out.radii.km = drawn(out.distances.km, fold.n_cal(), generator);
        if (fold.time_axis) out.radii.years = drawn(out.distances.years, fold.n_cal(), generator);
    }
    out.matched = matched_neighbours(fold, out.radii, config.k_grid);
    return out;
}

}  // namespace

MatchedGapNeighbours matched_gap_neighbours(const GapTarget& target, const torch::Tensor& recorded,
                                            const GapPlots& plots, const GapFittingConfig& config,
                                            const HidingRadii* radii) {
    config.validate();
    validate_inputs(target, torch::Tensor(), recorded, plots);
    const torch::Tensor recorded_values = as_double(recorded);
    const auto [cal_index, test_index] = divide(recorded_values);
    if (cal_index.size(0) < 2 || test_index.size(0) == 0) {
        throw std::invalid_argument("gap fitting: the matched test needs at least two complete plots "
                                    "and one gap");
    }
    const Prepared prepared =
        prepare(target, torch::Tensor(), recorded_values, cal_index, test_index, plots, config, radii);
    MatchedGapNeighbours out;
    out.complete = cal_index;
    out.gaps = test_index;
    out.radii = prepared.radii;
    out.species = prepared.matched[kSpecies];
    out.spatial = prepared.matched[kSpatial];
    return out;
}

GapFit fit_gaps(const GapTarget& target, const torch::Tensor& raw, const torch::Tensor& recorded,
                const GapPlots& plots, const GapFittingConfig& config, const HidingRadii* radii) {
    config.validate();
    if (!raw.defined()) throw std::invalid_argument("gap fitting: the raw prediction is required");
    validate_inputs(target, raw, recorded, plots);
    const int64_t n = plots.species.size(0);
    const torch::Tensor raw_values = as_double(raw);
    const torch::Tensor recorded_values = as_double(recorded);
    const auto [cal_index, test_index] = divide(recorded_values);

    GapFit out;
    out.report.target = target.name;
    out.report.scale = target.scale;
    out.report.n_complete = cal_index.size(0);
    out.report.n_gaps = test_index.size(0);
    out.fitted = recorded_values.clone();
    out.method = torch::full({n}, kGapRecorded, kLong);
    if (out.report.n_gaps == 0) return out;
    if (out.report.n_complete < 2) {
        out.report.skipped = "fewer than two complete plots";
        out.fitted.index_put_({test_index}, raw_values.index({test_index}));
        out.method.index_put_({test_index}, static_cast<int64_t>(GapMethod::Resolve));
        return out;
    }

    const Prepared prepared =
        prepare(target, raw_values, recorded_values, cal_index, test_index, plots, config, radii);
    const Fold& fold = prepared.fold;
    const Plane& plane = fold.plane;
    out.report.axes.push_back(summarise_axis("similarity", prepared.distances.similarity, true));
    out.report.axes.push_back(summarise_axis(
        "km", prepared.distances.km,
        fold.spatial_axis && prepared.radii.km.defined() &&
            torch::isfinite(prepared.distances.km).any().item<bool>()));
    out.report.axes.push_back(summarise_axis("years", prepared.distances.years,
                                             fold.time_axis && prepared.radii.years.defined()));

    std::vector<std::optional<NeighbourTerm>> terms(kNeighbourKinds);
    for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
        terms[kind] = neighbour_term(fold, static_cast<NeighbourKind>(kind), prepared.matched[kind],
                                     config.k_grid);
        if (terms[kind]) out.report.neighbours.push_back(terms[kind]->choice);
    }

    // The candidates this fit can evaluate, in GapMethod order.
    std::vector<const CandidateSpec*> usable;
    for (const auto& spec : kCandidates) {
        const bool asked = config.candidates.empty() ||
                           std::find(config.candidates.begin(), config.candidates.end(), spec.method) !=
                               config.candidates.end();
        if (!asked) continue;
        if (spec.reads[kSpatial] && !fold.spatial_axis) continue;
        bool available = true;
        for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
            if (spec.reads[kind] && !terms[kind]) available = false;
        }
        if (available) usable.push_back(&spec);
    }
    const torch::Tensor centre = plane.centre(fold.truth_cal);
    std::vector<CandidateFit> fits;
    fits.reserve(usable.size());
    for (const CandidateSpec* spec : usable) fits.push_back(fit_candidate(fold, *spec, centre, terms));

    const torch::Tensor located =
        fold.spatial_axis ? finite_rows(fold.coords_test) : torch::zeros({fold.n_test()}, torch::kBool);
    const torch::Tensor cal_located =
        fold.spatial_axis ? finite_rows(fold.coords_cal) : torch::ones({fold.n_cal()}, torch::kBool);
    torch::Tensor test_rows = torch::full({fold.n_test(), fold.truth_cal.size(1)}, kNaN, kDouble);
    torch::Tensor test_method = torch::full({fold.n_test()}, kGapRecorded, kLong);

    struct Group {
        const char* name;
        torch::Tensor gaps;
        torch::Tensor rows;
        bool spatial;
    };
    const Group groups[] = {
        {"located", located, cal_located, true},
        {"unlocated", located.logical_not(), torch::ones({fold.n_cal()}, torch::kBool), false},
    };
    for (const Group& group : groups) {
        const int64_t n_gaps = group.gaps.sum().item<int64_t>();
        if (n_gaps == 0) continue;
        // Judged on the complete plots the group's gaps resemble: those with a
        // position for gaps with one. A complete plot left without any
        // neighbour at the gaps' distance is read as the local constant,
        // which is all the dataset says about it there.
        const torch::Tensor truth = fold.truth_cal.index({group.rows});
        GapGroupChoice choice;
        choice.group = group.name;
        choice.n_complete = truth.size(0);
        choice.n_gaps = n_gaps;
        std::optional<std::size_t> best;
        double best_error = kNaN;
        for (std::size_t c = 0; c < usable.size(); ++c) {
            if (usable[c]->reads[kSpatial] && !group.spatial) continue;
            const torch::Tensor cal = fits[c].cal.index({group.rows});
            const torch::Tensor judged =
                torch::where(torch::isfinite(cal), cal, centre.unsqueeze(0).expand_as(cal));
            const double error = plane.aggregate(plane.errors(judged, truth));
            choice.matched_error.emplace_back(usable[c]->method, error);
            if (!best || error < best_error) {
                best = c;
                best_error = error;
            }
        }
        if (!best) {
            throw std::invalid_argument(
                "gap fitting: target '" + target.name + "' has " + std::to_string(n_gaps) + " " +
                group.name + " gaps and no candidate for them" +
                (group.spatial ? "" : "; gaps without coordinates need a candidate that reads no position"));
        }
        const CandidateFit& chosen = fits[*best];
        choice.method = usable[*best]->method;
        choice.coefficients = chosen.coefficients;
        choice.n_fitted = chosen.n_fitted;
        choice.k = chosen.k;
        test_rows.index_put_({group.gaps}, chosen.test.index({group.gaps}));
        test_method.index_put_({group.gaps}, static_cast<int64_t>(choice.method));
        out.report.groups.push_back(std::move(choice));
    }

    out.fitted.index_put_({test_index}, plane.back(test_rows));
    out.method.index_put_({test_index}, test_method);
    return out;
}

// ---------------------------------------------------------------------------
// Reading the tables
// ---------------------------------------------------------------------------

GapTables read_gap_tables(const std::optional<TableSource>& header, const TableSource& species,
                          const GapColumns& columns, const std::vector<std::string>& plot_ids) {
    if (columns.latitude.has_value() != columns.longitude.has_value()) {
        throw std::invalid_argument("gap fitting: name both a latitude and a longitude column, or neither");
    }
    const auto n = static_cast<int64_t>(plot_ids.size());
    std::unordered_map<std::string, int32_t> plot_of;
    plot_of.reserve(plot_ids.size());
    for (std::size_t i = 0; i < plot_ids.size(); ++i) plot_of.emplace(plot_ids[i], static_cast<int32_t>(i));

    // Every numeric plot-level column read, once each.
    std::vector<std::string> names;
    auto column = [&](const std::string& name) {
        const auto at = std::find(names.begin(), names.end(), name);
        if (at != names.end()) return static_cast<std::size_t>(at - names.begin());
        names.push_back(name);
        return names.size() - 1;
    };
    std::optional<std::size_t> lat_col, lon_col, year_col;
    if (columns.latitude) {
        lat_col = column(*columns.latitude);
        lon_col = column(*columns.longitude);
    }
    if (columns.survey_year) year_col = column(*columns.survey_year);
    std::vector<std::vector<std::size_t>> recorded_cols;
    for (const auto& [target, cols] : columns.recorded) {
        if (cols.empty() || cols.size() > 2) {
            throw std::invalid_argument("gap fitting: target '" + target +
                                        "' names " + std::to_string(cols.size()) +
                                        " recorded columns; one, or latitude and longitude");
        }
        std::vector<std::size_t> at;
        for (const auto& c : cols) at.push_back(column(c));
        recorded_cols.push_back(std::move(at));
    }

    std::vector<std::vector<double>> values(names.size(), std::vector<double>(plot_ids.size(), kNaN));
    std::vector<char> seen(plot_ids.size(), 0);
    const char* table = header ? "header" : "species";
    const auto read_plot_columns = [&](RowSource& source, bool first_row_only) {
        const int plot_col = require_column(source, columns.plot_id, "plot_id", table);
        std::vector<int> cols;
        for (const auto& name : names) cols.push_back(require_column(source, name, "gap fitting", table));
        source.read_rows([&](size_t, RowFields row) {
            const auto id = cell_at(row, plot_col);
            if (!id) return;
            const auto it = plot_of.find(std::string(*id));
            if (it == plot_of.end()) return;
            const auto p = static_cast<std::size_t>(it->second);
            if (seen[p]) {
                if (first_row_only) return;
                throw std::runtime_error("Duplicate plot_id '" + std::string(*id) + "' in header data.");
            }
            seen[p] = 1;
            for (std::size_t c = 0; c < cols.size(); ++c) {
                if (const auto v = parse_cell_at(row, cols[c])) values[c][p] = *v;
            }
        });
    };
    if (header) {
        const auto source = header->open();
        read_plot_columns(*source, /*first_row_only=*/false);
    } else {
        const auto source = species.open();
        read_plot_columns(*source, /*first_row_only=*/true);
    }
    for (std::size_t p = 0; p < plot_ids.size(); ++p) {
        if (!seen[p]) {
            throw std::runtime_error("gap fitting: plot '" + plot_ids[p] + "' is not in the " + table +
                                     " table");
        }
    }

    // Presence of every named species, merged per plot.
    PlotAssemblages::Builder builder;
    {
        const auto source = species.open();
        const int plot_col = require_column(*source, columns.plot_id, "plot_id", "species");
        const int species_col = require_column(*source, columns.species_id, "species_id", "species");
        source->read_rows([&](size_t, RowFields row) {
            const auto id = cell_at(row, plot_col);
            const auto name = cell_at(row, species_col);
            if (!id || !name || name->empty()) return;
            const auto it = plot_of.find(std::string(*id));
            if (it == plot_of.end()) return;
            builder.add(it->second, *name, 1.0f, {}, {});
        });
    }
    std::vector<int32_t> identity(plot_ids.size());
    for (std::size_t i = 0; i < identity.size(); ++i) identity[i] = static_cast<int32_t>(i);
    const PlotAssemblages assemblages =
        std::move(builder).finish(n, identity, RepresentationMode::PresenceAbsence);
    std::vector<int64_t> offsets{0};
    std::vector<int64_t> species_columns;
    offsets.reserve(plot_ids.size() + 1);
    species_columns.reserve(static_cast<std::size_t>(assemblages.n_entries()));
    for (int64_t p = 0; p < n; ++p) {
        const auto present = assemblages.species(p);
        std::vector<int64_t> row(present.begin(), present.end());
        std::sort(row.begin(), row.end());
        species_columns.insert(species_columns.end(), row.begin(), row.end());
        offsets.push_back(static_cast<int64_t>(species_columns.size()));
    }
    const auto nnz = static_cast<int64_t>(species_columns.size());

    auto tensor_of = [&](std::size_t c) { return torch::tensor(values[c], kDouble); };
    GapTables out;
    out.plots.species = make_sparse_rows(torch::tensor(offsets, kLong), torch::tensor(species_columns, kLong),
                                         torch::ones({nnz}, torch::kFloat32), assemblages.n_species());
    if (lat_col) out.plots.coordinates = torch::stack({tensor_of(*lat_col), tensor_of(*lon_col)}, 1);
    if (year_col) out.plots.years = tensor_of(*year_col);
    for (std::size_t t = 0; t < columns.recorded.size(); ++t) {
        const auto& cols = recorded_cols[t];
        out.recorded[columns.recorded[t].first] =
            cols.size() == 1 ? tensor_of(cols[0]) : torch::stack({tensor_of(cols[0]), tensor_of(cols[1])}, 1);
    }
    return out;
}

std::vector<GapFit> fit_gaps(const ResolvePredictions& predictions, const GapTables& tables,
                             const std::vector<GapFitRequest>& requests, const GapFittingConfig& config) {
    auto head = [&](const std::string& name) {
        const auto it = predictions.predictions.find(name);
        if (it == predictions.predictions.end()) {
            throw std::invalid_argument("gap fitting: the predictions hold no output '" + name + "'");
        }
        return as_double(it->second);
    };
    std::vector<GapFit> out;
    for (const GapFitRequest& request : requests) {
        const GapTarget& target = request.target;
        const std::size_t needed_min = target.scale == GapScale::Geographic ? 2 : 1;
        const std::size_t needed_max = target.scale == GapScale::Linear || target.scale == GapScale::Log ? 1 : 2;
        if (request.outputs.size() < needed_min || request.outputs.size() > needed_max) {
            throw std::invalid_argument("gap fitting: target '" + target.name + "' reads " +
                                        std::to_string(request.outputs.size()) + " outputs; a " +
                                        enum_to_name(target.scale) + " target reads " +
                                        (target.scale == GapScale::Geographic ? "two (latitude, longitude)"
                                         : target.scale == GapScale::Circular
                                             ? "one (the bearing) or two (its sine, then its cosine)"
                                             : "one"));
        }
        torch::Tensor raw;
        if (target.scale == GapScale::Geographic) {
            raw = torch::stack({head(request.outputs[0]), head(request.outputs[1])}, 1);
        } else if (request.outputs.size() == 2) {
            raw = as_double(bearing_from_components(head(request.outputs[0]), head(request.outputs[1]),
                                                    target.period));
        } else {
            raw = head(request.outputs[0]);
        }
        const auto it = tables.recorded.find(target.name);
        if (it == tables.recorded.end()) {
            throw std::invalid_argument("gap fitting: the tables carry no recorded values for target '" +
                                        target.name + "'");
        }
        out.push_back(fit_gaps(target, raw, it->second, tables.plots, config));
    }
    return out;
}

}  // namespace resolve
