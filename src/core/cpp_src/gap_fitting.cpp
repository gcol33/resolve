#include "resolve/gap_fitting.hpp"

#include "resolve/enum_names.hpp"
#include "resolve/geo.hpp"
#include "resolve/plot_columns.hpp"
#include "resolve/plot_neighbours.hpp"
#include "resolve/row_cells.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/species_encoding.hpp"
#include "resolve/suite.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
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

std::vector<double> values_of(const torch::Tensor& t) {
    if (!t.defined()) return {};
    const torch::Tensor c = as_double(t).contiguous();
    return {c.data_ptr<double>(), c.data_ptr<double>() + c.numel()};
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

    // The local constant of some plane rows: their geographic median, the
    // mean direction of a bearing, the median on a log scale (the constant
    // with the lowest absolute error there), the mean otherwise.
    [[nodiscard]] torch::Tensor centre(const torch::Tensor& truth) const {
        switch (scale_) {
            case GapScale::Geographic: {
                const torch::Tensor position = back(truth);
                const auto [lat, lon] = geographic_median(position.select(1, 0), position.select(1, 1));
                return to(torch::tensor({{lat, lon}}, kDouble)).squeeze(0);
            }
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

// ---------------------------------------------------------------------------
// One target divided into complete plots and gaps, in the plane
// ---------------------------------------------------------------------------

struct Split {
    Plane plane;
    torch::Tensor truth_cal;
    torch::Tensor raw_cal;
    torch::Tensor raw_test;
    torch::Tensor coords_cal;  // (n_cal, 2), undefined without positions
    torch::Tensor coords_test;
    torch::Tensor years_cal;   // undefined without years
    torch::Tensor years_test;
    bool spatial_axis = false;  // spatial neighbours are candidates
    bool km_axis = false;       // the fold layout is matched in km
    bool time_axis = false;     // the fold layout is matched in survey years

    int64_t n_test = 0;

    [[nodiscard]] int64_t n_cal() const { return truth_cal.size(0); }
    [[nodiscard]] bool circular() const { return plane.scale() == GapScale::Circular; }
};

// Each gap's distance to its nearest complete plot on every axis.
struct GapDistances {
    torch::Tensor similarity;  // (n_test,)
    torch::Tensor km;          // undefined without the km axis
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

// What hides one complete plot from another in a fold layout: the same fold,
// the held fold, or a buffer around the plot in km or in years (NaN: none). A
// search passes the km it computed itself, NaN when it did not.
struct FoldHiding {
    const std::vector<int64_t>& folds;
    int64_t held;
    // Per complete plot, its buffer in km and in years (NaN: none); null for
    // no buffer on the axis.
    const std::vector<double>* buffer_km;
    const std::vector<double>* buffer_years;
    const std::vector<double>& lat;
    const std::vector<double>& lon;
    const std::vector<double>& years;

    [[nodiscard]] bool hides(int64_t u, int64_t v, double km) const {
        const auto a = static_cast<std::size_t>(u), b = static_cast<std::size_t>(v);
        if (folds[b] == folds[a] || folds[b] == held) return true;
        if (buffer_km != nullptr && !std::isnan((*buffer_km)[a])) {
            if (std::isnan(km)) km = great_circle_km(lat[a], lon[a], lat[b], lon[b]);
            if (km < (*buffer_km)[a]) return true;
        }
        return buffer_years != nullptr && std::abs(years[a] - years[b]) < (*buffer_years)[a];
    }

    // Per query row, the km within which this rule hides every plot from it
    // (its km buffer, NaN for none), as PlaceIndex::nearest takes it; empty
    // without km buffers.
    [[nodiscard]] std::vector<double> radius(const std::vector<int64_t>& rows) const {
        std::vector<double> out;
        if (buffer_km == nullptr) return out;
        out.reserve(rows.size());
        for (const int64_t u : rows) out.push_back((*buffer_km)[static_cast<std::size_t>(u)]);
        return out;
    }
};

// A fold layout: each complete plot's fold, and its buffers (empty: none).
struct Layout {
    std::vector<int64_t> folds;
    std::vector<double> buffer_km;
    std::vector<double> buffer_years;

    [[nodiscard]] const std::vector<double>* km() const { return buffer_km.empty() ? nullptr : &buffer_km; }
    [[nodiscard]] const std::vector<double>* yrs() const { return buffer_years.empty() ? nullptr : &buffer_years; }
};

// The neighbour searches of one target over its plots, the complete plots
// first (rows [0, n_cal)) and the gaps after them: by species over a neighbour
// graph, by position over a k-d tree. Only complete plots are neighbours.
class PlotSearch {
public:
    PlotSearch(const Split& split, const torch::Tensor& species_cal, const torch::Tensor& species_test,
               const std::vector<int64_t>& grid)
        : n_cal_(species_cal.size(0)),
          n_test_(species_test.size(0)),
          sets_(concat_rows({species_cal, species_test})),
          k_species_(std::min(*std::max_element(grid.begin(), grid.end()), n_cal_ - 1)),
          graph_(sets_, n_cal_, SetGraphConfig{std::max<int64_t>(1, k_species_)}),
          open_(graph_.open()) {
        lat_.assign(static_cast<std::size_t>(n_cal_), kNaN);
        lon_.assign(static_cast<std::size_t>(n_cal_), kNaN);
        years_.assign(static_cast<std::size_t>(n_cal_), kNaN);
        if (split.coords_cal.defined()) {
            lat_ = values_of(split.coords_cal.select(1, 0));
            lon_ = values_of(split.coords_cal.select(1, 1));
        }
        if (split.years_cal.defined()) years_ = values_of(split.years_cal);
        if (split.spatial_axis || split.km_axis) {
            places_.emplace(torch::cat({split.coords_cal, split.coords_test}, 0), n_cal_);
        }
        if (split.spatial_axis) {
            const int64_t located = finite_rows(split.coords_cal).sum().item<int64_t>();
            k_spatial_ = std::min(*std::max_element(grid.begin(), grid.end()), located - 1);
        }
    }

    [[nodiscard]] const PresenceSets& sets() const noexcept { return sets_; }
    [[nodiscard]] const PlaceIndex* places() const noexcept { return places_ ? &*places_ : nullptr; }

    // The hiding rule of a layout over the complete plots.
    [[nodiscard]] FoldHiding hiding(const std::vector<int64_t>& folds, int64_t held,
                                    const std::vector<double>* buffer_km,
                                    const std::vector<double>* buffer_years) const {
        return {folds, held, buffer_km, buffer_years, lat_, lon_, years_};
    }

    // Each gap's distance to its nearest complete plot on every axis; a gap
    // sharing no species with any complete plot is at similarity 0.
    [[nodiscard]] GapDistances distances(const Split& split) const {
        GapDistances out;
        out.similarity = torch::nan_to_num(open_.similarities.slice(0, n_cal_).select(1, 0), 0.0);
        if (split.km_axis) out.km = places_->nearest(gap_rows(), 1).km.select(1, 0);
        if (split.time_axis) out.years = nearest_year_gap(split.years_test, split.years_cal);
        return out;
    }

    // Each complete plot's neighbours among the complete plots the layout
    // leaves it: those of the other folds, `held` (a fold, or -1) hidden from
    // every plot too, outside its buffers. By species (always) and by distance
    // (with the spatial axis).
    [[nodiscard]] std::vector<torch::Tensor> out_of_fold(const Layout& layout, int64_t held) {
        const FoldHiding rule = hiding(layout.folds, held, layout.km(), layout.yrs());
        std::vector<torch::Tensor> out(kNeighbourKinds);
        if (k_species_ >= 1) {
            out[kSpecies] =
                graph_.hidden(n_cal_, [&](int64_t u, int64_t v, double) { return rule.hides(u, v, kNaN); })
                    .indices.slice(1, 0, k_species_);
        }
        if (places_ && k_spatial_ >= 1) {
            std::vector<int64_t> rows(static_cast<std::size_t>(n_cal_));
            std::iota(rows.begin(), rows.end(), 0);
            out[kSpatial] = places_
                                ->nearest(rows, k_spatial_,
                                          [&](int64_t u, int64_t v, double km) { return rule.hides(u, v, km); },
                                          rule.radius(rows))
                                .indices;
        }
        return out;
    }

    // The gaps' k neighbours among the complete plots, of one kind.
    [[nodiscard]] torch::Tensor gap_neighbours(NeighbourKind kind, int64_t k) const {
        if (kind == kSpecies) return open_.indices.slice(0, n_cal_).slice(1, 0, k);
        return places_->nearest(gap_rows(), k).indices;
    }

private:
    [[nodiscard]] std::vector<int64_t> gap_rows() const {
        std::vector<int64_t> rows(static_cast<std::size_t>(n_test_));
        std::iota(rows.begin(), rows.end(), n_cal_);
        return rows;
    }

    int64_t n_cal_;
    int64_t n_test_;
    PresenceSets sets_;
    int64_t k_species_;
    int64_t k_spatial_ = 0;
    SetNeighbourGraph graph_;
    SetNeighbours open_;
    std::optional<PlaceIndex> places_;
    std::vector<double> lat_, lon_, years_;  // the complete plots'
};

// ---------------------------------------------------------------------------
// The fold layout
// ---------------------------------------------------------------------------

// Portable 32-bit hashing (the murmur3 finaliser over the words in turn), so
// every implementation deals the same layout from the same seed.
uint32_t mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

uint32_t hash_words(uint32_t seed, std::initializer_list<uint32_t> words) {
    uint32_t h = mix32(seed);
    for (const uint32_t w : words) h = mix32((h ^ w) + 0x9e3779b9u);
    return h;
}

uint32_t low_word(int64_t v) { return static_cast<uint32_t>(static_cast<uint64_t>(v)); }
uint32_t high_word(int64_t v) { return static_cast<uint32_t>(static_cast<uint64_t>(v) >> 32); }

// The streams the seed is hashed into.
constexpr uint32_t kGroupStream = 1;
constexpr uint32_t kSampleStream = 2;

// Squares of the equal-area plane per side, from half the complete plots'
// extent down by halving, and survey-year periods likewise down to one year.
constexpr int kBlockSizes = 10;
constexpr int kPeriodLengths = 8;

// Similar complete plots kept per matching plot before a fold layout falls
// back to recounting the plot's overlaps.
constexpr std::size_t kSimilarListLength = 256;

constexpr uint32_t kBufferStream = 3;

// The axes a layout's buffers lie on.
enum class Buffer { None, Km, Years, Both };

// What a layout groups the complete plots by, and the buffers it hides within.
struct LayoutSpec {
    double block_km = kNaN;
    double block_years = kNaN;
    Buffer buffer = Buffer::None;
};

double median_of(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) { return std::isnan(v); }), values.end());
    if (values.empty()) return kNaN;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

class LayoutMatcher {
public:
    LayoutMatcher(const Split& split, const PlotSearch& search, const GapDistances& gaps,
                  const GapFittingConfig& config)
        : split_(split), search_(search), n_(split.n_cal()), seed_(static_cast<uint32_t>(config.matching_seed)) {
        positions();
        survey_years();
        sample(config.matching_sample);
        gap_side(gaps);
    }

    // The candidate layouts, plot by plot first (ties go to the earlier):
    // squares, periods and both; then plot by plot with buffers drawn from
    // the gaps' distances in km, in years, and in both.
    [[nodiscard]] std::vector<LayoutSpec> specs() const {
        std::vector<LayoutSpec> out{{}};
        for (const double s : block_sizes_) out.push_back({s, kNaN, Buffer::None});
        for (const double w : period_lengths_) out.push_back({kNaN, w, Buffer::None});
        for (const double s : block_sizes_) {
            for (const double w : period_lengths_) out.push_back({s, w, Buffer::None});
        }
        if (!gap_km_.empty()) out.push_back({kNaN, kNaN, Buffer::Km});
        if (!gap_years_.empty()) out.push_back({kNaN, kNaN, Buffer::Years});
        if (!gap_km_.empty() && !gap_years_.empty()) out.push_back({kNaN, kNaN, Buffer::Both});
        return out;
    }

    // The buffers of `spec` added to its folds: each complete plot takes the
    // distances of one gap drawn by the hash of its row, among the gaps with
    // a distance on the buffer's axes (with both, the gap's two distances
    // together; a missing one buffers nothing on its axis).
    [[nodiscard]] Layout with_buffers(std::vector<int64_t> folds, const LayoutSpec& spec) const {
        Layout out{std::move(folds), {}, {}};
        if (spec.buffer == Buffer::None) return out;
        const bool km = spec.buffer != Buffer::Years, years = spec.buffer != Buffer::Km;
        std::vector<int64_t> pool;
        for (std::size_t g = 0; g < gap_km_all_.size(); ++g) {
            const bool has_km = std::isfinite(gap_km_all_[g]), has_years = std::isfinite(gap_years_all_[g]);
            if ((km && has_km) || (years && has_years)) pool.push_back(static_cast<int64_t>(g));
        }
        if (km) out.buffer_km.assign(static_cast<std::size_t>(n_), kNaN);
        if (years) out.buffer_years.assign(static_cast<std::size_t>(n_), kNaN);
        for (int64_t i = 0; i < n_; ++i) {
            const uint32_t h = hash_words(seed_, {kBufferStream, low_word(i), high_word(i)});
            const auto g = static_cast<std::size_t>(pool[h % static_cast<uint32_t>(pool.size())]);
            if (km) out.buffer_km[static_cast<std::size_t>(i)] = gap_km_all_[g];
            if (years) out.buffer_years[static_cast<std::size_t>(i)] = gap_years_all_[g];
        }
        return out;
    }

    // Deal the complete plots' groups under `spec` to `n_folds` folds: the
    // largest groups first (ties by hash, then by first plot), each to the
    // fold holding fewest plots (ties to the lower fold); empty folds are
    // dropped and the rest numbered in order. Returns the folds and the
    // largest fold's plot count.
    [[nodiscard]] std::pair<std::vector<int64_t>, int64_t> deal(const LayoutSpec& spec, int64_t n_folds) const {
        struct Key {
            int64_t a, b, c, own;
            bool operator==(const Key& o) const { return a == o.a && b == o.b && c == o.c && own == o.own; }
        };
        struct KeyHash {
            std::size_t operator()(const Key& k) const {
                return std::hash<int64_t>()(k.a) ^ (std::hash<int64_t>()(k.b) * 31) ^
                       (std::hash<int64_t>()(k.c) * 961) ^ (std::hash<int64_t>()(k.own) * 29791);
            }
        };
        struct Group {
            Key key;
            int64_t size = 0;
            int64_t first = 0;
            uint32_t hash = 0;
        };
        const bool space = !std::isnan(spec.block_km);
        const bool time = !std::isnan(spec.block_years);
        std::unordered_map<Key, int64_t, KeyHash> index;
        std::vector<Group> groups;
        std::vector<int64_t> group_of(static_cast<std::size_t>(n_));
        for (int64_t i = 0; i < n_; ++i) {
            const auto u = static_cast<std::size_t>(i);
            Key key{0, 0, 0, -1};
            bool own = !space && !time;
            if (space) {
                if (std::isfinite(east_[u]) && std::isfinite(north_[u])) {
                    key.a = static_cast<int64_t>(std::floor(east_[u] / spec.block_km));
                    key.b = static_cast<int64_t>(std::floor(north_[u] / spec.block_km));
                } else {
                    own = true;
                }
            }
            if (time) {
                if (std::isfinite(years_[u])) {
                    key.c = static_cast<int64_t>(std::floor((years_[u] - year_origin_) / spec.block_years));
                } else {
                    own = true;
                }
            }
            if (own) key = Key{0, 0, 0, i};
            const auto [it, added] = index.emplace(key, static_cast<int64_t>(groups.size()));
            if (added) groups.push_back({key, 0, i, 0});
            ++groups[static_cast<std::size_t>(it->second)].size;
            group_of[u] = it->second;
        }
        for (Group& g : groups) {
            g.hash = hash_words(seed_, {kGroupStream, low_word(g.key.a), high_word(g.key.a), low_word(g.key.b),
                                        high_word(g.key.b), low_word(g.key.c), high_word(g.key.c),
                                        low_word(g.key.own), high_word(g.key.own)});
        }
        std::vector<int64_t> order(groups.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int64_t x, int64_t y) {
            const Group& gx = groups[static_cast<std::size_t>(x)];
            const Group& gy = groups[static_cast<std::size_t>(y)];
            if (gx.size != gy.size) return gx.size > gy.size;
            if (gx.hash != gy.hash) return gx.hash < gy.hash;
            return gx.first < gy.first;
        });
        std::vector<int64_t> counts(static_cast<std::size_t>(n_folds), 0);
        std::vector<int64_t> fold_of_group(groups.size());
        for (const int64_t g : order) {
            const auto f = std::min_element(counts.begin(), counts.end()) - counts.begin();
            fold_of_group[static_cast<std::size_t>(g)] = f;
            counts[static_cast<std::size_t>(f)] += groups[static_cast<std::size_t>(g)].size;
        }
        std::vector<int64_t> renumber(static_cast<std::size_t>(n_folds), -1);
        int64_t next = 0;
        for (int64_t f = 0; f < n_folds; ++f) {
            if (counts[static_cast<std::size_t>(f)] > 0) renumber[static_cast<std::size_t>(f)] = next++;
        }
        std::vector<int64_t> folds(static_cast<std::size_t>(n_));
        for (int64_t i = 0; i < n_; ++i) {
            folds[static_cast<std::size_t>(i)] =
                renumber[static_cast<std::size_t>(fold_of_group[static_cast<std::size_t>(group_of[static_cast<std::size_t>(i)])])];
        }
        return {std::move(folds), *std::max_element(counts.begin(), counts.end())};
    }

    // Per matched axis, the rank-scale Wasserstein distance between the
    // matching plots' distances to the other folds and the gaps' distances to
    // the complete plots.
    [[nodiscard]] std::vector<std::pair<std::string, double>> mismatch(const Layout& layout) const {
        const FoldHiding rule = search_.hiding(layout.folds, -1, layout.km(), layout.yrs());
        std::vector<std::pair<std::string, double>> out;
        out.emplace_back("similarity", rank_wasserstein(held_dissimilarity(rule), gap_dissimilarity_));
        if (split_.km_axis) out.emplace_back("km", rank_wasserstein(held_km(rule), gap_km_));
        if (split_.time_axis) out.emplace_back("years", rank_wasserstein(held_years(rule), gap_years_));
        return out;
    }

    [[nodiscard]] int64_t n_matching() const noexcept { return static_cast<int64_t>(sample_.size()); }

private:
    void positions() {
        east_.assign(static_cast<std::size_t>(n_), kNaN);
        north_.assign(static_cast<std::size_t>(n_), kNaN);
        if (!split_.coords_cal.defined()) return;
        const torch::Tensor located = finite_rows(split_.coords_cal);
        if (located.sum().item<int64_t>() < 2) return;
        const torch::Tensor known = split_.coords_cal.index({located});
        const auto [lat, lon] = geographic_median(known.select(1, 0), known.select(1, 1));
        const LambertAzimuthalEqualArea projection(lat, lon);
        const auto [east, north] = projection.forward(split_.coords_cal.select(1, 0), split_.coords_cal.select(1, 1));
        east_ = values_of(east / 1000.0);
        north_ = values_of(north / 1000.0);
        double lo_e = std::numeric_limits<double>::infinity(), hi_e = -lo_e, lo_n = lo_e, hi_n = -lo_e;
        for (int64_t i = 0; i < n_; ++i) {
            const auto u = static_cast<std::size_t>(i);
            if (!std::isfinite(east_[u]) || !std::isfinite(north_[u])) continue;
            lo_e = std::min(lo_e, east_[u]);
            hi_e = std::max(hi_e, east_[u]);
            lo_n = std::min(lo_n, north_[u]);
            hi_n = std::max(hi_n, north_[u]);
        }
        const double extent = std::max(hi_e - lo_e, hi_n - lo_n);
        if (!(extent > 0.0)) return;
        double side = extent;
        for (int j = 0; j < kBlockSizes; ++j) block_sizes_.push_back(side /= 2.0);
    }

    void survey_years() {
        years_.assign(static_cast<std::size_t>(n_), kNaN);
        if (!split_.years_cal.defined()) return;
        years_ = values_of(split_.years_cal);
        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        int64_t known = 0;
        for (const double y : years_) {
            if (!std::isfinite(y)) continue;
            lo = std::min(lo, y);
            hi = std::max(hi, y);
            ++known;
        }
        if (known < 2) return;
        year_origin_ = lo;
        double length = hi - lo;
        for (int j = 0; j < kPeriodLengths; ++j) {
            length /= 2.0;
            if (length < 1.0) break;
            period_lengths_.push_back(length);
        }
    }

    // The complete plots the layouts' distances are measured on: all of them,
    // or the `size` with the lowest hash of their row.
    void sample(int64_t size) {
        std::vector<int64_t> rows(static_cast<std::size_t>(n_));
        std::iota(rows.begin(), rows.end(), 0);
        if (n_ > size) {
            std::vector<uint32_t> h(static_cast<std::size_t>(n_));
            for (int64_t i = 0; i < n_; ++i) {
                h[static_cast<std::size_t>(i)] = hash_words(seed_, {kSampleStream, low_word(i), high_word(i)});
            }
            std::sort(rows.begin(), rows.end(), [&](int64_t a, int64_t b) {
                const uint32_t ha = h[static_cast<std::size_t>(a)], hb = h[static_cast<std::size_t>(b)];
                return ha != hb ? ha < hb : a < b;
            });
            rows.resize(static_cast<std::size_t>(size));
            std::sort(rows.begin(), rows.end());
        }
        sample_ = std::move(rows);

        // Each matching plot's most similar complete plots, ties to the lower
        // row, and whether the list holds every plot sharing a species.
        similar_.resize(sample_.size());
        whole_.assign(sample_.size(), true);
        for (std::size_t s = 0; s < sample_.size(); ++s) {
            std::vector<std::pair<double, int64_t>> list = similarities(sample_[s]);
            if (list.size() > kSimilarListLength) {
                std::partial_sort(list.begin(), list.begin() + kSimilarListLength, list.end(), more_similar);
                list.resize(kSimilarListLength);
                whole_[s] = false;
            } else {
                std::sort(list.begin(), list.end(), more_similar);
            }
            similar_[s] = std::move(list);
        }

        // The complete plots in order of survey year (ties by row).
        for (int64_t i = 0; i < n_; ++i) {
            if (std::isfinite(years_[static_cast<std::size_t>(i)])) by_year_.push_back(i);
        }
        std::sort(by_year_.begin(), by_year_.end(), [&](int64_t a, int64_t b) {
            const double ya = years_[static_cast<std::size_t>(a)], yb = years_[static_cast<std::size_t>(b)];
            return ya != yb ? ya < yb : a < b;
        });
        year_rank_.assign(static_cast<std::size_t>(n_), -1);
        for (std::size_t r = 0; r < by_year_.size(); ++r) year_rank_[static_cast<std::size_t>(by_year_[r])] = static_cast<int64_t>(r);
    }

    void gap_side(const GapDistances& gaps) {
        const std::size_t n_gaps = static_cast<std::size_t>(gaps.similarity.size(0));
        for (const double s : values_of(gaps.similarity)) gap_dissimilarity_.push_back(1.0 - s);
        gap_km_all_ = split_.km_axis ? values_of(gaps.km) : std::vector<double>(n_gaps, kNaN);
        gap_years_all_ = split_.time_axis ? values_of(gaps.years) : std::vector<double>(n_gaps, kNaN);
        for (const double km : gap_km_all_) {
            if (std::isfinite(km)) gap_km_.push_back(km);
        }
        for (const double y : gap_years_all_) {
            if (std::isfinite(y)) gap_years_.push_back(y);
        }
    }

    static bool more_similar(const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    }

    // A complete plot's similarity to every other complete plot it shares a
    // species with, as PresenceSets::cosine computes it.
    [[nodiscard]] std::vector<std::pair<double, int64_t>> similarities(int64_t plot) const {
        const PresenceSets& sets = search_.sets();
        std::vector<std::pair<double, int64_t>> out;
        const double own = sets.count(plot);
        for (const auto& [other, shared] : sets.overlaps(plot, n_)) {
            out.emplace_back(shared / std::sqrt(own * sets.count(other)), other);
        }
        return out;
    }

    // Each matching plot's dissimilarity to the most similar plot the rule
    // leaves it (1 when it shares no species with any).
    [[nodiscard]] std::vector<double> held_dissimilarity(const FoldHiding& rule) const {
        std::vector<double> out;
        out.reserve(sample_.size());
        for (std::size_t s = 0; s < sample_.size(); ++s) {
            const int64_t plot = sample_[s];
            double best = kNaN;
            for (const auto& [similarity, other] : similar_[s]) {
                if (!rule.hides(plot, other, kNaN)) {
                    best = similarity;
                    break;
                }
            }
            if (std::isnan(best)) {
                best = 0.0;
                if (!whole_[s]) {
                    for (const auto& [similarity, other] : similarities(plot)) {
                        if (!rule.hides(plot, other, kNaN)) best = std::max(best, similarity);
                    }
                }
            }
            out.push_back(1.0 - best);
        }
        return out;
    }

    // Each located matching plot's km to the nearest plot the rule leaves it.
    [[nodiscard]] std::vector<double> held_km(const FoldHiding& rule) const {
        std::vector<int64_t> rows;
        for (const int64_t plot : sample_) {
            const auto u = static_cast<std::size_t>(plot);
            if (std::isfinite(east_[u]) && std::isfinite(north_[u])) rows.push_back(plot);
        }
        std::vector<double> out;
        if (rows.empty()) return out;
        const PlaceNeighbours found = search_.places()->nearest(
            rows, 1, [&](int64_t u, int64_t v, double km) { return rule.hides(u, v, km); }, rule.radius(rows));
        for (const double km : values_of(found.km.select(1, 0))) {
            if (std::isfinite(km)) out.push_back(km);
        }
        return out;
    }

    // Each dated matching plot's years to the nearest plot in time the rule
    // leaves it.
    [[nodiscard]] std::vector<double> held_years(const FoldHiding& rule) const {
        std::vector<double> out;
        const auto n_ranked = static_cast<int64_t>(by_year_.size());
        for (const int64_t plot : sample_) {
            const int64_t r = year_rank_[static_cast<std::size_t>(plot)];
            if (r < 0) continue;
            const double year = years_[static_cast<std::size_t>(plot)];
            double best = std::numeric_limits<double>::infinity();
            for (int64_t q = r - 1; q >= 0; --q) {
                const int64_t other = by_year_[static_cast<std::size_t>(q)];
                if (!rule.hides(plot, other, kNaN)) {
                    best = std::min(best, year - years_[static_cast<std::size_t>(other)]);
                    break;
                }
            }
            for (int64_t q = r + 1; q < n_ranked; ++q) {
                const int64_t other = by_year_[static_cast<std::size_t>(q)];
                if (!rule.hides(plot, other, kNaN)) {
                    best = std::min(best, years_[static_cast<std::size_t>(other)] - year);
                    break;
                }
            }
            if (std::isfinite(best)) out.push_back(best);
        }
        return out;
    }

    // The Wasserstein distance between two samples on the rank scale of the
    // two pooled: the mean over the pooled values of the absolute difference
    // between the samples' empirical distribution functions there. It lies in
    // [0, 1] whatever the axis's units, so the axes add up; a share p of one
    // sample moved off a value the other holds entirely costs about p. NaN
    // when either is empty.
    static double rank_wasserstein(std::vector<double> a, std::vector<double> b) {
        if (a.empty() || b.empty()) return kNaN;
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        const double na = static_cast<double>(a.size()), nb = static_cast<double>(b.size());
        std::size_t i = 0, j = 0;
        double sum = 0.0;
        while (i < a.size() || j < b.size()) {
            const double x = j >= b.size() || (i < a.size() && a[i] <= b[j]) ? a[i] : b[j];
            const std::size_t i0 = i, j0 = j;
            while (i < a.size() && a[i] == x) ++i;
            while (j < b.size() && b[j] == x) ++j;
            const double mass = static_cast<double>((i - i0) + (j - j0));
            sum += mass * std::abs(static_cast<double>(i) / na - static_cast<double>(j) / nb);
        }
        return sum / (na + nb);
    }

    const Split& split_;
    const PlotSearch& search_;
    int64_t n_;
    uint32_t seed_;
    std::vector<double> east_, north_, years_;
    double year_origin_ = 0.0;
    std::vector<double> block_sizes_, period_lengths_;
    std::vector<int64_t> sample_;
    std::vector<std::vector<std::pair<double, int64_t>>> similar_;
    std::vector<bool> whole_;
    std::vector<int64_t> by_year_;
    std::vector<int64_t> year_rank_;
    std::vector<double> gap_dissimilarity_, gap_km_, gap_years_;
    std::vector<double> gap_km_all_, gap_years_all_;  // per gap, NaN where unknown
};

// The matched layout: the candidate with the smallest total mismatch among
// those whose largest fold holds at most max_fold_share of the complete
// plots (the plot-by-plot layout always qualifies).
Layout match_layout(const Split& split, const PlotSearch& search, const GapDistances& gaps,
                    const GapFittingConfig& config, GapFoldLayout& report) {
    const LayoutMatcher matcher(split, search, gaps, config);
    const int64_t n = split.n_cal();
    const int64_t n_folds = std::min(config.n_folds, n);
    const std::vector<LayoutSpec> specs = matcher.specs();
    std::optional<Layout> best;
    report.n_layouts = 0;
    report.n_matching_plots = matcher.n_matching();
    for (std::size_t s = 0; s < specs.size(); ++s) {
        auto [folds, largest] = matcher.deal(specs[s], n_folds);
        const int64_t used = *std::max_element(folds.begin(), folds.end()) + 1;
        const bool plot_by_plot = std::isnan(specs[s].block_km) && std::isnan(specs[s].block_years);
        if (!plot_by_plot &&
            (used < 2 || static_cast<double>(largest) > static_cast<double>(config.max_fold_share) * static_cast<double>(n))) {
            continue;
        }
        ++report.n_layouts;
        Layout layout = matcher.with_buffers(std::move(folds), specs[s]);
        auto mismatch = matcher.mismatch(layout);
        double total = 0.0;
        for (const auto& [axis, w] : mismatch) {
            if (std::isfinite(w)) total += w;
        }
        if (!best || total < report.total_mismatch) {
            report.n_folds = used;
            report.block_km = specs[s].block_km;
            report.block_years = specs[s].block_years;
            report.buffer_km = median_of(layout.buffer_km);
            report.buffer_years = median_of(layout.buffer_years);
            report.mismatch = std::move(mismatch);
            report.total_mismatch = total;
            best = std::move(layout);
        }
    }
    return std::move(*best);
}

// Injected folds as the layout: one per complete plot, every fold from 0 to
// the largest holding a plot.
Layout injected_layout(const torch::Tensor& folds, int64_t n, GapFoldLayout& report) {
    if (!folds.defined() || folds.dim() != 1 || folds.size(0) != n) {
        throw std::invalid_argument("gap fitting: the injected folds must hold one fold per complete plot (" +
                                    std::to_string(n) + ")");
    }
    const torch::Tensor f = folds.to(torch::kCPU, torch::kInt64).contiguous();
    std::vector<int64_t> out(f.data_ptr<int64_t>(), f.data_ptr<int64_t>() + n);
    const int64_t largest = *std::max_element(out.begin(), out.end());
    std::vector<int64_t> counts(static_cast<std::size_t>(std::max<int64_t>(largest, 0) + 1), 0);
    for (const int64_t v : out) {
        if (v < 0) throw std::invalid_argument("gap fitting: an injected fold is negative");
        ++counts[static_cast<std::size_t>(v)];
    }
    if (std::find(counts.begin(), counts.end(), 0) != counts.end() || counts.size() < 2) {
        throw std::invalid_argument("gap fitting: the injected folds must number at least two, each from 0 to the "
                                    "largest holding a plot");
    }
    report.injected = true;
    report.n_folds = static_cast<int64_t>(counts.size());
    return Layout{std::move(out), {}, {}};
}

// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

// One cross-validation over the training plots (rows of the complete plots),
// predicting a set of query plots.
struct Run {
    torch::Tensor train;        // (n_t,) int64 complete rows
    torch::Tensor train_folds;  // (n_t,) int64
    // Per neighbour kind, (n_t, width) complete rows from the other folds,
    // and (n_q, width) complete rows among the training plots; undefined
    // without the kind.
    std::vector<torch::Tensor> train_neighbours;
    std::vector<torch::Tensor> query_neighbours;
    torch::Tensor query_raw;  // (n_q, d)
};

// A group of query plots choosing together: the training plots it is judged
// on and the query plots it predicts.
struct RunGroup {
    const char* name;
    bool spatial;
    torch::Tensor judged;     // (n_t,) bool
    torch::Tensor predicted;  // (n_q,) bool
};

struct RunChoice {
    GapGroupChoice choice;
    torch::Tensor query;  // (n_q, d), the chosen method's value on the predicted rows
};

struct RunResult {
    std::vector<GapNeighbourChoice> neighbours;
    std::vector<RunChoice> groups;
};

// One neighbour kind's median at the k the out-of-fold error chose: on the
// training plots (out of fold) and on the query plots.
struct NeighbourTerm {
    torch::Tensor train;
    torch::Tensor query;
    GapNeighbourChoice choice;
};

// A candidate's out-of-fold values on the training plots, its values on the
// query plots fitted on all of them, and what that fit holds.
struct CandidateFit {
    torch::Tensor train;
    torch::Tensor query;
    std::vector<GapCoefficient> coefficients;
    int64_t n_fitted = 0;
    std::vector<std::pair<std::string, int64_t>> k;
};

class CrossValidation {
public:
    CrossValidation(const Split& split, const Run& run, const GapFittingConfig& config)
        : split_(split), run_(run), config_(config) {
        truth_ = split.truth_cal.index({run.train});
        raw_ = split.raw_cal.index({run.train});
        std::vector<int64_t> folds = values_of_long(run.train_folds);
        std::sort(folds.begin(), folds.end());
        folds.erase(std::unique(folds.begin(), folds.end()), folds.end());
        for (const int64_t f : folds) held_.push_back(run.train_folds == f);
        constant_train_ = torch::full({truth_.size(0), truth_.size(1)}, kNaN, kDouble);
        for (const torch::Tensor& rows : held_) {
            constant_train_.index_put_({rows}, split.plane.centre(truth_.index({rows.logical_not()})));
        }
        centre_ = split.plane.centre(truth_);
    }

    [[nodiscard]] RunResult choose(const std::vector<RunGroup>& groups) {
        RunResult out;
        std::vector<std::optional<NeighbourTerm>> terms(kNeighbourKinds);
        for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
            terms[kind] = neighbour_term(static_cast<NeighbourKind>(kind));
            if (terms[kind]) out.neighbours.push_back(terms[kind]->choice);
        }

        // The candidates this fit can evaluate, in GapMethod order.
        std::vector<const CandidateSpec*> usable;
        for (const auto& spec : kCandidates) {
            const bool asked = config_.candidates.empty() ||
                               std::find(config_.candidates.begin(), config_.candidates.end(), spec.method) !=
                                   config_.candidates.end();
            if (!asked) continue;
            if (spec.reads[kSpatial] && !split_.spatial_axis) continue;
            bool available = true;
            for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
                if (spec.reads[kind] && !terms[kind]) available = false;
            }
            if (available) usable.push_back(&spec);
        }
        std::vector<CandidateFit> fits;
        fits.reserve(usable.size());
        for (const CandidateSpec* spec : usable) fits.push_back(fit(*spec, terms));

        const Plane& plane = split_.plane;
        const int64_t d = truth_.size(1);
        for (const RunGroup& group : groups) {
            // A plot without neighbours in the other folds is read as the
            // local constant, which is all the training plots say about it.
            const torch::Tensor truth = truth_.index({group.judged});
            const torch::Tensor fallback = constant_train_.index({group.judged});
            RunChoice result;
            GapGroupChoice& choice = result.choice;
            choice.group = group.name;
            choice.n_complete = truth.size(0);
            choice.n_gaps = group.predicted.sum().item<int64_t>();
            std::optional<std::size_t> best;
            double best_error = kNaN;
            for (std::size_t c = 0; c < usable.size(); ++c) {
                if (usable[c]->reads[kSpatial] && !group.spatial) continue;
                const torch::Tensor values = fits[c].train.index({group.judged});
                const torch::Tensor judged = torch::where(torch::isfinite(values), values, fallback);
                const double error = plane.aggregate(plane.errors(judged, truth));
                choice.cv_error.emplace_back(usable[c]->method, error);
                if (!best || error < best_error) {
                    best = c;
                    best_error = error;
                }
            }
            if (!best) {
                throw std::invalid_argument(
                    std::string("gap fitting: ") + std::to_string(choice.n_gaps) + " " + group.name +
                    " gaps and no candidate for them" +
                    (group.spatial ? "" : "; gaps without coordinates need a candidate that reads no position"));
            }
            const CandidateFit& chosen = fits[*best];
            choice.method = usable[*best]->method;
            choice.coefficients = chosen.coefficients;
            choice.n_fitted = chosen.n_fitted;
            choice.k = chosen.k;
            const torch::Tensor query = torch::where(torch::isfinite(chosen.query), chosen.query,
                                                     centre_.unsqueeze(0).expand_as(chosen.query));
            result.query = torch::full({query.size(0), d}, kNaN, kDouble);
            result.query.index_put_({group.predicted}, query.index({group.predicted}));
            out.groups.push_back(std::move(result));
        }
        return out;
    }

private:
    static std::vector<int64_t> values_of_long(const torch::Tensor& t) {
        const torch::Tensor c = t.to(torch::kCPU, torch::kInt64).contiguous();
        return {c.data_ptr<int64_t>(), c.data_ptr<int64_t>() + c.numel()};
    }

    // Choose k by the out-of-fold error on the training plots that have
    // neighbours in the other folds; nullopt when the grid holds no k the
    // neighbours allow.
    [[nodiscard]] std::optional<NeighbourTerm> neighbour_term(NeighbourKind kind) const {
        const torch::Tensor& train = run_.train_neighbours[kind];
        if (!train.defined()) return std::nullopt;
        const int64_t width = train.size(1);
        const bool circular = split_.circular();
        const torch::Tensor has = train.select(1, 0) >= 0;
        const torch::Tensor train_has = train.index({has});
        const torch::Tensor truth_has = truth_.index({has});
        NeighbourTerm term;
        term.choice.kind = kKindNames[kind];
        double best_error = kNaN;
        bool any = false;
        for (const int64_t k : config_.k_grid) {
            if (k > width) continue;
            const double error = split_.plane.aggregate(
                split_.plane.errors(neighbour_median(split_.truth_cal, train_has, k, circular), truth_has));
            term.choice.cv_error.emplace_back(k, error);
            // The first k of the grid wins a tie, and stands when no training
            // plot has a neighbour in the other folds (every error NaN).
            if (!any || error < best_error) {
                best_error = error;
                term.choice.k = k;
            }
            any = true;
        }
        if (!any) return std::nullopt;
        term.train = neighbour_median(split_.truth_cal, train, term.choice.k, circular);
        term.query = neighbour_median(split_.truth_cal, run_.query_neighbours[kind], term.choice.k, circular);
        return term;
    }

    [[nodiscard]] CandidateFit fit(const CandidateSpec& spec, const std::vector<std::optional<NeighbourTerm>>& terms) const {
        CandidateFit out;
        const int64_t n_q = run_.query_raw.size(0);
        switch (spec.combination) {
            case Combination::Constant:
                out.train = constant_train_;
                out.query = centre_.unsqueeze(0).expand({n_q, centre_.size(0)}).clone();
                return out;
            case Combination::Raw:
                out.train = raw_;
                out.query = run_.query_raw;
                return out;
            case Combination::Neighbours: {
                const std::size_t kind = spec.reads[kSpecies] ? kSpecies : kSpatial;
                out.train = terms[kind]->train;
                out.query = terms[kind]->query;
                out.k.emplace_back(kKindNames[kind], terms[kind]->choice.k);
                return out;
            }
            case Combination::LeastSquares: {
                std::vector<torch::Tensor> train{raw_};
                std::vector<torch::Tensor> query{run_.query_raw};
                std::vector<std::string> names{"resolve"};
                for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
                    if (!spec.reads[kind]) continue;
                    train.push_back(terms[kind]->train);
                    query.push_back(terms[kind]->query);
                    names.emplace_back(kKindNames[kind]);
                    out.k.emplace_back(kKindNames[kind], terms[kind]->choice.k);
                }
                // Out of fold: each fold predicted by the coefficients the
                // other folds fit.
                out.train = torch::full({truth_.size(0), truth_.size(1)}, kNaN, kDouble);
                for (const torch::Tensor& rows : held_) {
                    const torch::Tensor rest = rows.logical_not();
                    std::vector<torch::Tensor> fit_x, held_x;
                    for (const torch::Tensor& x : train) {
                        fit_x.push_back(x.index({rest}));
                        held_x.push_back(x.index({rows}));
                    }
                    out.train.index_put_({rows},
                                         least_squares(truth_.index({rest}), fit_x, held_x, split_.circular()).test);
                }
                const LeastSquares all = least_squares(truth_, train, query, split_.circular());
                out.query = all.test;
                out.n_fitted = all.n_fitted;
                for (std::size_t j = 0; j < names.size(); ++j) {
                    out.coefficients.push_back(describe_coefficient(names[j], all.coefficients[static_cast<int64_t>(j)]));
                }
                return out;
            }
        }
        throw std::logic_error("gap fitting: unknown combination");
    }

    const Split& split_;
    const Run& run_;
    const GapFittingConfig& config_;
    torch::Tensor truth_;
    torch::Tensor raw_;
    std::vector<torch::Tensor> held_;  // per fold, (n_t,) bool
    torch::Tensor constant_train_;
    torch::Tensor centre_;
};

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
    if (n_folds < 2) {
        throw std::invalid_argument("gap fitting: n_folds must be at least 2, got " + std::to_string(n_folds));
    }
    if (!(static_cast<double>(max_fold_share) > 1.0 / static_cast<double>(n_folds) && max_fold_share <= 1.0f)) {
        throw std::invalid_argument("gap fitting: max_fold_share must lie in (1 / n_folds, 1], got " +
                                    std::to_string(max_fold_share));
    }
    if (matching_sample < 1) {
        throw std::invalid_argument("gap fitting: matching_sample must be at least 1, got " +
                                    std::to_string(matching_sample));
    }
}

torch::Tensor recorded_plots(const torch::Tensor& recorded) {
    if (!recorded.defined() || recorded.dim() < 1 || recorded.dim() > 2) {
        throw std::invalid_argument("recorded values: expected (n,) or (n, 2) values");
    }
    return finite_rows(recorded.dim() == 1 ? recorded.unsqueeze(1) : recorded);
}

std::string gap_method_label(int64_t code) {
    if (code == kGapRecorded) return "recorded";
    if (code == kGapNotImputed) return "not_imputed";
    return enum_to_name(static_cast<GapMethod>(code));
}

json::Value gap_fit_report_to_json(const GapFitReport& report) {
    json::Value out = json::Value::object();
    out.set("target", report.target);
    out.set("scale", enum_to_name(report.scale));
    out.set("n_complete", report.n_complete);
    out.set("n_gaps", report.n_gaps);
    out.set("n_not_imputed", report.n_not_imputed);
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
    if (report.folds.n_folds > 0) {
        const GapFoldLayout& layout = report.folds;
        json::Value f = json::Value::object();
        f.set("n_folds", layout.n_folds);
        f.set("block_km", layout.block_km);
        f.set("block_years", layout.block_years);
        f.set("buffer_km", layout.buffer_km);
        f.set("buffer_years", layout.buffer_years);
        f.set("injected", layout.injected);
        json::Value mismatch = json::Value::object();
        for (const auto& [axis, w] : layout.mismatch) mismatch.set(axis, w);
        f.set("mismatch", std::move(mismatch));
        f.set("total_mismatch", layout.total_mismatch);
        f.set("n_layouts", layout.n_layouts);
        f.set("n_matching_plots", layout.n_matching_plots);
        out.set("folds", std::move(f));
    }
    json::Value neighbours = json::Value::object();
    for (const auto& choice : report.neighbours) {
        json::Value c = json::Value::object();
        c.set("k", choice.k);
        json::Value errors = json::Value::object();
        for (const auto& [k, error] : choice.cv_error) errors.set(std::to_string(k), error);
        c.set("cv_error", std::move(errors));
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
        for (const auto& [method, error] : group.cv_error) errors.set(enum_to_name(method), error);
        g.set("cv_error", std::move(errors));
        g.set("estimated_error", group.estimated_error);
        g.set("n_estimated", group.n_estimated);
        g.set("fold_error", json::Value::array_of(group.fold_error));
        json::Value fold_n = json::Value::array();
        for (const int64_t n : group.fold_n_estimated) fold_n.push_back(n);
        g.set("fold_n_estimated", std::move(fold_n));
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

// The plots of a fit, as rows: the complete plots (every recorded value
// finite), the gaps, and the plots lacking the value that were not imputed
// (a raw value not finite). With `raw_values` undefined every plot counts as
// imputed. A plot recording the value without a raw prediction is in none.
struct Division {
    torch::Tensor complete;
    torch::Tensor gaps;
    torch::Tensor not_imputed;
};

Division divide_plots(const torch::Tensor& recorded_values, const torch::Tensor& raw_values) {
    const torch::Tensor recorded = recorded_plots(recorded_values);
    const torch::Tensor imputed = raw_values.defined() ? recorded_plots(raw_values)
                                                       : torch::ones_like(recorded);
    const torch::Tensor lacking = recorded.logical_not();
    auto rows = [](const torch::Tensor& mask) { return mask.nonzero().select(1, 0); };
    return {rows(recorded.logical_and(imputed)), rows(lacking.logical_and(imputed)),
            rows(lacking.logical_and(imputed.logical_not()))};
}

// One target's plots in the plane, its searches, the gaps' distances and the
// fold layout.
struct Prepared {
    Split split;
    std::unique_ptr<PlotSearch> search;
    GapDistances distances;
    Layout folds;
    GapFoldLayout layout;
};

Prepared prepare(const GapTarget& target, const torch::Tensor& raw_values,
                 const torch::Tensor& recorded_values, const torch::Tensor& cal_index,
                 const torch::Tensor& test_index, const GapPlots& plots, const GapFittingConfig& config,
                 const torch::Tensor* injected) {
    const torch::Tensor recorded_cal = recorded_values.index({cal_index});
    const Plane plane(target, recorded_cal);
    const torch::Tensor species = plots.species.to(torch::kCPU);
    const torch::Tensor coords = plots.coordinates.defined() ? as_double(plots.coordinates) : torch::Tensor();
    const torch::Tensor years = plots.years.defined() ? as_double(plots.years) : torch::Tensor();
    auto rows = [](const torch::Tensor& t, const torch::Tensor& index) {
        return t.defined() ? t.index({index}) : torch::Tensor();
    };

    Prepared out{Split{plane,
                       plane.to(recorded_cal),
                       raw_values.defined() ? plane.to(raw_values.index({cal_index})) : torch::Tensor(),
                       raw_values.defined() ? plane.to(raw_values.index({test_index})) : torch::Tensor(),
                       rows(coords, cal_index),
                       rows(coords, test_index),
                       rows(years, cal_index),
                       rows(years, test_index)},
                 {},
                 {},
                 {},
                 {}};
    Split& split = out.split;
    split.n_test = test_index.size(0);
    // Plots are near or far in space unless the target is the position itself
    // (whose gaps have none), and near or far in time where both sides carry
    // survey years (a year target's gaps carry none).
    const bool located_cal = coords.defined() && finite_rows(split.coords_cal).sum().item<int64_t>() >= 2;
    split.spatial_axis = config.spatial && target.scale != GapScale::Geographic && located_cal;
    split.km_axis = target.scale != GapScale::Geographic && located_cal &&
                    finite_rows(split.coords_test).any().item<bool>();
    split.time_axis = years.defined() && torch::isfinite(split.years_cal).any().item<bool>() &&
                      torch::isfinite(split.years_test).any().item<bool>();

    out.search = std::make_unique<PlotSearch>(split, select_rows(species, cal_index),
                                              select_rows(species, test_index), config.k_grid);
    out.distances = out.search->distances(split);
    out.folds = injected != nullptr ? injected_layout(*injected, split.n_cal(), out.layout)
                                    : match_layout(split, *out.search, out.distances, config, out.layout);
    return out;
}

}  // namespace

GapFolds gap_folds(const GapTarget& target, const torch::Tensor& recorded, const GapPlots& plots,
                   const GapFittingConfig& config) {
    config.validate();
    validate_inputs(target, torch::Tensor(), recorded, plots);
    const torch::Tensor recorded_values = as_double(recorded);
    const Division division = divide_plots(recorded_values, torch::Tensor());
    const torch::Tensor& cal_index = division.complete;
    const torch::Tensor& test_index = division.gaps;
    if (cal_index.size(0) < 2 || test_index.size(0) == 0) {
        throw std::invalid_argument("gap fitting: a fold layout needs at least two complete plots and one gap");
    }
    const Prepared prepared =
        prepare(target, torch::Tensor(), recorded_values, cal_index, test_index, plots, config, nullptr);
    GapFolds out;
    out.complete = cal_index;
    out.gaps = test_index;
    out.folds = torch::tensor(prepared.folds.folds, kLong);
    out.layout = prepared.layout;
    return out;
}

GapFit fit_gaps(const GapTarget& target, const torch::Tensor& raw, const torch::Tensor& recorded,
                const GapPlots& plots, const GapFittingConfig& config, const torch::Tensor* folds) {
    config.validate();
    if (!raw.defined()) throw std::invalid_argument("gap fitting: the raw prediction is required");
    validate_inputs(target, raw, recorded, plots);
    const int64_t n = plots.species.size(0);
    const torch::Tensor raw_values = as_double(raw);
    const torch::Tensor recorded_values = as_double(recorded);
    const Division division = divide_plots(recorded_values, raw_values);
    const torch::Tensor& cal_index = division.complete;
    const torch::Tensor& test_index = division.gaps;

    GapFit out;
    out.report.target = target.name;
    out.report.scale = target.scale;
    out.report.n_complete = cal_index.size(0);
    out.report.n_gaps = test_index.size(0);
    out.report.n_not_imputed = division.not_imputed.size(0);
    out.fitted = recorded_values.clone();
    out.method = torch::full({n}, kGapRecorded, kLong);
    out.method.index_put_({division.not_imputed}, kGapNotImputed);
    if (out.report.n_gaps == 0) return out;
    if (out.report.n_complete < 2) {
        out.report.skipped = "fewer than two complete plots";
        out.fitted.index_put_({test_index}, raw_values.index({test_index}));
        out.method.index_put_({test_index}, static_cast<int64_t>(GapMethod::Resolve));
        return out;
    }

    Prepared prepared = prepare(target, raw_values, recorded_values, cal_index, test_index, plots, config, folds);
    const Split& split = prepared.split;
    const Plane& plane = split.plane;
    const bool layout_matched = !prepared.layout.injected;
    out.report.axes.push_back(summarise_axis("similarity", prepared.distances.similarity, layout_matched));
    out.report.axes.push_back(
        summarise_axis("km", prepared.distances.km, layout_matched && split.km_axis));
    out.report.axes.push_back(
        summarise_axis("years", prepared.distances.years, layout_matched && split.time_axis));
    out.report.folds = prepared.layout;

    const int64_t n_cal = split.n_cal();
    const int64_t n_test = split.n_test;
    const std::vector<int64_t>& layout = prepared.folds.folds;
    const torch::Tensor fold_of = torch::tensor(layout, kLong);
    const torch::Tensor located =
        split.spatial_axis ? finite_rows(split.coords_test) : torch::zeros({n_test}, torch::kBool);
    const torch::Tensor cal_located =
        split.spatial_axis ? finite_rows(split.coords_cal) : torch::ones({n_cal}, torch::kBool);
    const bool any_located = located.any().item<bool>();
    const bool any_unlocated = located.logical_not().any().item<bool>();

    // The groups of plots choosing together for the given query plots: those
    // with a position (judged on the training plots with one), and those
    // without (judged on every training plot, among candidates that read no
    // position).
    auto groups_for = [&](const torch::Tensor& train_located, const torch::Tensor& query_located,
                          const torch::Tensor& query_unlocated) {
        std::vector<RunGroup> out_groups;
        if (any_located) out_groups.push_back({"located", true, train_located, query_located});
        if (any_unlocated) {
            out_groups.push_back({"unlocated", false, torch::ones_like(train_located), query_unlocated});
        }
        return out_groups;
    };

    // Every complete plot predicted from the other folds; the gaps from all.
    Run top;
    top.train = torch::arange(n_cal, kLong);
    top.train_folds = fold_of;
    top.train_neighbours = prepared.search->out_of_fold(prepared.folds, -1);
    top.query_neighbours.resize(kNeighbourKinds);
    for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
        if (top.train_neighbours[kind].defined()) {
            top.query_neighbours[kind] = prepared.search->gap_neighbours(static_cast<NeighbourKind>(kind),
                                                                         top.train_neighbours[kind].size(1));
        }
    }
    top.query_raw = split.raw_test;
    const RunResult chosen =
        CrossValidation(split, top, config).choose(groups_for(cal_located, located, located.logical_not()));
    out.report.neighbours = chosen.neighbours;

    // The whole procedure repeated with each fold held out, scored on it.
    const int64_t n_folds = prepared.layout.n_folds;
    std::vector<torch::Tensor> estimate(chosen.groups.size());
    for (torch::Tensor& e : estimate) e = torch::full({n_cal, split.truth_cal.size(1)}, kNaN, kDouble);
    if (n_folds >= 3) {
        for (int64_t f = 0; f < n_folds; ++f) {
            const torch::Tensor held = fold_of == f;
            const torch::Tensor rest = held.logical_not();
            const std::vector<torch::Tensor> neighbours = prepared.search->out_of_fold(prepared.folds, f);
            Run run;
            run.train = rest.nonzero().select(1, 0);
            run.train_folds = fold_of.index({rest});
            run.train_neighbours.resize(kNeighbourKinds);
            run.query_neighbours.resize(kNeighbourKinds);
            for (std::size_t kind = 0; kind < kNeighbourKinds; ++kind) {
                if (!neighbours[kind].defined()) continue;
                run.train_neighbours[kind] = neighbours[kind].index({rest});
                run.query_neighbours[kind] = neighbours[kind].index({held});
            }
            run.query_raw = split.raw_cal.index({held});
            const torch::Tensor held_located = cal_located.index({held});
            const RunResult nested = CrossValidation(split, run, config)
                                         .choose(groups_for(cal_located.index({rest}), held_located,
                                                            torch::ones_like(held_located)));
            const torch::Tensor held_rows = held.nonzero().select(1, 0);
            for (std::size_t g = 0; g < nested.groups.size(); ++g) {
                const torch::Tensor predicted = torch::isfinite(nested.groups[g].query).all(1);
                estimate[g].index_put_({held_rows.index({predicted})}, nested.groups[g].query.index({predicted}));
            }
        }
    }

    torch::Tensor test_rows = torch::full({n_test, split.truth_cal.size(1)}, kNaN, kDouble);
    torch::Tensor test_method = torch::full({n_test}, kGapRecorded, kLong);
    for (std::size_t g = 0; g < chosen.groups.size(); ++g) {
        GapGroupChoice choice = chosen.groups[g].choice;
        const torch::Tensor scored = torch::isfinite(estimate[g]).all(1);
        choice.n_estimated = scored.sum().item<int64_t>();
        if (choice.n_estimated > 0) {
            choice.estimated_error =
                plane.aggregate(plane.errors(estimate[g].index({scored}), split.truth_cal.index({scored})));
        }
        if (n_folds >= 3) {
            choice.fold_error.assign(static_cast<std::size_t>(n_folds), kNaN);
            choice.fold_n_estimated.assign(static_cast<std::size_t>(n_folds), 0);
            for (int64_t f = 0; f < n_folds; ++f) {
                const torch::Tensor in_fold = scored.logical_and(fold_of == f);
                const int64_t n_in_fold = in_fold.sum().item<int64_t>();
                choice.fold_n_estimated[static_cast<std::size_t>(f)] = n_in_fold;
                if (n_in_fold > 0) {
                    choice.fold_error[static_cast<std::size_t>(f)] = plane.aggregate(
                        plane.errors(estimate[g].index({in_fold}), split.truth_cal.index({in_fold})));
                }
            }
        }
        const torch::Tensor predicted = torch::isfinite(chosen.groups[g].query).all(1);
        test_rows.index_put_({predicted}, chosen.groups[g].query.index({predicted}));
        test_method.index_put_({predicted}, static_cast<int64_t>(choice.method));
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

    const std::vector<std::vector<double>> values =
        read_plot_columns(header, species, columns.plot_id, names, plot_ids, "gap fitting");

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
