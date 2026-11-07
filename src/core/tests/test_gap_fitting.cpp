// Gap fitting (gap_fitting.hpp): recovery of known relations between the raw
// prediction, the neighbours and the truth; matched neighbours no nearer than
// the gaps in km and in years; the choice of spatial neighbours where the gaps
// are near and not where they are a far region; bearings and positions; the
// tables reader and the single-model entry point.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/gap_fitting.hpp"
#include "resolve/geo.hpp"
#include "resolve/row_source.hpp"
#include "resolve/sparse_rows.hpp"

#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace resolve;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const auto kDouble = torch::TensorOptions().dtype(torch::kFloat64);

// The recorded values with the plots of `gaps` missing.
torch::Tensor with_gaps(const torch::Tensor& truth, const torch::Tensor& gaps) {
    torch::Tensor out = truth.clone();
    if (out.dim() == 1) {
        out.masked_fill_(gaps, kNaN);
    } else {
        out.masked_fill_(gaps.unsqueeze(1).expand_as(out), kNaN);
    }
    return out;
}

// No species at all: every plot is as similar as every other.
torch::Tensor no_species(int64_t n) { return torch::zeros({n, 1}, torch::kFloat32); }

// `per_plot` species drawn per plot from a pool of `n_species`.
torch::Tensor random_species(int64_t n, int64_t n_species = 60, int64_t per_plot = 12) {
    const torch::Tensor order = torch::rand({n, n_species}).argsort(/*dim=*/1, /*descending=*/false).slice(1, 0, per_plot);
    return torch::zeros({n, n_species}, torch::kFloat32).scatter_(1, order, 1.0f);
}

GapFittingConfig only(std::vector<GapMethod> candidates) {
    GapFittingConfig config;
    config.candidates = std::move(candidates);
    return config;
}

const GapGroupChoice& group(const GapFit& fit, const std::string& name) {
    for (const auto& g : fit.report.groups) {
        if (g.group == name) return g;
    }
    FAIL("no group " << name);
    return fit.report.groups.front();
}

double coefficient(const GapGroupChoice& g, const std::string& term) {
    for (const auto& c : g.coefficients) {
        if (c.term == term) return c.value;
    }
    FAIL("no coefficient " << term);
    return kNaN;
}

double gap_mae(const GapFit& fit, const torch::Tensor& truth, const torch::Tensor& gaps) {
    return (fit.fitted.index({gaps}) - truth.index({gaps})).abs().mean().item<double>();
}

// Plots in two survey campaigns 300 km apart, each surveyed in its own year.
struct Campaigns {
    torch::Tensor coordinates;  // (n, 2)
    torch::Tensor year;         // (n,)
    torch::Tensor campaign;     // (n,) 0 or 1
};

Campaigns two_campaigns(int64_t n) {
    Campaigns out;
    out.campaign = torch::randint(0, 2, {n}, torch::kInt64);
    const torch::Tensor c = out.campaign.to(torch::kFloat64);
    const torch::Tensor lat = 47.0 + 3.0 * c + 0.05 * torch::randn({n}, kDouble);
    const torch::Tensor lon = 14.0 + 0.05 * torch::randn({n}, kDouble);
    out.coordinates = torch::stack({lat, lon}, 1);
    out.year = 1965.0 + 40.0 * c;
    return out;
}

// Each campaign records its own pool of species.
torch::Tensor campaign_species(const torch::Tensor& campaign, int64_t pool = 20, int64_t per_plot = 6) {
    const int64_t n = campaign.size(0);
    const torch::Tensor order = torch::rand({n, pool}).argsort(/*dim=*/1, /*descending=*/false).slice(1, 0, per_plot);
    const torch::Tensor columns = order + campaign.unsqueeze(1) * pool;
    return torch::zeros({n, 2 * pool}, torch::kFloat32).scatter_(1, columns, 1.0f);
}

}  // namespace

TEST_CASE("offset_slope recovers the local relation of raw prediction and truth", "[gap_fitting]") {
    const int seed = GENERATE(0, 1, 2, 3, 4);
    const double slope = GENERATE(0.0, 0.4, 1.0);
    torch::manual_seed(seed);
    const int64_t n = 3000;
    const torch::Tensor pred = 1950.0 + 70.0 * torch::rand({n}, kDouble);
    const torch::Tensor truth = 1990.0 + slope * (pred - pred.mean()) + 8.0 + 3.0 * torch::randn({n}, kDouble);
    const torch::Tensor gaps = torch::arange(n) >= 2400;
    const GapFit fit = fit_gaps({"year"}, pred, with_gaps(truth, gaps), {no_species(n)},
                                only({GapMethod::OffsetSlope}));
    const GapGroupChoice& g = group(fit, "unlocated");
    CHECK(g.method == GapMethod::OffsetSlope);
    CHECK_THAT(coefficient(g, "resolve"), WithinAbs(slope, 0.02));
    CHECK(g.n_fitted == 2400);
    // Complete plots keep their recorded value.
    CHECK(torch::equal(fit.fitted.slice(0, 0, 2400), truth.slice(0, 0, 2400)));
    CHECK((fit.method.slice(0, 0, 2400) == kGapRecorded).all().item<bool>());
    CHECK((fit.method.slice(0, 2400) == static_cast<int64_t>(GapMethod::OffsetSlope)).all().item<bool>());
}

TEST_CASE("A bearing offset is recovered as a complex coefficient: a turn and no shrink",
          "[gap_fitting]") {
    const int seed = GENERATE(0, 1, 2, 3, 4);
    torch::manual_seed(seed);
    const int64_t n = 2000;
    const torch::Tensor truth = 360.0 * torch::rand({n}, kDouble);
    const torch::Tensor pred = torch::remainder(truth - 40.0 + 5.0 * torch::randn({n}, kDouble), 360.0);
    const torch::Tensor gaps = torch::arange(n) >= 1500;
    const GapFit fit = fit_gaps({"aspect", GapScale::Circular, 360.0}, pred, with_gaps(truth, gaps),
                                {no_species(n)}, only({GapMethod::OffsetSlope}));
    const GapGroupChoice& g = group(fit, "unlocated");
    REQUIRE(g.coefficients.size() == 1);
    REQUIRE(g.coefficients[0].complex);
    CHECK_THAT(g.coefficients[0].angle_deg, WithinAbs(40.0, 1.0));
    CHECK_THAT(g.coefficients[0].modulus, WithinAbs(1.0, 0.05));
    const torch::Tensor d = torch::remainder((fit.fitted.index({gaps}) - truth.index({gaps})).abs(), 360.0);
    CHECK(torch::minimum(d, 360.0 - d).mean().item<double>() < 6.0);
}

TEST_CASE("The species stack recovers the weights of the raw prediction and the neighbours",
          "[gap_fitting]") {
    const int seed = GENERATE(0, 1, 2, 3, 4);
    torch::manual_seed(seed);
    const int64_t n = 3000;
    const Campaigns c = two_campaigns(n);
    const torch::Tensor pred = 1985.0 + 15.0 * torch::randn({n}, kDouble);
    const torch::Tensor truth = 1985.0 + 0.5 * (pred - 1985.0) + 0.8 * (c.year - 1985.0) +
                                torch::randn({n}, kDouble);
    const torch::Tensor gaps = torch::rand({n}) < 0.2;
    const GapFit fit = fit_gaps({"year"}, pred, with_gaps(truth, gaps), {campaign_species(c.campaign)},
                                only({GapMethod::StackSpecies}));
    const GapGroupChoice& g = group(fit, "unlocated");
    CHECK_THAT(coefficient(g, "resolve"), WithinAbs(0.5, 0.05));
    // The neighbour median is of recorded values, which already carry the
    // campaign component at its weight, so it enters at weight 1.
    CHECK_THAT(coefficient(g, "species"), WithinAbs(1.0, 0.05));
}

TEST_CASE("The spatial stack recovers the weight of the nearest plots", "[gap_fitting]") {
    const int seed = GENERATE(0, 1, 2, 3, 4);
    torch::manual_seed(seed);
    const int64_t n = 3000;
    const Campaigns c = two_campaigns(n);
    const torch::Tensor pred = 1985.0 + 15.0 * torch::randn({n}, kDouble);
    const torch::Tensor truth = 1985.0 + 0.5 * (pred - 1985.0) + 0.8 * (c.year - 1985.0) +
                                torch::randn({n}, kDouble);
    const torch::Tensor gaps = torch::rand({n}) < 0.2;
    const GapFit fit = fit_gaps({"altitude"}, pred, with_gaps(truth, gaps),
                                {random_species(n), c.coordinates},
                                only({GapMethod::StackSpeciesSpatial}));
    const GapGroupChoice& g = group(fit, "located");
    CHECK(g.method == GapMethod::StackSpeciesSpatial);
    CHECK_THAT(coefficient(g, "resolve"), WithinAbs(0.5, 0.05));
    CHECK_THAT(coefficient(g, "spatial"), WithinAbs(1.0, 0.05));
    CHECK(std::abs(coefficient(g, "species")) < 0.1);
}

TEST_CASE("Matched neighbours are no nearer in km than the gaps are to the complete plots",
          "[gap_fitting]") {
    torch::manual_seed(4);
    // Complete plots fill a band; every gap lies 100-150 km north of it.
    const int64_t n_cal = 800;
    const int64_t n_gap = 200;
    const int64_t n = n_cal + n_gap;
    const torch::Tensor lat = torch::cat({45.0 + 2.0 * torch::rand({n_cal}, kDouble),
                                          47.9 + 0.45 * torch::rand({n_gap}, kDouble)});
    const torch::Tensor lon = 14.0 + torch::rand({n}, kDouble);
    const torch::Tensor coords = torch::stack({lat, lon}, 1);
    const torch::Tensor gaps = torch::arange(n) >= n_cal;
    const torch::Tensor recorded = with_gaps(torch::zeros({n}, kDouble), gaps);
    const GapPlots plots{random_species(n), coords};

    const MatchedGapNeighbours m = matched_gap_neighbours({"altitude"}, recorded, plots);
    REQUIRE(m.spatial.defined());
    const torch::Tensor cal = coords.index({m.complete});
    const double gap_km_min =
        nearest_by_great_circle(coords.index({m.gaps}), cal, 1).km.min().item<double>();
    REQUIRE(gap_km_min > 90.0);
    for (const torch::Tensor& nb : {m.species, m.spatial}) {
        const torch::Tensor own = torch::arange(n_cal).unsqueeze(1);
        CHECK_FALSE(nb.eq(own).any().item<bool>());
        const torch::Tensor has = nb >= 0;
        REQUIRE(has.any().item<bool>());
        const torch::Tensor i = own.expand_as(nb).index({has});
        const torch::Tensor j = nb.index({has});
        const torch::Tensor km = great_circle_km(cal.index({i, 0}), cal.index({i, 1}), cal.index({j, 0}),
                                                 cal.index({j, 1}));
        CHECK(km.min().item<double>() >= gap_km_min - 1e-3);
    }
    // Without the hiding, each complete plot's nearest neighbour is far nearer.
    const PlaceNeighbours plain = nearest_by_great_circle(cal, cal, 1, /*exclude_self=*/true);
    CHECK(plain.km.max().item<double>() < gap_km_min);
}

TEST_CASE("Matched neighbours are no nearer in survey year than the gaps", "[gap_fitting]") {
    torch::manual_seed(5);
    // The gaps were surveyed 1930-1940, the complete plots 1960-2020.
    const int64_t n_cal = 600;
    const int64_t n_gap = 150;
    const int64_t n = n_cal + n_gap;
    const torch::Tensor year = torch::cat({1960.0 + 60.0 * torch::rand({n_cal}, kDouble),
                                           1930.0 + 10.0 * torch::rand({n_gap}, kDouble)});
    const torch::Tensor gaps = torch::arange(n) >= n_cal;
    GapPlots plots{random_species(n)};
    plots.years = year;
    const MatchedGapNeighbours m =
        matched_gap_neighbours({"altitude"}, with_gaps(torch::zeros({n}, kDouble), gaps), plots);
    CHECK_FALSE(m.spatial.defined());
    REQUIRE(m.radii.years.defined());
    const torch::Tensor cal_year = year.index({m.complete});
    const double gap_years_min = (year.index({m.gaps}).unsqueeze(1) - cal_year.unsqueeze(0))
                                     .abs()
                                     .amin(1)
                                     .min()
                                     .item<double>();
    const torch::Tensor has = m.species >= 0;
    REQUIRE(has.any().item<bool>());
    const torch::Tensor i = torch::arange(n_cal).unsqueeze(1).expand_as(m.species).index({has});
    const torch::Tensor j = m.species.index({has});
    CHECK((cal_year.index({i}) - cal_year.index({j})).abs().min().item<double>() >= gap_years_min - 1e-9);
}

TEST_CASE("Spatial neighbours are chosen where the gaps are near and not for a far region",
          "[gap_fitting]") {
    const int seed = GENERATE(0, 1, 2, 3, 4);
    torch::manual_seed(seed);
    // A value that changes every 55 km north-south, recorded with little
    // noise, and a raw prediction of it with much more.
    const int64_t n = 3000;
    const torch::Tensor lat = 45.0 + 5.0 * torch::rand({n}, kDouble);
    const torch::Tensor lon = 14.0 + 0.5 * torch::rand({n}, kDouble);
    const torch::Tensor truth = 1985.0 + 30.0 * torch::sin(2.0 * std::numbers::pi * lat / 0.5) +
                                torch::randn({n}, kDouble);
    const torch::Tensor pred = truth + 6.0 * torch::randn({n}, kDouble);
    const torch::Tensor coords = torch::stack({lat, lon}, 1);

    // Scattered gaps: a gap's nearest complete plot is a few km away.
    {
        const torch::Tensor gaps = torch::rand({n}) < 0.2;
        const GapPlots plots{no_species(n), coords};
        const torch::Tensor recorded = with_gaps(truth, gaps);
        const GapFit fit = fit_gaps({"altitude"}, pred, recorded, plots);
        const GapGroupChoice& g = group(fit, "located");
        CHECK((g.method == GapMethod::KnnSpatial || g.method == GapMethod::StackSpeciesSpatial));
        const double best_spatial = std::min(
            gap_mae(fit_gaps({"altitude"}, pred, recorded, plots, only({GapMethod::KnnSpatial})), truth, gaps),
            gap_mae(fit_gaps({"altitude"}, pred, recorded, plots, only({GapMethod::StackSpeciesSpatial})),
                    truth, gaps));
        CHECK(gap_mae(fit, truth, gaps) <= 1.1 * best_spatial);
    }

    // A missing region over 55 km from every complete plot: the nearest
    // complete plots no longer carry the field, and the matched test shows it.
    {
        const torch::Tensor keep = (lat < 48.5) | (lat > 49.0);
        const torch::Tensor idx = keep.nonzero().select(1, 0);
        const int64_t m = idx.size(0);
        const torch::Tensor t = truth.index({idx});
        const torch::Tensor p = pred.index({idx});
        const torch::Tensor gaps = lat.index({idx}) > 49.0;
        const GapPlots plots{no_species(m), coords.index({idx})};
        const torch::Tensor recorded = with_gaps(t, gaps);
        const GapFit fit = fit_gaps({"altitude"}, p, recorded, plots);
        const GapGroupChoice& g = group(fit, "located");
        // The nearest complete plots are not read as they are: the spatial
        // median alone is never chosen, and a stack chosen at the gaps'
        // distance gives it little weight.
        CHECK(g.method != GapMethod::KnnSpatial);
        if (g.method == GapMethod::StackSpeciesSpatial) {
            CHECK(std::abs(coefficient(g, "spatial")) < 0.25);
        }
        // The same stack judged by plain leave-one-out (nothing hidden).
        const int64_t n_complete = fit.report.n_complete;
        HidingRadii nothing_hidden;
        nothing_hidden.similarity = torch::full({n_complete}, 2.0, kDouble);
        nothing_hidden.km = torch::zeros({n_complete}, kDouble);
        const GapFit plain = fit_gaps({"altitude"}, p, recorded, plots,
                                      only({GapMethod::StackSpeciesSpatial}), &nothing_hidden);
        const GapFit calibrated = fit_gaps({"altitude"}, p, recorded, plots, only({GapMethod::OffsetSlope}));
        CHECK(gap_mae(fit, t, gaps) < 0.8 * gap_mae(plain, t, gaps));
        CHECK(gap_mae(fit, t, gaps) <= 1.1 * gap_mae(calibrated, t, gaps));
    }
}

TEST_CASE("A position target is fitted in the plane by species neighbours, never by its own "
          "position", "[gap_fitting]") {
    torch::manual_seed(9);
    const int64_t n = 1200;
    const Campaigns c = two_campaigns(n);
    // The raw prediction places each plot 40 km off at random.
    const torch::Tensor raw = c.coordinates + 0.36 * torch::randn({n, 2}, kDouble);
    const torch::Tensor gaps = torch::rand({n}) < 0.2;
    const torch::Tensor recorded = with_gaps(c.coordinates, gaps);
    // The caller passes the recorded position as the plots' coordinates too:
    // the gaps carry none, so no spatial axis is read.
    const GapPlots plots{campaign_species(c.campaign), recorded};
    const GapFit fit = fit_gaps({"location", GapScale::Geographic}, raw, recorded, plots);
    REQUIRE(fit.report.groups.size() == 1);
    const GapGroupChoice& g = group(fit, "unlocated");
    CHECK((g.method == GapMethod::KnnSpecies || g.method == GapMethod::StackSpecies));
    CHECK_FALSE(fit.report.axes[1].matched);
    const torch::Tensor fitted = fit.fitted.index({gaps});
    const torch::Tensor truth = c.coordinates.index({gaps});
    const torch::Tensor raw_gaps = raw.index({gaps});
    const double fitted_km = great_circle_km(fitted.select(1, 0), fitted.select(1, 1), truth.select(1, 0),
                                             truth.select(1, 1)).median().item<double>();
    const double raw_km = great_circle_km(raw_gaps.select(1, 0), raw_gaps.select(1, 1), truth.select(1, 0),
                                          truth.select(1, 1)).median().item<double>();
    CHECK(fitted_km < 0.5 * raw_km);
}

TEST_CASE("A year target reads no survey-year axis: its gaps carry no year", "[gap_fitting]") {
    torch::manual_seed(2);
    const int64_t n = 500;
    const torch::Tensor truth = 1950.0 + 60.0 * torch::rand({n}, kDouble);
    const torch::Tensor gaps = torch::rand({n}) < 0.2;
    const torch::Tensor recorded = with_gaps(truth, gaps);
    GapPlots plots{random_species(n)};
    plots.years = recorded;
    const GapFit fit = fit_gaps({"year"}, truth + 5.0 * torch::randn({n}, kDouble), recorded, plots);
    CHECK(fit.report.axes[2].axis == "years");
    CHECK_FALSE(fit.report.axes[2].matched);
    CHECK(fit.report.axes[0].matched);
}

TEST_CASE("Gap fitting without gaps or without complete plots, and its determinism",
          "[gap_fitting]") {
    torch::manual_seed(1);
    const int64_t n = 300;
    const torch::Tensor truth = torch::randn({n}, kDouble);
    const torch::Tensor raw = truth + 0.1;

    const GapFit none = fit_gaps({"slope"}, raw, truth, {no_species(n)});
    CHECK(torch::equal(none.fitted, truth));
    CHECK(none.report.groups.empty());

    torch::Tensor one = torch::full({n}, kNaN, kDouble);
    one[0] = 1.0;
    const GapFit skipped = fit_gaps({"slope"}, raw, one, {no_species(n)});
    CHECK_FALSE(skipped.report.skipped.empty());
    CHECK(torch::equal(skipped.fitted.slice(0, 1), raw.slice(0, 1)));

    const torch::Tensor recorded = with_gaps(truth, torch::rand({n}) < 0.3);
    const GapPlots plots{random_species(n)};
    const GapFit a = fit_gaps({"slope"}, raw, recorded, plots);
    const GapFit b = fit_gaps({"slope"}, raw, recorded, plots);
    CHECK(torch::equal(a.fitted, b.fitted));
    // The drawn radii, injected, reproduce the fit.
    const MatchedGapNeighbours m = matched_gap_neighbours({"slope"}, recorded, plots);
    const GapFit c = fit_gaps({"slope"}, raw, recorded, plots, {}, &m.radii);
    CHECK(torch::equal(a.fitted, c.fitted));

    CHECK_THROWS_AS(fit_gaps({"slope"}, raw.slice(0, 1), recorded, plots), std::invalid_argument);
    GapFittingConfig bad;
    bad.k_grid = {0};
    CHECK_THROWS_AS(fit_gaps({"slope"}, raw, recorded, plots, bad), std::invalid_argument);
    CHECK_THROWS_AS(fit_gaps({"slope"}, raw, recorded, plots, only({GapMethod::KnnSpatial})),
                    std::invalid_argument);
    const json::Value report = gap_fit_report_to_json(a.report);
    CHECK(report.at("groups").contains("unlocated"));
    CHECK(gap_method_label(kGapRecorded) == "recorded");
}

TEST_CASE("read_gap_tables reads every named species, positions, years and recorded values",
          "[gap_fitting]") {
    const ColumnTable header({"plot", "Year", "Lat", "Lon", "alt"},
                             {{"a", "b", "c"}, {"1990", "", "2001"}, {"46", "47", "NA"},
                              {"14", "15", "16"}, {"300", "NA", "500"}});
    const ColumnTable species({"plot", "taxon"},
                              {{"a", "a", "a", "b", "c", "c"},
                               {"Poa annua", "Poa annua", "Bellis perennis", "Poa annua", "", "Abies alba"}});
    GapColumns columns;
    columns.plot_id = "plot";
    columns.species_id = "taxon";
    columns.latitude = "Lat";
    columns.longitude = "Lon";
    columns.survey_year = "Year";
    columns.recorded = {{"altitude", {"alt"}}, {"location", {"Lat", "Lon"}}};
    const GapTables t = read_gap_tables(TableSource::table(header), TableSource::table(species), columns,
                                        {"c", "a", "b"});
    const torch::Tensor dense = sparse_rows_to_dense(t.plots.species);
    REQUIRE(dense.sizes() == std::vector<int64_t>{3, 3});
    CHECK(dense.sum(1).equal(torch::tensor({1.0f, 2.0f, 1.0f})));
    CHECK(dense.max().item<float>() == 1.0f);
    CHECK(t.plots.years[1].item<double>() == 1990.0);
    CHECK(std::isnan(t.plots.years[2].item<double>()));
    CHECK(std::isnan(t.plots.coordinates[0][0].item<double>()));
    CHECK(t.recorded.at("altitude")[0].item<double>() == 500.0);
    CHECK(t.recorded.at("location").sizes() == std::vector<int64_t>{3, 2});
    CHECK_THROWS(read_gap_tables(TableSource::table(header), TableSource::table(species), columns,
                                 {"a", "zzz"}));

    // One table: plot-level columns from each plot's first row.
    const ColumnTable single({"plot", "taxon", "alt"},
                             {{"a", "a", "b"}, {"x", "y", "x"}, {"10", "99", "20"}});
    GapColumns one;
    one.plot_id = "plot";
    one.species_id = "taxon";
    one.recorded = {{"altitude", {"alt"}}};
    const GapTables s = read_gap_tables(std::nullopt, TableSource::table(single), one, {"a", "b"});
    CHECK(s.recorded.at("altitude")[0].item<double>() == 10.0);
    CHECK(s.recorded.at("altitude")[1].item<double>() == 20.0);
}

TEST_CASE("fit_gaps on a model's predictions reads its heads by request", "[gap_fitting]") {
    torch::manual_seed(6);
    const int64_t n = 400;
    const torch::Tensor truth = 360.0 * torch::rand({n}, kDouble);
    const torch::Tensor angle = (truth - 30.0) * std::numbers::pi / 180.0;
    ResolvePredictions predictions;
    predictions.predictions["aspect_sin"] = torch::sin(angle).to(torch::kFloat32);
    predictions.predictions["aspect_cos"] = torch::cos(angle).to(torch::kFloat32);
    GapTables tables;
    tables.plots.species = no_species(n);
    tables.recorded["aspect"] = with_gaps(truth, torch::arange(n) >= 300);
    const std::vector<GapFit> fits = fit_gaps(
        predictions, tables, {{{"aspect", GapScale::Circular, 360.0}, {"aspect_sin", "aspect_cos"}}},
        only({GapMethod::OffsetSlope}));
    REQUIRE(fits.size() == 1);
    CHECK_THAT(fits[0].report.groups[0].coefficients[0].angle_deg, WithinAbs(30.0, 0.5));
    CHECK_THROWS_AS(fit_gaps(predictions, tables, {{{"aspect", GapScale::Circular}, {"missing"}}}),
                    std::invalid_argument);
}
