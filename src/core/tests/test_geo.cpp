// Positions on the Earth (geo.hpp): great-circle distance, the geographic
// median and the ellipsoidal Lambert azimuthal equal-area projection, against
// EPSG 3035 and values computed independently.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/geo.hpp"

#include <cmath>
#include <limits>
#include <numbers>

using namespace resolve;
using Catch::Matchers::WithinAbs;

namespace {

torch::Tensor degrees(std::initializer_list<double> values) {
    return torch::tensor(std::vector<double>(values), torch::kFloat64);
}

}  // namespace

TEST_CASE("great_circle_km measures a degree of a meridian and broadcasts", "[geo]") {
    const double one_degree = kEarthRadiusKm * std::numbers::pi / 180.0;
    const torch::Tensor d = great_circle_km(degrees({45.0}), degrees({10.0}), degrees({46.0}), degrees({10.0}));
    CHECK_THAT(d.item<double>(), WithinAbs(one_degree, 1e-9));

    const torch::Tensor grid = great_circle_km(degrees({0.0, 10.0}).unsqueeze(1), degrees({0.0, 0.0}).unsqueeze(1),
                                               degrees({0.0, 10.0, 20.0}).unsqueeze(0),
                                               degrees({0.0, 0.0, 0.0}).unsqueeze(0));
    REQUIRE(grid.sizes() == std::vector<int64_t>{2, 3});
    CHECK_THAT(grid[1][2].item<double>(), WithinAbs(10.0 * one_degree, 1e-9));
    CHECK_THAT(grid[1][1].item<double>(), WithinAbs(0.0, 1e-12));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK(std::isnan(great_circle_km(degrees({nan}), degrees({0.0}), degrees({1.0}), degrees({0.0})).item<double>()));
}

TEST_CASE("The projection reproduces EPSG 3035 and round-trips", "[geo]") {
    // EPSG 3035 is this projection about (52 N, 10 E) with a false origin of
    // (4321000, 3210000); Vienna lies at about (4794076, 2807806).
    const LambertAzimuthalEqualArea europe(52.0, 10.0);
    const auto [e, n] = europe.forward(degrees({52.0, 48.2}), degrees({10.0, 16.37}));
    CHECK_THAT(e[0].item<double>(), WithinAbs(0.0, 1e-6));
    CHECK_THAT(n[0].item<double>(), WithinAbs(0.0, 1e-6));
    CHECK_THAT(e[1].item<double>() + 4321000.0, WithinAbs(4794075.83587123, 1e-4));
    CHECK_THAT(n[1].item<double>() + 3210000.0, WithinAbs(2807806.4230653, 1e-4));

    const LambertAzimuthalEqualArea local(46.0, 14.0);
    const torch::Tensor lat = degrees({47.0, 40.5, 60.0, 46.0});
    const torch::Tensor lon = degrees({15.0, 2.25, 30.0, 14.0});
    const auto [east, north] = local.forward(lat, lon);
    CHECK_THAT(east[0].item<double>(), WithinAbs(76057.89583493, 1e-4));
    CHECK_THAT(north[0].item<double>(), WithinAbs(111636.51308403, 1e-4));
    CHECK_THAT(east[2].item<double>(), WithinAbs(891172.44334203, 1e-4));
    CHECK_THAT(north[2].item<double>(), WithinAbs(1649063.4623858, 1e-4));
    const auto [back_lat, back_lon] = local.inverse(east, north);
    CHECK(torch::allclose(back_lat, lat, 0.0, 1e-9));
    CHECK(torch::allclose(back_lon, lon, 0.0, 1e-9));
}

TEST_CASE("geographic_median is the point of least mean distance, ignoring missing positions",
          "[geo]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Three points whose angles are all below 120 degrees: the median is the
    // interior Fermat point, not one of them.
    const auto [lat, lon] = geographic_median(degrees({46.0, 47.0, 46.5, nan}),
                                              degrees({14.0, 14.0, 15.0, 3.0}));
    CHECK_THAT(lat, WithinAbs(46.502470682601775, 1e-9));
    CHECK_THAT(lon, WithinAbs(14.41939501364955, 1e-9));
    CHECK_THROWS_AS(geographic_median(degrees({nan}), degrees({nan})), std::invalid_argument);
}
