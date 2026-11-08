#include "resolve/geo.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace resolve {

namespace {

constexpr double kRadians = std::numbers::pi / 180.0;

torch::Tensor as_double(const torch::Tensor& t) {
    return t.to(torch::kCPU, torch::kFloat64);
}

}  // namespace

torch::Tensor great_circle_km(const torch::Tensor& lat1, const torch::Tensor& lon1,
                              const torch::Tensor& lat2, const torch::Tensor& lon2) {
    const torch::Tensor p1 = as_double(lat1) * kRadians;
    const torch::Tensor p2 = as_double(lat2) * kRadians;
    const torch::Tensor dl = (as_double(lon2) - as_double(lon1)) * kRadians;
    const torch::Tensor a = torch::sin((p2 - p1) / 2).pow(2) +
                            torch::cos(p1) * torch::cos(p2) * torch::sin(dl / 2).pow(2);
    return 2.0 * kEarthRadiusKm * torch::asin(torch::sqrt(torch::clamp(a, 0.0, 1.0)));
}

double great_circle_km(double lat1, double lon1, double lat2, double lon2) {
    const double p1 = lat1 * kRadians;
    const double p2 = lat2 * kRadians;
    const double dl = (lon2 - lon1) * kRadians;
    const double s1 = std::sin((p2 - p1) / 2);
    const double s2 = std::sin(dl / 2);
    const double a = s1 * s1 + std::cos(p1) * std::cos(p2) * s2 * s2;
    return 2.0 * kEarthRadiusKm * std::asin(std::sqrt(std::clamp(a, 0.0, 1.0)));
}

std::pair<double, double> geographic_median(const torch::Tensor& lat, const torch::Tensor& lon,
                                            int iterations) {
    const torch::Tensor la = as_double(lat).reshape({-1});
    const torch::Tensor lo = as_double(lon).reshape({-1});
    const torch::Tensor finite = torch::isfinite(la) & torch::isfinite(lo);
    if (!finite.any().item<bool>()) {
        throw std::invalid_argument("geographic_median: no position has finite coordinates");
    }
    const torch::Tensor phi = la.index({finite}) * kRadians;
    const torch::Tensor lambda = lo.index({finite}) * kRadians;
    const torch::Tensor xyz = torch::stack(
        {torch::cos(phi) * torch::cos(lambda), torch::cos(phi) * torch::sin(lambda), torch::sin(phi)},
        1);
    torch::Tensor point = xyz.mean(0);
    point = point / point.norm();
    for (int i = 0; i < iterations; ++i) {
        const torch::Tensor angle = torch::acos(torch::clamp(xyz.matmul(point), -1.0, 1.0));
        const torch::Tensor weight = 1.0 / torch::clamp_min(angle, 1e-9);
        torch::Tensor next = (xyz * weight.unsqueeze(1)).sum(0);
        next = next / next.norm();
        if ((next - point).norm().item<double>() < 1e-12) break;
        point = next;
    }
    const auto p = point.accessor<double, 1>();
    return {std::asin(p[2]) / kRadians, std::atan2(p[1], p[0]) / kRadians};
}

LambertAzimuthalEqualArea::LambertAzimuthalEqualArea(double latitude_of_origin,
                                                     double longitude_of_origin,
                                                     double semi_major_m,
                                                     double inverse_flattening)
    : lat0_(latitude_of_origin), lon0_(longitude_of_origin), a_(semi_major_m) {
    const double f = 1.0 / inverse_flattening;
    e2_ = f * (2.0 - f);
    e_ = std::sqrt(e2_);
    q_pole_ = q(1.0);
    radius_authalic_ = a_ * std::sqrt(q_pole_ / 2.0);
    const double phi0 = lat0_ * kRadians;
    const double sin0 = std::sin(phi0);
    beta0_ = std::asin(std::clamp(q(sin0) / q_pole_, -1.0, 1.0));
    d_ = a_ * std::cos(phi0) /
         (std::sqrt(1.0 - e2_ * sin0 * sin0) * radius_authalic_ * std::cos(beta0_));
}

double LambertAzimuthalEqualArea::q(double sin_phi) const {
    const double es = e_ * sin_phi;
    return (1.0 - e2_) *
           (sin_phi / (1.0 - e2_ * sin_phi * sin_phi) - std::log((1.0 - es) / (1.0 + es)) / (2.0 * e_));
}

torch::Tensor LambertAzimuthalEqualArea::q(const torch::Tensor& sin_phi) const {
    const torch::Tensor es = e_ * sin_phi;
    return (1.0 - e2_) *
           (sin_phi / (1.0 - e2_ * sin_phi * sin_phi) - torch::log((1.0 - es) / (1.0 + es)) / (2.0 * e_));
}

torch::Tensor LambertAzimuthalEqualArea::geodetic_latitude(const torch::Tensor& beta) const {
    constexpr double kTolerance = 1e-13;
    constexpr int kMaxIterations = 30;
    const torch::Tensor target = q_pole_ * torch::sin(beta);
    torch::Tensor phi = beta.clone();
    for (int i = 0; i < kMaxIterations; ++i) {
        const torch::Tensor sin_phi = torch::sin(phi);
        const torch::Tensor denominator = 1.0 - e2_ * sin_phi * sin_phi;
        const torch::Tensor residual = q(sin_phi) - target;
        const torch::Tensor derivative = 2.0 * (1.0 - e2_) * torch::cos(phi) / (denominator * denominator);
        const torch::Tensor step =
            torch::where(derivative.abs() > 1e-12, residual / derivative, torch::zeros_like(residual));
        phi = phi - step;
        // NaN steps (a NaN input) do not hold the iteration open.
        if (!(step.abs() >= kTolerance).any().item<bool>()) break;
    }
    return torch::clamp(phi, -std::numbers::pi / 2, std::numbers::pi / 2);
}

std::pair<torch::Tensor, torch::Tensor> LambertAzimuthalEqualArea::forward(
    const torch::Tensor& latitude, const torch::Tensor& longitude) const {
    const torch::Tensor phi = as_double(latitude) * kRadians;
    const torch::Tensor delta_lambda = (as_double(longitude) - lon0_) * kRadians;
    const torch::Tensor beta = torch::asin(torch::clamp(q(torch::sin(phi)) / q_pole_, -1.0, 1.0));
    const torch::Tensor sin_b = torch::sin(beta);
    const torch::Tensor cos_b = torch::cos(beta);
    const double sin_b0 = std::sin(beta0_);
    const double cos_b0 = std::cos(beta0_);
    const torch::Tensor common = 1.0 + sin_b0 * sin_b + cos_b0 * cos_b * torch::cos(delta_lambda);
    const torch::Tensor b = radius_authalic_ * torch::sqrt(2.0 / common);
    torch::Tensor easting = b * d_ * cos_b * torch::sin(delta_lambda);
    torch::Tensor northing = (b / d_) * (cos_b0 * sin_b - sin_b0 * cos_b * torch::cos(delta_lambda));
    return {std::move(easting), std::move(northing)};
}

std::pair<torch::Tensor, torch::Tensor> LambertAzimuthalEqualArea::inverse(
    const torch::Tensor& easting, const torch::Tensor& northing) const {
    const torch::Tensor x = as_double(easting);
    const torch::Tensor y = as_double(northing);
    const torch::Tensor rho = torch::hypot(x / d_, y * d_);
    const torch::Tensor at_origin = rho < 1e-9;
    const torch::Tensor safe_rho = torch::where(at_origin, torch::ones_like(rho), rho);
    const torch::Tensor c =
        2.0 * torch::asin(torch::clamp(safe_rho / (2.0 * radius_authalic_), -1.0, 1.0));
    const torch::Tensor sin_c = torch::sin(c);
    const torch::Tensor cos_c = torch::cos(c);
    const double sin_b0 = std::sin(beta0_);
    const double cos_b0 = std::cos(beta0_);
    const torch::Tensor sin_beta = cos_c * sin_b0 + (d_ * y * sin_c * cos_b0) / safe_rho;
    const torch::Tensor beta = torch::asin(torch::clamp(sin_beta, -1.0, 1.0));
    torch::Tensor latitude = geodetic_latitude(beta) / kRadians;
    torch::Tensor longitude =
        lon0_ + torch::atan2(x * sin_c, d_ * safe_rho * cos_b0 * cos_c - d_ * d_ * y * sin_b0 * sin_c) /
                    kRadians;
    latitude = torch::where(at_origin, torch::full_like(latitude, lat0_), latitude);
    longitude = torch::where(at_origin, torch::full_like(longitude, lon0_), longitude);
    return {std::move(latitude), std::move(longitude)};
}

}  // namespace resolve
