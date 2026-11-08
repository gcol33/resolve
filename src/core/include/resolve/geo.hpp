#pragma once

// Positions on the Earth: great-circle distance, the geographic median of a
// set of points, and the Lambert azimuthal equal-area projection of the GRS80
// ellipsoid (EPSG 9820), in which a set of positions can be fitted as plane
// coordinates without distorting area.
//
// Angles are in degrees, projected coordinates in metres, distances in km.
// Every tensor function computes in float64 on the CPU and returns float64.

#include <torch/torch.h>

#include <utility>

namespace resolve {

// The Earth radius great-circle distances are measured with, in km.
inline constexpr double kEarthRadiusKm = 6371.0;

// GRS80, the ellipsoid ETRS89 is realised on.
inline constexpr double kGrs80SemiMajorM = 6378137.0;
inline constexpr double kGrs80InverseFlattening = 298.257222101;

// The haversine distance in km between two sets of positions in degrees,
// broadcast against each other (a (n, 1) column against a (1, m) row gives
// the (n, m) distances). NaN wherever a coordinate is NaN.
[[nodiscard]] torch::Tensor great_circle_km(const torch::Tensor& lat1,
                                            const torch::Tensor& lon1,
                                            const torch::Tensor& lat2,
                                            const torch::Tensor& lon2);

// The haversine distance in km between two positions in degrees, by the same
// formula; NaN when a coordinate is NaN.
[[nodiscard]] double great_circle_km(double lat1, double lon1, double lat2, double lon2);

// The point minimising the mean great-circle distance to (lat, lon), by
// Weiszfeld's iteration on unit vectors started from their normalised mean.
// Positions with a NaN coordinate are left out. Returns (latitude, longitude)
// in degrees; throws std::invalid_argument when no position is finite.
[[nodiscard]] std::pair<double, double> geographic_median(const torch::Tensor& lat,
                                                          const torch::Tensor& lon,
                                                          int iterations = 200);

// The oblique Lambert azimuthal equal-area projection of an ellipsoid about an
// origin (Snyder 1987, "Map Projections: A Working Manual", pp. 187-190).
class LambertAzimuthalEqualArea {
public:
    LambertAzimuthalEqualArea(double latitude_of_origin,
                              double longitude_of_origin,
                              double semi_major_m = kGrs80SemiMajorM,
                              double inverse_flattening = kGrs80InverseFlattening);

    // Degrees to (easting, northing) in metres from the origin.
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& latitude,
                                                                  const torch::Tensor& longitude) const;
    // (easting, northing) in metres back to (latitude, longitude) in degrees.
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> inverse(const torch::Tensor& easting,
                                                                  const torch::Tensor& northing) const;

    [[nodiscard]] double latitude_of_origin() const noexcept { return lat0_; }
    [[nodiscard]] double longitude_of_origin() const noexcept { return lon0_; }

private:
    // Twice the area of the zone from the equator to sin_phi, in units of a^2.
    [[nodiscard]] torch::Tensor q(const torch::Tensor& sin_phi) const;
    [[nodiscard]] double q(double sin_phi) const;
    // The geodetic latitude whose authalic latitude is beta, by Newton.
    [[nodiscard]] torch::Tensor geodetic_latitude(const torch::Tensor& beta) const;

    double lat0_;
    double lon0_;
    double a_;
    double e2_;
    double e_;
    double q_pole_;
    double radius_authalic_;
    double beta0_;
    double d_;
};

}  // namespace resolve
