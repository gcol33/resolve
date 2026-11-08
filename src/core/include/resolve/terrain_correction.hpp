#pragma once

// Terrain correction: a prediction of altitude or slope combined with an
// elevation model read at the plot's coordinates.
//
// An elevation model read at a precise coordinate is close to a plot's
// recorded altitude, and far off where the coordinate is wrong or coarse; the
// composition estimate errs the other way round. The correction learns, per
// plot, how much to trust each source. A small MLP g reads the plot's features
// and returns
//
//   corrected = sum_k softmax(g(x))_k * expert_k + offset_limit * tanh(g(x)_offset)
//
// over three experts: the raw prediction, the elevation-model descriptor at
// the coordinate, and the same descriptor over 1 km (terrain_source.hpp). A
// softmax weight of one returns a source exactly, which a residual model
// cannot: on the paper's validation cells this gated form beat a per-class
// linear blend, an outlier-mixture fusion (Hogg, Bovy & Lang 2010) and
// gradient-boosted trees on the residual (dev_notes/terrain_correction_design.md).
//
// Features, every value scaled by the target's `scale`: the prediction, both
// descriptors and their differences from the prediction; log10(location
// uncertainty + 1) with an indicator of a recorded uncertainty; log1p of the
// members' spread (x 100 / scale); the share of the assemblage the vocabulary
// recognises; log1p of the number of species; an indicator of a held 1 km
// descriptor. A missing 1 km descriptor reads the point descriptor in its
// place, and training drops it at random (`dem_1km_dropout`), so a caller who
// supplies the point value alone is served. A plot without the point
// descriptor or without a prediction keeps the raw prediction.
//
// Fitted with the mean absolute error, the headline error of both targets.

#include "resolve/json.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace resolve {

inline constexpr int kTerrainSeed = 20261108;

struct TerrainCorrectionConfig {
    // The target's scale: values are divided by it before the network reads
    // them, and the free offset is bounded by offset_fraction * scale.
    float scale = 500.0f;
    float offset_fraction = 0.02f;
    int64_t hidden_dim = 128;
    int64_t n_layers = 2;
    int64_t epochs = 12;
    int64_t batch_size = 2048;
    float lr = 3e-3f;
    float weight_decay = 1e-4f;
    // Share of the training plots whose 1 km descriptor is hidden each epoch.
    float dem_1km_dropout = 0.2f;
    int seed = kTerrainSeed;
    torch::Device device = torch::kCPU;

    // Throws std::invalid_argument naming the first problem.
    void validate() const;
};

// One value per plot, each (n,) float64-convertible; NaN where unknown.
struct TerrainCorrectionInputs {
    torch::Tensor prediction;        // the raw prediction, in the target's units
    torch::Tensor dem;               // the descriptor at the coordinate
    torch::Tensor dem_1km;           // the descriptor over 1 km (undefined: none)
    torch::Tensor uncertainty_m;     // recorded location uncertainty (undefined: none)
    torch::Tensor spread;            // the suite members' standard deviation (undefined: none)
    torch::Tensor recognised_share;  // share of distinct species recognised (undefined: none)
    torch::Tensor n_species;         // species in the plot (undefined: none)

    // The number of plots; throws std::invalid_argument when the defined
    // tensors differ in length or `prediction` / `dem` is undefined.
    [[nodiscard]] int64_t size() const;
};

struct TerrainCorrectionResult {
    torch::Tensor corrected;  // (n,) float64, the raw prediction where not applied
    torch::Tensor weights;    // (n, 3) float64: raw prediction, descriptor, 1 km descriptor; NaN where not applied
    torch::Tensor offset;     // (n,) float64 free offset in the target's units; NaN where not applied
    torch::Tensor applied;    // (n,) bool
};

// The expert weights' column names, in order.
[[nodiscard]] const std::vector<std::string>& terrain_weight_names();

// What a fit records about itself.
struct TerrainCorrectionReport {
    std::string target;
    TerrainDescriptor descriptor = TerrainDescriptor::Elevation;
    int64_t n_train = 0;
    // Mean absolute error on the training plots of the raw prediction, the
    // point descriptor and the corrected value.
    double mae_raw = 0.0;
    double mae_dem = 0.0;
    double mae_corrected = 0.0;
    std::vector<double> loss_history;  // mean training loss per epoch (scaled units)
};

[[nodiscard]] json::Value terrain_correction_report_to_json(const TerrainCorrectionReport& report);

class TerrainCorrection {
public:
    TerrainCorrection(std::string target, TerrainDescriptor descriptor, TerrainCorrectionConfig config);

    [[nodiscard]] TerrainCorrectionResult apply(const TerrainCorrectionInputs& inputs) const;

    // Save to / load from a tensor pack (tensor_pack.hpp): the config, the
    // target, the descriptor, the report and the network's parameters.
    void save(const std::string& path) const;
    [[nodiscard]] static TerrainCorrection load(const std::string& path,
                                                torch::Device device = torch::kCPU);

    [[nodiscard]] const std::string& target() const noexcept { return target_; }
    [[nodiscard]] TerrainDescriptor descriptor() const noexcept { return descriptor_; }
    [[nodiscard]] const TerrainCorrectionConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TerrainCorrectionReport& report() const noexcept { return report_; }

    // The number of input features the network reads.
    static constexpr int64_t kFeatures = 11;

private:
    friend TerrainCorrection fit_terrain_correction(const std::string&, TerrainDescriptor,
                                                    const TerrainCorrectionInputs&, const torch::Tensor&,
                                                    const TerrainCorrectionConfig&);
    std::string target_;
    TerrainDescriptor descriptor_;
    TerrainCorrectionConfig config_;
    TerrainCorrectionReport report_;
    // Mutable: a forward pass on a const correction runs the module.
    mutable torch::nn::Sequential net_{nullptr};
    // What the network reads for an input a caller does not give (the
    // training plots' medians).
    double fill_spread_ = 0.0;
    double fill_recognised_share_ = 1.0;
    double fill_n_species_ = 10.0;
};

// Fit the correction of `target` on plots whose recorded value is `truth`
// ((n,), NaN where unrecorded; those plots and those without a prediction or
// a point descriptor are left out). Throws std::invalid_argument for an
// invalid config, lengths that differ or fewer than two usable plots.
[[nodiscard]] TerrainCorrection fit_terrain_correction(const std::string& target,
                                                       TerrainDescriptor descriptor,
                                                       const TerrainCorrectionInputs& inputs,
                                                       const torch::Tensor& truth,
                                                       const TerrainCorrectionConfig& config = {});

}  // namespace resolve
