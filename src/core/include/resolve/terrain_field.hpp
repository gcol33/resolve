#pragma once

// An elevation model compressed into neural fields, one per tile.
//
// Each tile of a TerrainSource is stored as
//
//   * a coarse base: the tile averaged onto base_cells x base_cells blocks, in
//     decimetres (int16), read bilinearly;
//   * the detail = elevation - base, modelled by a multiresolution hash
//     encoding of the tile's local position (Mueller, Evans, Schied & Keller
//     2022, "Instant neural graphics primitives with a multiresolution hash
//     encoding", ACM Transactions on Graphics 41(4), 102) followed by a small
//     MLP, its parameters stored as float16.
//
// Tiles are fitted independently (fit_terrain_tile), each to its own tensor
// pack, and assembled into one field file (assemble_terrain_field), so a
// continent is fitted in parallel and a query loads only the tiles it reaches.
// The fit minimises the mean absolute error of the elevation plus
// gradient_weight times that of the four-neighbour differences, so the slope
// survives compression (gradient supervision as in ImplicitTerrainV2,
// arXiv 2605.22556).
//
// TerrainField answers the elevation at a cell centre from its tile's field,
// and between cell centres bilinearly from those values, exactly as
// TerrainGrids does from the stored cells (terrain_source.hpp): a descriptor
// read from the field and the same descriptor read from the grids differ only
// by the field's reconstruction error, which each tile records (RMSE, mean
// and 99th-percentile absolute error, maximum, slope error) when it is fitted.

#include "resolve/json.hpp"
#include "resolve/lru_cache.hpp"
#include "resolve/tensor_pack.hpp"
#include "resolve/terrain_source.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace resolve {

inline constexpr const char* kTerrainFieldFormat = "resolve-terrain-field";
inline constexpr const char* kTerrainTileFormat = "resolve-terrain-tile";

struct TerrainFieldConfig {
    int64_t levels = 16;              // hash-encoding levels
    int64_t log2_table_size = 17;     // entries per level: 2^log2_table_size
    int64_t features = 2;             // features per entry
    int64_t coarsest_resolution = 16; // grid cells per side at the coarsest level
    float finest_fraction = 1.0f;     // finest level's cells per side / max(rows, cols)
    int64_t hidden_dim = 64;
    int64_t n_layers = 2;
    int64_t base_cells = 64;          // coarse base blocks per side
    int64_t steps = 6000;
    int64_t batch_size = 1 << 18;
    float lr = 1e-2f;
    float gradient_weight = 0.5f;
    int seed = 20261108;
    torch::Device device = torch::kCPU;

    void validate() const;
};

// The multiresolution hash encoding of points in [0, 1]^2: per level l a grid
// of N_l cells per side, N_l = floor(N_min * b^l) with b fixed so the finest
// level has N_max cells; a level whose (N_l + 1)^2 vertices fit the table is
// indexed densely, a finer one by the spatial hash x * 1 XOR y * 2654435761
// modulo the table size; the F features of the four corner vertices are read
// bilinearly. Output (n, levels * features).
class HashGridEncodingImpl : public torch::nn::Module {
public:
    HashGridEncodingImpl(int64_t levels, int64_t log2_table_size, int64_t features, int64_t coarsest,
                         int64_t finest);
    torch::Tensor forward(const torch::Tensor& xy);

    [[nodiscard]] const std::vector<int64_t>& resolutions() const noexcept { return resolutions_; }
    torch::Tensor table;  // (levels * 2^log2_table_size, features)

private:
    int64_t levels_;
    int64_t table_size_;
    int64_t features_;
    std::vector<int64_t> resolutions_;
};
TORCH_MODULE(HashGridEncoding);

// One tile's field: the base and the detail network.
class TerrainTileFieldImpl : public torch::nn::Module {
public:
    TerrainTileFieldImpl(const TerrainFieldConfig& config, int64_t rows, int64_t cols);

    // The elevation (m) at cells (row, col) of the tile; (n,) float32 on the
    // field's device. Rows and columns are clamped to the tile.
    torch::Tensor cells(const torch::Tensor& row, const torch::Tensor& col);
    // The detail network's output (scaled) at local positions in [0, 1]^2.
    torch::Tensor detail(const torch::Tensor& uv);
    // The base (m) at cells, bilinear between block centres.
    torch::Tensor base_at(const torch::Tensor& row, const torch::Tensor& col) const;

    int64_t rows;
    int64_t cols;
    double scale = 1.0;      // detail units: metres per network unit
    torch::Tensor base_dm;   // (base_cells, base_cells) int16, decimetres
    HashGridEncoding encoding{nullptr};
    torch::nn::Sequential mlp{nullptr};
};
TORCH_MODULE(TerrainTileField);

// How closely a fitted tile reproduces its grid, over every cell.
struct TerrainTileMetrics {
    double rmse = 0.0;
    double mae = 0.0;
    double p99 = 0.0;
    double max_abs = 0.0;
    double slope_mae = 0.0;  // degrees, the four-neighbour slope of the cells
    double seconds = 0.0;
    int64_t bytes = 0;       // stored size of the tile
};

[[nodiscard]] json::Value terrain_tile_metrics_to_json(const TerrainTileMetrics& m);
[[nodiscard]] TerrainTileMetrics terrain_tile_metrics_from_json(const json::Value& v);

// How closely `reconstructed` reproduces `grid` (both (rows, cols), laid out
// as `tile`): RMSE, mean, 99th-percentile and largest absolute error, and the
// mean absolute difference of the four-neighbour slopes. The quality figures
// both terrain field formats record per tile.
[[nodiscard]] TerrainTileMetrics terrain_reconstruction_metrics(const torch::Tensor& reconstructed,
                                                                const torch::Tensor& grid, const TerrainTile& tile);

// The four-neighbour slope (degrees) of every interior cell of a grid laid
// out as `tile`, (rows - 2, cols - 2): the stencil terrain_descriptors reads.
[[nodiscard]] torch::Tensor terrain_cell_slopes(const torch::Tensor& grid, const TerrainTile& tile);

// Fit tile `tile` of `source`'s grid `grid` ((rows, cols) float32) and write
// it to `path` as a tile pack. Returns its metrics.
TerrainTileMetrics fit_terrain_tile(const torch::Tensor& grid, const TerrainTile& tile,
                                    const TerrainFieldConfig& config, const std::string& path);

// Merge tile packs into one field file at `path`. Throws when two packs cover
// one tile or were fitted under different configs.
void assemble_terrain_field(const std::vector<std::string>& tile_paths, const std::string& path);

// A field file as a TerrainSource. At most `cache_tiles` tile fields are held
// at once. Thread-safe.
class TerrainField final : public TerrainSource {
public:
    explicit TerrainField(const std::string& path, torch::Device device = torch::kCPU,
                          int64_t cache_tiles = 64);

    [[nodiscard]] const TerrainTileIndex& index() const override { return index_; }
    [[nodiscard]] torch::Tensor elevation(const torch::Tensor& lat, const torch::Tensor& lon) const override;
    [[nodiscard]] std::string describe() const override;
    // Every cell of the tile evaluated from its field.
    [[nodiscard]] torch::Tensor grid(int64_t t) const override;

    // The elevation of each cell from its tile's field, NaN for tile -1.
    [[nodiscard]] torch::Tensor cell_values(const TerrainCells& cells) const;
    [[nodiscard]] const TerrainFieldConfig& config() const noexcept { return config_; }
    // Per tile (in index order), the metrics recorded when it was fitted.
    [[nodiscard]] const std::vector<TerrainTileMetrics>& metrics() const noexcept { return metrics_; }

private:
    TerrainTileField tile_field(int64_t t) const;

    TensorPackReader pack_;
    torch::Device device_;
    TerrainFieldConfig config_;
    TerrainTileIndex index_;
    std::vector<TerrainTileMetrics> metrics_;
    std::vector<double> scales_;
    LruCache<TerrainTileField> cache_;
};

}  // namespace resolve
