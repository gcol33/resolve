#pragma once

// An elevation model compressed with a bounded error: a coded terrain field.
//
// Each tile's cells are quantised to multiples of 2 * max_error (so no cell
// moves by more than max_error), cut into block x block blocks, and each block
// coded on its own: the Lorenzo predictor (west + north - north-west, the
// predictor of the SZ error-bounded compressor, Di & Cappello 2016) leaves
// small residuals, which are mapped to non-negative integers (zigzag) and
// written with a Golomb-Rice code of the block's best parameter k, or, where
// most residuals are zero, as exponential-Golomb runs of zeros between Rice-
// coded non-zero values (the run mode of JPEG-LS). A block is decoded from its
// own bytes alone.
//
// On the GLO-30 tiles measured for the design (Alps, Bohemia, Pannonia, the
// Netherlands) this takes 0.76-1.97 bits per cell at max_error = 2 m, at or
// below zstd over the same residuals, against 1.3-5.2 bits per cell and
// 1-10 m RMSE for the neural terrain field (terrain_field.hpp)
// (dev_notes/terrain_correction_design.md).
//
// The file is a tensor pack of format "resolve-terrain-coded": per tile its
// geometry, quantum, block size and the RMSE, maximum and slope error it was
// encoded with; per tile two tensors, the byte stream and the byte offset of
// each block, every tile's offsets written before the streams. Tiles are
// coded on all threads. TerrainCodedField reads and decodes only the blocks a
// query reaches.

#include "resolve/json.hpp"
#include "resolve/lru_cache.hpp"
#include "resolve/tensor_pack.hpp"
#include "resolve/terrain_field.hpp"
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

inline constexpr const char* kTerrainCodedFormat = "resolve-terrain-coded";

struct TerrainCodecConfig {
    float max_error = 2.0f;  // m: no cell moves by more than this
    int64_t block = 64;      // cells per block side

    void validate() const;
};

// One tile's code: the quantised grid's blocks as bytes.
struct TerrainTileCode {
    torch::Tensor stream;   // (bytes,) uint8
    torch::Tensor offsets;  // (n_blocks + 1,) int64, byte offset of each block and the end
};

// Code a (rows, cols) float32 grid; `quantum` = 2 * max_error.
[[nodiscard]] TerrainTileCode encode_terrain_tile(const torch::Tensor& grid, double quantum, int64_t block);
// The grid back, (rows, cols) float32, each cell within quantum / 2 of the
// original.
[[nodiscard]] torch::Tensor decode_terrain_tile(const TerrainTileCode& code, int64_t rows, int64_t cols,
                                                double quantum, int64_t block);

// Code every tile of `grids` (or those named in `tiles`) into one coded field
// file at `path`. Returns, per tile coded, its metrics (RMSE, mean, 99th
// percentile and largest error, slope error, seconds, bytes).
std::vector<std::pair<std::string, TerrainTileMetrics>> encode_terrain_grids(const TerrainGrids& grids,
                                                                             const TerrainCodecConfig& config,
                                                                             const std::string& path,
                                                                             const std::vector<std::string>& tiles = {});

// Decode block `b` (row-major over the tile's blocks) of a tile's code into
// `out`, whose rows are `stride` floats apart; `scratch` is reused between
// calls. The code must be on the CPU and contiguous
// (terrain_tile_code_on_cpu).
void decode_terrain_block(const TerrainTileCode& code, int64_t rows, int64_t cols, double quantum, int64_t block,
                          int64_t b, float* out, int64_t stride, std::vector<int64_t>& scratch);
[[nodiscard]] TerrainTileCode terrain_tile_code_on_cpu(const TerrainTileCode& code);

// A coded field file as a TerrainSource. A query reads and decodes only the
// blocks its cells fall in: their bytes are read in one request per run of
// nearby blocks (each tile's block offsets are read once and kept, eight
// bytes per block), the blocks are decoded on all threads, and at most
// `cache_blocks` decoded blocks are kept for later queries, the least
// recently used dropped first. Thread-safe.
class TerrainCodedField final : public TerrainSource {
public:
    explicit TerrainCodedField(const std::string& path, int64_t cache_blocks = 4096);

    [[nodiscard]] const TerrainTileIndex& index() const override { return index_; }
    [[nodiscard]] torch::Tensor elevation(const torch::Tensor& lat, const torch::Tensor& lon) const override;
    [[nodiscard]] std::string describe() const override;

    [[nodiscard]] torch::Tensor cell_values(const TerrainCells& cells) const;
    // Decoded on each call, not held.
    [[nodiscard]] torch::Tensor grid(int64_t t) const override;
    // Block `b` (row-major over the tile's blocks) of tile `t` decoded,
    // (h, w) float32.
    [[nodiscard]] torch::Tensor block(int64_t t, int64_t b) const;
    [[nodiscard]] const TerrainCodecConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<TerrainTileMetrics>& metrics() const noexcept { return metrics_; }

private:
    const TerrainTile& tile_at(int64_t t, const char* who) const;
    [[nodiscard]] double quantum() const noexcept;
    // Each tile's block offsets, (n_blocks + 1,) int64; those not yet held
    // are read together (every tile's offsets lie in one region of the file).
    [[nodiscard]] std::vector<torch::Tensor> offsets(const std::vector<int64_t>& tiles) const;
    // The decoded blocks under `keys` (each block_base_[t] + b, sorted and
    // distinct), from the cache or read and decoded together.
    [[nodiscard]] std::vector<torch::Tensor> blocks(const std::vector<int64_t>& keys) const;

    TensorPackReader pack_;
    TerrainCodecConfig config_;
    TerrainTileIndex index_;
    std::vector<TerrainTileMetrics> metrics_;
    // Each tile's first block's key: the blocks of the tiles before it.
    std::vector<int64_t> block_base_;
    mutable std::mutex offsets_mutex_;
    mutable std::vector<torch::Tensor> offsets_;
    LruCache<torch::Tensor> blocks_;
};

// Open a terrain field file of either format (a neural field,
// terrain_field.hpp, or a coded field), read from its header. `device` is
// where a neural field runs.
[[nodiscard]] std::shared_ptr<TerrainSource> open_terrain_field(const std::string& path,
                                                                torch::Device device = torch::kCPU);

}  // namespace resolve
