#pragma once

// Elevation at plot coordinates, and the terrain descriptors read from it.
//
// A digital elevation model is a set of tiles, each a grid of elevations whose
// cell centres sit at known geographic positions. Two sources answer
// "the elevation at (lat, lon)":
//
//   TerrainGrids   the grids themselves, float32 `.npy` files listed in a
//                  `tiles.json` (written by resolve-2026/scripts/terrain_tiles.py
//                  from Copernicus GLO-30), loaded tile by tile as queries reach
//                  them;
//   TerrainField   the same tiles compressed into neural fields
//                  (terrain_field.hpp), shipped with a suite.
//
// Both are TerrainSource, and terrain_descriptors() reads every descriptor
// through that one interface, so a descriptor computed from the shipped field
// and the same descriptor computed from the grids the correction was trained on
// follow one definition:
//
//   elevation       the elevation at the coordinate, bilinear between the four
//                   surrounding cell centres;
//   elevation_1km   the mean elevation over a 1 km x 1 km square centred on the
//                   coordinate, read on a kTerrainLattice x kTerrainLattice
//                   lattice of points;
//   slope           the slope of the cell nearest the coordinate from its four
//                   edge neighbours (central differences in metres, as
//                   Earth Engine's ee.Terrain.slope), in degrees;
//   slope_1km       the inclination of the least-squares plane through the
//                   1 km lattice, in degrees.
//
// Angles are in degrees (WGS84), elevations and distances in metres. Every
// tensor here is float64 on the CPU; NaN marks a value the source does not
// hold (a coordinate outside every tile, or a NaN coordinate).

#include "resolve/json.hpp"
#include "resolve/lru_cache.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

// The side of the square elevation_1km and slope_1km read, in metres, and the
// points per side of the lattice they read it on (odd, so the coordinate
// itself is a lattice point; 33 spaces the points 31.25 m apart, about one
// GLO-30 cell).
inline constexpr double kTerrainWindowM = 1000.0;
inline constexpr int64_t kTerrainLattice = 33;

// The Earth radius distances on the ground are converted to angles with, in m.
inline constexpr double kTerrainEarthRadiusM = 6371000.0;

// Points per task when the terrain sources spread a per-point loop over
// threads (at::parallel_for).
inline constexpr int64_t kTerrainParallelGrain = int64_t{1} << 14;

// One tile: a grid of `rows` x `cols` elevations, row-major from north to
// south, cell (r, c) centred at (lat_first - r * dlat, lon_first + c * dlon).
struct TerrainTile {
    std::string name;
    int64_t rows = 0;
    int64_t cols = 0;
    double lat_first = 0.0;
    double lon_first = 0.0;
    double dlat = 0.0;
    double dlon = 0.0;

    // The integer degree square the tile covers, keyed by its south-west
    // corner: (floor of the southern edge, floor of the western edge).
    [[nodiscard]] std::pair<int64_t, int64_t> key() const;

    bool operator==(const TerrainTile&) const = default;
};

// A tile's geometry as JSON (name, rows, cols, lat_first, lon_first, dlat,
// dlon) and back: how tiles.json and both terrain field formats record it.
[[nodiscard]] json::Value terrain_tile_to_json(const TerrainTile& tile);
[[nodiscard]] TerrainTile terrain_tile_from_json(const json::Value& value);

// Where a set of points falls: the nearest cell centre of the tile holding
// each point, -1 for a point outside every tile.
struct TerrainCells {
    torch::Tensor tile;  // (n,) int64 index into TerrainTileIndex::tiles(), -1 outside
    torch::Tensor row;   // (n,) int64
    torch::Tensor col;   // (n,) int64
};

// The tiles of a source, found by the degree square a point falls in. Tiles
// abut: the row after a tile's last row is the first row of the tile to its
// south, the column after its last column the first column of the tile to its
// east (GLO-30 lays its tiles out so, pixel-is-point).
class TerrainTileIndex {
public:
    TerrainTileIndex() = default;
    // Throws std::invalid_argument for a tile without cells or with a
    // non-positive spacing, or two tiles on one degree square.
    explicit TerrainTileIndex(std::vector<TerrainTile> tiles);

    [[nodiscard]] const std::vector<TerrainTile>& tiles() const noexcept { return tiles_; }
    [[nodiscard]] int64_t size() const noexcept { return static_cast<int64_t>(tiles_.size()); }
    // The tile on the degree square keyed (south, west), or -1.
    [[nodiscard]] int64_t find(int64_t south, int64_t west) const;

    // The tile holding each point (its degree square), -1 outside every tile.
    [[nodiscard]] torch::Tensor tile_of(const torch::Tensor& lat, const torch::Tensor& lon) const;

    // The cell nearest each point, moved into the neighbouring tile where the
    // nearest centre is that tile's first row or column.
    [[nodiscard]] TerrainCells nearest_cells(const torch::Tensor& lat, const torch::Tensor& lon) const;

    // Cells given as (tile, row, col) with row / col up to one tile beyond
    // the tile's edge on any side, moved into the tile that holds them; a
    // cell whose neighbour tile is absent gets tile -1.
    [[nodiscard]] TerrainCells normalise(const torch::Tensor& tile, const torch::Tensor& row,
                                         const torch::Tensor& col) const;

    // The centre of each cell, (lat, lon); NaN for tile -1.
    [[nodiscard]] std::pair<torch::Tensor, torch::Tensor> centres(const TerrainCells& cells) const;

private:
    std::vector<TerrainTile> tiles_;
    std::unordered_map<int64_t, int64_t> by_key_;
};

// A source of elevations.
class TerrainSource {
public:
    virtual ~TerrainSource() = default;

    [[nodiscard]] virtual const TerrainTileIndex& index() const = 0;

    // The elevation (m) at each point: bilinear between the four cell centres
    // around it, NaN where none of them is held. (n,) float64.
    [[nodiscard]] virtual torch::Tensor elevation(const torch::Tensor& lat,
                                                  const torch::Tensor& lon) const = 0;

    // Tile `t` (its position in index().tiles()) whole, (rows, cols) float32:
    // the stored cells of grids, the decoded or evaluated cells of a field.
    [[nodiscard]] virtual torch::Tensor grid(int64_t t) const = 0;

    // One line naming the source, for reports.
    [[nodiscard]] virtual std::string describe() const = 0;
};

struct TerrainDescriptors {
    torch::Tensor elevation;      // (n,) m
    torch::Tensor elevation_1km;  // (n,) m
    torch::Tensor slope;          // (n,) degrees
    torch::Tensor slope_1km;      // (n,) degrees
};

// The four descriptors at each coordinate, read in chunks of `chunk` points.
// A 1 km descriptor needs at least half of its lattice held, else NaN; the
// slope needs all four neighbours of its cell.
[[nodiscard]] TerrainDescriptors terrain_descriptors(const TerrainSource& source,
                                                     const torch::Tensor& lat,
                                                     const torch::Tensor& lon,
                                                     int64_t chunk = 4096);

// ---------------------------------------------------------------------------
// Grids on disk
// ---------------------------------------------------------------------------

inline constexpr const char* kTerrainGridsIndexFile = "tiles.json";
inline constexpr const char* kTerrainGridsFormat = "resolve-terrain-grids";

// A float32 C-order 2-D `.npy` array as a (rows, cols) float32 tensor.
// Throws io::IOError when the file cannot be read and std::runtime_error for
// any other layout.
[[nodiscard]] torch::Tensor read_npy_grid(const std::string& path);
// Write a (rows, cols) tensor as a float32 C-order `.npy` (version 1.0).
void write_npy_grid(const std::string& path, const torch::Tensor& grid);

// The tiles listed in `<dir>/tiles.json`, each read from its `.npy` when a
// query first reaches it; at most `cache_tiles` grids are held at once (the
// least recently used is dropped first). Thread-safe.
class TerrainGrids final : public TerrainSource {
public:
    explicit TerrainGrids(std::string dir, int64_t cache_tiles = 16);

    [[nodiscard]] const TerrainTileIndex& index() const override { return index_; }
    [[nodiscard]] torch::Tensor elevation(const torch::Tensor& lat,
                                          const torch::Tensor& lon) const override;
    [[nodiscard]] std::string describe() const override;

    // The elevation of each cell, NaN for tile -1.
    [[nodiscard]] torch::Tensor cell_values(const TerrainCells& cells) const;
    [[nodiscard]] torch::Tensor grid(int64_t t) const override;
    [[nodiscard]] const std::string& dir() const noexcept { return dir_; }

private:
    std::string dir_;
    std::vector<std::string> files_;
    TerrainTileIndex index_;
    LruCache<torch::Tensor> cache_;
};

// Bilinear interpolation at (lat, lon) from a function giving the elevation
// of cells: the shared reading of TerrainGrids and TerrainField, so both
// interpolate the same corners with the same weights. Corners not held are
// left out and the remaining weights renormalised.
using TerrainCellValues = std::function<torch::Tensor(const TerrainCells&)>;
[[nodiscard]] torch::Tensor bilinear_elevation(const TerrainTileIndex& index,
                                               const torch::Tensor& lat,
                                               const torch::Tensor& lon,
                                               const TerrainCellValues& cell_values);

}  // namespace resolve
