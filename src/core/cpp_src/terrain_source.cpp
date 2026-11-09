#include "resolve/terrain_source.hpp"

#include "resolve/io_retry.hpp"
#include "resolve/json.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace resolve {

namespace {

constexpr double kRadians = std::numbers::pi / 180.0;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

int64_t pack_key(int64_t south, int64_t west) {
    return (south + 1000) * 4000 + (west + 2000);
}

torch::Tensor as_double(const torch::Tensor& t) {
    return t.to(torch::kCPU, torch::kFloat64).reshape({-1}).contiguous();
}

torch::Tensor as_long(const torch::Tensor& t) {
    return t.to(torch::kCPU, torch::kInt64).reshape({-1}).contiguous();
}

}  // namespace

// ---------------------------------------------------------------------------
// TerrainTile / TerrainTileIndex
// ---------------------------------------------------------------------------

std::pair<int64_t, int64_t> TerrainTile::key() const {
    // A quarter cell inward from the south-west cell centre lies inside the
    // tile's degree square for a pixel-is-point grid (centres on the edges)
    // and for a pixel-is-area one (centres half a cell in).
    const double south = lat_first - static_cast<double>(rows) * dlat + 0.25 * dlat;
    const double west = lon_first + 0.25 * dlon;
    return {static_cast<int64_t>(std::floor(south)), static_cast<int64_t>(std::floor(west))};
}

json::Value terrain_tile_to_json(const TerrainTile& t) {
    json::Value v = json::Value::object();
    v.set("name", t.name);
    v.set("rows", t.rows);
    v.set("cols", t.cols);
    v.set("lat_first", t.lat_first);
    v.set("lon_first", t.lon_first);
    v.set("dlat", t.dlat);
    v.set("dlon", t.dlon);
    return v;
}

TerrainTile terrain_tile_from_json(const json::Value& v) {
    TerrainTile t;
    t.name = v.at("name").as_string();
    t.rows = v.at("rows").as_int();
    t.cols = v.at("cols").as_int();
    t.lat_first = v.at("lat_first").as_number();
    t.lon_first = v.at("lon_first").as_number();
    t.dlat = v.at("dlat").as_number();
    t.dlon = v.at("dlon").as_number();
    return t;
}

TerrainTileIndex::TerrainTileIndex(std::vector<TerrainTile> tiles) : tiles_(std::move(tiles)) {
    for (int64_t i = 0; i < size(); ++i) {
        const auto& t = tiles_[static_cast<size_t>(i)];
        if (t.rows < 2 || t.cols < 2) {
            throw std::invalid_argument("TerrainTileIndex: tile '" + t.name +
                                        "' has fewer than 2 x 2 cells");
        }
        if (!(t.dlat > 0.0) || !(t.dlon > 0.0)) {
            throw std::invalid_argument("TerrainTileIndex: tile '" + t.name +
                                        "' has a non-positive cell spacing");
        }
        const auto [s, w] = t.key();
        if (!by_key_.emplace(pack_key(s, w), i).second) {
            throw std::invalid_argument("TerrainTileIndex: tiles '" + t.name + "' and '" +
                                        tiles_[static_cast<size_t>(by_key_.at(pack_key(s, w)))].name +
                                        "' cover the same degree square");
        }
    }
}

int64_t TerrainTileIndex::find(int64_t south, int64_t west) const {
    const auto it = by_key_.find(pack_key(south, west));
    return it == by_key_.end() ? -1 : it->second;
}

torch::Tensor TerrainTileIndex::tile_of(const torch::Tensor& lat, const torch::Tensor& lon) const {
    const torch::Tensor la = as_double(lat);
    const torch::Tensor lo = as_double(lon);
    if (la.numel() != lo.numel()) {
        throw std::invalid_argument("TerrainTileIndex::tile_of: lat and lon differ in length");
    }
    torch::Tensor out = torch::full({la.numel()}, int64_t{-1}, torch::kInt64);
    const auto a = la.accessor<double, 1>();
    const auto o = lo.accessor<double, 1>();
    auto t = out.accessor<int64_t, 1>();
    at::parallel_for(0, la.numel(), kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (std::isfinite(a[i]) && std::isfinite(o[i])) {
                t[i] = find(static_cast<int64_t>(std::floor(a[i])), static_cast<int64_t>(std::floor(o[i])));
            }
        }
    });
    return out;
}

TerrainCells TerrainTileIndex::nearest_cells(const torch::Tensor& lat, const torch::Tensor& lon) const {
    const torch::Tensor la = as_double(lat);
    const torch::Tensor lo = as_double(lon);
    const torch::Tensor tile = tile_of(la, lo);
    const int64_t n = la.numel();
    torch::Tensor row = torch::zeros({n}, torch::kInt64);
    torch::Tensor col = torch::zeros({n}, torch::kInt64);
    const auto a = la.accessor<double, 1>();
    const auto o = lo.accessor<double, 1>();
    const auto t = tile.accessor<int64_t, 1>();
    auto r = row.accessor<int64_t, 1>();
    auto c = col.accessor<int64_t, 1>();
    at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (t[i] < 0) continue;
            const auto& tt = tiles_[static_cast<size_t>(t[i])];
            r[i] = static_cast<int64_t>(std::llround((tt.lat_first - a[i]) / tt.dlat));
            c[i] = static_cast<int64_t>(std::llround((o[i] - tt.lon_first) / tt.dlon));
        }
    });
    return normalise(tile, row, col);
}

TerrainCells TerrainTileIndex::normalise(const torch::Tensor& tile, const torch::Tensor& row,
                                         const torch::Tensor& col) const {
    TerrainCells out{as_long(tile).clone(), as_long(row).clone(), as_long(col).clone()};
    const int64_t n = out.tile.numel();
    if (out.row.numel() != n || out.col.numel() != n) {
        throw std::invalid_argument("TerrainTileIndex::normalise: tile, row and col differ in length");
    }
    auto t = out.tile.accessor<int64_t, 1>();
    auto r = out.row.accessor<int64_t, 1>();
    auto c = out.col.accessor<int64_t, 1>();
    at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (t[i] < 0) continue;
            // Rows first, then columns, so a diagonal neighbour passes through
            // the tile beside it.
            for (int axis = 0; axis < 2 && t[i] >= 0; ++axis) {
                const auto& tt = tiles_[static_cast<size_t>(t[i])];
                int64_t& v = axis == 0 ? r[i] : c[i];
                const int64_t extent = axis == 0 ? tt.rows : tt.cols;
                if (v >= 0 && v < extent) continue;
                const auto [s, w] = tt.key();
                const bool after = v >= extent;
                const int64_t next = axis == 0 ? find(after ? s - 1 : s + 1, w) : find(s, after ? w + 1 : w - 1);
                if (next < 0) {
                    t[i] = -1;
                    break;
                }
                const auto& nt = tiles_[static_cast<size_t>(next)];
                const int64_t next_extent = axis == 0 ? nt.rows : nt.cols;
                v = after ? v - extent : v + next_extent;
                if (v < 0 || v >= next_extent) {
                    t[i] = -1;
                    break;
                }
                // A tile north or south may space its columns differently
                // (GLO-30 widens them at 50, 60, 70 and 80 degrees): the
                // column is the one nearest the same longitude there.
                if (axis == 0 && nt.dlon != tt.dlon) {
                    c[i] = static_cast<int64_t>(
                        std::llround((tt.lon_first + static_cast<double>(c[i]) * tt.dlon - nt.lon_first) / nt.dlon));
                }
                t[i] = next;
            }
        }
    });
    return out;
}

std::pair<torch::Tensor, torch::Tensor> TerrainTileIndex::centres(const TerrainCells& cells) const {
    const int64_t n = cells.tile.numel();
    torch::Tensor lat = torch::full({n}, kNaN, torch::kFloat64);
    torch::Tensor lon = torch::full({n}, kNaN, torch::kFloat64);
    const torch::Tensor tile = as_long(cells.tile);
    const torch::Tensor row = as_long(cells.row);
    const torch::Tensor col = as_long(cells.col);
    const auto t = tile.accessor<int64_t, 1>();
    const auto r = row.accessor<int64_t, 1>();
    const auto c = col.accessor<int64_t, 1>();
    auto a = lat.accessor<double, 1>();
    auto o = lon.accessor<double, 1>();
    at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (t[i] < 0) continue;
            const auto& tt = tiles_[static_cast<size_t>(t[i])];
            a[i] = tt.lat_first - static_cast<double>(r[i]) * tt.dlat;
            o[i] = tt.lon_first + static_cast<double>(c[i]) * tt.dlon;
        }
    });
    return {lat, lon};
}

// ---------------------------------------------------------------------------
// Bilinear reading and descriptors
// ---------------------------------------------------------------------------

torch::Tensor bilinear_elevation(const TerrainTileIndex& index, const torch::Tensor& lat,
                                 const torch::Tensor& lon, const TerrainCellValues& cell_values) {
    const torch::Tensor la = as_double(lat);
    const torch::Tensor lo = as_double(lon);
    const int64_t n = la.numel();
    const torch::Tensor tile = index.tile_of(la, lo);
    torch::Tensor ctile = torch::empty({4 * n}, torch::kInt64);
    torch::Tensor crow = torch::empty({4 * n}, torch::kInt64);
    torch::Tensor ccol = torch::empty({4 * n}, torch::kInt64);
    torch::Tensor weight = torch::zeros({4 * n}, torch::kFloat64);
    {
        // The two rows around each point, each placed in the tile that holds
        // it: a row past the tile's edge lies in the tile north or south,
        // which may space its columns differently.
        const auto a = la.accessor<double, 1>();
        const auto t = tile.accessor<int64_t, 1>();
        const auto& tiles = index.tiles();
        torch::Tensor rtile = torch::full({2 * n}, int64_t{-1}, torch::kInt64);
        torch::Tensor rrow = torch::zeros({2 * n}, torch::kInt64);
        torch::Tensor wrow = torch::zeros({2 * n}, torch::kFloat64);
        auto rt = rtile.accessor<int64_t, 1>();
        auto rr = rrow.accessor<int64_t, 1>();
        auto wl = wrow.accessor<double, 1>();
        at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                if (t[i] < 0) continue;
                const auto& tt = tiles[static_cast<size_t>(t[i])];
                const double fr = (tt.lat_first - a[i]) / tt.dlat;
                const double r0 = std::floor(fr);
                const double wr = fr - r0;
                for (int dr = 0; dr < 2; ++dr) {
                    rt[2 * i + dr] = t[i];
                    rr[2 * i + dr] = static_cast<int64_t>(r0) + dr;
                    wl[2 * i + dr] = dr ? wr : 1.0 - wr;
                }
            }
        });
        const TerrainCells rows = index.normalise(rtile, rrow, torch::zeros({2 * n}, torch::kInt64));
        const auto nt = rows.tile.accessor<int64_t, 1>();
        const auto nr = rows.row.accessor<int64_t, 1>();
        const auto o = lo.accessor<double, 1>();
        auto ct = ctile.accessor<int64_t, 1>();
        auto cr = crow.accessor<int64_t, 1>();
        auto cc = ccol.accessor<int64_t, 1>();
        auto w = weight.accessor<double, 1>();
        at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                for (int k = 0; k < 4; ++k) ct[4 * i + k] = -1;
                for (int dr = 0; dr < 2; ++dr) {
                    const int64_t rt_i = nt[2 * i + dr];
                    if (rt_i < 0) continue;
                    const auto& tt = tiles[static_cast<size_t>(rt_i)];
                    const double fc = (o[i] - tt.lon_first) / tt.dlon;
                    const double c0 = std::floor(fc);
                    const double wc = fc - c0;
                    for (int dc = 0; dc < 2; ++dc) {
                        const int k = 2 * dr + dc;
                        ct[4 * i + k] = rt_i;
                        cr[4 * i + k] = nr[2 * i + dr];
                        cc[4 * i + k] = static_cast<int64_t>(c0) + dc;
                        w[4 * i + k] = wl[2 * i + dr] * (dc ? wc : 1.0 - wc);
                    }
                }
            }
        });
    }
    const TerrainCells cells = index.normalise(ctile, crow, ccol);
    const torch::Tensor values = as_double(cell_values(cells));
    const torch::Tensor held = torch::isfinite(values);
    const torch::Tensor w = torch::where(held, weight, torch::zeros_like(weight)).view({n, 4});
    const torch::Tensor v = torch::where(held, values, torch::zeros_like(values)).view({n, 4});
    const torch::Tensor wsum = w.sum(1);
    return torch::where(wsum > 0, (w * v).sum(1) / wsum, torch::full_like(wsum, kNaN));
}

namespace {

// The 1 km lattice's offsets (metres east, metres north) of every point, in
// row-major order: (kTerrainLattice^2,) each.
std::pair<torch::Tensor, torch::Tensor> lattice_offsets() {
    const torch::Tensor axis = torch::linspace(-kTerrainWindowM / 2, kTerrainWindowM / 2,
                                               kTerrainLattice, torch::kFloat64);
    const auto grid = torch::meshgrid({axis, axis}, "ij");
    return {grid[1].reshape({-1}), grid[0].reshape({-1})};
}

void descriptors_chunk(const TerrainSource& source, const torch::Tensor& lat, const torch::Tensor& lon,
                       TerrainDescriptors& out, int64_t offset) {
    const int64_t m = lat.numel();
    const auto& index = source.index();
    using torch::indexing::Slice;
    const auto rows = Slice(offset, offset + m);

    // The 1 km lattice: mean and least-squares plane. Read first: its points
    // cover the cells the point descriptors below read, so a caching source
    // meets those cells already loaded.
    static const auto offsets = lattice_offsets();
    const torch::Tensor& ex = offsets.first;
    const torch::Tensor& ny = offsets.second;
    const int64_t p = ex.numel();
    const torch::Tensor la = lat.view({m, 1});
    const torch::Tensor lo = lon.view({m, 1});
    const torch::Tensor qlat = la + ny.view({1, p}) / kTerrainEarthRadiusM / kRadians;
    const torch::Tensor qlon =
        lo + ex.view({1, p}) / (kTerrainEarthRadiusM * torch::cos(la * kRadians)) / kRadians;
    const torch::Tensor zz = source.elevation(qlat.reshape({-1}), qlon.reshape({-1})).view({m, p});
    const torch::Tensor held = torch::isfinite(zz);
    const torch::Tensor w = held.to(torch::kFloat64);
    const torch::Tensor v = torch::where(held, zz, torch::zeros_like(zz));
    const torch::Tensor count = w.sum(1);
    const torch::Tensor enough = count >= static_cast<double>((p + 1) / 2);
    const torch::Tensor nan = torch::full({m}, kNaN, torch::kFloat64);
    out.elevation_1km.index_put_({rows}, torch::where(enough, v.sum(1) / count.clamp_min(1.0), nan));

    // Normal equations of z = a + b x + c y over the held points, per plot.
    const torch::Tensor X = torch::stack({torch::ones_like(ex), ex, ny}, 1);  // (p, 3)
    const torch::Tensor wX = w.unsqueeze(2) * X.unsqueeze(0);                 // (m, p, 3)
    torch::Tensor A = torch::matmul(wX.transpose(1, 2), X.unsqueeze(0).expand({m, p, 3}));
    const torch::Tensor b = torch::matmul(wX.transpose(1, 2), v.unsqueeze(2));  // (m, 3, 1)
    // Plots without enough points get the identity so the batched solve
    // stays defined; their result is replaced by NaN below.
    const torch::Tensor eye = torch::eye(3, torch::kFloat64).expand({m, 3, 3});
    A = torch::where(enough.view({m, 1, 1}), A, eye);
    const torch::Tensor coef = torch::linalg_solve(A, b).squeeze(2);
    const torch::Tensor tilt =
        torch::atan(torch::sqrt(coef.select(1, 1).pow(2) + coef.select(1, 2).pow(2))) / kRadians;
    out.slope_1km.index_put_({rows}, torch::where(enough, tilt, nan));

    out.elevation.index_put_({rows}, source.elevation(lat, lon));

    // Slope of the nearest cell from its four edge neighbours.
    const TerrainCells centre = index.nearest_cells(lat, lon);
    const auto [clat, clon] = index.centres(centre);
    const torch::Tensor dr = torch::tensor({-1, 1, 0, 0}, torch::kInt64);
    const torch::Tensor dc = torch::tensor({0, 0, -1, 1}, torch::kInt64);
    const TerrainCells nb = index.normalise(centre.tile.repeat_interleave(4),
                                            centre.row.repeat_interleave(4) + dr.repeat({m}),
                                            centre.col.repeat_interleave(4) + dc.repeat({m}));
    const auto [nlat, nlon] = index.centres(nb);
    const torch::Tensor z = source.elevation(nlat, nlon).view({m, 4});
    torch::Tensor dlat = torch::full({m}, kNaN, torch::kFloat64);
    torch::Tensor dlon = torch::full({m}, kNaN, torch::kFloat64);
    {
        const auto t = centre.tile.accessor<int64_t, 1>();
        auto a = dlat.accessor<double, 1>();
        auto o = dlon.accessor<double, 1>();
        for (int64_t i = 0; i < m; ++i) {
            if (t[i] < 0) continue;
            const auto& tt = index.tiles()[static_cast<size_t>(t[i])];
            a[i] = tt.dlat;
            o[i] = tt.dlon;
        }
    }
    const torch::Tensor dy = dlat * kRadians * kTerrainEarthRadiusM;
    const torch::Tensor dx = dlon * kRadians * kTerrainEarthRadiusM * torch::cos(clat * kRadians);
    // Rows run north to south: neighbour 0 is the northern one.
    const torch::Tensor gy = (z.select(1, 0) - z.select(1, 1)) / (2.0 * dy);
    const torch::Tensor gx = (z.select(1, 3) - z.select(1, 2)) / (2.0 * dx);
    out.slope.index_put_({rows}, torch::atan(torch::sqrt(gx * gx + gy * gy)) / kRadians);
}

}  // namespace

TerrainDescriptors terrain_descriptors(const TerrainSource& source, const torch::Tensor& lat,
                                       const torch::Tensor& lon, int64_t chunk) {
    const torch::Tensor la = as_double(lat);
    const torch::Tensor lo = as_double(lon);
    if (la.numel() != lo.numel()) {
        throw std::invalid_argument("terrain_descriptors: lat and lon differ in length");
    }
    if (chunk < 1) throw std::invalid_argument("terrain_descriptors: chunk must be >= 1");
    const int64_t n = la.numel();
    const auto blank = [n] { return torch::full({n}, kNaN, torch::kFloat64); };
    // Coordinates are read grouped by tile, so a source that decodes or loads
    // whole tiles meets each tile in as few chunks as possible, whatever order
    // the caller's plots come in; results return in the caller's order.
    const torch::Tensor order = torch::argsort(source.index().tile_of(la, lo), /*stable=*/true, 0, false);
    const torch::Tensor sla = la.index_select(0, order);
    const torch::Tensor slo = lo.index_select(0, order);
    TerrainDescriptors sorted{blank(), blank(), blank(), blank()};
    for (int64_t i = 0; i < n; i += chunk) {
        const int64_t m = std::min(chunk, n - i);
        descriptors_chunk(source, sla.narrow(0, i, m), slo.narrow(0, i, m), sorted, i);
    }
    TerrainDescriptors out{blank(), blank(), blank(), blank()};
    out.elevation.index_put_({order}, sorted.elevation);
    out.elevation_1km.index_put_({order}, sorted.elevation_1km);
    out.slope.index_put_({order}, sorted.slope);
    out.slope_1km.index_put_({order}, sorted.slope_1km);
    return out;
}

// ---------------------------------------------------------------------------
// .npy grids
// ---------------------------------------------------------------------------

namespace {

std::string read_bytes(const std::string& path) {
    return io::with_retry<io::IOError>([&] {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw io::IOError("cannot open '" + path + "'");
        std::ostringstream buffer;
        buffer << in.rdbuf();
        if (!in.good() && !in.eof()) throw io::IOError("cannot read '" + path + "'");
        return buffer.str();
    }, "terrain grid read");
}

// The value of `key` in a numpy header dict, as written (quotes kept).
std::string header_value(const std::string& header, const std::string& key, const std::string& path) {
    const std::string needle = "'" + key + "':";
    const auto at = header.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error("read_npy_grid: '" + path + "' has no '" + key + "' in its header");
    }
    auto begin = header.find_first_not_of(' ', at + needle.size());
    auto end = begin;
    if (header[begin] == '(') {
        end = header.find(')', begin) + 1;
    } else {
        end = header.find_first_of(",}", begin);
    }
    return header.substr(begin, end - begin);
}

}  // namespace

torch::Tensor read_npy_grid(const std::string& path) {
    const std::string bytes = read_bytes(path);
    static const char magic[] = "\x93NUMPY";
    if (bytes.size() < 10 || std::memcmp(bytes.data(), magic, 6) != 0) {
        throw std::runtime_error("read_npy_grid: '" + path + "' is not a .npy file");
    }
    const auto major = static_cast<unsigned char>(bytes[6]);
    size_t header_len = 0;
    size_t header_at = 0;
    if (major == 1) {
        header_len = static_cast<unsigned char>(bytes[8]) | (static_cast<size_t>(static_cast<unsigned char>(bytes[9])) << 8);
        header_at = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12) throw std::runtime_error("read_npy_grid: '" + path + "' is truncated");
        for (int k = 0; k < 4; ++k) {
            header_len |= static_cast<size_t>(static_cast<unsigned char>(bytes[8 + k])) << (8 * k);
        }
        header_at = 12;
    } else {
        throw std::runtime_error("read_npy_grid: '" + path + "' has .npy version " + std::to_string(major));
    }
    if (bytes.size() < header_at + header_len) {
        throw std::runtime_error("read_npy_grid: '" + path + "' is truncated");
    }
    const std::string header = bytes.substr(header_at, header_len);
    const std::string descr = header_value(header, "descr", path);
    if (descr != "'<f4'") {
        throw std::runtime_error("read_npy_grid: '" + path + "' holds " + descr + ", not float32 ('<f4')");
    }
    if (header_value(header, "fortran_order", path) != "False") {
        throw std::runtime_error("read_npy_grid: '" + path + "' is Fortran-ordered");
    }
    const std::string shape = header_value(header, "shape", path);
    std::vector<int64_t> dims;
    for (size_t i = 1; i < shape.size();) {
        const auto end = shape.find_first_of(",)", i);
        const std::string token = shape.substr(i, end - i);
        if (token.find_first_not_of(' ') != std::string::npos) dims.push_back(std::stoll(token));
        i = end + 1;
    }
    if (dims.size() != 2 || dims[0] < 1 || dims[1] < 1) {
        throw std::runtime_error("read_npy_grid: '" + path + "' is not a 2-D grid (shape " + shape + ")");
    }
    const size_t data_at = header_at + header_len;
    const size_t expected = static_cast<size_t>(dims[0] * dims[1]) * sizeof(float);
    if (bytes.size() - data_at != expected) {
        throw std::runtime_error("read_npy_grid: '" + path + "' holds " +
                                 std::to_string(bytes.size() - data_at) + " data bytes, shape " + shape +
                                 " needs " + std::to_string(expected));
    }
    torch::Tensor out = torch::empty({dims[0], dims[1]}, torch::kFloat32);
    std::memcpy(out.data_ptr<float>(), bytes.data() + data_at, expected);
    return out;
}

void write_npy_grid(const std::string& path, const torch::Tensor& grid) {
    if (grid.dim() != 2) throw std::invalid_argument("write_npy_grid: grid must be 2-D");
    const torch::Tensor g = grid.to(torch::kCPU, torch::kFloat32).contiguous();
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(g.size(0)) + ", " + std::to_string(g.size(1)) + "), }";
    const size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    if (!out) throw io::IOError("cannot write '" + path + "'");
    out.write("\x93NUMPY\x01\x00", 8);
    const auto len = static_cast<uint16_t>(header.size());
    const char len_bytes[2] = {static_cast<char>(len & 0xff), static_cast<char>(len >> 8)};
    out.write(len_bytes, 2);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(g.data_ptr<float>()),
              static_cast<std::streamsize>(g.numel() * sizeof(float)));
    if (!out) throw io::IOError("cannot write '" + path + "'");
}

// ---------------------------------------------------------------------------
// TerrainGrids
// ---------------------------------------------------------------------------

TerrainGrids::TerrainGrids(std::string dir, int64_t cache_tiles)
    : dir_(std::move(dir)), cache_(cache_tiles) {
    const std::string index_path = (std::filesystem::path(dir_) / kTerrainGridsIndexFile).string();
    const json::Value meta = json::read_file(index_path);
    if (!meta.contains("format") || meta.at("format").as_string() != kTerrainGridsFormat) {
        throw std::runtime_error("TerrainGrids: '" + index_path + "' is not a " + kTerrainGridsFormat +
                                 " index");
    }
    std::vector<TerrainTile> tiles;
    for (const auto& e : meta.at("tiles").items()) {
        tiles.push_back(terrain_tile_from_json(e));
        files_.push_back((std::filesystem::path(dir_) / e.at("file").as_string()).string());
    }
    index_ = TerrainTileIndex(std::move(tiles));
}

torch::Tensor TerrainGrids::grid(int64_t t) const {
    if (t < 0 || t >= index_.size()) throw std::out_of_range("TerrainGrids::grid: no tile " + std::to_string(t));
    return cache_.get_or_make(t, [&] {
        torch::Tensor g = read_npy_grid(files_[static_cast<size_t>(t)]);
        const auto& tile = index_.tiles()[static_cast<size_t>(t)];
        if (g.size(0) != tile.rows || g.size(1) != tile.cols) {
            throw std::runtime_error("TerrainGrids: '" + files_[static_cast<size_t>(t)] + "' is " +
                                     std::to_string(g.size(0)) + " x " + std::to_string(g.size(1)) +
                                     ", tiles.json says " + std::to_string(tile.rows) + " x " +
                                     std::to_string(tile.cols));
        }
        return g;
    });
}

torch::Tensor TerrainGrids::cell_values(const TerrainCells& cells) const {
    const torch::Tensor tile = as_long(cells.tile);
    const torch::Tensor row = as_long(cells.row);
    const torch::Tensor col = as_long(cells.col);
    const int64_t n = tile.numel();
    torch::Tensor out = torch::full({n}, kNaN, torch::kFloat64);
    const auto t = tile.accessor<int64_t, 1>();
    const auto r = row.accessor<int64_t, 1>();
    const auto c = col.accessor<int64_t, 1>();
    auto o = out.accessor<double, 1>();
    std::vector<int64_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return t[a] < t[b]; });
    size_t i = 0;
    while (i < order.size()) {
        const int64_t tt = t[order[i]];
        size_t j = i;
        while (j < order.size() && t[order[j]] == tt) ++j;
        if (tt >= 0) {
            const torch::Tensor g = grid(tt);
            const auto ga = g.accessor<float, 2>();
            for (size_t k = i; k < j; ++k) {
                const int64_t p = order[k];
                o[p] = static_cast<double>(ga[r[p]][c[p]]);
            }
        }
        i = j;
    }
    return out;
}

torch::Tensor TerrainGrids::elevation(const torch::Tensor& lat, const torch::Tensor& lon) const {
    return bilinear_elevation(index_, lat, lon, [this](const TerrainCells& cells) { return cell_values(cells); });
}

std::string TerrainGrids::describe() const {
    return "terrain grids '" + dir_ + "' (" + std::to_string(index_.size()) + " tiles)";
}

}  // namespace resolve
