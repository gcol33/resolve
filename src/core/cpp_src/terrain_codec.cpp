#include "resolve/terrain_codec.hpp"

#include "resolve/config_json.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace resolve {

namespace {

constexpr int kCodedFormatVersion = 1;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// Rice codes with a unary part this long escape to a 32-bit raw value, so a
// cliff (or a void filled with a far value) costs 64 bits, not thousands.
constexpr uint64_t kEscape = 32;
constexpr int kMaxK = 15;

class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}
    void bit(bool b) {
        acc_ = static_cast<uint8_t>((acc_ << 1) | (b ? 1 : 0));
        if (++n_ == 8) flush_byte();
    }
    void bits(uint64_t v, int count) {
        for (int i = count - 1; i >= 0; --i) bit(((v >> i) & 1u) != 0);
    }
    void rice(uint64_t v, int k) {
        const uint64_t q = v >> k;
        if (q >= kEscape) {
            for (uint64_t i = 0; i < kEscape; ++i) bit(true);
            bits(v, 32);
            return;
        }
        for (uint64_t i = 0; i < q; ++i) bit(true);
        bit(false);
        bits(v & ((uint64_t{1} << k) - 1), k);
    }
    // Exponential-Golomb of order 0.
    void exp_golomb(uint64_t n) {
        const uint64_t v = n + 1;
        int len = 0;
        while ((v >> len) > 1) ++len;
        for (int i = 0; i < len; ++i) bit(false);
        bits(v, len + 1);
    }
    void finish() {
        while (n_ != 0) bit(false);
    }

private:
    void flush_byte() {
        out_.push_back(acc_);
        acc_ = 0;
        n_ = 0;
    }
    std::vector<uint8_t>& out_;
    uint8_t acc_ = 0;
    int n_ = 0;
};

// Reads the stream BitWriter writes, most significant bit first, through a
// 64-bit window: a unary run is counted with one leading-ones or
// leading-zeros instruction instead of bit by bit.
class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}
    bool bit() { return bits(1) != 0; }
    // count in [0, 32].
    uint64_t bits(int count) {
        if (count == 0) return 0;
        refill();
        need(count);
        const uint64_t v = window_ >> (64 - count);
        consume(count);
        return v;
    }
    uint64_t rice(int k) {
        refill();
        const int ones = std::countl_one(window_);
        if (ones >= static_cast<int>(kEscape)) {
            consume(static_cast<int>(kEscape));
            return bits(32);
        }
        need(ones + 1);
        consume(ones + 1);
        return (static_cast<uint64_t>(ones) << k) | bits(k);
    }
    uint64_t exp_golomb() {
        refill();
        const int len = std::countl_zero(window_);
        need(len + 1);
        if (len > 31) throw std::runtime_error("terrain codec: an exponential-Golomb code is too long");
        consume(len);
        return bits(len + 1) - 1;
    }

private:
    // Fill the window to at least 57 valid bits, or to the stream's end; the
    // bits past the valid ones are zero.
    void refill() {
        while (valid_ <= 56 && next_ < size_) {
            window_ |= static_cast<uint64_t>(data_[next_++]) << (56 - valid_);
            valid_ += 8;
        }
    }
    void need(int count) const {
        if (count > valid_) throw std::runtime_error("terrain codec: block stream ends early");
    }
    void consume(int count) {
        window_ = count >= 64 ? 0 : window_ << count;
        valid_ -= count;
    }

    const uint8_t* data_;
    size_t size_;
    size_t next_ = 0;
    uint64_t window_ = 0;
    int valid_ = 0;
};

uint64_t zigzag(int64_t r) { return r >= 0 ? static_cast<uint64_t>(r) << 1 : (static_cast<uint64_t>(-r) << 1) - 1; }
int64_t unzigzag(uint64_t z) { return (z & 1u) ? -static_cast<int64_t>((z + 1) >> 1) : static_cast<int64_t>(z >> 1); }

// The Rice cost of values in bits for every k, and the best k.
std::pair<int, uint64_t> best_rice(const std::vector<uint64_t>& values) {
    int best_k = 0;
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (int k = 0; k <= kMaxK; ++k) {
        uint64_t cost = 0;
        for (const uint64_t v : values) {
            const uint64_t q = v >> k;
            cost += q >= kEscape ? kEscape + 32 : q + 1 + static_cast<uint64_t>(k);
        }
        if (cost < best) {
            best = cost;
            best_k = k;
        }
    }
    return {best_k, best};
}

uint64_t exp_golomb_bits(uint64_t n) {
    const uint64_t v = n + 1;
    uint64_t len = 0;
    while ((v >> len) > 1) ++len;
    return 2 * len + 1;
}

// Quantised values of a block in row-major order -> its Lorenzo residuals as
// zigzag integers (the first cell's residual is its own value, stored apart).
std::vector<uint64_t> block_residuals(const std::vector<int64_t>& q, int64_t h, int64_t w) {
    std::vector<uint64_t> z(static_cast<size_t>(h * w));
    for (int64_t i = 0; i < h; ++i) {
        for (int64_t j = 0; j < w; ++j) {
            const int64_t v = q[static_cast<size_t>(i * w + j)];
            int64_t p = 0;
            if (i > 0 && j > 0) {
                p = q[static_cast<size_t>(i * w + j - 1)] + q[static_cast<size_t>((i - 1) * w + j)] -
                    q[static_cast<size_t>((i - 1) * w + j - 1)];
            } else if (j > 0) {
                p = q[static_cast<size_t>(i * w + j - 1)];
            } else if (i > 0) {
                p = q[static_cast<size_t>((i - 1) * w + j)];
            } else {
                p = v;
            }
            z[static_cast<size_t>(i * w + j)] = zigzag(v - p);
        }
    }
    return z;
}

void encode_block(const std::vector<int64_t>& q, int64_t h, int64_t w, std::vector<uint8_t>& out) {
    const std::vector<uint64_t> z = block_residuals(q, h, w);
    const std::vector<uint64_t> rest(z.begin() + 1, z.end());
    const auto [k, rice_cost] = best_rice(rest);
    // Run mode: zero runs as exponential-Golomb, each non-zero value v as
    // Rice(v - 1).
    std::vector<uint64_t> nonzero;
    std::vector<uint64_t> runs;
    uint64_t run = 0;
    for (const uint64_t v : rest) {
        if (v == 0) {
            ++run;
        } else {
            runs.push_back(run);
            run = 0;
            nonzero.push_back(v - 1);
        }
    }
    runs.push_back(run);
    uint64_t run_cost = 0;
    for (const uint64_t r : runs) run_cost += exp_golomb_bits(r);
    const auto [k_run, nonzero_cost] = nonzero.empty() ? std::pair<int, uint64_t>{0, 0} : best_rice(nonzero);
    run_cost += nonzero_cost;

    BitWriter bw(out);
    bw.bits(static_cast<uint32_t>(static_cast<int32_t>(q.front())), 32);
    const bool run_mode = run_cost < rice_cost;
    bw.bit(run_mode);
    if (!run_mode) {
        bw.bits(static_cast<uint64_t>(k), 4);
        for (const uint64_t v : rest) bw.rice(v, k);
    } else {
        bw.bits(static_cast<uint64_t>(k_run), 4);
        size_t next = 0;
        for (size_t r = 0; r < runs.size(); ++r) {
            bw.exp_golomb(runs[r]);
            if (r + 1 < runs.size()) bw.rice(nonzero[next++], k_run);
        }
    }
    bw.finish();
}

// Decode one block's bytes into its quantised values, row-major: the
// residuals are read into `q` and then turned into values in place, in raster
// order, so each prediction reads neighbours already restored.
void decode_block(const uint8_t* data, size_t size, int64_t h, int64_t w, std::vector<int64_t>& q) {
    BitReader br(data, size);
    const size_t n = static_cast<size_t>(h * w);
    q.assign(n, 0);
    const auto first = static_cast<int32_t>(static_cast<uint32_t>(br.bits(32)));
    const bool run_mode = br.bit();
    const int k = static_cast<int>(br.bits(4));
    if (!run_mode) {
        for (size_t i = 1; i < n; ++i) q[i] = unzigzag(br.rice(k));
    } else {
        size_t i = 1;
        while (true) {
            const uint64_t run = br.exp_golomb();
            if (run > n - i) throw std::runtime_error("terrain codec: a zero run passes the block's end");
            i += static_cast<size_t>(run);
            if (i >= n) break;
            q[i++] = unzigzag(br.rice(k) + 1);
        }
    }
    q[0] = first;
    const auto ws = static_cast<size_t>(w);
    for (size_t c = 1; c < ws; ++c) q[c] += q[c - 1];
    for (int64_t r = 1; r < h; ++r) {
        int64_t* row = q.data() + static_cast<size_t>(r) * ws;
        const int64_t* above = row - ws;
        row[0] += above[0];
        for (size_t c = 1; c < ws; ++c) row[c] += row[c - 1] + above[c] - above[c - 1];
    }
}

// One block's bytes decoded to elevations (quantised value x quantum) into
// `out`, whose rows are `stride` floats apart.
void decode_block_values(const uint8_t* data, size_t size, int64_t h, int64_t w, double quantum, float* out,
                         int64_t stride, std::vector<int64_t>& scratch) {
    decode_block(data, size, h, w, scratch);
    for (int64_t i = 0; i < h; ++i) {
        for (int64_t j = 0; j < w; ++j) {
            out[i * stride + j] =
                static_cast<float>(static_cast<double>(scratch[static_cast<size_t>(i * w + j)]) * quantum);
        }
    }
}

int64_t block_count(int64_t rows, int64_t cols, int64_t block) {
    return ((rows + block - 1) / block) * ((cols + block - 1) / block);
}

// The shape (h, w) of block b of a tile cut into block x block blocks.
std::pair<int64_t, int64_t> block_shape(int64_t rows, int64_t cols, int64_t block, int64_t b) {
    const int64_t block_cols = (cols + block - 1) / block;
    return {std::min(block, rows - (b / block_cols) * block), std::min(block, cols - (b % block_cols) * block)};
}

// Blocks whose bytes lie less than this far apart are read in one request:
// on a spinning disk reading this many bytes takes about as long as one seek,
// so scattered queries cost few reads without reading whole tiles.
constexpr int64_t kCoalesceBytes = int64_t{1} << 18;

}  // namespace

void TerrainCodecConfig::validate() const {
    if (!(max_error > 0.0f) || !std::isfinite(max_error)) {
        throw std::invalid_argument("terrain codec: max_error must be > 0");
    }
    if (block < 2 || block > 4096) throw std::invalid_argument("terrain codec: block must lie in [2, 4096]");
}

TerrainTileCode encode_terrain_tile(const torch::Tensor& grid, double quantum, int64_t block) {
    if (grid.dim() != 2) throw std::invalid_argument("encode_terrain_tile: grid must be 2-D");
    if (!(quantum > 0.0)) throw std::invalid_argument("encode_terrain_tile: quantum must be > 0");
    const torch::Tensor g = grid.to(torch::kCPU, torch::kFloat64).contiguous();
    if (!torch::isfinite(g).all().item<bool>()) {
        throw std::invalid_argument("encode_terrain_tile: the grid holds non-finite elevations");
    }
    const int64_t rows = g.size(0);
    const int64_t cols = g.size(1);
    const auto a = g.accessor<double, 2>();
    std::vector<uint8_t> stream;
    std::vector<int64_t> offsets{0};
    std::vector<int64_t> q;
    for (int64_t i0 = 0; i0 < rows; i0 += block) {
        for (int64_t j0 = 0; j0 < cols; j0 += block) {
            const int64_t h = std::min(block, rows - i0);
            const int64_t w = std::min(block, cols - j0);
            q.resize(static_cast<size_t>(h * w));
            for (int64_t i = 0; i < h; ++i) {
                for (int64_t j = 0; j < w; ++j) {
                    const double v = std::nearbyint(a[i0 + i][j0 + j] / quantum);
                    if (std::abs(v) > 2.0e9) throw std::invalid_argument("encode_terrain_tile: quantum too small");
                    q[static_cast<size_t>(i * w + j)] = static_cast<int64_t>(v);
                }
            }
            encode_block(q, h, w, stream);
            offsets.push_back(static_cast<int64_t>(stream.size()));
        }
    }
    TerrainTileCode code;
    code.stream = torch::empty({static_cast<int64_t>(stream.size())}, torch::kUInt8);
    std::copy(stream.begin(), stream.end(), code.stream.data_ptr<uint8_t>());
    code.offsets = torch::tensor(offsets, torch::kInt64);
    return code;
}

torch::Tensor decode_terrain_tile(const TerrainTileCode& code, int64_t rows, int64_t cols, double quantum,
                                  int64_t block) {
    const TerrainTileCode cpu = terrain_tile_code_on_cpu(code);
    const int64_t n_blocks = block_count(rows, cols, block);
    if (cpu.offsets.numel() != n_blocks + 1 || cpu.offsets[-1].item<int64_t>() != cpu.stream.numel()) {
        throw std::runtime_error("decode_terrain_tile: block offsets do not fit the stream");
    }
    torch::Tensor out = torch::empty({rows, cols}, torch::kFloat32);
    float* o = out.data_ptr<float>();
    std::vector<int64_t> q;
    const int64_t block_cols = (cols + block - 1) / block;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const int64_t i0 = (b / block_cols) * block;
        const int64_t j0 = (b % block_cols) * block;
        decode_terrain_block(cpu, rows, cols, quantum, block, b, o + i0 * cols + j0, cols, q);
    }
    return out;
}

TerrainTileCode terrain_tile_code_on_cpu(const TerrainTileCode& code) {
    return {code.stream.to(torch::kCPU, torch::kUInt8).contiguous(),
            code.offsets.to(torch::kCPU, torch::kInt64).contiguous()};
}

void decode_terrain_block(const TerrainTileCode& code, int64_t rows, int64_t cols, double quantum, int64_t block,
                          int64_t b, float* out, int64_t stride, std::vector<int64_t>& scratch) {
    const int64_t n_blocks = block_count(rows, cols, block);
    if (b < 0 || b >= n_blocks) throw std::out_of_range("decode_terrain_block: no block " + std::to_string(b));
    if (code.offsets.numel() != n_blocks + 1 || code.offsets.scalar_type() != torch::kInt64 ||
        code.stream.scalar_type() != torch::kUInt8 || !code.stream.is_contiguous() || !code.offsets.is_contiguous() ||
        !code.stream.device().is_cpu() || !code.offsets.device().is_cpu()) {
        throw std::runtime_error("decode_terrain_block: the code is not a contiguous CPU stream with one offset per block");
    }
    const int64_t* off = code.offsets.data_ptr<int64_t>();
    if (off[b] < 0 || off[b + 1] < off[b] || off[b + 1] > code.stream.numel()) {
        throw std::runtime_error("decode_terrain_block: block offsets do not fit the stream");
    }
    const auto [h, w] = block_shape(rows, cols, block, b);
    decode_block_values(code.stream.data_ptr<uint8_t>() + off[b], static_cast<size_t>(off[b + 1] - off[b]), h, w,
                        quantum, out, stride, scratch);
}

std::vector<std::pair<std::string, TerrainTileMetrics>> encode_terrain_grids(const TerrainGrids& grids,
                                                                             const TerrainCodecConfig& config,
                                                                             const std::string& path,
                                                                             const std::vector<std::string>& tiles) {
    config.validate();
    const double quantum = 2.0 * static_cast<double>(config.max_error);
    std::vector<int64_t> selected;
    for (int64_t t = 0; t < grids.index().size(); ++t) {
        const std::string& name = grids.index().tiles()[static_cast<size_t>(t)].name;
        if (tiles.empty() || std::find(tiles.begin(), tiles.end(), name) != tiles.end()) selected.push_back(t);
    }
    if (selected.empty()) throw std::invalid_argument("encode_terrain_grids: no tile selected");

    // Tiles are coded on all threads, each on its own.
    std::vector<TerrainTileCode> codes(selected.size());
    std::vector<TerrainTileMetrics> metrics(selected.size());
    at::parallel_for(0, static_cast<int64_t>(selected.size()), 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t t = selected[static_cast<size_t>(i)];
            const TerrainTile& tile = grids.index().tiles()[static_cast<size_t>(t)];
            const auto started = std::chrono::steady_clock::now();
            const torch::Tensor grid = grids.grid(t);
            TerrainTileCode code = encode_terrain_tile(grid, quantum, config.block);
            const torch::Tensor back = decode_terrain_tile(code, tile.rows, tile.cols, quantum, config.block);
            TerrainTileMetrics m = terrain_reconstruction_metrics(back, grid, tile);
            m.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            m.bytes = code.stream.numel() + code.offsets.numel() * 8;
            codes[static_cast<size_t>(i)] = std::move(code);
            metrics[static_cast<size_t>(i)] = m;
        }
    });

    // Every tile's block offsets come first in the file, then the streams in
    // tile order: a reader finds the offsets of the tiles a query reaches in
    // one small region, and then reads the streams in one pass.
    json::Value entries = json::Value::array();
    std::vector<std::pair<std::string, torch::Tensor>> tensors;
    std::vector<std::pair<std::string, TerrainTileMetrics>> out;
    for (size_t i = 0; i < selected.size(); ++i) {
        const TerrainTile& tile = grids.index().tiles()[static_cast<size_t>(selected[i])];
        json::Value e = json::Value::object();
        e.set("tile", terrain_tile_to_json(tile));
        e.set("quantum", quantum);
        e.set("metrics", terrain_tile_metrics_to_json(metrics[i]));
        entries.push_back(std::move(e));
        tensors.emplace_back(tile.name + "/offsets", codes[i].offsets);
        out.emplace_back(tile.name, metrics[i]);
    }
    for (size_t i = 0; i < selected.size(); ++i) {
        tensors.emplace_back(grids.index().tiles()[static_cast<size_t>(selected[i])].name + "/stream",
                             codes[i].stream);
    }
    json::Value meta = json::Value::object();
    meta.set("format", kTerrainCodedFormat);
    meta.set("version", kCodedFormatVersion);
    meta.set("config", config_to_json(config));
    meta.set("tiles", std::move(entries));
    write_tensor_pack(path, meta, tensors);
    return out;
}

TerrainCodedField::TerrainCodedField(const std::string& path, int64_t cache_blocks)
    : pack_(path), blocks_(cache_blocks) {
    const json::Value& meta = pack_.meta();
    if (!meta.contains("format") || meta.at("format").as_string() != kTerrainCodedFormat) {
        throw std::runtime_error("'" + path + "' is not a coded terrain field");
    }
    if (meta.at("version").as_int() != kCodedFormatVersion) {
        throw std::runtime_error("'" + path + "' is coded terrain field version " +
                                 std::to_string(meta.at("version").as_int()));
    }
    config_ = config_from_json<TerrainCodecConfig>(meta.at("config"));
    std::vector<TerrainTile> tiles;
    for (const auto& e : meta.at("tiles").items()) {
        tiles.push_back(terrain_tile_from_json(e.at("tile")));
        metrics_.push_back(terrain_tile_metrics_from_json(e.at("metrics")));
    }
    int64_t base = 0;
    for (const TerrainTile& tile : tiles) {
        block_base_.push_back(base);
        base += block_count(tile.rows, tile.cols, config_.block);
    }
    offsets_.resize(tiles.size());
    index_ = TerrainTileIndex(std::move(tiles));
}

const TerrainTile& TerrainCodedField::tile_at(int64_t t, const char* who) const {
    if (t < 0 || t >= index_.size()) {
        throw std::out_of_range(std::string("TerrainCodedField::") + who + ": no tile " + std::to_string(t));
    }
    return index_.tiles()[static_cast<size_t>(t)];
}

double TerrainCodedField::quantum() const noexcept { return 2.0 * static_cast<double>(config_.max_error); }

std::vector<torch::Tensor> TerrainCodedField::offsets(const std::vector<int64_t>& tiles) const {
    std::vector<torch::Tensor> out(tiles.size());
    std::vector<size_t> missing;
    {
        std::lock_guard<std::mutex> lock(offsets_mutex_);
        for (size_t i = 0; i < tiles.size(); ++i) {
            (void)tile_at(tiles[i], "offsets");
            const torch::Tensor& held = offsets_[static_cast<size_t>(tiles[i])];
            if (held.defined()) {
                out[i] = held;
            } else {
                missing.push_back(i);
            }
        }
    }
    if (missing.empty()) return out;
    std::vector<std::string> names;
    for (const size_t i : missing) names.push_back(index_.tiles()[static_cast<size_t>(tiles[i])].name + "/offsets");
    const std::vector<torch::Tensor> read = pack_.read_many(names, static_cast<uint64_t>(kCoalesceBytes));
    std::lock_guard<std::mutex> lock(offsets_mutex_);
    for (size_t m = 0; m < missing.size(); ++m) {
        const TerrainTile& tile = index_.tiles()[static_cast<size_t>(tiles[missing[m]])];
        const torch::Tensor& off = read[m];
        const int64_t n_blocks = block_count(tile.rows, tile.cols, config_.block);
        if (off.scalar_type() != torch::kInt64 || off.dim() != 1 || off.numel() != n_blocks + 1 ||
            off[0].item<int64_t>() != 0 || (off.diff() < 0).any().item<bool>()) {
            throw std::runtime_error("'" + pack_.path() + "': tile '" + tile.name +
                                     "' has block offsets that do not fit its blocks");
        }
        torch::Tensor& held = offsets_[static_cast<size_t>(tiles[missing[m]])];
        if (!held.defined()) held = off.contiguous();
        out[missing[m]] = held;
    }
    return out;
}

torch::Tensor TerrainCodedField::grid(int64_t t) const {
    const TerrainTile& tile = tile_at(t, "grid");
    return decode_terrain_tile({pack_.read(tile.name + "/stream"), offsets({t}).front()}, tile.rows, tile.cols, quantum(),
                               config_.block);
}

torch::Tensor TerrainCodedField::block(int64_t t, int64_t b) const {
    const TerrainTile& tile = tile_at(t, "block");
    if (b < 0 || b >= block_count(tile.rows, tile.cols, config_.block)) {
        throw std::out_of_range("TerrainCodedField::block: no block " + std::to_string(b));
    }
    return blocks({block_base_[static_cast<size_t>(t)] + b}).front();
}

std::vector<torch::Tensor> TerrainCodedField::blocks(const std::vector<int64_t>& keys) const {
    std::vector<torch::Tensor> out(keys.size());
    // A block to decode: where its result goes, its tile and position, and
    // where its bytes sit in the buffers read below.
    struct Job {
        size_t slot;
        int64_t tile;
        int64_t b;
        size_t buffer;
        int64_t at;
        int64_t bytes;
    };
    std::vector<Job> jobs;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (std::optional<torch::Tensor> hit = blocks_.find(keys[i])) {
            out[i] = *std::move(hit);
            continue;
        }
        const auto next = std::upper_bound(block_base_.begin(), block_base_.end(), keys[i]);
        const int64_t t = static_cast<int64_t>(next - block_base_.begin()) - 1;
        jobs.push_back({i, t, keys[i] - block_base_[static_cast<size_t>(t)], 0, 0, 0});
    }
    if (jobs.empty()) return out;

    // The tiles' block offsets, then one read per run of nearby blocks, both
    // in file order and one at a time: concurrent reads make a spinning disk
    // seek between them.
    std::sort(jobs.begin(), jobs.end(),
              [](const Job& x, const Job& y) { return x.tile != y.tile ? x.tile < y.tile : x.b < y.b; });
    std::vector<int64_t> tiles;
    for (const Job& job : jobs) {
        if (tiles.empty() || tiles.back() != job.tile) tiles.push_back(job.tile);
    }
    const std::vector<torch::Tensor> tile_offsets = offsets(tiles);
    // A byte range of one tile's stream.
    struct Run {
        int64_t tile;
        int64_t begin;
        int64_t end;
    };
    std::vector<Run> runs;
    for (size_t j = 0, ti = 0; j < jobs.size(); ++ti) {
        const int64_t t = tiles[ti];
        const int64_t* off = tile_offsets[ti].data_ptr<int64_t>();
        while (j < jobs.size() && jobs[j].tile == t) {
            const int64_t begin = off[jobs[j].b];
            int64_t end = off[jobs[j].b + 1];
            size_t last = j;
            while (last + 1 < jobs.size() && jobs[last + 1].tile == t &&
                   off[jobs[last + 1].b] - end < kCoalesceBytes) {
                ++last;
                end = std::max(end, off[jobs[last].b + 1]);
            }
            for (size_t k = j; k <= last; ++k) {
                jobs[k].buffer = runs.size();
                jobs[k].at = off[jobs[k].b] - begin;
                jobs[k].bytes = off[jobs[k].b + 1] - off[jobs[k].b];
            }
            runs.push_back({t, begin, end});
            j = last + 1;
        }
    }
    std::vector<std::vector<uint8_t>> buffers(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        const Run& run = runs[i];
        buffers[i].resize(static_cast<size_t>(run.end - run.begin));
        pack_.read_bytes(index_.tiles()[static_cast<size_t>(run.tile)].name + "/stream",
                         static_cast<uint64_t>(run.begin), static_cast<uint64_t>(run.end - run.begin),
                         buffers[i].data());
    }

    std::vector<torch::Tensor> decoded(jobs.size());
    const double q = quantum();
    at::parallel_for(0, static_cast<int64_t>(jobs.size()), 1, [&](int64_t begin, int64_t end) {
        std::vector<int64_t> scratch;
        for (int64_t j = begin; j < end; ++j) {
            const Job& job = jobs[static_cast<size_t>(j)];
            const TerrainTile& tile = index_.tiles()[static_cast<size_t>(job.tile)];
            const auto [h, w] = block_shape(tile.rows, tile.cols, config_.block, job.b);
            torch::Tensor values = torch::empty({h, w}, torch::kFloat32);
            decode_block_values(buffers[job.buffer].data() + job.at, static_cast<size_t>(job.bytes), h, w, q,
                                values.data_ptr<float>(), w, scratch);
            decoded[static_cast<size_t>(j)] = std::move(values);
        }
    });
    for (size_t j = 0; j < jobs.size(); ++j) {
        out[jobs[j].slot] = blocks_.put(keys[jobs[j].slot], std::move(decoded[j]));
    }
    return out;
}

torch::Tensor TerrainCodedField::cell_values(const TerrainCells& cells) const {
    const torch::Tensor tile = cells.tile.to(torch::kCPU, torch::kInt64).reshape({-1}).contiguous();
    const torch::Tensor row = cells.row.to(torch::kCPU, torch::kInt64).reshape({-1}).contiguous();
    const torch::Tensor col = cells.col.to(torch::kCPU, torch::kInt64).reshape({-1}).contiguous();
    const int64_t n = tile.numel();
    if (row.numel() != n || col.numel() != n) {
        throw std::invalid_argument("TerrainCodedField::cell_values: tile, row and col differ in length");
    }
    const int64_t* t = tile.data_ptr<int64_t>();
    const int64_t* r = row.data_ptr<int64_t>();
    const int64_t* c = col.data_ptr<int64_t>();
    const int64_t bs = config_.block;

    // Each cell's block key, -1 for a cell outside every tile.
    std::vector<int64_t> key(static_cast<size_t>(n));
    at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            if (t[i] < 0) {
                key[static_cast<size_t>(i)] = -1;
                continue;
            }
            const TerrainTile& tt = tile_at(t[i], "cell_values");
            if (r[i] < 0 || r[i] >= tt.rows || c[i] < 0 || c[i] >= tt.cols) {
                throw std::out_of_range("TerrainCodedField::cell_values: cell (" + std::to_string(r[i]) + ", " +
                                        std::to_string(c[i]) + ") lies outside tile '" + tt.name + "'");
            }
            const int64_t block_cols = (tt.cols + bs - 1) / bs;
            key[static_cast<size_t>(i)] =
                block_base_[static_cast<size_t>(t[i])] + (r[i] / bs) * block_cols + c[i] / bs;
        }
    });
    std::vector<int64_t> distinct;
    for (const int64_t k : key) {
        if (k >= 0 && (distinct.empty() || distinct.back() != k)) distinct.push_back(k);
    }
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    const std::vector<torch::Tensor> values = blocks(distinct);

    torch::Tensor out = torch::full({n}, kNaN, torch::kFloat64);
    double* o = out.data_ptr<double>();
    at::parallel_for(0, n, kTerrainParallelGrain, [&](int64_t begin, int64_t end) {
        int64_t current = -1;
        const float* data = nullptr;
        int64_t width = 0;
        for (int64_t i = begin; i < end; ++i) {
            const int64_t k = key[static_cast<size_t>(i)];
            if (k < 0) continue;
            if (k != current) {
                const auto at = std::lower_bound(distinct.begin(), distinct.end(), k) - distinct.begin();
                const torch::Tensor& v = values[static_cast<size_t>(at)];
                data = v.data_ptr<float>();
                width = v.size(1);
                current = k;
            }
            o[i] = static_cast<double>(data[(r[i] % bs) * width + c[i] % bs]);
        }
    });
    return out;
}

torch::Tensor TerrainCodedField::elevation(const torch::Tensor& lat, const torch::Tensor& lon) const {
    return bilinear_elevation(index_, lat, lon, [this](const TerrainCells& cells) { return cell_values(cells); });
}

std::string TerrainCodedField::describe() const {
    return "coded terrain field '" + pack_.path() + "' (" + std::to_string(index_.size()) + " tiles, max error " +
           std::to_string(config_.max_error) + " m)";
}

std::shared_ptr<TerrainSource> open_terrain_field(const std::string& path, torch::Device device) {
    const std::string format = TensorPackReader(path).meta().at("format").as_string();
    if (format == kTerrainCodedFormat) return std::make_shared<TerrainCodedField>(path);
    if (format == kTerrainFieldFormat) return std::make_shared<TerrainField>(path, device);
    throw std::runtime_error("'" + path + "' is a tensor pack of format '" + format + "', not a terrain field");
}

}  // namespace resolve
