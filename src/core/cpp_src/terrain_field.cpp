#include "resolve/terrain_field.hpp"

#include "resolve/config_json.hpp"
#include "resolve/lr_schedule.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace resolve {

namespace {

constexpr double kRadians = std::numbers::pi / 180.0;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kHashPrime = 2654435761LL;
constexpr int kTileFormatVersion = 1;
constexpr int kFieldFormatVersion = 1;

}  // namespace

TerrainTileMetrics terrain_tile_metrics_from_json(const json::Value& v) {
    TerrainTileMetrics m;
    m.rmse = v.at("rmse").as_number();
    m.mae = v.at("mae").as_number();
    m.p99 = v.at("p99").as_number();
    m.max_abs = v.at("max_abs").as_number();
    m.slope_mae = v.at("slope_mae").as_number();
    m.seconds = v.at("seconds").as_number();
    m.bytes = v.at("bytes").as_int();
    return m;
}

namespace {

int64_t finest_resolution(const TerrainFieldConfig& c, int64_t rows, int64_t cols) {
    const auto finest = static_cast<int64_t>(std::floor(static_cast<double>(std::max(rows, cols)) * c.finest_fraction));
    return std::max(finest, c.coarsest_resolution);
}

// The tile's local position of cells: (row / (rows - 1), col / (cols - 1)).
torch::Tensor local_uv(const torch::Tensor& row, const torch::Tensor& col, int64_t rows, int64_t cols) {
    return torch::stack({row.to(torch::kFloat32) / static_cast<float>(rows - 1),
                         col.to(torch::kFloat32) / static_cast<float>(cols - 1)},
                        1);
}

}  // namespace

torch::Tensor terrain_cell_slopes(const torch::Tensor& z, const TerrainTile& tile) {
    using torch::indexing::Slice;
    const int64_t rows = z.size(0);
    const double dy = tile.dlat * kRadians * kTerrainEarthRadiusM;
    const torch::Tensor lat = tile.lat_first - torch::arange(1, rows - 1, z.options()).to(torch::kFloat64) * tile.dlat;
    const torch::Tensor dx =
        (tile.dlon * kRadians * kTerrainEarthRadiusM * torch::cos(lat * kRadians)).to(z.scalar_type()).view({-1, 1});
    const torch::Tensor gy = (z.index({Slice(0, -2), Slice(1, -1)}) - z.index({Slice(2), Slice(1, -1)})) / (2.0 * dy);
    const torch::Tensor gx = (z.index({Slice(1, -1), Slice(2)}) - z.index({Slice(1, -1), Slice(0, -2)})) / (2.0 * dx);
    return torch::atan(torch::sqrt(gx * gx + gy * gy)) / kRadians;
}

TerrainTileMetrics terrain_reconstruction_metrics(const torch::Tensor& reconstructed, const torch::Tensor& grid,
                                                  const TerrainTile& tile) {
    const torch::Tensor err = (reconstructed - grid).to(torch::kFloat64);
    const torch::Tensor abs = err.abs().reshape({-1});
    TerrainTileMetrics out;
    out.rmse = err.pow(2).mean().sqrt().item<double>();
    out.mae = abs.mean().item<double>();
    out.max_abs = abs.max().item<double>();
    const int64_t stride = std::max<int64_t>(1, abs.numel() / (1 << 22));
    out.p99 = torch::quantile(abs.slice(0, 0, abs.numel(), stride), 0.99).item<double>();
    out.slope_mae =
        (terrain_cell_slopes(reconstructed, tile) - terrain_cell_slopes(grid, tile)).abs().mean().item<double>();
    return out;
}

void TerrainFieldConfig::validate() const {
    if (levels < 1) throw std::invalid_argument("terrain field: levels must be >= 1");
    if (log2_table_size < 4 || log2_table_size > 30) {
        throw std::invalid_argument("terrain field: log2_table_size must lie in [4, 30]");
    }
    if (features < 1) throw std::invalid_argument("terrain field: features must be >= 1");
    if (coarsest_resolution < 1) throw std::invalid_argument("terrain field: coarsest_resolution must be >= 1");
    if (!(finest_fraction > 0.0f)) throw std::invalid_argument("terrain field: finest_fraction must be > 0");
    if (hidden_dim < 1 || n_layers < 1) throw std::invalid_argument("terrain field: hidden_dim and n_layers must be >= 1");
    if (base_cells < 1) throw std::invalid_argument("terrain field: base_cells must be >= 1");
    if (steps < 1 || batch_size < 1) throw std::invalid_argument("terrain field: steps and batch_size must be >= 1");
    if (!(lr > 0.0f)) throw std::invalid_argument("terrain field: lr must be > 0");
    if (!(gradient_weight >= 0.0f)) throw std::invalid_argument("terrain field: gradient_weight must be >= 0");
}

// ---------------------------------------------------------------------------
// Hash encoding
// ---------------------------------------------------------------------------

HashGridEncodingImpl::HashGridEncodingImpl(int64_t levels, int64_t log2_table_size, int64_t features,
                                           int64_t coarsest, int64_t finest)
    : levels_(levels), table_size_(int64_t{1} << log2_table_size), features_(features) {
    const double growth =
        levels > 1 ? std::exp((std::log(static_cast<double>(finest)) - std::log(static_cast<double>(coarsest))) /
                              static_cast<double>(levels - 1))
                   : 1.0;
    for (int64_t l = 0; l < levels; ++l) {
        resolutions_.push_back(static_cast<int64_t>(std::floor(static_cast<double>(coarsest) * std::pow(growth, l))));
    }
    table = register_parameter("table", torch::empty({levels * table_size_, features}).uniform_(-1e-4, 1e-4));
}

torch::Tensor HashGridEncodingImpl::forward(const torch::Tensor& xy) {
    const int64_t n = xy.size(0);
    const auto opts = xy.options();
    const torch::Tensor res = torch::tensor(resolutions_, torch::kInt64).to(xy.device());
    const torch::Tensor resf = res.to(opts.dtype()).view({levels_, 1, 1});
    const torch::Tensor p = xy.unsqueeze(0) * resf;                 // (L, n, 2)
    const torch::Tensor p0 = torch::floor(p);
    const torch::Tensor f = p - p0;
    const torch::Tensor i0 = p0.to(torch::kInt64);
    const torch::Tensor stride = (res + 1).view({levels_, 1});
    const torch::Tensor dense = ((res + 1) * (res + 1) <= table_size_).view({levels_, 1});
    const torch::Tensor offset = (torch::arange(levels_, res.options()) * table_size_).view({levels_, 1});
    torch::Tensor acc = torch::zeros({levels_, n, features_}, table.options());
    for (int corner = 0; corner < 4; ++corner) {
        const int64_t dx = corner >> 1;
        const int64_t dy = corner & 1;
        const torch::Tensor cx = i0.select(2, 0) + dx;
        const torch::Tensor cy = i0.select(2, 1) + dy;
        const torch::Tensor dense_idx = cx * stride + cy;
        const torch::Tensor hash_idx = torch::bitwise_and(torch::bitwise_xor(cx, cy * kHashPrime), table_size_ - 1);
        const torch::Tensor idx = torch::where(dense, dense_idx, hash_idx) + offset;  // (L, n)
        const torch::Tensor w = (dx ? f.select(2, 0) : 1.0 - f.select(2, 0)) *
                                (dy ? f.select(2, 1) : 1.0 - f.select(2, 1));
        acc = acc + w.unsqueeze(2).to(table.scalar_type()) *
                        table.index_select(0, idx.reshape({-1})).view({levels_, n, features_});
    }
    return acc.permute({1, 0, 2}).reshape({n, levels_ * features_});
}

// ---------------------------------------------------------------------------
// Tile field
// ---------------------------------------------------------------------------

TerrainTileFieldImpl::TerrainTileFieldImpl(const TerrainFieldConfig& config, int64_t rows_, int64_t cols_)
    : rows(rows_), cols(cols_) {
    config.validate();
    encoding = register_module(
        "encoding", HashGridEncoding(config.levels, config.log2_table_size, config.features,
                                     config.coarsest_resolution, finest_resolution(config, rows, cols)));
    mlp = torch::nn::Sequential();
    int64_t width = config.levels * config.features;
    for (int64_t l = 0; l < config.n_layers; ++l) {
        mlp->push_back(torch::nn::Linear(width, config.hidden_dim));
        mlp->push_back(torch::nn::ReLU());
        width = config.hidden_dim;
    }
    mlp->push_back(torch::nn::Linear(width, 1));
    mlp = register_module("mlp", mlp);
    base_dm = register_buffer("base_dm", torch::zeros({config.base_cells, config.base_cells}, torch::kInt16));
}

torch::Tensor TerrainTileFieldImpl::detail(const torch::Tensor& uv) {
    return mlp->forward(encoding->forward(uv)).squeeze(1);
}

torch::Tensor TerrainTileFieldImpl::base_at(const torch::Tensor& row, const torch::Tensor& col) const {
    const int64_t b = base_dm.size(0);
    const torch::Tensor base = base_dm.to(torch::kFloat32) / 10.0f;
    const auto block = [b](const torch::Tensor& i, int64_t extent) {
        // Block coordinate of a cell (block centres at integers), as an
        // area-averaged grid is read with align_corners = false.
        return ((i.to(torch::kFloat32) + 0.5f) * static_cast<float>(b) / static_cast<float>(extent) - 0.5f)
            .clamp(0.0f, static_cast<float>(b - 1));
    };
    const torch::Tensor br = block(row, rows);
    const torch::Tensor bc = block(col, cols);
    const torch::Tensor r0 = torch::floor(br).to(torch::kInt64);
    const torch::Tensor c0 = torch::floor(bc).to(torch::kInt64);
    const torch::Tensor r1 = (r0 + 1).clamp_max(b - 1);
    const torch::Tensor c1 = (c0 + 1).clamp_max(b - 1);
    const torch::Tensor wr = br - r0.to(torch::kFloat32);
    const torch::Tensor wc = bc - c0.to(torch::kFloat32);
    const torch::Tensor flat = base.reshape({-1});
    const auto at = [&](const torch::Tensor& r, const torch::Tensor& c) { return flat.index_select(0, r * b + c); };
    return (1 - wr) * (1 - wc) * at(r0, c0) + (1 - wr) * wc * at(r0, c1) + wr * (1 - wc) * at(r1, c0) +
           wr * wc * at(r1, c1);
}

torch::Tensor TerrainTileFieldImpl::cells(const torch::Tensor& row, const torch::Tensor& col) {
    const torch::Tensor r = row.clamp(0, rows - 1);
    const torch::Tensor c = col.clamp(0, cols - 1);
    return detail(local_uv(r, c, rows, cols)) * static_cast<float>(scale) + base_at(r, c);
}

json::Value terrain_tile_metrics_to_json(const TerrainTileMetrics& m) {
    json::Value v = json::Value::object();
    v.set("rmse", m.rmse);
    v.set("mae", m.mae);
    v.set("p99", m.p99);
    v.set("max_abs", m.max_abs);
    v.set("slope_mae", m.slope_mae);
    v.set("seconds", m.seconds);
    v.set("bytes", m.bytes);
    return v;
}

namespace {

// The tile's parameters as stored: float16 network, int16 base.
std::vector<std::pair<std::string, torch::Tensor>> stored_tensors(const TerrainTileField& field,
                                                                  const std::string& prefix) {
    std::vector<std::pair<std::string, torch::Tensor>> out;
    for (const auto& p : field->named_parameters()) {
        out.emplace_back(prefix + p.key(), p.value().detach().to(torch::kCPU, torch::kFloat16));
    }
    out.emplace_back(prefix + "base_dm", field->base_dm.to(torch::kCPU));
    return out;
}

// Load stored tensors into `field` (float16 widened to the field's dtype).
void load_stored(TerrainTileField& field, const std::function<torch::Tensor(const std::string&)>& read,
                 const std::string& prefix) {
    torch::NoGradGuard no_grad;
    for (auto& p : field->named_parameters()) {
        const torch::Tensor t = read(prefix + p.key());
        if (t.sizes() != p.value().sizes()) {
            throw std::runtime_error("terrain field: parameter '" + prefix + p.key() + "' has the wrong shape");
        }
        p.value().copy_(t.to(p.value().device(), p.value().scalar_type()));
    }
    const torch::Tensor base = read(prefix + "base_dm");
    if (base.sizes() != field->base_dm.sizes()) {
        throw std::runtime_error("terrain field: '" + prefix + "base_dm' has the wrong shape");
    }
    field->base_dm.copy_(base.to(field->base_dm.device()));
}

TerrainTileMetrics evaluate_tile(TerrainTileField& field, const torch::Tensor& z, const TerrainTile& tile) {
    torch::NoGradGuard no_grad;
    field->eval();
    const int64_t rows = z.size(0);
    const int64_t cols = z.size(1);
    torch::Tensor rec = torch::empty_like(z);
    constexpr int64_t kChunkRows = 256;
    const torch::Tensor col = torch::arange(cols, z.options().dtype(torch::kInt64));
    for (int64_t r0 = 0; r0 < rows; r0 += kChunkRows) {
        const int64_t m = std::min(kChunkRows, rows - r0);
        const torch::Tensor row = torch::arange(r0, r0 + m, col.options()).repeat_interleave(cols);
        rec.narrow(0, r0, m).copy_(field->cells(row, col.repeat({m})).view({m, cols}));
    }
    return terrain_reconstruction_metrics(rec, z, tile);
}

}  // namespace

TerrainTileMetrics fit_terrain_tile(const torch::Tensor& grid, const TerrainTile& tile,
                                    const TerrainFieldConfig& config, const std::string& path) {
    config.validate();
    if (grid.dim() != 2 || grid.size(0) != tile.rows || grid.size(1) != tile.cols) {
        throw std::invalid_argument("fit_terrain_tile: grid shape does not match tile '" + tile.name + "'");
    }
    const auto started = std::chrono::steady_clock::now();
    const torch::Device device = config.device;
    const torch::Tensor z = grid.to(device, torch::kFloat32).contiguous();
    if (!torch::isfinite(z).all().item<bool>()) {
        throw std::invalid_argument("fit_terrain_tile: tile '" + tile.name + "' holds non-finite elevations");
    }
    torch::manual_seed(static_cast<uint64_t>(config.seed));
    TerrainTileField field(config, tile.rows, tile.cols);
    field->to(device);
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor pooled =
            torch::adaptive_avg_pool2d(z.view({1, 1, tile.rows, tile.cols}), {config.base_cells, config.base_cells});
        field->base_dm.copy_(torch::round(pooled.view({config.base_cells, config.base_cells}) * 10.0f)
                                 .clamp(-32768.0f, 32767.0f)
                                 .to(torch::kInt16));
        const int64_t sample = std::min<int64_t>(z.numel(), 1 << 20);
        const torch::Tensor r = torch::randint(tile.rows, {sample}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        const torch::Tensor c = torch::randint(tile.cols, {sample}, r.options());
        const torch::Tensor detail = z.index({r, c}) - field->base_at(r, c);
        field->scale = std::max(1.0, torch::quantile(detail.abs(), 0.999).item<double>());
    }
    const double scale = field->scale;

    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(field->encoding->parameters(),
                        std::make_unique<torch::optim::AdamWOptions>(
                            torch::optim::AdamWOptions(config.lr).betas({0.9, 0.99}).eps(1e-15).weight_decay(0.0)));
    groups.emplace_back(field->mlp->parameters(),
                        std::make_unique<torch::optim::AdamWOptions>(
                            torch::optim::AdamWOptions(config.lr).betas({0.9, 0.99}).eps(1e-15).weight_decay(1e-6)));
    torch::optim::AdamW opt(std::move(groups));
    const int64_t warmup = std::max<int64_t>(1, config.steps / 20);
    const auto lopts = torch::TensorOptions().dtype(torch::kInt64).device(device);
    field->train();
    for (int64_t step = 0; step < config.steps; ++step) {
        set_learning_rate(opt, warmup_linear_decay_rate(step, config.steps, warmup, config.lr));
        const torch::Tensor r = torch::randint(tile.rows, {config.batch_size}, lopts);
        const torch::Tensor c = torch::randint(tile.cols, {config.batch_size}, lopts);
        const torch::Tensor target = (z.index({r, c}) - field->base_at(r, c)) / static_cast<float>(scale);
        torch::Tensor loss = (field->detail(local_uv(r, c, tile.rows, tile.cols)) - target).abs().mean();
        if (config.gradient_weight > 0.0f) {
            // Central differences of the detail at interior cells, against the
            // grid's (the base's own differences cancel in neither, so the
            // target is the detail's).
            const torch::Tensor ri = r.clamp(1, tile.rows - 2);
            const torch::Tensor ci = c.clamp(1, tile.cols - 2);
            const auto d = [&](const torch::Tensor& rr, const torch::Tensor& cc) {
                return (z.index({rr, cc}) - field->base_at(rr, cc)) / static_cast<float>(scale);
            };
            const auto f = [&](const torch::Tensor& rr, const torch::Tensor& cc) {
                return field->detail(local_uv(rr, cc, tile.rows, tile.cols));
            };
            const torch::Tensor gy = f(ri - 1, ci) - f(ri + 1, ci);
            const torch::Tensor gx = f(ri, ci + 1) - f(ri, ci - 1);
            const torch::Tensor ty = d(ri - 1, ci) - d(ri + 1, ci);
            const torch::Tensor tx = d(ri, ci + 1) - d(ri, ci - 1);
            loss = loss + config.gradient_weight * ((gy - ty).abs().mean() + (gx - tx).abs().mean());
        }
        opt.zero_grad();
        loss.backward();
        opt.step();
    }

    // Evaluate as stored: parameters rounded to float16.
    const auto stored = stored_tensors(field, "");
    std::unordered_map<std::string, torch::Tensor> by_name(stored.begin(), stored.end());
    load_stored(field, [&](const std::string& name) { return by_name.at(name); }, "");
    TerrainTileMetrics metrics = evaluate_tile(field, z, tile);
    metrics.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    json::Value meta = json::Value::object();
    meta.set("format", kTerrainTileFormat);
    meta.set("version", kTileFormatVersion);
    meta.set("tile", terrain_tile_to_json(tile));
    meta.set("config", config_to_json(config));
    meta.set("scale", scale);
    meta.set("metrics", terrain_tile_metrics_to_json(metrics));
    write_tensor_pack(path, meta, stored);
    metrics.bytes = static_cast<int64_t>(std::filesystem::file_size(path));
    meta.set("metrics", terrain_tile_metrics_to_json(metrics));
    write_tensor_pack(path, meta, stored);
    return metrics;
}

void assemble_terrain_field(const std::vector<std::string>& tile_paths, const std::string& path) {
    if (tile_paths.empty()) throw std::invalid_argument("assemble_terrain_field: no tile packs given");
    json::Value tiles = json::Value::array();
    std::vector<std::pair<std::string, torch::Tensor>> tensors;
    std::string config_text;
    json::Value config;
    std::unordered_set<std::string> names;
    for (const auto& tp : tile_paths) {
        const TensorPackReader pack(tp);
        const json::Value& meta = pack.meta();
        if (!meta.contains("format") || meta.at("format").as_string() != kTerrainTileFormat) {
            throw std::runtime_error("assemble_terrain_field: '" + tp + "' is not a terrain tile pack");
        }
        const std::string text = json::dump(meta.at("config"), 0);
        if (config_text.empty()) {
            config_text = text;
            config = meta.at("config");
        } else if (text != config_text) {
            throw std::runtime_error("assemble_terrain_field: '" + tp + "' was fitted under another config");
        }
        const std::string name = meta.at("tile").at("name").as_string();
        if (!names.insert(name).second) {
            throw std::runtime_error("assemble_terrain_field: tile '" + name + "' is given twice");
        }
        json::Value entry = json::Value::object();
        entry.set("tile", meta.at("tile"));
        entry.set("scale", meta.at("scale"));
        entry.set("metrics", meta.at("metrics"));
        tiles.push_back(std::move(entry));
        for (const auto& t : pack.names()) tensors.emplace_back(name + "/" + t, pack.read(t));
    }
    json::Value meta = json::Value::object();
    meta.set("format", kTerrainFieldFormat);
    meta.set("version", kFieldFormatVersion);
    meta.set("config", config);
    meta.set("tiles", std::move(tiles));
    write_tensor_pack(path, meta, tensors);
}

// ---------------------------------------------------------------------------
// TerrainField
// ---------------------------------------------------------------------------

TerrainField::TerrainField(const std::string& path, torch::Device device, int64_t cache_tiles)
    : pack_(path), device_(device), cache_(cache_tiles) {
    const json::Value& meta = pack_.meta();
    if (!meta.contains("format") || meta.at("format").as_string() != kTerrainFieldFormat) {
        throw std::runtime_error("'" + path + "' is not a terrain field");
    }
    if (meta.at("version").as_int() != kFieldFormatVersion) {
        throw std::runtime_error("'" + path + "' is terrain field version " + std::to_string(meta.at("version").as_int()));
    }
    config_ = config_from_json<TerrainFieldConfig>(meta.at("config"));
    config_.device = device_;
    std::vector<TerrainTile> tiles;
    for (const auto& e : meta.at("tiles").items()) {
        tiles.push_back(terrain_tile_from_json(e.at("tile")));
        scales_.push_back(e.at("scale").as_number());
        metrics_.push_back(terrain_tile_metrics_from_json(e.at("metrics")));
    }
    index_ = TerrainTileIndex(std::move(tiles));
}

TerrainTileField TerrainField::tile_field(int64_t t) const {
    return cache_.get_or_make(t, [&] {
        const auto& tile = index_.tiles()[static_cast<size_t>(t)];
        TerrainTileField field(config_, tile.rows, tile.cols);
        field->to(device_);
        field->scale = scales_[static_cast<size_t>(t)];
        load_stored(field, [&](const std::string& name) { return pack_.read(name); }, tile.name + "/");
        field->eval();
        return field;
    });
}

torch::Tensor TerrainField::cell_values(const TerrainCells& cells) const {
    const torch::Tensor tile = cells.tile.to(torch::kCPU, torch::kInt64).reshape({-1}).contiguous();
    const torch::Tensor row = cells.row.to(torch::kCPU, torch::kInt64).reshape({-1});
    const torch::Tensor col = cells.col.to(torch::kCPU, torch::kInt64).reshape({-1});
    const int64_t n = tile.numel();
    torch::Tensor out = torch::full({n}, kNaN, torch::kFloat64);
    if (n == 0) return out;
    torch::NoGradGuard no_grad;
    const auto [sorted, order] = torch::sort(tile, /*stable=*/true, /*dim=*/0, /*descending=*/false);
    const auto unique = torch::_unique2(sorted, /*sorted=*/true, /*return_inverse=*/false, /*return_counts=*/true);
    const torch::Tensor values = std::get<0>(unique);
    const torch::Tensor counts = std::get<2>(unique);
    const auto v = values.accessor<int64_t, 1>();
    const auto k = counts.accessor<int64_t, 1>();
    int64_t start = 0;
    for (int64_t u = 0; u < values.numel(); ++u) {
        const int64_t len = k[u];
        if (v[u] >= 0) {
            const torch::Tensor idx = order.narrow(0, start, len);
            TerrainTileField field = tile_field(v[u]);
            constexpr int64_t kChunk = 1 << 20;
            for (int64_t i = 0; i < len; i += kChunk) {
                const int64_t m = std::min(kChunk, len - i);
                const torch::Tensor part = idx.narrow(0, i, m);
                const torch::Tensor z = field->cells(row.index_select(0, part).to(device_),
                                                     col.index_select(0, part).to(device_));
                out.index_put_({part}, z.to(torch::kCPU, torch::kFloat64));
            }
        }
        start += len;
    }
    return out;
}

torch::Tensor TerrainField::elevation(const torch::Tensor& lat, const torch::Tensor& lon) const {
    return bilinear_elevation(index_, lat, lon, [this](const TerrainCells& cells) { return cell_values(cells); });
}

torch::Tensor TerrainField::grid(int64_t t) const {
    if (t < 0 || t >= index_.size()) throw std::out_of_range("TerrainField::grid: no tile " + std::to_string(t));
    const TerrainTile& tile = index_.tiles()[static_cast<size_t>(t)];
    const torch::Tensor row = torch::arange(tile.rows, torch::kInt64).repeat_interleave(tile.cols);
    const torch::Tensor col = torch::arange(tile.cols, torch::kInt64).repeat({tile.rows});
    const TerrainCells cells{torch::full({tile.rows * tile.cols}, t, torch::kInt64), row, col};
    return cell_values(cells).to(torch::kFloat32).view({tile.rows, tile.cols});
}

std::string TerrainField::describe() const {
    return "terrain field '" + pack_.path() + "' (" + std::to_string(index_.size()) + " tiles)";
}

}  // namespace resolve
