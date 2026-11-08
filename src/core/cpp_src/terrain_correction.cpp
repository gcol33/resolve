#include "resolve/terrain_correction.hpp"

#include "resolve/config_json.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/lr_schedule.hpp"
#include "resolve/tensor_pack.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace resolve {

namespace {

constexpr const char* kFormat = "resolve-terrain-correction";
constexpr int kFormatVersion = 1;
constexpr int64_t kExperts = 3;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

torch::Tensor column(const torch::Tensor& t, int64_t n) {
    if (!t.defined()) return torch::full({n}, kNaN, torch::kFloat64);
    return t.to(torch::kCPU, torch::kFloat64).reshape({-1});
}

double finite_median(const torch::Tensor& t) {
    const torch::Tensor v = t.index({torch::isfinite(t)});
    return v.numel() == 0 ? 0.0 : v.median().item<double>();
}

// Values the network reads for an input a caller does not give: the training
// plots' medians.
struct Fills {
    double spread = 0.0;
    double recognised_share = 1.0;
    double n_species = 10.0;
};

struct Prepared {
    torch::Tensor X;        // (n, kFeatures) float32
    torch::Tensor experts;  // (n, kExperts) float32, scaled
    torch::Tensor applied;  // (n,) bool
};

// The network's features and experts. `hide_1km` (optional, (n,) bool) hides
// the 1 km descriptor of those plots, as a caller without it would.
Prepared prepare(const TerrainCorrectionInputs& in, double scale, const Fills& fills,
                 const torch::Tensor& hide_1km = {}) {
    const int64_t n = in.size();
    const torch::Tensor pred = column(in.prediction, n);
    const torch::Tensor dem = column(in.dem, n);
    torch::Tensor has1 = torch::isfinite(column(in.dem_1km, n));
    if (hide_1km.defined()) has1 = has1 & hide_1km.logical_not();
    const torch::Tensor dem1 = torch::where(has1, column(in.dem_1km, n), dem);
    const torch::Tensor u = column(in.uncertainty_m, n);
    const torch::Tensor has_u = torch::isfinite(u);
    const torch::Tensor lu = torch::where(has_u, torch::log10(u.clamp_min(0.0) + 1.0), torch::zeros_like(u));
    const auto filled = [](const torch::Tensor& t, double fill) {
        return torch::where(torch::isfinite(t), t, torch::full_like(t, fill));
    };
    const torch::Tensor spread = filled(column(in.spread, n), fills.spread).clamp_min(0.0);
    const torch::Tensor rec = filled(column(in.recognised_share, n), fills.recognised_share);
    const torch::Tensor ns = filled(column(in.n_species, n), fills.n_species).clamp_min(0.0);
    const torch::Tensor applied = torch::isfinite(pred) & torch::isfinite(dem);
    const torch::Tensor p = torch::where(applied, pred, torch::zeros_like(pred)) / scale;
    const torch::Tensor d = torch::where(applied, dem, torch::zeros_like(dem)) / scale;
    const torch::Tensor d1 = torch::where(applied, dem1, torch::zeros_like(dem1)) / scale;
    torch::Tensor X = torch::stack({p, d, d1, d - p, d1 - p, lu, has_u.to(torch::kFloat64),
                                    torch::log1p(spread / scale * 100.0), rec, torch::log1p(ns),
                                    has1.to(torch::kFloat64)},
                                   1);
    torch::Tensor E = torch::stack({p, d, d1}, 1);
    return {X.to(torch::kFloat32), E.to(torch::kFloat32), applied};
}

torch::nn::Sequential make_net(const TerrainCorrectionConfig& c) {
    torch::nn::Sequential net;
    int64_t width = TerrainCorrection::kFeatures;
    for (int64_t l = 0; l < c.n_layers; ++l) {
        net->push_back(torch::nn::Linear(width, c.hidden_dim));
        net->push_back(torch::nn::SiLU());
        width = c.hidden_dim;
    }
    net->push_back(torch::nn::Linear(width, kExperts + 1));
    return net;
}

// The gated output in scaled units, the expert weights and the offset in
// scaled units.
struct Gated {
    torch::Tensor y;
    torch::Tensor weights;
    torch::Tensor offset;
};

Gated gated(torch::nn::Sequential net, const torch::Tensor& X, const torch::Tensor& E,
            double offset_fraction) {
    const torch::Tensor o = net->forward(X);
    const torch::Tensor w = torch::softmax(o.narrow(1, 0, kExperts), 1);
    const torch::Tensor offset = offset_fraction * torch::tanh(o.select(1, kExperts));
    return {(w * E).sum(1) + offset, w, offset};
}

Fills fills_from(const json::Value& meta) {
    const json::Value& f = meta.at("fills");
    return {f.at("spread").as_number(), f.at("recognised_share").as_number(), f.at("n_species").as_number()};
}

json::Value fills_to_json(const Fills& f) {
    json::Value v = json::Value::object();
    v.set("spread", f.spread);
    v.set("recognised_share", f.recognised_share);
    v.set("n_species", f.n_species);
    return v;
}

}  // namespace

void TerrainCorrectionConfig::validate() const {
    if (!(scale > 0.0f) || !std::isfinite(scale)) throw std::invalid_argument("terrain correction: scale must be > 0");
    if (!(offset_fraction >= 0.0f)) throw std::invalid_argument("terrain correction: offset_fraction must be >= 0");
    if (hidden_dim < 1) throw std::invalid_argument("terrain correction: hidden_dim must be >= 1");
    if (n_layers < 1) throw std::invalid_argument("terrain correction: n_layers must be >= 1");
    if (epochs < 1) throw std::invalid_argument("terrain correction: epochs must be >= 1");
    if (batch_size < 1) throw std::invalid_argument("terrain correction: batch_size must be >= 1");
    if (!(lr > 0.0f)) throw std::invalid_argument("terrain correction: lr must be > 0");
    if (!(weight_decay >= 0.0f)) throw std::invalid_argument("terrain correction: weight_decay must be >= 0");
    if (!(dem_1km_dropout >= 0.0f && dem_1km_dropout <= 1.0f)) {
        throw std::invalid_argument("terrain correction: dem_1km_dropout must lie in [0, 1]");
    }
}

int64_t TerrainCorrectionInputs::size() const {
    if (!prediction.defined() || !dem.defined()) {
        throw std::invalid_argument("terrain correction: prediction and dem are required");
    }
    const int64_t n = prediction.numel();
    for (const torch::Tensor* t : {&dem, &dem_1km, &uncertainty_m, &spread, &recognised_share, &n_species}) {
        if (t->defined() && t->numel() != n) {
            throw std::invalid_argument("terrain correction: every input must have one value per plot (" +
                                        std::to_string(n) + ")");
        }
    }
    return n;
}

const std::vector<std::string>& terrain_weight_names() {
    static const std::vector<std::string> names = {"resolve", "dem", "dem_1km"};
    return names;
}

json::Value terrain_correction_report_to_json(const TerrainCorrectionReport& r) {
    json::Value v = json::Value::object();
    v.set("target", r.target);
    v.set("descriptor", enum_to_name(r.descriptor));
    v.set("n_train", r.n_train);
    v.set("mae_raw", r.mae_raw);
    v.set("mae_dem", r.mae_dem);
    v.set("mae_corrected", r.mae_corrected);
    v.set("loss_history", json::Value::array_of(r.loss_history));
    return v;
}

TerrainCorrection::TerrainCorrection(std::string target, TerrainDescriptor descriptor,
                                     TerrainCorrectionConfig config)
    : target_(std::move(target)), descriptor_(descriptor), config_(std::move(config)) {
    config_.validate();
    net_ = make_net(config_);
    net_->to(config_.device);
    report_.target = target_;
    report_.descriptor = descriptor_;
}

TerrainCorrectionResult TerrainCorrection::apply(const TerrainCorrectionInputs& inputs) const {
    const int64_t n = inputs.size();
    const double scale = config_.scale;
    Fills fills;
    fills.spread = fill_spread_;
    fills.recognised_share = fill_recognised_share_;
    fills.n_species = fill_n_species_;
    const Prepared p = prepare(inputs, scale, fills);
    const torch::Tensor raw = column(inputs.prediction, n);
    TerrainCorrectionResult out{raw.clone(), torch::full({n, kExperts}, kNaN, torch::kFloat64),
                                torch::full({n}, kNaN, torch::kFloat64), p.applied};
    if (n == 0) return out;
    torch::NoGradGuard no_grad;
    net_->eval();
    const torch::Device device = config_.device;
    constexpr int64_t kChunk = 1 << 16;
    for (int64_t i = 0; i < n; i += kChunk) {
        const int64_t m = std::min(kChunk, n - i);
        const torch::Tensor X = p.X.narrow(0, i, m).to(device);
        const torch::Tensor E = p.experts.narrow(0, i, m).to(device);
        const Gated g = gated(net_, X, E, config_.offset_fraction);
        const torch::Tensor sel = p.applied.narrow(0, i, m);
        const torch::Tensor yc = g.y.to(torch::kCPU, torch::kFloat64) * scale;
        const torch::Tensor wc = g.weights.to(torch::kCPU, torch::kFloat64);
        const torch::Tensor oc = g.offset.to(torch::kCPU, torch::kFloat64) * scale;
        auto rows = out.corrected.narrow(0, i, m);
        rows.copy_(torch::where(sel, yc, rows));
        out.weights.narrow(0, i, m).copy_(torch::where(sel.unsqueeze(1), wc, out.weights.narrow(0, i, m)));
        out.offset.narrow(0, i, m).copy_(torch::where(sel, oc, out.offset.narrow(0, i, m)));
    }
    return out;
}

TerrainCorrection fit_terrain_correction(const std::string& target, TerrainDescriptor descriptor,
                                         const TerrainCorrectionInputs& inputs, const torch::Tensor& truth,
                                         const TerrainCorrectionConfig& config) {
    config.validate();
    const int64_t n = inputs.size();
    if (truth.numel() != n) {
        throw std::invalid_argument("fit_terrain_correction: truth must have one value per plot (" +
                                    std::to_string(n) + ")");
    }
    const torch::Tensor y_all = column(truth, n);
    const torch::Tensor raw = column(inputs.prediction, n);
    const torch::Tensor dem = column(inputs.dem, n);
    const torch::Tensor usable = torch::isfinite(y_all) & torch::isfinite(raw) & torch::isfinite(dem);
    const torch::Tensor rows = torch::nonzero(usable).reshape({-1});
    const int64_t m = rows.numel();
    if (m < 2) throw std::invalid_argument("fit_terrain_correction: fewer than two plots with a recorded value, "
                                           "a prediction and a descriptor");

    TerrainCorrection model(target, descriptor, config);
    model.fill_spread_ = finite_median(column(inputs.spread, n).index({rows}));
    model.fill_recognised_share_ = inputs.recognised_share.defined()
                                       ? finite_median(column(inputs.recognised_share, n).index({rows}))
                                       : 1.0;
    model.fill_n_species_ = inputs.n_species.defined()
                                ? finite_median(column(inputs.n_species, n).index({rows}))
                                : 10.0;
    const Fills fills{model.fill_spread_, model.fill_recognised_share_, model.fill_n_species_};

    const auto take = [&](const torch::Tensor& t) { return t.defined() ? column(t, n).index({rows}) : t; };
    TerrainCorrectionInputs train{take(inputs.prediction), take(inputs.dem), take(inputs.dem_1km),
                                  take(inputs.uncertainty_m), take(inputs.spread),
                                  take(inputs.recognised_share), take(inputs.n_species)};
    const double scale = config.scale;
    const torch::Device device = config.device;
    const torch::Tensor y = (y_all.index({rows}) / scale).to(torch::kFloat32).to(device);

    torch::manual_seed(static_cast<uint64_t>(config.seed));
    model.net_ = make_net(config);
    model.net_->to(device);
    torch::optim::AdamW opt(model.net_->parameters(),
                            torch::optim::AdamWOptions(config.lr).weight_decay(config.weight_decay));
    auto gen = at::make_generator<at::CPUGeneratorImpl>(static_cast<uint64_t>(config.seed));
    const int64_t steps_per_epoch = (m + config.batch_size - 1) / config.batch_size;
    const int64_t total_steps = steps_per_epoch * config.epochs;
    const int64_t warmup = std::max<int64_t>(1, total_steps / 10);
    int64_t step = 0;
    model.net_->train();
    for (int64_t epoch = 0; epoch < config.epochs; ++epoch) {
        const torch::Tensor hide = config.dem_1km_dropout > 0.0f
                                       ? torch::rand({m}, gen, torch::kFloat64) < config.dem_1km_dropout
                                       : torch::Tensor();
        const Prepared p = prepare(train, scale, fills, hide);
        const torch::Tensor X = p.X.to(device);
        const torch::Tensor E = p.experts.to(device);
        const torch::Tensor perm = torch::randperm(m, gen, torch::kInt64).to(device);
        double epoch_loss = 0.0;
        for (int64_t i = 0; i < m; i += config.batch_size, ++step) {
            set_learning_rate(opt, warmup_linear_decay_rate(step, total_steps, warmup, config.lr));
            const torch::Tensor idx = perm.narrow(0, i, std::min(config.batch_size, m - i));
            const Gated g = gated(model.net_, X.index_select(0, idx), E.index_select(0, idx),
                                  config.offset_fraction);
            const torch::Tensor loss = (g.y - y.index_select(0, idx)).abs().mean();
            opt.zero_grad();
            loss.backward();
            opt.step();
            epoch_loss += loss.item<double>() * static_cast<double>(idx.numel());
        }
        model.report_.loss_history.push_back(epoch_loss / static_cast<double>(m));
    }
    model.net_->eval();

    const TerrainCorrectionResult fitted = model.apply(train);
    const torch::Tensor y_cpu = y_all.index({rows});
    model.report_.n_train = m;
    model.report_.mae_raw = (train.prediction.to(torch::kFloat64) - y_cpu).abs().mean().item<double>();
    model.report_.mae_dem = (train.dem.to(torch::kFloat64) - y_cpu).abs().mean().item<double>();
    model.report_.mae_corrected = (fitted.corrected - y_cpu).abs().mean().item<double>();
    return model;
}

void TerrainCorrection::save(const std::string& path) const {
    json::Value meta = json::Value::object();
    meta.set("format", kFormat);
    meta.set("version", kFormatVersion);
    meta.set("target", target_);
    meta.set("descriptor", enum_to_name(descriptor_));
    meta.set("config", config_to_json(config_));
    meta.set("fills", fills_to_json({fill_spread_, fill_recognised_share_, fill_n_species_}));
    meta.set("report", terrain_correction_report_to_json(report_));
    std::vector<std::pair<std::string, torch::Tensor>> tensors;
    for (const auto& p : net_->named_parameters()) {
        tensors.emplace_back(p.key(), p.value().detach().to(torch::kCPU, torch::kFloat32));
    }
    write_tensor_pack(path, meta, tensors);
}

TerrainCorrection TerrainCorrection::load(const std::string& path, torch::Device device) {
    const TensorPackReader pack(path);
    const json::Value& meta = pack.meta();
    if (!meta.contains("format") || meta.at("format").as_string() != kFormat) {
        throw std::runtime_error("'" + path + "' is not a terrain correction file");
    }
    if (meta.at("version").as_int() != kFormatVersion) {
        throw std::runtime_error("'" + path + "' is terrain correction version " +
                                 std::to_string(meta.at("version").as_int()));
    }
    TerrainCorrectionConfig config = config_from_json<TerrainCorrectionConfig>(meta.at("config"));
    config.device = device;
    TerrainCorrection model(meta.at("target").as_string(),
                            enum_from_name<TerrainDescriptor>(meta.at("descriptor").as_string()), config);
    const Fills fills = fills_from(meta);
    model.fill_spread_ = fills.spread;
    model.fill_recognised_share_ = fills.recognised_share;
    model.fill_n_species_ = fills.n_species;
    const json::Value& r = meta.at("report");
    model.report_.n_train = r.at("n_train").as_int();
    model.report_.mae_raw = r.at("mae_raw").as_number();
    model.report_.mae_dem = r.at("mae_dem").as_number();
    model.report_.mae_corrected = r.at("mae_corrected").as_number();
    for (const auto& l : r.at("loss_history").items()) model.report_.loss_history.push_back(l.as_number());
    torch::NoGradGuard no_grad;
    for (auto& p : model.net_->named_parameters()) {
        const torch::Tensor t = pack.read(p.key());
        if (t.sizes() != p.value().sizes()) {
            throw std::runtime_error("'" + path + "': parameter '" + p.key() + "' has the wrong shape");
        }
        p.value().copy_(t.to(device, p.value().scalar_type()));
    }
    model.net_->eval();
    return model;
}

}  // namespace resolve
