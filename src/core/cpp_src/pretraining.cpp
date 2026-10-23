#include "resolve/pretraining.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/dataset.hpp"
#include "resolve/sparse_rows.hpp"
#include "resolve/utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace resolve {

PlotInputs pretraining_inputs(const ResolveDataset& dataset, const ResolveModel& model) {
    PlotInputs inputs = dataset.plot_inputs();
    inputs.continuous = dataset.continuous_block(model->hash_in_continuous());
    return inputs.without_empty();
}

// =============================================================================
// PretrainRng
// =============================================================================

PretrainRng::PretrainRng(int seed)
    : base_seed_(static_cast<uint64_t>(seed)),
      gen_(at::detail::createCPUGenerator(static_cast<uint64_t>(seed)))
{
}

void PretrainRng::seed_epoch(int epoch) {
    if (!gen_.has_value()) return;  // global stream: nothing of ours to reseed
    gen_->set_current_seed(base_seed_ + static_cast<uint64_t>(epoch) + 1u);
}

torch::Tensor PretrainRng::to_device(torch::Tensor drawn, torch::Device device) {
    return drawn.device() == device ? drawn : drawn.to(device);
}

torch::Tensor PretrainRng::rand(
    at::IntArrayRef size, const torch::TensorOptions& options
) {
    return to_device(
        torch::rand(size, gen_, options.device(torch::kCPU)), options.device());
}

torch::Tensor PretrainRng::rand_like(const torch::Tensor& other) {
    return rand(other.sizes(), other.options());
}

torch::Tensor PretrainRng::beta_symmetric(
    at::IntArrayRef size, double alpha, const torch::TensorOptions& options
) {
    auto cpu = options.device(torch::kCPU);
    auto concentration = torch::full(size, alpha, cpu);
    // _standard_gamma is the only Gamma sampler in ATen that takes a generator.
    auto first = torch::_standard_gamma(concentration, gen_);
    auto second = torch::_standard_gamma(concentration, gen_);
    // Both draws are 0 only when alpha underflows; fall back to an even mix
    // rather than dividing by zero.
    auto total = first + second;
    auto weights = torch::where(total > 0, first / total,
                                torch::full_like(total, 0.5));
    return to_device(weights, options.device());
}

torch::Tensor PretrainRng::randn_like(const torch::Tensor& other) {
    return to_device(
        torch::randn(other.sizes(), gen_, other.options().device(torch::kCPU)),
        other.device());
}

torch::Tensor PretrainRng::randint(
    int64_t low, int64_t high, at::IntArrayRef size,
    const torch::TensorOptions& options
) {
    return to_device(
        torch::randint(low, high, size, gen_, options.device(torch::kCPU)),
        options.device());
}

torch::Tensor PretrainRng::randperm(int64_t n, torch::Device device) {
    return to_device(
        torch::randperm(n, gen_, torch::TensorOptions().dtype(torch::kLong)),
        device);
}

// =============================================================================
// Config validation
// =============================================================================

void PretrainConfig::validate() const {
    if (batch_size < 1) {
        throw std::invalid_argument(
            "PretrainConfig.batch_size must be >= 1 (got " +
            std::to_string(batch_size) + "); a batch_size of 0 divides by zero "
            "when computing the step count.");
    }
    if (!(mask_ratio > 0.0f && mask_ratio < 1.0f)) {
        throw std::invalid_argument(
            "PretrainConfig.mask_ratio must be in (0, 1) (got " +
            std::to_string(mask_ratio) + "); mask_ratio >= 1 makes the Block "
            "strategy's block size span all features and its randint bound <= 0.");
    }
    if (!(corruption_rate >= 0.0f && corruption_rate <= 1.0f)) {
        throw std::invalid_argument(
            "PretrainConfig.corruption_rate must be in [0, 1] (got " +
            std::to_string(corruption_rate) + ").");
    }
    if (!(mixup_alpha >= 0.0f)) {
        throw std::invalid_argument(
            "PretrainConfig.mixup_alpha must be >= 0 (got " +
            std::to_string(mixup_alpha) + "); 0 switches the mixup "
            "augmentation off.");
    }
}

void MLMPretrainConfig::validate() const {
    if (batch_size < 1) {
        throw std::invalid_argument(
            "MLMPretrainConfig.batch_size must be >= 1 (got " +
            std::to_string(batch_size) + ").");
    }
    if (!(mask_prob > 0.0f && mask_prob < 1.0f)) {
        throw std::invalid_argument(
            "MLMPretrainConfig.mask_prob must be in (0, 1) (got " +
            std::to_string(mask_prob) + ").");
    }
}

// =============================================================================
// FeatureMasker
// =============================================================================

FeatureMaskerImpl::FeatureMaskerImpl(
    int64_t n_features,
    float mask_ratio,
    MaskStrategy strategy,
    PretrainRng rng
) : n_features_(n_features),
    mask_ratio_(mask_ratio),
    strategy_(strategy)
{
    // Learnable mask token: one value per feature dimension. Initialized from
    // `rng` rather than torch::nn::init::normal_ (which has no generator seam)
    // so a masker rebuilt inside a seeded pretraining run does not draw from the
    // global RNG stream. NoGradGuard because mask_token_ is a leaf requiring
    // grad, which is what nn::init would have supplied.
    mask_token_ = register_parameter("mask_token",
        torch::zeros({1, n_features}));
    {
        torch::NoGradGuard no_grad;
        mask_token_.normal_(0.0, 0.02, rng.generator());
    }
}

torch::Tensor FeatureMaskerImpl::create_mask(
    int64_t batch_size, PretrainRng rng
) const {
    const auto opts = mask_token_.options();
    const auto index_opts = torch::TensorOptions().dtype(torch::kLong).device(opts.device());

    switch (strategy_) {
        case MaskStrategy::Random:
            return (rng.rand({batch_size, n_features_}, opts) > mask_ratio_)
                .to(torch::kFloat32);

        case MaskStrategy::Block: {
            // One contiguous block of mask_ratio * n_features features per
            // row, starting at a uniform position.
            const int64_t block_size = std::max(static_cast<int64_t>(1),
                static_cast<int64_t>(n_features_ * mask_ratio_));
            const auto starts = rng.randint(0, n_features_ - block_size + 1,
                                            {batch_size, 1}, index_opts);
            const auto column = torch::arange(n_features_, index_opts).unsqueeze(0);
            const auto masked = (column >= starts) & (column < starts + block_size);
            return masked.logical_not().to(opts.dtype());
        }

        case MaskStrategy::Structured: {
            // Whole contiguous feature groups, ~4 features each (at least one
            // group, the last absorbing the remainder), approximating the
            // block's layout of coordinates, species signal and covariates.
            // Each row masks n_mask groups drawn uniformly without
            // replacement: the groups with the n_mask smallest uniform keys.
            constexpr int64_t group_target_size = 4;
            const int64_t n_groups = std::max(int64_t(1), n_features_ / group_target_size);
            const int64_t group_size = n_features_ / n_groups;
            const int64_t n_mask = std::max(int64_t(1),
                static_cast<int64_t>(n_groups * mask_ratio_));
            const auto keys = rng.rand({batch_size, n_groups}, opts);
            const auto chosen = std::get<1>(keys.sort(/*dim=*/1)).slice(/*dim=*/1, 0, n_mask);
            const auto masked_group = torch::zeros({batch_size, n_groups}, opts)
                .scatter_(1, chosen, 1.0f);
            const auto group_of_feature = torch::clamp_max(
                torch::arange(n_features_, index_opts).div(group_size, "floor"),
                n_groups - 1);
            return 1.0f - masked_group.index_select(1, group_of_feature);
        }
    }
    throw std::logic_error("FeatureMasker: unknown mask strategy");
}

torch::Tensor FeatureMaskerImpl::apply_mask(torch::Tensor features, torch::Tensor mask) const {
    // Replace masked positions with learnable mask token
    // mask: 1 = keep original, 0 = replace with mask_token
    auto expanded_token = mask_token_.expand_as(features);
    return features * mask + expanded_token * (1.0f - mask);
}

// =============================================================================
// JEPAPredictor
// =============================================================================

JEPAPredictorImpl::JEPAPredictorImpl(
    int64_t latent_dim,
    int hidden_dim,
    int n_layers,
    float dropout
) {
    torch::nn::Sequential layers;

    int64_t in_dim = latent_dim;
    for (int i = 0; i < n_layers; ++i) {
        layers->push_back(torch::nn::Linear(in_dim, hidden_dim));
        layers->push_back(torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_dim})));
        layers->push_back(torch::nn::GELU());
        if (dropout > 0.0f) {
            layers->push_back(torch::nn::Dropout(dropout));
        }
        in_dim = hidden_dim;
    }

    // Final projection back to latent_dim
    layers->push_back(torch::nn::Linear(hidden_dim, latent_dim));

    mlp_ = register_module("mlp", layers);
}

torch::Tensor JEPAPredictorImpl::forward(torch::Tensor context_repr) {
    return mlp_->forward(context_repr);
}

// =============================================================================
// SCARFCorruptor
// =============================================================================

namespace {

// `values` with each column permuted across the rows independently: column j
// is reordered by the argsort of its own column of uniform keys.
torch::Tensor shuffle_columns(const torch::Tensor& values, PretrainRng& rng) {
    const auto keys = rng.rand(values.sizes(),
                               values.options().dtype(torch::kFloat32));
    return values.gather(0, std::get<1>(keys.sort(/*dim=*/0)));
}

}  // namespace

SCARFCorruptorImpl::SCARFCorruptorImpl(
    int64_t n_features,
    float corruption_rate
) : n_features_(n_features),
    corruption_rate_(corruption_rate)
{
}

torch::Tensor SCARFCorruptorImpl::corrupt(
    torch::Tensor features, PretrainRng rng
) const {
    const auto batch_size = features.size(0);

    // 1 = corrupt, 0 = keep; a corrupted cell takes its column's value in
    // another row of the batch.
    const auto corruption_mask =
        (rng.rand({batch_size, n_features_}, features.options()) < corruption_rate_)
            .to(torch::kFloat32);
    const auto shuffled = shuffle_columns(features, rng);
    return features * (1.0f - corruption_mask) + shuffled * corruption_mask;
}

torch::Tensor SCARFCorruptorImpl::corrupt_categoricals(
    const torch::Tensor& codes, PretrainRng rng
) const {
    if (!codes.defined() || codes.numel() == 0) return codes;
    const auto corrupted =
        rng.rand(codes.sizes(), codes.options().dtype(torch::kFloat32)) < corruption_rate_;
    return torch::where(corrupted, shuffle_columns(codes, rng), codes);
}

// =============================================================================
// ProjectionHead
// =============================================================================

ProjectionHeadImpl::ProjectionHeadImpl(
    int64_t input_dim,
    int64_t projection_dim
) {
    torch::nn::Sequential layers;
    layers->push_back(torch::nn::Linear(input_dim, input_dim));
    layers->push_back(torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({input_dim})));
    layers->push_back(torch::nn::GELU());
    layers->push_back(torch::nn::Linear(input_dim, projection_dim));

    mlp_ = register_module("mlp", layers);
}

torch::Tensor ProjectionHeadImpl::forward(torch::Tensor x) {
    return mlp_->forward(x);
}

// =============================================================================
// Pretext views
// =============================================================================

torch::Tensor mask_codes(const torch::Tensor& codes, float ratio, PretrainRng& rng) {
    if (!codes.defined() || codes.numel() == 0) return codes;
    const auto dropped =
        (codes != 0) &
        (rng.rand(codes.sizes(), codes.options().dtype(torch::kFloat32)) < ratio);
    return codes.masked_fill(dropped, 0);
}

PlotInputs mask_species_view(const PlotInputs& inputs, float ratio, PretrainRng& rng) {
    PlotInputs view = inputs;

    if (inputs.pool_mask.defined() && inputs.pool_mask.numel() > 0) {
        // A pooled species leaves the pool as a whole: one decision per slot,
        // applied to every pool tensor aligned with species_ids.
        const auto valid = inputs.pool_mask.to(torch::kBool);
        const auto dropped =
            valid & (rng.rand(valid.sizes(), torch::TensorOptions()
                                                 .dtype(torch::kFloat32)
                                                 .device(valid.device())) < ratio);
        auto drop_slots = [&](const torch::Tensor& t) {
            return t.defined() && t.numel() > 0 ? t.masked_fill(dropped, 0) : t;
        };
        view.species_ids = drop_slots(inputs.species_ids);
        view.pool_genus_ids = drop_slots(inputs.pool_genus_ids);
        view.pool_family_ids = drop_slots(inputs.pool_family_ids);
        view.pool_weights = drop_slots(inputs.pool_weights);
        view.pool_mask = inputs.pool_mask.masked_fill(dropped, 0);
    } else {
        view.species_ids = mask_codes(inputs.species_ids, ratio, rng);
    }

    // The taxonomy slots have their own count, so each is drawn from its own
    // shape.
    view.genus_ids = mask_codes(inputs.genus_ids, ratio, rng);
    view.family_ids = mask_codes(inputs.family_ids, ratio, rng);

    // An explicit species vector drops each recorded species (each stored
    // entry of its sparse rows) with probability `ratio`.
    if (inputs.species_vector.defined() && inputs.species_vector.numel() > 0) {
        require_sparse_rows(inputs.species_vector, "mask_species_view: species_vector");
        const auto values = inputs.species_vector.values();
        const auto keep = (rng.rand_like(values) >= ratio).to(values.dtype());
        view.species_vector = sparse_rows_with_values(inputs.species_vector, values * keep);
    }
    return view;
}

// =============================================================================
// Shared pretraining loop
// =============================================================================

PretrainResult run_pretrain_loop(
    PretrainLoopSpec spec,
    const PretrainBatchFn& batch_fn,
    const PretrainLoopHooks& hooks
) {
    if (spec.batch_size < 1) {
        throw std::invalid_argument(
            "run_pretrain_loop: batch_size must be >= 1 (got " +
            std::to_string(spec.batch_size) + "); a batch_size of 0 divides by "
            "zero when computing the step count.");
    }
    spec.inputs = spec.inputs.without_empty();
    const int64_t n_samples = spec.inputs.n_plots();
    if (n_samples == 0) {
        throw std::invalid_argument(
            "run_pretrain_loop: the inputs carry no defined, non-empty field "
            "(one supplies the sample count for the shuffle).");
    }
    if (!batch_fn) {
        throw std::invalid_argument(
            "run_pretrain_loop: batch_fn must be callable (it is the objective).");
    }

    const auto start_time = std::chrono::steady_clock::now();
    PretrainResult result;

    for (const auto& module : spec.modules) {
        if (module) module->to(spec.device);
    }
    spec.inputs = spec.inputs.to(spec.device);

    // A loader leaves a missing covariate or coordinate as NaN under
    // MissingValuePolicy::Indicate, and the Trainer fills it from the fitting
    // rows before any forward. A pretext task has no fitted scalers, so an
    // unfilled block would put NaN through the objective and from there into
    // every weight, silently. Fill from the column means of the rows this task
    // was given -- the same fill training computes for the same rows -- and say
    // so, because it is a property of the data the caller handed over.
    auto& continuous = spec.inputs.continuous;
    if (continuous.defined() && torch::isnan(continuous).any().item<bool>()) {
        continuous = fill_missing_continuous(continuous);
        spec.log("Pretraining: the continuous block carries missing cells; "
                 "filled each with its column mean");
    }

    torch::optim::AdamW optimizer(
        spec.params,
        torch::optim::AdamWOptions(spec.lr).weight_decay(spec.weight_decay));

    PretrainRng rng(spec.seed);
    const auto sum_options = torch::TensorOptions().dtype(torch::kFloat32).device(spec.device);

    for (int epoch = 0; epoch < spec.epochs; ++epoch) {
        rng.seed_epoch(epoch);

        for (const auto& module : spec.modules) {
            if (module) module->train();
        }
        if (hooks.on_epoch_begin) hooks.on_epoch_begin(epoch);

        // Summed on the device and read back once per epoch, so a batch adds
        // no host sync.
        auto epoch_loss = torch::zeros({}, sum_options);
        int n_batches = 0;

        auto perm = rng.randperm(n_samples, spec.device);

        for (int64_t start = 0; start < n_samples; start += spec.batch_size) {
            const int64_t end = std::min(
                start + static_cast<int64_t>(spec.batch_size), n_samples);
            const PretrainBatch batch = spec.inputs.index_select(perm.slice(0, start, end));

            auto loss = batch_fn(batch, rng);
            if (!loss.defined()) continue;  // the task opted out of this batch

            optimizer.zero_grad();
            loss.backward();
            if (spec.grad_clip_norm > 0.0f) {
                torch::nn::utils::clip_grad_norm_(spec.params, spec.grad_clip_norm);
            }
            optimizer.step();

            if (hooks.on_step_end) hooks.on_step_end();

            epoch_loss.add_(loss.detach().to(sum_options).reshape({}));
            ++n_batches;
        }

        // Guard an empty (or fully skipped) pretraining set so the history
        // records 0 rather than NaN from a divide-by-zero.
        const float avg_loss = (n_batches > 0)
            ? epoch_loss.item<float>() / static_cast<float>(n_batches) : 0.0f;
        result.loss_history.push_back(avg_loss);

        if (hooks.on_epoch_end) hooks.on_epoch_end(epoch, avg_loss);

        if (spec.log && spec.log_every > 0 && epoch % spec.log_every == 0) {
            std::ostringstream msg;
            msg << "Pretrain epoch " << (epoch + 1) << "/" << spec.epochs
                << " - " << spec.task_name << " loss: " << avg_loss;
            if (hooks.epoch_detail) msg << hooks.epoch_detail(epoch);
            spec.log(msg.str());
        }
    }

    result.epochs_completed = spec.epochs;
    result.total_time_seconds = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - start_time).count();

    return result;
}

namespace {

// `inputs` must carry the continuous block a JEPA / SCARF task masks.
void require_continuous(const PlotInputs& inputs, const char* task) {
    if (!inputs.continuous.defined() || inputs.continuous.dim() != 2) {
        throw std::invalid_argument(
            std::string(task) + " pretraining needs inputs.continuous, the "
            "(n_plots, n_features) continuous block it masks");
    }
}

// The (online, target) tensors of two modules of one architecture, matched by
// name in a single walk over both ordered lists.
std::vector<std::pair<torch::Tensor, torch::Tensor>> paired_by_name(
    const torch::OrderedDict<std::string, torch::Tensor>& online,
    const torch::OrderedDict<std::string, torch::Tensor>& target
) {
    if (online.size() != target.size()) {
        throw std::logic_error("JEPA: the target encoder holds " +
                               std::to_string(target.size()) + " tensors where the "
                               "online encoder holds " + std::to_string(online.size()));
    }
    std::vector<std::pair<torch::Tensor, torch::Tensor>> pairs;
    pairs.reserve(online.size());
    auto t = target.begin();
    for (auto o = online.begin(); o != online.end(); ++o, ++t) {
        if (o->key() != t->key()) {
            throw std::logic_error("JEPA: online tensor '" + o->key() +
                                   "' meets target tensor '" + t->key() + "'");
        }
        pairs.emplace_back(o->value(), t->value());
    }
    return pairs;
}

}  // namespace

// =============================================================================
// JEPAPretrainer
// =============================================================================

JEPAPretrainer::JEPAPretrainer(
    ResolveModel model,
    const PretrainConfig& config
) : context_encoder_(model),
    config_(config)
{
    config_.validate();
    int64_t latent = context_encoder_->latent_dim();

    predictor_ = JEPAPredictor(
        latent,
        config.predictor_hidden_dim,
        config.predictor_n_layers,
        config.predictor_dropout
    );

    // masker_ is built in pretrain() from the actual runtime feature count
    // (inputs.continuous.size(1)).

    // The target encoder is a copy of the online one that follows it by EMA
    // and takes no gradient.
    target_encoder_ = ResolveModel(context_encoder_->schema(), context_encoder_->config());
    ema_parameters_ = paired_by_name(context_encoder_->named_parameters(),
                                     target_encoder_->named_parameters());
    ema_buffers_ = paired_by_name(context_encoder_->named_buffers(),
                                  target_encoder_->named_buffers());
    {
        torch::NoGradGuard no_grad;
        for (auto& [online, target] : ema_parameters_) target.copy_(online);
        for (auto& [online, target] : ema_buffers_) target.copy_(online);
    }
    for (auto& p : target_encoder_->parameters()) {
        p.set_requires_grad(false);
    }
}

void JEPAPretrainer::update_target_encoder(float decay) {
    torch::NoGradGuard no_grad;
    for (auto& [online, target] : ema_parameters_) {
        target.mul_(decay).add_(online, 1.0f - decay);
    }
    // Buffers (BatchNorm running_mean/running_var/num_batches_tracked) take no
    // gradient and are not touched by the parameter EMA. Without copying them
    // the target encoder -- always run in eval() -- would normalize with the
    // construction-time init stats for the entire run, breaking the "target
    // is a slow copy of the online encoder" invariant. BYOL/JEPA copy buffers
    // from the online encoder rather than EMA-ing them.
    for (auto& [online, target] : ema_buffers_) {
        target.copy_(online);
    }
}

float JEPAPretrainer::get_ema_decay(int step, int total_steps) const {
    // Cosine schedule from ema_decay to ema_decay_end
    float progress = static_cast<float>(step) / static_cast<float>(std::max(1, total_steps));
    float cosine = cosine_ramp(progress);
    return config_.ema_decay_end - (config_.ema_decay_end - config_.ema_decay) * cosine;
}

PretrainResult JEPAPretrainer::pretrain(const PlotInputs& inputs) {
    require_continuous(inputs, "JEPA");
    const int64_t n_samples = inputs.continuous.size(0);
    const int64_t n_features = inputs.continuous.size(1);

    // Masker at the runtime feature count. Its mask-token init draws from the
    // pretraining RNG, not the global stream.
    masker_ = FeatureMasker(n_features, config_.mask_ratio, config_.mask_strategy,
                            PretrainRng(config_.seed));

    // The EMA target encoder is frozen and always run in eval(), so it is not
    // one of the loop's trainable modules; move it to the device here.
    target_encoder_->to(config_.device);

    PretrainLoopSpec spec;
    spec.modules = {context_encoder_.ptr(), predictor_.ptr(), masker_.ptr()};
    for (auto& p : context_encoder_->parameters()) spec.params.push_back(p);
    for (auto& p : predictor_->parameters()) spec.params.push_back(p);
    for (auto& p : masker_->parameters()) spec.params.push_back(p);
    spec.inputs = inputs;
    spec.epochs = config_.pretrain_epochs;
    spec.batch_size = config_.batch_size;
    spec.lr = config_.pretrain_lr;
    spec.weight_decay = config_.pretrain_weight_decay;
    spec.device = config_.device;
    spec.seed = config_.seed;
    spec.log = config_.log;
    spec.task_name = "JEPA";

    const int total_steps = config_.pretrain_epochs *
        static_cast<int>((n_samples + config_.batch_size - 1) / config_.batch_size);
    int global_step = 0;

    auto batch_fn = [this](const PretrainBatch& batch, PretrainRng& rng) {
        // The context view: every input masked (see "Pretext views").
        const auto mask = masker_->create_mask(batch.continuous.size(0), rng);
        PlotInputs context = mask_species_view(batch, config_.mask_ratio, rng);
        context.continuous = masker_->apply_mask(batch.continuous, mask);
        context.categorical_ids = mask_codes(batch.categorical_ids, config_.mask_ratio, rng);
        auto context_repr = context_encoder_->get_latent(context);

        // Predict target representation from context
        auto predicted_repr = predictor_->forward(context_repr);

        // Target representation (unmasked) -> target encoder (no grad)
        torch::Tensor target_repr;
        {
            torch::NoGradGuard no_grad;
            target_encoder_->eval();
            target_repr = target_encoder_->get_latent(batch);
        }

        // L2 loss between the normalized predicted and target representations
        // (the cosine-similarity form).
        auto pred_norm = torch::nn::functional::normalize(predicted_repr,
            torch::nn::functional::NormalizeFuncOptions().dim(1));
        auto target_norm = torch::nn::functional::normalize(target_repr,
            torch::nn::functional::NormalizeFuncOptions().dim(1));

        return torch::mse_loss(pred_norm, target_norm);
    };

    PretrainLoopHooks hooks;
    // Update target encoder via EMA, on the cosine schedule over global steps.
    hooks.on_step_end = [this, total_steps, &global_step]() {
        update_target_encoder(get_ema_decay(global_step, total_steps));
        ++global_step;
    };

    return run_pretrain_loop(std::move(spec), batch_fn, hooks);
}

// =============================================================================
// SCARFPretrainer
// =============================================================================

SCARFPretrainer::SCARFPretrainer(
    ResolveModel model,
    const PretrainConfig& config
) : model_(model),
    config_(config)
{
    config_.validate();
    int64_t latent = model_->latent_dim();

    // Corruptor is rebuilt with the runtime feature count in pretrain()
    corruptor_ = SCARFCorruptor(1, config.corruption_rate);

    // Projection head: maps latent -> contrastive space
    projection_head_ = ProjectionHead(latent, config.projection_dim);
}

PretrainResult SCARFPretrainer::pretrain(const PlotInputs& inputs) {
    require_continuous(inputs, "SCARF");
    corruptor_ = SCARFCorruptor(inputs.continuous.size(1), config_.corruption_rate);

    PretrainLoopSpec spec;
    spec.modules = {model_.ptr(), projection_head_.ptr()};
    for (auto& p : model_->parameters()) spec.params.push_back(p);
    for (auto& p : projection_head_->parameters()) spec.params.push_back(p);
    spec.inputs = inputs;
    spec.epochs = config_.pretrain_epochs;
    spec.batch_size = config_.batch_size;
    spec.lr = config_.pretrain_lr;
    spec.weight_decay = config_.pretrain_weight_decay;
    spec.device = config_.device;
    spec.seed = config_.seed;
    spec.log = config_.log;
    spec.task_name = "SCARF";

    auto batch_fn = [this](const PretrainBatch& batch, PretrainRng& rng) {
        // View 1: the plot's own inputs -> encoder -> projection
        auto repr_1 = model_->get_latent(batch);
        auto z_1 = projection_head_->forward(repr_1);

        // View 2: every input corrupted (see "Pretext views"). Feeding both
        // views identical species tensors would make the positive pair
        // matchable by species identity alone, and the encoder would
        // degenerate to passing species through.
        auto corrupted_cont = corruptor_->corrupt(batch.continuous, rng);
        if (config_.mixup_alpha > 0.0f) {
            // Mix each row of the second view with another row of the batch.
            // The weight is drawn from Beta(alpha, alpha) and then folded to
            // [0.5, 1] for the row's own share: a pretext task has no label to
            // mix, so the row has to stay recognizable as the positive pair of
            // view 1.
            const auto rows = corrupted_cont.size(0);
            auto share = rng.beta_symmetric({rows, 1},
                                            static_cast<double>(config_.mixup_alpha),
                                            corrupted_cont.options());
            share = torch::maximum(share, 1.0f - share);
            auto partner = rng.randperm(rows, corrupted_cont.device());
            corrupted_cont = share * corrupted_cont +
                             (1.0f - share) * corrupted_cont.index_select(0, partner);
        }
        PlotInputs corrupted = mask_species_view(batch, config_.corruption_rate, rng);
        corrupted.continuous = corrupted_cont;
        corrupted.categorical_ids = corruptor_->corrupt_categoricals(batch.categorical_ids, rng);
        auto repr_2 = model_->get_latent(corrupted);
        auto z_2 = projection_head_->forward(repr_2);

        // Normalize projections
        z_1 = torch::nn::functional::normalize(z_1,
            torch::nn::functional::NormalizeFuncOptions().dim(1));
        z_2 = torch::nn::functional::normalize(z_2,
            torch::nn::functional::NormalizeFuncOptions().dim(1));

        // InfoNCE loss
        // Similarity matrix: (batch, batch)
        auto sim = torch::mm(z_1, z_2.t()) / config_.temperature;

        // Labels: diagonal (positive pairs)
        auto labels = torch::arange(batch.continuous.size(0),
            torch::TensorOptions().dtype(torch::kLong).device(batch.continuous.device()));

        // Symmetric loss: both directions
        auto loss_12 = torch::nn::functional::cross_entropy(sim, labels);
        auto loss_21 = torch::nn::functional::cross_entropy(sim.t(), labels);
        return (loss_12 + loss_21) * 0.5f;
    };

    return run_pretrain_loop(std::move(spec), batch_fn);
}

// =============================================================================
// Masked Species Pretraining (BERT-style MLM)
// =============================================================================

MaskedSpeciesHeadImpl::MaskedSpeciesHeadImpl(int64_t d_model, int64_t n_species) {
    proj_ = register_module("proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, n_species).bias(false)));
}

torch::Tensor MaskedSpeciesHeadImpl::forward(torch::Tensor token_embeddings) {
    return proj_->forward(token_embeddings);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mask_species_batch(
    torch::Tensor species_ids,
    torch::Tensor valid_mask,
    int64_t n_species,
    float mask_prob,
    PretrainRng rng
) {
    auto device = species_ids.device();
    auto masked_ids = species_ids.clone();

    // Select positions to mask: valid positions with probability mask_prob
    auto rand_vals =
        rng.rand(valid_mask.sizes(), valid_mask.options().dtype(torch::kFloat32));
    auto mlm_mask = valid_mask & (rand_vals < mask_prob);

    // Save targets before masking
    auto mlm_targets = species_ids.index({mlm_mask}).clone();

    // The 80% subset whose token embedding the encoder swaps for the learned
    // mask embedding. Kept separate from mlm_mask so the 10%-random and
    // 10%-keep ids in masked_ids survive to the encoder (passing the full
    // mlm_mask here would overwrite every masked position with the mask
    // embedding, nullifying the 80/10/10 split entirely).
    auto mask_token_positions = torch::zeros_like(mlm_mask);

    // BERT-style 80/10/10 split, vectorized: one action value per masked slot,
    // scattered back onto the (B, S) grid in mlm_mask's row-major order and
    // applied with tensor ops, so the split adds no host sync.
    auto n_masked = mlm_mask.sum().item<int64_t>();  // single sync (empty-case guard)
    if (n_masked > 0) {
        auto action_rand = rng.rand({n_masked}, torch::TensorOptions().device(device));

        // 80% -> mask token; next 10% -> random species id; last 10% -> keep.
        auto mask_replace = action_rand < 0.8f;                          // (n_masked,)
        auto mask_random = (action_rand >= 0.8f) & (action_rand < 0.9f); // (n_masked,)

        // index_put_ with a boolean mask writes the 1-D values in the row-major
        // order of the mask's true positions — the same order action_rand was
        // drawn in — so these land on exactly the intended grid cells.
        auto replace_grid = torch::zeros_like(mlm_mask);
        auto random_grid = torch::zeros_like(mlm_mask);
        replace_grid.index_put_({mlm_mask}, mask_replace);
        random_grid.index_put_({mlm_mask}, mask_random);

        mask_token_positions = replace_grid;
        masked_ids = masked_ids.masked_fill(replace_grid, 0);

        // Random-id replacement: draw a full-grid random-id tensor and select it
        // only at the random positions. Needs at least two species ids (1..n-1).
        if (n_species > 1) {
            auto rand_ids = rng.randint(
                1, n_species, mlm_mask.sizes(),
                torch::TensorOptions().dtype(torch::kInt64).device(device));
            masked_ids = torch::where(random_grid, rand_ids, masked_ids);
        }
    }

    return {masked_ids, mlm_mask, mlm_targets, mask_token_positions};
}

MaskedSpeciesPretrainer::MaskedSpeciesPretrainer(
    PlotEncoderTransformer encoder,
    int64_t n_species,
    const MLMPretrainConfig& config
) : encoder_(std::move(encoder)),
    config_(config),
    n_species_(n_species)
{
    config_.validate();
    mlm_head_ = MaskedSpeciesHead(encoder_->d_model(), n_species);
}

MaskedSpeciesPretrainer::MaskedSpeciesPretrainer(
    ResolveModel model,
    const MLMPretrainConfig& config
) : MaskedSpeciesPretrainer(model->transformer_encoder(),
                            model->schema().n_species_vocab, config)
{
}

PretrainResult MaskedSpeciesPretrainer::pretrain(const PlotInputs& inputs) {
    const auto present = [](const torch::Tensor& t) { return t.defined() && t.numel() > 0; };
    if (!present(inputs.species_ids) || !present(inputs.pool_mask)) {
        throw std::invalid_argument(
            "masked-species pretraining needs inputs.species_ids and "
            "inputs.pool_mask, the plot's species tokens and which of them are real");
    }

    PretrainLoopSpec spec;
    spec.modules = {encoder_.ptr(), mlm_head_.ptr()};
    for (auto& p : encoder_->parameters()) spec.params.push_back(p);
    for (auto& p : mlm_head_->parameters()) spec.params.push_back(p);
    spec.inputs = {.species_ids = inputs.species_ids,
                   .pool_genus_ids = inputs.pool_genus_ids,
                   .pool_family_ids = inputs.pool_family_ids,
                   .pool_weights = inputs.pool_weights,
                   .pool_mask = inputs.pool_mask.to(torch::kBool)};
    spec.epochs = config_.pretrain_epochs;
    spec.batch_size = config_.batch_size;
    spec.lr = config_.pretrain_lr;
    spec.weight_decay = config_.pretrain_weight_decay;
    // The MLM objective is not gradient-clipped.
    spec.grad_clip_norm = 0.0f;
    spec.device = config_.device;
    spec.seed = config_.seed;
    spec.log = config_.log;
    spec.task_name = "MLM";
    spec.log_every = 1;

    auto batch_fn = [this](const PretrainBatch& batch,
                           PretrainRng& rng) -> torch::Tensor {
        auto [masked_ids, mlm_mask, mlm_targets, mask_token_positions] =
            mask_species_batch(batch.species_ids, batch.pool_mask, n_species_,
                               config_.mask_prob, rng);

        // Nothing was selected for masking in this batch, so there is no target
        // to learn from; an undefined loss tells the loop to move on.
        if (mlm_targets.numel() == 0) return {};

        // Pre-pooling token embeddings. Only the 80% mask-token subset is
        // replaced with the mask embedding; the 10%-random and 10%-keep ids
        // reach the encoder through masked_ids.
        auto tokens = encoder_->forward_tokens(
            masked_ids, batch.pool_genus_ids, batch.pool_family_ids, batch.pool_weights,
            batch.pool_mask, mask_token_positions);

        // Extract masked positions and project to species logits
        auto masked_tokens = tokens.index({mlm_mask});  // (N_masked, d_model)
        auto logits = mlm_head_->forward(masked_tokens);  // (N_masked, n_species)

        // Cross-entropy loss (ignore padding index 0)
        return torch::nn::functional::cross_entropy(
            logits, mlm_targets,
            torch::nn::functional::CrossEntropyFuncOptions().ignore_index(0));
    };

    return run_pretrain_loop(std::move(spec), batch_fn);
}

} // namespace resolve
