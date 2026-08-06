#include "resolve/predictor.hpp"
#include "resolve/dataset.hpp"
#include "resolve/encoder.hpp"  // Fp32NormImpl (inference-time BN fusion)
#include "resolve/utils.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace resolve {

namespace {

// Slice a tensor along dim 0 in [start, end). Returns an empty/undefined
// tensor if the input is itself undefined (mirrors the "defined or empty"
// contract every downstream predict() call already handles).
inline torch::Tensor slice0(const torch::Tensor& t, int64_t start, int64_t end) {
    // Only pass undefined / scalar tensors through unchanged. A defined but
    // 0-numel matrix (e.g. (n, 0)) must still be sliced along dim 0, otherwise
    // the whole n-row tensor comes back for a [start,end) chunk.
    if (!t.defined() || t.dim() == 0) return t;
    return t.slice(/*dim=*/0, start, end);
}

// True when the encoder looks species IDs up in an embedding table, i.e. when a
// species-code mismatch silently returns another species' row. Hash mode
// derives its features from the species STRING (content-derived, so the vocab
// codes never reach the model) unless it was built with an explicit species
// vector, which is indexed by code like sparse mode.
bool indexes_species_codes(const ResolveModel& model) {
    switch (model->species_encoding()) {
        case SpeciesEncodingMode::Embed:
        case SpeciesEncodingMode::Sparse:
        case SpeciesEncodingMode::RankPool:
        case SpeciesEncodingMode::Transformer:
            return true;
        case SpeciesEncodingMode::Hash:
            return model->uses_explicit_vector();
    }
    return false;
}

// Shared body of every vocabulary check: sizes first (cheap, and the only
// comparison a pre-issue-#102 checkpoint supports), then the ordered name lists
// when both sides carry them. `what` names the vocabulary in the error.
void require_matching_vocab(
    const char* what,
    int64_t model_size,
    int64_t dataset_size,
    const std::vector<std::string>& model_names,
    const std::vector<std::string>& dataset_names
) {
    auto fail = [&](const std::string& detail) {
        std::ostringstream msg;
        msg << "Predictor::predict: the dataset's " << what << " vocabulary is not "
               "the one the model was trained with (" << detail << "). Its integer "
               "codes therefore index the wrong embedding rows, which would "
               "produce wrong predictions with no error. Build the dataset in the "
               "model's namespace: ResolveDataset::from_csv_with_vocabs(header, "
               "species, roles, targets, predictor.external_vocabs(), "
               "dataset_config_from_checkpoint(predictor.schema(), "
               "predictor.model()->config())). See gcol33/resolve#102.";
        throw std::runtime_error(msg.str());
    };

    if (model_size != dataset_size) {
        fail("model has " + std::to_string(model_size) + " entries, dataset has " +
             std::to_string(dataset_size));
    }
    if (model_names.empty() || dataset_names.empty()) {
        return;  // pre-#102 checkpoint (or an unfit vocab): size check is all there is
    }
    if (model_names.size() != dataset_names.size()) {
        fail("model vocabulary lists " + std::to_string(model_names.size()) +
             " names, dataset lists " + std::to_string(dataset_names.size()));
    }
    for (size_t i = 0; i < model_names.size(); ++i) {
        if (model_names[i] != dataset_names[i]) {
            fail("code " + std::to_string(i) + " is '" + model_names[i] +
                 "' in the model but '" + dataset_names[i] + "' in the dataset");
        }
    }
}

}  // namespace

Predictor::Predictor(
    ResolveModel model,
    Scalers scalers,
    torch::Device device
) : model_(model), scalers_(scalers), device_(device)
{
    model_->to(device_);
    model_->eval();
}

Predictor::Predictor(
    ResolveModel model,
    Scalers scalers,
    CategoricalVocab categorical_vocab,
    torch::Device device
) : model_(model),
    scalers_(scalers),
    categorical_vocab_(std::move(categorical_vocab)),
    device_(device)
{
    model_->to(device_);
    model_->eval();
}

Predictor Predictor::load(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    auto [model, scalers, vocab] = Trainer::load(path, device, vram_fraction);
    Predictor predictor(std::move(model), std::move(scalers), std::move(vocab), device);

    // A checkpoint written before gcol33/resolve#102 recorded only the
    // vocabulary SIZES, so nothing can rebuild an inference dataset in this
    // model's ID namespace and predict() can only compare sizes. Say so once,
    // loudly: the failure it guards against is silent wrong predictions.
    if (indexes_species_codes(predictor.model()) &&
        !predictor.schema().has_species_vocab()) {
        std::cerr << "[resolve] warning: this checkpoint does not carry the fitted "
                     "species/taxonomy vocabularies (written before "
                     "gcol33/resolve#102). Its encoder indexes species embeddings "
                     "by an integer code that depends on the file the vocabulary "
                     "was fitted on, so scoring new data can silently look up the "
                     "wrong embedding rows. Only the vocabulary SIZES can be "
                     "checked. Retrain or re-save with a current build, or build "
                     "the inference dataset with "
                     "ResolveDataset::from_csv_with_schema(..., training_dataset, "
                     "...) against the original training data.\n";
    }
    return predictor;
}

ExternalVocabs Predictor::external_vocabs() const {
    // The schema carries species + taxonomy; the categorical string -> code
    // maps live on the Predictor (Trainer::save writes them under their own
    // archive block), so fold them in here to get the complete carrier.
    ExternalVocabs vocabs = external_vocabs_from_schema(schema());
    vocabs.categorical = categorical_vocab_;
    return vocabs;
}

void Predictor::validate_dataset_vocabs(const ResolveDataset& dataset) const {
    const ResolveSchema& model_schema = schema();
    const ResolveSchema& data_schema = dataset.schema();

    if (indexes_species_codes(model_)) {
        require_matching_vocab("species",
                               model_schema.n_species_vocab,
                               data_schema.n_species_vocab,
                               model_schema.species_vocab,
                               data_schema.species_vocab);
    }

    // Taxonomy IDs index genus/family embedding tables in every encoder that
    // has taxonomy, hash mode included (its fixed genus/family slots are
    // embedding lookups even though its species features are not).
    if (model_schema.has_taxonomy) {
        require_matching_vocab("genus",
                               model_schema.n_genera_vocab,
                               data_schema.n_genera_vocab,
                               model_schema.genus_vocab,
                               data_schema.genus_vocab);
        require_matching_vocab("family",
                               model_schema.n_families_vocab,
                               data_schema.n_families_vocab,
                               model_schema.family_vocab,
                               data_schema.family_vocab);
    }

    // Categorical covariates: the dataset's own per-column maps must be the
    // ones training fitted, because predict() forwards dataset.categorical_ids()
    // straight to the embedder. The dataset retains no raw strings, so a
    // mismatch cannot be repaired here -- only detected.
    if (!categorical_vocab_.column_names().empty()) {
        const auto& model_cols = categorical_vocab_.column_names();
        const auto& data_cols = dataset.categorical_vocab().column_names();
        if (model_cols != data_cols) {
            std::ostringstream msg;
            msg << "Predictor::predict: the dataset's categorical columns ("
                << data_cols.size() << ") do not match the model's ("
                << model_cols.size() << ") in name or order. Build the dataset "
                   "with ResolveDataset::from_csv_with_vocabs(..., "
                   "predictor.external_vocabs(), ...). See gcol33/resolve#102.";
            throw std::runtime_error(msg.str());
        }
        for (const auto& col : model_cols) {
            if (categorical_vocab_.column_map(col) !=
                dataset.categorical_vocab().column_map(col)) {
                std::ostringstream msg;
                msg << "Predictor::predict: the dataset's categorical column '"
                    << col << "' was factorized against its own values, so its "
                       "codes mean something different from the model's. Build "
                       "the dataset with ResolveDataset::from_csv_with_vocabs("
                       "..., predictor.external_vocabs(), ...). See "
                       "gcol33/resolve#102.";
                throw std::runtime_error(msg.str());
            }
        }
    }
}

ResolvePredictions Predictor::predict(
    const ResolveDataset& dataset,
    bool return_latent,
    int64_t batch_size
) {
    // Reject a dataset whose integer codes are not the model's BEFORE any
    // forward pass: every non-hash encoder would otherwise index the wrong
    // embedding rows and return plausible, wrong numbers (issue #102).
    validate_dataset_vocabs(dataset);

    const int64_t n = dataset.n_plots();

    // Validate batch_size: -1 (one-shot) or strictly positive. 0 / <-1 reject.
    if (batch_size != -1 && batch_size <= 0) {
        throw std::invalid_argument(
            "Predictor::predict: batch_size must be -1 (whole dataset) "
            "or a positive integer; got " + std::to_string(batch_size));
    }

    // SAINT (inter-sample attention) and the coordinate-kNN GNN build each
    // plot's representation from the OTHER plots in the same forward batch, so
    // a chunked forward would make a plot's prediction depend on its chunk-mates
    // and break the one-shot == chunked equivalence that holds for every other
    // encoder. Force a single forward for them rather than silently returning
    // chunk-composition-dependent predictions.
    const EncoderArchitecture arch = model_->config().encoder_architecture;
    const bool batch_dependent_encoder =
        (arch == EncoderArchitecture::SAINT || arch == EncoderArchitecture::GNN);
    if (batch_dependent_encoder && batch_size != -1 && n > batch_size) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::cerr << "[resolve] warning: the model's encoder uses "
                         "inter-sample / batch-local attention (SAINT or GNN); its "
                         "predictions depend on batch composition, so chunked "
                         "inference is disabled and a single full-batch forward is "
                         "used. Pass predict batch_size=-1 to silence this.\n";
        }
    }

    // Single-shot path: legacy behavior. Used when caller opts out of
    // chunking (-1), when the dataset already fits in one chunk
    // (n <= batch_size), or for a batch-composition-dependent encoder (above).
    // The pool-* tensors are threaded through so PlotEncoderRankPool /
    // PlotEncoderTransformer get the correct (n_plots, max_species) rank-pool
    // tensors instead of the embed-mode species_ids (shape (n_plots, top_k_species)).
    if (batch_size == -1 || n <= batch_size || batch_dependent_encoder) {
        auto result = predict(
            dataset.coordinates(),
            dataset.covariates(),
            dataset.hash_embedding(),
            dataset.species_ids(),
            dataset.species_vector(),
            dataset.genus_ids(),
            dataset.family_ids(),
            dataset.unknown_fraction(),
            dataset.unknown_count(),
            dataset.pool_genus_ids(),
            dataset.pool_family_ids(),
            dataset.pool_weights(),
            dataset.pool_mask(),
            dataset.pool_has_cover(),
            dataset.categorical_ids(),
            return_latent
        );

        // Use actual plot IDs from dataset
        result.plot_ids = dataset.plot_ids();

        // Copy targets from dataset for residual analysis
        result.targets = dataset.targets();

        return result;
    }

    // Chunked path: slice every input tensor along dim 0, forward each
    // chunk on `device_`, move the chunk outputs to CPU, accumulate, and
    // concat on CPU at the end. Keeping accumulators on CPU is what
    // bounds peak VRAM to a single chunk's footprint regardless of n.
    const auto& coordinates       = dataset.coordinates();
    const auto& covariates        = dataset.covariates();
    const auto& hash_embedding    = dataset.hash_embedding();
    const auto& species_ids       = dataset.species_ids();
    const auto& species_vector    = dataset.species_vector();
    const auto& genus_ids         = dataset.genus_ids();
    const auto& family_ids        = dataset.family_ids();
    const auto& unknown_fraction  = dataset.unknown_fraction();
    const auto& unknown_count     = dataset.unknown_count();
    const auto& pool_genus_ids    = dataset.pool_genus_ids();
    const auto& pool_family_ids   = dataset.pool_family_ids();
    const auto& pool_weights      = dataset.pool_weights();
    const auto& pool_mask         = dataset.pool_mask();
    const auto& pool_has_cover    = dataset.pool_has_cover();
    const auto& categorical_ids   = dataset.categorical_ids();

    // Per-target lists of CPU chunks to concatenate at the end.
    std::unordered_map<std::string, std::vector<torch::Tensor>> pred_chunks;
    std::vector<torch::Tensor> latent_chunks;

    for (int64_t start = 0; start < n; start += batch_size) {
        const int64_t end = std::min(start + batch_size, n);

        auto chunk = predict(
            slice0(coordinates,      start, end),
            slice0(covariates,       start, end),
            slice0(hash_embedding,   start, end),
            slice0(species_ids,      start, end),
            slice0(species_vector,   start, end),
            slice0(genus_ids,        start, end),
            slice0(family_ids,       start, end),
            slice0(unknown_fraction, start, end),
            slice0(unknown_count,    start, end),
            slice0(pool_genus_ids,   start, end),
            slice0(pool_family_ids,  start, end),
            slice0(pool_weights,     start, end),
            slice0(pool_mask,        start, end),
            slice0(pool_has_cover,   start, end),
            slice0(categorical_ids,  start, end),
            return_latent
        );

        for (auto& [name, tensor] : chunk.predictions) {
            pred_chunks[name].push_back(tensor.detach().to(torch::kCPU));
        }
        if (return_latent && chunk.latent.defined()) {
            latent_chunks.push_back(chunk.latent.detach().to(torch::kCPU));
        }
    }

    ResolvePredictions result;
    for (auto& [name, chunks] : pred_chunks) {
        result.predictions[name] = torch::cat(chunks, /*dim=*/0);
    }
    if (return_latent && !latent_chunks.empty()) {
        result.latent = torch::cat(latent_chunks, /*dim=*/0);
    }

    // Use actual plot IDs from dataset
    result.plot_ids = dataset.plot_ids();

    // Copy targets from dataset for residual analysis
    result.targets = dataset.targets();

    return result;
}

ResolvePredictions Predictor::predict(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor hash_embedding,
    torch::Tensor species_ids,
    torch::Tensor species_vector,
    torch::Tensor genus_ids,
    torch::Tensor family_ids,
    torch::Tensor unknown_fraction,
    torch::Tensor unknown_count,
    torch::Tensor pool_genus_ids,
    torch::Tensor pool_family_ids,
    torch::Tensor pool_weights,
    torch::Tensor pool_mask,
    torch::Tensor pool_has_cover,
    torch::Tensor categorical_ids,
    bool return_latent
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Build continuous features based on encoding mode (must match trainer.cpp)
    std::vector<torch::Tensor> continuous_parts;
    push_if_defined(continuous_parts, coordinates);
    push_if_defined(continuous_parts, covariates);
    push_if_defined(continuous_parts, unknown_fraction, 1);
    if (unknown_count.defined() && unknown_count.numel() > 0) {
        continuous_parts.push_back(unknown_count.to(torch::kFloat32).unsqueeze(1));
    }

    // For hash mode, include hash embedding in continuous
    if (model_->species_encoding() == SpeciesEncodingMode::Hash &&
        !model_->uses_explicit_vector()) {
        push_if_defined(continuous_parts, hash_embedding);
    }

    torch::Tensor continuous;
    if (!continuous_parts.empty()) {
        continuous = torch::cat(continuous_parts, /*dim=*/1);
    } else {
        int64_t n_samples = 0;
        if (hash_embedding.defined()) n_samples = hash_embedding.size(0);
        else if (species_ids.defined()) n_samples = species_ids.size(0);
        else if (species_vector.defined()) n_samples = species_vector.size(0);
        continuous = torch::zeros({n_samples, 0}, torch::kFloat32);
    }

    // Scale continuous features
    torch::Tensor scaled_continuous;
    if (scalers_.continuous_mean.defined() && continuous.size(1) > 0) {
        scaled_continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    } else {
        scaled_continuous = continuous;
    }
    scaled_continuous = scaled_continuous.to(device_);

    // Move tensors to device
    genus_ids = to_device_if_defined(genus_ids, device_);
    family_ids = to_device_if_defined(family_ids, device_);
    species_ids = to_device_if_defined(species_ids, device_);
    species_vector = to_device_if_defined(species_vector, device_);
    pool_genus_ids = to_device_if_defined(pool_genus_ids, device_);
    pool_family_ids = to_device_if_defined(pool_family_ids, device_);
    pool_weights = to_device_if_defined(pool_weights, device_);
    pool_mask = to_device_if_defined(pool_mask, device_);
    pool_has_cover = to_device_if_defined(pool_has_cover, device_);
    categorical_ids = to_device_if_defined(categorical_ids, device_);

    // Get predictions using appropriate encoding mode
    auto outputs = model_->forward(scaled_continuous, genus_ids, family_ids, species_ids, species_vector,
                                    pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                    categorical_ids);

    ResolvePredictions result;

    // Process each output
    for (const auto& cfg : model_->schema().targets) {
        auto out_it = outputs.find(cfg.name);
        if (out_it == outputs.end()) continue;

        auto pred = out_it->second;

        if (cfg.task == TaskType::Classification) {
            // Return class predictions
            result.predictions[cfg.name] = torch::argmax(pred, /*dim=*/1);
        } else {
            // Unscale and inverse transform
            pred = pred.squeeze(-1);

            auto scaler_it = scalers_.target_scalers.find(cfg.name);
            if (scaler_it != scalers_.target_scalers.end()) {
                pred = pred * scaler_it->second.second.to(device_) + scaler_it->second.first.to(device_);
            }

            if (cfg.transform == TransformType::Log1p) {
                pred = torch::expm1(torch::clamp(pred, kExpClampMin, kExpClampMax));
            }

            result.predictions[cfg.name] = pred;
        }
    }

    // Optionally return latent
    if (return_latent) {
        result.latent = model_->get_latent(scaled_continuous, genus_ids, family_ids, species_ids, species_vector,
                                           pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                                           categorical_ids);
    }

    // Create plot indices as strings
    int64_t n_samples = scaled_continuous.size(0);
    for (int64_t i = 0; i < n_samples; ++i) {
        result.plot_ids.push_back(std::to_string(i));
    }

    return result;
}

torch::Tensor Predictor::get_embeddings(
    torch::Tensor coordinates,
    torch::Tensor covariates,
    torch::Tensor hash_embedding,
    torch::Tensor genus_ids,
    torch::Tensor family_ids
) {
    torch::NoGradGuard no_grad;
    model_->eval();

    // Concatenate continuous features (hash_embedding may be empty for non-hash modes)
    std::vector<torch::Tensor> continuous_parts;
    if (coordinates.defined() && coordinates.numel() > 0) {
        continuous_parts.push_back(coordinates);
    }
    if (hash_embedding.defined() && hash_embedding.numel() > 0) {
        continuous_parts.push_back(hash_embedding);
    }
    if (covariates.defined() && covariates.size(1) > 0) {
        continuous_parts.push_back(covariates);
    }
    if (continuous_parts.empty()) {
        throw std::runtime_error("get_embeddings requires at least one non-empty input tensor");
    }
    auto continuous = torch::cat(continuous_parts, /*dim=*/1);

    // Scale continuous features, guarding an unfit scaler the same way predict()
    // does (a model whose continuous scalers were never fit leaves these
    // undefined; subtracting an undefined tensor would throw).
    auto scaled_continuous = continuous;
    if (scalers_.continuous_mean.defined() && continuous.size(1) > 0) {
        scaled_continuous = (continuous - scalers_.continuous_mean) / scalers_.continuous_scale;
    }
    scaled_continuous = scaled_continuous.to(device_);

    genus_ids = to_device_if_defined(genus_ids, device_);
    family_ids = to_device_if_defined(family_ids, device_);

    return model_->get_latent(scaled_continuous, genus_ids, family_ids);
}

void Predictor::optimize_for_inference() {
    model_->eval();
    torch::NoGradGuard no_grad;

    // Fuse Linear+BatchNorm1d pairs in all Sequential modules
    for (auto& module : model_->modules(/*include_self=*/false)) {
        auto seq = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(module);
        if (!seq) continue;

        for (size_t i = 0; i + 1 < seq->size(); ++i) {
            auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>((*seq)[i]);
            // Since the fp32-norm guard (gcol33/resolve#21), every norm in the
            // Sequential MLP is wrapped in Fp32Norm, so the child is an
            // Fp32NormImpl, not a bare BatchNorm1dImpl. Unwrap it to reach the
            // inner BatchNorm1d; fall back to a direct cast for any un-wrapped
            // norm (e.g. RESOLVE_FP32_NORM=0 builds).
            std::shared_ptr<torch::nn::BatchNorm1dImpl> bn;
            if (auto fp32 = std::dynamic_pointer_cast<Fp32NormImpl>((*seq)[i + 1])) {
                bn = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(fp32->inner_module());
            } else {
                bn = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>((*seq)[i + 1]);
            }
            if (!linear || !bn) continue;

            // Fuse: W_new = bn.weight / sqrt(var + eps) * W_linear
            //        b_new = bn.weight / sqrt(var + eps) * b_linear + bn.bias - bn.weight * mean / sqrt(var + eps)
            auto std_val = torch::sqrt(bn->running_var + bn->options.eps());
            auto scale = bn->weight / std_val;

            linear->weight.mul_(scale.unsqueeze(1));
            // Fold the BN shift into the Linear bias. If the Linear has no bias
            // (bias-free Linear immediately before BN, as in tabm/gnn/attention
            // sub-blocks), synthesize one so the shift term is not silently
            // dropped, which would change the fused output.
            auto shift = bn->bias - scale * bn->running_mean;
            if (linear->bias.defined()) {
                linear->bias.mul_(scale).add_(shift);
            } else {
                linear->bias = linear->register_parameter("bias", shift.clone());
            }

            // Replace BN with Identity (identity module in Sequential)
            seq->replace_module(std::to_string(i + 1), torch::nn::Identity());
        }
    }
}

torch::Tensor Predictor::get_genus_embeddings() const {
    return model_->get_genus_weights();
}

torch::Tensor Predictor::get_family_embeddings() const {
    return model_->get_family_weights();
}

torch::Tensor Predictor::get_species_embeddings() const {
    return model_->get_species_weights();
}

} // namespace resolve
