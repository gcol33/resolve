#include "resolve/predictor.hpp"
#include "resolve/dataset.hpp"
#include "resolve/continuous_block.hpp"
#include "resolve/encoder.hpp"  // Fp32NormImpl (inference-time BN fusion)
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace resolve {

namespace {

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
// comparison a sizes-only checkpoint supports), then the ordered name lists
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
        return;  // sizes-only checkpoint (or an unfit vocab): size check is all there is
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

    // A checkpoint that recorded only the vocabulary SIZES carries nothing to
    // rebuild an inference dataset in this model's ID namespace from, and
    // predict() can only compare sizes. Say so once, loudly: the failure it
    // guards against is silent wrong predictions.
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
    // embedding rows and return plausible, wrong numbers.
    validate_dataset_vocabs(dataset);

    auto result = predict(dataset.continuous_inputs(), dataset.plot_inputs(),
                          return_latent, batch_size);
    result.plot_ids = dataset.plot_ids();
    // The dataset's targets ride along for residual analysis.
    result.targets = dataset.targets();
    return result;
}

ResolvePredictions Predictor::predict(
    const ContinuousInputs& parts,
    const PlotInputs& inputs,
    bool return_latent,
    int64_t batch_size
) {
    auto result = run_inference(parts, inputs, /*want_outputs=*/true, return_latent, batch_size);
    const int64_t n = parts.n_rows() > 0 ? parts.n_rows() : inputs.n_plots();
    result.plot_ids.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        result.plot_ids.push_back(std::to_string(i));
    }
    return result;
}

torch::Tensor Predictor::get_embeddings(
    const ContinuousInputs& parts,
    const PlotInputs& inputs,
    int64_t batch_size
) {
    return run_inference(parts, inputs, /*want_outputs=*/false, /*want_latent=*/true,
                         batch_size).latent;
}

ResolvePredictions Predictor::run_inference(
    const ContinuousInputs& parts,
    const PlotInputs& inputs,
    bool want_outputs,
    bool want_latent,
    int64_t batch_size
) {
    // Validate batch_size: -1 (one-shot) or strictly positive. 0 / <-1 reject.
    if (batch_size != -1 && batch_size <= 0) {
        throw std::invalid_argument(
            "Predictor::predict: batch_size must be -1 (whole dataset) "
            "or a positive integer; got " + std::to_string(batch_size));
    }
    if (inputs.continuous.defined()) {
        throw std::invalid_argument(
            "Predictor::predict: PlotInputs::continuous must be empty; the "
            "continuous block is assembled from the ContinuousInputs parts");
    }

    const auto& schema = model_->schema();
    auto present = [](const torch::Tensor& t) { return t.defined() && t.numel() > 0; };
    if (schema.track_unknown_fraction && !present(parts.unknown_fraction)) {
        throw std::invalid_argument(
            "Predictor: the model reads an unknown-species fraction "
            "(track_unknown_fraction), so pass unknown_fraction");
    }
    if (schema.track_unknown_count && !present(parts.unknown_count)) {
        throw std::invalid_argument(
            "Predictor: the model reads an unknown-species count "
            "(track_unknown_count), so pass unknown_count");
    }

    ContinuousInputs block_parts = parts;
    if (!model_->hash_in_continuous()) block_parts.hash_embedding = torch::Tensor();
    const PlotInputs species_inputs = inputs.without_empty();
    const int64_t n = parts.n_rows() > 0 ? parts.n_rows() : species_inputs.n_plots();

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
    const bool one_chunk = batch_size == -1 || batch_dependent_encoder;
    const int64_t step = one_chunk ? std::max<int64_t>(n, 1) : batch_size;

    torch::NoGradGuard no_grad;
    model_->eval();

    // Per-target lists of CPU chunks to concatenate at the end. Keeping the
    // accumulators on the CPU bounds peak device memory to one chunk.
    std::unordered_map<std::string, std::vector<torch::Tensor>> pred_chunks;
    std::unordered_map<std::string, std::vector<torch::Tensor>> prob_chunks;
    std::vector<torch::Tensor> latent_chunks;

    // At least one chunk runs, so an empty input still yields every output
    // with zero rows.
    for (int64_t start = 0; start == 0 || start < n; start += step) {
        const int64_t end = std::min(start + step, n);

        // The continuous block, assembled and standardised exactly as the
        // Trainer did at fit time (continuous_block.hpp).
        PlotInputs chunk = species_inputs.slice(start, end);
        chunk.continuous = standardize_continuous(
            assemble_continuous(block_parts.slice(start, end), schema.missing_values,
                                end - start),
            scalers_);
        chunk = chunk.to(device_);

        if (want_outputs) {
            const auto outputs = model_->forward(chunk);
            for (const auto& cfg : schema.targets) {
                auto out_it = outputs.find(cfg.name);
                if (out_it == outputs.end()) continue;
                auto pred = out_it->second.detach();

                if (cfg.task == TaskType::Classification) {
                    // The head emits logits; the softmax row is the per-class
                    // probability and its argmax the predicted code, so the
                    // two outputs agree by construction.
                    auto probs = torch::softmax(pred, /*dim=*/1);
                    pred_chunks[cfg.name].push_back(
                        torch::argmax(probs, /*dim=*/1).to(torch::kCPU));
                    prob_chunks[cfg.name].push_back(probs.to(torch::kCPU));
                } else {
                    // Unscale and inverse transform
                    pred = pred.squeeze(-1);
                    auto scaler_it = scalers_.target_scalers.find(cfg.name);
                    if (scaler_it != scalers_.target_scalers.end()) {
                        pred = pred * scaler_it->second.second.to(device_) +
                               scaler_it->second.first.to(device_);
                    }
                    if (cfg.transform == TransformType::Log1p) {
                        pred = torch::expm1(torch::clamp(pred, kExpClampMin, kExpClampMax));
                    }
                    pred_chunks[cfg.name].push_back(pred.to(torch::kCPU));
                }
            }
        }
        if (want_latent) {
            latent_chunks.push_back(model_->get_latent(chunk).detach().to(torch::kCPU));
        }
    }

    ResolvePredictions result;
    for (auto& [name, chunks] : pred_chunks) {
        result.predictions[name] = torch::cat(chunks, /*dim=*/0);
    }
    for (auto& [name, chunks] : prob_chunks) {
        result.probabilities[name] = torch::cat(chunks, /*dim=*/0);
    }
    if (!latent_chunks.empty()) {
        result.latent = torch::cat(latent_chunks, /*dim=*/0);
    }
    return result;
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
            // Since the fp32-norm guard, every norm in the
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
