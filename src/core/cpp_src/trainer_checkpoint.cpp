// Trainer persistence: writing checkpoints (with the run progress resume()
// reads), loading them into a fresh model or into this trainer, resuming a
// run, and the in-memory snapshots of weights and optimizer state.

#include "resolve/trainer.hpp"
#include "resolve/checkpoint.hpp"
#include "resolve/gpu.hpp"
#include "resolve/io_retry.hpp"
#include "resolve/string_list_archive.hpp"

#ifdef RESOLVE_HAS_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <algorithm>
#include <mutex>
#include <sstream>

namespace resolve {

namespace {

// The key whose presence marks a checkpoint carrying run progress.
constexpr const char* kProgressKey = "resume_next_epoch";

// Where the best epoch's weights live in a progress checkpoint.
enum class BestWeightsSource : int64_t {
    None = 0,        // no best epoch yet (or a fixed-duration run)
    Checkpoint = 1,  // the checkpoint's own model weights
    Stored = 2,      // the resume_best_weights bytes
};

torch::Tensor bytes_tensor(const std::string& bytes) {
    if (bytes.empty()) return torch::empty({0}, torch::kUInt8);
    return torch::from_blob(const_cast<char*>(bytes.data()),
                            {static_cast<int64_t>(bytes.size())}, torch::kUInt8)
        .clone();
}

std::string tensor_bytes(const torch::Tensor& t) {
    const auto bytes = t.to(torch::kCPU).to(torch::kUInt8).contiguous();
    if (bytes.numel() == 0) return {};
    return std::string(reinterpret_cast<const char*>(bytes.data_ptr<uint8_t>()),
                       static_cast<size_t>(bytes.numel()));
}

torch::Tensor float_tensor(const std::vector<float>& values) {
    return values.empty() ? torch::empty({0}, torch::kFloat32)
                          : torch::tensor(values, torch::kFloat32);
}

std::vector<float> float_vector(const torch::Tensor& t) {
    const auto values = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* data = values.numel() > 0 ? values.data_ptr<float>() : nullptr;
    return std::vector<float>(data, data + values.numel());
}

torch::Tensor read_required(torch::serialize::InputArchive& archive, const std::string& key,
                            const std::string& path) {
    torch::Tensor t;
    if (!archive.try_read(key, t)) {
        throw std::invalid_argument("checkpoint '" + path + "' carries run progress but "
                                    "no '" + key + "' entry; it is incomplete");
    }
    return t;
}

// (n_targets, n_epochs): row i is `history[targets[i].name]`, an absent target
// a row of zeros.
std::vector<std::string> names_of(const std::vector<TargetConfig>& targets) {
    std::vector<std::string> names;
    names.reserve(targets.size());
    for (const auto& target : targets) names.push_back(target.name);
    return names;
}

torch::Tensor history_matrix(const std::unordered_map<std::string, std::vector<float>>& history,
                             const std::vector<std::string>& names, size_t n_epochs) {
    auto out = torch::zeros({static_cast<int64_t>(names.size()),
                             static_cast<int64_t>(n_epochs)}, torch::kFloat32);
    for (size_t i = 0; i < names.size(); ++i) {
        auto it = history.find(names[i]);
        if (it == history.end() || it->second.empty()) continue;
        out[static_cast<int64_t>(i)].copy_(float_tensor(it->second));
    }
    return out;
}

std::unordered_map<std::string, std::vector<float>> history_map(
    const torch::Tensor& matrix, const std::vector<std::string>& names) {
    std::unordered_map<std::string, std::vector<float>> out;
    if (matrix.numel() == 0 || matrix.size(1) == 0) return out;
    for (size_t i = 0; i < names.size(); ++i) {
        out[names[i]] = float_vector(matrix[static_cast<int64_t>(i)]);
    }
    return out;
}

Scalers scalers_on_cpu(Scalers scalers) {
    auto cpu = [](torch::Tensor& t) {
        if (t.defined()) t = t.to(torch::kCPU);
    };
    cpu(scalers.continuous_fill);
    cpu(scalers.continuous_mean);
    cpu(scalers.continuous_scale);
    for (auto& [name, mean_scale] : scalers.target_scalers) {
        cpu(mean_scale.first);
        cpu(mean_scale.second);
    }
    return scalers;
}

bool same_tensor(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.defined() != b.defined()) return false;
    if (!a.defined()) return true;
    const auto ca = a.to(torch::kCPU);
    const auto cb = b.to(torch::kCPU);
    return ca.sizes() == cb.sizes() && ca.dtype() == cb.dtype() && torch::equal(ca, cb);
}

bool same_scalers(const Scalers& a, const Scalers& b) {
    if (!same_tensor(a.continuous_fill, b.continuous_fill) ||
        !same_tensor(a.continuous_mean, b.continuous_mean) ||
        !same_tensor(a.continuous_scale, b.continuous_scale) ||
        a.target_scalers.size() != b.target_scalers.size()) {
        return false;
    }
    for (const auto& [name, mean_scale] : a.target_scalers) {
        auto it = b.target_scalers.find(name);
        if (it == b.target_scalers.end() || !same_tensor(mean_scale.first, it->second.first) ||
            !same_tensor(mean_scale.second, it->second.second)) {
            return false;
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// Snapshots
// =============================================================================

std::string Trainer::snapshot_weights() const {
    torch::serialize::OutputArchive archive;
    model_->save(archive);
    std::ostringstream out;
    archive.save_to(out);
    return out.str();
}

void Trainer::restore_weights(const std::string& snapshot) {
    std::istringstream in(snapshot);
    torch::serialize::InputArchive archive;
    archive.load_from(in, config_.device);
    model_->load(archive);
}

std::string Trainer::snapshot_optimizer() const {
    torch::serialize::OutputArchive archive;
    optimizer_->save(archive);
    std::ostringstream out;
    archive.save_to(out);
    return out.str();
}

void Trainer::restore_optimizer(const std::string& snapshot) {
    std::istringstream in(snapshot);
    torch::serialize::InputArchive archive;
    archive.load_from(in, config_.device);
    optimizer_->load(archive);
}

void Trainer::restore_rng_states(const RunProgress& progress) const {
    if (progress.cpu_rng_state.defined()) {
        auto gen = at::detail::getDefaultCPUGenerator();
        std::lock_guard<std::mutex> lock(gen.mutex());
        gen.set_state(progress.cpu_rng_state);
    }
#ifdef RESOLVE_HAS_CUDA
    if (progress.cuda_rng_state.defined() && config_.device.is_cuda()) {
        auto gen = at::cuda::detail::getDefaultCUDAGenerator(config_.device.index());
        std::lock_guard<std::mutex> lock(gen.mutex());
        gen.set_state(progress.cuda_rng_state);
    }
#endif
}

// =============================================================================
// Writing
// =============================================================================

void Trainer::save(const std::string& path, const RunMetadata* metadata) const {
    save_checkpoint(path, metadata, nullptr);
}

void Trainer::save_checkpoint(const std::string& path, const RunMetadata* metadata,
                              const RunProgress* progress) const {
    torch::serialize::OutputArchive archive;

    // Model parameters and buffers under prefixed keys, beside the checkpoint's
    // own metadata keys.
    for (const auto& pair : model_->named_parameters()) {
        archive.write("param_" + pair.key(), pair.value());
    }
    for (const auto& pair : model_->named_buffers()) {
        archive.write("buffer_" + pair.key(), pair.value());
    }

    save_model_config(archive, model_->config());
    save_schema(archive, model_->schema());
    save_scalers(archive, scalers_);

    // Save categorical vocabulary (string -> code maps for each categorical
    // column). An empty vocab writes a count of zero, which load() reads back
    // as an empty vocabulary.
    categorical_vocab_.save(archive, "trainer_categorical_");

    // save_train_config writes the REQUESTED batch size (passed explicitly) and
    // the EFFECTIVE one it reads off the config. fit() restores
    // config_.batch_size to the requested value before returning, so a save()
    // after fit() would otherwise record effective == requested and lose the
    // OOM fallback the run used; it gets a copy carrying the tracked value.
    TrainConfig persisted_config = config_;
    if (effective_batch_size_ > 0) {
        persisted_config.batch_size = effective_batch_size_;
    }
    save_train_config(archive, persisted_config, requested_batch_size_);

    // Save run metadata if provided (final checkpoint only)
    if (metadata != nullptr) {
        save_run_metadata(archive, *metadata);
    }

    if (progress != nullptr) {
        write_progress(archive, *progress);
    }

    // Retry a transient write fault on flaky storage; a small
    // checkpoint re-write is cheap, so any failure is retried.
    io::with_retry([&] { archive.save_to(path); }, "checkpoint save");

    // Write human-readable JSON metadata alongside checkpoint
    if (metadata != nullptr) {
        write_metadata_json(path, model_->config(), persisted_config, *metadata,
                            model_->schema(), requested_batch_size_);
    }
}

void Trainer::write_progress(torch::serialize::OutputArchive& archive,
                             const RunProgress& progress) const {
    const auto& targets = loss_fn_.targets();
    const TrainResult& result = progress.result;

    archive.write(kProgressKey, torch::tensor(static_cast<int64_t>(progress.next_epoch)));
    archive.write("resume_best_loss", torch::tensor(progress.best_loss));
    archive.write("resume_best_epoch", torch::tensor(static_cast<int64_t>(result.best_epoch)));
    archive.write("resume_patience",
                  torch::tensor(static_cast<int64_t>(progress.patience_counter)));

    archive.write("resume_train_loss", float_tensor(result.train_loss_history));
    archive.write("resume_test_loss", float_tensor(result.test_loss_history));
    archive.write("resume_lr", float_tensor(result.lr_history));

    // The reduce-on-plateau schedule's state, the rate in double as the
    // optimizer holds it.
    const PlateauState& plateau = progress.plateau;
    archive.write("resume_plateau_best", torch::tensor(plateau.best, torch::kFloat64));
    archive.write("resume_plateau_bad_epochs",
                  torch::tensor(static_cast<int64_t>(plateau.bad_epochs)));
    archive.write("resume_plateau_cooldown",
                  torch::tensor(static_cast<int64_t>(plateau.cooldown_counter)));
    archive.write("resume_plateau_lr", torch::tensor(plateau.lr, torch::kFloat64));
    archive.write("resume_train_loss_by_target",
                  history_matrix(result.train_loss_history_by_target, names_of(targets),
                                 result.train_loss_history.size()));
    archive.write("resume_test_loss_by_target",
                  history_matrix(result.test_loss_history_by_target, names_of(targets),
                                 result.test_loss_history.size()));
    archive.write("resume_aux_loss", history_matrix(result.train_aux_loss_history,
                                                    model_->aux_loss_names(),
                                                    result.train_loss_history.size()));

    // The best epoch's metrics as three parallel lists.
    std::vector<std::string> metric_targets;
    std::vector<std::string> metric_names;
    std::vector<float> metric_values;
    for (const auto& [target, metrics] : result.final_metrics) {
        for (const auto& [name, value] : metrics) {
            metric_targets.push_back(target);
            metric_names.push_back(name);
            metric_values.push_back(value);
        }
    }
    write_string_list(archive, "resume_metric_target", metric_targets);
    write_string_list(archive, "resume_metric_name", metric_names);
    archive.write("resume_metric_value", float_tensor(metric_values));

    // The best epoch's weights are the checkpoint's own when it is the epoch
    // just finished, and stored beside them otherwise.
    BestWeightsSource best = BestWeightsSource::None;
    if (!progress.best_weights.empty()) {
        best = result.best_epoch == progress.next_epoch - 1 ? BestWeightsSource::Checkpoint
                                                           : BestWeightsSource::Stored;
    }
    archive.write("resume_best_source", torch::tensor(static_cast<int64_t>(best)));
    if (best == BestWeightsSource::Stored) {
        archive.write("resume_best_weights", bytes_tensor(progress.best_weights));
    }

    archive.write("resume_optimizer", bytes_tensor(snapshot_optimizer()));
    archive.write("resume_amp_scale", torch::tensor(amp_scale_));
    archive.write("resume_amp_growth", torch::tensor(static_cast<int64_t>(amp_growth_tracker_)));
    archive.write("resume_elapsed",
                  torch::tensor(progress.elapsed_seconds, torch::kFloat64));
    write_string_list(archive, "resume_created_at", {created_at_});
    archive.write("resume_requested_batch",
                  torch::tensor(static_cast<int64_t>(requested_batch_size_)));
    archive.write("resume_effective_batch",
                  torch::tensor(static_cast<int64_t>(effective_batch_size_)));

    // The split the run trained on, which resume() requires again.
    archive.write("resume_train_indices", train_indices_.to(torch::kCPU));
    archive.write("resume_test_indices", test_indices_.to(torch::kCPU));

    // The global generators dropout and cover dropout draw from, so the
    // continued epochs draw what an uninterrupted run would have.
    {
        auto gen = at::detail::getDefaultCPUGenerator();
        std::lock_guard<std::mutex> lock(gen.mutex());
        archive.write("resume_cpu_rng", gen.get_state());
    }
#ifdef RESOLVE_HAS_CUDA
    if (config_.device.is_cuda()) {
        auto gen = at::cuda::detail::getDefaultCUDAGenerator(config_.device.index());
        std::lock_guard<std::mutex> lock(gen.mutex());
        archive.write("resume_cuda_rng", gen.get_state());
    }
#endif
}

// =============================================================================
// Reading
// =============================================================================

torch::serialize::InputArchive Trainer::open_checkpoint(
    const std::string& path, torch::Device device, float vram_fraction,
    const LogCallback& log) {
    // The VRAM cap goes in before any device allocation, so the weights
    // respect it on their first upload; it changes nothing on a CPU device.
    if (device.is_cuda()) {
        set_vram_fraction(static_cast<double>(vram_fraction), device.index(), log);
    }

    torch::serialize::InputArchive archive;
    // `device` is passed to the unpickler, which remaps every storage as it is
    // read. Without it each tensor is restored to the device it was SAVED on,
    // so loading a GPU-trained checkpoint on a machine with no CUDA device
    // throws inside deserialization ("No CUDA GPUs are available").
    //
    // Retry a transient read fault on flaky storage. A mmap-backed load can
    // still fault as an OS structured exception, which the crash handler
    // reports (process.hpp); this catches the throwing-read failures.
    io::with_retry([&] { archive.load_from(path, device); }, "checkpoint load");
    return archive;
}

void Trainer::restore_model_state(torch::serialize::InputArchive& archive,
                                  ResolveModel& model, torch::Device device) {
    load_weights_into(archive, model);
    model->to(device);
}

std::tuple<ResolveModel, Scalers, CategoricalVocab> Trainer::load(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    auto archive = open_checkpoint(path, device, vram_fraction, default_log);

    ModelConfig config = load_model_config(archive);
    ResolveSchema schema = load_schema(archive, &config);
    Scalers scalers = load_scalers(archive);

    // Load the categorical vocabulary; a checkpoint without categorical
    // columns yields an empty one (CategoricalVocab::load).
    CategoricalVocab vocab = CategoricalVocab::load(archive, "trainer_categorical_");

    ResolveModel model(schema, config);
    restore_model_state(archive, model, device);

    return {model, scalers, vocab};
}

TrainConfig Trainer::load_train_config(const std::string& path) {
    torch::serialize::InputArchive archive;
    // Read onto the CPU. Deserialization restores every tensor in the archive,
    // model weights included, so a GPU-saved checkpoint would need a CUDA
    // device present just to read back a handful of scalars.
    io::with_retry([&] { archive.load_from(path, torch::kCPU); },
                   "checkpoint load (train config)");
    // Qualify to the free function (checkpoint.cpp); the unqualified name would
    // resolve to this static member and recurse.
    return resolve::load_train_config(archive);
}

RunMetadata Trainer::load_run_metadata(const std::string& path) {
    torch::serialize::InputArchive archive;
    // CPU for the same reason as load_train_config above: metadata is scalars.
    io::with_retry([&] { archive.load_from(path, torch::kCPU); },
                   "checkpoint load (run metadata)");
    return resolve::load_run_metadata(archive);
}

void Trainer::load_weights_into(torch::serialize::InputArchive& archive, ResolveModel& model) {
    // Freshly-constructed model parameters are leaf tensors with
    // requires_grad=true. Calling .copy_() on them directly trips autograd's
    // check_inplace ("a leaf Variable that requires grad is being used in an
    // in-place operation"). Mirror PyTorch's copy-inside-torch.no_grad().
    torch::NoGradGuard no_grad;
    int64_t n_expected = 0, n_loaded = 0;
    std::string first_missing;
    auto note_missing = [&](const std::string& key) {
        if (first_missing.empty()) first_missing = key;
    };
    for (const auto& pair : model->named_parameters()) {
        ++n_expected;
        torch::Tensor t;
        if (archive.try_read("param_" + pair.key(), t)) {
            pair.value().copy_(t);
            ++n_loaded;
        } else {
            note_missing("param_" + pair.key());
        }
    }
    // A data buffer (the species graph, the trait matrix, SAINT's reference
    // rows) takes the shape it was saved with; every other tensor must match.
    const auto data_buffers = model->data_buffers();
    auto is_data_buffer = [&data_buffers](const torch::Tensor& buffer) {
        return std::any_of(data_buffers.begin(), data_buffers.end(),
                           [&buffer](const torch::Tensor& d) { return d.is_same(buffer); });
    };
    for (const auto& pair : model->named_buffers()) {
        ++n_expected;
        torch::Tensor t;
        if (archive.try_read("buffer_" + pair.key(), t)) {
            if (is_data_buffer(pair.value())) {
                pair.value().set_data(t.to(pair.value().scalar_type()));
            } else {
                pair.value().copy_(t);
            }
            ++n_loaded;
        } else {
            note_missing("buffer_" + pair.key());
        }
    }
    // A silent skip leaves those tensors at fresh random init, so a mismatched
    // architecture would load as a partly-untrained model. Fail loudly instead
    // (this is what makes an encoder_architecture / sub-config drift visible).
    if (n_loaded != n_expected) {
        std::ostringstream msg;
        msg << "load_weights_into: checkpoint is missing " << (n_expected - n_loaded)
            << " of " << n_expected << " model tensors, so the loaded model would "
               "keep randomly-initialized weights. The model architecture likely "
               "does not match the checkpoint (e.g. a different encoder_architecture "
               "or architecture sub-config, or a checkpoint written before a change "
               "to this architecture's layout, which has to be retrained). First "
               "missing tensor: " << first_missing;
        throw std::runtime_error(msg.str());
    }
    // The missing-tensor check above catches a subset model loaded from a subset
    // checkpoint, but not the reverse: a checkpoint from a SUPERSET architecture
    // whose overlapping tensor names + shapes match a smaller model loads a
    // silent partial subset (copy_ shape-checks miss same-name/same-shape drift).
    // Count the checkpoint's own weight tensors and reject a mismatch loudly.
    int64_t n_ckpt_weights = 0;
    for (const auto& key : archive.keys()) {
        if (key.rfind("param_", 0) == 0 || key.rfind("buffer_", 0) == 0) {
            ++n_ckpt_weights;
        }
    }
    if (n_ckpt_weights != n_expected) {
        std::ostringstream msg;
        msg << "load_weights_into: checkpoint holds " << n_ckpt_weights
            << " weight tensors but the model has " << n_expected
            << ". The architecture does not match the checkpoint (a superset "
               "checkpoint would load only its overlapping tensors, silently "
               "leaving the extra checkpoint weights unused).";
        throw std::runtime_error(msg.str());
    }
}

void Trainer::load_state(
    const std::string& path,
    torch::Device device,
    float vram_fraction
) {
    auto archive = open_checkpoint(path, device, vram_fraction, config_.log);

    // The architecture of model_ must already match the checkpoint.
    restore_model_state(archive, model_, device);
    Scalers loaded = scalers_on_cpu(load_scalers(archive));
    categorical_vocab_ = CategoricalVocab::load(archive, "trainer_categorical_");

    // The device copies and optimizer belong to the weights and scalers they
    // were made with; config_.device follows the weights, so the evaluators
    // move the held-out fold to where the model now lives.
    release_training_state();
    config_.device = device;

    // A prepared split is standardised with the scalers it was fitted with;
    // the evaluators score the loaded weights in the loaded scalers, so the
    // split is re-expressed in those (a no-op when they are the same).
    if (data_prepared_ && !same_scalers(scalers_, loaded)) {
        train_ = standardized(unstandardized(train_, scalers_), loaded);
        test_ = standardized(unstandardized(test_, scalers_), loaded);
    }
    scalers_ = std::move(loaded);
}

// =============================================================================
// Resuming a run
// =============================================================================

Trainer::RunProgress Trainer::read_progress(torch::serialize::InputArchive& archive,
                                            const std::string& path) const {
    const auto& targets = loss_fn_.targets();
    auto scalar_int = [&](const char* key) {
        return static_cast<int>(read_required(archive, key, path).item<int64_t>());
    };

    RunProgress progress;
    progress.next_epoch = scalar_int(kProgressKey);
    progress.best_loss = read_required(archive, "resume_best_loss", path).item<float>();
    progress.patience_counter = scalar_int("resume_patience");

    TrainResult& result = progress.result;
    result.best_epoch = scalar_int("resume_best_epoch");
    result.train_loss_history = float_vector(read_required(archive, "resume_train_loss", path));
    result.test_loss_history = float_vector(read_required(archive, "resume_test_loss", path));
    result.lr_history = float_vector(read_required(archive, "resume_lr", path));

    progress.plateau.best = read_required(archive, "resume_plateau_best", path).item<double>();
    progress.plateau.bad_epochs = scalar_int("resume_plateau_bad_epochs");
    progress.plateau.cooldown_counter = scalar_int("resume_plateau_cooldown");
    progress.plateau.lr = read_required(archive, "resume_plateau_lr", path).item<double>();
    result.train_loss_history_by_target =
        history_map(read_required(archive, "resume_train_loss_by_target", path), names_of(targets));
    result.test_loss_history_by_target =
        history_map(read_required(archive, "resume_test_loss_by_target", path), names_of(targets));
    result.train_aux_loss_history =
        history_map(read_required(archive, "resume_aux_loss", path), model_->aux_loss_names());

    const auto metric_targets = read_string_list(archive, "resume_metric_target");
    const auto metric_names = read_string_list(archive, "resume_metric_name");
    const auto metric_values =
        float_vector(read_required(archive, "resume_metric_value", path));
    if (metric_targets.size() != metric_names.size() ||
        metric_targets.size() != metric_values.size()) {
        throw std::invalid_argument("checkpoint '" + path + "' holds a corrupt metric "
                                    "table in its run progress");
    }
    for (size_t i = 0; i < metric_targets.size(); ++i) {
        result.final_metrics[metric_targets[i]][metric_names[i]] = metric_values[i];
    }

    switch (static_cast<BestWeightsSource>(
                read_required(archive, "resume_best_source", path).item<int64_t>())) {
        case BestWeightsSource::None:
            break;
        case BestWeightsSource::Checkpoint:
            progress.best_weights = snapshot_weights();
            break;
        case BestWeightsSource::Stored:
            progress.best_weights =
                tensor_bytes(read_required(archive, "resume_best_weights", path));
            break;
        default:
            throw std::invalid_argument("checkpoint '" + path + "' names an unknown "
                                        "source for its best weights");
    }

    progress.optimizer_state = tensor_bytes(read_required(archive, "resume_optimizer", path));
    progress.has_amp_state = true;
    progress.amp_scale = read_required(archive, "resume_amp_scale", path).item<float>();
    progress.amp_growth_tracker = scalar_int("resume_amp_growth");
    progress.elapsed_seconds = read_required(archive, "resume_elapsed", path).item<double>();
    const auto created = read_string_list(archive, "resume_created_at");
    progress.created_at = created.empty() ? std::string() : created.front();
    progress.requested_batch_size = scalar_int("resume_requested_batch");
    progress.effective_batch_size = scalar_int("resume_effective_batch");

    progress.cpu_rng_state = read_required(archive, "resume_cpu_rng", path).to(torch::kCPU);
    torch::Tensor cuda_rng;
    if (config_.device.is_cuda() && archive.try_read("resume_cuda_rng", cuda_rng)) {
        progress.cuda_rng_state = cuda_rng.to(torch::kCPU);
    }
    return progress;
}

TrainResult Trainer::resume(const std::string& checkpoint_path) {
    if (!data_prepared_) {
        throw std::runtime_error(
            "Trainer::resume: nothing is prepared. Call prepare_data() with the "
            "interrupted run's data and seed first, so the run continues on its "
            "own split.");
    }

    // Read on the host: the progress is scalars, bytes and strings, and the
    // weights are copied into model_ on its device. train() applies the VRAM
    // cap before anything else reaches the device.
    auto archive = open_checkpoint(checkpoint_path, torch::kCPU, 1.0f, config_.log);
    {
        torch::Tensor probe;
        if (!archive.try_read(kProgressKey, probe)) {
            throw std::invalid_argument(
                "Trainer::resume: checkpoint '" + checkpoint_path + "' carries no run "
                "progress. resume() reads the checkpoints fit() writes while it "
                "trains (best.pt and checkpoint_<N>.pt under "
                "TrainConfig::checkpoint_dir), not a final checkpoint or one "
                "save() wrote.");
        }
    }

    // The run continues only on the split it trained on.
    const auto saved_train = read_required(archive, "resume_train_indices", checkpoint_path);
    const auto saved_test = read_required(archive, "resume_test_indices", checkpoint_path);
    if (!same_tensor(saved_train, train_indices_) || !same_tensor(saved_test, test_indices_)) {
        throw std::invalid_argument(
            "Trainer::resume: the prepared split differs from the one the run in '" +
            checkpoint_path + "' trained on (" + std::to_string(saved_train.numel()) +
            " fitting / " + std::to_string(saved_test.numel()) + " held-out plots there, " +
            std::to_string(train_indices_.numel()) + " / " +
            std::to_string(test_indices_.numel()) + " here). Prepare the same data "
            "with the same hold-out and seed.");
    }
    if (!same_scalers(scalers_on_cpu(load_scalers(archive)), scalers_)) {
        throw std::invalid_argument(
            "Trainer::resume: the prepared data's scalers differ from the run's in '" +
            checkpoint_path + "', so the plots carry different values; prepare the "
            "interrupted run's data.");
    }

    restore_model_state(archive, model_, config_.device);
    RunProgress progress = read_progress(archive, checkpoint_path);
    return train(&progress);
}

}  // namespace resolve
