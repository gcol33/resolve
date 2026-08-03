// resolve_rcpp.cpp - Module registration + free functions.
//
// Issue #17: the R bindings are now a thin client over the `resolve_c` C ABI
// (resolve/resolve_capi.h). No libtorch C++ header is included anywhere under
// r/src/. The Rcpp module surface (class / method / free-function names and R
// argument types) is identical to the libtorch-linked version, so R/resolve.R
// and the testthat suite are unchanged.
// [[Rcpp::plugins(cpp17)]]

#include "rcpp_common.h"
#include "rcpp_dataset.h"
#include "rcpp_model.h"
#include "rcpp_trainer.h"
#include "rcpp_predictor.h"

// =============================================================================
// Expose module-managed wrapper classes to non-module Rcpp machinery so the
// factory-style free functions (from_csv, load, ...) can return wrappers by
// value. See the previous implementation's note: RCPP_EXPOSED_CLASS_NODECL
// emits the wrap()/as() traits routing through the module's S4 representation.
// =============================================================================

RCPP_EXPOSED_CLASS_NODECL(RResolveDataset)
RCPP_EXPOSED_CLASS_NODECL(RResolveModel)
RCPP_EXPOSED_CLASS_NODECL(RTrainer)
RCPP_EXPOSED_CLASS_NODECL(RPredictor)

RCPP_MODULE(resolve_module) {
    class_<RResolveDataset>("ResolveDataset")
        .method("coordinates", &RResolveDataset::coordinates, "Get coordinate matrix")
        .method("covariates", &RResolveDataset::covariates, "Get covariate matrix")
        .method("hash_embedding", &RResolveDataset::hash_embedding, "Get hash embedding matrix")
        .method("species_ids", &RResolveDataset::species_ids, "Get species IDs matrix")
        .method("species_vector", &RResolveDataset::species_vector, "Get explicit species vector")
        .method("genus_ids", &RResolveDataset::genus_ids, "Get genus IDs matrix")
        .method("family_ids", &RResolveDataset::family_ids, "Get family IDs matrix")
        .method("categorical_ids", &RResolveDataset::categorical_ids, "Get categorical covariate codes matrix")
        .method("pool_genus_ids", &RResolveDataset::pool_genus_ids, "Get rank-pool genus IDs matrix")
        .method("pool_family_ids", &RResolveDataset::pool_family_ids, "Get rank-pool family IDs matrix")
        .method("pool_weights", &RResolveDataset::pool_weights, "Get rank-pool per-species weights matrix")
        .method("pool_mask", &RResolveDataset::pool_mask, "Get rank-pool valid-species mask matrix")
        .method("pool_has_cover", &RResolveDataset::pool_has_cover, "Get per-plot has-abundance flag")
        .method("has_pool_data", &RResolveDataset::has_pool_data, "Check if rank-pool tensors are present")
        .method("unknown_fraction", &RResolveDataset::unknown_fraction, "Get unknown species fraction")
        .method("unknown_count", &RResolveDataset::unknown_count, "Get unknown species count")
        .method("targets", &RResolveDataset::targets, "Get target values as list")
        .method("schema", &RResolveDataset::schema, "Get dataset schema")
        .method("plot_ids", &RResolveDataset::plot_ids, "Get plot IDs")
        .method("species_vocab", &RResolveDataset::species_vocab, "Get species vocabulary")
        .method("n_plots", &RResolveDataset::n_plots, "Get number of plots")
        .method("config", &RResolveDataset::config, "Get dataset config")
        .method("has_raw_species_data", &RResolveDataset::has_raw_species_data, "Check if raw species data is available")
        .method("raw_species_ids", &RResolveDataset::raw_species_ids, "Get raw species IDs")
        .method("raw_weights", &RResolveDataset::raw_weights, "Get raw species weights")
        .method("plot_offsets", &RResolveDataset::plot_offsets, "Get plot offsets for raw data")
        .method("taxonomy_vocab", &RResolveDataset::taxonomy_vocab, "Get taxonomy vocabulary info")
        .method("categorical_vocab", &RResolveDataset::categorical_vocab, "Get categorical covariate vocabulary (per-column code maps)")
        ;

    function("ResolveDataset_from_csv", &RResolveDataset::from_csv, "Load dataset from CSV files");
    function("ResolveDataset_from_species_csv", &RResolveDataset::from_species_csv, "Load dataset from single species CSV");
    function("ResolveDataset_from_csv_with_schema", &RResolveDataset::from_csv_with_schema, "Load dataset reusing another dataset's vocabularies and class mappings");
    function("ResolveDataset_from_dataframe", &RResolveDataset::from_dataframe, "Load dataset from in-memory header + species column lists");
    function("ResolveDataset_from_dataframe_header", &RResolveDataset::from_dataframe_header, "Load dataset from in-memory header columns + species CSV path");
    function("ResolveDataset_from_species_dataframe", &RResolveDataset::from_species_dataframe, "Load dataset from a single in-memory long-format column list");
    function("ResolveDataset_from_dataframe_with_schema", &RResolveDataset::from_dataframe_with_schema, "In-memory analog of from_csv_with_schema");

    class_<RResolveModel>("ResolveModel")
        .constructor<List, List>("Create a ResolveModel")
        .method("forward", &RResolveModel::forward, "Forward pass")
        .method("get_latent", &RResolveModel::get_latent, "Get latent representations")
        .method("train", &RResolveModel::train, "Set training mode")
        .method("eval", &RResolveModel::eval, "Set evaluation mode")
        .method("to_device", &RResolveModel::to_device, "Move model to device")
        .method("latent_dim", &RResolveModel::latent_dim, "Get latent dimension")
        .method("forward_with_aux", &RResolveModel::forward_with_aux, "Forward pass with MoE auxiliary loss")
        .method("forward_single", &RResolveModel::forward_single, "Forward pass for single target")
        .method("encode_with_activations", &RResolveModel::encode_with_activations, "Encode with intermediate activations")
        .method("get_gate_probs", &RResolveModel::get_gate_probs, "Get MoE gating probabilities")
        .method("set_traits", &RResolveModel::set_traits, "Set species trait matrix for TraitNet")
        .method("species_encoding", &RResolveModel::species_encoding, "Get species encoding mode")
        .method("uses_explicit_vector", &RResolveModel::uses_explicit_vector, "Check if using explicit vector")
        .method("uses_moe", &RResolveModel::uses_moe, "Check if using Mixture of Experts")
        .method("n_experts", &RResolveModel::n_experts, "Get number of experts")
        .method("get_genus_weights", &RResolveModel::get_genus_weights, "Get genus embedding weights")
        .method("get_family_weights", &RResolveModel::get_family_weights, "Get family embedding weights")
        .method("get_species_weights", &RResolveModel::get_species_weights, "Get species embedding weights")
        ;

    class_<RTrainer>("Trainer")
        .constructor<RResolveModel&, List>("Create a Trainer")
        .method("prepare_data", &RTrainer::prepare_data, "Prepare training data from tensors")
        .method("prepare_data_pool", &RTrainer::prepare_data_pool, "Prepare training data for rank_pool/transformer modes")
        .method("prepare_data_from_dataset", &RTrainer::prepare_data_from_dataset, "Prepare training data from ResolveDataset")
        .method("fit", &RTrainer::fit, "Train the model")
        .method("save", &RTrainer::save, "Save model checkpoint")
        .method("get_scalers", &RTrainer::get_scalers, "Get fitted scalers")
        .method("get_config", &RTrainer::get_config, "Get training configuration")
        .method("compute_diagnostics", &RTrainer::compute_diagnostics, "Compute network diagnostics")
        .method("compute_calibration", &RTrainer::compute_calibration, "Compute calibration for a target")
        .method("compute_residuals", &RTrainer::compute_residuals, "Compute residual analysis for a target")
        .method("compute_classification_predictions", &RTrainer::compute_classification_predictions, "Per-plot test-fold predictions for a classification target")
        .method("load_state", &RTrainer::load_state, "Load checkpoint weights/scalers/vocab into this trainer in place")
        .method("test_indices", &RTrainer::test_indices, "Global plot indices of the held-out test fold")
        .method("train_indices", &RTrainer::train_indices, "Global plot indices of the training fold")
        .method("test_plot_ids", &RTrainer::test_plot_ids, "Plot IDs of the held-out test fold")
        .method("train_plot_ids", &RTrainer::train_plot_ids, "Plot IDs of the training fold")
        .method("categorical_vocab", &RTrainer::categorical_vocab, "Get categorical covariate vocabulary captured at prepare_data time")
        .method("cross_validate", &RTrainer::cross_validate, "Run k-fold cross-validation")
        .method("cross_validate_spatial", &RTrainer::cross_validate_spatial, "Run spatial block cross-validation")
        .method("predict_from_trainer", &RTrainer::predict_from_trainer, "Make predictions using trainer's model")
        ;

    class_<RPredictor>("Predictor")
        .method("predict", &RPredictor::predict, "Make predictions from tensors")
        .method("predict_dataset", &RPredictor::predict_dataset, "Make predictions from ResolveDataset")
        .method("get_embeddings", &RPredictor::get_embeddings, "Get latent embeddings")
        .method("get_genus_embeddings", &RPredictor::get_genus_embeddings, "Get genus embeddings")
        .method("get_family_embeddings", &RPredictor::get_family_embeddings, "Get family embeddings")
        .method("get_species_embeddings", &RPredictor::get_species_embeddings, "Get species embeddings")
        .method("optimize_for_inference", &RPredictor::optimize_for_inference, "Fuse BatchNorm for faster inference")
        .method("device", &RPredictor::device, "Get current device")
        .method("get_scalers", &RPredictor::get_scalers, "Get fitted scalers")
        .method("categorical_vocab", &RPredictor::categorical_vocab, "Get categorical covariate vocabulary loaded from the checkpoint")
        ;

    function("Predictor_load", &RPredictor::load, "Load predictor from checkpoint");
    function("Trainer_load_train_config", &RTrainer::load_train_config,
             "Recover the persisted TrainConfig from a checkpoint (as a list)");
    function("Trainer_load_run_metadata", &RTrainer::load_run_metadata,
             "Recover the persisted run metadata from a checkpoint (as a list)");
}

// =============================================================================
// Free functions (exported via compileAttributes)
// =============================================================================

// [[Rcpp::export]]
std::string resolve_version() {
    capi_require_loaded();
    return std::string(resolve_capi_version());
}

// Cap the PyTorch CUDA caching allocator at `fraction` of device VRAM.
// fraction in (0, 1]; 1.0 disables the cap. device_index = -1L uses the current
// CUDA device. No-op on CPU-only builds.
// [[Rcpp::export]]
void resolve_set_vram_fraction(double fraction, int device_index = -1) {
    capi_require_loaded();
    capi_check_status(resolve_capi_set_vram_fraction(fraction, device_index));
}

// Pin libtorch's intra-op / inter-op thread pools (issue #18). intraop /
// interop <= 0 keep libtorch's default. Best-effort; call at load before the
// first op. Removes the worker threads whose teardown join crashes the
// Rscript.exe launcher on Windows.
// [[Rcpp::export]]
void resolve_set_thread_pools(int intraop_threads, int interop_threads = -1) {
    capi_require_loaded();
    capi_check_status(resolve_capi_set_thread_pools(intraop_threads, interop_threads));
}

// Install the Windows crash handler: turn an unhandled native fault into an
// immediate TerminateProcess instead of a JIT-debugger hang (issue #19) or a
// teardown access violation under Rscript.exe (issue #18). No-op off Windows.
// [[Rcpp::export]]
void resolve_install_crash_handler(int shutdown_exit_code = 0) {
    capi_require_loaded();
    capi_check_status(resolve_capi_install_crash_handler(shutdown_exit_code));
}

// Mark all engine work complete so a subsequent native fault during process
// teardown exits with the shutdown code rather than a crash code (issue #18).
// Registered as an on-exit finalizer in zzz.R.
// [[Rcpp::export]]
void resolve_signal_work_complete() {
    capi_require_loaded();
    capi_check_status(resolve_capi_signal_work_complete());
}

//' Configure the PyTorch CUDA caching allocator
//'
//' Set the \code{PYTORCH_CUDA_ALLOC_CONF} environment variable to a
//' platform-aware default (Linux/macOS get an \code{expandable_segments:True,}
//' prefix; Windows omits it). Best-effort: if torch already initialized its
//' allocator, the change may not take effect for the running process.
//'
//' @param force If \code{TRUE}, overwrite any existing
//'   \code{PYTORCH_CUDA_ALLOC_CONF}; default \code{FALSE} only sets when unset.
//' @return The resulting value of \code{PYTORCH_CUDA_ALLOC_CONF} as a string.
//' @keywords internal
//' @export
// [[Rcpp::export]]
std::string resolve_configure_cuda_allocator(bool force = false) {
    capi_require_loaded();
    resolve_value_t* v = resolve_capi_configure_cuda_allocator(force ? 1 : 0);
    capi_check(v);
    ValuePtr guard(v);
    return std::string(resolve_value_as_string(v));
}

// =============================================================================
// resolve_c runtime-loader entry points (see resolve_capi_dynload.*). Called
// from .onLoad / resolve.available() / resolve.install_backend() in R. These are
// the ONLY package functions safe to call before the backend has been loaded.
// =============================================================================

// Load the resolve_c shared library from `path` and bind its symbols. Returns
// TRUE on success; on failure the reason is available via resolve_capi_load_error().
// [[Rcpp::export]]
bool resolve_capi_load_lib(std::string path) {
    return resolve_capi_load(path.c_str()) == 0;
}

// TRUE once resolve_c has been loaded. Cheap; safe before any load.
// [[Rcpp::export]]
bool resolve_capi_is_available() {
    return resolve_capi_available() != 0;
}

// Human-readable message for the most recent resolve_capi_load_lib() failure.
// [[Rcpp::export]]
std::string resolve_capi_load_error() {
    return std::string(resolve_capi_dynload_error());
}
