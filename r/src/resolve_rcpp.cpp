// resolve_rcpp.cpp - Main entry point and module registration
// [[Rcpp::plugins(cpp17)]]

// Include all class wrappers
#include "rcpp_common.h"
#include "rcpp_dataset.h"
#include "rcpp_model.h"
#include "rcpp_trainer.h"
#include "rcpp_predictor.h"

#include <cstdlib>
#include <string>
//
// The legacy standalone `SpeciesEncoder` wrapper (rcpp_encoder.h) bound a
// `resolve::SpeciesEncoder` class that has since been split in the C++ core
// into `resolve::RankPoolEncoder` (variable-length plot encoder) and
// `resolve::EmbeddingEncoder` (fixed top-k encoder), neither of which is a
// drop-in replacement for the old unified API (no hash_embedding output,
// no save/load, different state surface). The R-side `resolve.encoder()`
// facade is also a Python-POC mirror; the canonical modern path is
// `resolve.dataset.csv()` which dispatches to `resolve::ResolveDataset::from_csv()`
// and performs encoding inside the C++ engine. The standalone wrapper is
// therefore removed until a full port is done (which would also need to add
// save/load on the new C++ encoders); calling `resolve.encoder()` from R
// raises a clear error pointing at the modern path.

// =============================================================================
// Expose module-managed wrapper classes to non-module Rcpp machinery.
//
// Free `function(name, &Class::factory)` registrations inside RCPP_MODULE() are
// translated by Rcpp into ordinary CppFunction calls, which dispatch through
// the generic Rcpp::wrap()/Rcpp::as(). Without an explicit specialization those
// fail with "cannot convert type to SEXP" for module-managed classes, because
// Rcpp::wrap() does not know how to box a foreign C++ type into an S4 object.
//
// RCPP_EXPOSED_CLASS_NODECL emits the Rcpp::wrap() and Rcpp::as() traits that
// route through the module's S4 representation (the same Reference Class object
// you get from new(.resolve_module$ClassName)), so factory-style functions can
// return wrapper instances by value. The classes themselves are already
// forward-defined via the rcpp_*.h includes above.
// =============================================================================

RCPP_EXPOSED_CLASS_NODECL(RResolveDataset)
RCPP_EXPOSED_CLASS_NODECL(RResolveModel)
RCPP_EXPOSED_CLASS_NODECL(RTrainer)
RCPP_EXPOSED_CLASS_NODECL(RPredictor)

// =============================================================================
// Module exports via Rcpp modules
// =============================================================================

RCPP_MODULE(resolve_module) {
    // ResolveDataset - high-level data loading (mirrors Python ResolveDataset)
    class_<RResolveDataset>("ResolveDataset")
        .method("coordinates", &RResolveDataset::coordinates, "Get coordinate matrix")
        .method("covariates", &RResolveDataset::covariates, "Get covariate matrix")
        .method("hash_embedding", &RResolveDataset::hash_embedding, "Get hash embedding matrix")
        .method("species_ids", &RResolveDataset::species_ids, "Get species IDs matrix")
        .method("species_vector", &RResolveDataset::species_vector, "Get explicit species vector")
        .method("genus_ids", &RResolveDataset::genus_ids, "Get genus IDs matrix")
        .method("family_ids", &RResolveDataset::family_ids, "Get family IDs matrix")
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
        ;

    function("ResolveDataset_from_csv", &RResolveDataset::from_csv, "Load dataset from CSV files");
    function("ResolveDataset_from_species_csv", &RResolveDataset::from_species_csv, "Load dataset from single species CSV");

    // ResolveModel
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

    // Trainer
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
        .method("cross_validate", &RTrainer::cross_validate, "Run k-fold cross-validation")
        .method("cross_validate_spatial", &RTrainer::cross_validate_spatial, "Run spatial block cross-validation")
        .method("predict_from_trainer", &RTrainer::predict_from_trainer, "Make predictions using trainer's model")
        ;

    // Predictor
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
        ;

    function("Predictor_load", &RPredictor::load, "Load predictor from checkpoint");
}

// =============================================================================
// Package initialization
// =============================================================================

// [[Rcpp::export]]
std::string resolve_version() {
    return resolve::VERSION;
}

// Cap the PyTorch CUDA caching allocator at `fraction` of device VRAM.
// fraction must be in (0, 1]; 1.0 disables the cap. device_index = -1L
// uses the current CUDA device. No-op on CPU-only builds or when no
// CUDA device is present.
// [[Rcpp::export]]
void resolve_set_vram_fraction(double fraction, int device_index = -1) {
    resolve::set_vram_fraction(fraction, device_index);
}

//' Configure the PyTorch CUDA caching allocator
//'
//' Set the \code{PYTORCH_CUDA_ALLOC_CONF} environment variable to a
//' platform-aware default. Linux/macOS get \code{expandable_segments:True,}
//' prefixed; on Windows that prefix is intentionally omitted because the
//' cuMemMap-backed expandable-segments allocator is not implemented (libtorch
//' warns \dQuote{expandable_segments not supported on this platform}). The
//' baseline \code{garbage_collection_threshold:0.8,max_split_size_mb:256}
//' helps reduce reserved-but-unallocated fragmentation on both platforms.
//'
//' Mirrors \code{resolve_core.configure_cuda_allocator} in Python, with one
//' caveat: the R/Rcpp binding loads libtorch via the package's shared library,
//' which triggers torch initialization. By the time this function runs the
//' PyTorch CUDA caching allocator has typically already initialized and read
//' its config exactly once, so changes here are best-effort and may not affect
//' the running allocator. To force the default before torch initializes, set
//' \code{Sys.setenv(PYTORCH_CUDA_ALLOC_CONF = "...")} before \code{library(resolve)}.
//'
//' @param force If \code{TRUE}, overwrite any existing
//'   \code{PYTORCH_CUDA_ALLOC_CONF}; default \code{FALSE} only sets when unset.
//' @return The resulting value of \code{PYTORCH_CUDA_ALLOC_CONF} as a string.
//' @keywords internal
//' @export
// [[Rcpp::export]]
std::string resolve_configure_cuda_allocator(bool force = false) {
    std::string base = "garbage_collection_threshold:0.8,max_split_size_mb:256";
#if !defined(_WIN32)
    base = "expandable_segments:True," + base;
#endif

    const char* existing = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (force || existing == nullptr || existing[0] == '\0') {
#if defined(_WIN32)
        _putenv_s("PYTORCH_CUDA_ALLOC_CONF", base.c_str());
#else
        setenv("PYTORCH_CUDA_ALLOC_CONF", base.c_str(), 1);
#endif
        return base;
    }
    return std::string(existing);
}
