// resolve_rcpp.cpp - Module registration + free functions.
//
// The R bindings are a thin client over the `resolve_c` C ABI
// (resolve/resolve_capi.h); no libtorch C++ header is included anywhere under
// r/src/.
// [[Rcpp::plugins(cpp17)]]

#include "rcpp_common.h"
#include "rcpp_dataset.h"
#include "rcpp_fuzzy.h"
#include "rcpp_model.h"
#include "rcpp_trainer.h"
#include "rcpp_predictor.h"
#include "rcpp_suite.h"

// =============================================================================
// Expose module-managed wrapper classes to non-module Rcpp machinery so the
// factory-style free functions (load, for_dataset, ...) can return wrappers by
// value: RCPP_EXPOSED_CLASS_NODECL emits the wrap()/as() traits routing through
// the module's S4 representation. RResolveDataset's are declared in
// rcpp_dataset.h, where the loader first converts one.
// =============================================================================

RCPP_EXPOSED_CLASS_NODECL(RResolveModel)
RCPP_EXPOSED_CLASS_NODECL(RTrainer)
RCPP_EXPOSED_CLASS_NODECL(RPredictor)
RCPP_EXPOSED_CLASS_NODECL(RSuite)
RCPP_EXPOSED_CLASS_NODECL(RFuzzyIndex)

namespace {

// The default of every field of a configuration struct, as the list its
// reader takes: "dataset", "model", "train", "spatial_block", "retrieval",
// "pretrain", "masked_species" or "vae".
List default_config(std::string which) {
    capi_require_loaded();
    return as<List>(value_to_r_owned(resolve_default_config(which.c_str())));
}

// A numeric matrix as a standalone row-major value node.
ValuePtr matrix_value(const NumericMatrix& x) {
    const std::vector<double> buf = row_major<double>(x);
    return owned_value(resolve_value_new_double_matrix(buf.data(), x.nrow(), x.ncol()));
}

// The engine's exact cosine nearest-neighbour search: list(indices, similarities),
// the indices 0-based rows of `reference`.
List nearest_by_cosine(NumericMatrix queries, NumericMatrix reference, double k,
                       bool exclude_self, std::string device) {
    capi_require_loaded();
    const ValuePtr query_rows = matrix_value(queries);
    const ValuePtr reference_rows = matrix_value(reference);
    return as<List>(value_to_r_owned(resolve_nearest_by_cosine(
        query_rows.get(), reference_rows.get(), static_cast<int64_t>(k), exclude_self ? 1 : 0,
        device.c_str())));
}

// A numeric vector as a DOUBLE_ARRAY, a numeric matrix as a row-major
// DOUBLE_MATRIX, NULL as a NULL-kind node.
ValuePtr numeric_value(SEXP x, const char* what) {
    if (Rf_isNull(x)) return owned_value(resolve_value_new_null());
    if (!Rf_isNumeric(x) && !Rf_isLogical(x)) Rcpp::stop(std::string(what) + " must be numeric");
    if (Rf_isMatrix(x)) return matrix_value(NumericMatrix(x));
    const NumericVector v(x);
    return owned_value(resolve_value_new_double_array(v.begin(), v.size()));
}

// Species names resolved against a vocabulary (resolve_names_resolve): a list
// of name, token, route, source, via, distance.
RObject names_resolve(CharacterVector vocabulary, CharacterVector names, SEXP options) {
    capi_require_loaded();
    const ValuePtr vocabulary_v = owned_value(r_to_value(vocabulary));
    const ValuePtr names_v = owned_value(r_to_value(names));
    const ValuePtr options_v = owned_value(Rf_isNull(options)
                                               ? resolve_value_new_null()
                                               : r_list_to_value_map(List(options), "names"));
    return value_to_r_owned(resolve_names_resolve(vocabulary_v.get(), names_v.get(), options_v.get()));
}

// Gap fitting of one target from vectors (resolve_fit_gaps): list(fitted,
// method, report).
List fit_gaps(List target, SEXP raw, SEXP recorded, NumericMatrix species, SEXP coordinates,
              SEXP years, List config, SEXP folds) {
    capi_require_loaded();
    const ValuePtr target_v = owned_value(r_list_to_value_map(target, "target"));
    const ValuePtr raw_v = numeric_value(raw, "raw");
    const ValuePtr recorded_v = numeric_value(recorded, "recorded");
    const ValuePtr species_v = matrix_value(species);
    const ValuePtr coordinates_v = numeric_value(coordinates, "coordinates");
    const ValuePtr years_v = numeric_value(years, "years");
    const ValuePtr config_v = owned_value(r_list_to_value_map(config, "gapFitting"));
    std::vector<int64_t> fold_values;
    if (!Rf_isNull(folds)) {
        const IntegerVector f(folds);
        fold_values.assign(f.begin(), f.end());
    }
    const ValuePtr folds_v = owned_value(
        Rf_isNull(folds) ? resolve_value_new_null()
                         : resolve_value_new_int_array(fold_values.data(), static_cast<int64_t>(fold_values.size())));
    return as<List>(value_to_r_owned(resolve_fit_gaps(target_v.get(), raw_v.get(), recorded_v.get(),
                                                      species_v.get(), coordinates_v.get(),
                                                      years_v.get(), config_v.get(), folds_v.get())));
}

// Gap fitting of a single model's targets from its predictions and the
// dataset's tables (resolve_fit_prediction_gaps).
List fit_prediction_gaps(List header_cols, List species_cols, List columns, List predictions,
                         List requests, List config) {
    capi_require_loaded();
    const ValuePtr species = owned_value(r_table_to_value(species_cols, "species"));
    const ValuePtr header = owned_value(header_cols.size() > 0 ? r_table_to_value(header_cols, "header")
                                                               : resolve_value_new_null());
    const ValuePtr columns_v = owned_value(r_list_to_value_map(columns, "columns"));
    ValuePtr predictions_v = owned_value(resolve_value_new_map());
    map_put(predictions_v.get(), "plot_ids", owned_value(r_to_value(predictions["plot_ids"])));
    ValuePtr outputs = owned_value(resolve_value_new_map());
    const List heads = predictions["outputs"];
    const CharacterVector head_names = heads.names();
    for (R_xlen_t i = 0; i < heads.size(); ++i) {
        map_put(outputs.get(), r_string(head_names[i]), numeric_value(heads[i], "predictions"));
    }
    map_put(predictions_v.get(), "outputs", std::move(outputs));
    ValuePtr requests_v = owned_value(resolve_value_new_list());
    for (R_xlen_t i = 0; i < requests.size(); ++i) {
        list_put(requests_v.get(), owned_value(r_list_to_value_map(requests[i], "targets")));
    }
    const ValuePtr config_v = owned_value(r_list_to_value_map(config, "gapFitting"));
    return as<List>(value_to_r_owned(resolve_fit_prediction_gaps(
        header.get(), species.get(), columns_v.get(), predictions_v.get(), requests_v.get(),
        config_v.get())));
}

// A named list of per-plot numeric vectors as a MAP of DOUBLE_ARRAYs (NULL
// entries left out): the inputs of a terrain correction.
ValuePtr numeric_map(const List& x, const char* what) {
    ValuePtr out = owned_value(resolve_value_new_map());
    if (x.size() == 0) return out;
    const CharacterVector names = x.names();
    for (R_xlen_t i = 0; i < x.size(); ++i) {
        if (Rf_isNull(x[i])) continue;
        map_put(out.get(), r_string(names[i]), numeric_value(x[i], what));
    }
    return out;
}

// Elevation-model descriptors at coordinates (resolve_terrain_descriptors):
// list(elevation, elevation_1km, slope, slope_1km).
List terrain_descriptors(List source, SEXP latitude, SEXP longitude) {
    capi_require_loaded();
    const ValuePtr source_v = owned_value(r_list_to_value_map(source, "terrain source"));
    const ValuePtr lat = numeric_value(latitude, "latitude");
    const ValuePtr lon = numeric_value(longitude, "longitude");
    return as<List>(value_to_r_owned(resolve_terrain_descriptors(source_v.get(), lat.get(), lon.get())));
}

// One tile of an elevation model whole (resolve_terrain_grid):
// list(tile, elevation).
List terrain_grid(List source, std::string tile) {
    capi_require_loaded();
    const ValuePtr source_v = owned_value(r_list_to_value_map(source, "terrain source"));
    return as<List>(value_to_r_owned(resolve_terrain_grid(source_v.get(), tile.c_str())));
}

// Fit a terrain correction and save it (resolve_fit_terrain_correction): the
// fit's report.
List fit_terrain_correction(std::string target, std::string descriptor, List inputs, SEXP truth, List config,
                            std::string path) {
    capi_require_loaded();
    const ValuePtr inputs_v = numeric_map(inputs, "terrain inputs");
    const ValuePtr truth_v = numeric_value(truth, "truth");
    const ValuePtr config_v = owned_value(r_list_to_value_map(config, "config"));
    return as<List>(value_to_r_owned(resolve_fit_terrain_correction(
        target.c_str(), descriptor.c_str(), inputs_v.get(), truth_v.get(), config_v.get(), path.c_str())));
}

// Apply a saved terrain correction (resolve_terrain_correction_apply).
List terrain_correction_apply(std::string path, List inputs, std::string device) {
    capi_require_loaded();
    const ValuePtr inputs_v = numeric_map(inputs, "terrain inputs");
    return as<List>(value_to_r_owned(resolve_terrain_correction_apply(path.c_str(), inputs_v.get(), device.c_str())));
}

// Fit one tile of an elevation grid into a terrain tile pack
// (resolve_fit_terrain_tile): the tile's metrics.
List fit_terrain_tile(std::string grids, std::string tile, List config, std::string path) {
    capi_require_loaded();
    const ValuePtr config_v = owned_value(r_list_to_value_map(config, "config"));
    return as<List>(value_to_r_owned(
        resolve_fit_terrain_tile(grids.c_str(), tile.c_str(), config_v.get(), path.c_str())));
}

// Code elevation grids into one coded terrain field (resolve_encode_terrain_grids):
// a named list of tile -> metrics.
List encode_terrain_grids(std::string grids, List config, std::string path, CharacterVector tiles) {
    capi_require_loaded();
    const ValuePtr config_v = owned_value(r_list_to_value_map(config, "config"));
    const ValuePtr tiles_v = owned_value(tiles.size() == 0 ? resolve_value_new_null() : r_to_value(tiles));
    return as<List>(value_to_r_owned(
        resolve_encode_terrain_grids(grids.c_str(), config_v.get(), path.c_str(), tiles_v.get())));
}

// Merge tile packs into one terrain field file (resolve_assemble_terrain_field).
void assemble_terrain_field(CharacterVector tiles, std::string path) {
    capi_require_loaded();
    const ValuePtr tiles_v = owned_value(r_to_value(tiles));
    capi_check_status(resolve_assemble_terrain_field(tiles_v.get(), path.c_str()));
}

}  // namespace

RCPP_MODULE(resolve_module) {
    class_<RResolveDataset>("ResolveDataset")
        .method("coordinates", &RResolveDataset::coordinates, "Get coordinate matrix")
        .method("covariates", &RResolveDataset::covariates, "Get covariate matrix")
        .method("hash_embedding", &RResolveDataset::hash_embedding, "Get hash embedding matrix")
        .method("species_ids", &RResolveDataset::species_ids, "Get species IDs matrix")
        .method("species_vector", &RResolveDataset::species_vector,
                "Explicit species vector as sparse rows (row_offsets, species, values, n_species)")
        .method("species_vector_dense", &RResolveDataset::species_vector_dense,
                "Explicit species vector as a dense matrix")
        .method("genus_ids", &RResolveDataset::genus_ids, "Get genus IDs matrix")
        .method("family_ids", &RResolveDataset::family_ids, "Get family IDs matrix")
        .method("species_genus_ids", &RResolveDataset::species_genus_ids,
                "Genus code of each species in the vocabulary")
        .method("species_family_ids", &RResolveDataset::species_family_ids,
                "Family code of each species in the vocabulary")
        .method("species_graph", &RResolveDataset::species_graph,
                "Build the species graph a HeterogeneousGNN passes messages on")
        .method("categorical_ids", &RResolveDataset::categorical_ids, "Get categorical covariate codes matrix")
        .method("pool_genus_ids", &RResolveDataset::pool_genus_ids, "Get rank-pool genus IDs matrix")
        .method("pool_family_ids", &RResolveDataset::pool_family_ids, "Get rank-pool family IDs matrix")
        .method("pool_weights", &RResolveDataset::pool_weights, "Get rank-pool per-species weights matrix")
        .method("pool_mask", &RResolveDataset::pool_mask, "Get rank-pool valid-species mask matrix")
        .method("pool_has_cover", &RResolveDataset::pool_has_cover, "Get per-plot has-abundance flag")
        .method("has_pool_data", &RResolveDataset::has_pool_data, "Check if rank-pool tensors are present")
        .method("unknown_fraction", &RResolveDataset::unknown_fraction, "Get unknown species fraction")
        .method("unknown_count", &RResolveDataset::unknown_count, "Get unknown species count")
        .method("species_names", &RResolveDataset::species_names,
                "The species names a load with names did not find as written")
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
        .method("vocabs", &RResolveDataset::vocabs, "Get every fitted vocabulary, for the loader's vocabs")
        .method("pretrain_vae", &RResolveDataset::pretrain_vae,
                "Train a variational autoencoder on the dataset's species vectors")
        ;

    function("ResolveDataset_load", &RResolveDataset::load,
             "Load a dataset from CSV paths or in-memory column lists");
    function("ResolveModel_for_dataset", &RResolveModel::for_dataset,
             "A model sized for a dataset's plots");
    function("default_config", &default_config,
             "The default of every field of a configuration struct");
    function("nearest_by_cosine", &nearest_by_cosine,
             "The k most similar reference rows of each query row by cosine similarity");
    function("fit_gaps", &fit_gaps, "Gap fitting of one target from vectors");
    function("names_resolve", &names_resolve, "Resolve species names against a vocabulary");
    function("fit_prediction_gaps", &fit_prediction_gaps,
             "Gap fitting of a single model's targets from its predictions and tables");
    function("terrain_descriptors", &terrain_descriptors, "Elevation-model descriptors at coordinates");
    function("terrain_grid", &terrain_grid, "One tile of an elevation model whole");
    function("fit_terrain_correction", &fit_terrain_correction, "Fit and save a terrain correction");
    function("terrain_correction_apply", &terrain_correction_apply, "Apply a saved terrain correction");
    function("fit_terrain_tile", &fit_terrain_tile, "Fit one tile of a terrain field");
    function("assemble_terrain_field", &assemble_terrain_field, "Merge terrain tile packs into a field file");
    function("encode_terrain_grids", &encode_terrain_grids, "Code elevation grids into a coded terrain field");

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
        .method("composition_parameters", &RResolveModel::composition_parameters,
                "Species, genus and family tables the species encoder reads composition through")
        .method("config", &RResolveModel::config, "The model's configuration")
        .method("pretrain", &RResolveModel::pretrain,
                "Self-supervised pretraining of the encoder on a dataset's plots")
        .method("set_species_graph", &RResolveModel::set_species_graph,
                "Set the species graph a HeterogeneousGNN passes messages on")
        .method("requires_species_graph", &RResolveModel::requires_species_graph,
                "Whether this architecture reads a species graph")
        .method("has_species_graph", &RResolveModel::has_species_graph,
                "Whether a species graph has been set")
        .method("species_graph_edge_index", &RResolveModel::species_graph_edge_index,
                "Edge endpoints of the species graph, (2, n_edges)")
        .method("species_graph_edge_type", &RResolveModel::species_graph_edge_type,
                "Relation of each species-graph edge")
        ;

    class_<RTrainer>("Trainer")
        .constructor<RResolveModel&, List>("Create a Trainer")
        .method("prepare_data", &RTrainer::prepare_data, "Prepare training data from tensors")
        .method("prepare_data_pool", &RTrainer::prepare_data_pool, "Prepare training data for rank_pool/transformer modes")
        .method("prepare_data_from_dataset", &RTrainer::prepare_data_from_dataset, "Prepare training data from ResolveDataset")
        .method("prepare_data_stopping_fold", &RTrainer::prepare_data_stopping_fold, "Prepare training data from ResolveDataset, holding out the named plots for early stopping")
        .method("fit", &RTrainer::fit, "Train the model")
        .method("resume", &RTrainer::resume, "Continue a run from a checkpoint fit() wrote while training")
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
        .method("effective_batch_size", &RTrainer::effective_batch_size, "Batch size the last fit() actually trained at (post OOM auto-halve)")
        .method("run_metadata", &RTrainer::run_metadata, "Run metadata: the split prepare_data made and, after fit(), its timing, epochs and metrics")
        .method("categorical_vocab", &RTrainer::categorical_vocab, "Get categorical covariate vocabulary captured at prepare_data time")
        .method("cross_validate", &RTrainer::cross_validate, "Run k-fold cross-validation")
        .method("cross_validate_spatial", &RTrainer::cross_validate_spatial, "Run spatial block cross-validation")
        .method("predict_from_trainer", &RTrainer::predict_from_trainer, "Make predictions using trainer's model")
        ;

    class_<RPredictor>("Predictor")
        .method("predict", &RPredictor::predict, "Make predictions from tensors")
        .method("predict_dataset", &RPredictor::predict_dataset,
                "Make predictions from ResolveDataset")
        .method("get_embeddings", &RPredictor::get_embeddings, "Get latent embeddings")
        .method("get_genus_embeddings", &RPredictor::get_genus_embeddings, "Get genus embeddings")
        .method("get_family_embeddings", &RPredictor::get_family_embeddings,
                "Get family embeddings")
        .method("get_species_embeddings", &RPredictor::get_species_embeddings,
                "Get species embeddings")
        .method("optimize_for_inference", &RPredictor::optimize_for_inference,
                "Fuse BatchNorm for faster inference")
        .method("device", &RPredictor::device, "Get current device")
        .method("get_scalers", &RPredictor::get_scalers, "Get fitted scalers")
        .method("categorical_vocab", &RPredictor::categorical_vocab,
                "Get categorical covariate vocabulary loaded from the checkpoint")
        .method("schema", &RPredictor::schema,
                "Get the checkpoint's schema (incl. the fitted species/taxonomy vocabularies)")
        .method("vocabs", &RPredictor::vocabs,
                "Get every training vocabulary, for the *_with_vocabs loaders")
        .method("species_vocab", &RPredictor::species_vocab,
                "Ordered species vocabulary the model was trained with")
        .method("genus_vocab", &RPredictor::genus_vocab,
                "Ordered genus vocabulary the model was trained with")
        .method("family_vocab", &RPredictor::family_vocab,
                "Ordered family vocabulary the model was trained with")
        .method("dataset_config", &RPredictor::dataset_config,
                "The DatasetConfig this checkpoint implies")
        .method("model_config", &RPredictor::model_config, "The checkpoint's ModelConfig")
        .method("set_retrieval", &RPredictor::set_retrieval,
                "Blend predictions with the labels of the nearest reference plots")
        .method("clear_retrieval", &RPredictor::clear_retrieval,
                "Back to the model's own predictions")
        .method("retrieval", &RPredictor::retrieval, "The retrieval settings, or NULL")
        .method("set_plot_totals", &RPredictor::set_plot_totals,
                "Give predict() each plot's combined cover and species count, or withhold them")
        .method("plot_totals", &RPredictor::plot_totals,
                "Whether predict() gives the model each plot's totals");

    class_<RFuzzyIndex>("FuzzyIndex")
        .method("query", &RFuzzyIndex::query, "Best matches of each needle")
        .method("size", &RFuzzyIndex::size, "Number of indexed entries")
        .method("bucket_count", &RFuzzyIndex::bucket_count, "Buckets, the global one included")
        .method("max_supported_distance", &RFuzzyIndex::max_supported_distance,
                "The largest edit distance a query may ask for")
        ;
    function("FuzzyIndex_build", &RFuzzyIndex::build, "Build a fuzzy string index");

    class_<RSuite>("Suite")
        .method("manifest", &RSuite::manifest, "The suite's manifest as a nested list")
        .method("targets", &RSuite::targets, "The suite targets loaded, in manifest order")
        .method("n_encodings", &RSuite::n_encodings, "Distinct input encodings the loaded members need")
        .method("directory", &RSuite::directory, "The directory the suite was loaded from")
        .method("species_vocabularies", &RSuite::species_vocabularies,
                "Target name -> the species vocabulary its names resolve against")
        .method("predict_frame", &RSuite::predict_frame, "Score in-memory tables")
        .method("predict_csv", &RSuite::predict_csv, "Score CSV files")
        ;

    function("Predictor_load", &RPredictor::load, "Load predictor from checkpoint");
    function("Suite_load", &RSuite::load, "Load a model suite from its directory");
    function("Suite_verify", &RSuite::verify,
             "Read a suite's manifest and compare every member file with it");
    function("Suite_seal", &RSuite::seal,
             "Fill a suite manifest's member checksums and sizes, validate and write it");
    function("Trainer_read_checkpoint", &RTrainer::read_checkpoint,
             "Recover a checkpoint's persisted train config or run metadata (as a list)");
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

// Seed libtorch's CPU and CUDA random number generators. The seed arrives as a
// double so the whole range R can hold exactly (0 to 2^53) reaches the engine.
// [[Rcpp::export]]
void resolve_manual_seed(double seed) {
    capi_require_loaded();
    capi_check_status(resolve_capi_manual_seed(static_cast<uint64_t>(seed)));
}

// Pin libtorch's intra-op / inter-op thread pools. intraop / interop <= 0 keep
// libtorch's default. Best-effort; call at load before the first op. Removes
// the worker threads whose teardown join crashes the Rscript.exe launcher on
// Windows.
// [[Rcpp::export]]
void resolve_set_thread_pools(int intraop_threads, int interop_threads = -1) {
    capi_require_loaded();
    capi_check_status(resolve_capi_set_thread_pools(intraop_threads, interop_threads));
}

// Install the Windows crash handler: turn an unhandled native fault into an
// immediate TerminateProcess instead of a JIT-debugger hang or a teardown
// access violation under Rscript.exe. No-op off Windows.
// [[Rcpp::export]]
void resolve_install_crash_handler(int shutdown_exit_code = 0) {
    capi_require_loaded();
    capi_check_status(resolve_capi_install_crash_handler(shutdown_exit_code));
}

// Mark all engine work complete so a subsequent native fault during process
// teardown exits with the shutdown code rather than a crash code. Registered as
// an on-exit finalizer in zzz.R.
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
    const ValuePtr value = owned_value(resolve_capi_configure_cuda_allocator(force ? 1 : 0));
    return std::string(resolve_value_as_string(value.get()));
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
