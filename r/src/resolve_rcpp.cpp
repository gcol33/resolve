// resolve_rcpp.cpp - Main entry point and module registration
// [[Rcpp::plugins(cpp17)]]

// Include all class wrappers
#include "rcpp_common.hpp"
#include "rcpp_dataset.hpp"
#include "rcpp_encoder.hpp"
#include "rcpp_model.hpp"
#include "rcpp_trainer.hpp"
#include "rcpp_predictor.hpp"

// =============================================================================
// Module exports via Rcpp modules
// =============================================================================

RCPP_MODULE(resolve_module) {
    // SpeciesEncoder
    class_<RSpeciesEncoder>("SpeciesEncoder")
        .constructor<int, int, std::string, std::string, bool, std::string, std::string, int>(
            "Create a SpeciesEncoder")
        .method("fit", &RSpeciesEncoder::fit, "Fit encoder on species data")
        .method("transform", &RSpeciesEncoder::transform, "Transform species data")
        .method("is_fitted", &RSpeciesEncoder::is_fitted, "Check if encoder is fitted")
        .method("hash_dim", &RSpeciesEncoder::hash_dim, "Get hash dimension")
        .method("top_k", &RSpeciesEncoder::top_k, "Get top-k value")
        .method("n_genera", &RSpeciesEncoder::n_genera, "Get number of genera")
        .method("n_families", &RSpeciesEncoder::n_families, "Get number of families")
        .method("n_taxonomy_slots", &RSpeciesEncoder::n_taxonomy_slots, "Get taxonomy slot count")
        .method("uses_explicit_vector", &RSpeciesEncoder::uses_explicit_vector, "Check if using explicit vector")
        .method("n_species_vector", &RSpeciesEncoder::n_species_vector, "Get species vector size")
        .method("n_known_species", &RSpeciesEncoder::n_known_species, "Get known species count")
        .method("save", &RSpeciesEncoder::save, "Save encoder to file")
        ;

    function("SpeciesEncoder_load", &RSpeciesEncoder::load, "Load encoder from file");

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
        ;

    function("ResolveDataset_from_csv", &RResolveDataset::from_csv, "Load dataset from CSV files");

    // ResolveModel
    class_<RResolveModel>("ResolveModel")
        .constructor<List, List>("Create a ResolveModel")
        .method("forward", &RResolveModel::forward, "Forward pass")
        .method("get_latent", &RResolveModel::get_latent, "Get latent representations")
        .method("train", &RResolveModel::train, "Set training mode")
        .method("eval", &RResolveModel::eval, "Set evaluation mode")
        .method("to_device", &RResolveModel::to_device, "Move model to device")
        .method("latent_dim", &RResolveModel::latent_dim, "Get latent dimension")
        ;

    // Trainer
    class_<RTrainer>("Trainer")
        .constructor<RResolveModel&, List>("Create a Trainer")
        .method("prepare_data", &RTrainer::prepare_data, "Prepare training data from tensors")
        .method("prepare_data_pool", &RTrainer::prepare_data_pool, "Prepare training data for rank_pool/transformer modes")
        .method("prepare_data_from_dataset", &RTrainer::prepare_data_from_dataset, "Prepare training data from ResolveDataset")
        .method("fit", &RTrainer::fit, "Train the model")
        .method("save", &RTrainer::save, "Save model checkpoint")
        ;

    // Predictor
    class_<RPredictor>("Predictor")
        .method("predict", &RPredictor::predict, "Make predictions from tensors")
        .method("predict_dataset", &RPredictor::predict_dataset, "Make predictions from ResolveDataset")
        .method("get_embeddings", &RPredictor::get_embeddings, "Get latent embeddings")
        .method("get_genus_embeddings", &RPredictor::get_genus_embeddings, "Get genus embeddings")
        .method("get_family_embeddings", &RPredictor::get_family_embeddings, "Get family embeddings")
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
