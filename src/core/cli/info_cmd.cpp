// RESOLVE CLI - Info command implementation

#include <iostream>
#include <string>

#include "resolve/resolve.hpp"

int info_command(const std::string& model_path) {
    using namespace resolve;

    if (model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Model Information" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        // Load model
        auto [model, scalers] = Trainer::load(model_path, torch::kCPU);
        const auto& schema = model->schema();
        const auto& config = model->config();

        // Print schema information
        std::cout << "\nSchema:" << std::endl;
        std::cout << "  Plots: " << schema.n_plots << std::endl;
        std::cout << "  Species: " << schema.n_species << std::endl;
        std::cout << "  Species vocab: " << schema.n_species_vocab << std::endl;
        std::cout << "  Has coordinates: " << (schema.has_coordinates ? "yes" : "no") << std::endl;
        std::cout << "  Has abundance: " << (schema.has_abundance ? "yes" : "no") << std::endl;
        std::cout << "  Has taxonomy: " << (schema.has_taxonomy ? "yes" : "no") << std::endl;

        if (schema.has_taxonomy) {
            std::cout << "  Genera: " << schema.n_genera << " (vocab: " << schema.n_genera_vocab << ")" << std::endl;
            std::cout << "  Families: " << schema.n_families << " (vocab: " << schema.n_families_vocab << ")" << std::endl;
        }

        if (!schema.covariate_names.empty()) {
            std::cout << "  Covariates: ";
            for (size_t i = 0; i < schema.covariate_names.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << schema.covariate_names[i];
            }
            std::cout << std::endl;
        }

        // Print targets
        std::cout << "\nTargets:" << std::endl;
        for (const auto& target : schema.targets) {
            std::cout << "  " << target.name << ": ";
            if (target.task == TaskType::Classification) {
                std::cout << "classification (" << target.num_classes << " classes)";
            } else {
                std::cout << "regression";
                if (target.transform == TransformType::Log1p) {
                    std::cout << " [log1p]";
                }
            }
            std::cout << std::endl;
        }

        // Print model configuration
        std::cout << "\nModel Configuration:" << std::endl;

        std::string encoding_str = "hash";
        if (config.species_encoding == SpeciesEncodingMode::Embed) {
            encoding_str = "embed";
        } else if (config.species_encoding == SpeciesEncodingMode::Sparse) {
            encoding_str = "sparse";
        }
        std::cout << "  Species encoding: " << encoding_str << std::endl;
        std::cout << "  Hash dim: " << config.hash_dim << std::endl;
        std::cout << "  Species embed dim: " << config.species_embed_dim << std::endl;
        std::cout << "  Genus embed dim: " << config.genus_emb_dim << std::endl;
        std::cout << "  Family embed dim: " << config.family_emb_dim << std::endl;
        std::cout << "  Top-k: " << config.top_k << std::endl;
        std::cout << "  Top-k species: " << config.top_k_species << std::endl;
        std::cout << "  Taxonomy slots: " << config.n_taxonomy_slots << std::endl;
        std::cout << "  Dropout: " << config.dropout << std::endl;

        std::cout << "  Hidden dims: [";
        for (size_t i = 0; i < config.hidden_dims.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << config.hidden_dims[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "  Latent dim: " << model->latent_dim() << std::endl;

        // Print parameter count
        int64_t total_params = 0;
        for (const auto& param : model->parameters()) {
            total_params += param.numel();
        }
        std::cout << "\nTotal parameters: " << total_params << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
