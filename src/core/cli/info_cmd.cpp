// RESOLVE CLI - Info command implementation
//
// Prints everything that defines a checkpoint's architecture, so `info` alone
// answers "what model is this?" for every encoder family, not just the MLP.

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "config_report.hpp"

using resolve_cli::model_field_filter;
using resolve_cli::ParsedArgs;
using resolve_cli::print_config_block;

namespace {

void print_name_list(const char* label, const std::vector<std::string>& names) {
    if (names.empty()) return;
    std::cout << label;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << names[i];
    }
    std::cout << std::endl;
}

}  // namespace

int info_command(const ParsedArgs& args) {
    using namespace resolve;

    const std::string model_path = args.get("--model");
    if (model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Model Information" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        // Load model. Trainer::load returns (model, scalers, vocab); `info`
        // only reports schema/config/parameter counts, so the vocab is unused.
        auto [model, scalers, vocab] = Trainer::load(model_path, torch::kCPU);
        (void)vocab;
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

        print_name_list("  Covariates: ", schema.covariate_names);
        print_name_list("  Categoricals: ", schema.categorical_names);
        if (schema.has_categoricals()) {
            std::cout << "  Categorical vocab sizes: ";
            for (size_t i = 0; i < schema.categorical_vocab_sizes.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << schema.categorical_vocab_sizes[i];
            }
            std::cout << std::endl;
            std::cout << "  Categorical embed dim: " << schema.categorical_embed_dim
                      << std::endl;
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

        // The architecture the checkpoint records. Driven by the shared field
        // registry, so a hyperparameter added to ModelConfig or to one of its
        // architecture sub-configs is reported here in the edit that adds it.
        // model_field_filter (config_report.hpp) hides only a row whose feature
        // another field of the same struct switches off -- the six sub-configs
        // this checkpoint's architecture did not select, the mixture's
        // hyperparameters when routing is none, TabM and the parallel branches
        // when disabled, and the head's shape when it has no hidden layers.
        std::cout << "\nModel Configuration:" << std::endl;
        print_config_block(std::cout, "Model", config, /*persisted_only=*/false,
                           model_field_filter(config));

        std::cout << "  Latent dim: " << model->latent_dim() << std::endl;

        // The loading-side DatasetConfig this checkpoint implies -- the same one
        // `resolve predict` rebuilds to encode new data. Driven by the shared
        // field registry, so a loader knob added to DatasetConfig is reported
        // here in the edit that adds it. `selection` is the selection the run
        // APPLIED, not the one it was configured with (issue #113).
        std::cout << "\nData Encoding:" << std::endl;
        print_config_block(std::cout, "Dataset", dataset_config_from_checkpoint(schema, config));

        // The training recipe the checkpoint records. Driven by the same field
        // registry as the archive itself, so a hyperparameter added to
        // TrainConfig is reported here in the edit that adds it; only the rows
        // save_train_config actually persists are shown, since the rest read
        // back as struct defaults rather than as what this run used.
        std::cout << "\nTraining Configuration:" << std::endl;
        print_config_block(std::cout, "Train", Trainer::load_train_config(model_path),
                           /*persisted_only=*/true);

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
