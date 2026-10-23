// RESOLVE CLI - Info command implementation
//
// Prints a checkpoint's description, the document its JSON sidecar holds
// (checkpoint_description): the run, the schema, the model, the loader
// settings and the training recipe, so `info` alone answers "what model is
// this?" for every encoder family. The latent width and the parameter count
// come from the model the checkpoint builds.

#include <iostream>
#include <string>

#include "resolve/checkpoint.hpp"
#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "config_report.hpp"

using resolve_cli::ParsedArgs;

int info_suite_command(const ParsedArgs& args);

int info_command(const ParsedArgs& args) {
    using namespace resolve;

    if (args.has("--suite")) {
        if (args.has("--model")) {
            std::cerr << "Error: pass --model or --suite, not both" << std::endl;
            return 1;
        }
        return info_suite_command(args);
    }

    const std::string model_path = args.get("--model");
    if (model_path.empty()) {
        std::cerr << "Error: --model or --suite is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Model Information" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        const json::Value description = describe_checkpoint(model_path);
        auto [model, scalers, vocab] = Trainer::load(model_path, torch::kCPU);
        (void)scalers;
        (void)vocab;

        // The run metadata: the description's top-level scalars.
        std::cout << "\nRun:" << std::endl;
        resolve_cli::print_block(std::cout, description, "", [&description](const char* name) {
            const json::Value& v = description.at(name);
            return !v.is_object() && !v.is_array();
        });

        json::Value schema = description.at("schema");
        const json::Value dataset_config = schema.at("dataset_config");
        schema.erase("dataset_config");
        std::cout << "\nSchema:" << std::endl;
        resolve_cli::print_block(std::cout, schema, "");

        // Every sub-config row the selected architecture does not read, and the
        // rows another field switches off, are hidden (model_field_filter).
        std::cout << "\nModel Configuration:" << std::endl;
        resolve_cli::print_block(std::cout, description.at("model_config"), "Model",
                                 resolve_cli::model_field_filter(model->config()));
        std::cout << "  Latent dim: " << model->latent_dim() << std::endl;

        // The loader settings `resolve predict` encodes new plots with;
        // `selection` is the selection the training load applied.
        std::cout << "\nData Encoding:" << std::endl;
        resolve_cli::print_block(std::cout, dataset_config, "Dataset");

        std::cout << "\nTraining Configuration:" << std::endl;
        resolve_cli::print_block(std::cout, description.at("train_config"), "Train");

        const json::Value& metrics = description.at("final_metrics");
        if (metrics.size() > 0) {
            std::cout << "\nFinal metrics:" << std::endl;
            resolve_cli::print_block(std::cout, metrics, "");
        }

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
