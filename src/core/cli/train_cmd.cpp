// RESOLVE CLI - Train command implementation

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include "resolve/resolve.hpp"

int train_command(
    const std::string& header_path,
    const std::string& species_path,
    const std::string& output_path,
    const std::string& plot_id_col,
    const std::string& species_id_col,
    const std::optional<std::string>& abundance_col,
    const std::optional<std::string>& lon_col,
    const std::optional<std::string>& lat_col,
    const std::optional<std::string>& genus_col,
    const std::optional<std::string>& family_col,
    const std::vector<std::string>& target_cols,
    const std::vector<std::string>& target_types,
    const std::string& species_encoding,
    int hash_dim,
    int top_k,
    int batch_size,
    int max_epochs,
    int patience,
    float lr,
    float test_size,
    bool use_cuda
) {
    using namespace resolve;

    // Validate required arguments
    if (species_path.empty()) {
        std::cerr << "Error: --species is required" << std::endl;
        return 1;
    }

    if (target_cols.empty()) {
        std::cerr << "Error: At least one --target is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Training" << std::endl;
    std::cout << "================" << std::endl;

    // Set up role mapping
    RoleMapping roles;
    roles.plot_id = plot_id_col;
    roles.species_id = species_id_col;

    if (abundance_col) roles.abundance = *abundance_col;
    if (lon_col) roles.longitude = *lon_col;
    if (lat_col) roles.latitude = *lat_col;
    if (genus_col) roles.genus = *genus_col;
    if (family_col) roles.family = *family_col;

    // Parse target specifications
    std::vector<TargetSpec> targets;
    for (size_t i = 0; i < target_cols.size(); ++i) {
        TargetSpec spec;
        spec.column_name = target_cols[i];
        spec.target_name = target_cols[i];

        const std::string& type_str = target_types[i];
        if (type_str.find("classification") != std::string::npos) {
            spec.task = TaskType::Classification;
            // Parse number of classes: classification:9
            auto pos = type_str.find(':');
            if (pos != std::string::npos) {
                spec.num_classes = std::stoi(type_str.substr(pos + 1));
            }
        } else {
            spec.task = TaskType::Regression;
            if (type_str.find("log1p") != std::string::npos) {
                spec.transform = TransformType::Log1p;
            }
        }

        targets.push_back(spec);
        std::cout << "Target: " << spec.column_name
                  << " (" << (spec.task == TaskType::Classification ? "classification" : "regression") << ")"
                  << std::endl;
    }

    // Set up dataset configuration
    DatasetConfig dataset_config;
    if (species_encoding == "embed") {
        dataset_config.species_encoding = SpeciesEncodingMode::Embed;
    } else if (species_encoding == "sparse") {
        dataset_config.species_encoding = SpeciesEncodingMode::Sparse;
    } else {
        dataset_config.species_encoding = SpeciesEncodingMode::Hash;
    }
    dataset_config.hash_dim = hash_dim;
    dataset_config.top_k = top_k;

    // Load dataset
    std::cout << "\nLoading data..." << std::endl;
    ResolveDataset dataset;
    try {
        if (header_path.empty()) {
            dataset = ResolveDataset::from_species_csv(
                species_path, roles, targets, dataset_config
            );
        } else {
            dataset = ResolveDataset::from_csv(
                header_path, species_path, roles, targets, dataset_config
            );
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loaded " << dataset.n_plots() << " plots" << std::endl;
    std::cout << "Species vocabulary size: " << dataset.schema().n_species_vocab << std::endl;
    if (dataset.schema().has_taxonomy) {
        std::cout << "Genera: " << dataset.schema().n_genera << std::endl;
        std::cout << "Families: " << dataset.schema().n_families << std::endl;
    }

    // Set up model configuration
    ModelConfig model_config;
    model_config.species_encoding = dataset_config.species_encoding;
    model_config.hash_dim = hash_dim;
    model_config.top_k = top_k;

    // Create model
    std::cout << "\nCreating model..." << std::endl;
    ResolveModel model(dataset.schema(), model_config);

    // Set up training configuration
    TrainConfig train_config;
    train_config.batch_size = batch_size;
    train_config.max_epochs = max_epochs;
    train_config.patience = patience;
    train_config.lr = lr;

    if (use_cuda && torch::cuda::is_available()) {
        train_config.device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        train_config.device = torch::kCPU;
        std::cout << "Using CPU" << std::endl;
    }

    // Create trainer and prepare data
    Trainer trainer(model, train_config);
    trainer.prepare_data(dataset, test_size);

    // Train
    std::cout << "\nTraining..." << std::endl;
    auto result = trainer.fit();

    // Print results
    std::cout << "\n================" << std::endl;
    std::cout << "Training complete!" << std::endl;
    std::cout << "Best epoch: " << result.best_epoch << std::endl;
    std::cout << "Training time: " << result.train_time_seconds << "s" << std::endl;

    std::cout << "\nFinal metrics:" << std::endl;
    for (const auto& [target_name, metrics] : result.final_metrics) {
        std::cout << "  " << target_name << ":" << std::endl;
        for (const auto& [metric_name, value] : metrics) {
            std::cout << "    " << metric_name << ": " << value << std::endl;
        }
    }

    // Save model
    std::cout << "\nSaving model to: " << output_path << std::endl;
    trainer.save(output_path);

    std::cout << "Done!" << std::endl;
    return 0;
}
