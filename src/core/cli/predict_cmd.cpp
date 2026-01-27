// RESOLVE CLI - Predict command implementation

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include "resolve/resolve.hpp"

int predict_command(
    const std::string& model_path,
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
    bool use_cuda
) {
    using namespace resolve;

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    if (species_path.empty()) {
        std::cerr << "Error: --species is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Prediction" << std::endl;
    std::cout << "==================" << std::endl;

    // Load model
    std::cout << "Loading model from: " << model_path << std::endl;
    torch::Device device = torch::kCPU;
    if (use_cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    Predictor predictor = Predictor::load(model_path, device);
    const auto& schema = predictor.model()->schema();

    // Set up role mapping
    RoleMapping roles;
    roles.plot_id = plot_id_col;
    roles.species_id = species_id_col;

    if (abundance_col) roles.abundance = *abundance_col;
    if (lon_col) roles.longitude = *lon_col;
    if (lat_col) roles.latitude = *lat_col;
    if (genus_col) roles.genus = *genus_col;
    if (family_col) roles.family = *family_col;

    // Build target specs from schema (we're not training, so these are just placeholders)
    std::vector<TargetSpec> targets;
    for (const auto& target : schema.targets) {
        TargetSpec spec;
        spec.column_name = target.name;
        spec.target_name = target.name;
        spec.task = target.task;
        spec.transform = target.transform;
        spec.num_classes = target.num_classes;
        targets.push_back(spec);
    }

    // Set up dataset configuration based on model
    DatasetConfig dataset_config;
    dataset_config.species_encoding = predictor.model()->species_encoding();
    dataset_config.hash_dim = predictor.model()->config().hash_dim;
    dataset_config.top_k = predictor.model()->config().top_k;

    // Load dataset
    std::cout << "Loading data..." << std::endl;
    std::optional<ResolveDataset> dataset_opt;
    try {
        if (header_path.empty()) {
            dataset_opt = ResolveDataset::from_species_csv(
                species_path, roles, targets, dataset_config
            );
        } else {
            dataset_opt = ResolveDataset::from_csv(
                header_path, species_path, roles, targets, dataset_config
            );
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }
    auto& dataset = *dataset_opt;

    std::cout << "Loaded " << dataset.n_plots() << " plots" << std::endl;

    // Make predictions
    std::cout << "Making predictions..." << std::endl;
    auto predictions = predictor.predict(dataset);

    // Write predictions to CSV
    std::cout << "Writing predictions to: " << output_path << std::endl;
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
        return 1;
    }

    // Write header
    out << "plot_id";
    for (const auto& target : schema.targets) {
        out << "," << target.name;
    }
    out << "\n";

    // Write predictions
    for (size_t i = 0; i < predictions.plot_ids.size(); ++i) {
        out << predictions.plot_ids[i];

        for (const auto& target : schema.targets) {
            auto it = predictions.predictions.find(target.name);
            if (it != predictions.predictions.end()) {
                if (target.task == TaskType::Classification) {
                    out << "," << it->second[i].item<int64_t>();
                } else {
                    out << "," << it->second[i].item<float>();
                }
            } else {
                out << ",NA";
            }
        }
        out << "\n";
    }

    out.close();
    std::cout << "Done! Wrote " << predictions.plot_ids.size() << " predictions." << std::endl;

    return 0;
}
