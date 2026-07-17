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
    int batch_size_floor,
    int max_epochs,
    int patience,
    float lr,
    float test_size,
    bool use_cuda,
    float vram_fraction,
    const std::string& pool_weighting,
    int d_model,
    int n_heads,
    int n_attention_layers,
    const std::string& transformer_pooling
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

    // Set up dataset configuration. Reject an unknown encoding rather than
    // silently defaulting to hash (which would train the wrong model).
    DatasetConfig dataset_config;
    if (species_encoding == "hash") {
        dataset_config.species_encoding = SpeciesEncodingMode::Hash;
    } else if (species_encoding == "embed") {
        dataset_config.species_encoding = SpeciesEncodingMode::Embed;
    } else if (species_encoding == "sparse") {
        dataset_config.species_encoding = SpeciesEncodingMode::Sparse;
    } else if (species_encoding == "rank_pool") {
        dataset_config.species_encoding = SpeciesEncodingMode::RankPool;
    } else if (species_encoding == "transformer") {
        dataset_config.species_encoding = SpeciesEncodingMode::Transformer;
    } else {
        std::cerr << "Error: unknown --encoding '" << species_encoding
                  << "'. Valid values: hash, embed, sparse, rank_pool, transformer"
                  << std::endl;
        return 1;
    }
    dataset_config.hash_dim = hash_dim;
    dataset_config.top_k = top_k;

    // Pool weighting applies to rank_pool / transformer encoders.
    if (dataset_config.species_encoding == SpeciesEncodingMode::RankPool ||
        dataset_config.species_encoding == SpeciesEncodingMode::Transformer) {
        if (pool_weighting == "binary") {
            dataset_config.pool_weighting = PoolWeighting::Binary;
        } else if (pool_weighting == "abundance") {
            dataset_config.pool_weighting = PoolWeighting::Abundance;
        } else if (pool_weighting == "log1p") {
            dataset_config.pool_weighting = PoolWeighting::Log1p;
        } else if (pool_weighting == "norm") {
            dataset_config.pool_weighting = PoolWeighting::Norm;
        } else if (pool_weighting == "rank") {
            dataset_config.pool_weighting = PoolWeighting::Rank;
        } else {
            std::cerr << "Error: unknown --pool-weighting '" << pool_weighting
                      << "'. Valid values: binary, abundance, log1p, norm, rank"
                      << std::endl;
            return 1;
        }
    }

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

    // Transformer / rank_pool knobs. The transformer encoder rejects
    // pooling='cls' with 0 attention layers (the CLS vector would be constant),
    // so validate here for a clean CLI error instead of an exception.
    if (dataset_config.species_encoding == SpeciesEncodingMode::Transformer) {
        if (transformer_pooling != "attention" && transformer_pooling != "cls") {
            std::cerr << "Error: unknown --transformer-pooling '" << transformer_pooling
                      << "'. Valid values: attention, cls" << std::endl;
            return 1;
        }
        if (transformer_pooling == "cls" && n_attention_layers < 1) {
            std::cerr << "Error: --transformer-pooling cls requires "
                         "--n-attention-layers >= 1" << std::endl;
            return 1;
        }
        model_config.d_model = d_model;
        model_config.n_heads = n_heads;
        model_config.n_attention_layers = n_attention_layers;
        model_config.transformer_pooling = transformer_pooling;
    }

    // Create model
    std::cout << "\nCreating model..." << std::endl;
    ResolveModel model(dataset.schema(), model_config);

    // Set up training configuration
    TrainConfig train_config;
    train_config.batch_size = batch_size;
    train_config.batch_size_floor = batch_size_floor;
    train_config.max_epochs = max_epochs;
    train_config.patience = patience;
    train_config.lr = lr;
    train_config.vram_fraction = vram_fraction;

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
    // Show effective batch size: Trainer::fit mutates train_config.batch_size
    // to the post-halve value when the OOM auto-halve retry fires. Surface
    // it here so the operator can see whether a fallback run was used.
    if (trainer.config().batch_size != batch_size) {
        std::cout << "Effective batch size: " << trainer.config().batch_size
                  << " (requested " << batch_size
                  << ", floor " << trainer.config().batch_size_floor
                  << ") -- OOM auto-halve fired during training"
                  << std::endl;
    } else {
        std::cout << "Effective batch size: " << trainer.config().batch_size << std::endl;
    }

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
