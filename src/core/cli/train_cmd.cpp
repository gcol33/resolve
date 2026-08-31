// RESOLVE CLI - Train command implementation
//
// Reads its values from the ParsedArgs produced by the `train` flag table in
// cli_spec.hpp. Every DatasetConfig / ModelConfig / TrainConfig knob the CLI
// exposes is set here; the flag names below are the ones that table declares,
// and reading a name it does not declare throws rather than silently returning
// a default.

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"

using resolve_cli::ArgError;
using resolve_cli::ParsedArgs;

namespace {

// Parse one --target value: COL[:TYPE[:N][:log1p]].
//
//   area                        regression on `area`
//   area:regression:log1p       regression on log1p(area), inverted on predict
//   habitat:classification:9    9-class classification
//
// Fields after the type are order-free modifiers: an integer sets the class
// count, "log1p" selects the target transform.
resolve::TargetSpec parse_target_spec(const std::string& raw) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t sep = raw.find(':', start);
        if (sep == std::string::npos) {
            fields.push_back(raw.substr(start));
            break;
        }
        fields.push_back(raw.substr(start, sep - start));
        start = sep + 1;
    }

    if (fields.empty() || fields[0].empty()) {
        throw ArgError("--target expects COL[:TYPE[:N][:log1p]], got '" + raw + "'");
    }

    resolve::TargetSpec spec;
    spec.column_name = fields[0];
    spec.target_name = fields[0];

    const std::string type_str = fields.size() > 1 && !fields[1].empty()
                                     ? fields[1]
                                     : std::string("regression");
    try {
        spec.task = resolve::parse_task_type(type_str);
    } catch (const std::exception& e) {
        throw ArgError(std::string("--target ") + raw + ": " + e.what());
    }

    bool saw_log1p = false;
    bool saw_classes = false;
    for (size_t i = 2; i < fields.size(); ++i) {
        const std::string& modifier = fields[i];
        if (modifier.empty()) continue;
        if (modifier == "log1p") {
            saw_log1p = true;
            continue;
        }
        try {
            size_t consumed = 0;
            const int value = std::stoi(modifier, &consumed);
            if (consumed != modifier.size()) throw std::invalid_argument("trailing");
            spec.num_classes = value;
            saw_classes = true;
        } catch (const std::exception&) {
            throw ArgError("--target " + raw + ": unknown modifier '" + modifier +
                           "'. Expected a class count or 'log1p'.");
        }
    }

    if (spec.task == resolve::TaskType::Classification) {
        if (saw_log1p) {
            throw ArgError("--target " + raw +
                           ": log1p is a regression transform and cannot be "
                           "combined with classification");
        }
        if (!saw_classes || spec.num_classes < 2) {
            throw ArgError("--target " + raw +
                           ": classification needs a class count >= 2, e.g. "
                           "habitat:classification:9");
        }
    } else if (saw_log1p) {
        spec.transform = resolve::TransformType::Log1p;
    }

    return spec;
}

void print_metric_tree(
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& tree,
    const std::string& indent) {
    for (const auto& [target_name, metrics] : tree) {
        std::cout << indent << target_name << ":" << std::endl;
        for (const auto& [metric_name, value] : metrics) {
            std::cout << indent << "  " << metric_name << ": " << value << std::endl;
        }
    }
}

void print_cross_validation(const resolve::CrossValidationResult& cv) {
    std::cout << "\nCross-validation (" << cv.n_folds << " folds, "
              << cv.total_time_seconds << "s):" << std::endl;
    for (size_t fold = 0; fold < cv.fold_results.size(); ++fold) {
        std::cout << "  Fold " << (fold + 1) << " (best epoch "
                  << cv.fold_results[fold].best_epoch << "):" << std::endl;
        print_metric_tree(cv.fold_results[fold].final_metrics, "    ");
    }
    std::cout << "  Across folds (mean +/- std):" << std::endl;
    for (const auto& [target_name, metrics] : cv.mean_metrics) {
        std::cout << "    " << target_name << ":" << std::endl;
        for (const auto& [metric_name, mean] : metrics) {
            float stddev = 0.0f;
            auto target_it = cv.std_metrics.find(target_name);
            if (target_it != cv.std_metrics.end()) {
                auto metric_it = target_it->second.find(metric_name);
                if (metric_it != target_it->second.end()) stddev = metric_it->second;
            }
            std::cout << "      " << metric_name << ": " << mean << " +/- "
                      << stddev << std::endl;
        }
    }
}

}  // namespace

int train_command(const ParsedArgs& args) {
    using namespace resolve;

    const std::string header_path = args.get("--header");
    const std::string species_path = args.get("--species");
    const std::string output_path = args.get("--output");

    if (species_path.empty()) {
        std::cerr << "Error: --species is required" << std::endl;
        return 1;
    }

    const auto target_specs = args.get_all("--target");
    if (target_specs.empty()) {
        std::cerr << "Error: At least one --target is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Training" << std::endl;
    std::cout << "================" << std::endl;

    // ---------------------------------------------------------------- roles
    RoleMapping roles;
    roles.plot_id = args.get("--plot-id");
    roles.species_id = args.get("--species-id");

    std::string role_value;
    if (args.get_if_present("--abundance", role_value)) roles.abundance = role_value;
    if (args.get_if_present("--lon", role_value)) roles.longitude = role_value;
    if (args.get_if_present("--lat", role_value)) roles.latitude = role_value;
    if (args.get_if_present("--genus", role_value)) roles.genus = role_value;
    if (args.get_if_present("--family", role_value)) roles.family = role_value;

    roles.covariates = args.get_all("--covariate");
    roles.categoricals = args.get_all("--categorical");

    // Covariates and categoricals are header (plot-level) columns; the
    // single-table species loader has no plot-level row to read them from.
    if (header_path.empty() && (!roles.covariates.empty() || !roles.categoricals.empty())) {
        std::cerr << "Error: --covariate / --categorical name plot-level columns "
                     "and need --header" << std::endl;
        return 1;
    }
    for (const auto& categorical : roles.categoricals) {
        for (const auto& covariate : roles.covariates) {
            if (categorical == covariate) {
                std::cerr << "Error: column '" << categorical
                          << "' is listed as both --covariate and --categorical"
                          << std::endl;
                return 1;
            }
        }
    }
    if (!roles.covariates.empty()) {
        std::cout << "Covariates: ";
        for (size_t i = 0; i < roles.covariates.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << roles.covariates[i];
        }
        std::cout << std::endl;
    }
    if (!roles.categoricals.empty()) {
        std::cout << "Categoricals: ";
        for (size_t i = 0; i < roles.categoricals.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << roles.categoricals[i];
        }
        std::cout << std::endl;
    }

    // -------------------------------------------------------------- targets
    std::vector<TargetSpec> targets;
    targets.reserve(target_specs.size());
    for (const auto& raw : target_specs) {
        TargetSpec spec = parse_target_spec(raw);
        std::cout << "Target: " << spec.column_name << " ("
                  << task_type_to_string(spec.task);
        if (spec.task == TaskType::Classification) {
            std::cout << ", " << spec.num_classes << " classes";
        } else if (spec.transform == TransformType::Log1p) {
            std::cout << ", log1p";
        }
        std::cout << ")" << std::endl;
        targets.push_back(std::move(spec));
    }

    // ------------------------------------------------------- dataset config
    DatasetConfig dataset_config;
    dataset_config.species_encoding =
        parse_species_encoding_mode(args.get("--encoding"));
    dataset_config.hash_dim = args.get_int("--hash-dim");
    dataset_config.top_k = args.get_int("--top-k");
    dataset_config.top_k_species = args.get_int("--top-k-species");
    dataset_config.selection = parse_selection_mode(args.get("--selection"));
    dataset_config.species_budget = args.get_int("--species-budget");
    dataset_config.representation =
        parse_representation_mode(args.get("--representation"));
    dataset_config.normalization =
        parse_normalization_mode(args.get("--normalization"));
    dataset_config.aggregation = parse_aggregation_mode(args.get("--aggregation"));
    dataset_config.use_taxonomy = !args.has("--no-taxonomy");
    dataset_config.track_unknown_fraction = !args.has("--no-unknown-fraction");
    dataset_config.track_unknown_count = args.has("--unknown-count");
    dataset_config.use_cuda_hash = args.has("--use-cuda-hash");
    dataset_config.pool_weighting = parse_pool_weighting(args.get("--pool-weighting"));
    dataset_config.pool_species_cap = args.get_int("--pool-species-cap");

    const bool is_pool_encoder =
        dataset_config.species_encoding == SpeciesEncodingMode::RankPool ||
        dataset_config.species_encoding == SpeciesEncodingMode::Transformer;

    if (dataset_config.use_cuda_hash &&
        dataset_config.species_encoding != SpeciesEncodingMode::Hash) {
        std::cerr << "Error: --use-cuda-hash applies to --encoding hash only"
                  << std::endl;
        return 1;
    }

    // ----------------------------------------------------------------- load
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

    // ---------------------------------------------------------------- seed
    // Seed BEFORE the model is constructed: weight initialization draws from
    // the global torch RNG, so an unseeded process gives two identical
    // invocations different models. The same seed drives the train/test split,
    // the per-epoch shuffle, and the cross-validation folds below.
    const int seed = args.get_int("--seed");
    torch::manual_seed(static_cast<uint64_t>(seed));

    // --------------------------------------------------------- model config
    ModelConfig model_config;
    model_config.species_encoding = dataset_config.species_encoding;
    model_config.hash_dim = dataset_config.hash_dim;
    model_config.top_k = dataset_config.top_k;
    model_config.top_k_species = dataset_config.top_k_species;
    model_config.encoder_architecture =
        parse_encoder_architecture(args.get("--encoder-architecture"));
    model_config.dropout = args.get_float("--dropout");

    // The taxonomy-slot count is a property of the tensors the loader actually
    // built (top_k, doubled for top_bottom selection). Read it off the dataset
    // rather than recomputing the rule here, so the model can never be sized
    // against a different width than the data carries.
    const torch::Tensor& genus_ids = dataset.genus_ids();
    if (genus_ids.defined() && genus_ids.dim() == 2 && genus_ids.size(1) > 0) {
        model_config.n_taxonomy_slots = static_cast<int>(genus_ids.size(1));
    }

    const auto hidden_dims = args.get_list("--hidden-dims");
    if (hidden_dims.empty()) {
        std::cerr << "Error: --hidden-dims needs at least one width, e.g. 256,128"
                  << std::endl;
        return 1;
    }
    model_config.hidden_dims.clear();
    for (const auto& width : hidden_dims) {
        try {
            model_config.hidden_dims.push_back(std::stoll(width));
        } catch (const std::exception&) {
            std::cerr << "Error: --hidden-dims expects integers, got '" << width
                      << "'" << std::endl;
            return 1;
        }
    }

    // Mixture of experts. Parsing the two enums here turns an unknown spelling
    // into a CLI error naming the accepted values, rather than an exception out
    // of the model constructor.
    try {
        model_config.moe_routing = parse_moe_routing_type(args.get("--moe-routing"));
        model_config.moe_placement = parse_moe_placement(args.get("--moe-placement"));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    if (model_config.moe_routing != MoERoutingType::None) {
        model_config.n_experts = args.get_int("--n-experts");
        model_config.moe_top_k = args.get_int("--moe-top-k");
        model_config.moe_noise_std = args.get_float("--moe-noise-std");
        model_config.moe_aux_loss_weight = args.get_float("--moe-aux-loss-weight");

        const auto expert_dims = args.get_list("--expert-hidden-dims");
        model_config.expert_hidden_dims.clear();
        for (const auto& width : expert_dims) {
            try {
                model_config.expert_hidden_dims.push_back(std::stoll(width));
            } catch (const std::exception&) {
                std::cerr << "Error: --expert-hidden-dims expects integers, got '"
                          << width << "'" << std::endl;
                return 1;
            }
        }
    }

    // Transformer / rank_pool knobs. The transformer encoder rejects
    // pooling='cls' with 0 attention layers (the CLS vector would be constant),
    // so validate here for a clean CLI error instead of an exception.
    if (is_pool_encoder) {
        model_config.cover_dropout = args.get_float("--cover-dropout");
    }
    if (dataset_config.species_encoding == SpeciesEncodingMode::Transformer) {
        const std::string transformer_pooling = args.get("--transformer-pooling");
        const int n_attention_layers = args.get_int("--n-attention-layers");
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
        model_config.d_model = args.get_int("--d-model");
        model_config.n_heads = args.get_int("--n-heads");
        model_config.n_attention_layers = n_attention_layers;
        model_config.transformer_ff_dim = args.get_int("--transformer-ff-dim");
        model_config.transformer_pooling = transformer_pooling;
        model_config.transformer_dropout = args.get_float("--transformer-dropout");
    }

    std::cout << "\nCreating model..." << std::endl;
    ResolveModel model(dataset.schema(), model_config);

    // --------------------------------------------------------- train config
    TrainConfig train_config;
    const int requested_batch_size = args.get_int("--batch-size");
    train_config.batch_size = requested_batch_size;
    train_config.batch_size_floor = args.get_int("--batch-size-floor");
    train_config.max_epochs = args.get_int("--max-epochs");
    train_config.patience = args.get_int("--patience");
    train_config.lr = args.get_float("--lr");
    train_config.weight_decay = args.get_float("--weight-decay");
    train_config.loss_config = parse_loss_config_mode(args.get("--loss-config"));
    train_config.lr_scheduler = parse_lr_scheduler_type(args.get("--lr-scheduler"));
    train_config.lr_step_size = args.get_int("--lr-step-size");
    train_config.lr_gamma = args.get_float("--lr-gamma");
    train_config.lr_min = args.get_float("--lr-min");
    train_config.band_threshold = args.get_float("--band-threshold");
    train_config.nca_temperature = args.get_float("--nca-temperature");
    train_config.nca_neighbors = args.get_int("--nca-neighbors");
    train_config.nca_weight = args.get_float("--nca-weight");
    train_config.checkpoint_dir = args.get("--checkpoint-dir");
    train_config.checkpoint_every = args.get_int("--checkpoint-every");
    train_config.use_amp = args.get_switch("--amp", "--no-amp", false);
    train_config.cudnn_benchmark =
        args.get_switch("--cudnn-benchmark", "--no-cudnn-benchmark", true);
    train_config.allow_tf32 = !args.has("--no-tf32");
    train_config.vram_fraction = args.get_float("--vram-fraction");

    train_config.band_thresholds.clear();
    for (const auto& threshold : args.get_list("--band-thresholds")) {
        try {
            train_config.band_thresholds.push_back(std::stof(threshold));
        } catch (const std::exception&) {
            std::cerr << "Error: --band-thresholds expects numbers, got '"
                      << threshold << "'" << std::endl;
            return 1;
        }
    }

    if (train_config.checkpoint_every > 0 && train_config.checkpoint_dir.empty()) {
        std::cerr << "Error: --checkpoint-every needs --checkpoint-dir" << std::endl;
        return 1;
    }

    if (args.has("--cuda") && torch::cuda::is_available()) {
        train_config.device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        train_config.device = torch::kCPU;
        std::cout << "Using CPU" << std::endl;
    }

    // ------------------------------------------------------- prepare + train
    Trainer trainer(model, train_config);
    trainer.prepare_data(dataset, args.get_float("--test-size"), seed);

    // Cross-validation runs first and restores the trainer's weights and split
    // when it finishes, so the model saved below is the same one a run without
    // --cv-folds produces.
    const int cv_folds = args.get_int("--cv-folds");
    if (cv_folds > 0) {
        std::cout << "\nCross-validating..." << std::endl;
        try {
            if (args.has("--cv-spatial")) {
                if (!dataset.schema().has_coordinates) {
                    std::cerr << "Error: --cv-spatial needs coordinates; pass "
                                 "--lon and --lat" << std::endl;
                    return 1;
                }
                SpatialBlockConfig spatial_config;
                spatial_config.lat_size = args.get_float("--cv-lat-size");
                spatial_config.lon_size = args.get_float("--cv-lon-size");
                spatial_config.balance = args.has("--cv-balance");
                print_cross_validation(
                    trainer.cross_validate_spatial(spatial_config, cv_folds, seed));
            } else {
                print_cross_validation(trainer.cross_validate(cv_folds, seed));
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during cross-validation: " << e.what() << std::endl;
            return 1;
        }
    }

    std::cout << "\nTraining..." << std::endl;
    auto result = trainer.fit();

    // ------------------------------------------------------------- report
    std::cout << "\n================" << std::endl;
    std::cout << "Training complete!" << std::endl;
    std::cout << "Best epoch: " << result.best_epoch << std::endl;
    std::cout << "Training time: " << result.train_time_seconds << "s" << std::endl;
    // The batch size the run actually trained at. fit() restores
    // config().batch_size to the requested value before returning, so the
    // effective value is the one the result carries (issue #105).
    std::cout << "Effective batch size: " << result.effective_batch_size;
    if (result.effective_batch_size != requested_batch_size) {
        std::cout << " (requested " << requested_batch_size
                  << ", floor " << train_config.batch_size_floor
                  << ") -- OOM auto-halve fired during training";
    }
    std::cout << std::endl;

    std::cout << "\nFinal metrics:" << std::endl;
    print_metric_tree(result.final_metrics, "  ");

    std::cout << "\nSaving model to: " << output_path << std::endl;
    trainer.save(output_path);

    std::cout << "Done!" << std::endl;
    return 0;
}
