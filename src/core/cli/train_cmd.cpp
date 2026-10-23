// RESOLVE CLI - Train command implementation
//
// Reads its values from the ParsedArgs produced by the `train` flag table in
// cli_spec.hpp. The DatasetConfig, ModelConfig, TrainConfig and architecture
// fields come back through the field-registry readers of config_flags.hpp;
// the flags below are the ones that table declares, and reading a name it
// does not declare throws rather than silently returning a default.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "cli_common.hpp"
#include "config_flags.hpp"

using resolve_cli::ArgError;
using resolve_cli::ParsedArgs;
using resolve_cli::trim_whitespace;

namespace {

// The plot IDs of a --stopping-plots file: one per line, surrounding
// whitespace dropped, blank lines skipped.
std::vector<std::string> read_plot_id_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw ArgError("--stopping-plots: cannot open '" + path + "'");
    }
    std::vector<std::string> ids;
    std::string line;
    while (std::getline(in, line)) {
        std::string id = trim_whitespace(line);
        if (!id.empty()) ids.push_back(std::move(id));
    }
    return ids;
}

// Parse one --target value: COL[:TYPE[:N][:log1p][:mask]].
//
//   area                        regression on `area`
//   area:regression:log1p       regression on log1p(area), inverted on predict
//   habitat:classification:9    9-class classification
//   habitat:classification:9:mask
//                               plots without a habitat label stay in the fit
//
// Fields after the type are order-free modifiers: an integer sets the class
// count, "log1p" selects the target transform, and "drop" (the default) or
// "mask" sets what happens to a plot whose label for this target is missing.
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
        throw ArgError("--target expects COL[:TYPE[:N][:log1p][:mask]], got '" + raw + "'");
    }

    resolve::TargetSpec spec;
    spec.column_name = fields[0];
    spec.target_name = fields[0];

    const std::string type_str = fields.size() > 1 && !fields[1].empty()
                                     ? fields[1]
                                     : std::string("regression");
    try {
        spec.task = resolve::enum_from_name<resolve::TaskType>(type_str);
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
        if (modifier == "drop" || modifier == "mask") {
            spec.missing = resolve::enum_from_name<resolve::TargetMissingPolicy>(modifier);
            continue;
        }
        try {
            spec.num_classes = resolve_cli::parse_int_value("--target", modifier);
            saw_classes = true;
        } catch (const ArgError&) {
            throw ArgError("--target " + raw + ": unknown modifier '" + modifier +
                           "'. Expected a class count, 'log1p', 'drop' or 'mask'.");
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
    const RoleMapping roles = resolve_cli::read_roles(args);

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
        std::cout << "Target: " << spec.column_name << " (" << enum_to_name(spec.task);
        if (spec.task == TaskType::Classification) {
            std::cout << ", " << spec.num_classes << " classes";
        } else if (spec.transform == TransformType::Log1p) {
            std::cout << ", log1p";
        }
        if (spec.missing == TargetMissingPolicy::Mask) {
            std::cout << ", missing labels masked";
        }
        std::cout << ")" << std::endl;
        targets.push_back(std::move(spec));
    }

    // -------------------------------------------------------------- configs
    // Every field from its registry-generated flag; a field whose flag is not
    // given keeps its struct default.
    DatasetConfig dataset_config;
    resolve_cli::read_top_level_flags(dataset_config, args);
    ModelConfig model_config;
    resolve_cli::read_top_level_flags(model_config, args);
    resolve_cli::read_architecture_flags(model_config, args);
    TrainConfig train_config;
    resolve_cli::read_top_level_flags(train_config, args);
    {
        const resolve_cli::FlagContext context{dataset_config, model_config, train_config};
        resolve_cli::check_flag_gates(dataset_config, args, context);
        resolve_cli::check_flag_gates(model_config, args, context);
        resolve_cli::check_flag_gates(train_config, args, context);
    }

    // The parallel block's branches are variable-length, so they have their own
    // repeatable grammar rather than a value flag per field.
    for (const auto& spec : args.get_all("--parallel-branch")) {
        model_config.parallel_layers.branches.push_back(
            resolve_cli::parse_parallel_branch(spec));
    }
    if (model_config.parallel_layers.enabled &&
        model_config.parallel_layers.branches.empty()) {
        std::cerr << "Error: --parallel-enabled needs at least one "
                     "--parallel-branch, e.g. --parallel-branch 256,128"
                  << std::endl;
        return 1;
    }
    if (model_config.hidden_dims.empty()) {
        std::cerr << "Error: --hidden-dims needs at least one width, e.g. 256,128"
                  << std::endl;
        return 1;
    }
    train_config.device = resolve_cli::choose_device(args);

    // ----------------------------------------------------------------- load
    std::cout << "\nLoading data..." << std::endl;
    ResolveDataset dataset;
    try {
        dataset = resolve_cli::load_dataset(header_path, species_path, roles, targets,
                                            dataset_config);
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

    // ---------------------------------------------------------------- model
    // The knobs that size the model to the data come from the dataset config,
    // and the taxonomy-slot count is the width the loader built (top_k,
    // doubled for top_bottom selection), so the model is never sized against a
    // different width than the data carries.
    model_config.species_encoding = dataset_config.species_encoding;
    model_config.hash_dim = dataset_config.hash_dim;
    model_config.top_k = dataset_config.top_k;
    model_config.top_k_species = dataset_config.top_k_species;
    const torch::Tensor& genus_ids = dataset.genus_ids();
    if (genus_ids.defined() && genus_ids.dim() == 2 && genus_ids.size(1) > 0) {
        model_config.n_taxonomy_slots = static_cast<int>(genus_ids.size(1));
    }

    std::cout << "\nCreating model..." << std::endl;
    ResolveModel model(dataset.schema(), model_config);
    const int requested_batch_size = train_config.batch_size;

    // ------------------------------------------------------- prepare + train
    Trainer trainer(model, train_config);
    std::string stopping_plots_path;
    if (args.get_if_present("--stopping-plots", stopping_plots_path)) {
        if (args.has("--test-size")) {
            std::cerr << "Error: pass --test-size or --stopping-plots, not both"
                      << std::endl;
            return 1;
        }
        const auto stopping_ids = read_plot_id_list(stopping_plots_path);
        try {
            trainer.prepare_data(dataset, dataset.plot_indices(stopping_ids), seed);
            std::cout << "Stopping fold: " << stopping_ids.size() << " plots from "
                      << stopping_plots_path << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: --stopping-plots: " << e.what() << std::endl;
            return 1;
        }
    } else {
        trainer.prepare_data(dataset, args.get_float("--test-size"), seed);
    }

    std::string resume_path;
    const bool resume = args.get_if_present("--resume", resume_path);

    // Cross-validation runs first and restores the trainer's weights and split
    // when it finishes, so the model saved below is the same one a run without
    // --cv-folds produces.
    const int cv_folds = args.get_int("--cv-folds");
    if (resume && cv_folds > 0) {
        std::cerr << "Error: pass --resume or --cv-folds, not both" << std::endl;
        return 1;
    }
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

    TrainResult result;
    if (resume) {
        std::cout << "\nResuming from " << resume_path << "..." << std::endl;
        try {
            result = trainer.resume(resume_path);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: --resume: " << e.what() << std::endl;
            return 1;
        }
        std::cout << "Resumed at epoch " << result.resumed_from_epoch << std::endl;
    } else {
        std::cout << "\nTraining..." << std::endl;
        result = trainer.fit();
    }

    // ------------------------------------------------------------- report
    std::cout << "\n================" << std::endl;
    std::cout << "Training complete!" << std::endl;
    std::cout << "Best epoch: " << result.best_epoch << std::endl;
    std::cout << "Training time: " << result.train_time_seconds << "s" << std::endl;
    // The batch size the run actually trained at. fit() restores
    // config().batch_size to the requested value before returning, so the
    // effective value is the one the result carries.
    std::cout << "Effective batch size: " << result.effective_batch_size;
    if (result.effective_batch_size != requested_batch_size) {
        std::cout << " (requested " << requested_batch_size
                  << ", floor " << train_config.batch_size_floor
                  << ") -- OOM auto-halve fired during training";
    }
    std::cout << std::endl;

    std::cout << "\nFinal metrics:" << std::endl;
    print_metric_tree(result.final_metrics, "  ");

    // Saved with the run metadata, so the checkpoint records how the plots
    // were split, and the JSON sidecar beside it says so in plain text.
    std::cout << "\nSaving model to: " << output_path << std::endl;
    trainer.save(output_path, &trainer.run_metadata());

    std::cout << "Done!" << std::endl;
    return 0;
}
