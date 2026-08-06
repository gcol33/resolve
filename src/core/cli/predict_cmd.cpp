// RESOLVE CLI - Predict command implementation
//
// Reads its values from the ParsedArgs produced by the `predict` flag table in
// cli_spec.hpp.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"

using resolve_cli::ParsedArgs;

namespace {

// Quote a CSV field when it carries a comma, quote, or newline. Class labels
// come from a user's CSV column and may contain any of them; emitting them raw
// would shift every following column (issue #110 item 5).
std::string csv_field(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

}  // namespace

int predict_command(const ParsedArgs& args) {
    using namespace resolve;

    const std::string model_path = args.get("--model");
    const std::string header_path = args.get("--header");
    const std::string species_path = args.get("--species");
    const std::string output_path = args.get("--output");
    const int64_t predict_batch_size = args.get_int64("--predict-batch-size");

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
    if (args.has("--cuda") && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    Predictor predictor =
        Predictor::load(model_path, device, args.get_float("--vram-fraction"));
    const auto& schema = predictor.schema();

    // Set up role mapping
    RoleMapping roles;
    roles.plot_id = args.get("--plot-id");
    roles.species_id = args.get("--species-id");

    std::string role_value;
    if (args.get_if_present("--abundance", role_value)) roles.abundance = role_value;
    if (args.get_if_present("--lon", role_value)) roles.longitude = role_value;
    if (args.get_if_present("--lat", role_value)) roles.latitude = role_value;
    if (args.get_if_present("--genus", role_value)) roles.genus = role_value;
    if (args.get_if_present("--family", role_value)) roles.family = role_value;

    // Covariate / categorical columns. The checkpoint's schema already names
    // the columns the model was fitted on, in the order the encoder expects, so
    // that is the default and a covariate model needs no extra flags to score.
    // --covariate / --categorical override it for a prediction CSV whose columns
    // are named differently; the count must still match, because the encoder's
    // input width is fixed by the checkpoint.
    roles.covariates = args.get_all("--covariate");
    if (roles.covariates.empty()) {
        roles.covariates = schema.covariate_names;
    } else if (roles.covariates.size() != schema.covariate_names.size()) {
        std::cerr << "Error: the checkpoint was trained on "
                  << schema.covariate_names.size() << " covariate(s) but "
                  << roles.covariates.size() << " --covariate flag(s) were given"
                  << std::endl;
        return 1;
    }

    roles.categoricals = args.get_all("--categorical");
    if (roles.categoricals.empty()) {
        roles.categoricals = schema.categorical_names;
    } else if (roles.categoricals.size() != schema.categorical_names.size()) {
        std::cerr << "Error: the checkpoint was trained on "
                  << schema.categorical_names.size() << " categorical(s) but "
                  << roles.categoricals.size()
                  << " --categorical flag(s) were given" << std::endl;
        return 1;
    }

    if (header_path.empty() &&
        (!roles.covariates.empty() || !roles.categoricals.empty())) {
        std::cerr << "Error: this checkpoint uses plot-level covariates, so "
                     "--header is required" << std::endl;
        return 1;
    }

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

    // Rebuild the loading-side DatasetConfig the checkpoint was trained with.
    // Single source of truth (issue #102): species_encoding / hash_dim / top_k
    // come from ModelConfig (they size the model), everything else the loader
    // consumed -- top_k_species, selection, representation, normalization,
    // aggregation, track_unknown_*, use_taxonomy, pool_weighting and the
    // resolved pool_species_cap -- from the schema. Rebuilding only a subset
    // silently re-encoded the data with the struct defaults (#38 fixed
    // pool_weighting alone).
    DatasetConfig dataset_config =
        dataset_config_from_checkpoint(schema, predictor.model()->config());

    // Load dataset IN THE MODEL'S ID NAMESPACE. Every non-hash encoder indexes
    // an embedding table with a code that is a function of the file the vocab
    // was fitted on, so a plain from_csv here re-fits the species / taxonomy /
    // categorical codes and the model looks up the wrong rows -- wrong
    // predictions with no error (issue #102).
    std::cout << "Loading data..." << std::endl;
    const bool reuse_vocabs = schema.has_species_vocab();
    if (!reuse_vocabs) {
        std::cerr << "[resolve] warning: this checkpoint predates "
                     "gcol33/resolve#102 and carries no fitted vocabularies, so "
                     "the species / taxonomy codes are being re-fit from the "
                     "prediction data. That is safe only for a hash-encoded "
                     "model without taxonomy; anything else should be retrained "
                     "or re-saved with a current build."
                  << std::endl;
    }
    const ExternalVocabs vocabs =
        reuse_vocabs ? predictor.external_vocabs() : ExternalVocabs{};

    std::optional<ResolveDataset> dataset_opt;
    try {
        if (header_path.empty()) {
            dataset_opt = reuse_vocabs
                ? ResolveDataset::from_species_csv_with_vocabs(
                      species_path, roles, targets, vocabs, dataset_config)
                : ResolveDataset::from_species_csv(
                      species_path, roles, targets, dataset_config);
        } else {
            dataset_opt = reuse_vocabs
                ? ResolveDataset::from_csv_with_vocabs(
                      header_path, species_path, roles, targets, vocabs, dataset_config)
                : ResolveDataset::from_csv(
                      header_path, species_path, roles, targets, dataset_config);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }
    auto& dataset = *dataset_opt;

    std::cout << "Loaded " << dataset.n_plots() << " plots" << std::endl;

    // Make predictions
    std::cout << "Making predictions";
    if (predict_batch_size == -1) {
        std::cout << " (one-shot, no chunking)..." << std::endl;
    } else {
        std::cout << " (batch_size=" << predict_batch_size << ")..." << std::endl;
    }
    std::optional<ResolvePredictions> predictions_opt;
    try {
        predictions_opt = predictor.predict(
            dataset, /*return_latent=*/false, predict_batch_size);
    } catch (const std::exception& e) {
        // Includes the issue-#102 vocabulary guard: the dataset's integer codes
        // are not the model's, so scoring it would return plausible-looking
        // wrong numbers.
        std::cerr << "Error predicting: " << e.what() << std::endl;
        return 1;
    }
    auto& predictions = *predictions_opt;

    // Write predictions to CSV
    std::cout << "Writing predictions to: " << output_path << std::endl;
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
        return 1;
    }

    // Header. A classification target gets TWO columns (issue #110 item 5):
    //   <target>       the original CSV class label (schema.targets[].class_names,
    //                  persisted since #76), or the integer code when the
    //                  checkpoint has no class vocabulary (pre-#76, or a column
    //                  that was already integer-coded);
    //   <target>_code  always the integer code the model predicted.
    // Emitting the code alone forced every user to reconstruct the mapping by
    // hand; emitting the label alone would lose the code an already-integer
    // column carries.
    out << "plot_id";
    bool wrote_labels = false;
    for (const auto& target : schema.targets) {
        out << "," << csv_field(target.name);
        if (target.task == TaskType::Classification) {
            out << "," << csv_field(target.name + "_code");
            if (!target.class_names.empty()) wrote_labels = true;
        }
    }
    out << "\n";

    // Write predictions
    for (size_t i = 0; i < predictions.plot_ids.size(); ++i) {
        out << csv_field(predictions.plot_ids[i]);

        for (const auto& target : schema.targets) {
            auto it = predictions.predictions.find(target.name);
            if (it == predictions.predictions.end()) {
                out << (target.task == TaskType::Classification ? ",NA,NA" : ",NA");
                continue;
            }
            if (target.task == TaskType::Classification) {
                const int64_t code = it->second[i].item<int64_t>();
                const bool in_vocab =
                    code >= 0 &&
                    code < static_cast<int64_t>(target.class_names.size());
                out << "," << (in_vocab ? csv_field(target.class_names[
                                              static_cast<size_t>(code)])
                                        : std::to_string(code));
                out << "," << code;
            } else {
                out << "," << it->second[i].item<float>();
            }
        }
        out << "\n";
    }

    out.close();
    std::cout << "Done! Wrote " << predictions.plot_ids.size() << " predictions."
              << std::endl;
    if (wrote_labels) {
        std::cout << "Classification targets are written as two columns: "
                     "'<target>' (class label) and '<target>_code' (integer code)."
                  << std::endl;
    }

    return 0;
}
