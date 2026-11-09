// RESOLVE CLI - Predict command implementation
//
// Reads its values from the ParsedArgs produced by the `predict` flag table in
// cli_spec.hpp.

#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <optional>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "cli_common.hpp"
#include "config_flags.hpp"
#include "csv_output.hpp"
#include "gap_flags.hpp"
#include "names_flags.hpp"

using resolve_cli::csv_field;
using resolve_cli::ParsedArgs;

int predict_suite_command(const ParsedArgs& args);

int predict_command(const ParsedArgs& args) {
    using namespace resolve;

    if (args.has("--suite")) {
        if (args.has("--model")) {
            std::cerr << "Error: pass --model or --suite, not both" << std::endl;
            return 1;
        }
        for (const char* model_only : {"--retrieval-header", "--retrieval-species", "--retrieval-k",
                                       "--retrieval-alpha", "--retrieval-temperature",
                                       "--gap-target"}) {
            if (args.has(model_only)) {
                std::cerr << "Error: " << model_only << " applies to --model only" << std::endl;
                return 1;
            }
        }
        return predict_suite_command(args);
    }
    for (const char* suite_only : {"--target", "--members", "--no-verify", "--terrain", "--terrain-grids",
                                   "--terrain-elevation", "--terrain-elevation-1km", "--terrain-slope",
                                   "--terrain-slope-1km", "--location-uncertainty"}) {
        if (args.has(suite_only)) {
            std::cerr << "Error: " << suite_only << " applies to --suite only" << std::endl;
            return 1;
        }
    }

    const std::string model_path = args.get("--model");
    const std::string header_path = args.get("--header");
    const std::string species_path = args.get("--species");
    const std::string output_path = args.get("--output");
    const int64_t predict_batch_size = args.get_int64("--predict-batch-size");
    const bool write_probabilities = args.has("--probabilities");

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: --model or --suite is required" << std::endl;
        return 1;
    }

    if (species_path.empty()) {
        std::cerr << "Error: --species is required" << std::endl;
        return 1;
    }

    std::optional<GapFittingConfig> gap_fitting;
    std::optional<SpeciesNames> names;
    try {
        gap_fitting = resolve_cli::read_gap_fitting(args);
        names = resolve_cli::read_species_names(args);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    const std::string retrieval_header = args.get("--retrieval-header");
    const std::string retrieval_species = args.get("--retrieval-species");
    if (retrieval_species.empty()) {
        for (const char* flag : {"--retrieval-header", "--retrieval-k", "--retrieval-alpha",
                                 "--retrieval-temperature"}) {
            if (args.has(flag)) {
                std::cerr << "Error: " << flag << " needs --retrieval-species" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "RESOLVE Prediction" << std::endl;
    std::cout << "==================" << std::endl;

    // Load model
    std::cout << "Loading model from: " << model_path << std::endl;
    const torch::Device device = resolve_cli::choose_device(args);
    Predictor predictor =
        Predictor::load(model_path, device, args.get_float("--vram-fraction"));
    const auto& schema = predictor.schema();

    RoleMapping roles = resolve_cli::read_roles(args);

    // Covariate / categorical columns. The checkpoint's schema already names
    // the columns the model was fitted on, in the order the encoder expects, so
    // that is the default and a covariate model needs no extra flags to score.
    // --covariate / --categorical override it for a prediction CSV whose columns
    // are named differently; the count must still match, because the encoder's
    // input width is fixed by the checkpoint.
    if (roles.covariates.empty()) {
        roles.covariates = schema.covariate_names;
    } else if (roles.covariates.size() != schema.covariate_names.size()) {
        std::cerr << "Error: the checkpoint was trained on "
                  << schema.covariate_names.size() << " covariate(s) but "
                  << roles.covariates.size() << " --covariate flag(s) were given"
                  << std::endl;
        return 1;
    }

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

    // The plots to score carry no answer, so the loader is asked for no target:
    // a target it was asked for would have to be a column of the file, and a
    // plot missing it would be dropped from the output.
    const std::vector<TargetSpec> targets;

    // The loader settings the checkpoint was trained with.
    const DatasetConfig dataset_config =
        dataset_config_from_checkpoint(schema, predictor.model()->config());

    // Load dataset IN THE MODEL'S ID NAMESPACE. Every non-hash encoder indexes
    // an embedding table with a code that is a function of the file the vocab
    // was fitted on, so a plain from_csv here re-fits the species / taxonomy /
    // categorical codes and the model looks up the wrong rows -- wrong
    // predictions with no error.
    std::cout << "Loading data..." << std::endl;
    const bool reuse_vocabs = schema.has_species_vocab();
    if (!reuse_vocabs) {
        std::cerr << "[resolve] warning: this checkpoint carries no fitted "
                     "vocabularies, only their sizes, so "
                     "the species / taxonomy codes are being re-fit from the "
                     "prediction data. That is safe only for a hash-encoded "
                     "model without taxonomy; anything else should be retrained "
                     "or re-saved with a current build."
                  << std::endl;
    }
    const ExternalVocabs vocabs =
        reuse_vocabs ? predictor.external_vocabs() : ExternalVocabs{};
    if (names && !reuse_vocabs) {
        std::cerr << "Error: --names needs a checkpoint that carries its species vocabulary"
                  << std::endl;
        return 1;
    }
    const SpeciesNames* resolve_names = names ? &*names : nullptr;

    std::optional<ResolveDataset> dataset_opt;
    try {
        dataset_opt = resolve_cli::load_dataset(header_path, species_path, roles, targets,
                                                dataset_config,
                                                reuse_vocabs ? &vocabs : nullptr, resolve_names);
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }
    auto& dataset = *dataset_opt;

    std::cout << "Loaded " << dataset.n_plots() << " plots" << std::endl;
    if (names) {
        std::cout << "Names: " << resolve_cli::names_summary(dataset.species_names()) << std::endl;
    }

    // Retrieval: the reference plots, read with the model's targets in its
    // namespace (their class labels coded as at training), blend every
    // prediction with the labels of its nearest reference plots.
    if (!retrieval_species.empty()) {
        try {
            RetrievalConfig retrieval;
            resolve_cli::read_config_flags(
                retrieval, args, resolve_cli::FlagNaming::members(resolve_cli::kRetrievalConfigFlags));
            std::vector<TargetSpec> reference_targets = target_specs_from_schema(schema);
            // A reference plot unlabelled for one target still serves the others.
            for (auto& spec : reference_targets) spec.missing = TargetMissingPolicy::Mask;
            const ResolveDataset reference = resolve_cli::load_dataset(
                retrieval_header, retrieval_species, roles, reference_targets, dataset_config,
                reuse_vocabs ? &vocabs : nullptr, resolve_names);
            predictor.set_retrieval(reference, retrieval, predict_batch_size);
            std::cout << "Retrieval: " << reference.n_plots() << " reference plots, k = "
                      << retrieval.k_neighbors << ", alpha = " << retrieval.alpha << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error setting up retrieval: " << e.what() << std::endl;
            return 1;
        }
    }

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
        // Includes the vocabulary guard: the dataset's integer codes
        // are not the model's, so scoring it would return plausible-looking
        // wrong numbers.
        std::cerr << "Error predicting: " << e.what() << std::endl;
        return 1;
    }
    auto& predictions = *predictions_opt;

    // Gap fitting: each --recorded target from the dataset's own complete
    // plots, read from the same tables in the predictions' plot order.
    // Without it, each --recorded target says which plots record it.
    std::vector<GapFit> gap_fits;
    std::vector<GapFitRequest> gap_requests;
    std::vector<std::pair<std::string, torch::Tensor>> recorded_flags;
    if (!gap_fitting && args.has("--recorded")) {
        try {
            const auto recorded = resolve_cli::read_recorded(args);
            std::vector<std::string> recorded_columns;
            for (const auto& [name, cols] : recorded) {
                recorded_columns.insert(recorded_columns.end(), cols.begin(), cols.end());
            }
            std::optional<TableSource> header_source;
            if (!header_path.empty()) header_source = TableSource::csv(header_path);
            const auto values = read_plot_columns(header_source, TableSource::csv(species_path), roles.plot_id,
                                                  recorded_columns, predictions.plot_ids, "recorded values");
            std::size_t next = 0;
            for (const auto& [name, cols] : recorded) {
                std::vector<torch::Tensor> parts;
                for (std::size_t c = 0; c < cols.size(); ++c) {
                    parts.push_back(torch::tensor(values[next++], torch::kFloat64));
                }
                recorded_flags.emplace_back(name, recorded_plots(torch::stack(parts, 1)).contiguous());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading recorded values: " << e.what() << std::endl;
            return 1;
        }
    }
    if (gap_fitting) {
        try {
            GapColumns gap_columns;
            gap_columns.plot_id = roles.plot_id;
            gap_columns.species_id = roles.species_id;
            gap_columns.latitude = roles.latitude;
            gap_columns.longitude = roles.longitude;
            gap_columns.survey_year = resolve_cli::role_flag(args, "--survey-year");
            gap_columns.recorded = resolve_cli::read_recorded(args);
            std::vector<GapFitRequest> declared;
            for (const auto& item : args.get_all("--gap-target")) {
                declared.push_back(resolve_cli::parse_gap_target(item));
            }
            for (const auto& [name, cols] : gap_columns.recorded) {
                const auto it = std::find_if(declared.begin(), declared.end(),
                                             [&](const GapFitRequest& r) { return r.target.name == name; });
                if (it != declared.end()) {
                    gap_requests.push_back(*it);
                    continue;
                }
                GapFitRequest request;
                request.target.name = name;
                request.outputs = {name};
                const auto head = std::find_if(schema.targets.begin(), schema.targets.end(),
                                               [&](const TargetConfig& t) { return t.name == name; });
                if (head != schema.targets.end() && head->transform == TransformType::Log1p) {
                    request.target.scale = GapScale::Log;
                }
                gap_requests.push_back(std::move(request));
            }
            std::optional<TableSource> header_source;
            if (!header_path.empty()) header_source = TableSource::csv(header_path);
            const GapTables tables = read_gap_tables(header_source, TableSource::csv(species_path),
                                                     gap_columns, predictions.plot_ids);
            gap_fits = fit_gaps(predictions, tables, gap_requests, *gap_fitting);
        } catch (const std::exception& e) {
            std::cerr << "Error fitting gaps: " << e.what() << std::endl;
            return 1;
        }
    }

    // Write predictions to CSV
    std::cout << "Writing predictions to: " << output_path << std::endl;
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
        return 1;
    }

    // Header. A classification target gets TWO columns:
    //   <target>       the original CSV class label (schema.targets[].class_names),
    //                  or the integer code when the checkpoint has no class
    //                  vocabulary (a column that was already integer-coded);
    //   <target>_code  always the integer code the model predicted.
    // The code alone would leave every user to reconstruct the mapping by
    // hand; the label alone would lose the code an already-integer
    // column carries.
    //
    // With --probabilities a classification target additionally gets one
    // column per class, '<target>_prob_<class>' in code order, carrying the
    // softmax row the code was taken from.
    out << "plot_id";
    bool wrote_labels = false;
    bool wrote_probabilities = false;
    for (const auto& target : schema.targets) {
        out << "," << csv_field(target.name);
        if (target.task == TaskType::Classification) {
            out << "," << csv_field(target.name + "_code");
            if (!target.class_names.empty()) wrote_labels = true;
            if (write_probabilities) {
                for (int64_t k = 0; k < target.num_classes; ++k) {
                    out << "," << csv_field(target.name + "_prob_" +
                                            class_label(target.class_names, k));
                }
                wrote_probabilities = true;
            }
        }
    }
    // A gap-fitted target's fitted value (latitude / longitude for a
    // location) and the method that gave it, after every model output.
    for (const auto& request : gap_requests) {
        const std::string& name = request.target.name;
        if (request.target.scale == GapScale::Geographic) {
            out << "," << csv_field(name + "_fitted_latitude") << "," << csv_field(name + "_fitted_longitude");
        } else {
            out << "," << csv_field(name + "_fitted");
        }
        out << "," << csv_field(name + "_fit_method");
    }
    for (const auto& [name, flags] : recorded_flags) out << "," << csv_field(name + "_recorded");
    out << "\n";

    // Each target's outputs as contiguous CPU tensors in the dtype the writer
    // reads, so every cell below is a plain array read. Undefined for a target
    // the checkpoint produced no output for.
    struct TargetColumns {
        torch::Tensor values;         // (n,) float32 regression values or int64 codes
        torch::Tensor probabilities;  // (n, n_classes) float32 when written
    };
    std::vector<TargetColumns> columns;
    columns.reserve(schema.targets.size());
    for (const auto& target : schema.targets) {
        TargetColumns col;
        auto it = predictions.predictions.find(target.name);
        if (it != predictions.predictions.end()) {
            const bool is_classification = target.task == TaskType::Classification;
            col.values = it->second.to(torch::kCPU)
                             .to(is_classification ? torch::kInt64 : torch::kFloat32)
                             .contiguous();
            if (is_classification && write_probabilities) {
                col.probabilities = predictions.probabilities.at(target.name)
                                        .to(torch::kCPU).to(torch::kFloat32).contiguous();
            }
        }
        columns.push_back(std::move(col));
    }

    // Each gap fit as a contiguous (n, 1 or 2) float32 table and its method codes.
    std::vector<torch::Tensor> gap_fitted;
    std::vector<torch::Tensor> gap_methods;
    for (const auto& fit : gap_fits) {
        const torch::Tensor fitted = fit.fitted.to(torch::kCPU, torch::kFloat32);
        gap_fitted.push_back((fitted.dim() == 1 ? fitted.unsqueeze(1) : fitted).contiguous());
        gap_methods.push_back(fit.method.to(torch::kCPU, torch::kInt64).contiguous());
    }

    // Write predictions
    for (size_t i = 0; i < predictions.plot_ids.size(); ++i) {
        out << csv_field(predictions.plot_ids[i]);
        const auto row = static_cast<int64_t>(i);

        for (size_t t = 0; t < schema.targets.size(); ++t) {
            const auto& target = schema.targets[t];
            const auto& col = columns[t];
            const bool is_classification = target.task == TaskType::Classification;
            const int64_t n_prob_columns =
                (is_classification && write_probabilities) ? target.num_classes : 0;
            if (!col.values.defined()) {
                out << (is_classification ? ",NA,NA" : ",NA");
                for (int64_t k = 0; k < n_prob_columns; ++k) out << ",NA";
                continue;
            }
            if (is_classification) {
                const int64_t code = col.values.data_ptr<int64_t>()[row];
                out << "," << csv_field(class_label(target.class_names, code));
                out << "," << code;
                if (n_prob_columns > 0) {
                    const float* probs =
                        col.probabilities.data_ptr<float>() + row * col.probabilities.size(1);
                    for (int64_t k = 0; k < n_prob_columns; ++k) {
                        out << "," << resolve_cli::csv_number(probs[k]);
                    }
                }
            } else {
                out << "," << resolve_cli::csv_number(col.values.data_ptr<float>()[row]);
            }
        }
        for (std::size_t g = 0; g < gap_fits.size(); ++g) {
            const torch::Tensor& fitted = gap_fitted[g];
            const float* values = fitted.data_ptr<float>() + row * fitted.size(1);
            for (int64_t c = 0; c < fitted.size(1); ++c) out << "," << resolve_cli::csv_number(values[c]);
            out << "," << csv_field(gap_method_label(gap_methods[g].data_ptr<int64_t>()[row]));
        }
        for (const auto& [name, flags] : recorded_flags) {
            out << "," << (flags.data_ptr<bool>()[row] ? "TRUE" : "FALSE");
        }
        out << "\n";
    }

    out.close();
    std::cout << "Done! Wrote " << predictions.plot_ids.size() << " predictions."
              << std::endl;
    if (names) {
        // One vocabulary reads every target of a single model: the target
        // cell is left empty.
        try {
            std::cout << "Names report: "
                      << resolve_cli::write_names_report(args, output_path,
                                                         {{"", &dataset.species_names()}})
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }
    if (gap_fitting) {
        std::vector<std::pair<std::string, const GapFit*>> fits;
        for (std::size_t i = 0; i < gap_fits.size(); ++i) {
            fits.emplace_back(gap_requests[i].target.name, &gap_fits[i]);
            for (const auto& g : gap_fits[i].report.groups) {
                std::cout << "Gap fitting '" << gap_requests[i].target.name << "', " << g.n_gaps << " "
                          << g.group << " gaps: " << enum_to_name(g.method) << std::endl;
            }
        }
        try {
            std::cout << "Gap-fitting report: "
                      << resolve_cli::write_gap_report(args, output_path, fits) << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }
    if (wrote_labels) {
        std::cout << "Classification targets are written as two columns: "
                     "'<target>' (class label) and '<target>_code' (integer code)."
                  << std::endl;
    }
    if (wrote_probabilities) {
        std::cout << "Class probabilities are written as one column per class: "
                     "'<target>_prob_<class>', in code order."
                  << std::endl;
    }

    return 0;
}
