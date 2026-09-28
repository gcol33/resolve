// RESOLVE CLI - model suites: `resolve predict --suite` and `resolve info --suite`.
//
// Reads its values from the ParsedArgs produced by the `predict` and `info`
// flag tables in cli_spec.hpp.

#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "resolve/resolve.hpp"

#include "arg_parser.hpp"
#include "csv_output.hpp"

using resolve_cli::class_label;
using resolve_cli::csv_field;
using resolve_cli::csv_number;
using resolve_cli::ParsedArgs;

namespace {

// A role flag renames a contract column only when it was given; its table
// default belongs to `--model`, where no manifest supplies the name.
std::optional<std::string> column_flag(const ParsedArgs& args, const char* flag) {
    std::string value;
    if (args.get_if_present(flag, value)) return value;
    return std::nullopt;
}

void write_header(std::ostream& out, const resolve::SuitePredictions& p, bool probabilities,
                  bool members) {
    using resolve::SuiteCombine;
    out << "plot_id";
    for (const auto& t : p.targets) {
        const std::string n = t.name;
        out << "," << csv_field(n);
        switch (t.combine) {
            case SuiteCombine::Vote:
                out << "," << csv_field(n + "_code") << "," << csv_field(n + "_agreement");
                if (probabilities) {
                    for (int64_t k = 0; k < t.probabilities.size(1); ++k) {
                        out << "," << csv_field(n + "_prob_" + class_label(t.class_names, k));
                    }
                }
                break;
            case SuiteCombine::Mean:
                out << "," << csv_field(n + "_sd");
                break;
            case SuiteCombine::CircularMean:
                out << "," << csv_field(n + "_circular_sd");
                break;
        }
        out << "," << csv_field(n + "_n_species") << "," << csv_field(n + "_n_recognised")
            << "," << csv_field(n + "_recognised_share") << ","
            << csv_field(n + "_recognised_abundance_share");
        if (members) {
            for (const int64_t seed : t.member_seeds) {
                out << "," << csv_field(n + "_seed" + std::to_string(seed));
            }
        }
    }
    out << "\n";
}

double at(const torch::Tensor& t, int64_t i) {
    return t.defined() ? t[i].item<double>() : std::nan("");
}

void write_rows(std::ostream& out, const resolve::SuitePredictions& p, bool probabilities,
                bool members) {
    using resolve::SuiteCombine;
    // Contiguous CPU tensors, so the per-cell reads below are plain indexing.
    struct Columns {
        torch::Tensor value, agreement, probabilities, dispersion, members;
        torch::Tensor n_species, n_recognised, count_share, abundance_share;
    };
    std::vector<Columns> cols;
    auto cpu = [](const torch::Tensor& t) {
        return t.defined() ? t.to(torch::kCPU).contiguous() : t;
    };
    for (const auto& t : p.targets) {
        cols.push_back({cpu(t.value), cpu(t.agreement), cpu(t.probabilities),
                        cpu(t.dispersion), cpu(t.members), cpu(t.recognition.n_species),
                        cpu(t.recognition.n_recognised), cpu(t.recognition.count_share),
                        cpu(t.recognition.abundance_share)});
    }

    const auto n = static_cast<int64_t>(p.plot_ids.size());
    for (int64_t i = 0; i < n; ++i) {
        out << csv_field(p.plot_ids[static_cast<size_t>(i)]);
        for (std::size_t j = 0; j < p.targets.size(); ++j) {
            const auto& t = p.targets[j];
            const auto& c = cols[j];
            if (t.combine == SuiteCombine::Vote) {
                const int64_t code = c.value[i].item<int64_t>();
                out << "," << csv_field(class_label(t.class_names, code)) << "," << code << ","
                    << csv_number(at(c.agreement, i));
                if (probabilities) {
                    for (int64_t k = 0; k < c.probabilities.size(1); ++k) {
                        out << "," << csv_number(c.probabilities[i][k].item<double>());
                    }
                }
            } else {
                out << "," << csv_number(at(c.value, i)) << "," << csv_number(at(c.dispersion, i));
            }
            out << "," << csv_number(at(c.n_species, i)) << ","
                << csv_number(at(c.n_recognised, i)) << "," << csv_number(at(c.count_share, i))
                << "," << csv_number(at(c.abundance_share, i));
            if (members) {
                for (int64_t m = 0; m < c.members.size(0); ++m) {
                    if (t.combine == SuiteCombine::Vote) {
                        out << "," << csv_field(class_label(t.class_names,
                                                            c.members[m][i].item<int64_t>()));
                    } else {
                        out << "," << csv_number(c.members[m][i].item<double>());
                    }
                }
            }
        }
        out << "\n";
    }
}

void print_limits(const resolve::SuitePredictions& p) {
    for (const auto& t : p.targets) {
        if (t.status == resolve::SuiteTargetStatus::Experimental) {
            std::cout << "Note: '" << t.name << "' is released as experimental: " << t.limit
                      << std::endl;
        }
    }
}

}  // namespace

int predict_suite_command(const ParsedArgs& args) {
    using namespace resolve;

    const std::string suite_dir = args.get("--suite");
    const std::string header_path = args.get("--header");
    const std::string species_path = args.get("--species");
    const std::string output_path = args.get("--output");
    const bool probabilities = args.has("--probabilities");
    const bool members = args.has("--members");

    if (species_path.empty()) {
        std::cerr << "Error: --species is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Suite Prediction" << std::endl;
    std::cout << "========================" << std::endl;

    SuiteLoadOptions load;
    load.verify_checksums = !args.has("--no-verify");
    load.targets = args.get_all("--target");
    load.vram_fraction = args.get_float("--vram-fraction");
    if (args.has("--cuda") && torch::cuda::is_available()) {
        load.device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }

    std::optional<SuitePredictor> suite;
    try {
        suite.emplace(SuitePredictor::load(suite_dir, load));
    } catch (const std::exception& e) {
        std::cerr << "Error loading suite: " << e.what() << std::endl;
        return 1;
    }
    const SuiteManifest& manifest = suite->manifest();
    std::cout << "Suite: " << manifest.name << " (" << suite_dir << ")" << std::endl;
    std::cout << "Targets:";
    for (const auto& name : suite->target_names()) std::cout << " " << name;
    std::cout << std::endl;
    if (load.verify_checksums) std::cout << "Member checksums verified" << std::endl;

    SuitePredictOptions predict;
    predict.batch_size = args.get_int64("--predict-batch-size");
    predict.keep_members = members;
    predict.columns.plot_id = column_flag(args, "--plot-id");
    predict.columns.species_id = column_flag(args, "--species-id");
    predict.columns.abundance = column_flag(args, "--abundance");
    predict.columns.genus = column_flag(args, "--genus");
    predict.columns.family = column_flag(args, "--family");
    predict.columns.latitude = column_flag(args, "--lat");
    predict.columns.longitude = column_flag(args, "--lon");
    if (args.has("--covariate")) predict.columns.covariates = args.get_all("--covariate");
    if (args.has("--categorical")) predict.columns.categoricals = args.get_all("--categorical");

    std::optional<SuitePredictions> predictions;
    try {
        predictions.emplace(suite->predict(SuiteInput::csv(header_path, species_path), predict));
    } catch (const std::exception& e) {
        std::cerr << "Error predicting: " << e.what() << std::endl;
        return 1;
    }

    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_path << std::endl;
        return 1;
    }
    write_header(out, *predictions, probabilities, members);
    write_rows(out, *predictions, probabilities, members);
    out.close();

    std::cout << "Wrote " << predictions->plot_ids.size() << " plots to " << output_path
              << std::endl;
    print_limits(*predictions);
    return 0;
}

int info_suite_command(const ParsedArgs& args) {
    using namespace resolve;
    const std::string suite_dir = args.get("--suite");
    try {
        const SuiteManifest m = SuiteManifest::read(suite_dir);
        std::cout << "RESOLVE Suite Information" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "Suite: " << m.name << std::endl;
        if (!m.description.empty()) std::cout << "Description: " << m.description << std::endl;
        std::cout << "Licence: " << m.licence << std::endl;
        std::cout << "Trained with engine: " << m.engine_version << std::endl;
        std::cout << "Taxonomy: " << m.taxonomy << std::endl;
        std::cout << "Training scope: " << m.training_scope << std::endl;

        const auto& in = m.inputs;
        std::cout << "\nInputs:" << std::endl;
        std::cout << "  Plot id: " << in.plot_id << std::endl;
        std::cout << "  Species: " << in.species << std::endl;
        auto optional_column = [](const char* label, const std::string& column) {
            std::cout << "  " << label << ": " << (column.empty() ? "(not read)" : column)
                      << std::endl;
        };
        optional_column("Abundance", in.abundance);
        if (!in.abundance_units.empty()) {
            std::cout << "  Abundance units: " << in.abundance_units << std::endl;
        }
        if (in.zero_abundance_as != 0.0f) {
            std::cout << "  A recorded abundance of 0 is read as " << in.zero_abundance_as
                      << std::endl;
        }
        optional_column("Genus", in.genus);
        optional_column("Family", in.family);
        optional_column("Latitude", in.latitude);
        optional_column("Longitude", in.longitude);
        for (const auto& c : in.covariates) std::cout << "  Covariate: " << c << std::endl;
        for (const auto& c : in.categoricals) std::cout << "  Categorical: " << c << std::endl;
        if (!in.notes.empty()) std::cout << "  Notes: " << in.notes << std::endl;

        std::cout << "\nTargets:" << std::endl;
        for (const auto& t : m.targets) {
            std::cout << "  " << t.name << ": " << enum_to_name(t.combine) << " of "
                      << t.members.size() << " member(s)";
            if (!t.units.empty()) std::cout << ", " << t.units;
            std::cout << ", " << enum_to_name(t.status) << std::endl;
            if (!t.limit.empty()) std::cout << "    limit: " << t.limit << std::endl;
            for (const auto& [regime, metrics] : t.validation.members()) {
                std::cout << "    validation " << regime << ": " << json::dump(metrics, 0)
                          << std::endl;
            }
        }

        const auto problems = m.verify(suite_dir);
        std::cout << "\nMembers: ";
        if (problems.empty()) {
            std::cout << "all present, sizes and SHA-256 match the manifest" << std::endl;
        } else {
            std::cout << problems.size() << " problem(s)" << std::endl;
            for (const auto& p : problems) std::cout << "  - " << p << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading suite: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
