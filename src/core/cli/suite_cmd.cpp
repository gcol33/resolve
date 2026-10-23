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
#include "cli_common.hpp"
#include "csv_output.hpp"

using resolve_cli::csv_field;
using resolve_cli::csv_number;
using resolve_cli::ParsedArgs;

namespace {

// The engine's prediction table (suite_prediction_columns) as CSV.
void write_columns(std::ostream& out, const std::vector<resolve::SuiteColumn>& columns) {
    using Kind = resolve::SuiteColumn::Kind;
    for (std::size_t c = 0; c < columns.size(); ++c) {
        out << (c == 0 ? "" : ",") << csv_field(columns[c].name);
    }
    out << "\n";
    const std::size_t n_rows = columns.empty() ? 0 : columns.front().text.size();
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t c = 0; c < columns.size(); ++c) {
            const resolve::SuiteColumn& column = columns[c];
            if (c > 0) out << ",";
            switch (column.kind) {
                case Kind::Text: out << csv_field(column.text[i]); break;
                case Kind::Integer: out << column.integers[i]; break;
                case Kind::Number: out << csv_number(column.numbers[i]); break;
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
    load.device = resolve_cli::choose_device(args);

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
    // A role flag renames a contract column only when it was given; its table
    // default belongs to `--model`, where no manifest supplies the name.
    predict.columns = resolve_cli::read_suite_columns(args);

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
    write_columns(out, suite_prediction_columns(*predictions, probabilities));
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
            std::cout << "  A recorded abundance of 0 is read as "
                      << number_to_decimal(in.zero_abundance_as) << std::endl;
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
            if (t.inputs) {
                std::cout << "    reads: species";
                if (t.inputs->coordinates) std::cout << ", coordinates";
                for (const auto& c : t.inputs->covariates) std::cout << ", " << c;
                for (const auto& c : t.inputs->categoricals) std::cout << ", " << c;
                std::cout << std::endl;
            }
            if (!t.limit.empty()) std::cout << "    limit: " << t.limit << std::endl;
            for (const auto& member : t.members) {
                if (member.zero_abundance_as) {
                    std::cout << "    seed " << member.seed << " read a recorded abundance of 0 as "
                              << number_to_decimal(*member.zero_abundance_as) << std::endl;
                }
            }
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
