// RESOLVE CLI - model suites: `resolve predict --suite`, `resolve info --suite`
// and `resolve suite derive`.
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
#include "gap_flags.hpp"
#include "names_flags.hpp"
#include "terrain_flags.hpp"

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
                case Kind::Logical: out << (column.logical[i] ? "TRUE" : "FALSE"); break;
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

// Split "KEY=VALUE" at the first '='.
std::pair<std::string, std::string> key_value(const std::string& flag, const std::string& item) {
    const auto eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size()) {
        throw resolve_cli::ArgError(flag + " expects KEY=VALUE, got '" + item + "'");
    }
    return {item.substr(0, eq), item.substr(eq + 1)};
}

}  // namespace

int suite_derive_command(const ParsedArgs& args) {
    using namespace resolve;
    DerivedSuiteSpec spec;
    spec.parent = args.get("--parent");
    spec.name = args.get("--name");
    spec.description = args.get("--description");
    spec.licence = args.get("--licence");
    spec.training_scope = args.get("--training-scope");
    const std::string output = args.get("--output");
    for (const auto& [flag, value] : {std::pair<const char*, const std::string*>{"--parent", &spec.parent},
                                      {"--output", &output},
                                      {"--name", &spec.name},
                                      {"--licence", &spec.licence},
                                      {"--training-scope", &spec.training_scope}}) {
        if (value->empty()) {
            std::cerr << "Error: " << flag << " is required" << std::endl;
            return 1;
        }
    }

    auto target = [&spec](const std::string& name) -> DerivedSuiteTarget& {
        for (auto& t : spec.targets) {
            if (t.name == name) return t;
        }
        spec.targets.push_back(DerivedSuiteTarget{});
        spec.targets.back().name = name;
        return spec.targets.back();
    };
    auto known = [&spec](const char* flag, const std::string& name) -> DerivedSuiteTarget& {
        for (auto& t : spec.targets) {
            if (t.name == name) return t;
        }
        throw resolve_cli::ArgError(std::string(flag) + " names target '" + name +
                                    "', which no --member names");
    };

    try {
        for (const auto& item : args.get_all("--member")) {
            const auto [key, path] = key_value("--member", item);
            const auto colon = key.rfind(':');
            if (colon == std::string::npos || colon == 0 || colon + 1 == key.size()) {
                throw resolve_cli::ArgError("--member expects TARGET:SEED=PATH, got '" + item + "'");
            }
            target(key.substr(0, colon))
                .members.push_back({path, resolve_cli::parse_int_value("--member", key.substr(colon + 1))});
        }
        if (spec.targets.empty()) {
            std::cerr << "Error: at least one --member is required" << std::endl;
            return 1;
        }
        for (const auto& item : args.get_all("--units")) {
            const auto [name, units] = key_value("--units", item);
            known("--units", name).units = units;
        }
        for (const auto& item : args.get_all("--outputs")) {
            const auto [name, heads] = key_value("--outputs", item);
            auto& t = known("--outputs", name);
            std::size_t start = 0;
            while (start <= heads.size()) {
                const std::size_t end = std::min(heads.find(',', start), heads.size());
                t.outputs.push_back(heads.substr(start, end - start));
                start = end + 1;
            }
        }
        for (const auto& item : args.get_all("--combine")) {
            const auto [name, rule] = key_value("--combine", item);
            known("--combine", name).combine = enum_from_name<SuiteCombine>(rule);
        }
        for (const auto& item : args.get_all("--period")) {
            const auto [name, period] = key_value("--period", item);
            known("--period", name).period = resolve_cli::parse_float_value("--period", period);
        }
        for (const auto& item : args.get_all("--experimental")) {
            const auto [name, limit] = key_value("--experimental", item);
            auto& t = known("--experimental", name);
            t.status = SuiteTargetStatus::Experimental;
            t.limit = limit;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    try {
        const SuiteManifest manifest = write_derived_suite(output, spec);
        std::cout << "Derived suite '" << manifest.name << "' written to " << output << std::endl;
        for (const auto& t : manifest.targets) {
            std::cout << "  " << t.name << ": " << t.members.size() << " member(s) from '"
                      << t.derived_from->suite << "' target '" << t.derived_from->target << "'"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

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
    try {
        predict.gap_fitting = resolve_cli::read_gap_fitting(args);
        predict.columns.recorded = resolve_cli::read_recorded(args);
        if (predict.gap_fitting) {
            predict.columns.survey_year = resolve_cli::role_flag(args, "--survey-year");
        }
        predict.names = resolve_cli::read_species_names(args);
        predict.terrain = resolve_cli::read_terrain(args, predict.columns, load.device);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

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
    if (predict.names) {
        std::vector<std::pair<std::string, const SpeciesNameReport*>> reports;
        for (const auto& t : predictions->targets) {
            reports.emplace_back(t.name, &t.species_names);
            std::cout << "Names '" << t.name << "': " << resolve_cli::names_summary(t.species_names)
                      << std::endl;
        }
        try {
            std::cout << "Names report: " << resolve_cli::write_names_report(args, output_path, reports)
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }
    if (predict.gap_fitting) {
        std::vector<std::pair<std::string, const GapFit*>> fits;
        for (const auto& t : predictions->targets) {
            if (!t.gap_fit) continue;
            fits.emplace_back(t.name, &*t.gap_fit);
            for (const auto& g : t.gap_fit->report.groups) {
                std::cout << "Gap fitting '" << t.name << "', " << g.n_gaps << " " << g.group
                          << " gaps: " << enum_to_name(g.method) << std::endl;
            }
        }
        try {
            std::cout << "Gap-fitting report: " << resolve_cli::write_gap_report(args, output_path, fits)
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }
    if (predict.terrain) {
        for (const auto& t : predictions->targets) {
            if (!t.terrain) continue;
            const auto applied = t.terrain->result.applied.sum().item<int64_t>();
            std::cout << "Terrain correction '" << t.name << "' (" << enum_to_name(t.terrain->source)
                      << "): applied to " << applied << " of " << predictions->plot_ids.size() << " plots"
                      << std::endl;
        }
    }
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
            if (!t.classes.empty()) {
                std::cout << "    classes:";
                for (const auto& c : t.classes) std::cout << " " << c;
                std::cout << std::endl;
            }
            if (t.derived_from) {
                const SuiteDerivation& d = *t.derived_from;
                std::cout << "    fine-tuned from suite '" << d.suite << "' target '" << d.target
                          << "'";
                if (!d.licence.empty()) std::cout << " (" << d.licence << ")";
                std::cout << std::endl;
                if (!d.training_scope.empty()) {
                    std::cout << "    parent training scope: " << d.training_scope << std::endl;
                }
            }
            if (t.terrain_correction) {
                std::cout << "    terrain correction: " << t.terrain_correction->file << std::endl;
            }
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

        if (m.terrain_field) std::cout << "\nTerrain field: " << m.terrain_field->file << std::endl;

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
