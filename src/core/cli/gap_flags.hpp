// RESOLVE CLI - gap fitting for `resolve predict` (--model and --suite): the
// flags, the recorded columns they name, and the report written beside the
// predictions.

#ifndef RESOLVE_CLI_GAP_FLAGS_HPP
#define RESOLVE_CLI_GAP_FLAGS_HPP

#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "resolve/enum_names.hpp"
#include "resolve/gap_fitting.hpp"
#include "resolve/json.hpp"

#include "arg_parser.hpp"
#include "config_flags.hpp"

namespace resolve_cli {

// GapFittingConfig's flags.
inline constexpr FieldFlag kGapFittingConfigFlags[] = {
    {"candidates", "--gap-candidates",
     "With --fit-gaps: the methods the choice is made\namong (default all)"},
    {"k_grid", "--gap-k-grid",
     "With --fit-gaps: the neighbour counts k is chosen\nfrom by the matched leave-one-out"},
    {"spatial", "--spatial",
     "With --fit-gaps: spatial neighbours as candidates\nwhere the gaps carry coordinates; --no-spatial\nreads composition only"},
    {"matching_seed", "--gap-seed",
     "With --fit-gaps: seed of the draw of the gaps'\ndistances the matched test hides by"},
};

// The gap-fitting rows of the `predict` flag table.
inline void append_gap_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--fit-gaps", Arity::Flag, "", "",
                     "Fit each target --recorded names from the dataset's\n"
                     "own complete plots: '<target>_fitted' and\n"
                     "'<target>_fit_method' beside the raw prediction,\n"
                     "and a report of what was chosen (--gap-report)."});
    flags.push_back({"--recorded", Arity::Repeatable, "T=COL", "",
                     "With --fit-gaps: the column holding target T's\n"
                     "recorded value, missing on the plots to fill;\n"
                     "T=LAT,LON for a location. Repeat per target."});
    flags.push_back({"--survey-year", Arity::Value, "COL", "",
                     "With --fit-gaps: the survey-year column; the\n"
                     "matched test then also hides plots closer in time\n"
                     "than the gaps are."});
    flags.push_back({"--gap-target", Arity::Repeatable, "T=SCALE[:PERIOD]:HEADS", "",
                     "With --fit-gaps and --model: how target T's values\n"
                     "lie and the head(s) it reads, e.g.\n"
                     "aspect=circular:360:asp_sin,asp_cos or\n"
                     "location=geographic:lat,lon. Default: the head\n"
                     "named T, log scale when it was trained on log1p."});
    flags.push_back({"--gap-report", Arity::Value, "PATH", "",
                     "With --fit-gaps: where the report is written (JSON;\n"
                     "default '<output>.gap_fitting.json')."});
    append_config_flags(flags, "Gap fitting", resolve::GapFittingConfig{},
                        FlagNaming::members(kGapFittingConfigFlags));
}

// TARGET=COL[,COL] values of --recorded.
inline std::vector<std::pair<std::string, std::vector<std::string>>> read_recorded(
    const ParsedArgs& args) {
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    for (const std::string& item : args.get_all("--recorded")) {
        const auto eq = item.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == item.size()) {
            throw ArgError("--recorded expects TARGET=COLUMN or TARGET=LAT,LON, got '" + item + "'");
        }
        std::vector<std::string> columns;
        std::string rest = item.substr(eq + 1);
        for (std::size_t start = 0;;) {
            const auto comma = rest.find(',', start);
            columns.push_back(rest.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        out.emplace_back(item.substr(0, eq), std::move(columns));
    }
    return out;
}

// The gap-fitting config, or nullopt without --fit-gaps (where a gap flag
// given anyway is an error).
inline std::optional<resolve::GapFittingConfig> read_gap_fitting(const ParsedArgs& args) {
    if (!args.has("--fit-gaps")) {
        for (const char* flag : {"--recorded", "--survey-year", "--gap-target", "--gap-report",
                                 "--gap-candidates", "--gap-k-grid", "--spatial", "--no-spatial",
                                 "--gap-seed"}) {
            if (args.has(flag)) throw ArgError(std::string(flag) + " needs --fit-gaps");
        }
        return std::nullopt;
    }
    if (!args.has("--recorded")) {
        throw ArgError("--fit-gaps needs --recorded TARGET=COLUMN for each target to fit");
    }
    resolve::GapFittingConfig config;
    read_config_flags(config, args, FlagNaming::members(kGapFittingConfigFlags));
    config.validate();
    return config;
}

// A --gap-target value: NAME=SCALE[:PERIOD]:HEAD[,HEAD].
inline resolve::GapFitRequest parse_gap_target(const std::string& item) {
    const auto eq = item.find('=');
    const auto first = eq == std::string::npos ? std::string::npos : item.find(':', eq);
    if (eq == std::string::npos || first == std::string::npos) {
        throw ArgError("--gap-target expects NAME=SCALE[:PERIOD]:HEAD[,HEAD], got '" + item + "'");
    }
    resolve::GapFitRequest request;
    request.target.name = item.substr(0, eq);
    try {
        request.target.scale =
            resolve::enum_from_name<resolve::GapScale>(item.substr(eq + 1, first - eq - 1));
    } catch (const std::exception&) {
        throw ArgError("--gap-target: unknown scale in '" + item + "'; accepted: " +
                       resolve::enum_spellings<resolve::GapScale>());
    }
    std::string heads = item.substr(first + 1);
    const auto second = heads.find(':');
    if (second != std::string::npos) {
        request.target.period = parse_float_value("--gap-target", heads.substr(0, second));
        heads = heads.substr(second + 1);
    }
    for (std::size_t start = 0;;) {
        const auto comma = heads.find(',', start);
        request.outputs.push_back(heads.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return request;
}

// Write each target's report into one JSON object, target name -> report.
inline std::string write_gap_report(const ParsedArgs& args, const std::string& output_path,
                                    const std::vector<std::pair<std::string, const resolve::GapFit*>>& fits) {
    std::string path = args.get("--gap-report");
    if (path.empty()) path = output_path + ".gap_fitting.json";
    resolve::json::Value report = resolve::json::Value::object();
    for (const auto& [name, fit] : fits) report.set(name, resolve::gap_fit_report_to_json(fit->report));
    std::ofstream out(path);
    if (!out.is_open()) throw ArgError("cannot open the gap-fitting report " + path);
    out << resolve::json::dump(report) << "\n";
    return path;
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_GAP_FLAGS_HPP
