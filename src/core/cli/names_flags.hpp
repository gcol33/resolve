// RESOLVE CLI - name resolution for `resolve predict` (--model and --suite):
// the flags and the report of the names not found as written.

#ifndef RESOLVE_CLI_NAMES_FLAGS_HPP
#define RESOLVE_CLI_NAMES_FLAGS_HPP

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "resolve/enum_names.hpp"
#include "resolve/name_resolution.hpp"

#include "arg_parser.hpp"
#include "csv_output.hpp"

namespace resolve_cli {

// The name-resolution rows of the `predict` flag table.
inline void append_names_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--names", Arity::Value, "PATH", "",
                     "Resolve the species names before encoding, with the\n"
                     "release's name table (names.json): a name the\n"
                     "vocabulary lacks is read through the table (EVA's\n"
                     "entered names, backbone synonyms), else as the\n"
                     "nearest known name within --max-edit-distance.\n"
                     "The names not found as written go to --names-report."});
    flags.push_back({"--spelling-only", Arity::Flag, "", "",
                     "Resolve exact names and spellings against the\n"
                     "vocabulary alone, without a name table."});
    flags.push_back({"--max-edit-distance", Arity::Value, "N", "1",
                     "With --names or --spelling-only: the edits a\n"
                     "misspelled name may be away from a known name (0\n"
                     "turns the spelling step off; 2 can also rename\n"
                     "species the models never saw)."});
    flags.push_back({"--names-report", Arity::Value, "PATH", "",
                     "With --names or --spelling-only: where the names not\n"
                     "found as written are listed (CSV; default\n"
                     "'<output>.names.csv')."});
}

// The name-resolution setting, or nullopt without --names / --spelling-only
// (where a name flag given anyway is an error).
inline std::optional<resolve::SpeciesNames> read_species_names(const ParsedArgs& args) {
    const bool table = args.has("--names");
    const bool spelling_only = args.has("--spelling-only");
    if (table && spelling_only) throw ArgError("--names and --spelling-only exclude each other");
    if (!table && !spelling_only) {
        for (const char* flag : {"--max-edit-distance", "--names-report"}) {
            if (args.has(flag)) throw ArgError(std::string(flag) + " needs --names or --spelling-only");
        }
        return std::nullopt;
    }
    resolve::SpeciesNames names;
    names.options.max_edit_distance = static_cast<int>(args.get_int64("--max-edit-distance"));
    if (names.options.max_edit_distance < 0 ||
        names.options.max_edit_distance > resolve::kMaxNameEditDistance) {
        throw ArgError("--max-edit-distance must be in [0, " +
                       std::to_string(resolve::kMaxNameEditDistance) + "]");
    }
    if (table) names.table = resolve::NameTable::open(args.get("--names"));
    return names;
}

// Write the names each target did not find as written, one row per target and
// name: target (empty for --model, whose targets share one vocabulary), name,
// token, route, source, via, distance, n_records.
inline std::string write_names_report(
    const ParsedArgs& args, const std::string& output_path,
    const std::vector<std::pair<std::string, const resolve::SpeciesNameReport*>>& reports) {
    std::string path = args.get("--names-report");
    if (path.empty()) path = output_path + ".names.csv";
    std::ofstream out(path);
    if (!out.is_open()) throw ArgError("cannot open the names report " + path);
    out << "target,name,token,route,source,via,distance,n_records\n";
    for (const auto& [target, report] : reports) {
        for (std::size_t i = 0; i < report->size(); ++i) {
            out << csv_field(target) << ',' << csv_field(report->name[i]) << ','
                << csv_field(report->token[i]) << ',' << resolve::enum_to_name(report->route[i])
                << ',' << csv_field(report->source[i]) << ',' << csv_field(report->via[i]) << ','
                << report->distance[i] << ',' << report->n_records[i] << '\n';
        }
    }
    return path;
}

// A line per route: how many of the names not found as written each took.
inline std::string names_summary(const resolve::SpeciesNameReport& report) {
    std::size_t counts[4] = {0, 0, 0, 0};
    for (const resolve::NameRoute r : report.route) ++counts[static_cast<std::size_t>(r)];
    return std::to_string(counts[static_cast<std::size_t>(resolve::NameRoute::Alias)]) +
           " through the name table, " +
           std::to_string(counts[static_cast<std::size_t>(resolve::NameRoute::Spelling)]) +
           " by spelling, " +
           std::to_string(counts[static_cast<std::size_t>(resolve::NameRoute::Unresolved)]) +
           " unresolved";
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_NAMES_FLAGS_HPP
