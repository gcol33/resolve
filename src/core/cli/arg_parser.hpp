#pragma once

// Declarative flag table for the RESOLVE CLI.
//
// Each subcommand declares its flags once, as rows in a CommandSpec. Everything
// else reads that table:
//
//   * parse_args   rejects a flag the table does not declare (a silently
//                  ignored `--maxepochs` is a wrong run that looks fine), a
//                  value flag with no value, and a stray positional;
//   * render_usage generates the help text, so help cannot drift from what the
//                  parser accepts;
//   * ParsedArgs   refuses to read a flag the table does not declare, so a
//                  typo on the reading side is a loud error rather than a
//                  silent default.
//
// Adding a flag is one row plus one read. No torch, no engine types: this
// header is pure string handling so the parsing rules can be unit-tested
// without a process (see tests/test_effective_batch.cpp).

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace resolve_cli {

// How many values a flag takes and how repeats are treated.
enum class Arity {
    Flag,        // presence only, no value    (--cuda)
    Value,       // --name VALUE, last wins    (--batch-size 4096)
    Repeatable   // --name VALUE, all kept     (--target area:regression)
};

// One row of a command's flag table.
struct FlagSpec {
    const char* name;           // "--batch-size", always with the leading "--"
    Arity arity;
    const char* value_label;    // "N" / "PATH" / "COL"; "" for Arity::Flag
    const char* default_value;  // "" when the flag has no default
    const char* help;           // "\n" separates continuation lines
};

// Raised for every user-facing argument problem. main() prints what() and
// exits non-zero.
class ArgError : public std::runtime_error {
public:
    explicit ArgError(const std::string& msg) : std::runtime_error(msg) {}
};

namespace detail {

// Lowercase and drop '-' / '_' so "--maxepochs" and "--max_epochs" both
// normalize to the same key as "--max-epochs". Those two spellings are exactly
// the typos an ignored-unknown-flag parser used to swallow, so the suggestion
// is derived from the same normalization rather than an edit distance.
inline std::string normalize_flag(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '-' || c == '_') continue;
        out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

}  // namespace detail

// The flag table for one subcommand, plus the prose that heads its help block.
class CommandSpec {
public:
    CommandSpec(std::string name, std::string summary, std::vector<FlagSpec> flags)
        : name_(std::move(name)), summary_(std::move(summary)), flags_(std::move(flags)) {}

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::string& summary() const noexcept { return summary_; }
    [[nodiscard]] const std::vector<FlagSpec>& flags() const noexcept { return flags_; }

    [[nodiscard]] const FlagSpec* find(const std::string& flag) const noexcept {
        for (const auto& f : flags_) {
            if (flag == f.name) return &f;
        }
        return nullptr;
    }

    // Closest declared flag under the hyphen/underscore/case-insensitive
    // normalization, or "" when nothing matches.
    [[nodiscard]] std::string suggest(const std::string& flag) const {
        const std::string key = detail::normalize_flag(flag);
        for (const auto& f : flags_) {
            if (detail::normalize_flag(f.name) == key) return f.name;
        }
        return {};
    }

private:
    std::string name_;
    std::string summary_;
    std::vector<FlagSpec> flags_;
};

// Values collected for one subcommand invocation. Reads are checked against the
// spec, so a name that is not a declared flag throws instead of quietly
// returning a default.
class ParsedArgs {
public:
    ParsedArgs() = default;
    explicit ParsedArgs(const CommandSpec* spec) : spec_(spec) {}

    [[nodiscard]] bool has(const std::string& flag) const {
        require_declared(flag);
        for (const auto& kv : values_) {
            if (kv.first == flag) return true;
        }
        return false;
    }

    // Last supplied value, or the table's default when the flag was omitted.
    [[nodiscard]] std::string get(const std::string& flag) const {
        const FlagSpec* spec = require_declared(flag);
        std::string out = spec->default_value;
        for (const auto& kv : values_) {
            if (kv.first == flag) out = kv.second;
        }
        return out;
    }

    // Last supplied value, or empty when the flag was omitted. For roles that
    // must stay unset rather than fall back to a default column name.
    [[nodiscard]] bool get_if_present(const std::string& flag, std::string& out) const {
        require_declared(flag);
        bool found = false;
        for (const auto& kv : values_) {
            if (kv.first == flag) {
                out = kv.second;
                found = true;
            }
        }
        return found;
    }

    // Every value supplied for a repeatable flag, in command-line order.
    [[nodiscard]] std::vector<std::string> get_all(const std::string& flag) const {
        require_declared(flag);
        std::vector<std::string> out;
        for (const auto& kv : values_) {
            if (kv.first == flag) out.push_back(kv.second);
        }
        return out;
    }

    // Comma-separated list value ("64,32" -> {64, 32}). Empty string -> {}.
    [[nodiscard]] std::vector<std::string> get_list(const std::string& flag) const {
        const std::string raw = get(flag);
        std::vector<std::string> out;
        std::string item;
        std::istringstream stream(raw);
        while (std::getline(stream, item, ',')) {
            const std::string trimmed = trim(item);
            if (!trimmed.empty()) out.push_back(trimmed);
        }
        return out;
    }

    [[nodiscard]] int get_int(const std::string& flag) const {
        return static_cast<int>(parse_int64(flag, get(flag)));
    }
    [[nodiscard]] int64_t get_int64(const std::string& flag) const {
        return parse_int64(flag, get(flag));
    }
    [[nodiscard]] float get_float(const std::string& flag) const {
        return parse_float(flag, get(flag));
    }

    // Boolean carried by a pair of opposing presence flags (--amp / --no-amp).
    // The one that appears later on the command line wins, so a default in a
    // wrapper script stays overridable.
    [[nodiscard]] bool get_switch(const std::string& on_flag,
                                  const std::string& off_flag,
                                  bool default_value) const {
        require_declared(on_flag);
        require_declared(off_flag);
        bool value = default_value;
        for (const auto& kv : values_) {
            if (kv.first == on_flag) value = true;
            else if (kv.first == off_flag) value = false;
        }
        return value;
    }

    // Used by parse_args; also the seam a test uses to build a ParsedArgs
    // directly.
    void add(std::string flag, std::string value) {
        values_.emplace_back(std::move(flag), std::move(value));
    }

    [[nodiscard]] const CommandSpec* spec() const noexcept { return spec_; }

private:
    const FlagSpec* require_declared(const std::string& flag) const {
        if (spec_ == nullptr) {
            throw ArgError("ParsedArgs has no command spec; cannot read " + flag);
        }
        const FlagSpec* found = spec_->find(flag);
        if (found == nullptr) {
            // A programming error in the command implementation, not user
            // input: the command asked for a flag its own table never declared.
            throw ArgError("internal error: command '" + spec_->name() +
                           "' read undeclared flag " + flag);
        }
        return found;
    }

    static std::string trim(const std::string& s) {
        size_t begin = 0;
        size_t end = s.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
        while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
        return s.substr(begin, end - begin);
    }

    static int64_t parse_int64(const std::string& flag, const std::string& raw) {
        try {
            size_t consumed = 0;
            const long long value = std::stoll(raw, &consumed);
            if (consumed != raw.size()) throw std::invalid_argument("trailing characters");
            return static_cast<int64_t>(value);
        } catch (const std::exception&) {
            throw ArgError(flag + " expects an integer, got '" + raw + "'");
        }
    }

    static float parse_float(const std::string& flag, const std::string& raw) {
        try {
            size_t consumed = 0;
            const float value = std::stof(raw, &consumed);
            if (consumed != raw.size()) throw std::invalid_argument("trailing characters");
            return value;
        } catch (const std::exception&) {
            throw ArgError(flag + " expects a number, got '" + raw + "'");
        }
    }

    const CommandSpec* spec_ = nullptr;
    std::vector<std::pair<std::string, std::string>> values_;
};

// Parse the tokens that follow the subcommand name. Throws ArgError on an
// unknown flag, a missing value, or a positional argument.
inline ParsedArgs parse_args(const CommandSpec& spec,
                             const std::vector<std::string>& tokens) {
    ParsedArgs parsed(&spec);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];

        if (token.rfind("--", 0) != 0) {
            throw ArgError("unexpected argument '" + token + "' for command '" +
                           spec.name() + "'. Values must follow their flag, e.g. " +
                           "--target area:regression. Run `resolve help` for the "
                           "full flag list.");
        }

        const FlagSpec* flag = spec.find(token);
        if (flag == nullptr) {
            std::string msg = "unknown flag '" + token + "' for command '" +
                              spec.name() + "'";
            const std::string suggestion = spec.suggest(token);
            if (!suggestion.empty()) msg += ". Did you mean " + suggestion + "?";
            else msg += ". Run `resolve help` for the full flag list.";
            throw ArgError(msg);
        }

        if (flag->arity == Arity::Flag) {
            parsed.add(token, "1");
            continue;
        }

        if (i + 1 >= tokens.size()) {
            throw ArgError(std::string(flag->name) + " expects a value (" +
                           flag->value_label + ")");
        }
        // A declared flag sitting where a value should be is a forgotten value,
        // not a value that happens to look like a flag. A bare negative number
        // ("-1") starts with a single dash and is accepted normally.
        const std::string& next = tokens[i + 1];
        if (next.rfind("--", 0) == 0 && spec.find(next) != nullptr) {
            throw ArgError(std::string(flag->name) + " expects a value (" +
                           flag->value_label + "), found flag '" + next + "'");
        }
        parsed.add(token, next);
        ++i;
    }

    return parsed;
}

// Render one command's flag block. The flag column widens to fit the longest
// entry (with a floor so short blocks line up with the long ones), and a flag
// wider than the column puts its help on the following line.
inline std::string render_usage(const CommandSpec& spec) {
    constexpr size_t kIndent = 2;
    constexpr size_t kMinColumn = 25;
    constexpr size_t kMaxColumn = 30;

    auto flag_text = [](const FlagSpec& f) {
        std::string text = f.name;
        if (f.arity != Arity::Flag && f.value_label[0] != '\0') {
            text += " ";
            text += f.value_label;
        }
        return text;
    };

    size_t column = kMinColumn;
    for (const auto& f : spec.flags()) {
        column = std::max(column, kIndent + flag_text(f).size() + 2);
    }
    column = std::min(column, kMaxColumn);

    std::ostringstream out;
    out << spec.summary() << "\n";
    for (const auto& f : spec.flags()) {
        const std::string text = flag_text(f);
        std::string line = std::string(kIndent, ' ') + text;
        if (line.size() + 2 > column) {
            out << line << "\n";
            line.clear();
        }
        line.append(column - line.size(), ' ');

        std::string help = f.help;
        if (f.default_value[0] != '\0') {
            help += " (default: ";
            help += f.default_value;
            help += ")";
        }

        std::istringstream help_lines(help);
        std::string help_line;
        bool first = true;
        while (std::getline(help_lines, help_line)) {
            if (first) {
                out << line << help_line << "\n";
                first = false;
            } else {
                out << std::string(column, ' ') << help_line << "\n";
            }
        }
        if (first) out << line << "\n";
    }
    return out.str();
}

}  // namespace resolve_cli
