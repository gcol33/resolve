#pragma once

// Name resolution: the species names of a table read as the tokens a model's
// vocabulary holds.
//
// A checkpoint's species vocabulary holds the name each training record was
// resolved to (EVA's WFO_TAXON), and an encoding matches a table's species
// names exactly against it. A user writes a species as it was entered in the
// field, by a current accepted name, or misspelled, and none of these need be
// the token: EVA's records entered as Anemone nemorosa carry the token
// "Anemone altaica", and WFO's accepted Anemonoides nemorosa is no token at
// all. The resolver reads each name, first hit wins:
//
//   exact     the vocabulary holds the name as written;
//   alias     the release's name table maps the name to a token the
//             vocabulary holds (EVA's entered names, then the synonymy of the
//             plant backbones, each row naming its source);
//   spelling  within max_edit_distance edits (Damerau-Levenshtein, letter
//             case ignored) of a token or a table name whose token the
//             vocabulary holds, searched within the name's genus (its first
//             word) first and across every entry only when the genus holds
//             none; used only when every nearest entry gives one token.
//
// A name none of these reach stays as written and encodes as an unknown
// species. Every name that is not exact is reported with its route.

#include "resolve/fuzzy.hpp"
#include "resolve/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace resolve {

// Transparent string hash, so a map keyed by std::string is searched with a
// string_view without building a string.
struct NameHash {
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const noexcept {
        return std::hash<std::string_view>{}(s);
    }
};

// The release's name table: name -> the token it stands for, and the source
// that linked them ("eva" for EVA's entered names, else the backbone, e.g.
// "wfo"). A name that is a token of a released vocabulary is never in it.
class NameTable {
public:
    struct Alias {
        std::string token;
        std::string source;
    };

    NameTable() = default;

    // The table in `path`: a JSON object {"source": <text>, "aliases":
    // {name: [token, source], ...}}, the file the release ships beside its
    // suites. Records the file's SHA-256.
    [[nodiscard]] static NameTable read(const std::string& path);

    // read(path), shared: a process reads each file once and hands every later
    // caller the same table while the file keeps its size and modification
    // time. What the bindings call, so a 17 MB release table passed by path on
    // every prediction is parsed once.
    [[nodiscard]] static std::shared_ptr<const NameTable> open(const std::string& path);

    // A table from parallel columns (bindings, tests). Throws on unequal
    // lengths, an empty name or token, or a name given twice.
    [[nodiscard]] static NameTable from_columns(const std::vector<std::string>& names,
                                                const std::vector<std::string>& tokens,
                                                const std::vector<std::string>& sources);

    // The row for `name`, or nullptr.
    [[nodiscard]] const Alias* find(std::string_view name) const;

    [[nodiscard]] std::size_t size() const noexcept { return names_.size(); }
    // The names in the order they were read; name(i) maps to alias(i).
    [[nodiscard]] const std::string& name(std::size_t i) const { return names_.at(i); }
    [[nodiscard]] const Alias& alias(std::size_t i) const { return aliases_.at(i); }
    // The row index of `name`, or -1.
    [[nodiscard]] int64_t row(std::string_view name) const;
    // The file's description of where its rows come from; empty for a table
    // built from columns.
    [[nodiscard]] const std::string& description() const noexcept { return description_; }
    // SHA-256 of the file read; empty for a table built from columns.
    [[nodiscard]] const std::string& sha256() const noexcept { return sha256_; }

private:
    void add(std::string name, std::string token, std::string source);

    std::vector<std::string> names_;
    std::vector<Alias> aliases_;
    std::unordered_map<std::string, std::size_t, NameHash, std::equal_to<>> index_;
    std::string description_;
    std::string sha256_;
};

// What a caller asks of the resolver.
struct NameResolverOptions {
    // The edits a misspelled name may be away from a token or a table name:
    // 0 turns the spelling step off. At 1, misspelled copies of the sPlotOpen
    // example names resolved 99.5% right and none wrong; 2 also renames some
    // species the models never saw to a different species.
    int max_edit_distance = 1;
};

// The largest max_edit_distance the resolver accepts.
inline constexpr int kMaxNameEditDistance = 3;

// What the resolver gives for one name.
struct ResolvedName {
    // The token the name is read as; the name itself for Exact and Unresolved.
    std::string token;
    NameRoute route = NameRoute::Unresolved;
    // Alias: the table's source. Spelling: the source of `via` when it is a
    // table name, empty when it is a token.
    std::string source;
    // Spelling: the nearest entry (a token or a table name).
    std::string via;
    // Spelling: the edits from the name to `via` (0 when only the letter case
    // differs).
    int distance = 0;
};

// How a load resolves a table's species names: the name table (may be null:
// exact and spelling against the vocabulary alone) and the options. A loader
// given one builds a NameResolver over the vocabulary it encodes with.
struct SpeciesNames {
    std::shared_ptr<const NameTable> table;
    NameResolverOptions options;
};

// Resolves names against one species vocabulary. The spelling index is built
// on the first name that needs it, so a table whose names are all exact or
// aliases never pays for it. resolve() is safe to call from several threads.
class NameResolver {
public:
    // `vocabulary` as a checkpoint orders it ([0] = "<UNK>", skipped); `table`
    // may be null.
    NameResolver(const std::vector<std::string>& vocabulary, std::shared_ptr<const NameTable> table,
                 NameResolverOptions options = {});
    ~NameResolver();
    NameResolver(const NameResolver&) = delete;
    NameResolver& operator=(const NameResolver&) = delete;

    [[nodiscard]] ResolvedName resolve(std::string_view name) const;
    [[nodiscard]] std::vector<ResolvedName> resolve(const std::vector<std::string>& names) const;

    [[nodiscard]] const NameResolverOptions& options() const noexcept { return options_; }

private:
    struct Spelling;

    [[nodiscard]] const Spelling& spelling() const;
    [[nodiscard]] ResolvedName nearest(std::string_view name) const;

    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, int32_t, NameHash, std::equal_to<>> codes_;
    std::shared_ptr<const NameTable> table_;
    NameResolverOptions options_;
    mutable std::once_flag spelling_once_;
    mutable std::unique_ptr<Spelling> spelling_;
};

// The names of a table that are not exact, one row per distinct name in the
// order they first appear, with the records that carry each. Columns are
// parallel.
struct SpeciesNameReport {
    std::vector<std::string> name;
    std::vector<std::string> token;
    std::vector<NameRoute> route;
    std::vector<std::string> source;
    std::vector<std::string> via;
    std::vector<int32_t> distance;
    std::vector<int64_t> n_records;

    [[nodiscard]] std::size_t size() const noexcept { return name.size(); }
    void add(const std::string& written, const ResolvedName& resolved, int64_t records);
};

}  // namespace resolve
