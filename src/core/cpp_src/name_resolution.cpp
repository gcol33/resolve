#include "resolve/name_resolution.hpp"

#include "resolve/json.hpp"
#include "resolve/sha256.hpp"

#include <filesystem>
#include <map>
#include <stdexcept>
#include <utility>

namespace resolve {

// =============================================================================
// NameTable
// =============================================================================

void NameTable::add(std::string name, std::string token, std::string source) {
    if (name.empty() || token.empty()) {
        throw std::invalid_argument("NameTable: a row needs a name and a token (name '" + name +
                                    "', token '" + token + "')");
    }
    if (!index_.emplace(name, names_.size()).second) {
        throw std::invalid_argument("NameTable: the name '" + name + "' is given twice");
    }
    names_.push_back(std::move(name));
    aliases_.push_back(Alias{std::move(token), std::move(source)});
}

NameTable NameTable::read(const std::string& path) {
    const json::Value root = json::read_file(path);
    if (!root.is_object()) {
        throw std::runtime_error("NameTable: " + path + " is not a JSON object");
    }
    const json::Value* aliases = root.find("aliases");
    if (aliases == nullptr || !aliases->is_object()) {
        throw std::runtime_error("NameTable: " + path + " has no \"aliases\" object");
    }
    NameTable table;
    table.names_.reserve(aliases->members().size());
    table.aliases_.reserve(aliases->members().size());
    table.index_.reserve(aliases->members().size());
    for (const auto& [name, row] : aliases->members()) {
        if (!row.is_array() || row.items().size() != 2 || !row.items()[0].is_string() ||
            !row.items()[1].is_string()) {
            throw std::runtime_error("NameTable: " + path + ": the row for '" + name +
                                     "' is not [token, source]");
        }
        table.add(name, row.items()[0].as_string(), row.items()[1].as_string());
    }
    if (const json::Value* source = root.find("source"); source != nullptr && source->is_string()) {
        table.description_ = source->as_string();
    }
    table.sha256_ = sha256_file(path);
    return table;
}

std::shared_ptr<const NameTable> NameTable::open(const std::string& path) {
    struct Cached {
        std::uintmax_t size;
        std::filesystem::file_time_type modified;
        std::shared_ptr<const NameTable> table;
    };
    static std::mutex mutex;
    static std::map<std::string, Cached> cache;

    const std::filesystem::path file(path);
    const std::uintmax_t size = std::filesystem::file_size(file);
    const auto modified = std::filesystem::last_write_time(file);
    const std::string key = std::filesystem::absolute(file).lexically_normal().string();
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(key);
        it != cache.end() && it->second.size == size && it->second.modified == modified) {
        return it->second.table;
    }
    auto table = std::make_shared<const NameTable>(read(path));
    cache[key] = Cached{size, modified, table};
    return table;
}

NameTable NameTable::from_columns(const std::vector<std::string>& names,
                                  const std::vector<std::string>& tokens,
                                  const std::vector<std::string>& sources) {
    if (tokens.size() != names.size() || sources.size() != names.size()) {
        throw std::invalid_argument("NameTable: names, tokens and sources must have one length");
    }
    NameTable table;
    for (std::size_t i = 0; i < names.size(); ++i) table.add(names[i], tokens[i], sources[i]);
    return table;
}

int64_t NameTable::row(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? -1 : static_cast<int64_t>(it->second);
}

const NameTable::Alias* NameTable::find(std::string_view name) const {
    const int64_t i = row(name);
    return i < 0 ? nullptr : &aliases_[static_cast<std::size_t>(i)];
}

// =============================================================================
// NameResolver
// =============================================================================

struct NameResolver::Spelling {
    fuzzy::FuzzyIndex index;
    std::vector<int32_t> token_of;  // entry -> vocabulary code
    std::vector<int64_t> alias_of;  // entry -> table row, -1 for a token
};

namespace {

// The genus of a name: its first word, the bucket a query searches first.
std::string genus_of(std::string_view name) {
    return std::string(name.substr(0, name.find(' ')));
}

}  // namespace

NameResolver::NameResolver(const std::vector<std::string>& vocabulary,
                           std::shared_ptr<const NameTable> table, NameResolverOptions options)
    : vocabulary_(vocabulary), table_(std::move(table)), options_(options) {
    if (options_.max_edit_distance < 0 || options_.max_edit_distance > kMaxNameEditDistance) {
        throw std::invalid_argument("NameResolver: max_edit_distance must be in [0, " +
                                    std::to_string(kMaxNameEditDistance) + "], got " +
                                    std::to_string(options_.max_edit_distance));
    }
    codes_.reserve(vocabulary_.size());
    for (std::size_t c = 1; c < vocabulary_.size(); ++c) {
        if (!vocabulary_[c].empty()) codes_.emplace(vocabulary_[c], static_cast<int32_t>(c));
    }
}

NameResolver::~NameResolver() = default;

const NameResolver::Spelling& NameResolver::spelling() const {
    std::call_once(spelling_once_, [this] {
        auto sp = std::make_unique<Spelling>();
        std::vector<std::string> entries;
        for (std::size_t c = 1; c < vocabulary_.size(); ++c) {
            if (vocabulary_[c].empty() || codes_.at(vocabulary_[c]) != static_cast<int32_t>(c)) continue;
            entries.push_back(vocabulary_[c]);
            sp->token_of.push_back(static_cast<int32_t>(c));
            sp->alias_of.push_back(-1);
        }
        if (table_) {
            for (std::size_t i = 0; i < table_->size(); ++i) {
                if (codes_.contains(table_->name(i))) continue;
                const auto code = codes_.find(table_->alias(i).token);
                if (code == codes_.end()) continue;
                entries.push_back(table_->name(i));
                sp->token_of.push_back(code->second);
                sp->alias_of.push_back(static_cast<int64_t>(i));
            }
        }
        fuzzy::BuildOptions build;
        build.max_edit_distance = options_.max_edit_distance;
        build.bucket_fn = genus_of;
        sp->index = fuzzy::FuzzyIndex::build(entries, std::move(build));
        spelling_ = std::move(sp);
    });
    return *spelling_;
}

ResolvedName NameResolver::nearest(std::string_view name) const {
    ResolvedName out;
    out.token = std::string(name);
    const Spelling& sp = spelling();
    fuzzy::QueryOptions query;
    query.max_edit_distance = options_.max_edit_distance;
    query.bucket_hint = genus_of(name);
    // The index keeps the top_n nearest and drops a tie past them, so the
    // query widens until fewer than top_n entries share the best distance:
    // only then are all of them seen.
    for (query.top_n = 8;; query.top_n *= 4) {
        const auto matches = sp.index.query(name, query);
        if (matches.empty()) return out;
        const int best = matches.front().distance;
        std::size_t n_best = 0;
        while (n_best < matches.size() && matches[n_best].distance == best) ++n_best;
        if (n_best == static_cast<std::size_t>(query.top_n) &&
            static_cast<std::size_t>(query.top_n) < sp.index.size()) {
            continue;
        }
        const int32_t token = sp.token_of[static_cast<std::size_t>(matches.front().id)];
        for (std::size_t i = 1; i < n_best; ++i) {
            if (sp.token_of[static_cast<std::size_t>(matches[i].id)] != token) return out;
        }
        const auto& via = matches.front();
        out.token = vocabulary_[static_cast<std::size_t>(token)];
        out.route = NameRoute::Spelling;
        out.via = via.entry;
        out.distance = best;
        if (const int64_t row = sp.alias_of[static_cast<std::size_t>(via.id)]; row >= 0) {
            out.source = table_->alias(static_cast<std::size_t>(row)).source;
        }
        return out;
    }
}

ResolvedName NameResolver::resolve(std::string_view name) const {
    ResolvedName out;
    out.token = std::string(name);
    if (name.empty()) return out;
    if (codes_.contains(name)) {
        out.route = NameRoute::Exact;
        return out;
    }
    if (table_) {
        if (const NameTable::Alias* alias = table_->find(name);
            alias != nullptr && codes_.contains(alias->token)) {
            out.token = alias->token;
            out.route = NameRoute::Alias;
            out.source = alias->source;
            return out;
        }
    }
    if (options_.max_edit_distance == 0) return out;
    return nearest(name);
}

std::vector<ResolvedName> NameResolver::resolve(const std::vector<std::string>& names) const {
    std::vector<ResolvedName> out;
    out.reserve(names.size());
    for (const auto& name : names) out.push_back(resolve(name));
    return out;
}

// =============================================================================
// SpeciesNameReport
// =============================================================================

void SpeciesNameReport::add(const std::string& written, const ResolvedName& resolved,
                            int64_t records) {
    name.push_back(written);
    token.push_back(resolved.token);
    route.push_back(resolved.route);
    source.push_back(resolved.source);
    via.push_back(resolved.via);
    distance.push_back(resolved.distance);
    n_records.push_back(records);
}

}  // namespace resolve
