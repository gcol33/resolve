#include "resolve/suite.hpp"

#include "resolve/config_registry.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/number_text.hpp"
#include "resolve/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace resolve {

namespace fs = std::filesystem;

// ============================================================================
// Manifest <-> JSON
// ============================================================================

namespace {

// Collects every problem a manifest has, so one read reports all of them.
class Problems {
public:
    void add(std::string where, const std::string& what) {
        items_.push_back(std::move(where) + ": " + what);
    }
    [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
    [[noreturn]] void raise(const std::string& heading) const {
        std::string msg = heading;
        for (const auto& item : items_) msg += "\n  - " + item;
        throw std::runtime_error(msg);
    }

private:
    std::vector<std::string> items_;
};

// Read helpers over one JSON object that name the member's path on error and
// reject a member the format does not define, so a misspelt key fails rather
// than silently taking its default.
class ObjectReader {
public:
    ObjectReader(const json::Value& object, std::string where)
        : object_(object), where_(std::move(where)) {
        if (!object_.is_object()) fail("", std::string("expected an object, found ") +
                                              json::kind_name(object_.kind()));
    }

    [[nodiscard]] std::string path(const std::string& key) const {
        return where_.empty() ? key : where_ + "." + key;
    }

    const json::Value* get(const std::string& key) {
        seen_.insert(key);
        return object_.find(key);
    }

    std::string string(const std::string& key, bool required) {
        const json::Value* v = get(key);
        if (!v) {
            if (required) fail(key, "is required");
            return {};
        }
        if (!v->is_string()) fail(key, "must be a string");
        return v->as_string();
    }

    std::vector<std::string> strings(const std::string& key) {
        const json::Value* v = get(key);
        if (!v) return {};
        if (!v->is_array()) fail(key, "must be an array of strings");
        std::vector<std::string> out;
        for (const auto& item : v->items()) {
            if (!item.is_string()) fail(key, "must be an array of strings");
            out.push_back(item.as_string());
        }
        return out;
    }

    bool boolean(const std::string& key, bool fallback) {
        const json::Value* v = get(key);
        if (!v) return fallback;
        if (!v->is_bool()) fail(key, "must be true or false");
        return v->as_bool();
    }

    double number(const std::string& key, double fallback) {
        const json::Value* v = get(key);
        if (!v) return fallback;
        if (!v->is_number()) fail(key, "must be a number");
        return v->as_number();
    }

    int64_t integer(const std::string& key, bool required) {
        const json::Value* v = get(key);
        if (!v) {
            if (required) fail(key, "is required");
            return 0;
        }
        try {
            return v->as_int();
        } catch (const std::exception&) {
            fail(key, "must be an integer");
        }
    }

    json::Value object_or_empty(const std::string& key) {
        const json::Value* v = get(key);
        if (!v) return json::Value::object();
        if (!v->is_object()) fail(key, "must be an object");
        return *v;
    }

    const json::Value& required(const std::string& key) {
        const json::Value* v = get(key);
        if (!v) fail(key, "is required");
        return *v;
    }

    // Every member of the object must have been asked for.
    void finish() const {
        for (const auto& [key, value] : object_.members()) {
            (void)value;
            if (!seen_.count(key)) fail(key, "is not a member of the suite manifest format");
        }
    }

    [[noreturn]] void fail(const std::string& key, const std::string& what) const {
        const std::string where = key.empty() ? (where_.empty() ? "manifest" : where_) : path(key);
        throw std::runtime_error("suite manifest: " + where + " " + what);
    }

private:
    const json::Value& object_;
    std::string where_;
    std::unordered_set<std::string> seen_;
};

template <typename EnumT>
EnumT enum_member(ObjectReader& reader, const std::string& key) {
    const std::string name = reader.string(key, /*required=*/true);
    try {
        return enum_from_name<EnumT>(name);
    } catch (const std::exception& e) {
        reader.fail(key, e.what());
    }
}

bool is_hex_digest(const std::string& s) {
    if (s.size() != 64) return false;
    for (const char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// A member path must stay inside the suite directory and read the same on
// every platform: relative, '/'-separated, no '..' or '.' segment.
std::string member_path_problem(const std::string& file) {
    if (file.empty()) return "is empty";
    if (file.front() == '/') return "must be relative to the suite directory";
    if (file.find(':') != std::string::npos) return "must not name a drive";
    if (file.find('\\') != std::string::npos) return "must use '/' as the separator";
    std::size_t start = 0;
    while (start <= file.size()) {
        const std::size_t end = std::min(file.find('/', start), file.size());
        const std::string segment = file.substr(start, end - start);
        if (segment.empty() || segment == "." || segment == "..") {
            return "must not contain an empty, '.' or '..' segment";
        }
        start = end + 1;
    }
    return {};
}

fs::path member_location(const std::string& dir, const std::string& file) {
    return fs::path(dir) / fs::path(file);
}

json::Value inputs_to_json(const SuiteInputContract& in) {
    json::Value o = json::Value::object();
    o.set("plot_id", in.plot_id);
    o.set("species", in.species);
    o.set("abundance", in.abundance);
    o.set("genus", in.genus);
    o.set("family", in.family);
    o.set("latitude", in.latitude);
    o.set("longitude", in.longitude);
    o.set("covariates", json::Value::array_of(in.covariates));
    o.set("categoricals", json::Value::array_of(in.categoricals));
    o.set("abundance_units", in.abundance_units);
    o.set("zero_abundance_as", static_cast<double>(in.zero_abundance_as));
    o.set("notes", in.notes);
    return o;
}

SuiteInputContract inputs_from_json(const json::Value& value) {
    ObjectReader r(value, "inputs");
    SuiteInputContract in;
    in.plot_id = r.string("plot_id", true);
    in.species = r.string("species", true);
    in.abundance = r.string("abundance", false);
    in.genus = r.string("genus", false);
    in.family = r.string("family", false);
    in.latitude = r.string("latitude", false);
    in.longitude = r.string("longitude", false);
    in.covariates = r.strings("covariates");
    in.categoricals = r.strings("categoricals");
    in.abundance_units = r.string("abundance_units", false);
    in.zero_abundance_as = static_cast<float>(r.number("zero_abundance_as", 0.0));
    in.notes = r.string("notes", false);
    r.finish();
    return in;
}

json::Value target_inputs_to_json(const SuiteTargetInputs& in) {
    json::Value o = json::Value::object();
    o.set("coordinates", in.coordinates);
    o.set("covariates", json::Value::array_of(in.covariates));
    o.set("categoricals", json::Value::array_of(in.categoricals));
    return o;
}

SuiteTargetInputs target_inputs_from_json(const json::Value& value, const std::string& where) {
    ObjectReader r(value, where);
    SuiteTargetInputs in;
    in.coordinates = r.boolean("coordinates", false);
    in.covariates = r.strings("covariates");
    in.categoricals = r.strings("categoricals");
    r.finish();
    return in;
}

json::Value target_to_json(const SuiteTarget& t) {
    json::Value o = json::Value::object();
    o.set("name", t.name);
    o.set("task", enum_to_name(t.task));
    o.set("combine", enum_to_name(t.combine));
    o.set("outputs", json::Value::array_of(t.outputs));
    o.set("units", t.units);
    if (t.inputs) o.set("inputs", target_inputs_to_json(*t.inputs));
    if (combine_rule(t.combine).needs_period) o.set("period", t.period);
    o.set("status", enum_to_name(t.status));
    if (!t.limit.empty()) o.set("limit", t.limit);
    json::Value members = json::Value::array();
    for (const auto& m : t.members) {
        json::Value mo = json::Value::object();
        mo.set("seed", m.seed);
        mo.set("file", m.file);
        mo.set("sha256", m.sha256);
        mo.set("bytes", m.bytes);
        if (m.zero_abundance_as) mo.set("zero_abundance_as", *m.zero_abundance_as);
        members.push_back(std::move(mo));
    }
    o.set("members", std::move(members));
    o.set("training", t.training);
    o.set("validation", t.validation);
    return o;
}

SuiteTarget target_from_json(const json::Value& value, const std::string& where) {
    ObjectReader r(value, where);
    SuiteTarget t;
    t.name = r.string("name", true);
    t.task = enum_member<TaskType>(r, "task");
    t.combine = enum_member<SuiteCombine>(r, "combine");
    t.outputs = r.strings("outputs");
    t.units = r.string("units", false);
    if (const json::Value* in = r.get("inputs")) {
        t.inputs = target_inputs_from_json(*in, r.path("inputs"));
    }
    t.period = r.number("period", 0.0);
    t.status = enum_member<SuiteTargetStatus>(r, "status");
    t.limit = r.string("limit", false);
    const json::Value& members = r.required("members");
    if (!members.is_array()) r.fail("members", "must be an array");
    for (std::size_t i = 0; i < members.items().size(); ++i) {
        ObjectReader mr(members.items()[i], r.path("members") + "[" + std::to_string(i) + "]");
        SuiteMember m;
        m.seed = mr.integer("seed", true);
        m.file = mr.string("file", true);
        m.sha256 = mr.string("sha256", false);
        m.bytes = mr.integer("bytes", false);
        if (mr.get("zero_abundance_as") != nullptr) {
            m.zero_abundance_as = static_cast<float>(mr.number("zero_abundance_as", 0.0));
        }
        mr.finish();
        t.members.push_back(std::move(m));
    }
    t.training = r.object_or_empty("training");
    t.validation = r.object_or_empty("validation");
    r.finish();
    return t;
}

}  // namespace

json::Value SuiteManifest::to_json() const {
    json::Value o = json::Value::object();
    o.set("format", kSuiteFormat);
    o.set("format_version", format_version());
    o.set("name", name);
    o.set("description", description);
    o.set("licence", licence);
    o.set("engine_version", engine_version);
    o.set("taxonomy", taxonomy);
    o.set("training_scope", training_scope);
    o.set("inputs", inputs_to_json(inputs));
    json::Value ts = json::Value::array();
    for (const auto& t : targets) ts.push_back(target_to_json(t));
    o.set("targets", std::move(ts));
    return o;
}

SuiteManifest SuiteManifest::from_json(const json::Value& value) {
    ObjectReader r(value, "");
    const std::string format = r.string("format", true);
    if (format != kSuiteFormat) {
        r.fail("format", "is '" + format + "', not '" + kSuiteFormat + "'");
    }
    const int64_t version = r.integer("format_version", true);
    if (version < 1 || version > kSuiteFormatVersion) {
        r.fail("format_version", "is " + std::to_string(version) + "; this engine reads "
               "version " + std::to_string(kSuiteFormatVersion) + " and below. A newer "
               "engine is needed to read this suite.");
    }
    SuiteManifest m;
    m.name = r.string("name", true);
    m.description = r.string("description", false);
    m.licence = r.string("licence", false);
    m.engine_version = r.string("engine_version", false);
    m.taxonomy = r.string("taxonomy", false);
    m.training_scope = r.string("training_scope", false);
    m.inputs = inputs_from_json(r.required("inputs"));
    const json::Value& targets = r.required("targets");
    if (!targets.is_array()) r.fail("targets", "must be an array");
    for (std::size_t i = 0; i < targets.items().size(); ++i) {
        m.targets.push_back(
            target_from_json(targets.items()[i], "targets[" + std::to_string(i) + "]"));
    }
    r.finish();
    if (version < m.format_version()) {
        r.fail("format_version", "is " + std::to_string(version) + ", but the manifest uses "
               "features that need version " + std::to_string(m.format_version()) +
               " (per-target inputs and the spherical mean need version 2, a member's own "
               "zero_abundance_as version 3)");
    }
    return m;
}

SuiteTargetInputs SuiteManifest::inputs_of(const SuiteTarget& target) const {
    if (target.inputs) return *target.inputs;
    return SuiteTargetInputs{inputs.reads_coordinates(), inputs.covariates, inputs.categoricals};
}

int SuiteManifest::format_version() const {
    int version = 1;
    for (const auto& t : targets) {
        version = std::max(version, combine_rule(t.combine).format_version);
        if (t.inputs) version = std::max(version, 2);
        for (const auto& m : t.members) {
            if (m.zero_abundance_as) version = std::max(version, 3);
        }
    }
    return version;
}

void SuiteManifest::validate() const {
    Problems problems;
    auto require = [&](const std::string& where, const std::string& value) {
        if (value.empty()) problems.add(where, "is required");
    };
    require("name", name);
    require("licence", licence);
    require("engine_version", engine_version);
    require("taxonomy", taxonomy);
    require("training_scope", training_scope);

    require("inputs.plot_id", inputs.plot_id);
    require("inputs.species", inputs.species);
    if (inputs.latitude.empty() != inputs.longitude.empty()) {
        problems.add("inputs", "latitude and longitude are read together or not at all");
    }
    if (!std::isfinite(inputs.zero_abundance_as) || inputs.zero_abundance_as < 0.0f) {
        problems.add("inputs.zero_abundance_as", "must be a finite value >= 0");
    }
    if (inputs.zero_abundance_as != 0.0f && inputs.abundance.empty()) {
        problems.add("inputs.zero_abundance_as", "is set, but no abundance column is read");
    }

    if (targets.empty()) problems.add("targets", "must hold at least one target");
    std::unordered_set<std::string> names;
    for (std::size_t i = 0; i < targets.size(); ++i) {
        const SuiteTarget& t = targets[i];
        const std::string where = "targets[" + std::to_string(i) + "]";
        if (t.name.empty()) problems.add(where + ".name", "is required");
        else if (!names.insert(t.name).second) {
            problems.add(where + ".name", "'" + t.name + "' names two targets");
        }

        const SuiteCombineRule& rule = combine_rule(t.combine);
        if (t.task != rule.task) {
            problems.add(where, std::string(rule.noun) + " combines " + rule.combines +
                         "; task must be " + enum_to_name(rule.task));
        }
        if (t.outputs.size() < rule.min_outputs || t.outputs.size() > rule.max_outputs) {
            problems.add(where + ".outputs", std::string(rule.noun) + " reads " + rule.outputs);
        }
        if (rule.required_units != nullptr && t.units != rule.required_units) {
            problems.add(where + ".units", std::string(rule.noun) + " reads its outputs in " +
                         rule.required_units + "; units must be '" + rule.required_units + "'");
        }
        if (rule.needs_period && !(std::isfinite(t.period) && t.period > 0.0)) {
            problems.add(where + ".period", std::string(rule.noun) + " needs a positive period");
        }
        if (!rule.needs_period && t.period != 0.0) {
            problems.add(where + ".period", "is read only by a combine rule with a period");
        }
        for (const auto& out : t.outputs) {
            if (out.empty()) problems.add(where + ".outputs", "an output name is empty");
        }
        if (t.status == SuiteTargetStatus::Experimental && t.limit.empty()) {
            problems.add(where + ".limit", "an experimental target states its limit");
        }

        if (t.members.empty()) problems.add(where + ".members", "must hold at least one member");
        std::unordered_set<int64_t> seeds;
        std::unordered_set<std::string> files;
        for (std::size_t j = 0; j < t.members.size(); ++j) {
            const SuiteMember& m = t.members[j];
            const std::string mw = where + ".members[" + std::to_string(j) + "]";
            if (!seeds.insert(m.seed).second) {
                problems.add(mw + ".seed", std::to_string(m.seed) + " appears twice");
            }
            if (const std::string p = member_path_problem(m.file); !p.empty()) {
                problems.add(mw + ".file", p);
            } else if (!files.insert(m.file).second) {
                problems.add(mw + ".file", "'" + m.file + "' appears twice");
            }
            if (!is_hex_digest(m.sha256)) {
                problems.add(mw + ".sha256", "must be 64 lower-case hex digits (seal the "
                             "manifest to fill it)");
            }
            if (m.bytes <= 0) problems.add(mw + ".bytes", "must be positive (seal the manifest)");
            if (m.zero_abundance_as &&
                !(std::isfinite(*m.zero_abundance_as) && *m.zero_abundance_as >= 0.0f)) {
                problems.add(mw + ".zero_abundance_as", "must be a finite value >= 0");
            }
        }

        if (t.inputs) {
            const SuiteTargetInputs& in = *t.inputs;
            if (in.coordinates && !inputs.reads_coordinates()) {
                problems.add(where + ".inputs.coordinates",
                             "the target reads coordinates, the input contract names none");
            }
            auto subset = [&](const char* key, const std::vector<std::string>& read,
                              const std::vector<std::string>& contract) {
                std::unordered_set<std::string> seen;
                for (const auto& column : read) {
                    if (std::find(contract.begin(), contract.end(), column) == contract.end()) {
                        problems.add(where + ".inputs." + key, "'" + column +
                                     "' is not a column of the input contract");
                    } else if (!seen.insert(column).second) {
                        problems.add(where + ".inputs." + key, "'" + column + "' appears twice");
                    }
                }
            };
            subset("covariates", in.covariates, inputs.covariates);
            subset("categoricals", in.categoricals, inputs.categoricals);
        }
    }

    // The contract is the union of what the targets read: a column no target
    // reads would make a caller supply data no model looks at.
    if (!targets.empty()) {
        bool coordinates = false;
        std::unordered_set<std::string> covariates, categoricals;
        for (const auto& t : targets) {
            const SuiteTargetInputs read = inputs_of(t);
            coordinates = coordinates || read.coordinates;
            covariates.insert(read.covariates.begin(), read.covariates.end());
            categoricals.insert(read.categoricals.begin(), read.categoricals.end());
        }
        if (inputs.reads_coordinates() && !coordinates) {
            problems.add("inputs", "names coordinates, but no target reads them");
        }
        for (const auto& c : inputs.covariates) {
            if (!covariates.count(c)) {
                problems.add("inputs.covariates", "'" + c + "' is read by no target");
            }
        }
        for (const auto& c : inputs.categoricals) {
            if (!categoricals.count(c)) {
                problems.add("inputs.categoricals", "'" + c + "' is read by no target");
            }
        }
    }

    if (!problems.empty()) problems.raise("suite manifest '" + name + "' is invalid:");
}

const SuiteTarget& SuiteManifest::target(const std::string& target_name) const {
    for (const auto& t : targets) {
        if (t.name == target_name) return t;
    }
    std::string known;
    for (const auto& t : targets) known += (known.empty() ? "" : ", ") + t.name;
    throw std::runtime_error("suite '" + name + "' has no target '" + target_name +
                             "'; it has: " + known);
}

SuiteManifest SuiteManifest::read(const std::string& dir) {
    const std::string path = (fs::path(dir) / kSuiteManifestFile).string();
    if (!fs::exists(path)) {
        throw std::runtime_error("no suite manifest at '" + path + "'");
    }
    SuiteManifest m = from_json(json::read_file(path));
    m.validate();
    return m;
}

void SuiteManifest::write(const std::string& dir) const {
    validate();
    json::write_file((fs::path(dir) / kSuiteManifestFile).string(), to_json());
}

void SuiteManifest::seal(const std::string& dir) {
    for (auto& t : targets) {
        for (auto& m : t.members) {
            if (const std::string p = member_path_problem(m.file); !p.empty()) {
                throw std::runtime_error("suite target '" + t.name + "' member file '" +
                                         m.file + "' " + p);
            }
            const fs::path where = member_location(dir, m.file);
            if (!fs::exists(where)) {
                throw std::runtime_error("suite target '" + t.name + "': no member file at '" +
                                         where.string() + "'");
            }
            m.bytes = static_cast<int64_t>(fs::file_size(where));
            m.sha256 = sha256_file(where.string());
        }
    }
}

namespace {

std::string member_problem(const std::string& dir, const std::string& target,
                           const SuiteMember& m) {
    const fs::path where = member_location(dir, m.file);
    const std::string label = "target '" + target + "' seed " + std::to_string(m.seed) +
                              " (" + m.file + ")";
    if (!fs::exists(where)) return label + ": file missing";
    const auto bytes = static_cast<int64_t>(fs::file_size(where));
    if (bytes != m.bytes) {
        return label + ": " + std::to_string(bytes) + " bytes, the manifest records " +
               std::to_string(m.bytes);
    }
    const std::string digest = sha256_file(where.string());
    if (digest != m.sha256) {
        return label + ": SHA-256 " + digest + ", the manifest records " + m.sha256;
    }
    return {};
}

}  // namespace

std::vector<std::string> SuiteManifest::verify(const std::string& dir,
                                               const std::vector<std::string>& names) const {
    std::vector<std::string> problems;
    for (const auto& t : targets) {
        if (!names.empty() && std::find(names.begin(), names.end(), t.name) == names.end()) {
            continue;
        }
        for (const auto& m : t.members) {
            if (std::string p = member_problem(dir, t.name, m); !p.empty()) {
                problems.push_back(std::move(p));
            }
        }
    }
    return problems;
}

// ============================================================================
// Input
// ============================================================================

SuiteInput SuiteInput::csv(std::string header_path, std::string species_path) {
    if (species_path.empty()) throw std::invalid_argument("SuiteInput: a species CSV is required");
    SuiteInput in;
    in.header_path_ = std::move(header_path);
    in.species_path_ = std::move(species_path);
    return in;
}

SuiteInput SuiteInput::tables(const ColumnTable* header, const ColumnTable& species) {
    SuiteInput in;
    in.header_table_ = header;
    in.species_table_ = &species;
    return in;
}

bool SuiteInput::has_header() const noexcept {
    return header_table_ != nullptr || !header_path_.empty();
}

ResolveDataset SuiteInput::encode(const RoleMapping& roles, const ExternalVocabs& vocabs,
                                  const DatasetConfig& config) const {
    // No targets: the plots to score carry none, and a loader asked for one
    // would drop every plot where it is missing.
    const std::vector<TargetSpec> no_targets;
    if (species_table_ != nullptr) {
        if (header_table_ != nullptr) {
            return ResolveDataset::from_dataframe_with_vocabs(
                *header_table_, *species_table_, roles, no_targets, vocabs, config);
        }
        return ResolveDataset::from_species_dataframe_with_vocabs(
            *species_table_, roles, no_targets, vocabs, config);
    }
    if (!header_path_.empty()) {
        return ResolveDataset::from_csv_with_vocabs(
            header_path_, species_path_, roles, no_targets, vocabs, config);
    }
    return ResolveDataset::from_species_csv_with_vocabs(
        species_path_, roles, no_targets, vocabs, config);
}

// ============================================================================
// Combination rules
// ============================================================================

namespace {

void require_stack(const torch::Tensor& t, const char* what) {
    if (!t.defined() || t.dim() != 2 || t.size(0) < 1) {
        throw std::invalid_argument(std::string(what) +
                                    ": expected a (n_members, n_plots) tensor with at least one member");
    }
}

constexpr double kTwoPi = 2.0 * std::numbers::pi;

}  // namespace

CombinedPrediction combine_vote(const torch::Tensor& member_codes, int64_t n_classes,
                                const torch::Tensor& mean_probabilities) {
    require_stack(member_codes, "combine_vote");
    if (n_classes < 1) throw std::invalid_argument("combine_vote: n_classes must be >= 1");
    const auto codes = member_codes.to(torch::kCPU, torch::kLong);
    if (codes.numel() > 0 &&
        (codes.min().item<int64_t>() < 0 || codes.max().item<int64_t>() >= n_classes)) {
        throw std::invalid_argument("combine_vote: a class code lies outside [0, n_classes)");
    }
    const int64_t n_members = codes.size(0);
    const int64_t n_plots = codes.size(1);
    // (n_plots, n_classes) votes, and which classes hold the most.
    const auto votes = torch::one_hot(codes, n_classes).sum(0);
    const auto leading = votes == std::get<0>(votes.max(1, /*keepdim=*/true));
    // Among the leading classes the highest mean probability wins; argmax
    // returns the first maximum, so a tie that remains goes to the lowest code.
    torch::Tensor score = leading.to(torch::kFloat64);
    if (mean_probabilities.defined()) {
        if (mean_probabilities.dim() != 2 || mean_probabilities.size(0) != n_plots ||
            mean_probabilities.size(1) != n_classes) {
            throw std::invalid_argument(
                "combine_vote: mean_probabilities must be (n_plots, n_classes) = (" +
                std::to_string(n_plots) + ", " + std::to_string(n_classes) + ")");
        }
        score = mean_probabilities.to(torch::kCPU, torch::kFloat64)
                    .masked_fill(leading.logical_not(),
                                 -std::numeric_limits<double>::infinity());
    }
    CombinedPrediction out;
    out.value = score.argmax(1);
    out.agreement = votes.gather(1, out.value.unsqueeze(1)).squeeze(1).to(torch::kFloat32) /
                    static_cast<float>(n_members);
    return out;
}

CombinedPrediction combine_mean(const torch::Tensor& member_values) {
    require_stack(member_values, "combine_mean");
    const auto values = member_values.to(torch::kCPU, torch::kFloat64);
    CombinedPrediction out;
    out.value = values.mean(0).to(torch::kFloat32);
    out.dispersion = values.size(0) > 1
        ? values.std(/*dim=*/{0}, /*unbiased=*/true, /*keepdim=*/false).to(torch::kFloat32)
        : torch::full({values.size(1)}, std::numeric_limits<float>::quiet_NaN());
    return out;
}

CombinedPrediction combine_circular(const torch::Tensor& member_bearings, double period) {
    require_stack(member_bearings, "combine_circular");
    if (!(std::isfinite(period) && period > 0.0)) {
        throw std::invalid_argument("combine_circular: period must be positive");
    }
    const auto radians = member_bearings.to(torch::kCPU, torch::kFloat64) * (kTwoPi / period);
    const auto s = radians.sin().mean(0);
    const auto c = radians.cos().mean(0);
    // Mean resultant length R in [0, 1]; the circular standard deviation
    // sqrt(-2 ln R) is 0 when every member names the same bearing and grows
    // without bound as the bearings spread round the circle. Members whose
    // vectors cancel name no bearing: both are left undefined rather than
    // chosen arbitrarily.
    const auto resultant = torch::sqrt(s * s + c * c).clamp_max(1.0);
    const auto cancelled = resultant < kCancelledResultant;
    const auto nan = torch::full_like(resultant, std::numeric_limits<double>::quiet_NaN());
    CombinedPrediction out;
    out.value = torch::where(cancelled, nan,
                             torch::remainder(torch::atan2(s, c) * (period / kTwoPi), period))
                    .to(torch::kFloat32);
    out.dispersion = member_bearings.size(0) == 1
        ? torch::full({member_bearings.size(1)}, std::numeric_limits<float>::quiet_NaN())
        : torch::where(cancelled, nan,
                       torch::sqrt(-2.0 * torch::log(resultant)) * (period / kTwoPi))
              .to(torch::kFloat32);
    return out;
}

CombinedPrediction combine_spherical(const torch::Tensor& member_coordinates) {
    const torch::Tensor& c = member_coordinates;
    if (!c.defined() || c.dim() != 3 || c.size(0) < 1 || c.size(2) != 2) {
        throw std::invalid_argument("combine_spherical: expected a (n_members, n_plots, 2) "
                                    "tensor of latitude / longitude with at least one member");
    }
    constexpr double kRadians = std::numbers::pi / 180.0;
    const auto degrees = c.to(torch::kCPU, torch::kFloat64);
    const auto lat = degrees.select(2, 0) * kRadians;
    const auto lon = degrees.select(2, 1) * kRadians;
    // (n_members, n_plots, 3) unit vectors.
    const auto points = torch::stack({lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()}, 2);
    const auto sum = points.sum(0);
    const auto norm = sum.norm(2, /*dim=*/{1}, /*keepdim=*/true);
    // Members whose vectors cancel name no direction: the combined point is
    // left undefined rather than chosen arbitrarily.
    const auto resultant = norm / static_cast<double>(c.size(0));
    const auto centre = torch::where(resultant >= kCancelledResultant, sum / norm,
                                     torch::full_like(sum, std::numeric_limits<double>::quiet_NaN()));
    const auto x = centre.select(1, 0);
    const auto y = centre.select(1, 1);
    const auto z = centre.select(1, 2);
    CombinedPrediction out;
    out.value = torch::stack({torch::atan2(z, torch::sqrt(x * x + y * y)), torch::atan2(y, x)}, 1)
                    .div(kRadians)
                    .to(torch::kFloat32);
    if (c.size(0) == 1) {
        out.dispersion = torch::full({c.size(1)}, std::numeric_limits<float>::quiet_NaN());
        return out;
    }
    // Great-circle angle atan2(|a x b|, a . b), well conditioned at small angles.
    const auto across = torch::linalg_cross(points, centre.unsqueeze(0).expand_as(points), 2);
    const auto angle = torch::atan2(across.norm(2, /*dim=*/{2}), (points * centre.unsqueeze(0)).sum(2));
    out.dispersion = (torch::quantile(angle, 0.5, /*dim=*/0) * kEarthRadiusKm).to(torch::kFloat32);
    return out;
}

torch::Tensor bearing_from_components(const torch::Tensor& sine, const torch::Tensor& cosine,
                                      double period) {
    if (!(std::isfinite(period) && period > 0.0)) {
        throw std::invalid_argument("bearing_from_components: period must be positive");
    }
    const auto angle = torch::atan2(sine.to(torch::kFloat64), cosine.to(torch::kFloat64));
    return torch::remainder(angle * (period / kTwoPi), period).to(torch::kFloat32);
}

// ============================================================================
// Combine rules
// ============================================================================

namespace {

torch::Tensor member_head(const ResolvePredictions& member, const std::string& name) {
    return member.predictions.at(name).detach().to(torch::kCPU);
}

SuiteMemberOutput extract_vote(const SuiteTarget& target, const ResolvePredictions& member) {
    const std::string& head = target.outputs.front();
    return {member_head(member, head).to(torch::kLong),
            member.probabilities.at(head).detach().to(torch::kCPU)};
}

SuiteMemberOutput extract_mean(const SuiteTarget& target, const ResolvePredictions& member) {
    return {member_head(member, target.outputs.front()).to(torch::kFloat32), {}};
}

// A bearing from one head, or from a sine / cosine pair.
SuiteMemberOutput extract_bearing(const SuiteTarget& target, const ResolvePredictions& member) {
    if (target.outputs.size() == 2) {
        return {bearing_from_components(member_head(member, target.outputs[0]),
                                        member_head(member, target.outputs[1]), target.period),
                {}};
    }
    return {torch::remainder(member_head(member, target.outputs.front()).to(torch::kFloat64),
                             target.period)
                .to(torch::kFloat32),
            {}};
}

SuiteMemberOutput extract_coordinates(const SuiteTarget& target, const ResolvePredictions& member) {
    return {torch::stack({member_head(member, target.outputs[0]).to(torch::kFloat32),
                          member_head(member, target.outputs[1]).to(torch::kFloat32)},
                         1),
            {}};
}

CombinedPrediction combine_by_vote(const torch::Tensor& members, const torch::Tensor& probabilities,
                                   const SuiteTarget&) {
    return combine_vote(members, probabilities.size(1), probabilities);
}

CombinedPrediction combine_by_mean(const torch::Tensor& members, const torch::Tensor&,
                                   const SuiteTarget&) {
    return combine_mean(members);
}

CombinedPrediction combine_by_circular_mean(const torch::Tensor& members, const torch::Tensor&,
                                            const SuiteTarget& target) {
    return combine_circular(members, target.period);
}

CombinedPrediction combine_by_spherical_mean(const torch::Tensor& members, const torch::Tensor&,
                                             const SuiteTarget&) {
    return combine_spherical(members);
}

// In SuiteCombine order.
const SuiteCombineRule kSuiteCombineRules[] = {
    {SuiteCombine::Mean, "a mean", TaskType::Regression, "values", 1, 1, "one output", false,
     nullptr, 1, DispersionUnits::TargetUnits, {"", nullptr}, "_sd",
     "members' standard deviation", &extract_mean, &combine_by_mean},
    {SuiteCombine::Vote, "a vote", TaskType::Classification, "class predictions", 1, 1,
     "one output", false, nullptr, 1, DispersionUnits::None, {"", nullptr}, nullptr,
     nullptr, &extract_vote, &combine_by_vote},
    {SuiteCombine::CircularMean, "a circular mean", TaskType::Regression, "bearings", 1, 2,
     "one output (the bearing) or two (its sine, then its cosine)", true, nullptr, 1,
     DispersionUnits::TargetUnits, {"", nullptr}, "_circular_sd",
     "members' circular standard deviation", &extract_bearing, &combine_by_circular_mean},
    {SuiteCombine::SphericalMean, "a spherical mean", TaskType::Regression, "coordinates", 2, 2,
     "two outputs (latitude, then longitude)", false, "degrees", 2,
     DispersionUnits::Kilometres, {"_latitude", "_longitude"}, "_spread_km",
     "median great-circle distance of the members from the combined point, km",
     &extract_coordinates, &combine_by_spherical_mean},
};

}  // namespace

std::span<const SuiteCombineRule> suite_combine_rules() { return kSuiteCombineRules; }

const SuiteCombineRule& combine_rule(SuiteCombine combine) {
    for (const auto& rule : kSuiteCombineRules) {
        if (rule.combine == combine) return rule;
    }
    throw std::invalid_argument("suite: no combine rule for value " +
                                std::to_string(static_cast<int>(combine)));
}

// ============================================================================
// Predictions as columns
// ============================================================================

namespace {

SuiteColumn number_column(std::string name, const torch::Tensor& values, int64_t n_rows,
                          bool probability = false) {
    SuiteColumn column;
    column.name = std::move(name);
    column.kind = SuiteColumn::Kind::Number;
    column.probability = probability;
    if (!values.defined()) {
        column.numbers.assign(static_cast<size_t>(n_rows), std::numeric_limits<float>::quiet_NaN());
        return column;
    }
    const auto v = values.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* data = v.numel() > 0 ? v.data_ptr<float>() : nullptr;
    column.numbers.assign(data, data + v.numel());
    return column;
}

SuiteColumn label_column(std::string name, const torch::Tensor& codes,
                         const std::vector<std::string>& class_names) {
    SuiteColumn column;
    column.name = std::move(name);
    column.kind = SuiteColumn::Kind::Text;
    const auto c = codes.to(torch::kCPU).to(torch::kLong).contiguous();
    const int64_t* data = c.numel() > 0 ? c.data_ptr<int64_t>() : nullptr;
    column.text.reserve(static_cast<size_t>(c.numel()));
    for (int64_t i = 0; i < c.numel(); ++i) column.text.push_back(class_label(class_names, data[i]));
    return column;
}

// Component `i` of a value laid out as `rule.value_columns`: the tensor itself
// for one column, its column i for two.
torch::Tensor value_component(const torch::Tensor& value, const SuiteCombineRule& rule, std::size_t i) {
    return rule.n_value_columns() == 1 ? value : value.select(-1, static_cast<int64_t>(i));
}

}  // namespace

std::vector<SuiteColumn> suite_prediction_columns(const SuitePredictions& predictions,
                                                  bool probabilities) {
    const auto n = static_cast<int64_t>(predictions.plot_ids.size());
    std::vector<SuiteColumn> columns;
    SuiteColumn plot_ids;
    plot_ids.name = "plot_id";
    plot_ids.kind = SuiteColumn::Kind::Text;
    plot_ids.text = predictions.plot_ids;
    columns.push_back(std::move(plot_ids));

    for (const auto& t : predictions.targets) {
        const SuiteCombineRule& rule = combine_rule(t.combine);
        const std::string& name = t.name;
        if (rule.task == TaskType::Classification) {
            columns.push_back(label_column(name, t.value, t.class_names));
            SuiteColumn codes;
            codes.name = name + "_code";
            codes.kind = SuiteColumn::Kind::Integer;
            const auto c = t.value.to(torch::kCPU).to(torch::kLong).contiguous();
            const int64_t* data = c.numel() > 0 ? c.data_ptr<int64_t>() : nullptr;
            codes.integers.assign(data, data + c.numel());
            columns.push_back(std::move(codes));
            columns.push_back(number_column(name + "_agreement", t.agreement, n));
            if (probabilities && t.probabilities.defined()) {
                for (int64_t k = 0; k < t.probabilities.size(1); ++k) {
                    columns.push_back(number_column(name + "_prob_" + class_label(t.class_names, k),
                                                    t.probabilities.select(1, k), n,
                                                    /*probability=*/true));
                }
            }
        } else {
            for (std::size_t i = 0; i < rule.n_value_columns(); ++i) {
                columns.push_back(number_column(name + rule.value_columns[i],
                                                value_component(t.value, rule, i), n));
            }
        }
        if (rule.dispersion_column != nullptr) {
            columns.push_back(number_column(name + rule.dispersion_column, t.dispersion, n));
        }
        columns.push_back(number_column(name + "_n_species", t.recognition.n_species, n));
        columns.push_back(number_column(name + "_n_recognised", t.recognition.n_recognised, n));
        columns.push_back(number_column(name + "_recognised_share", t.recognition.count_share, n));
        columns.push_back(number_column(name + "_recognised_abundance_share",
                                        t.recognition.abundance_share, n));
        if (t.members.defined()) {
            for (std::size_t m = 0; m < t.member_seeds.size(); ++m) {
                const std::string seed = name + "_seed" + std::to_string(t.member_seeds[m]);
                const torch::Tensor member = t.members.select(0, static_cast<int64_t>(m));
                if (rule.task == TaskType::Classification) {
                    columns.push_back(label_column(seed, member, t.class_names));
                    continue;
                }
                for (std::size_t i = 0; i < rule.n_value_columns(); ++i) {
                    columns.push_back(number_column(seed + rule.value_columns[i],
                                                    value_component(member, rule, i), n));
                }
            }
        }
    }
    return columns;
}

// ============================================================================
// SuitePredictor
// ============================================================================

struct SuitePredictor::Member {
    int64_t seed;
    Predictor predictor;
};

struct SuitePredictor::Encoding {
    ExternalVocabs vocabs;
    DatasetConfig config;
    SuiteTargetInputs reads;  // the part of the input contract its members read
    std::string fingerprint;
};

struct SuitePredictor::LoadedTarget {
    std::size_t spec = 0;       // index into manifest_.targets
    std::size_t encoding = 0;   // index into encodings_
    std::vector<Member> members;
    std::vector<std::string> class_names;
};

SuitePredictor::SuitePredictor() = default;
SuitePredictor::SuitePredictor(SuitePredictor&&) noexcept = default;
SuitePredictor& SuitePredictor::operator=(SuitePredictor&&) noexcept = default;
SuitePredictor::~SuitePredictor() = default;

namespace {

// Everything that decides how a member encodes the input plots: its three
// vocabularies, its categorical maps, every loader knob, and the columns it
// reads. Two members with equal fingerprints read the same dataset.
struct FingerprintWriter {
    std::ostringstream& out;
    template <typename T>
    void operator()(const char* name, const char* /*key*/, const T& value) const {
        out << name << '=';
        if constexpr (std::is_enum_v<T>) {
            out << enum_to_name(value);
        } else if constexpr (std::is_same_v<T, bool>) {
            out << (value ? 1 : 0);
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
            out << value;
        } else {
            static_assert(registry_detail::always_false<T>,
                          "DatasetConfig field type has no fingerprint spelling");
        }
        out << ';';
    }
};

std::string encoding_fingerprint(const ExternalVocabs& vocabs, const DatasetConfig& config,
                                 const SuiteTargetInputs& reads) {
    Sha256 hash;
    auto list = [&](const char* label, const std::vector<std::string>& names) {
        hash.update(label);
        for (const auto& n : names) {
            hash.update(n);
            hash.update(std::string_view("\x1f", 1));
        }
        hash.update(std::string_view("\x1e", 1));
    };
    list("species", vocabs.species_vocab);
    list("genus", vocabs.taxonomy.genus_names());
    list("family", vocabs.taxonomy.family_names());
    for (const auto& col : vocabs.categorical.column_names()) {
        std::vector<std::pair<std::string, int64_t>> entries(
            vocabs.categorical.column_map(col).begin(), vocabs.categorical.column_map(col).end());
        std::sort(entries.begin(), entries.end());
        hash.update("categorical:" + col);
        for (const auto& [value, code] : entries) {
            hash.update(value + "=" + std::to_string(code));
            hash.update(std::string_view("\x1f", 1));
        }
    }
    hash.update("species_width=" + std::to_string(vocabs.pool_species_cap) + ";");
    std::ostringstream fields;
    for_each_field(config, FingerprintWriter{fields});
    hash.update(fields.str());
    hash.update(std::string("reads_coordinates=") + (reads.coordinates ? "1" : "0") + ";");
    list("reads_covariates", reads.covariates);
    list("reads_categoricals", reads.categoricals);
    return hash.hex_digest();
}

std::string joined(const std::vector<std::string>& names) {
    std::string s;
    for (const auto& n : names) s += (s.empty() ? "" : ", ") + n;
    return s;
}

// Check one loaded member against the manifest's contract and target entry.
void check_member(const SuiteManifest& manifest, const SuiteTarget& target,
                  const SuiteMember& member, const Predictor& predictor) {
    const ResolveSchema& schema = predictor.schema();
    const SuiteInputContract& in = manifest.inputs;
    const SuiteTargetInputs reads = manifest.inputs_of(target);
    const std::string source = target.inputs ? "the target's declared inputs" : "the input contract";
    const std::string label = "suite '" + manifest.name + "' target '" + target.name +
                              "' seed " + std::to_string(member.seed) + " (" + member.file + ")";
    auto fail = [&](const std::string& what) { throw std::runtime_error(label + ": " + what); };

    if (!schema.has_species_vocab()) {
        fail("the checkpoint carries no species vocabulary, only its size, so new plots "
             "cannot be encoded in its namespace");
    }
    for (const auto& out : target.outputs) {
        const TargetConfig* head = nullptr;
        for (const auto& t : schema.targets) {
            if (t.name == out) head = &t;
        }
        if (!head) {
            std::string heads;
            for (const auto& t : schema.targets) heads += (heads.empty() ? "" : ", ") + t.name;
            fail("the manifest reads output '" + out + "', the checkpoint's heads are: " + heads);
        }
        if (head->task != target.task) {
            fail("output '" + out + "' is a " + enum_to_name(head->task) + " head, the manifest "
                 "declares " + enum_to_name(target.task));
        }
    }
    if (schema.covariate_names != reads.covariates) {
        fail("the checkpoint reads covariates [" + joined(schema.covariate_names) + "], which " +
             source + " does not list in that order");
    }
    if (schema.categorical_names != reads.categoricals) {
        fail("the checkpoint's categorical columns [" + joined(schema.categorical_names) +
             "] differ from those of " + source);
    }
    if (schema.has_coordinates != reads.coordinates) {
        fail(schema.has_coordinates
                 ? "the checkpoint reads coordinates, " + source + " names none"
                 : source + " names coordinates, the checkpoint reads none");
    }
    // A taxonomy vocabulary always holds <UNK>; a level the data named holds
    // more.
    if (schema.has_taxonomy && schema.n_genera > 1 && in.genus.empty()) {
        fail("the checkpoint reads genus, the input contract names no genus column");
    }
    if (schema.has_taxonomy && schema.n_families > 1 && in.family.empty()) {
        fail("the checkpoint reads family, the input contract names no family column");
    }
    if (schema.has_abundance && in.abundance.empty()) {
        fail("the checkpoint was trained on recorded abundances, the input contract names no "
             "abundance column");
    }

    // The abundance a recorded 0 was read at in training: the checkpoint's
    // record, or for a checkpoint that predates the setting the value the
    // manifest states for the member. A member with neither is read at the
    // contract's value.
    const auto& unrecorded = schema.unrecorded_dataset_settings;
    const bool recorded = std::find(unrecorded.begin(), unrecorded.end(), "zero_abundance_as") ==
                          unrecorded.end();
    auto as_text = [](float v) { return number_to_decimal(v); };
    if (recorded) {
        const float trained = schema.dataset_config.zero_abundance_as;
        if (member.zero_abundance_as && *member.zero_abundance_as != trained) {
            fail("the manifest states the member read a zero abundance as " +
                 as_text(*member.zero_abundance_as) + ", the checkpoint records " +
                 as_text(trained));
        }
        if (trained != in.zero_abundance_as) {
            fail("the checkpoint was trained reading a zero abundance as " + as_text(trained) +
                 ", the input contract reads it as " + as_text(in.zero_abundance_as));
        }
    } else if (member.zero_abundance_as && *member.zero_abundance_as != in.zero_abundance_as) {
        fail("the manifest states the member read a zero abundance as " +
             as_text(*member.zero_abundance_as) + ", the input contract reads it as " +
             as_text(in.zero_abundance_as));
    }
}

// Index that reorders `ids` into `reference` order: reference[i] == ids[perm[i]].
torch::Tensor alignment(const std::vector<std::string>& reference,
                        const std::vector<std::string>& ids) {
    if (reference.size() != ids.size()) {
        throw std::runtime_error("suite: two encodings of the same input hold " +
                                 std::to_string(reference.size()) + " and " +
                                 std::to_string(ids.size()) + " plots");
    }
    std::unordered_map<std::string, int64_t> where;
    where.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) where.emplace(ids[i], static_cast<int64_t>(i));
    std::vector<int64_t> perm(reference.size());
    for (std::size_t i = 0; i < reference.size(); ++i) {
        auto it = where.find(reference[i]);
        if (it == where.end()) {
            throw std::runtime_error("suite: plot '" + reference[i] +
                                     "' is missing from one encoding of the input");
        }
        perm[i] = it->second;
    }
    return torch::tensor(perm, torch::kLong);
}

// Undefined tensors pass through, so an absent recognition stays absent.
torch::Tensor reorder(const torch::Tensor& t, const torch::Tensor& perm, int64_t dim) {
    if (!t.defined() || !perm.defined()) return t;
    return t.index_select(dim, perm);
}

}  // namespace

SuitePredictor SuitePredictor::load(const std::string& dir, const SuiteLoadOptions& options) {
    SuitePredictor suite;
    suite.dir_ = dir;
    suite.manifest_ = SuiteManifest::read(dir);
    const SuiteManifest& manifest = suite.manifest_;

    std::vector<std::size_t> selected;
    if (options.targets.empty()) {
        for (std::size_t i = 0; i < manifest.targets.size(); ++i) selected.push_back(i);
    } else {
        for (const auto& name : options.targets) {
            const SuiteTarget& t = manifest.target(name);
            const auto index = static_cast<std::size_t>(&t - manifest.targets.data());
            if (std::find(selected.begin(), selected.end(), index) == selected.end()) {
                selected.push_back(index);
            }
        }
        std::sort(selected.begin(), selected.end());
    }

    if (options.verify_checksums) {
        std::vector<std::string> names;
        for (const std::size_t i : selected) names.push_back(manifest.targets[i].name);
        const std::vector<std::string> problems = manifest.verify(dir, names);
        if (!problems.empty()) {
            std::string msg = "suite '" + manifest.name + "' in '" + dir +
                              "' does not match its manifest:";
            for (const auto& p : problems) msg += "\n  - " + p;
            throw std::runtime_error(msg);
        }
    }

    for (const std::size_t i : selected) {
        const SuiteTarget& spec = manifest.targets[i];
        LoadedTarget loaded;
        loaded.spec = i;
        std::optional<std::size_t> encoding;
        for (const auto& member : spec.members) {
            Predictor predictor = Predictor::load(member_location(dir, member.file).string(),
                                                  options.device, options.vram_fraction);
            check_member(manifest, spec, member, predictor);

            if (spec.combine == SuiteCombine::Vote) {
                const std::string& out = spec.outputs.front();
                std::vector<std::string> names;
                for (const auto& t : predictor.schema().targets) {
                    if (t.name == out) names = t.class_names;
                }
                if (loaded.members.empty()) {
                    loaded.class_names = names;
                } else if (names != loaded.class_names) {
                    throw std::runtime_error(
                        "suite '" + manifest.name + "' target '" + spec.name + "' seed " +
                        std::to_string(member.seed) + ": its classes differ from seed " +
                        std::to_string(loaded.members.front().seed) + "'s");
                }
            }

            ExternalVocabs vocabs = predictor.external_vocabs();
            // The loader is asked for no target, so the class mappings the
            // vocabularies carry for replay play no part in the encoding.
            vocabs.targets.clear();
            DatasetConfig config =
                dataset_config_from_checkpoint(predictor.schema(), predictor.model()->config());
            config.zero_abundance_as = manifest.inputs.zero_abundance_as;
            SuiteTargetInputs reads = manifest.inputs_of(spec);
            const std::string fingerprint = encoding_fingerprint(vocabs, config, reads);

            std::size_t index = suite.encodings_.size();
            for (std::size_t e = 0; e < suite.encodings_.size(); ++e) {
                if (suite.encodings_[e]->fingerprint == fingerprint) index = e;
            }
            if (index == suite.encodings_.size()) {
                auto enc = std::make_unique<Encoding>();
                enc->vocabs = std::move(vocabs);
                enc->config = config;
                enc->reads = std::move(reads);
                enc->fingerprint = fingerprint;
                suite.encodings_.push_back(std::move(enc));
            }
            // One target reports one recognition per plot, which needs one
            // vocabulary: its members are one recipe under different seeds.
            if (encoding && *encoding != index) {
                throw std::runtime_error(
                    "suite '" + manifest.name + "' target '" + spec.name + "' seed " +
                    std::to_string(member.seed) + " encodes its input differently from seed " +
                    std::to_string(loaded.members.front().seed) + " (another species "
                    "vocabulary or loader setting); a target's members must share one");
            }
            encoding = index;
            loaded.members.push_back(Member{member.seed, std::move(predictor)});
        }
        loaded.encoding = *encoding;
        suite.targets_.push_back(std::move(loaded));
    }
    return suite;
}

std::vector<std::string> SuitePredictor::target_names() const {
    std::vector<std::string> out;
    for (const auto& t : targets_) out.push_back(manifest_.targets[t.spec].name);
    return out;
}

RoleMapping SuitePredictor::roles_for(const SuiteColumns& columns) const {
    const SuiteInputContract& in = manifest_.inputs;
    auto pick = [](const std::optional<std::string>& over, const std::string& base) {
        return over ? *over : base;
    };
    auto optional_role = [&](const std::optional<std::string>& over,
                             const std::string& base) -> std::optional<std::string> {
        const std::string name = pick(over, base);
        if (base.empty() && over && !over->empty()) {
            throw std::invalid_argument("the suite reads no such column, so it cannot be renamed "
                                        "(asked to read '" + *over + "')");
        }
        if (name.empty()) return std::nullopt;
        return name;
    };

    RoleMapping roles;
    roles.plot_id = pick(columns.plot_id, in.plot_id);
    roles.species_id = pick(columns.species_id, in.species);
    roles.abundance = optional_role(columns.abundance, in.abundance);
    roles.genus = optional_role(columns.genus, in.genus);
    roles.family = optional_role(columns.family, in.family);
    roles.latitude = optional_role(columns.latitude, in.latitude);
    roles.longitude = optional_role(columns.longitude, in.longitude);
    roles.covariates = columns.covariates ? *columns.covariates : in.covariates;
    roles.categoricals = columns.categoricals ? *columns.categoricals : in.categoricals;
    if (roles.covariates.size() != in.covariates.size()) {
        throw std::invalid_argument("the suite reads " + std::to_string(in.covariates.size()) +
                                    " covariate(s); " + std::to_string(roles.covariates.size()) +
                                    " column name(s) were given");
    }
    if (roles.categoricals.size() != in.categoricals.size()) {
        throw std::invalid_argument("the suite reads " + std::to_string(in.categoricals.size()) +
                                    " categorical column(s); " +
                                    std::to_string(roles.categoricals.size()) +
                                    " column name(s) were given");
    }
    if (roles.plot_id.empty() || roles.species_id.empty()) {
        throw std::invalid_argument("the plot id and species columns cannot be empty");
    }
    return roles;
}

RoleMapping SuitePredictor::roles_for(const RoleMapping& roles, const Encoding& encoding) const {
    const SuiteInputContract& in = manifest_.inputs;
    // A contract column is renamed by position, so the caller's name for a
    // column is the one at the contract column's index.
    auto renamed = [](const std::vector<std::string>& read, const std::vector<std::string>& contract,
                      const std::vector<std::string>& caller) {
        std::vector<std::string> out;
        for (const auto& column : read) {
            const auto at = std::find(contract.begin(), contract.end(), column) - contract.begin();
            out.push_back(caller[static_cast<std::size_t>(at)]);
        }
        return out;
    };
    RoleMapping out = roles;
    if (!encoding.reads.coordinates) {
        out.latitude.reset();
        out.longitude.reset();
    }
    out.covariates = renamed(encoding.reads.covariates, in.covariates, roles.covariates);
    out.categoricals = renamed(encoding.reads.categoricals, in.categoricals, roles.categoricals);
    return out;
}

SuitePredictions SuitePredictor::predict(const SuiteInput& input,
                                         const SuitePredictOptions& options) {
    if (manifest_.inputs.needs_header() && !input.has_header()) {
        throw std::invalid_argument(
            "suite '" + manifest_.name + "' reads plot-level columns (coordinates or "
            "covariates), so a header table is required");
    }
    const RoleMapping roles = roles_for(options.columns);

    SuitePredictions result;
    result.targets.resize(targets_.size());
    bool have_reference = false;

    // One encoding at a time: build its dataset, score every target that reads
    // it, and release it before the next.
    for (std::size_t e = 0; e < encodings_.size(); ++e) {
        std::vector<std::size_t> users;
        for (std::size_t t = 0; t < targets_.size(); ++t) {
            if (targets_[t].encoding == e) users.push_back(t);
        }
        if (users.empty()) continue;

        const Encoding& enc = *encodings_[e];
        const ResolveDataset dataset = input.encode(roles_for(roles, enc), enc.vocabs, enc.config);

        torch::Tensor perm;
        if (!have_reference) {
            result.plot_ids = dataset.plot_ids();
            have_reference = true;
        } else if (dataset.plot_ids() != result.plot_ids) {
            perm = alignment(result.plot_ids, dataset.plot_ids());
        }

        for (const std::size_t t : users) {
            LoadedTarget& loaded = targets_[t];
            const SuiteTarget& spec = manifest_.targets[loaded.spec];
            SuiteTargetPrediction& out = result.targets[t];
            out.name = spec.name;
            out.task = spec.task;
            out.combine = spec.combine;
            out.status = spec.status;
            out.limit = spec.limit;
            out.units = spec.units;
            const SuiteCombineRule& rule = combine_rule(spec.combine);
            switch (rule.dispersion_units) {
                case DispersionUnits::None: out.dispersion_units.clear(); break;
                case DispersionUnits::TargetUnits: out.dispersion_units = spec.units; break;
                case DispersionUnits::Kilometres: out.dispersion_units = "km"; break;
            }
            out.class_names = loaded.class_names;

            std::vector<torch::Tensor> member_values;
            std::vector<torch::Tensor> member_probabilities;
            for (auto& member : loaded.members) {
                out.member_seeds.push_back(member.seed);
                SuiteMemberOutput output = rule.extract(
                    spec, member.predictor.predict(dataset, /*return_latent=*/false,
                                                   options.batch_size));
                member_values.push_back(std::move(output.value));
                if (output.probabilities.defined()) {
                    member_probabilities.push_back(std::move(output.probabilities));
                }
            }

            const torch::Tensor stack = reorder(torch::stack(member_values, 0), perm, 1);
            torch::Tensor mean_probabilities;
            if (!member_probabilities.empty()) {
                mean_probabilities = reorder(torch::stack(member_probabilities, 0).mean(0), perm, 0)
                                         .to(torch::kFloat32);
                out.probabilities = mean_probabilities;
            }
            const CombinedPrediction combined = rule.combine_members(stack, mean_probabilities, spec);
            out.value = combined.value;
            out.agreement = combined.agreement;
            out.dispersion = combined.dispersion;
            if (options.keep_members) out.members = stack;

            const SpeciesRecognition& r = dataset.species_recognition();
            out.recognition = SpeciesRecognition{
                reorder(r.n_species, perm, 0), reorder(r.n_recognised, perm, 0),
                reorder(r.count_share, perm, 0), reorder(r.abundance_share, perm, 0)};
        }
    }
    return result;
}

const SuiteTargetPrediction& SuitePredictions::target(const std::string& name) const {
    for (const auto& t : targets) {
        if (t.name == name) return t;
    }
    throw std::runtime_error("suite predictions hold no target '" + name + "'");
}

}  // namespace resolve
