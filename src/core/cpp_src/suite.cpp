#include "resolve/suite.hpp"

#include "resolve/config_registry.hpp"
#include "resolve/enum_names.hpp"
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

json::Value target_to_json(const SuiteTarget& t) {
    json::Value o = json::Value::object();
    o.set("name", t.name);
    o.set("task", enum_to_name(t.task));
    o.set("combine", enum_to_name(t.combine));
    o.set("outputs", json::Value::array_of(t.outputs));
    o.set("units", t.units);
    if (t.combine == SuiteCombine::CircularMean) o.set("period", t.period);
    o.set("status", enum_to_name(t.status));
    if (!t.limit.empty()) o.set("limit", t.limit);
    json::Value members = json::Value::array();
    for (const auto& m : t.members) {
        json::Value mo = json::Value::object();
        mo.set("seed", m.seed);
        mo.set("file", m.file);
        mo.set("sha256", m.sha256);
        mo.set("bytes", m.bytes);
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
    o.set("format_version", kSuiteFormatVersion);
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
    return m;
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

        switch (t.combine) {
            case SuiteCombine::Vote:
                if (t.task != TaskType::Classification) {
                    problems.add(where, "a vote combines class predictions; task must be classification");
                }
                if (t.outputs.size() != 1) problems.add(where + ".outputs", "a vote reads one output");
                break;
            case SuiteCombine::Mean:
                if (t.task != TaskType::Regression) {
                    problems.add(where, "a mean combines values; task must be regression");
                }
                if (t.outputs.size() != 1) problems.add(where + ".outputs", "a mean reads one output");
                break;
            case SuiteCombine::CircularMean:
                if (t.task != TaskType::Regression) {
                    problems.add(where, "a circular mean combines bearings; task must be regression");
                }
                if (t.outputs.size() != 1 && t.outputs.size() != 2) {
                    problems.add(where + ".outputs", "a circular mean reads one output (the "
                                 "bearing) or two (its sine, then its cosine)");
                }
                if (!(std::isfinite(t.period) && t.period > 0.0)) {
                    problems.add(where + ".period", "a circular mean needs a positive period");
                }
                break;
        }
        if (t.combine != SuiteCombine::CircularMean && t.period != 0.0) {
            problems.add(where + ".period", "is read only by a circular mean");
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

std::vector<std::string> SuiteManifest::verify(const std::string& dir) const {
    std::vector<std::string> problems;
    for (const auto& t : targets) {
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

CombinedPrediction combine_vote(const torch::Tensor& member_codes, int64_t n_classes) {
    require_stack(member_codes, "combine_vote");
    if (n_classes < 1) throw std::invalid_argument("combine_vote: n_classes must be >= 1");
    const auto codes = member_codes.to(torch::kCPU, torch::kLong);
    if (codes.numel() > 0 &&
        (codes.min().item<int64_t>() < 0 || codes.max().item<int64_t>() >= n_classes)) {
        throw std::invalid_argument("combine_vote: a class code lies outside [0, n_classes)");
    }
    const int64_t n_members = codes.size(0);
    // (n_plots, n_classes) votes; argmax returns the first maximum, so a tie
    // goes to the lowest class code.
    const auto votes = torch::one_hot(codes, n_classes).sum(0);
    CombinedPrediction out;
    out.value = votes.argmax(1);
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
    CombinedPrediction out;
    out.value = torch::remainder(torch::atan2(s, c) * (period / kTwoPi), period).to(torch::kFloat32);
    // Mean resultant length R in [0, 1]; the circular standard deviation
    // sqrt(-2 ln R) is 0 when every member names the same bearing and grows
    // without bound as the bearings spread round the circle.
    const auto resultant = torch::sqrt(s * s + c * c).clamp_max(1.0);
    out.dispersion = (torch::sqrt(-2.0 * torch::log(resultant)) * (period / kTwoPi))
                         .to(torch::kFloat32);
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
// SuitePredictor
// ============================================================================

struct SuitePredictor::Member {
    int64_t seed;
    Predictor predictor;
};

struct SuitePredictor::Encoding {
    ExternalVocabs vocabs;
    DatasetConfig config;
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
// vocabularies, its categorical maps, and every loader knob. Two members with
// equal fingerprints read the same dataset.
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

std::string encoding_fingerprint(const ExternalVocabs& vocabs, const DatasetConfig& config) {
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
    std::ostringstream fields;
    for_each_field(config, FingerprintWriter{fields});
    hash.update(fields.str());
    return hash.hex_digest();
}

// Check one loaded member against the manifest's contract and target entry.
void check_member(const SuiteManifest& manifest, const SuiteTarget& target,
                  const SuiteMember& member, const Predictor& predictor) {
    const ResolveSchema& schema = predictor.schema();
    const SuiteInputContract& in = manifest.inputs;
    const std::string label = "suite '" + manifest.name + "' target '" + target.name +
                              "' seed " + std::to_string(member.seed) + " (" + member.file + ")";
    auto fail = [&](const std::string& what) { throw std::runtime_error(label + ": " + what); };

    if (!schema.has_species_vocab()) {
        fail("the checkpoint carries no species vocabulary (written before "
             "gcol33/resolve#102), so new plots cannot be encoded in its namespace");
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
    if (schema.covariate_names != in.covariates) {
        fail("the checkpoint reads covariates [" + [&] {
            std::string s;
            for (const auto& c : schema.covariate_names) s += (s.empty() ? "" : ", ") + c;
            return s;
        }() + "], which the input contract does not list in that order");
    }
    if (schema.categorical_names != in.categoricals) {
        fail("the checkpoint's categorical columns differ from the input contract's");
    }
    if (schema.has_coordinates != in.reads_coordinates()) {
        fail(schema.has_coordinates
                 ? "the checkpoint reads coordinates, the input contract names none"
                 : "the input contract names coordinates, the checkpoint reads none");
    }
    if (schema.has_taxonomy && in.genus.empty() && in.family.empty()) {
        fail("the checkpoint reads genus / family, the input contract names neither column");
    }
    if (schema.zero_abundance_as != 0.0f && schema.zero_abundance_as != in.zero_abundance_as) {
        fail("the checkpoint was trained reading a zero abundance as " +
             std::to_string(schema.zero_abundance_as) + ", the input contract as " +
             std::to_string(in.zero_abundance_as));
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
        std::vector<std::string> problems;
        for (const std::size_t i : selected) {
            const SuiteTarget& t = manifest.targets[i];
            for (const auto& m : t.members) {
                if (std::string p = member_problem(dir, t.name, m); !p.empty()) {
                    problems.push_back(std::move(p));
                }
            }
        }
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
            const std::string fingerprint = encoding_fingerprint(vocabs, config);

            std::size_t index = suite.encodings_.size();
            for (std::size_t e = 0; e < suite.encodings_.size(); ++e) {
                if (suite.encodings_[e]->fingerprint == fingerprint) index = e;
            }
            if (index == suite.encodings_.size()) {
                auto enc = std::make_unique<Encoding>();
                enc->vocabs = std::move(vocabs);
                enc->config = config;
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
        const ResolveDataset dataset = input.encode(roles, enc.vocabs, enc.config);

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
            out.class_names = loaded.class_names;

            std::vector<torch::Tensor> member_values;
            std::vector<torch::Tensor> member_probabilities;
            for (auto& member : loaded.members) {
                out.member_seeds.push_back(member.seed);
                ResolvePredictions p =
                    member.predictor.predict(dataset, /*return_latent=*/false, options.batch_size);
                auto head = [&](const std::string& name) {
                    return p.predictions.at(name).detach().to(torch::kCPU);
                };
                switch (spec.combine) {
                    case SuiteCombine::Vote:
                        member_values.push_back(head(spec.outputs.front()).to(torch::kLong));
                        member_probabilities.push_back(
                            p.probabilities.at(spec.outputs.front()).detach().to(torch::kCPU));
                        break;
                    case SuiteCombine::Mean:
                        member_values.push_back(head(spec.outputs.front()).to(torch::kFloat32));
                        break;
                    case SuiteCombine::CircularMean:
                        member_values.push_back(
                            spec.outputs.size() == 2
                                ? bearing_from_components(head(spec.outputs[0]),
                                                          head(spec.outputs[1]), spec.period)
                                : torch::remainder(head(spec.outputs.front()).to(torch::kFloat64),
                                                   spec.period).to(torch::kFloat32));
                        break;
                }
            }

            const torch::Tensor stack = reorder(torch::stack(member_values, 0), perm, 1);
            CombinedPrediction combined;
            switch (spec.combine) {
                case SuiteCombine::Vote: {
                    const auto probabilities =
                        reorder(torch::stack(member_probabilities, 0).mean(0), perm, 0);
                    combined = combine_vote(stack, probabilities.size(1));
                    out.probabilities = probabilities.to(torch::kFloat32);
                    break;
                }
                case SuiteCombine::Mean:
                    combined = combine_mean(stack);
                    break;
                case SuiteCombine::CircularMean:
                    combined = combine_circular(stack, spec.period);
                    break;
            }
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
