// Exhaustiveness tests for the configuration field registries (issue #108).
//
// The registries in resolve/config_registry.hpp exist so that a field cannot be
// added to a config struct and then silently missed by one of the places that
// has to say something about it. Two of those guarantees are enforced by the
// compiler and need no test:
//
//   * a member with no registry row fails the arity static_assert in
//     RESOLVE_DEFINE_FIELD_REGISTRY, and
//   * a registry row whose type no visitor handles fails that visitor's
//     `static_assert(always_false<T>)` fallthrough.
//
// What a test still has to pin is the runtime behaviour the compiler cannot see:
// that a value written under a row actually comes back, and that the archive
// keys are the spellings earlier releases wrote, so existing checkpoints keep
// loading. Both are checked by mutating EVERY field away from its default,
// round-tripping, and comparing a registry-driven digest of the whole struct --
// so a future field is covered the moment its row exists.

#include <catch2/catch_test_macros.hpp>

#include "resolve/checkpoint.hpp"
#include "resolve/config_registry.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace resolve;

namespace {

// ---------------------------------------------------------------------------
// Registry-driven helpers
// ---------------------------------------------------------------------------

// Moves every field away from its default, deterministically and distinctly, so
// a round-trip that drops a field shows up as a mismatch rather than as a value
// that happened to already equal the default.
struct FieldMutator {
    int* counter;

    template <typename T>
    void operator()(const char*, const char*, T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            // No serialized form; the device would also need a CUDA build.
        } else if constexpr (is_registered_config_v<T>) {
            for_each_field(value, FieldMutator{counter});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            value.clear();
            for (int i = 0; i < 2; ++i) {
                ParallelBranchConfig branch;
                for_each_field(branch, FieldMutator{counter});
                value.push_back(std::move(branch));
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            value = !value;
        } else if constexpr (std::is_enum_v<T>) {
            const auto& table = EnumNames<T>::table;
            const std::size_t n = std::size(table);
            T candidate = table[n - 1].value;
            if (candidate == value) candidate = table[0].value;
            value = candidate;
        } else if constexpr (std::is_same_v<T, int>) {
            // Offset from the CURRENT value, so the result differs from the
            // default whatever the default is, and from every other int field.
            value = value + 1 + (*counter)++;
        } else if constexpr (std::is_same_v<T, float>) {
            // Dyadic offsets: exactly representable in float32, so the archive
            // round-trip is bit-exact and a mismatch means a real drop.
            value = value + 0.25f + 0.03125f * static_cast<float>((*counter)++);
        } else if constexpr (std::is_same_v<T, std::string>) {
            value = value + "_x" + std::to_string((*counter)++);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            const int n = (*counter)++;
            value = {11 + n, 22 + n, 33 + n};
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            const float n = static_cast<float>((*counter)++);
            value = {0.125f + n, 0.375f + n};
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            const int n = (*counter)++;
            value = {value.first + 7 + n, value.second + 9 + n};
        } else {
            static_assert(registry_detail::always_false<T>,
                          "add a branch to FieldMutator for this field type");
        }
    }
};

using FieldDigest = std::vector<std::pair<std::string, std::string>>;

// Renders every field as (dotted path, text). Comparing two digests element-wise
// says which field diverged, by name, instead of just that something did.
//
// `persisted_only` drops the rows with no checkpoint key (the device, the
// checkpoint destination and the AMP / cuDNN switches on TrainConfig), which the
// archive legitimately never carries.
struct FieldDigestWriter {
    FieldDigest* out;
    std::string path;
    bool persisted_only;

    void emit(const char* name, const std::string& text) const {
        out->emplace_back(path + name, text);
    }

    template <typename Seq>
    static std::string join(const Seq& values) {
        std::ostringstream os;
        os << std::setprecision(9);
        bool first = true;
        for (const auto& v : values) {
            if (!first) os << ",";
            first = false;
            os << v;
        }
        return os.str();
    }

    static std::string number(float v) {
        std::ostringstream os;
        os << std::setprecision(9) << v;
        return os.str();
    }

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        (void)key;
        if constexpr (std::is_same_v<T, LogCallback>) {
            (void)name;
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            if (persisted_only) return;
            emit(name, value.is_cuda() ? "cuda" : "cpu");
        } else if constexpr (is_registered_config_v<T>) {
            for_each_field(value, FieldDigestWriter{out, path + name + ".", persisted_only});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            if (persisted_only && !has_checkpoint_key(key)) return;
            emit(name, std::to_string(value.size()));
            for (std::size_t i = 0; i < value.size(); ++i) {
                for_each_field(value[i], FieldDigestWriter{
                    out, path + name + "[" + std::to_string(i) + "].", persisted_only});
            }
        } else {
            if (persisted_only && !has_checkpoint_key(key)) return;
            if constexpr (std::is_same_v<T, bool>) {
                emit(name, value ? "true" : "false");
            } else if constexpr (std::is_enum_v<T>) {
                emit(name, enum_to_name(value));
            } else if constexpr (std::is_same_v<T, int>) {
                emit(name, std::to_string(value));
            } else if constexpr (std::is_same_v<T, float>) {
                emit(name, number(value));
            } else if constexpr (std::is_same_v<T, std::string>) {
                emit(name, value);
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                                 std::is_same_v<T, std::vector<float>>) {
                emit(name, join(value));
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                emit(name, std::to_string(value.first) + "," + std::to_string(value.second));
            } else {
                static_assert(registry_detail::always_false<T>,
                              "add a branch to FieldDigestWriter for this field type");
            }
        }
    }
};

template <typename Cfg>
FieldDigest digest_of(const Cfg& config, bool persisted_only) {
    FieldDigest out;
    for_each_field(config, FieldDigestWriter{&out, std::string(), persisted_only});
    return out;
}

// Every field whose rendering differs, as "path: left != right".
std::vector<std::string> digest_differences(const FieldDigest& a, const FieldDigest& b) {
    std::vector<std::string> diffs;
    if (a.size() != b.size()) {
        diffs.push_back("digest length " + std::to_string(a.size()) + " vs " +
                        std::to_string(b.size()));
        return diffs;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].first != b[i].first) {
            diffs.push_back("field order: " + a[i].first + " vs " + b[i].first);
        } else if (a[i].second != b[i].second) {
            diffs.push_back(a[i].first + ": " + a[i].second + " != " + b[i].second);
        }
    }
    return diffs;
}

std::string describe(const std::vector<std::string>& diffs) {
    std::string out;
    for (const auto& d : diffs) {
        if (!out.empty()) out += "; ";
        out += d;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Name-keyed round-trip: the structural stand-in for the C-ABI value tree
// ---------------------------------------------------------------------------
//
// resolve_capi.cpp's value tree lives in the resolve_c shared library, which the
// test binary does not link. Its parse / emit pair is the same registry walk as
// the one below, over the same member names and the same enum spellings, so this
// pins the mechanism: a field reachable here is reachable there.

using NameKeyedMap = std::map<std::string, std::string>;

struct NameKeyedWriter {
    NameKeyedMap* out;
    std::string path;

    template <typename T>
    void operator()(const char* name, const char*, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            (void)name;
        } else if constexpr (is_registered_config_v<T>) {
            for_each_field(value, NameKeyedWriter{out, path + name + "."});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            (*out)[path + name] = std::to_string(value.size());
            for (std::size_t i = 0; i < value.size(); ++i) {
                for_each_field(value[i], NameKeyedWriter{
                    out, path + name + "[" + std::to_string(i) + "]."});
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            (*out)[path + name] = value ? "1" : "0";
        } else if constexpr (std::is_enum_v<T>) {
            (*out)[path + name] = enum_to_name(value);
        } else if constexpr (std::is_same_v<T, int>) {
            (*out)[path + name] = std::to_string(value);
        } else if constexpr (std::is_same_v<T, float>) {
            std::ostringstream os;
            os << std::setprecision(9) << value;
            (*out)[path + name] = os.str();
        } else if constexpr (std::is_same_v<T, std::string>) {
            (*out)[path + name] = value;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            (*out)[path + name] = FieldDigestWriter::join(value);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            (*out)[path + name] =
                std::to_string(value.first) + "," + std::to_string(value.second);
        } else {
            static_assert(registry_detail::always_false<T>,
                          "add a branch to NameKeyedWriter for this field type");
        }
    }
};

struct NameKeyedReader {
    const NameKeyedMap* in;
    std::string path;

    template <typename Out>
    void split_into(const std::string& text, Out& values) const {
        values.clear();
        std::istringstream is(text);
        std::string token;
        while (std::getline(is, token, ',')) {
            if (token.empty()) continue;
            std::istringstream item(token);
            typename Out::value_type v{};
            item >> v;
            values.push_back(v);
        }
    }

    template <typename T>
    void operator()(const char* name, const char*, T& value) const {
        const std::string key = path + name;
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            (void)name;
            (void)value;
        } else if constexpr (is_registered_config_v<T>) {
            for_each_field(value, NameKeyedReader{in, key + "."});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            auto it = in->find(key);
            if (it == in->end()) return;
            const std::size_t n = static_cast<std::size_t>(std::stoul(it->second));
            value.clear();
            for (std::size_t i = 0; i < n; ++i) {
                ParallelBranchConfig branch;
                for_each_field(branch, NameKeyedReader{
                    in, key + "[" + std::to_string(i) + "]."});
                value.push_back(std::move(branch));
            }
        } else {
            auto it = in->find(key);
            if (it == in->end()) return;
            const std::string& text = it->second;
            if constexpr (std::is_same_v<T, bool>) {
                value = (text == "1");
            } else if constexpr (std::is_enum_v<T>) {
                value = enum_from_name<T>(text);
            } else if constexpr (std::is_same_v<T, int>) {
                value = std::stoi(text);
            } else if constexpr (std::is_same_v<T, float>) {
                value = std::stof(text);
            } else if constexpr (std::is_same_v<T, std::string>) {
                value = text;
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                                 std::is_same_v<T, std::vector<float>>) {
                split_into(text, value);
            } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                std::vector<int64_t> parts;
                split_into(text, parts);
                if (parts.size() >= 2) {
                    value = {static_cast<int>(parts[0]), static_cast<int>(parts[1])};
                }
            } else {
                static_assert(registry_detail::always_false<T>,
                              "add a branch to NameKeyedReader for this field type");
            }
        }
    }
};

template <typename Cfg>
Cfg name_keyed_round_trip(const Cfg& config) {
    NameKeyedMap map;
    for_each_field(config, NameKeyedWriter{&map, std::string()});
    Cfg out;
    for_each_field(out, NameKeyedReader{&map, std::string()});
    return out;
}

std::string temp_path(const char* stem) {
    return (std::filesystem::temp_directory_path() / stem).string();
}

bool archive_has_key(const std::string& path, const std::string& key) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::Tensor probe;
    return archive.try_read(key, probe);
}

}  // namespace

// ===========================================================================
// Registry coverage
// ===========================================================================

TEST_CASE("Every config struct's registry covers every member", "[config][registry]") {
    // The build already fails when these disagree (the static_assert inside
    // RESOLVE_DEFINE_FIELD_REGISTRY). Restating the counts here makes the shape
    // of the guarantee visible in the test output, and pins the numbers so a
    // reviewer sees an intended addition in the diff.
    CHECK(field_registry_size(static_cast<const FTTransformerConfig*>(nullptr)) == 7);
    CHECK(field_registry_size(static_cast<const TabNetConfig*>(nullptr)) == 7);
    CHECK(field_registry_size(static_cast<const SAINTConfig*>(nullptr)) == 7);
    CHECK(field_registry_size(static_cast<const GNNConfig*>(nullptr)) == 8);
    CHECK(field_registry_size(static_cast<const TraitNetConfig*>(nullptr)) == 5);
    CHECK(field_registry_size(static_cast<const ExcelFormerConfig*>(nullptr)) == 7);
    CHECK(field_registry_size(static_cast<const HeterogeneousGNNConfig*>(nullptr)) == 10);
    CHECK(field_registry_size(static_cast<const TabMConfig*>(nullptr)) == 3);
    CHECK(field_registry_size(static_cast<const ParallelBranchConfig*>(nullptr)) == 5);
    CHECK(field_registry_size(static_cast<const ParallelLayersConfig*>(nullptr)) == 5);
    CHECK(field_registry_size(static_cast<const ModelConfig*>(nullptr)) == 45);
    CHECK(field_registry_size(static_cast<const DatasetConfig*>(nullptr)) == 15);
    CHECK(field_registry_size(static_cast<const TrainConfig*>(nullptr)) == 29);

    // Same comparison the static_assert makes, spelled out once at runtime.
    CHECK(field_registry_size(static_cast<const ModelConfig*>(nullptr)) ==
          registry_detail::aggregate_field_count<ModelConfig>());
    CHECK(field_registry_size(static_cast<const TrainConfig*>(nullptr)) ==
          registry_detail::aggregate_field_count<TrainConfig>());
    CHECK(field_registry_size(static_cast<const DatasetConfig*>(nullptr)) ==
          registry_detail::aggregate_field_count<DatasetConfig>());
}

TEST_CASE("The mutator really moves every field off its default", "[config][registry]") {
    // The round-trip tests below are only meaningful if the value written differs
    // from the value a failed read would leave behind, so check the mutator
    // against the defaults on the structs whose digests are fixed-length (the
    // parallel-branch list changes the row count and is covered by the
    // round-trips themselves).
    const auto every_field_moved = [](const auto& mutated, const auto& defaults) {
        const FieldDigest a = digest_of(mutated, false);
        const FieldDigest b = digest_of(defaults, false);
        if (a.size() != b.size() || a.empty()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i].second == b[i].second) return false;
        }
        return true;
    };

    int counter = 0;
    DatasetConfig dataset;
    for_each_field(dataset, FieldMutator{&counter});
    CHECK(every_field_moved(dataset, DatasetConfig{}));

    counter = 0;
    ExcelFormerConfig excelformer;
    for_each_field(excelformer, FieldMutator{&counter});
    CHECK(every_field_moved(excelformer, ExcelFormerConfig{}));

    counter = 0;
    TabNetConfig tabnet;
    for_each_field(tabnet, FieldMutator{&counter});
    CHECK(every_field_moved(tabnet, TabNetConfig{}));

    counter = 0;
    HeterogeneousGNNConfig hgnn;
    for_each_field(hgnn, FieldMutator{&counter});
    CHECK(every_field_moved(hgnn, HeterogeneousGNNConfig{}));
}

// ===========================================================================
// Checkpoint round-trips
// ===========================================================================

TEST_CASE("ModelConfig round-trips through the checkpoint with every field changed",
          "[config][registry][checkpoint]") {
    int counter = 0;
    ModelConfig config;
    for_each_field(config, FieldMutator{&counter});
    // Two branches, so the numbered parallel_branch_<i>_ blocks are exercised.
    REQUIRE(config.parallel_layers.branches.size() == 2);

    const std::string path = temp_path("resolve_registry_modelcfg.pt");
    {
        torch::serialize::OutputArchive archive;
        save_model_config(archive, config);
        archive.save_to(path);
    }
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    const ModelConfig loaded = load_model_config(archive);
    std::filesystem::remove(path);

    const auto diffs = digest_differences(digest_of(config, true), digest_of(loaded, true));
    INFO("fields that did not survive: " << describe(diffs));
    CHECK(diffs.empty());
}

TEST_CASE("TrainConfig round-trips through the checkpoint with every persisted field changed",
          "[config][registry][checkpoint]") {
    int counter = 0;
    TrainConfig config;
    for_each_field(config, FieldMutator{&counter});

    const std::string path = temp_path("resolve_registry_traincfg.pt");
    {
        torch::serialize::OutputArchive archive;
        save_train_config(archive, config);
        archive.save_to(path);
    }
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    const TrainConfig loaded = load_train_config(archive);

    SECTION("every persisted field survives") {
        const auto diffs = digest_differences(digest_of(config, true), digest_of(loaded, true));
        INFO("fields that did not survive: " << describe(diffs));
        CHECK(diffs.empty());
    }

    SECTION("fields the archive does not carry keep their defaults") {
        const TrainConfig defaults;
        CHECK(loaded.use_amp == defaults.use_amp);
        CHECK(loaded.checkpoint_dir == defaults.checkpoint_dir);
        CHECK(loaded.checkpoint_every == defaults.checkpoint_every);
        CHECK(loaded.cudnn_benchmark == defaults.cudnn_benchmark);
        CHECK(loaded.device == defaults.device);
    }

    std::filesystem::remove(path);
}

TEST_CASE("The requested batch size is what a checkpoint restores",
          "[config][registry][checkpoint]") {
    // save_train_config writes the REQUESTED batch size under train_batch_size
    // and the effective one alongside it, so an OOM fallback stays visible while
    // load_train_config recovers the recipe the operator asked for (issue #86).
    TrainConfig config;
    config.batch_size = 512;  // the value the OOM retry ended up training at

    const std::string path = temp_path("resolve_registry_batchsize.pt");
    {
        torch::serialize::OutputArchive archive;
        save_train_config(archive, config, /*requested_batch_size=*/4096);
        archive.save_to(path);
    }
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    const TrainConfig loaded = load_train_config(archive);

    torch::Tensor effective;
    REQUIRE(archive.try_read(kEffectiveBatchSizeKey, effective));
    std::filesystem::remove(path);

    CHECK(loaded.batch_size == 4096);
    CHECK(effective.item<int>() == 512);
}

// ===========================================================================
// Archive key spellings
// ===========================================================================

TEST_CASE("Checkpoint keys keep the spellings earlier releases wrote",
          "[config][registry][checkpoint][backcompat]") {
    // The registry supplies these spellings; it does not get to change them. A
    // renamed key would silently drop the field on load of an existing
    // checkpoint, which is the failure this whole exercise is about.
    int counter = 0;
    ModelConfig model;
    for_each_field(model, FieldMutator{&counter});

    const std::string model_path = temp_path("resolve_registry_keys_model.pt");
    {
        torch::serialize::OutputArchive archive;
        save_model_config(archive, model);
        archive.save_to(model_path);
    }
    for (const char* key : {
             "species_encoding", "uses_explicit_vector", "hash_dim", "hidden_dims",
             "categorical_embed_dim", "encoder_architecture", "head_hidden_dims",
             "moe_routing", "moe_placement", "n_experts", "expert_hidden_dims",
             "cover_dropout", "d_model", "transformer_pooling_len", "transformer_pooling",
             "ft_d_model", "ft_pre_norm",
             "tabnet_n_steps", "tabnet_use_sparsemax",
             "saint_use_contrastive_pretrain",
             "gnn_type", "gnn_graph_mode",
             "trait_trait_dim", "trait_shared_trait_encoder",
             "excel_importance_threshold", "excel_pre_norm",
             "hgnn_cooccurrence_threshold", "hgnn_use_cooccurrence_edges",
             "tabm_enabled", "tabm_aggregation_len", "tabm_aggregation",
             "parallel_enabled", "parallel_n_branches",
             "parallel_branch_0_hidden_dims", "parallel_branch_0_branch_weight",
             "parallel_branch_1_dropout",
         }) {
        INFO("model-config key " << key);
        CHECK(archive_has_key(model_path, key));
    }
    std::filesystem::remove(model_path);

    counter = 0;
    TrainConfig train;
    for_each_field(train, FieldMutator{&counter});
    const std::string train_path = temp_path("resolve_registry_keys_train.pt");
    {
        torch::serialize::OutputArchive archive;
        save_train_config(archive, train);
        archive.save_to(train_path);
    }
    for (const char* key : {
             "train_batch_size", "train_effective_batch_size", "train_batch_size_floor",
             "train_max_epochs", "train_patience", "train_lr", "train_weight_decay",
             "train_phase_boundary_1", "train_phase_boundary_2",
             "train_loss_config", "train_lr_scheduler", "train_lr_step_size",
             "train_lr_gamma", "train_lr_min", "train_vram_fraction",
             "train_band_thresholds", "train_band_threshold",
             "train_nca_temperature", "train_nca_neighbors", "train_nca_weight",
         }) {
        INFO("train-config key " << key);
        CHECK(archive_has_key(train_path, key));
    }
    std::filesystem::remove(train_path);
}

TEST_CASE("A checkpoint missing a long-standing model-config key is rejected",
          "[config][registry][checkpoint]") {
    // Defaulting these would build a differently-shaped model whose weights then
    // fail to load with a shape error far from the cause.
    const std::string path = temp_path("resolve_registry_truncated.pt");
    {
        torch::serialize::OutputArchive archive;
        archive.write("species_encoding", torch::tensor(0));
        archive.save_to(path);
    }
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    CHECK_THROWS_AS(load_model_config(archive), std::runtime_error);
    std::filesystem::remove(path);
}

// ===========================================================================
// Name-keyed (value-tree shaped) round-trip
// ===========================================================================

TEST_CASE("Config structs survive a name-keyed round-trip", "[config][registry]") {
    SECTION("ModelConfig") {
        int counter = 0;
        ModelConfig config;
        for_each_field(config, FieldMutator{&counter});
        const auto diffs = digest_differences(digest_of(config, false),
                                              digest_of(name_keyed_round_trip(config), false));
        INFO("fields that did not survive: " << describe(diffs));
        CHECK(diffs.empty());
    }
    SECTION("DatasetConfig") {
        int counter = 0;
        DatasetConfig config;
        for_each_field(config, FieldMutator{&counter});
        const auto diffs = digest_differences(digest_of(config, false),
                                              digest_of(name_keyed_round_trip(config), false));
        INFO("fields that did not survive: " << describe(diffs));
        CHECK(diffs.empty());
    }
    SECTION("TrainConfig") {
        int counter = 0;
        TrainConfig config;
        for_each_field(config, FieldMutator{&counter});
        const auto diffs = digest_differences(digest_of(config, false),
                                              digest_of(name_keyed_round_trip(config), false));
        INFO("fields that did not survive: " << describe(diffs));
        CHECK(diffs.empty());
    }
}

TEST_CASE("Every enum field round-trips through its shared name table",
          "[config][registry][enums]") {
    // The value tree, the CLI flags and `resolve info` all read the tables in
    // enum_names.hpp. Parsing back what the emitter produced is what keeps them
    // from drifting apart.
    for (const auto& row : kEncoderArchitectureNames) {
        CHECK(enum_from_name<EncoderArchitecture>(row.name) == row.value);
        CHECK(std::string(enum_to_name(row.value)) == row.name);
    }
    for (const auto& row : kPoolWeightingNames) {
        CHECK(enum_from_name<PoolWeighting>(row.name) == row.value);
        CHECK(std::string(enum_to_name(row.value)) == row.name);
    }
    for (const auto& row : kSelectionModeNames) {
        CHECK(enum_from_name<SelectionMode>(row.name) == row.value);
        CHECK(std::string(enum_to_name(row.value)) == row.name);
    }
    for (const auto& row : kLossConfigModeNames) {
        CHECK(enum_from_name<LossConfigMode>(row.name) == row.value);
        CHECK(std::string(enum_to_name(row.value)) == row.name);
    }
    CHECK_THROWS_AS(enum_from_name<EncoderArchitecture>("not_an_architecture"),
                    std::runtime_error);
}

// ===========================================================================
// JSON sidecar
// ===========================================================================

TEST_CASE("The metadata sidecar carries the whole model and train config",
          "[config][registry][checkpoint]") {
    int counter = 0;
    ModelConfig model;
    for_each_field(model, FieldMutator{&counter});
    model.encoder_architecture = EncoderArchitecture::ExcelFormer;

    TrainConfig train;
    train.loss_config = LossConfigMode::SMAPE;

    RunMetadata metadata;
    metadata.created_at = "2026-08-06T09:00:00Z";
    metadata.final_metrics["area"] = {{"rmse", 1.25f}};

    ResolveSchema schema;
    schema.n_plots = 10;

    const std::string checkpoint = temp_path("resolve_registry_sidecar.pt");
    const std::string sidecar = temp_path("resolve_registry_sidecar.json");
    write_metadata_json(checkpoint, model, train, metadata, schema, /*requested_batch_size=*/8192);

    std::ifstream file(sidecar);
    REQUIRE(file.is_open());
    const std::string text((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    file.close();
    std::filesystem::remove(sidecar);

    // A nested sub-config, a field that only the registry pass reaches, and the
    // enum names rather than raw ordinals.
    CHECK(text.find("\"excelformer\"") != std::string::npos);
    CHECK(text.find("\"importance_threshold\"") != std::string::npos);
    CHECK(text.find("\"encoder_architecture\": \"excelformer\"") != std::string::npos);
    CHECK(text.find("\"loss_config\": \"smape\"") != std::string::npos);
    CHECK(text.find("\"batch_size\": 8192") != std::string::npos);
    CHECK(text.find("\"effective_batch_size\"") != std::string::npos);
    // Braces balance, i.e. the nested emission did not lose one.
    std::size_t opens = 0, closes = 0;
    for (char c : text) {
        if (c == '{') ++opens;
        if (c == '}') ++closes;
    }
    CHECK(opens == closes);
    CHECK(opens > 10);
}
