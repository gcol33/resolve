#include "resolve/fine_tune.hpp"

#include "resolve/checkpoint.hpp"
#include "resolve/config_json.hpp"
#include "resolve/config_registry.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/sha256.hpp"
#include "resolve/suite.hpp"
#include "resolve/trainer.hpp"

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace resolve {

namespace fs = std::filesystem;

namespace {

template <std::size_t N>
bool listed(const char* const (&names)[N], const std::string& name) {
    return std::any_of(std::begin(names), std::end(names),
                       [&](const char* n) { return name == n; });
}

std::string joined(const std::vector<std::string>& names) {
    std::string s;
    for (const auto& n : names) s += (s.empty() ? "" : ", ") + n;
    return s;
}

// Every field of two registered configs whose values differ, outside `allowed`,
// as "<what> '<field>' is <a> here and <b> in the parent".
template <typename Cfg, std::size_t N>
void config_mismatches(const char* what, const Cfg& here, const Cfg& parent,
                       const char* const (&allowed)[N], std::vector<std::string>& out) {
    const json::Value a = config_to_json(here);
    const json::Value b = config_to_json(parent);
    for (const auto& [key, value] : b.members()) {
        if (listed(allowed, key)) continue;
        const json::Value* mine = a.find(key);
        if (mine == nullptr || !(*mine == value)) {
            out.push_back(std::string(what) + " '" + key + "' is " +
                          (mine ? json::dump(*mine, 0) : std::string("unset")) + " here and " +
                          json::dump(value, 0) + " in the parent");
        }
    }
}

// The suite member a source names, or the source itself as a checkpoint.
struct ResolvedSource {
    std::string checkpoint;
    std::string suite;
    std::string suite_target;
    int64_t suite_seed = 0;
    std::string licence;
    std::string training_scope;
};

ResolvedSource resolve_source(const std::string& source, const InitFromConfig& config) {
    std::error_code ec;
    if (!fs::is_directory(source, ec)) {
        if (!config.suite_target.empty() || config.suite_seed != -1) {
            throw std::invalid_argument(
                "init_from: '" + source + "' is a checkpoint, not a suite directory, so "
                "suite_target and suite_seed do not apply");
        }
        return {source, {}, {}, 0, {}, {}};
    }

    const SuiteManifest manifest = SuiteManifest::read(source);
    const SuiteTarget* target = nullptr;
    if (config.suite_target.empty()) {
        if (manifest.targets.size() != 1) {
            std::vector<std::string> names;
            for (const auto& t : manifest.targets) names.push_back(t.name);
            throw std::invalid_argument(
                "init_from: suite '" + manifest.name + "' holds several targets (" +
                joined(names) + "); name the one to start from (suite_target)");
        }
        target = &manifest.targets.front();
    } else {
        try {
            target = &manifest.target(config.suite_target);
        } catch (const std::runtime_error& e) {
            throw std::invalid_argument(std::string("init_from: ") + e.what());
        }
    }

    const SuiteMember* member = nullptr;
    if (config.suite_seed == -1) {
        member = &target->members.front();
    } else {
        for (const auto& m : target->members) {
            if (m.seed == config.suite_seed) member = &m;
        }
        if (member == nullptr) {
            std::string seeds;
            for (const auto& m : target->members) {
                seeds += (seeds.empty() ? "" : ", ") + std::to_string(m.seed);
            }
            throw std::invalid_argument("init_from: suite '" + manifest.name + "' target '" +
                                        target->name + "' has no member of seed " +
                                        std::to_string(config.suite_seed) + "; its seeds are " +
                                        seeds);
        }
    }
    if (const std::string problem = SuiteManifest::member_problem(source, target->name, *member);
        !problem.empty()) {
        throw std::runtime_error("init_from: suite '" + manifest.name + "' in '" + source +
                                 "' does not match its manifest: " + problem);
    }
    return {(fs::path(source) / fs::path(member->file)).string(), manifest.name, target->name,
            member->seed, manifest.licence, manifest.training_scope};
}

}  // namespace

ParentModel ParentModel::open(const std::string& source, const InitFromConfig& config) {
    const ResolvedSource resolved = resolve_source(source, config);

    torch::serialize::InputArchive archive;
    read_checkpoint_archive(archive, resolved.checkpoint, torch::kCPU, "parent checkpoint load");

    ParentModel parent;
    parent.model_config_ = load_model_config(archive);
    parent.schema_ = load_schema(archive, &parent.model_config_);
    parent.scalers_ = load_scalers(archive);
    parent.categorical_vocab_ = CategoricalVocab::load(archive, "trainer_categorical_");
    if (!parent.schema_.has_species_vocab()) {
        throw std::invalid_argument(
            "init_from: '" + resolved.checkpoint + "' carries no species vocabulary, only its "
            "size, so new plots cannot be read in its namespace; re-save it with a current "
            "build");
    }

    for (const auto& key : archive.keys()) {
        torch::Tensor t;
        if (key.rfind("param_", 0) == 0 && archive.try_read(key, t)) {
            parent.parameters_.emplace(key.substr(6), t);
        } else if (key.rfind("buffer_", 0) == 0 && archive.try_read(key, t)) {
            parent.buffers_.emplace(key.substr(7), t);
        }
    }

    ParentRecord& record = parent.record_;
    record.checkpoint = resolved.checkpoint;
    record.sha256 = sha256_file(resolved.checkpoint);
    record.bytes = static_cast<int64_t>(fs::file_size(resolved.checkpoint));
    record.resolve_version = load_run_metadata(archive).resolve_version;
    record.suite = resolved.suite;
    record.suite_target = resolved.suite_target;
    record.suite_seed = resolved.suite_seed;
    record.licence = resolved.licence;
    record.training_scope = resolved.training_scope;
    return parent;
}

ExternalVocabs ParentModel::vocabs() const {
    ExternalVocabs vocabs = external_vocabs_from_schema(schema_);
    vocabs.categorical = categorical_vocab_;
    return vocabs;
}

DatasetConfig ParentModel::dataset_config() const {
    return dataset_config_from_checkpoint(schema_, model_config_);
}

std::vector<std::string> fine_tune_mismatches(const ParentModel& parent,
                                              const ModelConfig& model_config,
                                              const ResolveSchema& schema) {
    std::vector<std::string> out;
    const ResolveSchema& ps = parent.schema();
    config_mismatches("model config", model_config, parent.model_config(), kFineTuneModelFields, out);
    config_mismatches("loader setting", schema.dataset_config, ps.dataset_config,
                      kFineTuneDatasetFields, out);

    auto same_list = [&](const char* what, const std::vector<std::string>& here,
                         const std::vector<std::string>& there) {
        if (here == there) return;
        out.push_back(std::string(what) + " differ: " + std::to_string(here.size()) + " here, " +
                      std::to_string(there.size()) + " in the parent" +
                      (here.size() == there.size() ? " (another order or other names)" : ""));
    };
    same_list("species vocabularies", schema.species_vocab, ps.species_vocab);
    same_list("genus vocabularies", schema.genus_vocab, ps.genus_vocab);
    same_list("family vocabularies", schema.family_vocab, ps.family_vocab);
    if (schema.covariate_names != ps.covariate_names) {
        out.push_back("covariates [" + joined(schema.covariate_names) + "] here and [" +
                      joined(ps.covariate_names) + "] in the parent");
    }
    if (schema.categorical_names != ps.categorical_names ||
        schema.categorical_vocab_sizes != ps.categorical_vocab_sizes) {
        out.push_back("categorical columns [" + joined(schema.categorical_names) + "] here and [" +
                      joined(ps.categorical_names) + "] in the parent, or their classes differ");
    }
    auto same_flag = [&](const char* what, bool here, bool there) {
        if (here != there) {
            out.push_back(std::string(what) + (here ? " read here, not by the parent"
                                                    : " read by the parent, not here"));
        }
    };
    same_flag("coordinates", schema.has_coordinates, ps.has_coordinates);
    same_flag("taxonomy", schema.has_taxonomy, ps.has_taxonomy);
    same_flag("abundances", schema.has_abundance, ps.has_abundance);
    return out;
}

// =============================================================================
// Trainer::init_from
// =============================================================================

void Trainer::init_from(const ParentModel& parent, InitReuse reuse) {
    std::vector<std::string> problems =
        fine_tune_mismatches(parent, model_->config(), model_->schema());

    // The heads carried over: a target of the new model the parent also has,
    // when reuse asks for it and the two read the same labels the same way.
    std::vector<std::string> heads;
    if (reuse == InitReuse::EncoderHeads) {
        for (const auto& mine : model_->schema().targets) {
            const auto& parent_targets = parent.schema().targets;
            auto it = std::find_if(parent_targets.begin(), parent_targets.end(),
                                   [&](const TargetConfig& t) { return t.name == mine.name; });
            if (it == parent_targets.end()) continue;
            if (it->task != mine.task || it->transform != mine.transform ||
                it->num_classes != mine.num_classes || it->class_names != mine.class_names) {
                problems.push_back(
                    "target '" + mine.name + "' is a " + enum_to_name(mine.task) +
                    " target here and a " + enum_to_name(it->task) + " target in the parent, "
                    "or its transform or classes differ, so its head cannot be carried over; "
                    "rename the target or reuse the encoder alone");
                continue;
            }
            heads.push_back(mine.name);
        }
    }
    auto carried = [&](const std::string& name) {
        const auto head = model_->head_of(name);
        return !head || std::find(heads.begin(), heads.end(), *head) != heads.end();
    };

    // Every tensor the model takes from the parent must be there at its
    // shape; a data buffer (species graph, trait matrix, SAINT's reference
    // rows) takes the parent's shape.
    const auto data_buffers = model_->data_buffers();
    auto is_data_buffer = [&](const torch::Tensor& buffer) {
        return std::any_of(data_buffers.begin(), data_buffers.end(),
                           [&](const torch::Tensor& d) { return d.is_same(buffer); });
    };
    std::vector<std::pair<torch::Tensor, torch::Tensor>> copies;  // (destination, source)
    std::vector<std::pair<torch::Tensor, torch::Tensor>> data_copies;
    auto take = [&](const char* kind, const std::string& name, const torch::Tensor& dest,
                    const std::unordered_map<std::string, torch::Tensor>& from, bool data) {
        auto it = from.find(name);
        if (it == from.end()) {
            if (!data || !model_->absent_buffer_is_empty(dest)) {
                problems.push_back(std::string("the parent holds no ") + kind + " '" + name + "'");
            }
            return;
        }
        if (data) {
            data_copies.emplace_back(dest, it->second);
        } else if (it->second.sizes() != dest.sizes()) {
            std::ostringstream msg;
            msg << kind << " '" << name << "' is " << dest.sizes() << " here and "
                << it->second.sizes() << " in the parent";
            problems.push_back(msg.str());
        } else {
            copies.emplace_back(dest, it->second);
        }
    };
    std::size_t n_model_tensors = 0;
    for (const auto& pair : model_->named_parameters()) {
        if (!carried(pair.key())) continue;
        ++n_model_tensors;
        take("parameter", pair.key(), pair.value(), parent.parameters(), false);
    }
    for (const auto& pair : model_->named_buffers()) {
        if (!carried(pair.key())) continue;
        ++n_model_tensors;
        take("buffer", pair.key(), pair.value(), parent.buffers(), is_data_buffer(pair.value()));
    }

    // A parent tensor outside its heads that the model has no place for would
    // be part of the encoder left behind.
    auto parent_head = [&](const std::string& name) {
        for (const auto& t : parent.schema().targets) {
            if (name.rfind("head_" + t.name + ".", 0) == 0) return true;
        }
        return false;
    };
    const auto own_parameters = model_->named_parameters();
    const auto own_buffers = model_->named_buffers();
    for (const auto& [name, tensor] : parent.parameters()) {
        (void)tensor;
        if (!parent_head(name) && own_parameters.find(name) == nullptr) {
            problems.push_back("the parent's parameter '" + name + "' has no place in this model");
        }
    }
    for (const auto& [name, tensor] : parent.buffers()) {
        (void)tensor;
        if (!parent_head(name) && own_buffers.find(name) == nullptr) {
            problems.push_back("the parent's buffer '" + name + "' has no place in this model");
        }
    }

    if (!problems.empty()) {
        std::string msg = "Trainer::init_from: the model cannot start from '" +
                          parent.record().checkpoint + "':";
        for (const auto& p : problems) msg += "\n  - " + p;
        throw std::invalid_argument(msg);
    }

    {
        // Freshly constructed parameters are leaves that require grad; copy
        // under no_grad, as load_weights_into does.
        torch::NoGradGuard no_grad;
        for (auto& [dest, source] : copies) dest.copy_(source);
        for (auto& [dest, source] : data_copies) {
            dest.set_data(source.to(dest.device(), dest.scalar_type()));
        }
    }

    // The scalers the encoder was trained with, and those of each carried
    // head's target.
    Scalers inherited;
    inherited.continuous_fill = parent.scalers().continuous_fill;
    inherited.continuous_mean = parent.scalers().continuous_mean;
    inherited.continuous_scale = parent.scalers().continuous_scale;
    for (const auto& name : heads) {
        auto it = parent.scalers().target_scalers.find(name);
        if (it != parent.scalers().target_scalers.end()) inherited.target_scalers[name] = it->second;
    }

    // A split already prepared moves to the inherited scalers; regression
    // targets without one keep the scalers fitted on their fitting rows.
    if (data_prepared_) {
        Scalers merged = scalers_;
        merged.continuous_fill = inherited.continuous_fill;
        merged.continuous_mean = inherited.continuous_mean;
        merged.continuous_scale = inherited.continuous_scale;
        for (const auto& [name, mean_scale] : inherited.target_scalers) {
            merged.target_scalers[name] = mean_scale;
        }
        train_ = standardized(unstandardized(train_, scalers_), merged);
        test_ = standardized(unstandardized(test_, scalers_), merged);
        scalers_ = std::move(merged);
    }
    release_training_state();
    inherited_scalers_ = std::move(inherited);
    parent_categorical_vocab_ = parent.categorical_vocab();

    ParentRecord record = parent.record();
    record.reuse = reuse;
    record.n_tensors = static_cast<int64_t>(n_model_tensors);
    record.heads = heads;
    parent_record_ = record;
    run_metadata_.parent = parent_record_;

    // Cross-validation restarts every fold from these weights.
    pristine_model_state_ = snapshot_weights();
    config_.log("Started from " + record.checkpoint + ": " + std::to_string(n_model_tensors) +
                " tensors" + (heads.empty() ? std::string() : ", heads " + joined(heads)));
}

}  // namespace resolve
