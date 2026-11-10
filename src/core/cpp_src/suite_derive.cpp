// write_derived_suite: fine-tuned checkpoints released as a suite of their own.

#include "resolve/suite.hpp"

#include "resolve/enum_names.hpp"
#include "resolve/trainer.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <unordered_set>

namespace resolve {

namespace fs = std::filesystem;

namespace {

// The parent member a fine-tuned checkpoint records, checked against the
// parent suite.
const SuiteTarget& parent_target_of(const SuiteManifest& parent, const std::string& parent_dir,
                                    const std::string& label, const ParentRecord& record) {
    if (record.suite != parent.name) {
        throw std::invalid_argument(
            label + " started from " +
            (record.suite.empty() ? "the checkpoint '" + record.checkpoint + "', not a suite member"
                                  : "suite '" + record.suite + "'") +
            "; the derived suite's parent is suite '" + parent.name + "' in '" + parent_dir + "'");
    }
    const SuiteTarget* target = nullptr;
    for (const auto& t : parent.targets) {
        if (t.name == record.suite_target) target = &t;
    }
    if (target == nullptr) {
        throw std::invalid_argument(label + " started from target '" + record.suite_target +
                                    "', which suite '" + parent.name + "' does not hold");
    }
    const auto member = std::find_if(target->members.begin(), target->members.end(),
                                     [&](const SuiteMember& m) { return m.seed == record.suite_seed; });
    if (member == target->members.end() || member->sha256 != record.sha256) {
        throw std::invalid_argument(
            label + " started from a checkpoint of SHA-256 " + record.sha256 + ", which is not "
            "suite '" + parent.name + "' target '" + target->name + "' seed " +
            std::to_string(record.suite_seed) + " in '" + parent_dir + "'");
    }
    return *target;
}

// `columns` in the order `order` lists them.
std::vector<std::string> in_order(const std::vector<std::string>& order,
                                  const std::unordered_set<std::string>& columns) {
    std::vector<std::string> out;
    for (const auto& c : order) {
        if (columns.count(c)) out.push_back(c);
    }
    return out;
}

}  // namespace

SuiteManifest write_derived_suite(const std::string& dir, const DerivedSuiteSpec& spec) {
    if (fs::exists(fs::path(dir) / kSuiteManifestFile)) {
        throw std::invalid_argument("write_derived_suite: '" + dir + "' already holds a suite "
                                    "manifest");
    }
    const SuiteManifest parent = SuiteManifest::read(spec.parent);

    SuiteManifest out;
    out.name = spec.name;
    out.description = spec.description;
    out.licence = spec.licence;
    out.training_scope = spec.training_scope;
    out.taxonomy = parent.taxonomy;
    out.inputs = parent.inputs;

    bool coordinates = false;
    std::unordered_set<std::string> covariates, categoricals;
    for (const auto& want : spec.targets) {
        if (want.members.empty()) {
            throw std::invalid_argument("write_derived_suite: target '" + want.name +
                                        "' names no member");
        }
        // The name becomes a directory of the release.
        if (want.name.empty() || want.name == "." || want.name == ".." ||
            want.name.find_first_of("/\\:") != std::string::npos) {
            throw std::invalid_argument("write_derived_suite: target name '" + want.name +
                                        "' must be non-empty and hold no '/', '\\' or ':'");
        }
        SuiteTarget target;
        target.name = want.name;
        target.outputs = want.outputs.empty() ? std::vector<std::string>{want.name} : want.outputs;
        target.period = want.period;
        target.units = want.units;
        target.status = want.status;
        target.limit = want.limit;
        target.training = want.training;
        target.validation = want.validation;

        const SuiteTarget* parent_target = nullptr;
        for (const auto& m : want.members) {
            const std::string label = "write_derived_suite: target '" + want.name + "' seed " +
                                      std::to_string(m.seed) + " (" + m.checkpoint + ")";
            const RunMetadata meta = Trainer::load_run_metadata(m.checkpoint);
            if (!meta.parent) {
                throw std::invalid_argument(label + " records no parent; it was not started "
                                            "from a suite member (Trainer::init_from)");
            }
            const SuiteTarget& from = parent_target_of(parent, spec.parent, label, *meta.parent);
            if (parent_target != nullptr && parent_target != &from) {
                throw std::invalid_argument(label + " started from target '" + from.name +
                                            "', another member from '" + parent_target->name +
                                            "'; a target's members share one parent target");
            }
            parent_target = &from;
            if (out.engine_version.empty()) out.engine_version = meta.resolve_version;

            SuiteMember member;
            member.seed = m.seed;
            member.file = "members/" + want.name + "/seed_" + std::to_string(m.seed) + ".pt";
            member.parent_sha256 = meta.parent->sha256;
            target.members.push_back(std::move(member));
        }

        // The task and classes from the first member's head.
        const Predictor first = Predictor::load(want.members.front().checkpoint, torch::kCPU);
        const TargetConfig* head = nullptr;
        for (const auto& t : first.schema().targets) {
            if (t.name == target.outputs.front()) head = &t;
        }
        if (head == nullptr) {
            throw std::invalid_argument("write_derived_suite: target '" + want.name + "': the "
                                        "checkpoints have no head '" + target.outputs.front() + "'");
        }
        target.task = head->task;
        target.combine = want.combine.value_or(head->task == TaskType::Classification
                                                   ? SuiteCombine::Vote
                                                   : SuiteCombine::Mean);
        if (target.combine == SuiteCombine::Vote) target.classes = head->class_names;

        target.inputs = parent.inputs_of(*parent_target);
        coordinates = coordinates || target.inputs->coordinates;
        covariates.insert(target.inputs->covariates.begin(), target.inputs->covariates.end());
        categoricals.insert(target.inputs->categoricals.begin(), target.inputs->categoricals.end());

        target.derived_from = SuiteDerivation{parent.name, parent_target->name,
                                              parent.engine_version, parent.licence,
                                              parent.training_scope};
        out.targets.push_back(std::move(target));
    }

    // The contract is what the targets read of the parent's.
    out.inputs.covariates = in_order(parent.inputs.covariates, covariates);
    out.inputs.categoricals = in_order(parent.inputs.categoricals, categoricals);
    if (!coordinates) {
        out.inputs.latitude.clear();
        out.inputs.longitude.clear();
    }

    for (std::size_t i = 0; i < out.targets.size(); ++i) {
        const SuiteTarget& target = out.targets[i];
        for (std::size_t j = 0; j < target.members.size(); ++j) {
            const fs::path to = fs::path(dir) / fs::path(target.members[j].file);
            fs::create_directories(to.parent_path());
            fs::copy_file(spec.targets[i].members[j].checkpoint, to,
                          fs::copy_options::overwrite_existing);
        }
    }
    out.seal(dir);
    out.write(dir);
    // Every member checked against the manifest just written.
    (void)SuitePredictor::load(dir);
    return out;
}

}  // namespace resolve
