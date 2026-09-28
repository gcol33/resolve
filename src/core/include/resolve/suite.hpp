#pragma once

// Model suites: a released set of checkpoints scored as one model.
//
// A suite is a directory holding a manifest (`manifest.json`) and the weight
// files it names. Each target of the suite is predicted by several members --
// the same recipe trained under different seeds -- and the suite reports, per
// plot, the members' combined prediction beside how far the members agree.
//
// The manifest is the release's contract with whoever scores it:
//
//   * the INPUT CONTRACT: which columns the plots must carry (species names
//     with abundance, taxonomy, coordinates, covariates) and how a raw value is
//     read (the abundance a recorded 0 stands for). Every member checkpoint is
//     checked against it at load, so a suite cannot claim inputs its models
//     were not trained on.
//   * per TARGET: the members (file, seed, SHA-256, size), how their
//     predictions combine (a class vote, a mean, a circular mean of bearings,
//     read from one output or from a sine / cosine pair), the output units,
//     whether the target is released or experimental and the limit an
//     experimental release states, and free-form training and validation
//     records (the validation metrics by regime the release reports).
//   * provenance: the engine version that trained the members, the species
//     taxonomy, the training scope and the licence.
//
// Loading verifies every member's checksum before anything is read, so a
// truncated download or a swapped file fails loudly rather than predicting.
//
// Members of one target may be encoded against their own species vocabulary --
// a target trains on the plots where it is recorded, and the vocabulary is
// fitted on those -- so the suite encodes the input plots once per distinct
// vocabulary and data encoding, never once per member.
//
// Per plot the suite reports, each on its own:
//   * the combined prediction;
//   * for a vote, the share of members naming the winning class, and the
//     members' mean class probabilities;
//   * for a mean, the members' standard deviation, in the target's units;
//   * for a circular mean, the members' circular standard deviation, in the
//     target's units;
//   * how much of the plot's assemblage the target's vocabulary recognises, by
//     distinct species and by abundance (compute_species_recognition).
// None of these is combined into a single support level, and none is a
// calibrated interval: seed spread measures how much the members disagree, not
// how far the combined prediction is from the truth.

#include "resolve/dataset.hpp"
#include "resolve/json.hpp"
#include "resolve/predictor.hpp"
#include "resolve/row_source.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace resolve {

// ============================================================================
// Manifest
// ============================================================================

inline constexpr const char* kSuiteManifestFile = "manifest.json";
inline constexpr const char* kSuiteFormat = "resolve-suite";
inline constexpr int kSuiteFormatVersion = 1;

// The columns a suite reads and how their raw values are interpreted. Column
// names are the defaults a caller's table is read with; a caller whose table
// names them differently renames them through SuiteColumns at predict time.
struct SuiteInputContract {
    std::string plot_id;
    std::string species;
    std::string abundance;   // empty: no abundance column is read
    std::string genus;       // empty: not read
    std::string family;      // empty: not read
    std::string latitude;    // empty: the members read no coordinates
    std::string longitude;
    std::vector<std::string> covariates;    // header columns, in the members' order
    std::vector<std::string> categoricals;  // header columns, in the members' order
    // What an abundance value measures (e.g. "percent cover"). Documentation
    // for the caller; the engine does not convert units.
    std::string abundance_units;
    // The abundance a recorded 0 is read at (DatasetConfig::zero_abundance_as).
    float zero_abundance_as = 0.0f;
    // Anything else the caller must know about the inputs (coordinate
    // reference system, the species naming the vocabulary expects, ...).
    std::string notes;

    [[nodiscard]] bool reads_coordinates() const noexcept { return !latitude.empty(); }
    // True when the members read header-level columns, so a header table is
    // required to score.
    [[nodiscard]] bool needs_header() const noexcept {
        return reads_coordinates() || !covariates.empty() || !categoricals.empty();
    }
};

// One weight file of a target.
struct SuiteMember {
    int64_t seed = 0;
    std::string file;    // path relative to the suite directory, '/'-separated
    std::string sha256;  // lower-case hex
    int64_t bytes = 0;
};

// One released target.
struct SuiteTarget {
    std::string name;             // the suite's name for it, e.g. "aspect"
    TaskType task = TaskType::Regression;
    SuiteCombine combine = SuiteCombine::Mean;
    // The member checkpoint's target head(s) the prediction is read from: one
    // for a vote or a mean; for a circular mean either one (a bearing in the
    // target's units) or two (its sine and its cosine, in that order).
    std::vector<std::string> outputs;
    std::string units;            // e.g. "m", "degrees", "m^2"; empty for classes
    double period = 0.0;          // circular mean: the length of a full turn
    SuiteTargetStatus status = SuiteTargetStatus::Released;
    std::string limit;            // required for an experimental target
    std::vector<SuiteMember> members;
    json::Value training = json::Value::object();    // free-form record
    json::Value validation = json::Value::object();  // regime -> metrics
};

struct SuiteManifest {
    std::string name;
    std::string description;
    std::string licence;
    std::string engine_version;   // the engine that trained the members
    std::string taxonomy;         // the species naming the vocabularies follow
    std::string training_scope;   // which plots the members were trained on
    SuiteInputContract inputs;
    std::vector<SuiteTarget> targets;

    // Structural validation: every required field present and well-formed,
    // target names unique, combine rule consistent with task and outputs,
    // members with unique seeds and relative paths. Throws std::runtime_error
    // listing every problem found, not just the first.
    void validate() const;

    [[nodiscard]] json::Value to_json() const;
    [[nodiscard]] static SuiteManifest from_json(const json::Value& value);

    [[nodiscard]] const SuiteTarget& target(const std::string& name) const;

    // Read `<dir>/manifest.json` and validate it.
    [[nodiscard]] static SuiteManifest read(const std::string& dir);
    // Validate, then write `<dir>/manifest.json`.
    void write(const std::string& dir) const;

    // Fill every member's sha256 and byte count from the files under `dir`.
    // What a release builder calls after listing the files and seeds.
    void seal(const std::string& dir);

    // Problems with the files under `dir`: a member that is missing, of the
    // wrong size, or whose digest differs. Empty when the suite is intact.
    [[nodiscard]] std::vector<std::string> verify(const std::string& dir) const;
};

// ============================================================================
// Input
// ============================================================================

// Renames the contract's columns for a caller whose table spells them
// differently. An unset field keeps the manifest's name. Field names follow
// RoleMapping, as the roles argument of every loader spells them.
struct SuiteColumns {
    std::optional<std::string> plot_id;
    std::optional<std::string> species_id;
    std::optional<std::string> abundance;
    std::optional<std::string> genus;
    std::optional<std::string> family;
    std::optional<std::string> latitude;
    std::optional<std::string> longitude;
    // Replaces the covariate / categorical list; must match the contract's
    // length, since the members' input width is fixed.
    std::optional<std::vector<std::string>> covariates;
    std::optional<std::vector<std::string>> categoricals;
};

// The plots to score: a header table (one row per plot; required when the
// contract reads header columns, and when given it defines which plots are
// scored and their order) and a species table (one row per record), each
// either a CSV path or an in-memory table. No target column is read.
class SuiteInput {
public:
    static SuiteInput csv(std::string header_path, std::string species_path);
    // `header` may be null. Both tables must outlive the SuiteInput.
    static SuiteInput tables(const ColumnTable* header, const ColumnTable& species);

    [[nodiscard]] bool has_header() const noexcept;

    // Encode the plots in a model's namespace.
    [[nodiscard]] ResolveDataset encode(const RoleMapping& roles, const ExternalVocabs& vocabs,
                                        const DatasetConfig& config) const;

private:
    std::string header_path_;
    std::string species_path_;
    const ColumnTable* header_table_ = nullptr;
    const ColumnTable* species_table_ = nullptr;
};

// ============================================================================
// Prediction
// ============================================================================

struct SuiteLoadOptions {
    torch::Device device = torch::kCPU;
    float vram_fraction = 1.0f;
    // Compare every member's size and SHA-256 with the manifest before loading.
    bool verify_checksums = true;
    // Load only these targets (by suite name); empty loads every target.
    std::vector<std::string> targets;
};

struct SuitePredictOptions {
    SuiteColumns columns;
    int64_t batch_size = 4096;  // per-member forward chunk, as Predictor::predict
    // Keep every member's own prediction in SuiteTargetPrediction::members.
    bool keep_members = false;
};

struct SuiteTargetPrediction {
    std::string name;
    TaskType task = TaskType::Regression;
    SuiteCombine combine = SuiteCombine::Mean;
    SuiteTargetStatus status = SuiteTargetStatus::Released;
    std::string limit;
    std::string units;
    std::vector<std::string> class_names;  // vote: label of each class code
    std::vector<int64_t> member_seeds;

    // (n_plots,) combined prediction: int64 class code for a vote, float32 in
    // the target's units otherwise (a bearing in [0, period)).
    torch::Tensor value;
    // Vote: (n_plots,) float32 share of members naming `value`, in (0, 1].
    torch::Tensor agreement;
    // Vote: (n_plots, n_classes) float32 mean of the members' class
    // probabilities.
    torch::Tensor probabilities;
    // Mean: members' standard deviation (n - 1 denominator; NaN for a single
    // member). Circular mean: circular standard deviation sqrt(-2 ln R) of the
    // members' bearings, R their mean resultant length. (n_plots,) float32 in
    // the target's units; undefined for a vote.
    torch::Tensor dispersion;
    // With keep_members: (n_members, n_plots), each member's own prediction in
    // the form of `value`.
    torch::Tensor members;
    // Recognition of each plot by the target's vocabulary.
    SpeciesRecognition recognition;
};

struct SuitePredictions {
    std::vector<std::string> plot_ids;
    std::vector<SuiteTargetPrediction> targets;

    [[nodiscard]] const SuiteTargetPrediction& target(const std::string& name) const;
};

// The combination rules on their own, over a (n_members, n_plots) stack of
// member predictions. Shared by SuitePredictor and exposed so a caller holding
// member predictions from elsewhere combines them identically.
struct CombinedPrediction {
    torch::Tensor value;
    torch::Tensor agreement;   // vote only
    torch::Tensor dispersion;  // mean / circular mean only
};
// Vote over int64 class codes. A tie between classes with the most votes goes
// to the one with the highest mean probability when `mean_probabilities`
// ((n_plots, n_classes), the members' mean class probabilities) is given, and
// to the lowest code otherwise or where those probabilities tie too.
[[nodiscard]] CombinedPrediction combine_vote(const torch::Tensor& member_codes,
                                              int64_t n_classes,
                                              const torch::Tensor& mean_probabilities = {});
[[nodiscard]] CombinedPrediction combine_mean(const torch::Tensor& member_values);
// Bearings in [0, period) (any real value is reduced modulo the period).
[[nodiscard]] CombinedPrediction combine_circular(const torch::Tensor& member_bearings,
                                                  double period);
// A member's bearing from its sine and cosine outputs, in [0, period). The
// pair need not have unit length: the network's two outputs are read as a
// direction.
[[nodiscard]] torch::Tensor bearing_from_components(const torch::Tensor& sine,
                                                    const torch::Tensor& cosine,
                                                    double period);

class SuitePredictor {
public:
    // Load the suite in `dir`: read and validate the manifest, verify the
    // members' checksums, load every member, and check each against the
    // manifest (its heads, their tasks, the input contract).
    static SuitePredictor load(const std::string& dir, const SuiteLoadOptions& options = {});

    [[nodiscard]] const SuiteManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const std::string& directory() const noexcept { return dir_; }
    // The suite targets loaded, in manifest order.
    [[nodiscard]] std::vector<std::string> target_names() const;
    // How many distinct encodings the loaded members need (one dataset each).
    [[nodiscard]] std::size_t n_encodings() const noexcept { return encodings_.size(); }

    [[nodiscard]] SuitePredictions predict(const SuiteInput& input,
                                           const SuitePredictOptions& options = {});

    SuitePredictor(SuitePredictor&&) noexcept;
    SuitePredictor& operator=(SuitePredictor&&) noexcept;
    ~SuitePredictor();

private:
    struct Member;
    struct Encoding;
    struct LoadedTarget;

    SuitePredictor();

    [[nodiscard]] RoleMapping roles_for(const SuiteColumns& columns) const;

    std::string dir_;
    SuiteManifest manifest_;
    std::vector<std::unique_ptr<Encoding>> encodings_;
    std::vector<LoadedTarget> targets_;
};

}  // namespace resolve
