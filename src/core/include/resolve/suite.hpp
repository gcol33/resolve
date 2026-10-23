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
//   * per TARGET: optionally the part of the contract its members read (a
//     model of the survey year cannot read the year), the members (file,
//     seed, SHA-256, size), how their predictions combine (a class vote, a
//     mean, a circular mean of bearings read from one output or from a sine /
//     cosine pair, a spherical mean of latitude / longitude), the output units,
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
// fitted on those -- and may read their own part of the input contract, so the
// suite encodes the input plots once per distinct vocabulary, data encoding
// and set of columns read, never once per member.
//
// Per plot the suite reports, each on its own:
//   * the combined prediction;
//   * for a vote, the share of members naming the winning class, and the
//     members' mean class probabilities;
//   * for a mean, the members' standard deviation, in the target's units;
//   * for a circular mean, the members' circular standard deviation, in the
//     target's units;
//   * for a spherical mean, the median great-circle distance of the members
//     from the combined point, in kilometres;
//   * how much of the plot's assemblage the target's vocabulary recognises, by
//     distinct species and by abundance (compute_species_recognition).
// None of these is combined into a single support level, and none is a
// calibrated interval: seed spread measures how much the members disagree, not
// how far the combined prediction is from the truth. A single member measures
// no spread, so every dispersion of a one-member target is NaN; the share of
// members naming the winning class is then 1.
//
// Each combine rule is one SuiteCombineRule (combine_rule): what it reads,
// how it combines, and the columns its prediction is written as. The manifest
// validation, the predictor, suite_prediction_columns and the CLI usage text
// all read that table.

#include "resolve/dataset.hpp"
#include "resolve/json.hpp"
#include "resolve/predictor.hpp"
#include "resolve/row_source.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace resolve {

// ============================================================================
// Manifest
// ============================================================================

inline constexpr const char* kSuiteManifestFile = "manifest.json";
inline constexpr const char* kSuiteFormat = "resolve-suite";
// Version 2 adds per-target inputs and the spherical mean, version 3 a
// member's own zero_abundance_as. A manifest is written as the lowest version
// that holds it, so an older engine still reads one that uses neither.
inline constexpr int kSuiteFormatVersion = 3;

// The Earth radius a spherical mean measures member spread with, in km.
inline constexpr double kEarthRadiusKm = 6371.0;

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

// The part of the input contract a target's members read. Every column it
// names is one the contract names, and the contract is the union of what its
// targets read. A target that declares none reads the whole contract.
struct SuiteTargetInputs {
    bool coordinates = false;               // reads the contract's latitude / longitude
    std::vector<std::string> covariates;    // contract covariates, in the members' order
    std::vector<std::string> categoricals;  // contract categoricals, in the members' order

    bool operator==(const SuiteTargetInputs&) const = default;
};

// One weight file of a target.
struct SuiteMember {
    int64_t seed = 0;
    std::string file;    // path relative to the suite directory, '/'-separated
    std::string sha256;  // lower-case hex
    int64_t bytes = 0;
    // The abundance a recorded 0 was read at while this member trained, stated
    // for a checkpoint that does not record it (one written before the engine
    // had the setting). Loading compares it with the input contract. Unset for
    // a checkpoint that records its own; when both are given they must agree.
    std::optional<float> zero_abundance_as = std::nullopt;
};

// One released target.
struct SuiteTarget {
    std::string name;             // the suite's name for it, e.g. "aspect"
    TaskType task = TaskType::Regression;
    SuiteCombine combine = SuiteCombine::Mean;
    // The member checkpoint's target head(s) the prediction is read from: one
    // for a vote or a mean; for a circular mean either one (a bearing in the
    // target's units) or two (its sine and its cosine, in that order); for a
    // spherical mean two (latitude, then longitude, in degrees).
    std::vector<std::string> outputs;
    std::string units;            // e.g. "m", "degrees", "m^2"; empty for classes
    // The columns the members read; unset reads the whole input contract.
    std::optional<SuiteTargetInputs> inputs;
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

    // What a target's members read: its declared inputs, or the whole contract.
    [[nodiscard]] SuiteTargetInputs inputs_of(const SuiteTarget& target) const;

    // The lowest format version that holds this manifest: 3 when a member
    // states its zero_abundance_as, 2 when a target declares its inputs or
    // combines by a spherical mean, else 1.
    [[nodiscard]] int format_version() const;

    // Read `<dir>/manifest.json` and validate it.
    [[nodiscard]] static SuiteManifest read(const std::string& dir);
    // Validate, then write `<dir>/manifest.json`.
    void write(const std::string& dir) const;

    // Fill every member's sha256 and byte count from the files under `dir`.
    // What a release builder calls after listing the files and seeds.
    void seal(const std::string& dir);

    // Problems with the files under `dir`: a member that is missing, of the
    // wrong size, or whose digest differs. Empty when the suite is intact.
    // `targets` (suite names) narrows the check to those targets' members.
    [[nodiscard]] std::vector<std::string> verify(
        const std::string& dir, const std::vector<std::string>& targets = {}) const;
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
    // The units of `dispersion`: the target's units for a mean or a circular
    // mean, "km" for a spherical mean, empty for a vote.
    std::string dispersion_units;
    std::vector<std::string> class_names;  // vote: label of each class code
    std::vector<int64_t> member_seeds;

    // Combined prediction: (n_plots,) int64 class code for a vote; (n_plots,)
    // float32 in the target's units for a mean or a circular mean (a bearing
    // in [0, period)); (n_plots, 2) float32 latitude and longitude in degrees
    // for a spherical mean.
    torch::Tensor value;
    // Vote: (n_plots,) float32 share of members naming `value`, in (0, 1].
    torch::Tensor agreement;
    // Vote: (n_plots, n_classes) float32 mean of the members' class
    // probabilities.
    torch::Tensor probabilities;
    // Mean: members' standard deviation (n - 1 denominator). Circular mean:
    // circular standard deviation sqrt(-2 ln R) of the members' bearings, R
    // their mean resultant length. Spherical mean: the median great-circle
    // distance of the members from the combined point. NaN for a single member
    // and where the members' directions cancel. (n_plots,) float32 in
    // `dispersion_units`; undefined for a vote.
    torch::Tensor dispersion;
    // With keep_members: (n_members, n_plots) -- (n_members, n_plots, 2) for a
    // spherical mean -- each member's own prediction in the form of `value`.
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
// member predictions from elsewhere combines them identically. Every
// dispersion is NaN for a single member, which measures no spread.
struct CombinedPrediction {
    torch::Tensor value;
    torch::Tensor agreement;   // vote only
    torch::Tensor dispersion;  // mean / circular / spherical mean only
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
// Where the members' unit vectors cancel (mean resultant length below
// kCancelledResultant) the combined bearing and its spread are NaN.
[[nodiscard]] CombinedPrediction combine_circular(const torch::Tensor& member_bearings,
                                                  double period);
// Points on the sphere: a (n_members, n_plots, 2) stack of latitude and
// longitude in degrees. The combined point is the members' mean unit vector
// projected back onto the sphere, as (n_plots, 2) latitude / longitude in
// degrees; its dispersion is the median great-circle distance of the members
// from it, in km (NaN for a single member). Where the members' vectors cancel
// (mean resultant length below kCancelledResultant) the combined point is
// undefined and both are NaN.
[[nodiscard]] CombinedPrediction combine_spherical(const torch::Tensor& member_coordinates);
// A member's bearing from its sine and cosine outputs, in [0, period). The
// pair need not have unit length: the network's two outputs are read as a
// direction.
[[nodiscard]] torch::Tensor bearing_from_components(const torch::Tensor& sine,
                                                    const torch::Tensor& cosine,
                                                    double period);

// The mean resultant length below which the members' unit vectors of a
// circular or spherical mean count as cancelling: they name no direction.
inline constexpr double kCancelledResultant = 1e-12;

// ============================================================================
// Combine rules
// ============================================================================

// The units of a rule's dispersion: none (a vote reports agreement), the
// target's own, or kilometres.
enum class DispersionUnits { None, TargetUnits, Kilometres };

// What one member contributes to a target: its prediction in the form of
// SuiteTargetPrediction::value, and for a vote its class probabilities.
struct SuiteMemberOutput {
    torch::Tensor value;
    torch::Tensor probabilities;
};

// One combine rule. A manifest target is checked against it, the predictor
// reads and combines member outputs through it, and its prediction is laid out
// as columns from it (suite_prediction_columns, the CLI and its usage text).
struct SuiteCombineRule {
    SuiteCombine combine;
    const char* noun;          // "a vote"
    TaskType task;             // the task of the heads it reads
    const char* combines;      // what it combines, for messages: "class predictions"
    std::size_t min_outputs;   // heads read, inclusive bounds
    std::size_t max_outputs;
    const char* outputs;       // the heads in words: "one output"
    bool needs_period;         // reads SuiteTarget::period
    const char* required_units;  // the units its outputs must be in; null: any
    int format_version;        // the first manifest format that holds it
    DispersionUnits dispersion_units;
    // Column suffixes of the combined value: one column, or one per component
    // (latitude, longitude). A classification rule writes the label under the
    // target's name and the code, agreement and probabilities beside it.
    std::array<const char*, 2> value_columns;
    const char* dispersion_column;   // suffix; null for a vote
    const char* dispersion_meaning;  // in words, for the usage text
    SuiteMemberOutput (*extract)(const SuiteTarget& target, const ResolvePredictions& member);
    // `mean_probabilities` is the members' mean class probabilities for a
    // vote and undefined otherwise.
    CombinedPrediction (*combine_members)(const torch::Tensor& members,
                                          const torch::Tensor& mean_probabilities,
                                          const SuiteTarget& target);

    [[nodiscard]] std::size_t n_value_columns() const noexcept {
        return value_columns[1] == nullptr ? 1 : 2;
    }
};

// Every rule, in SuiteCombine order, and the rule of one combine.
[[nodiscard]] std::span<const SuiteCombineRule> suite_combine_rules();
[[nodiscard]] const SuiteCombineRule& combine_rule(SuiteCombine combine);

// ============================================================================
// Predictions as columns
// ============================================================================

// One named column of a suite prediction table: text (plot IDs, class
// labels), integers (class codes) or numbers (NaN where undefined).
struct SuiteColumn {
    enum class Kind { Text, Integer, Number };
    std::string name;
    Kind kind = Kind::Number;
    bool probability = false;  // a class-probability column
    std::vector<std::string> text;
    std::vector<int64_t> integers;
    std::vector<float> numbers;
};

// The predictions as one row per plot: `plot_id`, then per target in order
// the prediction (for a classification rule `<t>` the class label, `<t>_code`
// and `<t>_agreement`, with `probabilities` `<t>_prob_<label>` per class; for
// a spherical mean `<t>_latitude` and `<t>_longitude`), its dispersion
// (`<t>_sd`, `<t>_circular_sd`, `<t>_spread_km`), `<t>_n_species`,
// `<t>_n_recognised`, `<t>_recognised_share`, `<t>_recognised_abundance_share`,
// and with kept members `<t>_seed<N>` per member (a label for a vote,
// `_latitude` / `_longitude` for a spherical mean). A class code is labelled
// by class_label. The CLI writes this table, and the Python and R data frames
// are built from it.
[[nodiscard]] std::vector<SuiteColumn> suite_prediction_columns(const SuitePredictions& predictions,
                                                                bool probabilities);

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

    // The caller's columns for the whole contract.
    [[nodiscard]] RoleMapping roles_for(const SuiteColumns& columns) const;
    // The part of `roles` an encoding's members read.
    [[nodiscard]] RoleMapping roles_for(const RoleMapping& roles, const Encoding& encoding) const;

    std::string dir_;
    SuiteManifest manifest_;
    std::vector<std::unique_ptr<Encoding>> encodings_;
    std::vector<LoadedTarget> targets_;
};

}  // namespace resolve
