// Shared fixtures for the Catch2 suite: scratch files and directories that
// clean up after themselves, the role / config builders most cases start from,
// the small synthetic CSV corpora several files read, and numeric helpers.
//
// Everything is inline in resolve::test, so a test file includes this header
// and adds `using namespace resolve::test;`.

#pragma once

#include "resolve/dataset.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/role_mapping.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace resolve::test {

// ============================================================================
// Scratch files
// ============================================================================

// Scratch names carry a per-process token: CTest can run the fast and the
// recovery entries of the one test executable side by side, and two processes
// each counting from zero would otherwise hand out the same file.
inline const std::string& process_token() {
    static const std::string token = [] {
        std::random_device rd;
        std::ostringstream out;
        out << std::hex << ((static_cast<std::uint64_t>(rd()) << 32) ^ rd());
        return out.str();
    }();
    return token;
}

inline std::filesystem::path unique_temp_path(const std::string& suffix) {
    static std::atomic<std::uint64_t> counter{0};
    return std::filesystem::temp_directory_path() /
           ("resolve_test_" + process_token() + "_" + std::to_string(counter.fetch_add(1)) +
            suffix);
}

// Text lets the platform translate '\n' (to CRLF on Windows); Binary writes
// the bytes as given.
enum class WriteMode { Text, Binary };

inline void write_file(const std::filesystem::path& path, const std::string& content,
                       WriteMode mode) {
    std::ofstream out(path, mode == WriteMode::Binary ? std::ios::out | std::ios::binary
                                                      : std::ios::out);
    out << content;
}

// A unique scratch path, removed on destruction. Nothing is created on disk
// until write() (or until the code under test writes there).
class TempPath {
public:
    explicit TempPath(const std::string& suffix = "") : path_(unique_temp_path(suffix)) {}
    ~TempPath() {
        if (path_.empty()) return;
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempPath(const TempPath&) = delete;
    TempPath& operator=(const TempPath&) = delete;
    TempPath(TempPath&& other) noexcept
        : path_(std::exchange(other.path_, std::filesystem::path{})) {}
    TempPath& operator=(TempPath&&) = delete;

    [[nodiscard]] std::string path() const { return path_.string(); }

    void write(const std::string& content, WriteMode mode = WriteMode::Text) const {
        write_file(path_, content, mode);
    }

private:
    std::filesystem::path path_;
};

// A scratch file holding `content`, removed on destruction.
class TempFile : public TempPath {
public:
    explicit TempFile(const std::string& content, const std::string& suffix = ".csv",
                      WriteMode mode = WriteMode::Text)
        : TempPath(suffix) {
        write(content, mode);
    }
};

// A scratch directory, removed with everything in it on destruction. Files are
// written byte-exact, so checksums over them match the content given.
class TempDir {
public:
    TempDir() : path_(unique_temp_path("_dir")) { std::filesystem::create_directories(path_); }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::string path() const { return path_.string(); }
    [[nodiscard]] std::string file(const std::string& name) const {
        return (path_ / name).string();
    }
    void write(const std::string& name, const std::string& content) const {
        std::filesystem::create_directories((path_ / name).parent_path());
        write_file(path_ / name, content, WriteMode::Binary);
    }

private:
    std::filesystem::path path_;
};

// ============================================================================
// Roles and configs
// ============================================================================

// A RoleMapping over the plot / species / "cover" columns the fixtures write,
// extended with the optional roles a case reads:
//
//   const RoleMapping roles = Roles().taxonomy().coordinates().covariates({"cov1"});
//
// taxonomy() maps "genus" / "family", coordinates() maps "lat" / "lon".
class Roles {
public:
    explicit Roles(std::string plot_id = "plot_id", std::string species_id = "sp") {
        roles_.plot_id = std::move(plot_id);
        roles_.species_id = std::move(species_id);
        roles_.abundance = "cover";
    }
    Roles& taxonomy() {
        roles_.genus = "genus";
        roles_.family = "family";
        return *this;
    }
    Roles& coordinates() {
        roles_.latitude = "lat";
        roles_.longitude = "lon";
        return *this;
    }
    Roles& covariates(std::vector<std::string> names) {
        roles_.covariates = std::move(names);
        return *this;
    }
    Roles& categoricals(std::vector<std::string> names) {
        roles_.categoricals = std::move(names);
        return *this;
    }
    operator RoleMapping() const { return roles_; }  // NOLINT(google-explicit-constructor)

private:
    RoleMapping roles_;
};

// No taxonomy and no unknown-species columns, so the continuous block holds
// only the plot's own coordinates and covariates.
inline DatasetConfig bare_dataset_config(SpeciesEncodingMode mode) {
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.use_taxonomy = false;
    cfg.track_unknown_fraction = false;
    cfg.track_unknown_count = false;
    return cfg;
}

inline DatasetConfig hash_dataset_config(int hash_dim = 4) {
    DatasetConfig cfg = bare_dataset_config(SpeciesEncodingMode::Hash);
    cfg.hash_dim = hash_dim;
    return cfg;
}

// Two genus/family slots, top-2 hash and top-3 embed selection, taxonomy on:
// the dataset side of vocab_model_config(). `track_unknown` adds the
// unknown-species fraction and count columns.
inline DatasetConfig vocab_dataset_config(SpeciesEncodingMode mode, bool track_unknown) {
    DatasetConfig cfg;
    cfg.species_encoding = mode;
    cfg.top_k = 2;
    cfg.top_k_species = 3;
    cfg.use_taxonomy = true;
    cfg.track_unknown_fraction = track_unknown;
    cfg.track_unknown_count = track_unknown;
    return cfg;
}

// A hash-encoded MLP.
inline ModelConfig hash_mlp_config(std::vector<int64_t> hidden_dims, int hash_dim = 4) {
    ModelConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Hash;
    cfg.encoder_architecture = EncoderArchitecture::MLP;
    cfg.hash_dim = hash_dim;
    cfg.hidden_dims = std::move(hidden_dims);
    return cfg;
}

// An MLP small enough to build quickly under any species encoding, matching
// vocab_dataset_config(): embeddings of width 8 (species) and 4 (genus,
// family), a transformer encoder without attention layers.
inline ModelConfig vocab_model_config(SpeciesEncodingMode mode) {
    ModelConfig cfg;
    cfg.species_encoding = mode;
    cfg.encoder_architecture = EncoderArchitecture::MLP;
    cfg.top_k = 2;
    cfg.top_k_species = 3;
    cfg.n_taxonomy_slots = 2;
    cfg.species_embed_dim = 8;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hidden_dims = {16, 8};
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.n_attention_layers = 0;
    return cfg;
}

// Every species encoding at a small width: 32 hash buckets, embeddings of 8
// (species) and 4 (genus, family), a one-layer two-head transformer.
inline ModelConfig small_encoder_config(SpeciesEncodingMode mode,
                                        std::vector<int64_t> hidden_dims) {
    ModelConfig cfg;
    cfg.species_encoding = mode;
    cfg.hidden_dims = std::move(hidden_dims);
    cfg.species_embed_dim = 8;
    cfg.genus_emb_dim = 4;
    cfg.family_emb_dim = 4;
    cfg.hash_dim = 32;
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.n_attention_layers = 1;
    cfg.transformer_ff_dim = 16;
    return cfg;
}

// The spec of a standalone species encoder: the given table sizes, the knobs
// of small_encoder_config(mode, hidden_dims).
inline SpeciesEncoderSpec encoder_spec(SpeciesEncodingMode mode, int64_t n_continuous,
                                       int64_t n_species, int64_t n_genera = 0,
                                       int64_t n_families = 0,
                                       std::vector<int64_t> hidden_dims = {32, 16}) {
    SpeciesEncoderSpec spec;
    spec.n_continuous = n_continuous;
    spec.n_species = n_species;
    spec.n_genera = n_genera;
    spec.n_families = n_families;
    spec.config = small_encoder_config(mode, std::move(hidden_dims));
    return spec;
}

// A silent CPU run of `epochs` epochs at batch size 8: for cases that need a
// prepared, saved or briefly trained trainer rather than a fitted one.
inline TrainConfig quiet_train_config(int epochs = 1) {
    TrainConfig cfg;
    cfg.batch_size = 8;
    cfg.max_epochs = epochs;
    cfg.patience = epochs;
    cfg.lr = 1e-3f;
    cfg.device = torch::kCPU;
    cfg.log = null_log;
    return cfg;
}

// The fit-to-convergence schedule of the recovery cases.
inline TrainConfig recovery_train_config(int max_epochs = 300) {
    TrainConfig cfg;
    cfg.batch_size = 64;
    cfg.max_epochs = max_epochs;
    cfg.patience = 40;
    cfg.lr = 1e-2f;
    cfg.device = torch::kCPU;
    return cfg;
}

// ============================================================================
// Synthetic corpora
// ============================================================================

// Plots on a straight line: plot i ("p<i>") sits at
// (lon0 + lon_step i, lat0 + lat_step i) with elevation elev0 + elev_step i and
// target y0 + y_step i. Columns: plot_id,lon,lat,elev,y.
struct PlotLine {
    int n_plots;
    double lon0, lon_step;
    double lat0, lat_step;
    double elev0, elev_step;
    double y0, y_step;
};

inline std::string line_header_csv(const PlotLine& line) {
    std::ostringstream out;
    out << "plot_id,lon,lat,elev,y\n";
    for (int i = 0; i < line.n_plots; ++i) {
        out << "p" << i << "," << (line.lon0 + line.lon_step * i) << ","
            << (line.lat0 + line.lat_step * i) << "," << (line.elev0 + line.elev_step * i)
            << "," << (line.y0 + line.y_step * i) << "\n";
    }
    return out.str();
}

// Every plot records all `n_species` species ("sp_<j>", genus gen_<j % 3>,
// family fam_<j % 2>), with covers cycling through 1..n_species so each plot
// ranks them differently. Columns: plot_id,sp,cover,genus,family.
inline std::string cyclic_species_csv(int n_plots, int n_species) {
    std::ostringstream out;
    out << "plot_id,sp,cover,genus,family\n";
    for (int i = 0; i < n_plots; ++i) {
        for (int j = 0; j < n_species; ++j) {
            out << "p" << i << ",sp_" << j << "," << (((j + i) % n_species) + 1)
                << ",gen_" << (j % 3) << ",fam_" << (j % 2) << "\n";
        }
    }
    return out.str();
}

// A 40-plot line (lon 10.0 + 0.1 i, lat 47.0 + 0.1 i, elev 100 + 5 i,
// y 1.0 + 0.25 i) with six species in every plot under cyclic covers, on disk
// for the lifetime of the object: enough rows for a short fit under any
// species encoding, with an "elev" covariate and the "y" regression target.
class SmallCorpus {
public:
    static constexpr int kPlots = 40;
    static constexpr int kSpecies = 6;

    SmallCorpus()
        : header_(line_header_csv({kPlots, 10.0, 0.1, 47.0, 0.1, 100.0, 5.0, 1.0, 0.25})),
          species_(cyclic_species_csv(kPlots, kSpecies)) {}

    [[nodiscard]] std::string header_path() const { return header_.path(); }
    [[nodiscard]] std::string species_path() const { return species_.path(); }

    // The corpus under `mode`; `taxonomy` maps and encodes genus / family.
    [[nodiscard]] ResolveDataset load(SpeciesEncodingMode mode, bool taxonomy = true) const {
        Roles roles;
        roles.coordinates().covariates({"elev"});
        if (taxonomy) roles.taxonomy();
        DatasetConfig cfg;
        cfg.species_encoding = mode;
        cfg.use_taxonomy = taxonomy;
        return ResolveDataset::from_csv(header_path(), species_path(), roles,
                                        {TargetSpec::regression("y")}, cfg);
    }

private:
    TempFile header_;
    TempFile species_;
};

// 80 plots with covariates "cov1" / "cov2", a regression target "y" and two
// classification targets, on disk for the lifetime of the object: "cls"
// (three classes set by cov1, with one plot in seven relabelled, so the
// held-out loss and accuracy of a short fit move apart) and "cls2" (two
// classes). Each plot holds one of ten background species and one species
// marking its true "cls" class. load() reads the targets a case names, hash
// encoded.
class ClassificationCorpus {
public:
    static constexpr int kPlots = 80;
    static constexpr int kClasses = 3;

    ClassificationCorpus() : header_(header_csv()), species_(species_csv()) {}

    [[nodiscard]] ResolveDataset load(std::vector<TargetSpec> targets) const {
        return ResolveDataset::from_csv(header_.path(), species_.path(),
                                        Roles().coordinates().covariates({"cov1", "cov2"}),
                                        std::move(targets), hash_dataset_config());
    }

    // The class cov1 sets for plot i, before relabelling.
    [[nodiscard]] static int true_class(int i) {
        const double c1 = std::sin(0.37 * i);
        return std::min(kClasses - 1, static_cast<int>((c1 + 1.0) * 0.5 * kClasses));
    }

private:
    static std::string header_csv() {
        std::ostringstream out;
        out << "plot_id,lat,lon,cov1,cov2,y,cls,cls2\n";
        for (int i = 0; i < kPlots; ++i) {
            const double c1 = std::sin(0.37 * i);
            const double c2 = std::cos(0.11 * i);
            const int cls = i % 7 == 0 ? (true_class(i) + 1) % kClasses : true_class(i);
            out << "P" << i << "," << (45.0 + 0.01 * i) << "," << (7.0 + 0.02 * i) << "," << c1
                << "," << c2 << "," << (3.0 * c1 - c2 + 1.0) << "," << cls << "," << (i % 2)
                << "\n";
        }
        return out.str();
    }
    static std::string species_csv() {
        std::ostringstream out;
        out << "plot_id,sp,cover\n";
        for (int i = 0; i < kPlots; ++i) {
            out << "P" << i << ",sp" << (i % 10) << "," << (1 + i % 4) << "\n";
            out << "P" << i << ",mark" << true_class(i) << ",1\n";
        }
        return out.str();
    }

    TempFile header_;
    TempFile species_;
};

// ============================================================================
// Encodings and modules
// ============================================================================

// Every species encoding, read off the engine's name table so a new encoding
// joins each per-encoding case. species_encoding_to_string() names one.
inline std::vector<SpeciesEncodingMode> all_species_encodings() {
    std::vector<SpeciesEncodingMode> modes;
    for (const auto& entry : kSpeciesEncodingNames) modes.push_back(entry.value);
    return modes;
}

inline std::set<std::string> parameter_names(const torch::nn::Module& module) {
    std::set<std::string> names;
    for (const auto& item : module.named_parameters()) names.insert(item.key());
    return names;
}

inline bool any_starts_with(const std::set<std::string>& names, const std::string& prefix) {
    for (const auto& name : names) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

// ============================================================================
// Numeric helpers
// ============================================================================

// Pearson correlation of two equal-length series; 0 when either is constant.
inline double pearson(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = a.size();
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= static_cast<double>(n);
    mb /= static_cast<double>(n);
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if (saa <= 0 || sbb <= 0) return 0.0;
    return sab / std::sqrt(saa * sbb);
}

}  // namespace resolve::test
