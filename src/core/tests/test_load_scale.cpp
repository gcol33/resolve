#include <catch2/catch_test_macros.hpp>

#include "resolve/dataset.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/env.hpp"
#include "resolve/role_mapping.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace resolve;
using namespace resolve::test;

// ============================================================================
// Loading at archive scale
//
// Writes a synthetic vegetation archive (plots with a skewed species
// frequency, several records per plot, genus / family, cover with some zeros)
// and times one load of it, reporting the process memory the load peaked at.
// Hidden from the default runs; run one case per process, since the peak is a
// process-wide high-water mark:
//
//   resolve_tests "[.scale]"
//
// RESOLVE_SCALE_PLOTS (200000), RESOLVE_SCALE_SPECIES (20000),
// RESOLVE_SCALE_MEAN_SPECIES (10, records per plot) and RESOLVE_SCALE_ENCODING
// (hash) size the run; RESOLVE_SCALE_DIR keeps the CSVs there and reuses
// header.csv / species.csv already in it.
// ============================================================================

namespace {

struct MemorySample {
    double working_set_gb = 0.0;
    double peak_working_set_gb = 0.0;
    double commit_gb = 0.0;
    double peak_commit_gb = 0.0;
};

MemorySample sample_memory() {
    constexpr double kGb = 1024.0 * 1024.0 * 1024.0;
    MemorySample m;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        m.working_set_gb = static_cast<double>(counters.WorkingSetSize) / kGb;
        m.peak_working_set_gb = static_cast<double>(counters.PeakWorkingSetSize) / kGb;
        m.commit_gb = static_cast<double>(counters.PagefileUsage) / kGb;
        m.peak_commit_gb = static_cast<double>(counters.PeakPagefileUsage) / kGb;
    }
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    const double peak = static_cast<double>(usage.ru_maxrss) / kGb;  // bytes
#else
    const double peak = static_cast<double>(usage.ru_maxrss) * 1024.0 / kGb;  // KiB
#endif
    m.peak_working_set_gb = peak;
    m.peak_commit_gb = peak;
#endif
    return m;
}

int64_t env_int(const char* name, int64_t fallback) {
    const auto value = get_env(name);
    return value ? std::stoll(*value) : fallback;
}

// Writes header.csv (plot_id, lon, lat, c1..c3, area) and species.csv
// (plot_id, species, genus, family, cover) into `dir`.
void write_archive(const std::filesystem::path& dir, int64_t n_plots, int64_t n_species,
                   double mean_records) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::poisson_distribution<int> extra(std::max(mean_records - 1.0, 0.0));

    std::vector<double> cumulative(static_cast<size_t>(n_species));
    double total = 0.0;
    for (int64_t s = 0; s < n_species; ++s) {
        total += 1.0 / std::pow(static_cast<double>(s) + 10.0, 1.1);
        cumulative[static_cast<size_t>(s)] = total;
    }

    std::ofstream species(dir / "species.csv", std::ios::binary);
    std::ofstream header(dir / "header.csv", std::ios::binary);
    species << "plot_id,species,genus,family,cover\n";
    header << "plot_id,lon,lat,c1,c2,c3,area\n";
    std::string buffer;
    char line[160];
    for (int64_t p = 0; p < n_plots; ++p) {
        const int n = 1 + std::min(extra(rng), 119);
        for (int j = 0; j < n; ++j) {
            const double u = unit(rng) * total;
            const auto s = static_cast<int64_t>(
                std::lower_bound(cumulative.begin(), cumulative.end(), u) - cumulative.begin());
            const double cover = unit(rng) < 0.05 ? 0.0 : std::round(unit(rng) * 400.0) / 10.0;
            std::snprintf(line, sizeof(line), "P%07lld,G%04lld sp%05lld,G%04lld,F%03lld,%.1f\n",
                          static_cast<long long>(p), static_cast<long long>(s / 8),
                          static_cast<long long>(s), static_cast<long long>(s / 8),
                          static_cast<long long>(s / 160), cover);
            buffer += line;
        }
        if (buffer.size() > (size_t{1} << 20)) {
            species << buffer;
            buffer.clear();
        }
        std::snprintf(line, sizeof(line), "P%07lld,%.5f,%.5f,%.4f,%.4f,%.4f,%.2f\n",
                      static_cast<long long>(p), -10.0 + 40.0 * unit(rng),
                      35.0 + 35.0 * unit(rng), unit(rng) - 0.5, unit(rng) - 0.5,
                      unit(rng) - 0.5, 1.0 + 100.0 * unit(rng));
        header << line;
    }
    species << buffer;
}

}  // namespace

TEST_CASE("Loading a synthetic archive: time and peak memory", "[.scale]") {
    const int64_t n_plots = env_int("RESOLVE_SCALE_PLOTS", 200000);
    const int64_t n_species = env_int("RESOLVE_SCALE_SPECIES", 20000);
    const double mean_records = static_cast<double>(env_int("RESOLVE_SCALE_MEAN_SPECIES", 10));
    const std::string encoding = get_env("RESOLVE_SCALE_ENCODING").value_or("hash");

    std::unique_ptr<TempDir> scratch;
    std::filesystem::path dir;
    if (const auto given = get_env("RESOLVE_SCALE_DIR")) {
        dir = *given;
        std::filesystem::create_directories(dir);
    } else {
        scratch = std::make_unique<TempDir>();
        dir = scratch->path();
    }
    if (!std::filesystem::exists(dir / "species.csv") ||
        !std::filesystem::exists(dir / "header.csv")) {
        write_archive(dir, n_plots, n_species, mean_records);
    }

    RoleMapping roles;
    roles.plot_id = "plot_id";
    roles.species_id = "species";
    roles.abundance = "cover";
    roles.genus = "genus";
    roles.family = "family";
    roles.longitude = "lon";
    roles.latitude = "lat";
    roles.covariates = {"c1", "c2", "c3"};
    DatasetConfig config;
    config.species_encoding = parse_species_encoding_mode(encoding);

    const MemorySample before = sample_memory();
    const auto start = std::chrono::steady_clock::now();
    const auto dataset = ResolveDataset::from_csv(
        (dir / "header.csv").string(), (dir / "species.csv").string(), roles,
        {TargetSpec::regression("area")}, config);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const MemorySample after = sample_memory();

    std::printf("[scale] encoding=%s plots=%lld species_vocab=%lld load_s=%.2f "
                "peak_ws_delta_gb=%.3f peak_commit_delta_gb=%.3f ws_after_delta_gb=%.3f\n",
                encoding.c_str(), static_cast<long long>(dataset.n_plots()),
                static_cast<long long>(dataset.schema().n_species_vocab), seconds,
                after.peak_working_set_gb - before.working_set_gb,
                after.peak_commit_gb - before.commit_gb,
                after.working_set_gb - before.working_set_gb);
    std::fflush(stdout);
    REQUIRE(dataset.n_plots() > 0);
}
