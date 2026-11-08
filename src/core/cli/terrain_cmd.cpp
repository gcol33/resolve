// RESOLVE CLI - `resolve terrain <action>`: fit a terrain field tile by tile,
// assemble the tiles into one field file, read the terrain descriptors at
// plot coordinates, and fit a terrain correction.
//
// Reads its values from the ParsedArgs produced by the terrain flag tables in
// terrain_flags.hpp.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "resolve/plot_columns.hpp"
#include "resolve/resolve.hpp"
#include "resolve/terrain_codec.hpp"
#include "resolve/terrain_correction.hpp"
#include "resolve/terrain_field.hpp"
#include "resolve/terrain_source.hpp"

#include "arg_parser.hpp"
#include "cli_common.hpp"
#include "csv_output.hpp"
#include "terrain_flags.hpp"

using resolve_cli::ArgError;
using resolve_cli::ParsedArgs;
namespace fs = std::filesystem;

namespace {

// --shard I/N as (I, N); (0, 1) when not given.
std::pair<int64_t, int64_t> read_shard(const ParsedArgs& args) {
    const std::string value = args.get("--shard");
    if (value.empty()) return {0, 1};
    const auto slash = value.find('/');
    if (slash == std::string::npos) throw ArgError("--shard expects I/N, got '" + value + "'");
    const int64_t i = resolve_cli::parse_int64_value("--shard", value.substr(0, slash));
    const int64_t n = resolve_cli::parse_int64_value("--shard", value.substr(slash + 1));
    if (n < 1 || i < 0 || i >= n) throw ArgError("--shard I/N needs 0 <= I < N, got '" + value + "'");
    return {i, n};
}

torch::Tensor column_tensor(const std::vector<double>& v) { return torch::tensor(v, torch::kFloat64); }

}  // namespace

int terrain_fit_field_command(const ParsedArgs& args) {
    using namespace resolve;
    try {
        const std::string grids_dir = args.get("--grids");
        const std::string out_dir = args.get("--out");
        if (grids_dir.empty() || out_dir.empty()) throw ArgError("terrain fit-field needs --grids and --out");
        TerrainFieldConfig config;
        resolve_cli::read_config_flags(config, args, resolve_cli::FlagNaming::members(resolve_cli::kTerrainFieldConfigFlags));
        config.device = resolve_cli::choose_device(args);
        config.validate();
        const auto [shard, shards] = read_shard(args);
        const std::vector<std::string> only = args.get_all("--tile");

        const TerrainGrids grids(grids_dir, 2);
        fs::create_directories(out_dir);
        int64_t fitted = 0;
        int64_t skipped = 0;
        int64_t position = 0;
        for (int64_t t = 0; t < grids.index().size(); ++t) {
            const TerrainTile& tile = grids.index().tiles()[static_cast<size_t>(t)];
            if (!only.empty() && std::find(only.begin(), only.end(), tile.name) == only.end()) continue;
            if (position++ % shards != shard) continue;
            const std::string path = (fs::path(out_dir) / (tile.name + ".tile")).string();
            if (fs::exists(path)) {
                ++skipped;
                continue;
            }
            const TerrainTileMetrics m = fit_terrain_tile(grids.grid(t), tile, config, path);
            ++fitted;
            std::cout << tile.name << ": rmse " << m.rmse << " m, p99 " << m.p99 << " m, max " << m.max_abs
                      << " m, slope mae " << m.slope_mae << " deg, " << m.bytes << " bytes, " << m.seconds << " s"
                      << std::endl;
        }
        std::cout << "Fitted " << fitted << " tile(s), " << skipped << " already in " << out_dir << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int terrain_encode_command(const ParsedArgs& args) {
    using namespace resolve;
    try {
        const std::string grids_dir = args.get("--grids");
        if (grids_dir.empty()) throw ArgError("terrain encode needs --grids");
        TerrainCodecConfig config;
        resolve_cli::read_config_flags(config, args, resolve_cli::FlagNaming::members(resolve_cli::kTerrainCodecConfigFlags));
        config.validate();
        const TerrainGrids grids(grids_dir, 2);
        const auto metrics = encode_terrain_grids(grids, config, args.get("--out"), args.get_all("--tile"));
        int64_t bytes = 0;
        int64_t cells = 0;
        double worst = 0.0;
        double sq = 0.0;
        for (const auto& [name, m] : metrics) {
            bytes += m.bytes;
            const TerrainTile& t = grids.index().tiles()[static_cast<size_t>(
                std::find_if(grids.index().tiles().begin(), grids.index().tiles().end(),
                             [&](const TerrainTile& x) { return x.name == name; }) -
                grids.index().tiles().begin())];
            cells += t.rows * t.cols;
            worst = std::max(worst, m.max_abs);
            sq += m.rmse * m.rmse * static_cast<double>(t.rows * t.cols);
        }
        std::cout << "Coded " << metrics.size() << " tile(s) into " << args.get("--out") << ": " << bytes << " bytes, "
                  << 8.0 * static_cast<double>(bytes) / static_cast<double>(cells) << " bits per cell, rmse "
                  << std::sqrt(sq / static_cast<double>(cells)) << " m, largest error " << worst << " m" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int terrain_assemble_command(const ParsedArgs& args) {
    using namespace resolve;
    try {
        const std::string tiles_dir = args.get("--tiles");
        const std::string out = args.get("--out");
        if (tiles_dir.empty()) throw ArgError("terrain assemble needs --tiles");
        std::vector<std::string> paths;
        for (const auto& entry : fs::directory_iterator(tiles_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".tile") paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
        if (paths.empty()) throw ArgError("no .tile files in '" + tiles_dir + "'");
        assemble_terrain_field(paths, out);
        const TerrainField field(out);
        std::vector<double> rmse;
        int64_t bytes = 0;
        for (const auto& m : field.metrics()) {
            rmse.push_back(m.rmse);
            bytes += m.bytes;
        }
        std::sort(rmse.begin(), rmse.end());
        std::cout << "Assembled " << paths.size() << " tile(s) into " << out << " ("
                  << fs::file_size(out) << " bytes); tile rmse median " << rmse[rmse.size() / 2] << " m, max "
                  << rmse.back() << " m" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int terrain_describe_command(const ParsedArgs& args) {
    using namespace resolve;
    try {
        const std::string grids_dir = args.get("--grids");
        const std::string field_path = args.get("--field");
        if (grids_dir.empty() == field_path.empty()) {
            throw ArgError("terrain describe reads --grids DIR or --field PATH (one of them)");
        }
        const std::string header = args.get("--header");
        if (header.empty()) throw ArgError("terrain describe needs --header");
        const TableColumns table = read_table_columns(TableSource::csv(header), args.get("--plot-id"),
                                                      {args.get("--lat"), args.get("--lon")}, "terrain describe");
        std::shared_ptr<TerrainSource> source;
        if (!grids_dir.empty()) {
            source = std::make_shared<TerrainGrids>(grids_dir);
        } else {
            source = open_terrain_field(field_path, resolve_cli::choose_device(args));
        }
        const TerrainDescriptors d =
            terrain_descriptors(*source, column_tensor(table.values[0]), column_tensor(table.values[1]));
        std::ofstream out(args.get("--output"));
        if (!out.is_open()) throw ArgError("cannot open " + args.get("--output"));
        out << "plot_id,elevation,elevation_1km,slope,slope_1km\n";
        const auto e = d.elevation.accessor<double, 1>();
        const auto e1 = d.elevation_1km.accessor<double, 1>();
        const auto s = d.slope.accessor<double, 1>();
        const auto s1 = d.slope_1km.accessor<double, 1>();
        for (size_t i = 0; i < table.plot_ids.size(); ++i) {
            const auto r = static_cast<int64_t>(i);
            out << resolve_cli::csv_field(table.plot_ids[i]) << "," << resolve_cli::csv_number(static_cast<float>(e[r]))
                << "," << resolve_cli::csv_number(static_cast<float>(e1[r])) << ","
                << resolve_cli::csv_number(static_cast<float>(s[r])) << ","
                << resolve_cli::csv_number(static_cast<float>(s1[r])) << "\n";
        }
        std::cout << "Described " << table.plot_ids.size() << " plot(s) from " << source->describe() << " into "
                  << args.get("--output") << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int terrain_fit_correction_command(const ParsedArgs& args) {
    using namespace resolve;
    try {
        for (const char* flag : {"--table", "--target", "--truth", "--prediction", "--dem", "--out"}) {
            if (args.get(flag).empty()) throw ArgError(std::string("terrain fit-correction needs ") + flag);
        }
        TerrainDescriptor descriptor;
        try {
            descriptor = enum_from_name<TerrainDescriptor>(args.get("--descriptor"));
        } catch (const std::exception&) {
            throw ArgError("--descriptor: accepted: " + enum_spellings<TerrainDescriptor>());
        }
        std::vector<std::string> names = {args.get("--truth"), args.get("--prediction"), args.get("--dem")};
        const std::vector<const char*> optional = {"--dem-1km", "--uncertainty", "--spread", "--recognised-share",
                                                   "--n-species"};
        for (const char* flag : optional) {
            if (!args.get(flag).empty()) names.push_back(args.get(flag));
        }
        const TableColumns table =
            read_table_columns(TableSource::csv(args.get("--table")), args.get("--plot-id"), names, "terrain fit-correction");
        std::size_t next = 3;
        const auto take = [&](const char* flag) {
            return args.get(flag).empty() ? torch::Tensor() : column_tensor(table.values[next++]);
        };
        TerrainCorrectionInputs in;
        in.prediction = column_tensor(table.values[1]);
        in.dem = column_tensor(table.values[2]);
        in.dem_1km = take("--dem-1km");
        in.uncertainty_m = take("--uncertainty");
        in.spread = take("--spread");
        in.recognised_share = take("--recognised-share");
        in.n_species = take("--n-species");

        TerrainCorrectionConfig config;
        resolve_cli::read_config_flags(config, args,
                                       resolve_cli::FlagNaming::members(resolve_cli::kTerrainCorrectionConfigFlags));
        config.device = resolve_cli::choose_device(args);
        const TerrainCorrection model =
            fit_terrain_correction(args.get("--target"), descriptor, in, column_tensor(table.values[0]), config);
        model.save(args.get("--out"));
        const TerrainCorrectionReport& r = model.report();
        std::cout << "Fitted the terrain correction of '" << r.target << "' on " << r.n_train
                  << " plots: mean absolute error raw " << r.mae_raw << ", elevation model " << r.mae_dem
                  << ", corrected " << r.mae_corrected << " (training plots); wrote " << args.get("--out")
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
