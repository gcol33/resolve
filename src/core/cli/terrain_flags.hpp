// RESOLVE CLI - terrain correction: the `resolve predict --suite` flags that
// switch it on, the columns they name, and the flag tables of
// `resolve terrain <action>`.

#ifndef RESOLVE_CLI_TERRAIN_FLAGS_HPP
#define RESOLVE_CLI_TERRAIN_FLAGS_HPP

#include <optional>
#include <string>
#include <vector>

#include "resolve/enum_names.hpp"
#include "resolve/suite.hpp"
#include "resolve/terrain_codec.hpp"
#include "resolve/terrain_correction.hpp"
#include "resolve/terrain_field.hpp"

#include "arg_parser.hpp"
#include "cli_common.hpp"
#include "config_flags.hpp"

namespace resolve_cli {

// The terrain rows of the `predict` flag table.
inline void append_terrain_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--terrain", Arity::Value, "SOURCE", "",
                     "With --suite: correct each target the suite gives a\n"
                     "terrain correction (altitude, slope) by an\n"
                     "elevation model read at the plots: 'field' (the\n"
                     "suite's terrain field, at --lat/--lon), 'grids'\n"
                     "(--terrain-grids, at --lat/--lon) or 'columns'\n"
                     "(your own values, --terrain-elevation ...). Writes\n"
                     "'<target>_terrain', the weight of each source and\n"
                     "the descriptors read beside the raw prediction."});
    flags.push_back({"--terrain-grids", Arity::Value, "DIR", "",
                     "With --terrain grids: the directory holding the\n"
                     "elevation grids' tiles.json."});
    flags.push_back({"--terrain-elevation", Arity::Value, "COL", "",
                     "With --terrain columns: the elevation (m) at each\n"
                     "plot, read by an altitude target."});
    flags.push_back({"--terrain-elevation-1km", Arity::Value, "COL", "",
                     "With --terrain columns: the mean elevation (m) over\n"
                     "the 1 km square around each plot (optional)."});
    flags.push_back({"--terrain-slope", Arity::Value, "COL", "",
                     "With --terrain columns: the slope (degrees) at each\n"
                     "plot, read by a slope target."});
    flags.push_back({"--terrain-slope-1km", Arity::Value, "COL", "",
                     "With --terrain columns: the inclination (degrees) of\n"
                     "the 1 km square around each plot (optional)."});
    flags.push_back({"--location-uncertainty", Arity::Value, "COL", "",
                     "With --terrain: the recorded location uncertainty\n"
                     "(m) of each plot; unset reads every plot as not\n"
                     "recorded."});
}

// The terrain options and their columns, or nullopt without --terrain (where
// a terrain flag given anyway is an error).
inline std::optional<resolve::SuiteTerrainOptions> read_terrain(const ParsedArgs& args,
                                                                resolve::SuiteColumns& columns,
                                                                torch::Device device) {
    if (!args.has("--terrain")) {
        for (const char* flag : {"--terrain-grids", "--terrain-elevation", "--terrain-elevation-1km",
                                 "--terrain-slope", "--terrain-slope-1km", "--location-uncertainty"}) {
            if (args.has(flag)) throw ArgError(std::string(flag) + " needs --terrain");
        }
        return std::nullopt;
    }
    resolve::SuiteTerrainOptions options;
    try {
        options.source = resolve::enum_from_name<resolve::TerrainSourceKind>(args.get("--terrain"));
    } catch (const std::exception&) {
        throw ArgError("--terrain: unknown source '" + args.get("--terrain") + "'; accepted: " +
                       resolve::enum_spellings<resolve::TerrainSourceKind>());
    }
    options.grids = args.get("--terrain-grids");
    options.device = device;
    if (options.source == resolve::TerrainSourceKind::Grids && options.grids.empty()) {
        throw ArgError("--terrain grids needs --terrain-grids DIR");
    }
    if (options.source != resolve::TerrainSourceKind::Grids && !options.grids.empty()) {
        throw ArgError("--terrain-grids is read by --terrain grids only");
    }
    columns.terrain.elevation = role_flag(args, "--terrain-elevation");
    columns.terrain.elevation_1km = role_flag(args, "--terrain-elevation-1km");
    columns.terrain.slope = role_flag(args, "--terrain-slope");
    columns.terrain.slope_1km = role_flag(args, "--terrain-slope-1km");
    columns.terrain.uncertainty = role_flag(args, "--location-uncertainty");
    const bool any_column = columns.terrain.elevation || columns.terrain.elevation_1km ||
                            columns.terrain.slope || columns.terrain.slope_1km;
    if (options.source != resolve::TerrainSourceKind::Columns && any_column) {
        throw ArgError("--terrain-elevation / --terrain-slope (and their -1km) are read by --terrain columns only");
    }
    return options;
}

// ---------------------------------------------------------------------------
// resolve terrain <action>
// ---------------------------------------------------------------------------

inline constexpr FieldFlag kTerrainFieldConfigFlags[] = {
    {"levels", "--levels", "Hash-encoding levels"},
    {"log2_table_size", "--log2-table-size", "Entries per level, as a power of two"},
    {"features", "--features", "Features per hash-table entry"},
    {"coarsest_resolution", "--coarsest-resolution", "Grid cells per side at the coarsest level"},
    {"finest_fraction", "--finest-fraction",
     "Finest level's cells per side as a fraction of the\ntile's larger side"},
    {"hidden_dim", "--hidden-dim", "Width of the detail network"},
    {"n_layers", "--n-layers", "Hidden layers of the detail network"},
    {"base_cells", "--base-cells", "Blocks per side of the coarse base"},
    {"steps", "--steps", "Optimizer steps per tile"},
    {"batch_size", "--batch-size", "Cells per step"},
    {"lr", "--lr", "Peak learning rate"},
    {"gradient_weight", "--gradient-weight",
     "Weight of the four-neighbour difference term\n(slope fidelity)"},
    {"seed", "--seed", "Seed of the initialisation and the cell draws"},
};

inline constexpr FieldFlag kTerrainCorrectionConfigFlags[] = {
    {"scale", "--scale", "The target's scale (500 for altitude in m, 20 for\nslope in degrees)"},
    {"offset_fraction", "--offset-fraction", "Bound of the free offset, as a fraction of --scale"},
    {"hidden_dim", "--hidden-dim", "Width of the gating network"},
    {"n_layers", "--n-layers", "Hidden layers of the gating network"},
    {"epochs", "--epochs", "Training epochs"},
    {"batch_size", "--batch-size", "Plots per step"},
    {"lr", "--lr", "Peak learning rate"},
    {"weight_decay", "--weight-decay", "AdamW weight decay"},
    {"dem_1km_dropout", "--dem-1km-dropout",
     "Share of plots whose 1 km descriptor is hidden each\nepoch"},
    {"seed", "--seed", "Seed of the initialisation and the batches"},
};

inline const CommandSpec& terrain_fit_field_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--grids", Arity::Value, "DIR", "",
                         "Elevation grids (a tiles.json and its .npy tiles)"});
        flags.push_back({"--out", Arity::Value, "DIR", "",
                         "Directory the tile fields are written to,\n'<tile>.tile' each; a tile already there is skipped"});
        flags.push_back({"--tile", Arity::Repeatable, "NAME", "",
                         "Fit only this tile. Repeat per tile; default every\ntile of --grids."});
        flags.push_back({"--shard", Arity::Value, "I/N", "",
                         "Fit the I-th of N interleaved shares of the tiles\n(0-based), for N parallel runs"});
        flags.push_back({"--device", Arity::Value, "DEVICE", "cpu", "cpu, cuda or cuda:N"});
        append_config_flags(flags, "Terrain field", resolve::TerrainFieldConfig{},
                            FlagNaming::members(kTerrainFieldConfigFlags));
        return CommandSpec("terrain fit-field", "Terrain fit-field Options:", std::move(flags));
    }();
    return spec;
}

inline constexpr FieldFlag kTerrainCodecConfigFlags[] = {
    {"max_error", "--max-error", "Largest error a cell may carry, m"},
    {"block", "--block", "Cells per side of an independently coded block"},
};

inline const CommandSpec& terrain_encode_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--grids", Arity::Value, "DIR", "",
                         "Elevation grids (a tiles.json and its .npy tiles)"});
        flags.push_back({"--out", Arity::Value, "PATH", "terrain_field.bin",
                         "The coded terrain field file to write"});
        flags.push_back({"--tile", Arity::Repeatable, "NAME", "",
                         "Code only this tile. Repeat per tile; default every\ntile of --grids."});
        append_config_flags(flags, "Terrain codec", resolve::TerrainCodecConfig{},
                            FlagNaming::members(kTerrainCodecConfigFlags));
        return CommandSpec("terrain encode", "Terrain encode Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& terrain_assemble_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--tiles", Arity::Value, "DIR", "",
                         "Directory of tile fields written by fit-field"});
        flags.push_back({"--out", Arity::Value, "PATH", "terrain_field.bin",
                         "The terrain field file to write"});
        return CommandSpec("terrain assemble", "Terrain assemble Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& terrain_describe_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--grids", Arity::Value, "DIR", "", "Read the elevation grids in DIR"});
        flags.push_back({"--field", Arity::Value, "PATH", "",
                         "Read a terrain field file (neural or coded) instead"});
        flags.push_back({"--header", Arity::Value, "PATH", "",
                         "CSV with one row per plot and its coordinates"});
        flags.push_back({"--plot-id", Arity::Value, "COL", "plot_id", "Plot id column"});
        flags.push_back({"--lat", Arity::Value, "COL", "latitude", "Latitude column (degrees)"});
        flags.push_back({"--lon", Arity::Value, "COL", "longitude", "Longitude column (degrees)"});
        flags.push_back({"--output", Arity::Value, "PATH", "terrain.csv",
                         "Writes plot_id, elevation, elevation_1km, slope,\nslope_1km"});
        flags.push_back({"--device", Arity::Value, "DEVICE", "cpu", "cpu, cuda or cuda:N (--field)"});
        return CommandSpec("terrain describe", "Terrain describe Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& terrain_fit_correction_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--table", Arity::Value, "PATH", "",
                         "CSV with one row per plot: its recorded value, the\nprediction and the descriptors"});
        flags.push_back({"--target", Arity::Value, "NAME", "", "The suite target corrected"});
        flags.push_back({"--descriptor", Arity::Value, "KIND", "elevation",
                         "elevation (an altitude target) or slope"});
        flags.push_back({"--plot-id", Arity::Value, "COL", "plot_id", "Plot id column"});
        flags.push_back({"--truth", Arity::Value, "COL", "", "Recorded value"});
        flags.push_back({"--prediction", Arity::Value, "COL", "", "The suite's combined prediction"});
        flags.push_back({"--dem", Arity::Value, "COL", "", "Descriptor at the coordinate"});
        flags.push_back({"--dem-1km", Arity::Value, "COL", "", "Descriptor over 1 km (optional)"});
        flags.push_back({"--uncertainty", Arity::Value, "COL", "", "Location uncertainty, m (optional)"});
        flags.push_back({"--spread", Arity::Value, "COL", "", "Members' standard deviation (optional)"});
        flags.push_back({"--recognised-share", Arity::Value, "COL", "",
                         "Share of species recognised (optional)"});
        flags.push_back({"--n-species", Arity::Value, "COL", "", "Species per plot (optional)"});
        flags.push_back({"--out", Arity::Value, "PATH", "", "The terrain correction file to write"});
        flags.push_back({"--device", Arity::Value, "DEVICE", "cpu", "cpu, cuda or cuda:N"});
        append_config_flags(flags, "Terrain correction", resolve::TerrainCorrectionConfig{},
                            FlagNaming::members(kTerrainCorrectionConfigFlags));
        return CommandSpec("terrain fit-correction", "Terrain fit-correction Options:", std::move(flags));
    }();
    return spec;
}

}  // namespace resolve_cli

#endif  // RESOLVE_CLI_TERRAIN_FLAGS_HPP
