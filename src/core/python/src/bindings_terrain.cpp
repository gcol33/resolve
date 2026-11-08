#include "bindings_common.hpp"

#include "resolve/config_registry.hpp"
#include "resolve/json.hpp"
#include "resolve/terrain_codec.hpp"
#include "resolve/terrain_correction.hpp"
#include "resolve/terrain_field.hpp"
#include "resolve/terrain_source.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

using namespace resolve;

namespace {

nb::object json_to_python(const json::Value& value) {
    return nb::module_::import_("json").attr("loads")(json::dump(value, 0));
}

TerrainCorrectionInputs inputs_from(const nb::object& prediction, const nb::object& dem, const nb::object& dem_1km,
                                    const nb::object& uncertainty_m, const nb::object& spread,
                                    const nb::object& recognised_share, const nb::object& n_species) {
    TerrainCorrectionInputs in;
    in.prediction = unpack_required_tensor(prediction, "prediction");
    in.dem = unpack_required_tensor(dem, "dem");
    in.dem_1km = unpack_optional_tensor(dem_1km);
    in.uncertainty_m = unpack_optional_tensor(uncertainty_m);
    in.spread = unpack_optional_tensor(spread);
    in.recognised_share = unpack_optional_tensor(recognised_share);
    in.n_species = unpack_optional_tensor(n_species);
    return in;
}

}  // namespace

void register_terrain(nb::module_& m) {
    {
        using Cfg = TerrainCorrectionConfig;
        auto cls = nb::class_<Cfg>(m, "TerrainCorrectionConfig",
                                   "Terrain correction settings (fit_terrain_correction).");
        cls.def(nb::init<>());
        RESOLVE_TERRAIN_CORRECTION_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
        bind_device_property(cls, &Cfg::device);
        cls.def("validate", &Cfg::validate);
    }
    {
        using Cfg = TerrainFieldConfig;
        auto cls = nb::class_<Cfg>(m, "TerrainFieldConfig", "Terrain field settings (fit_terrain_tile).");
        cls.def(nb::init<>());
        RESOLVE_TERRAIN_FIELD_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
        bind_device_property(cls, &Cfg::device);
        cls.def("validate", &Cfg::validate);
    }

    {
        using Cfg = TerrainCodecConfig;
        auto cls = nb::class_<Cfg>(m, "TerrainCodecConfig",
                                   "Coded terrain field settings (encode_terrain_grids): the largest error a "
                                   "cell may carry and the block size.");
        cls.def(nb::init<>());
        RESOLVE_TERRAIN_CODEC_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
        cls.def("validate", &Cfg::validate);
    }

    nb::class_<TerrainTile>(m, "TerrainTile",
                            "One tile of an elevation model: rows x cols cells, cell (r, c) centred at "
                            "(lat_first - r * dlat, lon_first + c * dlon).")
        .def(nb::init<>())
        .def_rw("name", &TerrainTile::name)
        .def_rw("rows", &TerrainTile::rows)
        .def_rw("cols", &TerrainTile::cols)
        .def_rw("lat_first", &TerrainTile::lat_first)
        .def_rw("lon_first", &TerrainTile::lon_first)
        .def_rw("dlat", &TerrainTile::dlat)
        .def_rw("dlon", &TerrainTile::dlon);

    nb::class_<TerrainDescriptors>(m, "TerrainDescriptors",
                                   "Elevation-model descriptors per plot, NaN where the source holds nothing.")
        .def_prop_ro("elevation", [](const TerrainDescriptors& d) { return tensor_or_none(d.elevation); })
        .def_prop_ro("elevation_1km", [](const TerrainDescriptors& d) { return tensor_or_none(d.elevation_1km); })
        .def_prop_ro("slope", [](const TerrainDescriptors& d) { return tensor_or_none(d.slope); })
        .def_prop_ro("slope_1km", [](const TerrainDescriptors& d) { return tensor_or_none(d.slope_1km); });

    nb::class_<TerrainSource>(m, "TerrainSource", "A source of elevations (TerrainGrids, TerrainField).")
        .def("elevation",
             [](const TerrainSource& s, const nb::object& lat, const nb::object& lon) {
                 const torch::Tensor la = unpack_required_tensor(lat, "lat");
                 const torch::Tensor lo = unpack_required_tensor(lon, "lon");
                 torch::Tensor z;
                 {
                     nb::gil_scoped_release release;
                     z = s.elevation(la, lo);
                 }
                 return tensor_or_none(z);
             },
             nb::arg("lat"), nb::arg("lon"),
             "The elevation (m) at each point, bilinear between cell centres; NaN outside every tile.")
        .def_prop_ro("tiles", [](const TerrainSource& s) { return s.index().tiles(); })
        .def(
            "grid",
            [](const TerrainSource& s, int64_t t) {
                torch::Tensor g;
                {
                    nb::gil_scoped_release release;
                    g = s.grid(t);
                }
                return tensor_or_none(g);
            },
            nb::arg("tile"),
            "Tile `tile` (its position in `tiles`) whole, (rows, cols) float32: the stored cells of grids, the "
            "decoded or evaluated cells of a field.")
        .def("describe", &TerrainSource::describe);

    nb::class_<TerrainGrids, TerrainSource>(m, "TerrainGrids",
                                            "Elevation grids on disk: a tiles.json and its float32 .npy tiles.")
        .def(nb::init<std::string, int64_t>(), nb::arg("dir"), nb::arg("cache_tiles") = 16);

    nb::class_<TerrainField, TerrainSource>(m, "TerrainField",
                                            "An elevation model compressed into one neural field per tile.")
        .def(
            "__init__",
            [](TerrainField* self, const std::string& path, const std::string& device, int64_t cache_tiles) {
                new (self) TerrainField(path, parse_device(device), cache_tiles);
            },
            nb::arg("path"), nb::arg("device") = "cpu", nb::arg("cache_tiles") = 64)
        .def_prop_ro("config", [](const TerrainField& f) { return f.config(); })
        .def_prop_ro("metrics",
                     [](const TerrainField& f) {
                         nb::list out;
                         for (const auto& m : f.metrics()) out.append(json_to_python(terrain_tile_metrics_to_json(m)));
                         return out;
                     },
                     "Per tile, how closely its field reproduced the grid when it was fitted.");

    nb::class_<TerrainCodedField, TerrainSource>(
        m, "TerrainCodedField", "An elevation model coded tile by tile with a bounded error per cell.")
        .def(nb::init<std::string, int64_t>(), nb::arg("path"), nb::arg("cache_blocks") = 4096)
        .def_prop_ro("config", [](const TerrainCodedField& f) { return f.config(); })
        .def_prop_ro("metrics", [](const TerrainCodedField& f) {
            nb::list out;
            for (const auto& m : f.metrics()) out.append(json_to_python(terrain_tile_metrics_to_json(m)));
            return out;
        });

    m.def(
        "open_terrain_field",
        [](const std::string& path, const std::string& device) { return open_terrain_field(path, parse_device(device)); },
        nb::arg("path"), nb::arg("device") = "cpu",
        "A terrain field file of either format (neural or coded), as a TerrainSource.");

    m.def(
        "encode_terrain_grids",
        [](const TerrainGrids& grids, const TerrainCodecConfig& config, const std::string& path,
           const std::vector<std::string>& tiles) {
            std::vector<std::pair<std::string, TerrainTileMetrics>> metrics;
            {
                nb::gil_scoped_release release;
                metrics = encode_terrain_grids(grids, config, path, tiles);
            }
            nb::dict out;
            for (const auto& [name, mt] : metrics) out[name.c_str()] = json_to_python(terrain_tile_metrics_to_json(mt));
            return out;
        },
        nb::arg("grids"), nb::arg("config") = TerrainCodecConfig{}, nb::arg("path"),
        nb::arg("tiles") = std::vector<std::string>{},
        "Code the grids' tiles into one coded terrain field file; returns each tile's metrics.");

    m.def(
        "terrain_descriptors",
        [](const TerrainSource& source, const nb::object& lat, const nb::object& lon, int64_t chunk) {
            const torch::Tensor la = unpack_required_tensor(lat, "lat");
            const torch::Tensor lo = unpack_required_tensor(lon, "lon");
            nb::gil_scoped_release release;
            return terrain_descriptors(source, la, lo, chunk);
        },
        nb::arg("source"), nb::arg("lat"), nb::arg("lon"), nb::arg("chunk") = 4096,
        "The elevation at each coordinate, the mean elevation over the 1 km square around it, the slope "
        "of its cell and the inclination of the 1 km square (degrees).");

    m.def("read_npy_grid", [](const std::string& path) { return tensor_or_none(read_npy_grid(path)); },
          nb::arg("path"), "A float32 2-D .npy grid as a tensor.");
    m.def("write_npy_grid",
          [](const std::string& path, const nb::object& grid) { write_npy_grid(path, unpack_required_tensor(grid, "grid")); },
          nb::arg("path"), nb::arg("grid"), "Write a 2-D tensor as a float32 .npy grid.");

    m.def(
        "fit_terrain_tile",
        [](const nb::object& grid, const TerrainTile& tile, const TerrainFieldConfig& config, const std::string& path) {
            const torch::Tensor g = unpack_required_tensor(grid, "grid");
            TerrainTileMetrics metrics;
            {
                nb::gil_scoped_release release;
                metrics = fit_terrain_tile(g, tile, config, path);
            }
            return json_to_python(terrain_tile_metrics_to_json(metrics));
        },
        nb::arg("grid"), nb::arg("tile"), nb::arg("config") = TerrainFieldConfig{}, nb::arg("path"),
        "Fit one tile's neural field and write it as a tile pack; returns its metrics.");
    m.def(
        "assemble_terrain_field",
        [](const std::vector<std::string>& tiles, const std::string& path) {
            nb::gil_scoped_release release;
            assemble_terrain_field(tiles, path);
        },
        nb::arg("tiles"), nb::arg("path"), "Merge tile packs into one terrain field file.");

    nb::class_<TerrainCorrectionResult>(m, "TerrainCorrectionResult")
        .def_prop_ro("corrected", [](const TerrainCorrectionResult& r) { return tensor_or_none(r.corrected); },
                     "(n,) the corrected value; the raw prediction where not applied.")
        .def_prop_ro("weights", [](const TerrainCorrectionResult& r) { return tensor_or_none(r.weights); },
                     "(n, 3) the weight of the raw prediction, the descriptor and the 1 km descriptor.")
        .def_prop_ro("offset", [](const TerrainCorrectionResult& r) { return tensor_or_none(r.offset); })
        .def_prop_ro("applied", [](const TerrainCorrectionResult& r) { return tensor_or_none(r.applied); });

    nb::class_<TerrainCorrection>(m, "TerrainCorrection",
                                  "A fitted terrain correction: the prediction combined with an elevation model.")
        .def_static(
            "load",
            [](const std::string& path, const std::string& device) {
                return TerrainCorrection::load(path, parse_device(device));
            },
            nb::arg("path"), nb::arg("device") = "cpu")
        .def("save", &TerrainCorrection::save, nb::arg("path"))
        .def(
            "apply",
            [](const TerrainCorrection& c, const nb::object& prediction, const nb::object& dem,
               const nb::object& dem_1km, const nb::object& uncertainty_m, const nb::object& spread,
               const nb::object& recognised_share, const nb::object& n_species) {
                const TerrainCorrectionInputs in =
                    inputs_from(prediction, dem, dem_1km, uncertainty_m, spread, recognised_share, n_species);
                nb::gil_scoped_release release;
                return c.apply(in);
            },
            nb::arg("prediction"), nb::arg("dem"), nb::arg("dem_1km") = nb::none(),
            nb::arg("uncertainty_m") = nb::none(), nb::arg("spread") = nb::none(),
            nb::arg("recognised_share") = nb::none(), nb::arg("n_species") = nb::none())
        .def_prop_ro("target", &TerrainCorrection::target)
        .def_prop_ro("descriptor", &TerrainCorrection::descriptor)
        .def_prop_ro("config", [](const TerrainCorrection& c) { return c.config(); })
        .def_prop_ro("report",
                     [](const TerrainCorrection& c) { return json_to_python(terrain_correction_report_to_json(c.report())); })
        .def_prop_ro_static("weight_names", [](nb::handle) { return terrain_weight_names(); });

    m.def(
        "fit_terrain_correction",
        [](const std::string& target, TerrainDescriptor descriptor, const nb::object& prediction,
           const nb::object& dem, const nb::object& truth, const nb::object& dem_1km, const nb::object& uncertainty_m,
           const nb::object& spread, const nb::object& recognised_share, const nb::object& n_species,
           const TerrainCorrectionConfig& config) {
            const TerrainCorrectionInputs in =
                inputs_from(prediction, dem, dem_1km, uncertainty_m, spread, recognised_share, n_species);
            const torch::Tensor t = unpack_required_tensor(truth, "truth");
            nb::gil_scoped_release release;
            return fit_terrain_correction(target, descriptor, in, t, config);
        },
        nb::arg("target"), nb::arg("descriptor"), nb::arg("prediction"), nb::arg("dem"), nb::arg("truth"),
        nb::arg("dem_1km") = nb::none(), nb::arg("uncertainty_m") = nb::none(), nb::arg("spread") = nb::none(),
        nb::arg("recognised_share") = nb::none(), nb::arg("n_species") = nb::none(),
        nb::arg("config") = TerrainCorrectionConfig{},
        "Fit the gated terrain correction of one target on plots whose recorded value is `truth` (NaN: "
        "unrecorded).");
}
