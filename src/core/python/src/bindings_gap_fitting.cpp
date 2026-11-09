#include "bindings_common.hpp"

#include "resolve/config_registry.hpp"
#include "resolve/gap_fitting.hpp"
#include "resolve/json.hpp"
#include "resolve/retrieval.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

using namespace resolve;

namespace {

// A JSON value (a gap-fitting report) as the Python dict / list / scalars it
// spells, NaN read back as None.
nb::object json_to_python(const json::Value& value) {
    return nb::module_::import_("json").attr("loads")(json::dump(value, 0));
}

nb::dict layout_to_python(const GapFoldLayout& layout) {
    nb::dict out;
    out["n_folds"] = layout.n_folds;
    out["block_km"] = layout.block_km;
    out["block_years"] = layout.block_years;
    out["buffer_km"] = layout.buffer_km;
    out["buffer_years"] = layout.buffer_years;
    out["injected"] = layout.injected;
    nb::dict mismatch;
    for (const auto& [axis, w] : layout.mismatch) mismatch[axis.c_str()] = w;
    out["mismatch"] = mismatch;
    out["total_mismatch"] = layout.total_mismatch;
    out["n_layouts"] = layout.n_layouts;
    out["n_matching_plots"] = layout.n_matching_plots;
    return out;
}

GapPlots plots_from(const nb::object& species, const nb::object& coordinates, const nb::object& years) {
    GapPlots plots;
    plots.species = unpack_required_tensor(species, "species");
    plots.coordinates = unpack_optional_tensor(coordinates);
    plots.years = unpack_optional_tensor(years);
    return plots;
}

std::vector<std::string> method_labels(const torch::Tensor& codes) {
    std::vector<std::string> out;
    const torch::Tensor c = codes.to(torch::kCPU, torch::kInt64).contiguous();
    out.reserve(static_cast<std::size_t>(c.numel()));
    const int64_t* data = c.data_ptr<int64_t>();
    for (int64_t i = 0; i < c.numel(); ++i) out.push_back(gap_method_label(data[i]));
    return out;
}

}  // namespace

void register_gap_fitting(nb::module_& m) {
    {
        using Cfg = GapFittingConfig;
        auto cls = nb::class_<Cfg>(m, "GapFittingConfig",
                                   "Gap fitting settings (fit_gaps, SuitePredictor.predict).");
        cls.def(nb::init<>());
        RESOLVE_GAP_FITTING_CONFIG_FIELDS(RESOLVE_BIND_FIELD)
        cls.def("validate", &Cfg::validate);
    }

    nb::class_<GapTarget>(m, "GapTarget", "The target a gap fit is for: its name, scale and period.")
        .def(
            "__init__",
            [](GapTarget* self, std::string name, GapScale scale, double period) {
                new (self) GapTarget{std::move(name), scale, period};
            },
            nb::arg("name") = "", nb::arg("scale") = GapScale::Linear, nb::arg("period") = 360.0)
        .def_rw("name", &GapTarget::name)
        .def_rw("scale", &GapTarget::scale)
        .def_rw("period", &GapTarget::period);

    nb::class_<GapFit>(m, "GapFit", "A gap fit: fitted values, the method per plot, the report.")
        .def_prop_ro("fitted", [](const GapFit& f) { return tensor_or_none(f.fitted); },
                     "Per plot in the target's units ((n,) or (n, 2) latitude / longitude): the "
                     "recorded value of a complete plot, the chosen method's value for a gap.")
        .def_prop_ro("method_codes", [](const GapFit& f) { return tensor_or_none(f.method); },
                     "(n,) int64: -1 for a complete plot, the GapMethod integer for a gap.")
        .def_prop_ro("method", [](const GapFit& f) { return method_labels(f.method); },
                     "Per plot: 'recorded' for a complete plot, the method's name for a gap.")
        .def_prop_ro("report", [](const GapFit& f) { return json_to_python(gap_fit_report_to_json(f.report)); },
                     "What was chosen and why: n complete / n gaps, the gaps' distance per axis, "
                     "the fold layout, the k chosen per neighbour kind, and per group of gaps "
                     "(located / unlocated) every candidate's cross-validated error, the error "
                     "estimated for the whole procedure and the chosen method's coefficients.");

    m.def(
        "fit_gaps",
        [](const GapTarget& target, const nb::object& raw, const nb::object& recorded,
           const nb::object& species, const nb::object& coordinates, const nb::object& years,
           const GapFittingConfig& config, const nb::object& folds) {
            const torch::Tensor raw_values = unpack_required_tensor(raw, "raw");
            const torch::Tensor recorded_values = unpack_required_tensor(recorded, "recorded");
            const GapPlots plots = plots_from(species, coordinates, years);
            const torch::Tensor injected = unpack_optional_tensor(folds);
            GapFit fit;
            {
                nb::gil_scoped_release release;
                fit = fit_gaps(target, raw_values, recorded_values, plots, config,
                               injected.defined() ? &injected : nullptr);
            }
            return fit;
        },
        nb::arg("target"), nb::arg("raw"), nb::arg("recorded"), nb::arg("species"),
        nb::arg("coordinates") = nb::none(), nb::arg("years") = nb::none(),
        nb::arg("config") = GapFittingConfig{}, nb::arg("folds") = nb::none(),
        "Fit the gaps of one target from the dataset's own complete plots.\n"
        "\n"
        "raw, recorded: the model's prediction and the recorded value per plot in\n"
        "the target's units, (n,) or (n, 2) latitude / longitude for a geographic\n"
        "target; a NaN recorded value marks a gap.\n"
        "species: (n, n_species) composition, dense or sparse CSR; an entry other\n"
        "than 0 marks a species present.\n"
        "coordinates: (n, 2) latitude / longitude, NaN where unknown; years: (n,)\n"
        "survey years, NaN where unknown.\n"
        "folds: (n_complete,) int64, the fold of each complete plot in plot order\n"
        "(numbered from 0), replacing the matched layout (reproduces another\n"
        "implementation's layout).");

    m.def(
        "gap_folds",
        [](const GapTarget& target, const nb::object& recorded, const nb::object& species,
           const nb::object& coordinates, const nb::object& years, const GapFittingConfig& config) {
            const torch::Tensor recorded_values = unpack_required_tensor(recorded, "recorded");
            const GapPlots plots = plots_from(species, coordinates, years);
            GapFolds found;
            {
                nb::gil_scoped_release release;
                found = gap_folds(target, recorded_values, plots, config);
            }
            nb::dict out;
            out["complete"] = tensor_or_none(found.complete);
            out["gaps"] = tensor_or_none(found.gaps);
            out["folds"] = tensor_or_none(found.folds);
            out["layout"] = layout_to_python(found.layout);
            return out;
        },
        nb::arg("target"), nb::arg("recorded"), nb::arg("species"),
        nb::arg("coordinates") = nb::none(), nb::arg("years") = nb::none(),
        nb::arg("config") = GapFittingConfig{},
        "The fold layout fit_gaps matches to the gaps: a dict of complete / gaps\n"
        "(plot rows), folds ((n_complete,) fold of each complete plot) and layout\n"
        "(n_folds, block_km, block_years, mismatch per axis, total_mismatch,\n"
        "n_layouts, n_matching_plots).");

    nb::class_<GapColumns>(m, "GapColumns", "The columns of a dataset's tables gap fitting reads.")
        .def(nb::init<>())
        .def_rw("plot_id", &GapColumns::plot_id)
        .def_rw("species_id", &GapColumns::species_id)
        .def_rw("latitude", &GapColumns::latitude)
        .def_rw("longitude", &GapColumns::longitude)
        .def_rw("survey_year", &GapColumns::survey_year)
        .def_rw("recorded", &GapColumns::recorded);

    nb::class_<GapTables>(m, "GapTables", "What gap fitting reads about a dataset's plots.")
        .def_prop_ro("species", [](const GapTables& t) { return tensor_or_none(t.plots.species); })
        .def_prop_ro("coordinates", [](const GapTables& t) { return tensor_or_none(t.plots.coordinates); })
        .def_prop_ro("years", [](const GapTables& t) { return tensor_or_none(t.plots.years); })
        .def_prop_ro("recorded", [](const GapTables& t) {
            nb::dict out;
            for (const auto& [name, values] : t.recorded) out[name.c_str()] = tensor_or_none(values);
            return out;
        });

    m.def(
        "read_gap_tables_csv",
        [](const std::string& species_path, const std::string& header_path, const GapColumns& columns,
           const std::vector<std::string>& plot_ids) {
            nb::gil_scoped_release release;
            std::optional<TableSource> header;
            if (!header_path.empty()) header = TableSource::csv(header_path);
            return read_gap_tables(header, TableSource::csv(species_path), columns, plot_ids);
        },
        nb::arg("species_path"), nb::arg("header_path"), nb::arg("columns"), nb::arg("plot_ids"),
        "read_gap_tables over CSV files (header_path '' reads plot-level columns from\n"
        "each plot's first species row).");

    m.def(
        "read_gap_tables_columns",
        [](const std::vector<std::string>& species_names,
           const std::vector<std::vector<std::string>>& species_columns,
           const std::optional<std::vector<std::string>>& header_names,
           const std::optional<std::vector<std::vector<std::string>>>& header_columns,
           const GapColumns& columns, const std::vector<std::string>& plot_ids) {
            nb::gil_scoped_release release;
            const ColumnTable species(species_names, species_columns);
            std::optional<ColumnTable> header_table;
            std::optional<TableSource> header;
            if (header_names) {
                header_table.emplace(*header_names, header_columns.value());
                header = TableSource::table(*header_table);
            }
            return read_gap_tables(header, TableSource::table(species), columns, plot_ids);
        },
        nb::arg("species_names"), nb::arg("species_columns"), nb::arg("header_names").none(),
        nb::arg("header_columns").none(), nb::arg("columns"), nb::arg("plot_ids"),
        "read_gap_tables over in-memory string columns (the pandas path).");

    nb::class_<GapFitRequest>(m, "GapFitRequest",
                              "One target of a single model to fit: its GapTarget and the output "
                              "head(s) it reads.")
        .def(
            "__init__",
            [](GapFitRequest* self, GapTarget target, std::vector<std::string> outputs) {
                new (self) GapFitRequest{std::move(target), std::move(outputs)};
            },
            nb::arg("target"), nb::arg("outputs"))
        .def_rw("target", &GapFitRequest::target)
        .def_rw("outputs", &GapFitRequest::outputs);

    m.def(
        "fit_prediction_gaps_tables",
        [](const ResolvePredictions& predictions, const GapTables& tables,
           const std::vector<GapFitRequest>& requests, const GapFittingConfig& config) {
            nb::gil_scoped_release release;
            return fit_gaps(predictions, tables, requests, config);
        },
        nb::arg("predictions"), nb::arg("tables"), nb::arg("requests"),
        nb::arg("config") = GapFittingConfig{},
        "Fit the gaps of every request from one model's predictions and the tables\n"
        "read for its plots (in predictions.plot_ids order).");

    m.def(
        "nearest_by_great_circle",
        [](const nb::object& queries, const nb::object& reference, int64_t k, bool exclude_self) {
            const at::Tensor query_rows = unpack_required_tensor(queries, "queries");
            const at::Tensor reference_rows = unpack_required_tensor(reference, "reference");
            PlaceNeighbours found;
            {
                nb::gil_scoped_release release;
                found = nearest_by_great_circle(query_rows, reference_rows, k, exclude_self);
            }
            return nb::make_tuple(tensor_or_none(found.indices), tensor_or_none(found.km));
        },
        nb::arg("queries"), nb::arg("reference"), nb::arg("k"), nb::arg("exclude_self") = false,
        "The k rows of `reference` nearest to each row of `queries` by great-circle\n"
        "distance: (rows, 2) latitude / longitude in degrees; an exact blocked\n"
        "search. Returns (indices, km), -1 / NaN where a row has fewer than k (a\n"
        "position without coordinates has none).");
}
