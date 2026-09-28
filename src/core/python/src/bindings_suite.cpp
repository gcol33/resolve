// Model suites (suite.hpp): the manifest, the suite predictor, its results, the
// combination rules and the species recognition report.

#include "bindings_common.hpp"
#include "resolve/json.hpp"
#include "resolve/sha256.hpp"
#include "resolve/suite.hpp"

#include <cmath>

namespace {

// 2^53: integral JSON numbers below it come back to Python as int.
constexpr double kExactIntegerLimit = 9007199254740992.0;

nb::object tensor_or_none(const torch::Tensor& t) {
    if (!t.defined()) return nb::none();
    auto cpu = t.detach().cpu().contiguous();
    return nb::steal(THPVariable_Wrap(cpu));
}

// JSON <-> Python: objects to dicts (member order kept), arrays to lists, and
// an integral number to int so a seed or an epoch count reads as one.
nb::object json_to_py(const resolve::json::Value& v) {
    using Kind = resolve::json::Value::Kind;
    switch (v.kind()) {
        case Kind::Null: return nb::none();
        case Kind::Bool: return nb::bool_(v.as_bool());
        case Kind::Number: {
            const double n = v.as_number();
            if (n == std::floor(n) && std::fabs(n) < kExactIntegerLimit) {
                return nb::int_(static_cast<int64_t>(n));
            }
            return nb::float_(n);
        }
        case Kind::String: return nb::str(v.as_string().c_str());
        case Kind::Array: {
            nb::list out;
            for (const auto& item : v.items()) out.append(json_to_py(item));
            return out;
        }
        case Kind::Object: {
            nb::dict out;
            for (const auto& [key, value] : v.members()) out[key.c_str()] = json_to_py(value);
            return out;
        }
    }
    return nb::none();
}

resolve::json::Value py_to_json(nb::handle h) {
    using resolve::json::Value;
    if (h.is_none()) return Value();
    if (nb::isinstance<nb::bool_>(h)) return Value(nb::cast<bool>(h));
    if (nb::isinstance<nb::int_>(h)) return Value(nb::cast<int64_t>(h));
    if (nb::isinstance<nb::float_>(h)) return Value(nb::cast<double>(h));
    if (nb::isinstance<nb::str>(h)) return Value(nb::cast<std::string>(h));
    if (nb::isinstance<nb::dict>(h)) {
        Value out = Value::object();
        for (auto [key, value] : nb::borrow<nb::dict>(h)) {
            if (!nb::isinstance<nb::str>(key)) {
                throw std::invalid_argument("a JSON object's keys must be strings");
            }
            out.set(nb::cast<std::string>(key), py_to_json(value));
        }
        return out;
    }
    if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
        Value out = Value::array();
        for (auto item : h) out.push_back(py_to_json(item));
        return out;
    }
    throw std::invalid_argument("value has no JSON form: " +
                                nb::cast<std::string>(nb::str(h.type())));
}

resolve::json::Value py_to_json_object(nb::handle h, const char* what) {
    resolve::json::Value v = py_to_json(h);
    if (!v.is_object()) throw std::invalid_argument(std::string(what) + " must be a dict");
    return v;
}

torch::Device parse_device(const std::string& device) {
    if (device == "cpu") return torch::kCPU;
    if (device == "cuda") return torch::kCUDA;
    throw std::invalid_argument("device must be 'cpu' or 'cuda', not '" + device + "'");
}

}  // namespace

void register_suite(nb::module_& m) {
    using namespace resolve;

    // --- manifest -----------------------------------------------------------

    nb::class_<SuiteInputContract>(m, "SuiteInputContract",
                                   "The columns a suite reads and how their raw values are read.")
        .def(nb::init<>())
        .def_rw("plot_id", &SuiteInputContract::plot_id)
        .def_rw("species", &SuiteInputContract::species)
        .def_rw("abundance", &SuiteInputContract::abundance)
        .def_rw("genus", &SuiteInputContract::genus)
        .def_rw("family", &SuiteInputContract::family)
        .def_rw("latitude", &SuiteInputContract::latitude)
        .def_rw("longitude", &SuiteInputContract::longitude)
        .def_rw("covariates", &SuiteInputContract::covariates)
        .def_rw("categoricals", &SuiteInputContract::categoricals)
        .def_rw("abundance_units", &SuiteInputContract::abundance_units)
        .def_rw("zero_abundance_as", &SuiteInputContract::zero_abundance_as)
        .def_rw("notes", &SuiteInputContract::notes)
        .def("reads_coordinates", &SuiteInputContract::reads_coordinates)
        .def("needs_header", &SuiteInputContract::needs_header);

    nb::class_<SuiteMember>(m, "SuiteMember", "One weight file of a suite target.")
        .def("__init__",
             [](SuiteMember* self, int64_t seed, const std::string& file,
                const std::string& sha256, int64_t bytes) {
                 new (self) SuiteMember{seed, file, sha256, bytes};
             },
             nb::arg("seed"), nb::arg("file"), nb::arg("sha256") = "", nb::arg("bytes") = 0)
        .def_rw("seed", &SuiteMember::seed)
        .def_rw("file", &SuiteMember::file)
        .def_rw("sha256", &SuiteMember::sha256)
        .def_rw("bytes", &SuiteMember::bytes);

    nb::class_<SuiteTarget>(m, "SuiteTarget", "One released target of a suite.")
        .def(nb::init<>())
        .def_rw("name", &SuiteTarget::name)
        .def_rw("task", &SuiteTarget::task)
        .def_rw("combine", &SuiteTarget::combine)
        .def_rw("outputs", &SuiteTarget::outputs)
        .def_rw("units", &SuiteTarget::units)
        .def_rw("period", &SuiteTarget::period)
        .def_rw("status", &SuiteTarget::status)
        .def_rw("limit", &SuiteTarget::limit)
        .def_rw("members", &SuiteTarget::members)
        .def_prop_rw(
            "training", [](const SuiteTarget& t) { return json_to_py(t.training); },
            [](SuiteTarget& t, nb::handle v) { t.training = py_to_json_object(v, "training"); },
            "Free-form training record (a dict).")
        .def_prop_rw(
            "validation", [](const SuiteTarget& t) { return json_to_py(t.validation); },
            [](SuiteTarget& t, nb::handle v) { t.validation = py_to_json_object(v, "validation"); },
            "Validation metrics by regime (a dict of dicts).");

    nb::class_<SuiteManifest>(m, "SuiteManifest",
                              "A suite's manifest: input contract, targets, members, provenance.")
        .def(nb::init<>())
        .def_rw("name", &SuiteManifest::name)
        .def_rw("description", &SuiteManifest::description)
        .def_rw("licence", &SuiteManifest::licence)
        .def_rw("engine_version", &SuiteManifest::engine_version)
        .def_rw("taxonomy", &SuiteManifest::taxonomy)
        .def_rw("training_scope", &SuiteManifest::training_scope)
        .def_rw("inputs", &SuiteManifest::inputs)
        .def_rw("targets", &SuiteManifest::targets,
                "The targets. A copy: assign a whole list to change them.")
        .def("validate", &SuiteManifest::validate,
             "Raise listing every structural problem, or return None.")
        .def("to_json", [](const SuiteManifest& s, int indent) {
                 return json::dump(s.to_json(), indent);
             }, nb::arg("indent") = 2)
        .def_static("from_json", [](const std::string& text) {
                        return SuiteManifest::from_json(json::parse(text));
                    }, nb::arg("text"),
                    "Parse a manifest from JSON text (not validated; call validate()).")
        .def("target", &SuiteManifest::target, nb::arg("name"))
        .def_static("read", &SuiteManifest::read, nb::arg("dir"),
                    "Read and validate <dir>/manifest.json.")
        .def("write", &SuiteManifest::write, nb::arg("dir"),
             "Validate, then write <dir>/manifest.json.")
        .def("seal", &SuiteManifest::seal, nb::arg("dir"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Fill every member's sha256 and byte count from the files under dir.")
        .def("verify", &SuiteManifest::verify, nb::arg("dir"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Problems with the member files under dir; empty when intact.");

    m.def("sha256_file", &sha256_file, nb::arg("path"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Lower-case hex SHA-256 of a file's bytes.");

    // --- recognition and combination rules -----------------------------------

    nb::class_<SpeciesRecognition>(m, "SpeciesRecognition",
                                   "How much of each plot a species vocabulary recognises.")
        .def(nb::init<>())
        .def_prop_ro("n_species", [](const SpeciesRecognition& r) { return tensor_or_none(r.n_species); })
        .def_prop_ro("n_recognised", [](const SpeciesRecognition& r) { return tensor_or_none(r.n_recognised); })
        .def_prop_ro("count_share", [](const SpeciesRecognition& r) { return tensor_or_none(r.count_share); })
        .def_prop_ro("abundance_share",
                     [](const SpeciesRecognition& r) { return tensor_or_none(r.abundance_share); });

    m.def("compute_species_recognition", &compute_species_recognition,
          nb::arg("records"), nb::arg("plot_ids"), nb::arg("vocab"),
          "Per-plot distinct species, how many the vocab recognises, and the recognised "
          "share by count and by abundance (NaN where undefined).");

    nb::class_<CombinedPrediction>(m, "CombinedPrediction")
        .def_prop_ro("value", [](const CombinedPrediction& c) { return tensor_or_none(c.value); })
        .def_prop_ro("agreement", [](const CombinedPrediction& c) { return tensor_or_none(c.agreement); })
        .def_prop_ro("dispersion", [](const CombinedPrediction& c) { return tensor_or_none(c.dispersion); });

    m.def("combine_vote", [](nb::object codes, int64_t n_classes, nb::object probabilities) {
              return combine_vote(unpack_required_tensor(codes, "member_codes"), n_classes,
                                  unpack_optional_tensor(probabilities));
          }, nb::arg("member_codes"), nb::arg("n_classes"),
          nb::arg("mean_probabilities") = nb::none(),
          "Majority class over a (n_members, n_plots) stack. A tie goes to the class with "
          "the higher mean probability when mean_probabilities (n_plots, n_classes) is "
          "given, else to the lowest code.");
    m.def("combine_mean", [](nb::object values) {
              return combine_mean(unpack_required_tensor(values, "member_values"));
          }, nb::arg("member_values"),
          "Mean and standard deviation over a (n_members, n_plots) stack.");
    m.def("combine_circular", [](nb::object bearings, double period) {
              return combine_circular(unpack_required_tensor(bearings, "member_bearings"), period);
          }, nb::arg("member_bearings"), nb::arg("period"),
          "Circular mean and circular standard deviation over a (n_members, n_plots) stack.");
    m.def("bearing_from_components", [](nb::object sine, nb::object cosine, double period) {
              return tensor_or_none(bearing_from_components(
                  unpack_required_tensor(sine, "sine"), unpack_required_tensor(cosine, "cosine"),
                  period));
          }, nb::arg("sine"), nb::arg("cosine"), nb::arg("period"),
          "A bearing in [0, period) from its sine and cosine.");

    // --- prediction ------------------------------------------------------------

    nb::class_<SuiteColumns>(m, "SuiteColumns",
                             "Renames the manifest's input columns; None keeps the manifest's.")
        .def(nb::init<>())
        .def_rw("plot_id", &SuiteColumns::plot_id)
        .def_rw("species_id", &SuiteColumns::species_id)
        .def_rw("abundance", &SuiteColumns::abundance)
        .def_rw("genus", &SuiteColumns::genus)
        .def_rw("family", &SuiteColumns::family)
        .def_rw("latitude", &SuiteColumns::latitude)
        .def_rw("longitude", &SuiteColumns::longitude)
        .def_rw("covariates", &SuiteColumns::covariates)
        .def_rw("categoricals", &SuiteColumns::categoricals);

    nb::class_<SuiteTargetPrediction>(m, "SuiteTargetPrediction")
        .def_ro("name", &SuiteTargetPrediction::name)
        .def_ro("task", &SuiteTargetPrediction::task)
        .def_ro("combine", &SuiteTargetPrediction::combine)
        .def_ro("status", &SuiteTargetPrediction::status)
        .def_ro("limit", &SuiteTargetPrediction::limit)
        .def_ro("units", &SuiteTargetPrediction::units)
        .def_ro("class_names", &SuiteTargetPrediction::class_names)
        .def_ro("member_seeds", &SuiteTargetPrediction::member_seeds)
        .def_prop_ro("value", [](const SuiteTargetPrediction& t) { return tensor_or_none(t.value); })
        .def_prop_ro("agreement", [](const SuiteTargetPrediction& t) { return tensor_or_none(t.agreement); })
        .def_prop_ro("probabilities",
                     [](const SuiteTargetPrediction& t) { return tensor_or_none(t.probabilities); })
        .def_prop_ro("dispersion", [](const SuiteTargetPrediction& t) { return tensor_or_none(t.dispersion); })
        .def_prop_ro("members", [](const SuiteTargetPrediction& t) { return tensor_or_none(t.members); })
        .def_ro("recognition", &SuiteTargetPrediction::recognition);

    nb::class_<SuitePredictions>(m, "SuitePredictions")
        .def_ro("plot_ids", &SuitePredictions::plot_ids)
        .def_ro("targets", &SuitePredictions::targets)
        .def("target", &SuitePredictions::target, nb::arg("name"), nb::rv_policy::copy);

    nb::class_<SuitePredictor>(m, "SuitePredictor",
                               "A model suite loaded from its directory, scored as one model.")
        .def_static("load",
                    [](const std::string& dir, const std::string& device, float vram_fraction,
                       bool verify, std::optional<std::vector<std::string>> targets) {
                        SuiteLoadOptions options;
                        options.device = parse_device(device);
                        options.vram_fraction = vram_fraction;
                        options.verify_checksums = verify;
                        if (targets) options.targets = *targets;
                        nb::gil_scoped_release nogil;
                        return SuitePredictor::load(dir, options);
                    },
                    nb::arg("dir"), nb::arg("device") = "cpu", nb::arg("vram_fraction") = 1.0f,
                    nb::arg("verify") = true, nb::arg("targets") = nb::none(),
                    "Read the manifest, verify every member's checksum, load and check the members.")
        .def_prop_ro("manifest", [](const SuitePredictor& s) { return s.manifest(); })
        .def_prop_ro("directory", &SuitePredictor::directory)
        .def_prop_ro("target_names", &SuitePredictor::target_names)
        .def_prop_ro("n_encodings", &SuitePredictor::n_encodings)
        .def("predict_csv",
             [](SuitePredictor& s, const std::string& species_path, const std::string& header_path,
                const SuiteColumns& columns, int64_t batch_size, bool keep_members) {
                 SuitePredictOptions options;
                 options.columns = columns;
                 options.batch_size = batch_size;
                 options.keep_members = keep_members;
                 nb::gil_scoped_release nogil;
                 return s.predict(SuiteInput::csv(header_path, species_path), options);
             },
             nb::arg("species_path"), nb::arg("header_path") = "",
             nb::arg("columns") = SuiteColumns{}, nb::arg("batch_size") = 4096,
             nb::arg("keep_members") = false)
        .def("predict_columns",
             [](SuitePredictor& s, std::vector<std::string> species_names,
                std::vector<std::vector<std::string>> species_columns,
                std::optional<std::vector<std::string>> header_names,
                std::optional<std::vector<std::vector<std::string>>> header_columns,
                const SuiteColumns& columns, int64_t batch_size, bool keep_members) {
                 if (header_names.has_value() != header_columns.has_value()) {
                     throw std::invalid_argument("pass header_names and header_columns together");
                 }
                 SuitePredictOptions options;
                 options.columns = columns;
                 options.batch_size = batch_size;
                 options.keep_members = keep_members;
                 nb::gil_scoped_release nogil;
                 const ColumnTable species(std::move(species_names), std::move(species_columns));
                 std::optional<ColumnTable> header;
                 if (header_names) header.emplace(std::move(*header_names), std::move(*header_columns));
                 return s.predict(SuiteInput::tables(header ? &*header : nullptr, species), options);
             },
             nb::arg("species_names"), nb::arg("species_columns"),
             nb::arg("header_names") = nb::none(), nb::arg("header_columns") = nb::none(),
             nb::arg("columns") = SuiteColumns{}, nb::arg("batch_size") = 4096,
             nb::arg("keep_members") = false,
             "Low-level entry point over string columns; SuitePredictor.predict wraps it "
             "for pandas DataFrames.");
}
