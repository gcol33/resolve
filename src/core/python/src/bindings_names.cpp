// Python bindings for name resolution (name_resolution.hpp): the release's
// name table, the resolver, the setting a load or a suite prediction takes,
// and the report of the names that were not exact.

#include "bindings_common.hpp"
#include "resolve/enum_names.hpp"
#include "resolve/name_resolution.hpp"

#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <string>
#include <vector>

namespace {

using resolve::NameResolver;
using resolve::NameTable;
using resolve::ResolvedName;
using resolve::SpeciesNameReport;
using resolve::SpeciesNames;

// The bindings hold a table as shared_ptr<NameTable> and bind no method that
// changes it; the engine reads it as shared_ptr<const NameTable>.
std::shared_ptr<NameTable> mutable_handle(std::shared_ptr<const NameTable> table) {
    return std::const_pointer_cast<NameTable>(std::move(table));
}

void init_species_names(SpeciesNames* self, std::shared_ptr<NameTable> table, int max_edit_distance) {
    new (self) SpeciesNames{std::move(table), resolve::NameResolverOptions{max_edit_distance}};
}

}  // namespace

void register_names(nb::module_& m) {
    nb::class_<NameTable>(m, "NameTable",
                          "The release's name table: name -> the vocabulary token it stands "
                          "for, and the source that linked them (\"eva\" or a backbone).")
        .def_static("read",
                    [](const std::string& path) {
                        return std::make_shared<NameTable>(NameTable::read(path));
                    },
                    nb::arg("path"), nb::call_guard<nb::gil_scoped_release>(),
                    "Read a names.json file ({\"source\": ..., \"aliases\": {name: [token, "
                    "source]}}) and record its SHA-256.")
        .def_static("open",
                    [](const std::string& path) { return mutable_handle(NameTable::open(path)); },
                    nb::arg("path"), nb::call_guard<nb::gil_scoped_release>(),
                    "read(path), parsed once per process while the file keeps its size and "
                    "modification time.")
        .def_static("from_columns",
                    [](const std::vector<std::string>& names, const std::vector<std::string>& tokens,
                       const std::vector<std::string>& sources) {
                        return std::make_shared<NameTable>(
                            NameTable::from_columns(names, tokens, sources));
                    },
                    nb::arg("names"), nb::arg("tokens"), nb::arg("sources"),
                    "A table from parallel lists.")
        .def("find",
             [](const NameTable& t, const std::string& name) -> nb::object {
                 const NameTable::Alias* alias = t.find(name);
                 if (alias == nullptr) return nb::none();
                 return nb::make_tuple(alias->token, alias->source);
             },
             nb::arg("name"), "(token, source) for `name`, or None.")
        .def("__len__", &NameTable::size)
        .def("__contains__", [](const NameTable& t, const std::string& name) {
            return t.find(name) != nullptr;
        })
        .def_prop_ro("description", &NameTable::description)
        .def_prop_ro("sha256", &NameTable::sha256);

    nb::class_<SpeciesNames>(m, "SpeciesNames",
                             "How species names are resolved before encoding: the name table "
                             "(None: the vocabulary alone) and the edits a misspelling may be "
                             "away (0 turns the spelling step off).")
        .def("__init__", &init_species_names, nb::arg("table") = nb::none(),
             nb::arg("max_edit_distance") = 1)
        .def_prop_ro("table",
                     [](const SpeciesNames& s) -> std::shared_ptr<NameTable> {
                         return mutable_handle(s.table);
                     })
        .def_prop_ro("max_edit_distance",
                     [](const SpeciesNames& s) { return s.options.max_edit_distance; });

    nb::class_<ResolvedName>(m, "ResolvedName", "What the resolver gives for one name.")
        .def_ro("token", &ResolvedName::token)
        .def_ro("route", &ResolvedName::route)
        .def_ro("source", &ResolvedName::source)
        .def_ro("via", &ResolvedName::via)
        .def_ro("distance", &ResolvedName::distance)
        .def("__repr__", [](const ResolvedName& r) {
            return "ResolvedName(token='" + r.token + "', route='" + enum_to_name(r.route) +
                   "', source='" + r.source + "', via='" + r.via +
                   "', distance=" + std::to_string(r.distance) + ")";
        });

    nb::class_<NameResolver>(m, "NameResolver",
                             "Resolves species names against one vocabulary: exact, then the "
                             "name table, then the nearest spelling within the edits allowed.")
        .def("__init__",
             [](NameResolver* self, const std::vector<std::string>& vocabulary,
                std::shared_ptr<NameTable> table, int max_edit_distance) {
                 new (self) NameResolver(vocabulary, std::move(table),
                                         resolve::NameResolverOptions{max_edit_distance});
             },
             nb::arg("vocabulary"), nb::arg("table") = nb::none(), nb::arg("max_edit_distance") = 1,
             "`vocabulary` as a checkpoint orders it ([0] = \"<UNK>\").")
        .def("resolve",
             [](const NameResolver& r, const std::string& name) { return r.resolve(name); },
             nb::arg("name"), nb::call_guard<nb::gil_scoped_release>())
        .def("resolve_batch",
             [](const NameResolver& r, const std::vector<std::string>& names) {
                 return r.resolve(names);
             },
             nb::arg("names"), nb::call_guard<nb::gil_scoped_release>());

    nb::class_<SpeciesNameReport>(m, "SpeciesNameReport",
                                  "The species names that were not found as written, one row per "
                                  "name in the order they first appear.")
        .def_ro("name", &SpeciesNameReport::name)
        .def_ro("token", &SpeciesNameReport::token)
        .def_prop_ro("route",
                     [](const SpeciesNameReport& r) {
                         std::vector<std::string> out;
                         out.reserve(r.size());
                         for (const resolve::NameRoute route : r.route) out.emplace_back(enum_to_name(route));
                         return out;
                     })
        .def_ro("source", &SpeciesNameReport::source)
        .def_ro("via", &SpeciesNameReport::via)
        .def_ro("distance", &SpeciesNameReport::distance)
        .def_ro("n_records", &SpeciesNameReport::n_records)
        .def("__len__", &SpeciesNameReport::size);
}
