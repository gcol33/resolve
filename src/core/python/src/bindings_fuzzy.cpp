// Python bindings for resolve::fuzzy::FuzzyIndex.
//
// Exposed under `_resolve_core.fuzzy.FuzzyIndex` per the plan in
// dev_notes/fuzzy_index_plan.md.

#include "bindings_common.hpp"
#include "resolve/fuzzy.hpp"

#include <nanobind/stl/function.h>

#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Convert an optional Python callable into a std::function the C++ core can
// call directly. Holds the GIL while calling back.
std::function<std::string(std::string_view)> make_bucket_fn(nb::object py_callable) {
    if (py_callable.is_none()) return nullptr;
    // nb::object's lambda capture is reference-counted, so this is safe even
    // after the binding call returns.
    return [py_callable](std::string_view sv) -> std::string {
        nb::gil_scoped_acquire gil;
        nb::object result = py_callable(nb::str(sv.data(), sv.size()));
        return nb::cast<std::string>(result);
    };
}

resolve::fuzzy::FuzzyIndex build_index(std::vector<std::string> entries,
                                       int max_edit_distance,
                                       nb::object bucket_fn,
                                       bool case_insensitive,
                                       bool damerau)
{
    resolve::fuzzy::BuildOptions opts;
    opts.max_edit_distance = max_edit_distance;
    opts.bucket_fn = make_bucket_fn(bucket_fn);
    opts.case_insensitive = case_insensitive;
    opts.damerau = damerau;
    return resolve::fuzzy::FuzzyIndex::build(entries, std::move(opts));
}

std::vector<resolve::fuzzy::Match> do_query(const resolve::fuzzy::FuzzyIndex& self,
                                            const std::string& needle,
                                            int max_edit_distance,
                                            int top_n,
                                            nb::object bucket_hint)
{
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_edit_distance;
    q.top_n = top_n;
    if (!bucket_hint.is_none()) {
        q.bucket_hint = nb::cast<std::string>(bucket_hint);
    }
    return self.query(needle, q);
}

std::vector<std::vector<resolve::fuzzy::Match>>
do_query_batch(const resolve::fuzzy::FuzzyIndex& self,
               std::vector<std::string> needles,
               int max_edit_distance,
               int top_n,
               nb::object bucket_fn,
               nb::object bucket_hint)
{
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_edit_distance;
    q.top_n = top_n;
    if (!bucket_hint.is_none()) {
        q.bucket_hint = nb::cast<std::string>(bucket_hint);
    }
    auto fn = make_bucket_fn(bucket_fn);
    // Release the GIL while the OpenMP parallel section runs; reacquire only
    // inside per-needle callbacks (handled in make_bucket_fn).
    nb::gil_scoped_release nogil;
    return self.query_batch(needles, q, fn);
}

} // namespace

void register_fuzzy(nb::module_& m) {
    auto sub = m.def_submodule("fuzzy",
        "Generic Damerau-Levenshtein fuzzy-string index.");

    nb::class_<resolve::fuzzy::Match>(sub, "Match",
        "A single top-N match returned by FuzzyIndex.query.")
        .def_ro("entry", &resolve::fuzzy::Match::entry,
                "Matched dictionary entry (original casing).")
        .def_ro("id", &resolve::fuzzy::Match::id,
                "Index into the `entries` list passed to FuzzyIndex.build.")
        .def_ro("distance", &resolve::fuzzy::Match::distance,
                "Damerau-Levenshtein distance from the query.")
        .def("__repr__", [](const resolve::fuzzy::Match& match) {
            return "Match(entry='" + match.entry +
                   "', id=" + std::to_string(match.id) +
                   ", distance=" + std::to_string(match.distance) + ")";
        });

    nb::class_<resolve::fuzzy::FuzzyIndex>(sub, "FuzzyIndex",
        "Trie + Levenshtein-automaton index over a string vocabulary.")
        .def_static("build",
            &build_index,
            nb::arg("entries"),
            nb::arg("max_edit_distance") = 3,
            nb::arg("bucket_fn") = nb::none(),
            nb::arg("case_insensitive") = true,
            nb::arg("damerau") = true,
            "Build an index over `entries`. If `bucket_fn` is provided it is\n"
            "called on each entry to derive a bucket key; queries can then pass\n"
            "`bucket_hint` to prune the search before falling back to the\n"
            "global trie. `max_edit_distance` caps the distance the index is\n"
            "sized for; per-query distances must be <= this value.")
        .def("query",
            &do_query,
            nb::arg("needle"),
            nb::arg("max_edit_distance") = 2,
            nb::arg("top_n") = 5,
            nb::arg("bucket_hint") = nb::none(),
            "Return top-N matches within `max_edit_distance` of `needle`,\n"
            "sorted by (distance asc, id asc). If `bucket_hint` is provided\n"
            "and matches are found in that bucket, only those are returned;\n"
            "otherwise the global trie is searched.")
        .def("query_batch",
            &do_query_batch,
            nb::arg("needles"),
            nb::arg("max_edit_distance") = 2,
            nb::arg("top_n") = 5,
            nb::arg("bucket_fn") = nb::none(),
            nb::arg("bucket_hint") = nb::none(),
            "OpenMP-parallel batch query. If `bucket_fn` is provided it is\n"
            "called per-needle to derive a bucket hint; otherwise the single\n"
            "`bucket_hint` is reused for every needle.")
        .def("__len__", &resolve::fuzzy::FuzzyIndex::size)
        .def_prop_ro("bucket_count", &resolve::fuzzy::FuzzyIndex::bucket_count)
        .def_prop_ro("max_supported_distance",
                     &resolve::fuzzy::FuzzyIndex::max_supported_distance);
}
