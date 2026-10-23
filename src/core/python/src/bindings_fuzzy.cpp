// Python bindings for resolve::fuzzy::FuzzyIndex, as the submodule
// `_resolve_core.fuzzy`.

#include "bindings_common.hpp"
#include "resolve/fuzzy.hpp"

#include <exception>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// A Python callable as the engine's bucket function, for the length of one
// binding call. It holds the callable by reference (the caller's argument
// keeps it alive) and takes the GIL to call it, so the engine may call it from
// its own threads with the GIL released. A Python exception is stored rather
// than thrown through the engine (query_batch calls it inside an OpenMP loop,
// where an escaping exception terminates the process); every later call then
// returns "" without calling Python, and rethrow_if_failed() raises the stored
// exception once the engine has returned.
class PythonBucketFn {
public:
    explicit PythonBucketFn(nb::handle callable) : callable_(callable) {}
    PythonBucketFn(const PythonBucketFn&) = delete;
    PythonBucketFn& operator=(const PythonBucketFn&) = delete;

    // The engine-side callback, or an empty function for None.
    std::function<std::string(std::string_view)> function() {
        if (callable_.is_none()) return nullptr;
        return [this](std::string_view key) { return call(key); };
    }

    // Raise the first exception the callable threw. Call with the GIL held.
    void rethrow_if_failed() {
        if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    }

private:
    std::string call(std::string_view key) {
        nb::gil_scoped_acquire gil;
        if (error_) return {};
        try {
            return nb::cast<std::string>(callable_(nb::str(key.data(), key.size())));
        } catch (...) {
            error_ = std::current_exception();
            return {};
        }
    }

    nb::handle callable_;
    std::exception_ptr error_;
};

resolve::fuzzy::FuzzyIndex build_index(const std::vector<std::string>& entries,
                                       int max_edit_distance,
                                       nb::handle bucket_fn,
                                       bool case_insensitive,
                                       bool damerau)
{
    PythonBucketFn bucket(bucket_fn);
    resolve::fuzzy::BuildOptions opts;
    opts.max_edit_distance = max_edit_distance;
    opts.bucket_fn = bucket.function();
    opts.case_insensitive = case_insensitive;
    opts.damerau = damerau;
    resolve::fuzzy::FuzzyIndex index = resolve::fuzzy::FuzzyIndex::build(entries, std::move(opts));
    bucket.rethrow_if_failed();
    return index;
}

std::vector<resolve::fuzzy::Match> do_query(const resolve::fuzzy::FuzzyIndex& self,
                                            const std::string& needle,
                                            int max_edit_distance,
                                            int top_n,
                                            std::optional<std::string> bucket_hint)
{
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_edit_distance;
    q.top_n = top_n;
    q.bucket_hint = std::move(bucket_hint);
    nb::gil_scoped_release nogil;
    return self.query(needle, q);
}

std::vector<std::vector<resolve::fuzzy::Match>>
do_query_batch(const resolve::fuzzy::FuzzyIndex& self,
               const std::vector<std::string>& needles,
               int max_edit_distance,
               int top_n,
               nb::handle bucket_fn,
               std::optional<std::string> bucket_hint)
{
    resolve::fuzzy::QueryOptions q;
    q.max_edit_distance = max_edit_distance;
    q.top_n = top_n;
    q.bucket_hint = std::move(bucket_hint);
    PythonBucketFn bucket(bucket_fn);
    std::vector<std::vector<resolve::fuzzy::Match>> matches;
    {
        // The GIL is released around the parallel search and taken back only
        // inside the bucket callback.
        nb::gil_scoped_release nogil;
        matches = self.query_batch(needles, q, bucket.function());
    }
    bucket.rethrow_if_failed();
    return matches;
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
            "sized for; per-query distances must be <= this value. An exception\n"
            "raised by `bucket_fn` propagates.")
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
            "Parallel batch query. If `bucket_fn` is provided it is called\n"
            "per needle to derive a bucket hint; otherwise the single\n"
            "`bucket_hint` is reused for every needle. An exception raised by\n"
            "`bucket_fn` propagates once the batch has run.")
        .def("__len__", &resolve::fuzzy::FuzzyIndex::size)
        .def_prop_ro("bucket_count", &resolve::fuzzy::FuzzyIndex::bucket_count)
        .def_prop_ro("max_supported_distance",
                     &resolve::fuzzy::FuzzyIndex::max_supported_distance);
}
