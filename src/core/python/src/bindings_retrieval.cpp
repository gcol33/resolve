#include "bindings_common.hpp"

#include "resolve/retrieval.hpp"

void register_retrieval(nb::module_& m) {
    m.def(
        "nearest_by_cosine",
        [](const nb::object& queries, const nb::object& reference, int64_t k, bool exclude_self) {
            const at::Tensor query_rows = unpack_required_tensor(queries, "queries");
            const at::Tensor reference_rows = unpack_required_tensor(reference, "reference");
            resolve::Neighbours found;
            {
                nb::gil_scoped_release release;
                found = resolve::nearest_by_cosine(query_rows, reference_rows, k, exclude_self);
            }
            return nb::make_tuple(tensor_or_none(found.indices),
                                  tensor_or_none(found.similarities));
        },
        nb::arg("queries"), nb::arg("reference"), nb::arg("k"), nb::arg("exclude_self") = false,
        "The k rows of `reference` most similar to each row of `queries` by cosine\n"
        "similarity, most similar first: the engine's exact blocked search.\n"
        "\n"
        "queries, reference: dense (rows, dim) torch tensors of one dim, any\n"
        "floating type (computed in float32; a numpy array goes through\n"
        "torch.from_numpy). The search runs on the device `reference` is on;\n"
        "`queries` may stay in host memory and is moved over one block of rows\n"
        "at a time. Memory stays bounded whatever the size of either set.\n"
        "k: neighbours per query, clamped to the reference size (one less with\n"
        "exclude_self).\n"
        "exclude_self: query row i is reference row i (pass the same tensor\n"
        "twice) and is never returned as its own neighbour; another row with\n"
        "identical values still is.\n"
        "\n"
        "Returns (indices, similarities): (n_queries, k) int64 rows of\n"
        "`reference` and their float32 cosine similarities, on the CPU.");
}
