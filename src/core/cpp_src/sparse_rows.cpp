#include "resolve/sparse_rows.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>

namespace resolve {

namespace {

// libtorch announces its CSR layout as beta once per process, from the first
// CSR tensor it constructs. Sparse rows are an internal storage format here,
// not an API the user opted into, so that announcement is swallowed at the one
// construction site.
class DiscardWarnings final : public c10::WarningHandler {
public:
    void process(const c10::Warning&) override {}
};

int64_t n_rows_of(const torch::Tensor& rows) { return rows.size(0); }

// The row of every entry, (nnz,) int64 on the rows' device.
torch::Tensor row_of_entries(const torch::Tensor& rows) {
    const auto crow = rows.crow_indices();
    const int64_t n = n_rows_of(rows);
    const auto counts = crow.slice(0, 1, n + 1) - crow.slice(0, 0, n);
    return torch::repeat_interleave(torch::arange(n, crow.options()), counts,
                                    /*dim=*/0, rows.col_indices().numel());
}

void require_two_dim(const torch::Tensor& t, const char* what) {
    if (!t.defined() || t.dim() != 2) {
        throw std::invalid_argument(std::string(what) + " must be a defined 2-D tensor");
    }
}

}  // namespace

torch::Tensor make_sparse_rows(torch::Tensor row_offsets, torch::Tensor columns,
                               torch::Tensor values, int64_t n_columns) {
    TORCH_CHECK(row_offsets.dim() == 1 && row_offsets.numel() >= 1,
                "sparse rows: row offsets must be a non-empty 1-D tensor");
    TORCH_CHECK(columns.dim() == 1 && values.dim() == 1 && columns.numel() == values.numel(),
                "sparse rows: columns (", columns.numel(), ") and values (", values.numel(),
                ") must be 1-D and of one length");
    TORCH_CHECK(row_offsets.scalar_type() == torch::kInt64 &&
                    columns.scalar_type() == torch::kInt64,
                "sparse rows: row offsets and columns must be int64");
    TORCH_CHECK(row_offsets.device() == columns.device() &&
                    columns.device() == values.device(),
                "sparse rows: the components must share one device");
    TORCH_CHECK(n_columns >= 0, "sparse rows: n_columns must be >= 0");
    const int64_t n_rows = row_offsets.numel() - 1;
    const auto options = values.options().layout(torch::kSparseCsr);
    static DiscardWarnings discard;
    c10::WarningUtils::WarningHandlerGuard guard(&discard);
    return at::_sparse_csr_tensor_unsafe(std::move(row_offsets), std::move(columns),
                                         std::move(values), {n_rows, n_columns}, options);
}

bool is_sparse_rows(const torch::Tensor& t) noexcept {
    return t.defined() && t.layout() == torch::kSparseCsr;
}

void require_sparse_rows(const torch::Tensor& t, const char* what) {
    if (!is_sparse_rows(t)) {
        throw std::invalid_argument(
            std::string(what) + " must be a sparse-rows (CSR) tensor; a dense "
            "matrix converts with to_sparse_rows");
    }
}

torch::Tensor to_sparse_rows(const torch::Tensor& t) {
    if (is_sparse_rows(t)) return t;
    require_two_dim(t, "to_sparse_rows: the matrix");
    const int64_t n_rows = t.size(0);
    const int64_t n_columns = t.size(1);
    const auto dense = t.contiguous();
    const auto nonzero = dense.nonzero();  // (nnz, 2), row-major order
    const auto rows = nonzero.select(1, 0);
    const auto columns = nonzero.select(1, 1).contiguous();
    const auto values = dense.reshape({-1}).index_select(0, rows * n_columns + columns);
    auto offsets = torch::zeros({n_rows + 1}, rows.options());
    if (n_rows > 0) {
        offsets.slice(0, 1).copy_(torch::bincount(rows, {}, n_rows).cumsum(0));
    }
    return make_sparse_rows(std::move(offsets), columns, values, n_columns);
}

torch::Tensor sparse_rows_to_dense(const torch::Tensor& rows) {
    require_sparse_rows(rows, "sparse_rows_to_dense: the input");
    auto dense = torch::zeros({rows.size(0), rows.size(1)}, rows.values().options());
    if (rows.col_indices().numel() > 0) {
        dense.index_put_({row_of_entries(rows), rows.col_indices()}, rows.values());
    }
    return dense;
}

torch::Tensor select_rows(const torch::Tensor& t, const torch::Tensor& rows) {
    if (!is_sparse_rows(t)) return t.index_select(0, rows.to(t.device()));
    const auto crow = t.crow_indices();
    const auto idx = rows.to(crow.device(), torch::kInt64);
    const int64_t n = idx.numel();
    const auto starts = crow.index_select(0, idx);
    const auto counts = crow.index_select(0, idx + 1) - starts;
    auto offsets = torch::zeros({n + 1}, crow.options());
    if (n > 0) offsets.slice(0, 1).copy_(counts.cumsum(0));
    const int64_t total = n > 0 ? offsets[n].item<int64_t>() : 0;
    // Entry j of the result is entry starts[r] + (j - offsets[r]) of the
    // source, r being the result row that holds j.
    const auto gather =
        torch::repeat_interleave(starts - offsets.slice(0, 0, n), counts, /*dim=*/0, total) +
        torch::arange(total, crow.options());
    return make_sparse_rows(std::move(offsets), t.col_indices().index_select(0, gather),
                            t.values().index_select(0, gather), t.size(1));
}

torch::Tensor slice_rows(const torch::Tensor& t, int64_t start, int64_t end) {
    if (!is_sparse_rows(t)) return t.slice(/*dim=*/0, start, end);
    const int64_t n = n_rows_of(t);
    start = std::clamp<int64_t>(start, 0, n);
    end = std::clamp<int64_t>(end, start, n);
    const auto window = t.crow_indices().slice(0, start, end + 1);
    const auto bounds = torch::stack({window[0], window[end - start]}).cpu();
    const int64_t first = bounds[0].item<int64_t>();
    const int64_t last = bounds[1].item<int64_t>();
    return make_sparse_rows(window - first, t.col_indices().slice(0, first, last),
                            t.values().slice(0, first, last), t.size(1));
}

torch::Tensor concat_rows(const std::vector<torch::Tensor>& parts) {
    TORCH_CHECK(!parts.empty(), "concat_rows: no parts");
    const bool sparse = is_sparse_rows(parts.front());
    for (const auto& part : parts) {
        TORCH_CHECK(is_sparse_rows(part) == sparse,
                    "concat_rows: the parts mix sparse-rows and dense layouts");
    }
    if (!sparse) return torch::cat(parts, /*dim=*/0);

    const int64_t n_columns = parts.front().size(1);
    std::vector<torch::Tensor> offsets{parts.front().crow_indices().slice(0, 0, 1)};
    std::vector<torch::Tensor> columns;
    std::vector<torch::Tensor> values;
    int64_t n_entries = 0;
    for (const auto& part : parts) {
        TORCH_CHECK(part.size(1) == n_columns, "concat_rows: the parts have ",
                    n_columns, " and ", part.size(1), " columns");
        offsets.push_back(part.crow_indices().slice(0, 1) + n_entries);
        columns.push_back(part.col_indices());
        values.push_back(part.values());
        n_entries += part.col_indices().numel();
    }
    return make_sparse_rows(torch::cat(offsets), torch::cat(columns), torch::cat(values),
                            n_columns);
}

torch::Tensor sparse_rows_matmul(const torch::Tensor& rows, const torch::Tensor& table) {
    require_sparse_rows(rows, "sparse_rows_matmul: the rows");
    require_two_dim(table, "sparse_rows_matmul: the table");
    TORCH_CHECK(table.size(0) == rows.size(1), "sparse_rows_matmul: the rows have ",
                rows.size(1), " columns but the table has ", table.size(0), " rows");
    const auto weights = rows.values().to(table.scalar_type());
    auto bags = torch::embedding_bag(table, rows.col_indices(), rows.crow_indices(),
                                     /*scale_grad_by_freq=*/false, /*mode=sum*/ 0,
                                     /*sparse=*/false, weights,
                                     /*include_last_offset=*/true);
    return std::get<0>(bags);
}

torch::Tensor sparse_rows_linear(const torch::Tensor& rows, const torch::Tensor& weight,
                                 const torch::Tensor& bias) {
    auto out = sparse_rows_matmul(rows, weight.t());
    return bias.defined() ? out + bias : out;
}

torch::Tensor sparse_rows_normalized(const torch::Tensor& rows, double eps) {
    require_sparse_rows(rows, "sparse_rows_normalized: the rows");
    const auto values = rows.values();
    const auto row_of = row_of_entries(rows);
    const auto sums = torch::zeros({rows.size(0)}, values.options()).index_add_(0, row_of, values);
    return sparse_rows_with_values(rows, values / (sums.index_select(0, row_of) + eps));
}

torch::Tensor sparse_rows_with_values(const torch::Tensor& rows, torch::Tensor values) {
    require_sparse_rows(rows, "sparse_rows_with_values: the rows");
    TORCH_CHECK(values.dim() == 1 && values.numel() == rows.col_indices().numel(),
                "sparse_rows_with_values: one value per entry is needed");
    return make_sparse_rows(rows.crow_indices(), rows.col_indices(), std::move(values),
                            rows.size(1));
}

}  // namespace resolve
