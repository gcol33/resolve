#pragma once

// Sparse rows: an (n_rows, n_columns) matrix that stores only its non-zero
// entries, row after row, as a torch sparse CSR tensor:
//
//   crow_indices  (n_rows + 1,) int64  row r owns entries [crow[r], crow[r + 1])
//   col_indices   (nnz,)        int64  the column of each entry
//   values        (nnz,)               the value of each entry
//
// The explicit species vector of the sparse encoding is held this way: a plot
// records tens of species out of a vocabulary of tens of thousands, so the
// dense (n_plots, n_species) matrix is almost entirely zeros and at archive
// scale does not fit in memory. The row operations below are what the model
// inputs need (select, slice, concatenate rows), built from the three
// components so they cost O(entries touched) on any device; the products are
// the ones the models compute from a species vector.

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace resolve {

// The sparse-rows tensor over the three components (moved, not copied). The
// entries of a row are expected in increasing column order with no column
// repeated; the components must share one device.
[[nodiscard]] torch::Tensor make_sparse_rows(torch::Tensor row_offsets,
                                             torch::Tensor columns,
                                             torch::Tensor values,
                                             int64_t n_columns);

// Whether `t` is a sparse-rows (CSR) tensor.
[[nodiscard]] bool is_sparse_rows(const torch::Tensor& t) noexcept;

// Throws std::invalid_argument naming `what` unless `t` is a defined
// sparse-rows tensor.
void require_sparse_rows(const torch::Tensor& t, const char* what);

// The non-zero entries of a dense 2-D matrix as sparse rows, in row-major
// order; a sparse-rows tensor is returned as it is.
[[nodiscard]] torch::Tensor to_sparse_rows(const torch::Tensor& t);

// The dense (n_rows, n_columns) matrix of a sparse-rows tensor.
[[nodiscard]] torch::Tensor sparse_rows_to_dense(const torch::Tensor& rows);

// ---------------------------------------------------------------------------
// Row operations on any layout: the dense ops for a strided tensor, the
// component ops for a sparse-rows one.
// ---------------------------------------------------------------------------

// Rows `rows` (int64 indices, on any device), in that order.
[[nodiscard]] torch::Tensor select_rows(const torch::Tensor& t, const torch::Tensor& rows);

// Rows [start, end), clamped to the rows `t` holds.
[[nodiscard]] torch::Tensor slice_rows(const torch::Tensor& t, int64_t start, int64_t end);

// The rows of every part, stacked in order. The parts share one layout, one
// device and their trailing sizes.
[[nodiscard]] torch::Tensor concat_rows(const std::vector<torch::Tensor>& parts);

// ---------------------------------------------------------------------------
// Products
// ---------------------------------------------------------------------------

// rows @ table: (n_rows, d) from a sparse-rows tensor and a dense
// (n_columns, d) table, each output row the value-weighted sum of the table
// rows its entries name (an embedding bag). Gradients reach `table`.
[[nodiscard]] torch::Tensor sparse_rows_matmul(const torch::Tensor& rows,
                                               const torch::Tensor& table);

// A linear layer over sparse rows: rows @ weight^T + bias, weight
// (out_features, n_columns), bias (out_features) or undefined. The same
// product as torch::nn::Linear over the dense matrix.
[[nodiscard]] torch::Tensor sparse_rows_linear(const torch::Tensor& rows,
                                               const torch::Tensor& weight,
                                               const torch::Tensor& bias);

// Each row divided by its sum plus `eps`: the weights of a value-weighted mean.
[[nodiscard]] torch::Tensor sparse_rows_normalized(const torch::Tensor& rows, double eps);

// The same rows and columns carrying `values` (one per entry) instead.
[[nodiscard]] torch::Tensor sparse_rows_with_values(const torch::Tensor& rows,
                                                    torch::Tensor values);

}  // namespace resolve
