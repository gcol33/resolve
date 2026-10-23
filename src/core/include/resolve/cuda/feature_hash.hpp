#pragma once

#include <torch/torch.h>

namespace resolve {
namespace cuda {

/**
 * Signed feature-hash embedding of a batch of plots on the GPU, from species
 * records stored in CSR form (records of plot p are rows plot_offsets[p] ..
 * plot_offsets[p + 1] - 1). Bucket and sign per record come from
 * resolve::hash_bucket_sign (hash_core.hpp), the definition the CPU encoder
 * uses.
 *
 * Inputs not on batch_indices' device are moved there; all are made
 * contiguous. The kernel runs on the current CUDA stream.
 *
 * @param batch_indices    (batch_size,) int64 plot indices of the batch
 * @param raw_species_ids  (n_total_records,) int64 species keys
 * @param raw_weights      (n_total_records,) float32 species weights
 * @param plot_offsets     (n_plots + 1,) int64 CSR offsets
 * @param hash_dim         embedding width, > 0
 * @return (batch_size, hash_dim) float32 embedding on batch_indices' device;
 *         rows of out-of-range plots are zero
 */
torch::Tensor compute_batch_hash_embedding_cuda(
    torch::Tensor batch_indices,
    torch::Tensor raw_species_ids,
    torch::Tensor raw_weights,
    torch::Tensor plot_offsets,
    int32_t hash_dim
);

} // namespace cuda
} // namespace resolve
