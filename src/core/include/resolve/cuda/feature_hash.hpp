#pragma once

#include <torch/torch.h>
#include <tuple>

namespace resolve {
namespace cuda {

/**
 * Compute hash embedding on GPU using CUDA kernels.
 *
 * This function uses MurmurHash3-style hashing to map species IDs to
 * a fixed-dimensional hash embedding space, then aggregates weighted
 * contributions per plot using atomic operations.
 *
 * @param plot_indices (n_rows,) int64 tensor of plot indices
 * @param species_ids (n_rows,) int64 tensor of species IDs
 * @param weights (n_rows,) float tensor of weights (e.g., abundances)
 * @param n_plots Number of output plots
 * @param hash_dim Dimension of hash embedding
 * @return (n_plots, hash_dim) float tensor of aggregated hash embeddings
 *
 * @throws std::runtime_error if inputs are not on CUDA or not contiguous
 */
torch::Tensor compute_hash_embedding_cuda(
    torch::Tensor plot_indices,
    torch::Tensor species_ids,
    torch::Tensor weights,
    int64_t n_plots,
    int32_t hash_dim
);

/**
 * Compute hash indices and signs for species IDs.
 *
 * Returns the hash bucket index and sign (+1 or -1) for each species.
 * Useful for debugging or when separate processing steps are needed.
 *
 * @param species_ids (n_rows,) int64 tensor of species IDs
 * @param hash_dim Dimension of hash embedding
 * @return Tuple of (hash_indices, signs) tensors
 */
std::tuple<torch::Tensor, torch::Tensor> compute_hash_indices_cuda(
    torch::Tensor species_ids,
    int32_t hash_dim
);

/**
 * Check if CUDA is available for hash operations.
 */
inline bool cuda_available() {
    return torch::cuda::is_available();
}

} // namespace cuda
} // namespace resolve
