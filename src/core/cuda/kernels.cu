// CUDA hash kernel. Compiled by nvcc without PyTorch headers; the torch-side
// wrapper is cuda/feature_hash.cpp.
//
// Every index read from user data is bounds-checked before use: a plot index
// outside the dataset or a malformed CSR range skips that plot instead of
// reading or writing out of bounds.

#include "resolve/cuda/hash_kernels.h"
#include "launch_common.cuh"

namespace resolve {
namespace cuda {

namespace {

// One block per batch element: the plot's CSR record range is hashed into a
// shared-memory row, which is then written to the batch element's output row.
__global__ void hash_batch_csr_kernel(
    const int64_t* __restrict__ batch_indices,
    const int64_t* __restrict__ plot_offsets,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t batch_size,
    int32_t hash_dim,
    int64_t n_total_plots,
    int64_t n_total_records
) {
    const int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const int64_t plot_idx = batch_indices[batch_idx];
    if (plot_idx < 0 || plot_idx >= n_total_plots) return;

    const int64_t start = plot_offsets[plot_idx];
    const int64_t end   = plot_offsets[plot_idx + 1];
    if (start < 0 || end < start || end > n_total_records) return;

    extern __shared__ float shared_row[];
    zero_row(shared_row, hash_dim);
    __syncthreads();

    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        accumulate_hashed(shared_row, species_ids[i], weights[i], hash_dim);
    }
    __syncthreads();

    copy_row(output + batch_idx * hash_dim, shared_row, hash_dim);
}

}  // namespace

cudaError_t launch_hash_batch_csr(
    const int64_t* batch_indices,
    const int64_t* plot_offsets,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t batch_size,
    int32_t hash_dim,
    int64_t n_total_plots,
    int64_t n_total_records,
    cudaStream_t stream
) {
    if (!batch_indices || !plot_offsets || !species_ids || !weights || !output) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || hash_dim <= 0 || n_total_plots <= 0 || n_total_records <= 0) {
        return cudaErrorInvalidValue;
    }

    size_t shared_mem = 0;
    const cudaError_t err = shared_row_bytes(hash_dim, shared_mem);
    if (err != cudaSuccess) return err;

    hash_batch_csr_kernel<<<static_cast<unsigned int>(batch_size), kHashBlockThreads,
                            shared_mem, stream>>>(
        batch_indices, plot_offsets, species_ids, weights, output,
        batch_size, hash_dim, n_total_plots, n_total_records);

    return cudaGetLastError();
}

}  // namespace cuda
}  // namespace resolve
