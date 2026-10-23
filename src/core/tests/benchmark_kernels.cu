// Benchmark-only alternatives to the production CSR hash kernel
// (cuda/kernels.cu). Declarations and semantics in benchmark_kernels.h.

#include "benchmark_kernels.h"
#include "launch_common.cuh"

namespace resolve {
namespace cuda {
namespace bench {

namespace {

bool valid_coo_args(const int64_t* plot_indices, const int64_t* species_ids,
                    const float* weights, const float* output,
                    int64_t n, int64_t n_plots, int32_t hash_dim) {
    return plot_indices && species_ids && weights && output
        && n > 0 && n_plots > 0 && hash_dim > 0;
}

__global__ void hash_aggregate_atomic_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int64_t plot_idx = plot_indices[idx];
    if (plot_idx < 0 || plot_idx >= n_plots) return;

    accumulate_hashed(output + plot_idx * hash_dim, species_ids[idx], weights[idx], hash_dim);
}

__global__ void hash_aggregate_shared_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    const int64_t plot_idx = blockIdx.x;
    if (plot_idx >= n_plots) return;

    extern __shared__ float shared_row[];
    zero_row(shared_row, hash_dim);
    __syncthreads();

    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (plot_indices[i] == plot_idx) {
            accumulate_hashed(shared_row, species_ids[i], weights[i], hash_dim);
        }
    }
    __syncthreads();

    copy_row(output + plot_idx * hash_dim, shared_row, hash_dim);
}

__global__ void hash_aggregate_chunked_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    int64_t chunk_size
) {
    const int64_t chunk_start = static_cast<int64_t>(blockIdx.x) * chunk_size;
    if (chunk_start >= n) return;
    const int64_t chunk_end = min(chunk_start + chunk_size, n);

    // The dominant plot is uniform across the block, so an invalid one makes
    // the whole block return before any barrier.
    const int64_t dominant_plot = plot_indices[chunk_start];
    if (dominant_plot < 0 || dominant_plot >= n_plots) return;

    extern __shared__ float shared_row[];
    zero_row(shared_row, hash_dim);
    __syncthreads();

    for (int64_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        const int64_t plot_idx = plot_indices[i];
        if (plot_idx < 0 || plot_idx >= n_plots) continue;
        float* row = (plot_idx == dominant_plot) ? shared_row : output + plot_idx * hash_dim;
        accumulate_hashed(row, species_ids[i], weights[i], hash_dim);
    }
    __syncthreads();

    float* out_row = output + dominant_plot * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        if (shared_row[i] != 0.0f) atomicAdd(&out_row[i], shared_row[i]);
    }
}

}  // namespace

cudaError_t launch_hash_aggregate_atomic(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream
) {
    if (!valid_coo_args(plot_indices, species_ids, weights, output, n, n_plots, hash_dim)) {
        return cudaErrorInvalidValue;
    }
    hash_aggregate_atomic_kernel<<<blocks_for(n, kHashBlockThreads), kHashBlockThreads,
                                   0, stream>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim);
    return cudaGetLastError();
}

cudaError_t launch_hash_aggregate_shared(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream
) {
    if (!valid_coo_args(plot_indices, species_ids, weights, output, n, n_plots, hash_dim)) {
        return cudaErrorInvalidValue;
    }
    size_t shared_mem = 0;
    const cudaError_t err = shared_row_bytes(hash_dim, shared_mem);
    if (err != cudaSuccess) return err;

    hash_aggregate_shared_kernel<<<static_cast<unsigned int>(n_plots), kHashBlockThreads,
                                   shared_mem, stream>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim);
    return cudaGetLastError();
}

cudaError_t launch_hash_aggregate_chunked(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, int64_t chunk_size,
    cudaStream_t stream
) {
    if (!valid_coo_args(plot_indices, species_ids, weights, output, n, n_plots, hash_dim)
        || chunk_size <= 0) {
        return cudaErrorInvalidValue;
    }
    size_t shared_mem = 0;
    const cudaError_t err = shared_row_bytes(hash_dim, shared_mem);
    if (err != cudaSuccess) return err;

    hash_aggregate_chunked_kernel<<<blocks_for(n, chunk_size), kHashBlockThreads,
                                    shared_mem, stream>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, chunk_size);
    return cudaGetLastError();
}

cudaError_t launch_hash_aggregate_auto(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream
) {
    const size_t row_bytes = static_cast<size_t>(hash_dim > 0 ? hash_dim : 0) * sizeof(float);
    if (n_plots <= 100 && n <= 10000 && row_bytes <= 48 * 1024) {
        return launch_hash_aggregate_shared(
            plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
    }
    return launch_hash_aggregate_atomic(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
}

}  // namespace bench
}  // namespace cuda
}  // namespace resolve
