/**
 * Pure CUDA kernels — header-only subset for inline use.
 *
 * This file contains only CUDA runtime API and standard C++ types.
 * It can be compiled with any CUDA version including 13.x.
 *
 * NOTE: The full kernel suite (shared-memory, chunked, CSR) lives in
 * kernels.cu and is accessed via extern "C" launchers. This header
 * provides lightweight inline wrappers for the two simplest kernels.
 */

#ifndef RESOLVE_CUDA_KERNELS_CUH
#define RESOLVE_CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

namespace resolve {
namespace cuda {

// ---------------------------------------------------------------------------
// Hash function
// ---------------------------------------------------------------------------

/// MurmurHash3 64→32-bit finalizer. Pure arithmetic, no memory access.
__device__ __forceinline__ int32_t murmur_hash32(int64_t key) {
    uint64_t h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int32_t>(h);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * Kernel to compute hash indices and signs for species IDs.
 */
__global__ void compute_hash_kernel(
    const int64_t* __restrict__ species_ids,
    int32_t*       __restrict__ hash_indices,
    int8_t*        __restrict__ signs,
    int64_t n,
    int32_t hash_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int32_t h = murmur_hash32(species_ids[idx]);
    hash_indices[idx] = (h < 0 ? -h : h) % hash_dim;
    signs[idx] = (h >= 0) ? 1 : -1;
}

/**
 * Combined kernel: compute hash and aggregate in one pass.
 * All plot indices are bounds-checked before use.
 */
__global__ void hash_and_aggregate_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int64_t plot_idx = plot_indices[idx];
    if (plot_idx < 0 || plot_idx >= n_plots) return;

    int32_t h = murmur_hash32(species_ids[idx]);
    int32_t hash_idx = (h < 0 ? -h : h) % hash_dim;
    float sign = (h >= 0) ? 1.0f : -1.0f;

    atomicAdd(&output[plot_idx * hash_dim + hash_idx], sign * weights[idx]);
}

// ---------------------------------------------------------------------------
// Inline launchers (lightweight — no error flag, no shared mem validation)
// For production use, prefer the extern "C" launchers in kernels.cu.
// ---------------------------------------------------------------------------

extern "C" {

inline void launch_hash_and_aggregate(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    cudaStream_t stream = nullptr
) {
    if (n <= 0 || n_plots <= 0 || hash_dim <= 0) return;

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    hash_and_aggregate_kernel<<<blocks, threads, 0, stream>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim
    );
}

inline void launch_compute_hash(
    const int64_t* species_ids,
    int32_t* hash_indices,
    int8_t* signs,
    int64_t n,
    int32_t hash_dim,
    cudaStream_t stream = nullptr
) {
    if (n <= 0 || hash_dim <= 0) return;

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    compute_hash_kernel<<<blocks, threads, 0, stream>>>(
        species_ids, hash_indices, signs, n, hash_dim
    );
}

} // extern "C"

} // namespace cuda
} // namespace resolve

#endif // RESOLVE_CUDA_KERNELS_CUH
