/**
 * Pure CUDA kernel implementations.
 *
 * This file is compiled by nvcc WITHOUT PyTorch headers.
 * It only uses CUDA runtime API - compatible with CUDA 13.x.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace resolve {
namespace cuda {

// MurmurHash3 32-bit finalizer
__device__ __forceinline__ int32_t murmur_hash32(int64_t key) {
    uint64_t h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int32_t>(h);
}

/**
 * Kernel to compute hash indices and signs for species IDs.
 */
__global__ void compute_hash_kernel(
    const int64_t* species_ids,
    int32_t* hash_indices,
    int8_t* signs,
    int64_t n,
    int32_t hash_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int32_t h = murmur_hash32(species_ids[idx]);
        hash_indices[idx] = (h < 0 ? -h : h) % hash_dim;
        signs[idx] = (h >= 0) ? 1 : -1;
    }
}

/**
 * Combined kernel: compute hash and aggregate in one pass.
 */
__global__ void hash_and_aggregate_kernel(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t plot_idx = plot_indices[idx];
        int32_t h = murmur_hash32(species_ids[idx]);
        int32_t hash_idx = (h < 0 ? -h : h) % hash_dim;
        float sign = (h >= 0) ? 1.0f : -1.0f;
        float contribution = sign * weights[idx];

        atomicAdd(&output[plot_idx * hash_dim + hash_idx], contribution);
    }
}

// Extern "C" launcher functions - callable from C++ without nvcc

extern "C" {

cudaError_t resolve_launch_hash_and_aggregate(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    void* stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    hash_and_aggregate_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim
    );

    return cudaGetLastError();
}

cudaError_t resolve_launch_compute_hash(
    const int64_t* species_ids,
    int32_t* hash_indices,
    int8_t* signs,
    int64_t n,
    int32_t hash_dim,
    void* stream
) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    compute_hash_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        species_ids, hash_indices, signs, n, hash_dim
    );

    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace resolve
