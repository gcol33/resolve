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
 * Basic kernel: compute hash and aggregate in one pass.
 * Uses global atomics - simple but can have contention.
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

/**
 * Optimized kernel using shared memory.
 *
 * Each block processes a chunk of species and accumulates into shared memory
 * before writing to global memory. This reduces global atomic contention.
 *
 * For small hash_dim (≤1024), uses shared memory for the entire hash table.
 * Much faster when many species map to the same plot.
 */
__global__ void hash_and_aggregate_shared_kernel(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    // Shared memory for local hash accumulation per plot
    // We process one plot per block when possible
    extern __shared__ float shared_hash[];

    int64_t plot_idx = blockIdx.x;
    if (plot_idx >= n_plots) return;

    // Initialize shared memory
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        shared_hash[i] = 0.0f;
    }
    __syncthreads();

    // Each thread scans the input and processes species belonging to this plot
    // This is efficient when species are roughly sorted by plot or when
    // we have many species per plot
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (plot_indices[i] == plot_idx) {
            int32_t h = murmur_hash32(species_ids[i]);
            int32_t hash_idx = (h < 0 ? -h : h) % hash_dim;
            float sign = (h >= 0) ? 1.0f : -1.0f;
            // Shared memory atomics are much faster than global
            atomicAdd(&shared_hash[hash_idx], sign * weights[i]);
        }
    }
    __syncthreads();

    // Write results to global memory (coalesced writes)
    float* out_row = output + plot_idx * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        out_row[i] = shared_hash[i];
    }
}

/**
 * Two-phase kernel for better load balancing.
 *
 * Phase 1: Each block processes a chunk of input, accumulates into shared memory.
 * Phase 2: Atomically add shared results to global output.
 *
 * Better for large datasets with many plots.
 */
__global__ void hash_and_aggregate_chunked_kernel(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    int64_t chunk_size
) {
    // Each block processes a chunk of input
    extern __shared__ float shared_accum[];  // [hash_dim] per unique plot in chunk

    int64_t chunk_start = blockIdx.x * chunk_size;
    int64_t chunk_end = min(chunk_start + chunk_size, n);

    if (chunk_start >= n) return;

    // Initialize shared memory
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        shared_accum[i] = 0.0f;
    }
    __syncthreads();

    // Find the dominant plot in this chunk (heuristic: first element)
    // This optimization works best when data is sorted by plot
    int64_t dominant_plot = plot_indices[chunk_start];

    // Process chunk - accumulate dominant plot locally, others globally
    for (int64_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        int64_t plot_idx = plot_indices[i];
        int32_t h = murmur_hash32(species_ids[i]);
        int32_t hash_idx = (h < 0 ? -h : h) % hash_dim;
        float sign = (h >= 0) ? 1.0f : -1.0f;
        float contribution = sign * weights[i];

        if (plot_idx == dominant_plot) {
            // Fast path: use shared memory
            atomicAdd(&shared_accum[hash_idx], contribution);
        } else {
            // Slow path: global atomic (rare if data is sorted)
            atomicAdd(&output[plot_idx * hash_dim + hash_idx], contribution);
        }
    }
    __syncthreads();

    // Write accumulated values for dominant plot to global memory
    float* out_row = output + dominant_plot * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        if (shared_accum[i] != 0.0f) {
            atomicAdd(&out_row[i], shared_accum[i]);
        }
    }
}

// Extern "C" launcher functions - callable from C++ without nvcc

extern "C" {

// Basic kernel launcher (original)
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

// Shared memory kernel launcher - best when n_plots is small
cudaError_t resolve_launch_hash_and_aggregate_shared(
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
    const int blocks = n_plots;  // One block per plot
    const size_t shared_mem = hash_dim * sizeof(float);

    hash_and_aggregate_shared_kernel<<<blocks, threads, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim
    );

    return cudaGetLastError();
}

// Chunked kernel launcher - best for large datasets with many plots
cudaError_t resolve_launch_hash_and_aggregate_chunked(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    int64_t chunk_size,
    void* stream
) {
    const int threads = 256;
    const int blocks = (n + chunk_size - 1) / chunk_size;
    const size_t shared_mem = hash_dim * sizeof(float);

    hash_and_aggregate_chunked_kernel<<<blocks, threads, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, chunk_size
    );

    return cudaGetLastError();
}

// Auto-select best kernel based on data characteristics
cudaError_t resolve_launch_hash_and_aggregate_auto(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    void* stream
) {
    // Heuristics based on benchmarking:
    // - basic kernel: best for most cases due to efficient global atomics
    // - shared kernel: only good for VERY small datasets (n < 10K, n_plots < 100)
    //   because it does O(n_plots * n) work scanning all rows per plot
    // - chunked kernel: good alternative but basic is usually faster
    //
    // The basic kernel with global atomics is surprisingly efficient on modern GPUs.
    // Only use shared memory for tiny datasets where atomic contention is high.

    const size_t max_shared_mem = 48 * 1024;  // 48KB typical shared memory limit
    const size_t required_shared = hash_dim * sizeof(float);

    // Shared kernel only beneficial for very small datasets where:
    // 1. Few plots (each block processes one plot)
    // 2. Few total rows (each thread scans all rows)
    // 3. High species-per-plot ratio (amortizes the full scan cost)
    if (n_plots <= 100 && n <= 10000 && required_shared <= max_shared_mem) {
        return resolve_launch_hash_and_aggregate_shared(
            plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
    }

    // For everything else, basic kernel is fastest
    return resolve_launch_hash_and_aggregate(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
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
