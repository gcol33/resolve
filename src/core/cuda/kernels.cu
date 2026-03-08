/**
 * Pure CUDA kernel implementations.
 *
 * This file is compiled by nvcc WITHOUT PyTorch headers.
 * It only uses CUDA runtime API - compatible with CUDA 13.x.
 *
 * Safety guarantees:
 *   - Every index loaded from user data is bounds-checked before use
 *   - CSR offset ranges are validated (start <= end, within data bounds)
 *   - Launchers validate parameters and shared memory limits before dispatch
 *   - Invalid inputs cause early return / skip, never out-of-bounds access
 *   - An optional device-side error flag reports violations to host code
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>  // fprintf for launcher diagnostics

namespace resolve {
namespace cuda {

// ---------------------------------------------------------------------------
// Device-side error flag
// ---------------------------------------------------------------------------
// Kernels atomically set this when they encounter invalid data (e.g. an
// out-of-range plot index). The host can read it back after a sync to
// detect data integrity issues without crashing the driver.

static __device__ uint32_t d_error_flag = 0;

/// Reset the error flag from host before a kernel launch.
static void reset_error_flag() {
    uint32_t zero = 0;
    cudaMemcpyToSymbol(d_error_flag, &zero, sizeof(uint32_t));
}

/// Read the error flag from host after synchronization.
/// Returns the number of invalid indices encountered.
static uint32_t read_error_flag() {
    uint32_t val = 0;
    cudaMemcpyFromSymbol(&val, d_error_flag, sizeof(uint32_t));
    return val;
}

/// Called from device code when an invalid index is detected.
/// Uses atomicAdd so concurrent threads don't race.
__device__ __forceinline__ void report_invalid_index() {
    atomicAdd(&d_error_flag, 1u);
}

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
 *
 * All accesses guarded by idx < n. No user-data indices involved.
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
    // Use unsigned cast to avoid UB when h == INT32_MIN (-h overflows)
    hash_indices[idx] = static_cast<int32_t>(static_cast<uint32_t>(h < 0 ? -h : h) % static_cast<uint32_t>(hash_dim));
    signs[idx] = (h >= 0) ? 1 : -1;
}

/**
 * Basic kernel: compute hash and aggregate in one pass.
 * Uses global atomics — simple but can have contention.
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
    if (plot_idx < 0 || plot_idx >= n_plots) {
        report_invalid_index();
        return;
    }

    int32_t h = murmur_hash32(species_ids[idx]);
    int32_t hash_idx = static_cast<int32_t>(static_cast<uint32_t>(h < 0 ? -h : h) % static_cast<uint32_t>(hash_dim));
    float sign = (h >= 0) ? 1.0f : -1.0f;

    atomicAdd(&output[plot_idx * hash_dim + hash_idx], sign * weights[idx]);
}

/**
 * Optimized kernel using shared memory.
 *
 * One block per plot. Each thread scans all input rows and accumulates
 * species belonging to its assigned plot into shared memory.
 *
 * Requires: shared memory >= hash_dim * sizeof(float).
 */
__global__ void hash_and_aggregate_shared_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim
) {
    extern __shared__ float shared_hash[];

    int64_t plot_idx = blockIdx.x;
    if (plot_idx >= n_plots) return;

    // Initialize shared memory
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        shared_hash[i] = 0.0f;
    }
    __syncthreads();

    // Scan all input and accumulate matches
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (plot_indices[i] == plot_idx) {
            int32_t h = murmur_hash32(species_ids[i]);
            int32_t hash_idx = static_cast<int32_t>(static_cast<uint32_t>(h < 0 ? -h : h) % static_cast<uint32_t>(hash_dim));
            float sign = (h >= 0) ? 1.0f : -1.0f;
            atomicAdd(&shared_hash[hash_idx], sign * weights[i]);
        }
    }
    __syncthreads();

    // Write results to global memory
    float* out_row = output + plot_idx * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        out_row[i] = shared_hash[i];
    }
}

/**
 * Two-phase chunked kernel for better load balancing.
 *
 * Phase 1: Each block processes a chunk of input, accumulates the
 *          "dominant" plot (first in chunk) into shared memory,
 *          everything else via global atomics.
 * Phase 2: Flush shared memory to global output.
 */
__global__ void hash_and_aggregate_chunked_kernel(
    const int64_t* __restrict__ plot_indices,
    const int64_t* __restrict__ species_ids,
    const float*   __restrict__ weights,
    float*         output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    int64_t chunk_size
) {
    extern __shared__ float shared_accum[];

    int64_t chunk_start = blockIdx.x * chunk_size;
    int64_t chunk_end = min(chunk_start + chunk_size, n);
    if (chunk_start >= n) return;

    // Initialize shared memory
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        shared_accum[i] = 0.0f;
    }
    __syncthreads();

    // Dominant plot = first element in this chunk
    int64_t dominant_plot = plot_indices[chunk_start];
    if (dominant_plot < 0 || dominant_plot >= n_plots) {
        report_invalid_index();
        return;  // entire block bails — shared mem was zeroed, nothing to flush
    }

    // Process chunk
    for (int64_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x) {
        int64_t plot_idx = plot_indices[i];
        if (plot_idx < 0 || plot_idx >= n_plots) {
            report_invalid_index();
            continue;
        }

        int32_t h = murmur_hash32(species_ids[i]);
        int32_t hash_idx = static_cast<int32_t>(static_cast<uint32_t>(h < 0 ? -h : h) % static_cast<uint32_t>(hash_dim));
        float sign = (h >= 0) ? 1.0f : -1.0f;
        float contribution = sign * weights[i];

        if (plot_idx == dominant_plot) {
            atomicAdd(&shared_accum[hash_idx], contribution);
        } else {
            atomicAdd(&output[plot_idx * hash_dim + hash_idx], contribution);
        }
    }
    __syncthreads();

    // Flush shared accumulator to global output
    float* out_row = output + dominant_plot * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        if (shared_accum[i] != 0.0f) {
            atomicAdd(&out_row[i], shared_accum[i]);
        }
    }
}

/**
 * CSR-based kernel for batch hash computation.
 *
 * One block per plot in the batch. Reads CSR offsets to find the species
 * range, hashes into shared memory, then writes to output.
 *
 * @param n_total_plots  Number of plots in the full dataset (for bounds check)
 * @param n_total_records  Total records in species_ids/weights (for CSR range check)
 */
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
    int64_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int64_t plot_idx = batch_indices[batch_idx];

    // Validate plot index against dataset bounds
    if (plot_idx < 0 || plot_idx >= n_total_plots) {
        report_invalid_index();
        return;
    }

    // Read CSR offsets and validate the range
    int64_t start = plot_offsets[plot_idx];
    int64_t end   = plot_offsets[plot_idx + 1];

    if (start < 0 || end < start || end > n_total_records) {
        report_invalid_index();
        return;
    }

    // Shared memory for local accumulation
    extern __shared__ float shared_hash[];

    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        shared_hash[i] = 0.0f;
    }
    __syncthreads();

    // Process all species for this plot
    for (int64_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        int32_t h = murmur_hash32(species_ids[i]);
        int32_t hash_idx = static_cast<int32_t>(static_cast<uint32_t>(h < 0 ? -h : h) % static_cast<uint32_t>(hash_dim));
        float sign = (h >= 0) ? 1.0f : -1.0f;
        atomicAdd(&shared_hash[hash_idx], sign * weights[i]);
    }
    __syncthreads();

    // Write results to output row for this batch element
    float* out_row = output + batch_idx * hash_dim;
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) {
        out_row[i] = shared_hash[i];
    }
}

// ---------------------------------------------------------------------------
// Extern "C" launcher functions — callable from C++ without nvcc
// ---------------------------------------------------------------------------
//
// Each launcher:
//   1. Validates parameters on the host (nullptr, non-positive dims, shared mem)
//   2. Resets the device error flag
//   3. Launches the kernel
//   4. Returns cudaGetLastError()
//
// The caller can optionally synchronize and call resolve_read_kernel_errors()
// to detect out-of-range indices that were safely skipped on the device.
// ---------------------------------------------------------------------------

extern "C" {

/// Read device-side error count (call after cudaDeviceSynchronize).
/// Returns number of invalid indices encountered across all threads.
uint32_t resolve_read_kernel_errors() {
    return read_error_flag();
}

// ---- Basic kernel launcher ------------------------------------------------

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
    // Host-side parameter validation
    if (!plot_indices || !species_ids || !weights || !output) return cudaErrorInvalidValue;
    if (n <= 0 || n_plots <= 0 || hash_dim <= 0) return cudaErrorInvalidValue;

    reset_error_flag();

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    hash_and_aggregate_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim
    );

    return cudaGetLastError();
}

// ---- Shared memory kernel launcher ----------------------------------------

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
    if (!plot_indices || !species_ids || !weights || !output) return cudaErrorInvalidValue;
    if (n <= 0 || n_plots <= 0 || hash_dim <= 0) return cudaErrorInvalidValue;

    const size_t shared_mem = static_cast<size_t>(hash_dim) * sizeof(float);

    // Query device shared memory limit
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (shared_mem > prop.sharedMemPerBlock) {
        fprintf(stderr, "[RESOLVE] hash_dim %d requires %zu B shared memory, "
                "device supports %zu B\n", hash_dim, shared_mem,
                prop.sharedMemPerBlock);
        return cudaErrorInvalidConfiguration;
    }

    reset_error_flag();

    const int threads = 256;
    const int blocks = static_cast<int>(n_plots);

    hash_and_aggregate_shared_kernel<<<blocks, threads, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim
    );

    return cudaGetLastError();
}

// ---- Chunked kernel launcher ----------------------------------------------

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
    if (!plot_indices || !species_ids || !weights || !output) return cudaErrorInvalidValue;
    if (n <= 0 || n_plots <= 0 || hash_dim <= 0 || chunk_size <= 0) return cudaErrorInvalidValue;

    const size_t shared_mem = static_cast<size_t>(hash_dim) * sizeof(float);

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (shared_mem > prop.sharedMemPerBlock) {
        fprintf(stderr, "[RESOLVE] hash_dim %d requires %zu B shared memory, "
                "device supports %zu B\n", hash_dim, shared_mem,
                prop.sharedMemPerBlock);
        return cudaErrorInvalidConfiguration;
    }

    reset_error_flag();

    const int threads = 256;
    const int blocks = static_cast<int>((n + chunk_size - 1) / chunk_size);

    hash_and_aggregate_chunked_kernel<<<blocks, threads, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, chunk_size
    );

    return cudaGetLastError();
}

// ---- Auto-select launcher -------------------------------------------------

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
    if (!plot_indices || !species_ids || !weights || !output) return cudaErrorInvalidValue;
    if (n <= 0 || n_plots <= 0 || hash_dim <= 0) return cudaErrorInvalidValue;

    const size_t required_shared = static_cast<size_t>(hash_dim) * sizeof(float);

    // Shared kernel only beneficial for very small datasets
    if (n_plots <= 100 && n <= 10000 && required_shared <= 48 * 1024) {
        return resolve_launch_hash_and_aggregate_shared(
            plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
    }

    return resolve_launch_hash_and_aggregate(
        plot_indices, species_ids, weights, output, n, n_plots, hash_dim, stream);
}

// ---- Compute hash launcher ------------------------------------------------

cudaError_t resolve_launch_compute_hash(
    const int64_t* species_ids,
    int32_t* hash_indices,
    int8_t* signs,
    int64_t n,
    int32_t hash_dim,
    void* stream
) {
    if (!species_ids || !hash_indices || !signs) return cudaErrorInvalidValue;
    if (n <= 0 || hash_dim <= 0) return cudaErrorInvalidValue;

    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    compute_hash_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        species_ids, hash_indices, signs, n, hash_dim
    );

    return cudaGetLastError();
}

// ---- CSR batch hash launcher ----------------------------------------------

cudaError_t resolve_launch_hash_batch_csr(
    const int64_t* batch_indices,
    const int64_t* plot_offsets,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t batch_size,
    int32_t hash_dim,
    int64_t n_total_plots,
    int64_t n_total_records,
    void* stream
) {
    if (!batch_indices || !plot_offsets || !species_ids || !weights || !output) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || hash_dim <= 0 || n_total_plots <= 0 || n_total_records <= 0) {
        return cudaErrorInvalidValue;
    }

    const size_t shared_mem = static_cast<size_t>(hash_dim) * sizeof(float);

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (shared_mem > prop.sharedMemPerBlock) {
        fprintf(stderr, "[RESOLVE] hash_dim %d requires %zu B shared memory, "
                "device supports %zu B\n", hash_dim, shared_mem,
                prop.sharedMemPerBlock);
        return cudaErrorInvalidConfiguration;
    }

    reset_error_flag();

    const int threads = 256;
    const int blocks = static_cast<int>(batch_size);

    hash_batch_csr_kernel<<<blocks, threads, shared_mem, static_cast<cudaStream_t>(stream)>>>(
        batch_indices, plot_offsets, species_ids, weights, output,
        batch_size, hash_dim, n_total_plots, n_total_records
    );

    return cudaGetLastError();
}

} // extern "C"

} // namespace cuda
} // namespace resolve
