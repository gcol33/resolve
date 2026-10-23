// Launch-side helpers shared by every hash-kernel translation unit
// (cuda/kernels.cu and the benchmark kernels). nvcc only; no PyTorch headers.

#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "resolve/hash_core.hpp"

namespace resolve {
namespace cuda {

constexpr int kHashBlockThreads = 256;

// Adds `weight` to `row` at the key's hashed bucket with the key's sign. `row`
// may point to shared or global memory.
__device__ __forceinline__ void accumulate_hashed(float* row, int64_t key,
                                                  float weight, int32_t hash_dim) {
    const HashBucketSign slot = hash_bucket_sign(key, hash_dim);
    atomicAdd(&row[slot.bucket], slot.sign * weight);
}

__device__ __forceinline__ void zero_row(float* row, int32_t hash_dim) {
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) row[i] = 0.0f;
}

__device__ __forceinline__ void copy_row(float* dst, const float* src, int32_t hash_dim) {
    for (int i = threadIdx.x; i < hash_dim; i += blockDim.x) dst[i] = src[i];
}

// cudaDevAttrMaxSharedMemoryPerBlock of the current device. The attribute is
// queried once per device and cached; devices beyond the cache are queried on
// every call.
inline cudaError_t max_shared_memory_per_block(size_t& out) {
    constexpr int kCachedDevices = 64;
    static std::atomic<int> cache[kCachedDevices] = {};

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return err;

    const bool cacheable = device >= 0 && device < kCachedDevices;
    int bytes = cacheable ? cache[device].load(std::memory_order_relaxed) : 0;
    if (bytes == 0) {
        err = cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, device);
        if (err != cudaSuccess) return err;
        if (cacheable) cache[device].store(bytes, std::memory_order_relaxed);
    }
    out = static_cast<size_t>(bytes);
    return cudaSuccess;
}

// Dynamic shared memory for one float accumulator row of width hash_dim, or
// cudaErrorInvalidConfiguration (with a diagnostic on stderr) when the row does
// not fit in one block's shared memory.
inline cudaError_t shared_row_bytes(int32_t hash_dim, size_t& out) {
    size_t limit = 0;
    const cudaError_t err = max_shared_memory_per_block(limit);
    if (err != cudaSuccess) return err;
    const size_t bytes = static_cast<size_t>(hash_dim) * sizeof(float);
    if (bytes > limit) {
        std::fprintf(stderr, "[RESOLVE] hash_dim %d requires %zu B shared memory, "
                     "device supports %zu B\n", hash_dim, bytes, limit);
        return cudaErrorInvalidConfiguration;
    }
    out = bytes;
    return cudaSuccess;
}

inline int blocks_for(int64_t n, int64_t per_block) {
    return static_cast<int>((n + per_block - 1) / per_block);
}

}  // namespace cuda
}  // namespace resolve
