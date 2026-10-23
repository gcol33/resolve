#pragma once

// Signed feature hashing of an int64 species key into a fixed-width embedding.
// Torch-free and CUDA-free so the CPU encoder (species_encoding.cpp), the CUDA
// kernels (cuda/kernels.cu) and the tests all compile the same definition: a
// model trained with the CUDA hash path and scored on the CPU path (or the
// reverse) reads the same bucket and sign for every species.

#include <cstdint>

#if defined(__CUDACC__)
#define RESOLVE_HASH_INLINE __host__ __device__ __forceinline__
#else
#define RESOLVE_HASH_INLINE inline
#endif

namespace resolve {

// One species' slot under signed feature hashing: the embedding column it adds
// to and the sign (+1 / -1) its weight carries there.
struct HashBucketSign {
    int32_t bucket;
    float sign;
};

// MurmurHash3 fmix64 finalizer, truncated to its low 32 bits.
RESOLVE_HASH_INLINE int32_t murmur_fmix_i32(int64_t key) {
    uint64_t h = static_cast<uint64_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int32_t>(h);
}

// bucket = |h| mod hash_dim, sign = sign of h, with h = murmur_fmix_i32(key).
// |h| is taken in unsigned arithmetic so h == INT32_MIN maps to 2^31 instead
// of overflowing a signed negation. Requires hash_dim > 0.
RESOLVE_HASH_INLINE HashBucketSign hash_bucket_sign(int64_t key, int32_t hash_dim) {
    const int32_t h = murmur_fmix_i32(key);
    const uint32_t magnitude = (h < 0) ? (0u - static_cast<uint32_t>(h))
                                       : static_cast<uint32_t>(h);
    HashBucketSign out;
    out.bucket = static_cast<int32_t>(magnitude % static_cast<uint32_t>(hash_dim));
    out.sign = (h >= 0) ? 1.0f : -1.0f;
    return out;
}

}  // namespace resolve
