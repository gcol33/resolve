#pragma once

// Inputs, CPU reference and CUDA kernel dispatch shared by the hash-embedding
// benchmarks (benchmark_cuda.cpp, benchmark_catch2.cpp).

#include <torch/torch.h>

#include "resolve/hash_core.hpp"

#include <array>
#include <cstdint>

#ifdef RESOLVE_HAS_CUDA
#include "benchmark_kernels.h"
#include "resolve/cuda/feature_hash.hpp"
#include <ATen/cuda/CUDAContext.h>
#endif

namespace resolve {
namespace bench {

// Unsorted COO species records on the CPU: record i adds weights[i] to plot
// plot_indices[i] at species_ids[i]'s hashed slot.
struct HashRecords {
    torch::Tensor plot_indices;  // (n,) int64
    torch::Tensor species_ids;   // (n,) int64
    torch::Tensor weights;       // (n,) float32
    int64_t n_plots = 0;
    int32_t hash_dim = 0;
};

inline HashRecords random_hash_records(int64_t n_rows, int64_t n_plots, int32_t hash_dim) {
    return {torch::randint(0, n_plots, {n_rows}, torch::kInt64),
            torch::randint(0, 100000, {n_rows}, torch::kInt64),
            torch::rand({n_rows}, torch::kFloat32),
            n_plots, hash_dim};
}

// Single-threaded CPU aggregation, the baseline every kernel is timed against.
inline torch::Tensor hash_embedding_cpu(const HashRecords& r) {
    auto output = torch::zeros({r.n_plots, r.hash_dim}, torch::kFloat32);
    auto plot = r.plot_indices.accessor<int64_t, 1>();
    auto species = r.species_ids.accessor<int64_t, 1>();
    auto weight = r.weights.accessor<float, 1>();
    auto out = output.accessor<float, 2>();
    for (int64_t i = 0; i < r.plot_indices.size(0); ++i) {
        const HashBucketSign slot = hash_bucket_sign(species[i], r.hash_dim);
        out[plot[i]][slot.bucket] += slot.sign * weight[i];
    }
    return output;
}

#ifdef RESOLVE_HAS_CUDA

enum class HashKernel { Csr, Atomic, Shared, Chunked, Auto };

struct HashKernelInfo {
    HashKernel kind;
    const char* label;
};

// Csr is the kernel training runs; the others are benchmark-only alternatives.
inline constexpr std::array<HashKernelInfo, 5> kHashKernels = {{
    {HashKernel::Csr, "csr (production)"},
    {HashKernel::Atomic, "atomic"},
    {HashKernel::Shared, "shared"},
    {HashKernel::Chunked, "chunked"},
    {HashKernel::Auto, "auto"},
}};

constexpr int64_t kChunkedKernelChunk = 4096;

// The records on the GPU in both layouts: COO as given, and CSR (records
// sorted by plot, plot p owning rows offsets[p] .. offsets[p + 1] - 1) with a
// batch of every plot, the form the training path hashes.
struct CudaHashRecords {
    torch::Tensor plot_indices, species_ids, weights;
    torch::Tensor csr_species_ids, csr_weights, csr_offsets, csr_batch;
    int64_t n_plots = 0;
    int32_t hash_dim = 0;
};

inline CudaHashRecords to_cuda(const HashRecords& r) {
    const auto gpu = torch::Device(torch::kCUDA);
    auto order = torch::argsort(r.plot_indices);
    auto counts = torch::bincount(r.plot_indices, /*weights=*/{}, r.n_plots);
    auto offsets = torch::cat({torch::zeros({1}, torch::kInt64), counts.cumsum(0)});
    return {r.plot_indices.to(gpu), r.species_ids.to(gpu), r.weights.to(gpu),
            r.species_ids.index_select(0, order).to(gpu),
            r.weights.index_select(0, order).to(gpu),
            offsets.to(gpu),
            torch::arange(r.n_plots, torch::kInt64).to(gpu),
            r.n_plots, r.hash_dim};
}

// Runs one kernel on the current CUDA stream and returns its
// (n_plots, hash_dim) embedding.
inline torch::Tensor run_hash_kernel(HashKernel kind, const CudaHashRecords& r) {
    if (kind == HashKernel::Csr) {
        return resolve::cuda::compute_batch_hash_embedding_cuda(
            r.csr_batch, torch::Tensor(), r.csr_species_ids, r.csr_weights,
            r.csr_offsets, r.hash_dim);
    }

    auto output = torch::zeros({r.n_plots, r.hash_dim},
                               torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const int64_t* plot = r.plot_indices.data_ptr<int64_t>();
    const int64_t* species = r.species_ids.data_ptr<int64_t>();
    const float* weight = r.weights.data_ptr<float>();
    float* out = output.data_ptr<float>();
    const int64_t n = r.plot_indices.size(0);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    namespace kb = resolve::cuda::bench;
    cudaError_t err = cudaSuccess;
    switch (kind) {
        case HashKernel::Atomic:
            err = kb::launch_hash_aggregate_atomic(plot, species, weight, out, n,
                                                   r.n_plots, r.hash_dim, stream);
            break;
        case HashKernel::Shared:
            err = kb::launch_hash_aggregate_shared(plot, species, weight, out, n,
                                                   r.n_plots, r.hash_dim, stream);
            break;
        case HashKernel::Chunked:
            err = kb::launch_hash_aggregate_chunked(plot, species, weight, out, n,
                                                    r.n_plots, r.hash_dim,
                                                    kChunkedKernelChunk, stream);
            break;
        case HashKernel::Auto:
            err = kb::launch_hash_aggregate_auto(plot, species, weight, out, n,
                                                 r.n_plots, r.hash_dim, stream);
            break;
        case HashKernel::Csr:
            break;
    }
    TORCH_CHECK(err == cudaSuccess, "hash kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

#endif  // RESOLVE_HAS_CUDA

}  // namespace bench
}  // namespace resolve
