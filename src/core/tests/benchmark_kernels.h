#pragma once

// Alternative CUDA hash-aggregation kernels, built only for the benchmarks
// (RESOLVE_BUILD_BENCHMARKS) to compare against the production CSR kernel
// (resolve/cuda/hash_kernels.h). All take unsorted COO records: record i adds
// weights[i] to plot plot_indices[i] at species_ids[i]'s hashed slot in the
// zero-initialised (n_plots x hash_dim) float32 `output`. Records whose plot
// index is outside [0, n_plots) are skipped.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace resolve {
namespace cuda {
namespace bench {

// One thread per record, global atomics.
cudaError_t launch_hash_aggregate_atomic(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream);

// One block per plot accumulating in shared memory; each block scans all n
// records, so the cost grows with n_plots * n.
cudaError_t launch_hash_aggregate_shared(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream);

// One block per chunk of chunk_size records: the chunk's first plot
// accumulates in shared memory, every other plot through global atomics.
cudaError_t launch_hash_aggregate_chunked(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, int64_t chunk_size,
    cudaStream_t stream);

// The shared kernel for small inputs (n_plots <= 100, n <= 10000, row fits in
// 48 KiB), the atomic kernel otherwise.
cudaError_t launch_hash_aggregate_auto(
    const int64_t* plot_indices, const int64_t* species_ids, const float* weights,
    float* output, int64_t n, int64_t n_plots, int32_t hash_dim, cudaStream_t stream);

}  // namespace bench
}  // namespace cuda
}  // namespace resolve
