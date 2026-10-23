#pragma once

// Host entry point of the CUDA hash kernel (cuda/kernels.cu). Torch-free, so
// it is shared by the nvcc translation unit that defines it and the C++ code
// that calls it.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace resolve {
namespace cuda {

// Signed feature-hash embedding of a batch of plots whose species records are
// stored in CSR form. Row b of `output` (batch_size x hash_dim, float32) is
// overwritten with the hashed sum over the records of plot batch_indices[b],
// i.e. records plot_offsets[p] .. plot_offsets[p + 1] - 1 of species_ids /
// weights. A plot index outside [0, n_total_plots) or a CSR range outside
// [0, n_total_records] leaves its row untouched. Enqueued on `stream`; returns
// the launch status (cudaErrorInvalidValue for null pointers or non-positive
// sizes, cudaErrorInvalidConfiguration when a hash_dim row does not fit in one
// block's shared memory).
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
    cudaStream_t stream);

}  // namespace cuda
}  // namespace resolve
