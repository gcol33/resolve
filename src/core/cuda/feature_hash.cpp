/**
 * PyTorch interface for CUDA hash embedding kernels.
 *
 * This file is compiled by the C++ compiler (not nvcc) and includes PyTorch headers.
 * It calls extern "C" kernel launchers from kernels.cu.
 *
 * This separation allows CUDA 13.x compatibility while using PyTorch.
 */

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Extern "C" declarations for CUDA kernel launchers (defined in kernels.cu)
extern "C" {

// Basic kernel (global atomics)
cudaError_t resolve_launch_hash_and_aggregate(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    void* stream
);

// Shared memory kernel (one block per plot)
cudaError_t resolve_launch_hash_and_aggregate_shared(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    void* stream
);

// Chunked kernel (better for sorted data)
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
);

// Auto-select best kernel
cudaError_t resolve_launch_hash_and_aggregate_auto(
    const int64_t* plot_indices,
    const int64_t* species_ids,
    const float* weights,
    float* output,
    int64_t n,
    int64_t n_plots,
    int32_t hash_dim,
    void* stream
);

cudaError_t resolve_launch_compute_hash(
    const int64_t* species_ids,
    int32_t* hash_indices,
    int8_t* signs,
    int64_t n,
    int32_t hash_dim,
    void* stream
);

} // extern "C"

namespace resolve {
namespace cuda {

/**
 * Compute hash embedding on GPU.
 *
 * @param plot_indices (n_rows,) int64 tensor of plot indices
 * @param species_ids (n_rows,) int64 tensor of species IDs
 * @param weights (n_rows,) float tensor of weights
 * @param n_plots Number of output plots
 * @param hash_dim Dimension of hash embedding
 * @return (n_plots, hash_dim) float tensor
 */
torch::Tensor compute_hash_embedding_cuda(
    torch::Tensor plot_indices,
    torch::Tensor species_ids,
    torch::Tensor weights,
    int64_t n_plots,
    int32_t hash_dim
) {
    // Input validation
    TORCH_CHECK(plot_indices.is_cuda(), "plot_indices must be on CUDA");
    TORCH_CHECK(species_ids.is_cuda(), "species_ids must be on CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
    TORCH_CHECK(plot_indices.is_contiguous(), "plot_indices must be contiguous");
    TORCH_CHECK(species_ids.is_contiguous(), "species_ids must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    int64_t n = plot_indices.size(0);

    // Create output tensor (zero-initialized)
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(plot_indices.device());
    torch::Tensor output = torch::zeros({n_plots, hash_dim}, options);

    // Get CUDA stream from PyTorch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel via extern "C" wrapper
    cudaError_t err = resolve_launch_hash_and_aggregate(
        plot_indices.data_ptr<int64_t>(),
        species_ids.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        n_plots,
        hash_dim,
        static_cast<void*>(stream)
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

/**
 * Compute hash indices and signs (separate step).
 *
 * Useful when indices are needed for debugging or multi-step processing.
 */
std::tuple<torch::Tensor, torch::Tensor> compute_hash_indices_cuda(
    torch::Tensor species_ids,
    int32_t hash_dim
) {
    TORCH_CHECK(species_ids.is_cuda(), "species_ids must be on CUDA");
    TORCH_CHECK(species_ids.is_contiguous(), "species_ids must be contiguous");

    int64_t n = species_ids.size(0);

    auto options_i32 = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(species_ids.device());
    auto options_i8 = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(species_ids.device());

    torch::Tensor hash_indices = torch::empty({n}, options_i32);
    torch::Tensor signs = torch::empty({n}, options_i8);

    // Get CUDA stream from PyTorch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = resolve_launch_compute_hash(
        species_ids.data_ptr<int64_t>(),
        hash_indices.data_ptr<int32_t>(),
        signs.data_ptr<int8_t>(),
        n,
        hash_dim,
        static_cast<void*>(stream)
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return {hash_indices, signs};
}

} // namespace cuda
} // namespace resolve
