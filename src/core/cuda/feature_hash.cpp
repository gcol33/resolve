/**
 * PyTorch interface for CUDA hash embedding kernels.
 *
 * This file is compiled by the C++ compiler (not nvcc) and includes PyTorch headers.
 * It calls extern "C" kernel launchers from kernels.cu.
 *
 * This separation allows CUDA 13.x compatibility while using PyTorch.
 *
 * Validation strategy (defense-in-depth):
 *   Host side (this file):  device, dtype, contiguity, shape consistency,
 *                           non-empty tensors, positive dimensions
 *   Device side (kernels.cu): per-thread bounds checks on user-data indices,
 *                              CSR range validation, error flag reporting
 */

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

// Extern "C" declarations for CUDA kernel launchers (defined in kernels.cu)
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
);

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
);

/// Read device-side error count (call after cudaDeviceSynchronize).
uint32_t resolve_read_kernel_errors();

} // extern "C"

namespace resolve {
namespace cuda {

/**
 * Compute hash embedding on GPU.
 *
 * @param plot_indices (n_rows,) int64 tensor of plot indices
 * @param species_ids  (n_rows,) int64 tensor of species IDs
 * @param weights      (n_rows,) float32 tensor of weights
 * @param n_plots      Number of output plots (must be > 0)
 * @param hash_dim     Dimension of hash embedding (must be > 0)
 * @return (n_plots, hash_dim) float32 tensor
 */
torch::Tensor compute_hash_embedding_cuda(
    torch::Tensor plot_indices,
    torch::Tensor species_ids,
    torch::Tensor weights,
    int64_t n_plots,
    int32_t hash_dim
) {
    // --- Device & dtype validation ---
    TORCH_CHECK(plot_indices.is_cuda(), "plot_indices must be on CUDA");
    TORCH_CHECK(species_ids.is_cuda(), "species_ids must be on CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be on CUDA");
    TORCH_CHECK(plot_indices.dtype() == torch::kInt64,
                "plot_indices must be int64, got ", plot_indices.dtype());
    TORCH_CHECK(species_ids.dtype() == torch::kInt64,
                "species_ids must be int64, got ", species_ids.dtype());
    TORCH_CHECK(weights.dtype() == torch::kFloat32,
                "weights must be float32, got ", weights.dtype());

    // --- Contiguity ---
    TORCH_CHECK(plot_indices.is_contiguous(), "plot_indices must be contiguous");
    TORCH_CHECK(species_ids.is_contiguous(), "species_ids must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    // --- Shape consistency ---
    int64_t n = plot_indices.size(0);
    TORCH_CHECK(species_ids.size(0) == n,
                "species_ids length (", species_ids.size(0),
                ") must match plot_indices length (", n, ")");
    TORCH_CHECK(weights.size(0) == n,
                "weights length (", weights.size(0),
                ") must match plot_indices length (", n, ")");

    // --- Dimension validation ---
    TORCH_CHECK(n > 0, "Input tensors must be non-empty");
    TORCH_CHECK(n_plots > 0, "n_plots must be > 0, got ", n_plots);
    TORCH_CHECK(hash_dim > 0, "hash_dim must be > 0, got ", hash_dim);

    // Create output tensor (zero-initialized)
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(plot_indices.device());
    torch::Tensor output = torch::zeros({n_plots, hash_dim}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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

    TORCH_CHECK(err == cudaSuccess,
                "CUDA hash_and_aggregate kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

/**
 * Compute hash indices and signs (separate step).
 */
std::tuple<torch::Tensor, torch::Tensor> compute_hash_indices_cuda(
    torch::Tensor species_ids,
    int32_t hash_dim
) {
    TORCH_CHECK(species_ids.is_cuda(), "species_ids must be on CUDA");
    TORCH_CHECK(species_ids.dtype() == torch::kInt64,
                "species_ids must be int64, got ", species_ids.dtype());
    TORCH_CHECK(species_ids.is_contiguous(), "species_ids must be contiguous");

    int64_t n = species_ids.size(0);
    TORCH_CHECK(n > 0, "species_ids must be non-empty");
    TORCH_CHECK(hash_dim > 0, "hash_dim must be > 0, got ", hash_dim);

    auto options_i32 = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(species_ids.device());
    auto options_i8 = torch::TensorOptions()
        .dtype(torch::kInt8)
        .device(species_ids.device());

    torch::Tensor hash_indices = torch::empty({n}, options_i32);
    torch::Tensor signs = torch::empty({n}, options_i8);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = resolve_launch_compute_hash(
        species_ids.data_ptr<int64_t>(),
        hash_indices.data_ptr<int32_t>(),
        signs.data_ptr<int8_t>(),
        n,
        hash_dim,
        static_cast<void*>(stream)
    );

    TORCH_CHECK(err == cudaSuccess,
                "CUDA compute_hash kernel launch failed: ",
                cudaGetErrorString(err));

    return {hash_indices, signs};
}

/**
 * Compute hash embedding for a batch of plots using CSR-format raw species data.
 *
 * OPTIMIZED VERSION: Uses a dedicated CUDA kernel that reads CSR offsets directly
 * on the GPU, eliminating all CPU work per batch.
 */
torch::Tensor compute_batch_hash_embedding_cuda(
    torch::Tensor batch_indices,
    torch::Tensor raw_plot_indices,  // Unused in optimized version
    torch::Tensor raw_species_ids,
    torch::Tensor raw_weights,
    torch::Tensor plot_offsets,
    int32_t hash_dim
) {
    // --- Non-empty / dimension validation ---
    TORCH_CHECK(batch_indices.defined() && batch_indices.numel() > 0,
                "batch_indices must be non-empty");
    TORCH_CHECK(plot_offsets.defined() && plot_offsets.numel() >= 2,
                "plot_offsets must have at least 2 elements (one plot)");
    TORCH_CHECK(hash_dim > 0, "hash_dim must be > 0, got ", hash_dim);

    int64_t batch_size = batch_indices.size(0);

    // Determine device
    torch::Device device = batch_indices.device();

    // --- Ensure CUDA & contiguous ---
    auto ensure_cuda_contiguous = [&](torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.defined(), name, " must be defined");
        if (!t.is_cuda()) t = t.to(device);
        t = t.contiguous();
    };

    ensure_cuda_contiguous(batch_indices, "batch_indices");
    ensure_cuda_contiguous(raw_species_ids, "raw_species_ids");
    ensure_cuda_contiguous(raw_weights, "raw_weights");
    ensure_cuda_contiguous(plot_offsets, "plot_offsets");

    // --- Dtype validation ---
    TORCH_CHECK(batch_indices.dtype() == torch::kInt64,
                "batch_indices must be int64, got ", batch_indices.dtype());
    TORCH_CHECK(raw_species_ids.dtype() == torch::kInt64,
                "raw_species_ids must be int64, got ", raw_species_ids.dtype());
    TORCH_CHECK(raw_weights.dtype() == torch::kFloat32,
                "raw_weights must be float32, got ", raw_weights.dtype());
    TORCH_CHECK(plot_offsets.dtype() == torch::kInt64,
                "plot_offsets must be int64, got ", plot_offsets.dtype());

    // --- Shape consistency ---
    TORCH_CHECK(raw_species_ids.size(0) == raw_weights.size(0),
                "raw_species_ids length (", raw_species_ids.size(0),
                ") must match raw_weights length (", raw_weights.size(0), ")");

    // n_total_plots = plot_offsets.size(0) - 1 (CSR format: n+1 entries)
    int64_t n_total_plots = plot_offsets.size(0) - 1;
    int64_t n_total_records = raw_species_ids.size(0);

    // Create output tensor (zero-initialized)
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device);
    torch::Tensor output = torch::zeros({batch_size, hash_dim}, options);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaError_t err = resolve_launch_hash_batch_csr(
        batch_indices.data_ptr<int64_t>(),
        plot_offsets.data_ptr<int64_t>(),
        raw_species_ids.data_ptr<int64_t>(),
        raw_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hash_dim,
        n_total_plots,
        n_total_records,
        static_cast<void*>(stream)
    );

    TORCH_CHECK(err == cudaSuccess,
                "CUDA hash_batch_csr kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

} // namespace cuda
} // namespace resolve
