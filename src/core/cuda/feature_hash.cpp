// PyTorch-facing wrapper for the CUDA hash kernel. Compiled by the C++
// compiler with PyTorch headers; the kernel itself lives in kernels.cu, an nvcc
// translation unit without them.
//
// This side checks device, dtype, contiguity and shape consistency; the kernel
// bounds-checks every plot index and CSR range it reads.

#include "resolve/cuda/feature_hash.hpp"
#include "resolve/cuda/hash_kernels.h"

#include <ATen/cuda/CUDAContext.h>

namespace resolve {
namespace cuda {

torch::Tensor compute_batch_hash_embedding_cuda(
    torch::Tensor batch_indices,
    torch::Tensor /*raw_plot_indices*/,
    torch::Tensor raw_species_ids,
    torch::Tensor raw_weights,
    torch::Tensor plot_offsets,
    int32_t hash_dim
) {
    TORCH_CHECK(batch_indices.defined() && batch_indices.numel() > 0,
                "batch_indices must be non-empty");
    TORCH_CHECK(plot_offsets.defined() && plot_offsets.numel() >= 2,
                "plot_offsets must have at least 2 elements (one plot)");
    TORCH_CHECK(hash_dim > 0, "hash_dim must be > 0, got ", hash_dim);

    const int64_t batch_size = batch_indices.size(0);
    const torch::Device device = batch_indices.device();

    auto ensure_cuda_contiguous = [&](torch::Tensor& t, const char* name) {
        TORCH_CHECK(t.defined(), name, " must be defined");
        if (!t.is_cuda()) t = t.to(device);
        t = t.contiguous();
    };

    ensure_cuda_contiguous(batch_indices, "batch_indices");
    ensure_cuda_contiguous(raw_species_ids, "raw_species_ids");
    ensure_cuda_contiguous(raw_weights, "raw_weights");
    ensure_cuda_contiguous(plot_offsets, "plot_offsets");

    TORCH_CHECK(batch_indices.dtype() == torch::kInt64,
                "batch_indices must be int64, got ", batch_indices.dtype());
    TORCH_CHECK(raw_species_ids.dtype() == torch::kInt64,
                "raw_species_ids must be int64, got ", raw_species_ids.dtype());
    TORCH_CHECK(raw_weights.dtype() == torch::kFloat32,
                "raw_weights must be float32, got ", raw_weights.dtype());
    TORCH_CHECK(plot_offsets.dtype() == torch::kInt64,
                "plot_offsets must be int64, got ", plot_offsets.dtype());

    TORCH_CHECK(raw_species_ids.size(0) == raw_weights.size(0),
                "raw_species_ids length (", raw_species_ids.size(0),
                ") must match raw_weights length (", raw_weights.size(0), ")");

    const int64_t n_total_plots = plot_offsets.size(0) - 1;
    const int64_t n_total_records = raw_species_ids.size(0);

    torch::Tensor output = torch::zeros(
        {batch_size, hash_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    const cudaError_t err = launch_hash_batch_csr(
        batch_indices.data_ptr<int64_t>(),
        plot_offsets.data_ptr<int64_t>(),
        raw_species_ids.data_ptr<int64_t>(),
        raw_weights.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        hash_dim,
        n_total_plots,
        n_total_records,
        at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(err == cudaSuccess,
                "CUDA hash_batch_csr kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

} // namespace cuda
} // namespace resolve
