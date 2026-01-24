/**
 * Benchmark: CPU vs GPU for RESOLVE operations
 */

#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef RESOLVE_HAS_CUDA
#include "resolve/cuda/feature_hash.hpp"
#endif

// Simple CPU implementation for comparison
torch::Tensor compute_hash_embedding_cpu(
    torch::Tensor plot_indices,
    torch::Tensor species_ids,
    torch::Tensor weights,
    int64_t n_plots,
    int32_t hash_dim
) {
    auto output = torch::zeros({n_plots, hash_dim}, torch::kFloat32);
    auto plot_acc = plot_indices.accessor<int64_t, 1>();
    auto species_acc = species_ids.accessor<int64_t, 1>();
    auto weight_acc = weights.accessor<float, 1>();
    auto out_acc = output.accessor<float, 2>();

    for (int64_t i = 0; i < plot_indices.size(0); ++i) {
        int64_t plot_idx = plot_acc[i];
        int64_t species_id = species_acc[i];

        // MurmurHash3-style mixing
        uint64_t h = static_cast<uint64_t>(species_id);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        int32_t hash = static_cast<int32_t>(h);

        int32_t hash_idx = (hash < 0 ? -hash : hash) % hash_dim;
        float sign = (hash >= 0) ? 1.0f : -1.0f;

        out_acc[plot_idx][hash_idx] += sign * weight_acc[i];
    }

    return output;
}

template<typename Func>
double benchmark(Func&& fn, int warmup = 3, int iterations = 10) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn();
    }

    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }

    if (torch::cuda::is_available()) {
        torch::cuda::synchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0 / iterations;  // ms per iteration
}

void run_hash_embedding_benchmark(int64_t n_rows, int64_t n_plots, int32_t hash_dim) {
    std::cout << "\n=== Hash Embedding Benchmark ===" << std::endl;
    std::cout << "n_rows: " << n_rows << ", n_plots: " << n_plots << ", hash_dim: " << hash_dim << std::endl;

    // Generate random test data
    auto plot_indices = torch::randint(0, n_plots, {n_rows}, torch::kInt64);
    auto species_ids = torch::randint(0, 100000, {n_rows}, torch::kInt64);
    auto weights = torch::rand({n_rows}, torch::kFloat32);

    // CPU benchmark
    double cpu_time = benchmark([&]() {
        auto result = compute_hash_embedding_cpu(plot_indices, species_ids, weights, n_plots, hash_dim);
    });
    std::cout << "CPU:  " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;

#ifdef RESOLVE_HAS_CUDA
    if (torch::cuda::is_available()) {
        // Move data to GPU
        auto plot_indices_cuda = plot_indices.to(torch::kCUDA);
        auto species_ids_cuda = species_ids.to(torch::kCUDA);
        auto weights_cuda = weights.to(torch::kCUDA);

        // GPU benchmark
        double gpu_time = benchmark([&]() {
            auto result = resolve::cuda::compute_hash_embedding_cuda(
                plot_indices_cuda, species_ids_cuda, weights_cuda, n_plots, hash_dim);
        });
        std::cout << "GPU:  " << std::fixed << std::setprecision(3) << gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(1) << (cpu_time / gpu_time) << "x" << std::endl;

        // Verify results match
        auto cpu_result = compute_hash_embedding_cpu(plot_indices, species_ids, weights, n_plots, hash_dim);
        auto gpu_result = resolve::cuda::compute_hash_embedding_cuda(
            plot_indices_cuda, species_ids_cuda, weights_cuda, n_plots, hash_dim);
        auto diff = (cpu_result - gpu_result.to(torch::kCPU)).abs().max().item<float>();
        std::cout << "Max diff: " << diff << (diff < 1e-5 ? " (OK)" : " (MISMATCH!)") << std::endl;
    } else {
        std::cout << "GPU:  N/A (CUDA not available)" << std::endl;
    }
#else
    std::cout << "GPU:  N/A (compiled without CUDA)" << std::endl;
#endif
}

void run_matmul_benchmark(int64_t m, int64_t k, int64_t n) {
    std::cout << "\n=== Matrix Multiply Benchmark (PyTorch) ===" << std::endl;
    std::cout << "Shape: [" << m << " x " << k << "] @ [" << k << " x " << n << "]" << std::endl;

    auto a_cpu = torch::rand({m, k}, torch::kFloat32);
    auto b_cpu = torch::rand({k, n}, torch::kFloat32);

    // CPU benchmark
    double cpu_time = benchmark([&]() {
        auto c = torch::mm(a_cpu, b_cpu);
    });
    std::cout << "CPU:  " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;

    if (torch::cuda::is_available()) {
        auto a_cuda = a_cpu.to(torch::kCUDA);
        auto b_cuda = b_cpu.to(torch::kCUDA);

        // GPU benchmark
        double gpu_time = benchmark([&]() {
            auto c = torch::mm(a_cuda, b_cuda);
        });
        std::cout << "GPU:  " << std::fixed << std::setprecision(3) << gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(1) << (cpu_time / gpu_time) << "x" << std::endl;
    } else {
        std::cout << "GPU:  N/A (CUDA not available)" << std::endl;
    }
}

void run_forward_pass_benchmark(int64_t batch_size, int64_t input_dim, int64_t hidden_dim) {
    std::cout << "\n=== Neural Network Forward Pass Benchmark ===" << std::endl;
    std::cout << "batch: " << batch_size << ", input: " << input_dim << ", hidden: " << hidden_dim << std::endl;

    // Simple 3-layer MLP
    auto fc1 = torch::nn::Linear(input_dim, hidden_dim);
    auto fc2 = torch::nn::Linear(hidden_dim, hidden_dim);
    auto fc3 = torch::nn::Linear(hidden_dim, 1);

    auto x_cpu = torch::rand({batch_size, input_dim}, torch::kFloat32);

    // CPU benchmark
    double cpu_time = benchmark([&]() {
        auto h = torch::relu(fc1->forward(x_cpu));
        h = torch::relu(fc2->forward(h));
        auto out = fc3->forward(h);
    });
    std::cout << "CPU:  " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;

    if (torch::cuda::is_available()) {
        fc1->to(torch::kCUDA);
        fc2->to(torch::kCUDA);
        fc3->to(torch::kCUDA);
        auto x_cuda = x_cpu.to(torch::kCUDA);

        // GPU benchmark
        double gpu_time = benchmark([&]() {
            auto h = torch::relu(fc1->forward(x_cuda));
            h = torch::relu(fc2->forward(h));
            auto out = fc3->forward(h);
        });
        std::cout << "GPU:  " << std::fixed << std::setprecision(3) << gpu_time << " ms" << std::endl;
        std::cout << "Speedup: " << std::fixed << std::setprecision(1) << (cpu_time / gpu_time) << "x" << std::endl;
    } else {
        std::cout << "GPU:  N/A (CUDA not available)" << std::endl;
    }
}

int main() {
    std::cout << "=== RESOLVE CPU vs GPU Benchmark ===" << std::endl;
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA devices: " << torch::cuda::device_count() << std::endl;
    }

    // Small dataset
    std::cout << "\n--- Small Dataset ---" << std::endl;
    run_hash_embedding_benchmark(10000, 100, 256);
    run_matmul_benchmark(256, 256, 256);
    run_forward_pass_benchmark(64, 256, 128);

    // Medium dataset
    std::cout << "\n--- Medium Dataset ---" << std::endl;
    run_hash_embedding_benchmark(100000, 1000, 512);
    run_matmul_benchmark(1024, 1024, 1024);
    run_forward_pass_benchmark(256, 512, 256);

    // Large dataset
    std::cout << "\n--- Large Dataset ---" << std::endl;
    run_hash_embedding_benchmark(1000000, 10000, 1024);
    run_matmul_benchmark(2048, 2048, 2048);
    run_forward_pass_benchmark(1024, 1024, 512);

    return 0;
}
