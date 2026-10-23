/**
 * Benchmark: CPU vs GPU for RESOLVE operations
 */

#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "benchmark_hash.hpp"

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

    const auto records = resolve::bench::random_hash_records(n_rows, n_plots, hash_dim);
    const auto cpu_result = resolve::bench::hash_embedding_cpu(records);

    double cpu_time = benchmark([&]() {
        auto result = resolve::bench::hash_embedding_cpu(records);
    });
    std::cout << std::left << std::setw(18) << "CPU" << std::fixed << std::setprecision(3)
              << cpu_time << " ms" << std::endl;

#ifdef RESOLVE_HAS_CUDA
    if (torch::cuda::is_available()) {
        const auto cuda_records = resolve::bench::to_cuda(records);

        for (const auto& kernel : resolve::bench::kHashKernels) {
            const auto result = resolve::bench::run_hash_kernel(kernel.kind, cuda_records);
            const float diff = (cpu_result - result.to(torch::kCPU)).abs().max().item<float>();

            double gpu_time = benchmark([&]() {
                auto output = resolve::bench::run_hash_kernel(kernel.kind, cuda_records);
            });
            std::cout << std::left << std::setw(18) << kernel.label << std::fixed
                      << std::setprecision(3) << gpu_time << " ms"
                      << " (" << std::setprecision(1) << (cpu_time / gpu_time) << "x)"
                      << (diff < 1e-3f ? "" : "  MISMATCH vs CPU") << std::endl;
        }
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
