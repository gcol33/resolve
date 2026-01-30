/**
 * Catch2 Benchmarks for RESOLVE
 *
 * Includes:
 * - Hash embedding CPU vs GPU
 * - Matrix multiply
 * - Forward pass
 * - End-to-end training (small model)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <torch/torch.h>

#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/types.hpp"

#ifdef RESOLVE_HAS_CUDA
#include "resolve/cuda/feature_hash.hpp"
#include <ATen/cuda/CUDAContext.h>
#endif

using namespace resolve;

// ============================================================================
// CPU Hash Embedding (baseline for comparison)
// ============================================================================

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

// ============================================================================
// Hash Embedding Benchmarks
// ============================================================================

TEST_CASE("Hash Embedding Benchmarks", "[benchmark][hash]") {
    SECTION("Small: 10K rows, 100 plots, hash_dim=256") {
        const int64_t n_rows = 10000;
        const int64_t n_plots = 100;
        const int32_t hash_dim = 256;

        auto plot_indices = torch::randint(0, n_plots, {n_rows}, torch::kInt64);
        auto species_ids = torch::randint(0, 100000, {n_rows}, torch::kInt64);
        auto weights = torch::rand({n_rows}, torch::kFloat32);

        BENCHMARK("CPU") {
            return compute_hash_embedding_cpu(plot_indices, species_ids, weights, n_plots, hash_dim);
        };

#ifdef RESOLVE_HAS_CUDA
        if (torch::cuda::is_available()) {
            auto plot_cuda = plot_indices.to(torch::kCUDA);
            auto species_cuda = species_ids.to(torch::kCUDA);
            auto weights_cuda = weights.to(torch::kCUDA);

            BENCHMARK("GPU (CUDA)") {
                return cuda::compute_hash_embedding_cuda(
                    plot_cuda, species_cuda, weights_cuda, n_plots, hash_dim);
            };
        }
#endif
    }

    SECTION("Medium: 100K rows, 1K plots, hash_dim=512") {
        const int64_t n_rows = 100000;
        const int64_t n_plots = 1000;
        const int32_t hash_dim = 512;

        auto plot_indices = torch::randint(0, n_plots, {n_rows}, torch::kInt64);
        auto species_ids = torch::randint(0, 100000, {n_rows}, torch::kInt64);
        auto weights = torch::rand({n_rows}, torch::kFloat32);

        BENCHMARK("CPU") {
            return compute_hash_embedding_cpu(plot_indices, species_ids, weights, n_plots, hash_dim);
        };

#ifdef RESOLVE_HAS_CUDA
        if (torch::cuda::is_available()) {
            auto plot_cuda = plot_indices.to(torch::kCUDA);
            auto species_cuda = species_ids.to(torch::kCUDA);
            auto weights_cuda = weights.to(torch::kCUDA);

            BENCHMARK("GPU (CUDA)") {
                return cuda::compute_hash_embedding_cuda(
                    plot_cuda, species_cuda, weights_cuda, n_plots, hash_dim);
            };
        }
#endif
    }

    SECTION("Large: 1M rows, 10K plots, hash_dim=1024") {
        const int64_t n_rows = 1000000;
        const int64_t n_plots = 10000;
        const int32_t hash_dim = 1024;

        auto plot_indices = torch::randint(0, n_plots, {n_rows}, torch::kInt64);
        auto species_ids = torch::randint(0, 100000, {n_rows}, torch::kInt64);
        auto weights = torch::rand({n_rows}, torch::kFloat32);

        BENCHMARK("CPU") {
            return compute_hash_embedding_cpu(plot_indices, species_ids, weights, n_plots, hash_dim);
        };

#ifdef RESOLVE_HAS_CUDA
        if (torch::cuda::is_available()) {
            auto plot_cuda = plot_indices.to(torch::kCUDA);
            auto species_cuda = species_ids.to(torch::kCUDA);
            auto weights_cuda = weights.to(torch::kCUDA);

            BENCHMARK("GPU (CUDA)") {
                return cuda::compute_hash_embedding_cuda(
                    plot_cuda, species_cuda, weights_cuda, n_plots, hash_dim);
            };
        }
#endif
    }
}

// ============================================================================
// Matrix Multiply Benchmarks (PyTorch mm)
// ============================================================================

TEST_CASE("Matrix Multiply Benchmarks", "[benchmark][matmul]") {
    SECTION("Small: 256x256") {
        auto a_cpu = torch::rand({256, 256}, torch::kFloat32);
        auto b_cpu = torch::rand({256, 256}, torch::kFloat32);

        BENCHMARK("CPU") {
            return torch::mm(a_cpu, b_cpu);
        };

        if (torch::cuda::is_available()) {
            auto a_cuda = a_cpu.to(torch::kCUDA);
            auto b_cuda = b_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto result = torch::mm(a_cuda, b_cuda);
                torch::cuda::synchronize();
                return result;
            };
        }
    }

    SECTION("Medium: 1024x1024") {
        auto a_cpu = torch::rand({1024, 1024}, torch::kFloat32);
        auto b_cpu = torch::rand({1024, 1024}, torch::kFloat32);

        BENCHMARK("CPU") {
            return torch::mm(a_cpu, b_cpu);
        };

        if (torch::cuda::is_available()) {
            auto a_cuda = a_cpu.to(torch::kCUDA);
            auto b_cuda = b_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto result = torch::mm(a_cuda, b_cuda);
                torch::cuda::synchronize();
                return result;
            };
        }
    }

    SECTION("Large: 2048x2048") {
        auto a_cpu = torch::rand({2048, 2048}, torch::kFloat32);
        auto b_cpu = torch::rand({2048, 2048}, torch::kFloat32);

        BENCHMARK("CPU") {
            return torch::mm(a_cpu, b_cpu);
        };

        if (torch::cuda::is_available()) {
            auto a_cuda = a_cpu.to(torch::kCUDA);
            auto b_cuda = b_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto result = torch::mm(a_cuda, b_cuda);
                torch::cuda::synchronize();
                return result;
            };
        }
    }
}

// ============================================================================
// Forward Pass Benchmarks
// ============================================================================

TEST_CASE("Forward Pass Benchmarks", "[benchmark][forward]") {
    // Simple 3-layer MLP (similar to RESOLVE's hidden layers)
    auto fc1 = torch::nn::Linear(256, 128);
    auto fc2 = torch::nn::Linear(128, 64);
    auto fc3 = torch::nn::Linear(64, 1);

    SECTION("Batch=64, Input=256") {
        auto x_cpu = torch::rand({64, 256}, torch::kFloat32);

        BENCHMARK("CPU") {
            auto h = torch::relu(fc1->forward(x_cpu));
            h = torch::relu(fc2->forward(h));
            return fc3->forward(h);
        };

        if (torch::cuda::is_available()) {
            fc1->to(torch::kCUDA);
            fc2->to(torch::kCUDA);
            fc3->to(torch::kCUDA);
            auto x_cuda = x_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto h = torch::relu(fc1->forward(x_cuda));
                h = torch::relu(fc2->forward(h));
                auto result = fc3->forward(h);
                torch::cuda::synchronize();
                return result;
            };

            // Move back to CPU for other sections
            fc1->to(torch::kCPU);
            fc2->to(torch::kCPU);
            fc3->to(torch::kCPU);
        }
    }

    SECTION("Batch=256, Input=512") {
        auto fc1_big = torch::nn::Linear(512, 256);
        auto fc2_big = torch::nn::Linear(256, 128);
        auto fc3_big = torch::nn::Linear(128, 1);
        auto x_cpu = torch::rand({256, 512}, torch::kFloat32);

        BENCHMARK("CPU") {
            auto h = torch::relu(fc1_big->forward(x_cpu));
            h = torch::relu(fc2_big->forward(h));
            return fc3_big->forward(h);
        };

        if (torch::cuda::is_available()) {
            fc1_big->to(torch::kCUDA);
            fc2_big->to(torch::kCUDA);
            fc3_big->to(torch::kCUDA);
            auto x_cuda = x_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto h = torch::relu(fc1_big->forward(x_cuda));
                h = torch::relu(fc2_big->forward(h));
                auto result = fc3_big->forward(h);
                torch::cuda::synchronize();
                return result;
            };
        }
    }

    SECTION("Batch=1024, Input=1024 (large)") {
        auto fc1_large = torch::nn::Linear(1024, 512);
        auto fc2_large = torch::nn::Linear(512, 256);
        auto fc3_large = torch::nn::Linear(256, 1);
        auto x_cpu = torch::rand({1024, 1024}, torch::kFloat32);

        BENCHMARK("CPU") {
            auto h = torch::relu(fc1_large->forward(x_cpu));
            h = torch::relu(fc2_large->forward(h));
            return fc3_large->forward(h);
        };

        if (torch::cuda::is_available()) {
            fc1_large->to(torch::kCUDA);
            fc2_large->to(torch::kCUDA);
            fc3_large->to(torch::kCUDA);
            auto x_cuda = x_cpu.to(torch::kCUDA);

            BENCHMARK("GPU") {
                auto h = torch::relu(fc1_large->forward(x_cuda));
                h = torch::relu(fc2_large->forward(h));
                auto result = fc3_large->forward(h);
                torch::cuda::synchronize();
                return result;
            };
        }
    }
}

// ============================================================================
// ResolveModel Forward Pass Benchmarks
// ============================================================================

TEST_CASE("ResolveModel Forward Benchmarks", "[benchmark][model]") {
    // Create a small schema for benchmarking
    ResolveSchema schema;
    schema.n_plots = 1000;
    schema.n_species = 5000;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 500;
    schema.n_families = 100;
    schema.n_genera_vocab = 501;
    schema.n_families_vocab = 101;
    schema.targets.push_back({
        "area",
        TaskType::Regression,
        TransformType::Log1p,
        0,
        1.0f
    });

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 256;
    config.hidden_dims = {512, 256, 128, 64};  // Smaller for benchmark
    config.dropout = 0.0f;  // No dropout for deterministic benchmarks
    config.top_k = 3;
    config.n_taxonomy_slots = 3;

    ResolveModel model(schema, config);

    // Input dimensions: coords(2) + hash(256) + unknown_frac(1) = 259 continuous features
    const int64_t n_continuous = 2 + config.hash_dim + 1;

    SECTION("Batch=64") {
        auto continuous = torch::rand({64, n_continuous}, torch::kFloat32);
        auto genus_ids = torch::randint(0, 501, {64, 3}, torch::kInt64);
        auto family_ids = torch::randint(0, 101, {64, 3}, torch::kInt64);

        BENCHMARK("CPU forward") {
            return model->forward(continuous, genus_ids, family_ids);
        };

        if (torch::cuda::is_available()) {
            model->to(torch::kCUDA);
            auto cont_cuda = continuous.to(torch::kCUDA);
            auto genus_cuda = genus_ids.to(torch::kCUDA);
            auto family_cuda = family_ids.to(torch::kCUDA);

            BENCHMARK("GPU forward") {
                auto result = model->forward(cont_cuda, genus_cuda, family_cuda);
                torch::cuda::synchronize();
                return result;
            };

            model->to(torch::kCPU);
        }
    }

    SECTION("Batch=256") {
        auto continuous = torch::rand({256, n_continuous}, torch::kFloat32);
        auto genus_ids = torch::randint(0, 501, {256, 3}, torch::kInt64);
        auto family_ids = torch::randint(0, 101, {256, 3}, torch::kInt64);

        BENCHMARK("CPU forward") {
            return model->forward(continuous, genus_ids, family_ids);
        };

        if (torch::cuda::is_available()) {
            model->to(torch::kCUDA);
            auto cont_cuda = continuous.to(torch::kCUDA);
            auto genus_cuda = genus_ids.to(torch::kCUDA);
            auto family_cuda = family_ids.to(torch::kCUDA);

            BENCHMARK("GPU forward") {
                auto result = model->forward(cont_cuda, genus_cuda, family_cuda);
                torch::cuda::synchronize();
                return result;
            };

            model->to(torch::kCPU);
        }
    }

    SECTION("Batch=1024") {
        auto continuous = torch::rand({1024, n_continuous}, torch::kFloat32);
        auto genus_ids = torch::randint(0, 501, {1024, 3}, torch::kInt64);
        auto family_ids = torch::randint(0, 101, {1024, 3}, torch::kInt64);

        BENCHMARK("CPU forward") {
            return model->forward(continuous, genus_ids, family_ids);
        };

        if (torch::cuda::is_available()) {
            model->to(torch::kCUDA);
            auto cont_cuda = continuous.to(torch::kCUDA);
            auto genus_cuda = genus_ids.to(torch::kCUDA);
            auto family_cuda = family_ids.to(torch::kCUDA);

            BENCHMARK("GPU forward") {
                auto result = model->forward(cont_cuda, genus_cuda, family_cuda);
                torch::cuda::synchronize();
                return result;
            };

            model->to(torch::kCPU);
        }
    }
}

// ============================================================================
// End-to-End Training Benchmark
// ============================================================================

TEST_CASE("End-to-End Training Benchmark Production", "[benchmark][training-prod]") {
    // Production-scale benchmark (similar to ASAAS paper)
    // Real dataset: ~1.9M plots, but we use 100K for benchmark
    const int64_t n_plots = 100000;
    const int32_t hash_dim = 1024;
    const int64_t n_continuous = 2 + hash_dim + 1;

    // Create schema matching paper configuration
    ResolveSchema schema;
    schema.n_plots = n_plots;
    schema.n_species = 150000;  // Realistic species count
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 10000;
    schema.n_families = 500;
    schema.n_genera_vocab = 10001;
    schema.n_families_vocab = 501;

    // Regression target: area prediction
    schema.targets.push_back({
        "area",
        TaskType::Regression,
        TransformType::Log1p,
        0,
        1.0f
    });

    // Production model architecture
    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::Hash;
    model_config.hash_dim = hash_dim;
    model_config.hidden_dims = {2048, 1024, 512, 256};  // Production network
    model_config.dropout = 0.2f;
    model_config.top_k = 5;
    model_config.n_taxonomy_slots = 5;

    // Create synthetic training data
    auto coordinates = torch::rand({n_plots, 2}, torch::kFloat32) * 180.0f - 90.0f;
    auto hash_embedding = torch::randn({n_plots, hash_dim}, torch::kFloat32);
    auto unknown_fraction = torch::rand({n_plots}, torch::kFloat32) * 0.1f;
    auto genus_ids = torch::randint(0, 10001, {n_plots, 5}, torch::kInt64);
    auto family_ids = torch::randint(0, 501, {n_plots, 5}, torch::kInt64);

    // Synthetic target: area in m² (log-normal distribution)
    auto area = torch::exp(torch::randn({n_plots}) * 2.0f + 4.0f);
    std::unordered_map<std::string, torch::Tensor> targets;
    targets["area"] = area;

    SECTION("5 epochs CPU") {
        TrainConfig train_config;
        train_config.batch_size = 1024;
        train_config.max_epochs = 5;
        train_config.patience = 10;
        train_config.lr = 1e-3f;
        train_config.device = torch::kCPU;
        train_config.loss_config = LossConfigMode::MAE;

        BENCHMARK_ADVANCED("Training 5 epochs (100K plots, production)")(Catch::Benchmark::Chronometer meter) {
            ResolveModel fresh_model(schema, model_config);
            Trainer fresh_trainer(fresh_model, train_config);

            fresh_trainer.prepare_data(
                coordinates,
                torch::Tensor(),
                hash_embedding,
                torch::Tensor(),
                torch::Tensor(),
                genus_ids,
                family_ids,
                unknown_fraction,
                torch::Tensor(),
                targets,
                0.2f,
                42
            );

            meter.measure([&fresh_trainer] {
                return fresh_trainer.fit();
            });
        };
    }

    if (torch::cuda::is_available()) {
        SECTION("5 epochs GPU") {
            TrainConfig train_config;
            train_config.batch_size = 1024;
            train_config.max_epochs = 5;
            train_config.patience = 10;
            train_config.lr = 1e-3f;
            train_config.device = torch::kCUDA;
            train_config.loss_config = LossConfigMode::MAE;

            BENCHMARK_ADVANCED("Training 5 epochs (100K plots, production)")(Catch::Benchmark::Chronometer meter) {
                ResolveModel fresh_model(schema, model_config);
                Trainer fresh_trainer(fresh_model, train_config);

                fresh_trainer.prepare_data(
                    coordinates,
                    torch::Tensor(),
                    hash_embedding,
                    torch::Tensor(),
                    torch::Tensor(),
                    genus_ids,
                    family_ids,
                    unknown_fraction,
                    torch::Tensor(),
                    targets,
                    0.2f,
                    42
                );

                meter.measure([&fresh_trainer] {
                    auto result = fresh_trainer.fit();
                    torch::cuda::synchronize();
                    return result;
                });
            };
        }
    }
}

TEST_CASE("End-to-End Training Benchmark Large", "[benchmark][training-large]") {
    // Larger synthetic dataset for training benchmark
    const int64_t n_plots = 10000;
    const int32_t hash_dim = 512;
    const int64_t n_continuous = 2 + hash_dim + 1;  // coords + hash + unknown_frac

    // Create schema
    ResolveSchema schema;
    schema.n_plots = n_plots;
    schema.n_species = 50000;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 2000;
    schema.n_families = 200;
    schema.n_genera_vocab = 2001;
    schema.n_families_vocab = 201;
    schema.targets.push_back({
        "area",
        TaskType::Regression,
        TransformType::Log1p,
        0,
        1.0f
    });

    // Create model with larger architecture
    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::Hash;
    model_config.hash_dim = hash_dim;
    model_config.hidden_dims = {1024, 512, 256, 128};  // Larger network
    model_config.dropout = 0.1f;
    model_config.top_k = 3;
    model_config.n_taxonomy_slots = 3;

    // Create synthetic training data
    auto coordinates = torch::rand({n_plots, 2}, torch::kFloat32) * 180.0f - 90.0f;
    auto hash_embedding = torch::randn({n_plots, hash_dim}, torch::kFloat32);
    auto unknown_fraction = torch::rand({n_plots}, torch::kFloat32) * 0.1f;
    auto genus_ids = torch::randint(0, 2001, {n_plots, 3}, torch::kInt64);
    auto family_ids = torch::randint(0, 201, {n_plots, 3}, torch::kInt64);

    // Synthetic target: area in m² (log-normal distribution)
    auto area = torch::exp(torch::randn({n_plots}) * 2.0f + 4.0f);
    std::unordered_map<std::string, torch::Tensor> targets;
    targets["area"] = area;

    SECTION("10 epochs CPU") {
        ResolveModel model(schema, model_config);

        TrainConfig train_config;
        train_config.batch_size = 512;
        train_config.max_epochs = 10;
        train_config.patience = 20;  // No early stopping for benchmark
        train_config.lr = 1e-3f;
        train_config.device = torch::kCPU;
        train_config.loss_config = LossConfigMode::MAE;

        Trainer trainer(model, train_config);

        BENCHMARK_ADVANCED("Training 10 epochs (10K plots)")(Catch::Benchmark::Chronometer meter) {
            // Reset model for each run
            ResolveModel fresh_model(schema, model_config);
            Trainer fresh_trainer(fresh_model, train_config);

            fresh_trainer.prepare_data(
                coordinates,
                torch::Tensor(),  // no covariates
                hash_embedding,
                torch::Tensor(),  // no species_ids
                torch::Tensor(),  // no species_vector
                genus_ids,
                family_ids,
                unknown_fraction,
                torch::Tensor(),  // no unknown_count
                targets,
                0.2f,
                42
            );

            meter.measure([&fresh_trainer] {
                return fresh_trainer.fit();
            });
        };
    }

    if (torch::cuda::is_available()) {
        SECTION("10 epochs GPU") {
            TrainConfig train_config;
            train_config.batch_size = 512;
            train_config.max_epochs = 10;
            train_config.patience = 20;
            train_config.lr = 1e-3f;
            train_config.device = torch::kCUDA;
            train_config.loss_config = LossConfigMode::MAE;

            BENCHMARK_ADVANCED("Training 10 epochs (10K plots)")(Catch::Benchmark::Chronometer meter) {
                ResolveModel fresh_model(schema, model_config);
                Trainer fresh_trainer(fresh_model, train_config);

                fresh_trainer.prepare_data(
                    coordinates,
                    torch::Tensor(),
                    hash_embedding,
                    torch::Tensor(),
                    torch::Tensor(),
                    genus_ids,
                    family_ids,
                    unknown_fraction,
                    torch::Tensor(),
                    targets,
                    0.2f,
                    42
                );

                meter.measure([&fresh_trainer] {
                    auto result = fresh_trainer.fit();
                    torch::cuda::synchronize();
                    return result;
                });
            };
        }
    }
}

TEST_CASE("End-to-End Training Benchmark", "[benchmark][training]") {
    // Small synthetic dataset for training benchmark
    const int64_t n_plots = 500;
    const int32_t hash_dim = 128;
    const int64_t n_continuous = 2 + hash_dim + 1;  // coords + hash + unknown_frac

    // Create schema
    ResolveSchema schema;
    schema.n_plots = n_plots;
    schema.n_species = 1000;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 100;
    schema.n_families = 20;
    schema.n_genera_vocab = 101;
    schema.n_families_vocab = 21;
    schema.targets.push_back({
        "area",
        TaskType::Regression,
        TransformType::Log1p,
        0,
        1.0f
    });

    // Create model with smaller architecture
    ModelConfig model_config;
    model_config.species_encoding = SpeciesEncodingMode::Hash;
    model_config.hash_dim = hash_dim;
    model_config.hidden_dims = {256, 128, 64};  // Small network
    model_config.dropout = 0.1f;
    model_config.top_k = 3;
    model_config.n_taxonomy_slots = 3;

    // Create synthetic training data
    auto coordinates = torch::rand({n_plots, 2}, torch::kFloat32) * 180.0f - 90.0f;
    auto hash_embedding = torch::randn({n_plots, hash_dim}, torch::kFloat32);
    auto unknown_fraction = torch::rand({n_plots}, torch::kFloat32) * 0.1f;
    auto genus_ids = torch::randint(0, 101, {n_plots, 3}, torch::kInt64);
    auto family_ids = torch::randint(0, 21, {n_plots, 3}, torch::kInt64);

    // Synthetic target: area in m² (log-normal distribution)
    auto area = torch::exp(torch::randn({n_plots}) * 2.0f + 4.0f);
    std::unordered_map<std::string, torch::Tensor> targets;
    targets["area"] = area;

    SECTION("5 epochs CPU") {
        ResolveModel model(schema, model_config);

        TrainConfig train_config;
        train_config.batch_size = 64;
        train_config.max_epochs = 5;
        train_config.patience = 10;  // No early stopping for benchmark
        train_config.lr = 1e-3f;
        train_config.device = torch::kCPU;
        train_config.loss_config = LossConfigMode::MAE;

        Trainer trainer(model, train_config);

        BENCHMARK_ADVANCED("Training 5 epochs")(Catch::Benchmark::Chronometer meter) {
            // Reset model for each run
            ResolveModel fresh_model(schema, model_config);
            Trainer fresh_trainer(fresh_model, train_config);

            fresh_trainer.prepare_data(
                coordinates,
                torch::Tensor(),  // no covariates
                hash_embedding,
                torch::Tensor(),  // no species_ids
                torch::Tensor(),  // no species_vector
                genus_ids,
                family_ids,
                unknown_fraction,
                torch::Tensor(),  // no unknown_count
                targets,
                0.2f,
                42
            );

            meter.measure([&fresh_trainer] {
                return fresh_trainer.fit();
            });
        };
    }

    if (torch::cuda::is_available()) {
        SECTION("5 epochs GPU") {
            TrainConfig train_config;
            train_config.batch_size = 64;
            train_config.max_epochs = 5;
            train_config.patience = 10;
            train_config.lr = 1e-3f;
            train_config.device = torch::kCUDA;
            train_config.loss_config = LossConfigMode::MAE;

            BENCHMARK_ADVANCED("Training 5 epochs")(Catch::Benchmark::Chronometer meter) {
                ResolveModel fresh_model(schema, model_config);
                Trainer fresh_trainer(fresh_model, train_config);

                fresh_trainer.prepare_data(
                    coordinates,
                    torch::Tensor(),
                    hash_embedding,
                    torch::Tensor(),
                    torch::Tensor(),
                    genus_ids,
                    family_ids,
                    unknown_fraction,
                    torch::Tensor(),
                    targets,
                    0.2f,
                    42
                );

                meter.measure([&fresh_trainer] {
                    auto result = fresh_trainer.fit();
                    torch::cuda::synchronize();
                    return result;
                });
            };
        }
    }
}

// ============================================================================
// Training Components Benchmark (individual parts)
// ============================================================================

TEST_CASE("Training Components Benchmark", "[benchmark][components]") {
    const int64_t batch_size = 256;
    const int64_t n_continuous = 259;  // Typical RESOLVE input size

    // Create schema and model
    ResolveSchema schema;
    schema.n_plots = 1000;
    schema.n_species = 5000;
    schema.has_coordinates = true;
    schema.has_taxonomy = true;
    schema.n_genera = 500;
    schema.n_families = 100;
    schema.n_genera_vocab = 501;
    schema.n_families_vocab = 101;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::Log1p, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 256;
    config.hidden_dims = {512, 256, 128, 64};
    config.dropout = 0.0f;
    config.top_k = 3;
    config.n_taxonomy_slots = 3;

    SECTION("Forward + Backward pass") {
        ResolveModel model(schema, config);
        model->train();

        auto continuous = torch::rand({batch_size, n_continuous}, torch::kFloat32);
        auto genus_ids = torch::randint(0, 501, {batch_size, 3}, torch::kInt64);
        auto family_ids = torch::randint(0, 101, {batch_size, 3}, torch::kInt64);
        auto target = torch::rand({batch_size}, torch::kFloat32);

        BENCHMARK("CPU forward+backward") {
            model->zero_grad();
            auto output = model->forward(continuous, genus_ids, family_ids);
            auto loss = torch::mse_loss(output["area"].squeeze(), target);
            loss.backward();
            return loss;
        };

        if (torch::cuda::is_available()) {
            model->to(torch::kCUDA);
            auto cont_cuda = continuous.to(torch::kCUDA);
            auto genus_cuda = genus_ids.to(torch::kCUDA);
            auto family_cuda = family_ids.to(torch::kCUDA);
            auto target_cuda = target.to(torch::kCUDA);

            BENCHMARK("GPU forward+backward") {
                model->zero_grad();
                auto output = model->forward(cont_cuda, genus_cuda, family_cuda);
                auto loss = torch::mse_loss(output["area"].squeeze(), target_cuda);
                loss.backward();
                torch::cuda::synchronize();
                return loss;
            };

            model->to(torch::kCPU);
        }
    }

    SECTION("Optimizer step") {
        ResolveModel model(schema, config);
        model->train();
        auto optimizer = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(1e-3));

        auto continuous = torch::rand({batch_size, n_continuous}, torch::kFloat32);
        auto genus_ids = torch::randint(0, 501, {batch_size, 3}, torch::kInt64);
        auto family_ids = torch::randint(0, 101, {batch_size, 3}, torch::kInt64);
        auto target = torch::rand({batch_size}, torch::kFloat32);

        BENCHMARK("CPU forward+backward+step") {
            optimizer.zero_grad();
            auto output = model->forward(continuous, genus_ids, family_ids);
            auto loss = torch::mse_loss(output["area"].squeeze(), target);
            loss.backward();
            optimizer.step();
            return loss;
        };

        if (torch::cuda::is_available()) {
            model->to(torch::kCUDA);
            auto optimizer_cuda = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(1e-3));
            auto cont_cuda = continuous.to(torch::kCUDA);
            auto genus_cuda = genus_ids.to(torch::kCUDA);
            auto family_cuda = family_ids.to(torch::kCUDA);
            auto target_cuda = target.to(torch::kCUDA);

            BENCHMARK("GPU forward+backward+step") {
                optimizer_cuda.zero_grad();
                auto output = model->forward(cont_cuda, genus_cuda, family_cuda);
                auto loss = torch::mse_loss(output["area"].squeeze(), target_cuda);
                loss.backward();
                optimizer_cuda.step();
                torch::cuda::synchronize();
                return loss;
            };
        }
    }
}
