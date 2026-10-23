#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/dataset.hpp"
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"
#include "resolve/plot_inputs.hpp"
#include "resolve/sparse_rows.hpp"
#include "test_support.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <sstream>
#include <vector>

using namespace resolve;
using namespace resolve::test;

// ============================================================================
// Sparse rows: the layout of the sparse encoding's species vector
//
// Every row operation and product is checked against the dense tensor op it
// stands for, so the sparse path is the dense path without the zeros.
// ============================================================================

namespace {

// A (rows, cols) matrix with about `density` of its cells non-zero, every
// value positive, and some rows empty.
torch::Tensor sparse_matrix(int64_t rows, int64_t cols, double density, uint64_t seed) {
    torch::manual_seed(seed);
    auto keep = (torch::rand({rows, cols}) < density).to(torch::kFloat32);
    keep.select(0, 1).zero_();
    return keep * (torch::rand({rows, cols}) + 0.5f);
}

void require_same_rows(const torch::Tensor& sparse, const torch::Tensor& dense) {
    REQUIRE(is_sparse_rows(sparse));
    REQUIRE(sparse.sizes() == dense.sizes());
    REQUIRE(torch::equal(sparse_rows_to_dense(sparse).cpu(), dense.cpu()));
}

}  // namespace

TEST_CASE("A dense matrix round-trips through sparse rows", "[sparse_rows]") {
    const auto dense = sparse_matrix(7, 11, 0.3, 1);
    const auto rows = to_sparse_rows(dense);
    REQUIRE(is_sparse_rows(rows));
    REQUIRE_FALSE(is_sparse_rows(dense));
    REQUIRE(rows.crow_indices().numel() == 8);
    REQUIRE(rows.col_indices().numel() == (dense != 0).sum().item<int64_t>());
    // Row 1 is empty.
    REQUIRE(rows.crow_indices()[1].item<int64_t>() == rows.crow_indices()[2].item<int64_t>());
    require_same_rows(rows, dense);
    // Already sparse rows: returned as is.
    REQUIRE(to_sparse_rows(rows).col_indices().data_ptr() == rows.col_indices().data_ptr());
    // Columns increase within each row.
    const auto crow = rows.crow_indices();
    const auto col = rows.col_indices();
    for (int64_t r = 0; r < 7; ++r) {
        for (int64_t i = crow[r].item<int64_t>() + 1; i < crow[r + 1].item<int64_t>(); ++i) {
            REQUIRE(col[i - 1].item<int64_t>() < col[i].item<int64_t>());
        }
    }
    REQUIRE_THROWS_AS(require_sparse_rows(dense, "x"), std::invalid_argument);
}

TEST_CASE("Row selection, slicing and concatenation match the dense ops",
          "[sparse_rows]") {
    const auto dense = sparse_matrix(20, 9, 0.35, 2);
    const auto rows = to_sparse_rows(dense);

    SECTION("select, with repeats and in any order") {
        const auto pick = torch::tensor(std::vector<int64_t>{5, 1, 5, 19, 0, 1});
        require_same_rows(select_rows(rows, pick), dense.index_select(0, pick));
        require_same_rows(select_rows(rows, torch::empty({0}, torch::kInt64)),
                          dense.index_select(0, torch::empty({0}, torch::kInt64)));
        // A dense tensor takes the dense op.
        REQUIRE(torch::equal(select_rows(dense, pick), dense.index_select(0, pick)));
    }
    SECTION("slice, clamped like a dense slice") {
        require_same_rows(slice_rows(rows, 3, 11), dense.slice(0, 3, 11));
        require_same_rows(slice_rows(rows, 15, 40), dense.slice(0, 15, 40));
        require_same_rows(slice_rows(rows, 6, 6), dense.slice(0, 6, 6));
        require_same_rows(slice_rows(rows, 0, 20), dense);
    }
    SECTION("concatenate") {
        const auto joined = concat_rows({slice_rows(rows, 0, 4), slice_rows(rows, 4, 4),
                                         slice_rows(rows, 4, 20)});
        require_same_rows(joined, dense);
        REQUIRE_THROWS(concat_rows({rows, dense}));
        REQUIRE_THROWS(concat_rows({rows, to_sparse_rows(sparse_matrix(3, 4, 0.5, 3))}));
    }
}

TEST_CASE("The sparse products equal the dense ones", "[sparse_rows]") {
    const auto dense = sparse_matrix(16, 30, 0.2, 4);
    const auto rows = to_sparse_rows(dense);

    SECTION("rows @ table, gradients included") {
        torch::manual_seed(5);
        auto table = torch::randn({30, 6}, torch::requires_grad());
        auto table_ref = table.detach().clone().requires_grad_(true);
        const auto got = sparse_rows_matmul(rows, table);
        const auto want = torch::matmul(dense, table_ref);
        REQUIRE(torch::allclose(got, want, 1e-5, 1e-6));
        got.sum().backward();
        want.sum().backward();
        REQUIRE(torch::allclose(table.grad(), table_ref.grad(), 1e-5, 1e-6));
    }
    SECTION("a linear layer over sparse rows is torch::nn::Linear over the matrix") {
        torch::manual_seed(6);
        torch::nn::Linear linear(30, 8);
        const auto want = linear->forward(dense);
        const auto got = sparse_rows_linear(rows, linear->weight, linear->bias);
        REQUIRE(torch::allclose(got, want, 1e-5, 1e-6));
        got.pow(2).sum().backward();
        const auto weight_grad = linear->weight.grad().clone();
        const auto bias_grad = linear->bias.grad().clone();
        linear->zero_grad();
        linear->forward(dense).pow(2).sum().backward();
        REQUIRE(torch::allclose(weight_grad, linear->weight.grad(), 1e-5, 1e-6));
        REQUIRE(torch::allclose(bias_grad, linear->bias.grad(), 1e-5, 1e-6));
    }
    SECTION("row normalization") {
        const auto want = dense / (dense.sum(1, true) + 1e-8);
        REQUIRE(torch::allclose(sparse_rows_to_dense(sparse_rows_normalized(rows, 1e-8)), want,
                                1e-6, 1e-7));
    }
    SECTION("new values on the same entries") {
        const auto doubled = sparse_rows_with_values(rows, rows.values() * 2);
        require_same_rows(doubled, dense * 2);
        REQUIRE_THROWS(sparse_rows_with_values(rows, torch::ones({1})));
    }
}

TEST_CASE("PlotInputs carries a sparse species vector through every row operation",
          "[sparse_rows][plot_inputs]") {
    const auto dense = sparse_matrix(10, 12, 0.3, 7);
    PlotInputs in;
    in.continuous = torch::arange(10, torch::kFloat32).unsqueeze(1);
    in.species_vector = to_sparse_rows(dense);

    const auto pick = torch::tensor(std::vector<int64_t>{9, 2, 4});
    const auto picked = in.index_select(pick);
    require_same_rows(picked.species_vector, dense.index_select(0, pick));
    REQUIRE(picked.n_plots() == 3);

    require_same_rows(in.slice(2, 7).species_vector, dense.slice(0, 2, 7));
    const auto joined = PlotInputs::cat({in.slice(0, 5), in.slice(5, 10)});
    require_same_rows(joined.species_vector, dense);
    require_same_rows(in.to(torch::kCPU).species_vector, dense);

    SECTION("a dense species vector given to a binding becomes sparse rows") {
        PlotInputs raw;
        raw.species_vector = dense.to(torch::kFloat64);
        raw.species_ids = torch::zeros({10, 2}, torch::kInt32);
        const auto conformed = raw.conformed();
        require_same_rows(conformed.species_vector, dense.to(torch::kFloat32));
        REQUIRE(conformed.species_vector.scalar_type() == torch::kFloat32);
        REQUIRE(conformed.species_ids.scalar_type() == torch::kInt64);
        REQUIRE_FALSE(conformed.genus_ids.defined());
    }
}

TEST_CASE("Sparse rows work on the GPU", "[sparse_rows][cuda]") {
    if (!torch::cuda::is_available()) SKIP("no CUDA device");
    const torch::Device gpu(torch::kCUDA, 0);
    const auto dense = sparse_matrix(40, 25, 0.25, 8);
    const auto rows = to_sparse_rows(dense).to(gpu);
    REQUIRE(rows.is_cuda());
    const auto pick = torch::tensor(std::vector<int64_t>{3, 39, 3, 0});
    require_same_rows(select_rows(rows, pick), dense.index_select(0, pick));
    require_same_rows(slice_rows(rows, 10, 30), dense.slice(0, 10, 30));
    require_same_rows(concat_rows({slice_rows(rows, 0, 10), slice_rows(rows, 10, 40)}), dense);
    torch::manual_seed(9);
    const auto table = torch::randn({25, 5});
    REQUIRE(torch::allclose(sparse_rows_matmul(rows, table.to(gpu)).cpu(),
                            torch::matmul(dense, table), 1e-5, 1e-6));
}

TEST_CASE("The sparse encoder reads sparse rows and refuses a dense matrix",
          "[sparse_rows][model]") {
    torch::manual_seed(10);
    PlotEncoderSparse encoder(/*n_continuous=*/3, /*n_species=*/20, /*species_embed_dim=*/4,
                              /*n_genera=*/0, /*n_families=*/0, /*genus_emb_dim=*/0,
                              /*family_emb_dim=*/0, /*top_k=*/0, std::vector<int64_t>{8},
                              /*dropout=*/0.0f);
    encoder->eval();
    const auto continuous = torch::randn({6, 3});
    const auto dense = sparse_matrix(6, 20, 0.3, 11);
    const auto out = encoder->forward(continuous, to_sparse_rows(dense));
    REQUIRE(out.size(0) == 6);
    REQUIRE(torch::isfinite(out).all().item<bool>());
    REQUIRE_THROWS(encoder->forward(continuous, dense));
}

TEST_CASE("A sparse-encoded dataset stores each plot's species as one sparse row",
          "[sparse_rows][dataset]") {
    // P1 records sp_b twice (summed), sp_c at 0 (not stored) and sp_x, which a
    // vocabulary fitted on other data does not hold.
    TempFile header("plot_id,y\nP1,1\nP2,2\n");
    TempFile species(
        "plot_id,sp,cover\n"
        "P1,sp_b,2\nP1,sp_a,1\nP1,sp_b,3\nP1,sp_c,0\n"
        "P2,sp_a,4\nP2,sp_c,1\n");
    DatasetConfig cfg;
    cfg.species_encoding = SpeciesEncodingMode::Sparse;
    cfg.track_unknown_fraction = false;
    const auto ds = ResolveDataset::from_csv(header.path(), species.path(), Roles(),
                                             {TargetSpec::regression("y")}, cfg);
    const auto& rows = ds.species_vector();
    REQUIRE(is_sparse_rows(rows));
    REQUIRE(rows.size(0) == 2);
    REQUIRE(rows.size(1) == ds.schema().n_species_vocab);

    const auto& vocab = ds.species_vocab();
    auto code = [&](const std::string& name) {
        return static_cast<int64_t>(std::find(vocab.begin(), vocab.end(), name) - vocab.begin());
    };
    const auto matrix = sparse_rows_to_dense(rows);
    REQUIRE(matrix[0][code("sp_b")].item<float>() == 5.0f);
    REQUIRE(matrix[0][code("sp_a")].item<float>() == 1.0f);
    REQUIRE(matrix[1][code("sp_a")].item<float>() == 4.0f);
    REQUIRE(matrix[1][code("sp_c")].item<float>() == 1.0f);
    // P1 stores sp_a and sp_b only: the zero cover stores nothing.
    REQUIRE(rows.crow_indices()[1].item<int64_t>() == 2);
    REQUIRE(rows.col_indices().numel() == 4);
}
