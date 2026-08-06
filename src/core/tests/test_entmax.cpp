#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/attention.hpp"
#include "resolve/checkpoint.hpp"
#include "resolve/types.hpp"

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// 1.5-entmax (gcol33/resolve#103)
//
// Reference: Peters, Niculae & Martins, "Sparse Sequence-to-Sequence Models",
// ACL 2019 (arXiv:1905.05702).
//   Eq. 13         alpha-entmax(z) = [ (alpha-1) z - tau 1 ]_+ ^ (1/(alpha-1))
//   Algorithm 2    exact sort-based solution at alpha = 1.5
//   Proposition 1  d alpha-entmax(z)/dz = diag(s) - s s^T / ||s||_1,
//                  s_i = (p*_i)^(2-alpha) on the support, 0 elsewhere
// ============================================================================

namespace {

// Number of strictly positive entries in each row of a (1, n) or (batch, n)
// simplex projection -- the support size.
int64_t support_size(const torch::Tensor& p, int64_t row = 0) {
    return (p[row] > 0).to(torch::kLong).sum().item<int64_t>();
}

}  // namespace

// ----------------------------------------------------------------------------
// Pinned values, hand-derived from Eq. 13 / Algorithm 2
// ----------------------------------------------------------------------------

TEST_CASE("entmax15 matches hand-computed Algorithm 2 values", "[entmax]") {
    SECTION("z = [1, 0, 0] -- full support, dense") {
        // Algorithm 2 line 1: alpha-entmax is shift invariant, so center on the
        // max and apply the (alpha - 1) = 1/2 rescaling of Eq. 13:
        //     z' = ([1, 0, 0] - 1) / 2 = [0, -1/2, -1/2],
        // already in descending order.
        //   rho = 1: M = 0,     S = 0,
        //            tau = 0 - sqrt((1 - 0)/1) = -1
        //   rho = 2: M = -1/4,  S = 2 (1/4)^2 = 1/8,
        //            tau = -1/4 - sqrt((1 - 1/8)/2) = -0.911437...
        //   rho = 3: M = -1/3,  S = (1/3)^2 + 2 (1/6)^2 = 1/9 + 1/18 = 1/6,
        //            tau = -1/3 - sqrt((1 - 1/6)/3) = -1/3 - sqrt(10)/6
        //                = -0.86037961...
        // Line 6 selects rho* = 3: tau(3) <= z'_[3] = -1/2 and there is no
        // z'_[4]. Line 7:
        //     p_1 = (0        + 0.86037961)^2 = 0.74025307
        //     p_2 = p_3 = (-1/2 + 0.86037961)^2 = 0.12987346
        //     sum  = 0.74025307 + 2 (0.12987346) = 1
        auto z = torch::tensor({{1.0f, 0.0f, 0.0f}});
        auto p = entmax15(z, /*dim=*/-1);

        REQUIRE_THAT(p[0][0].item<float>(), WithinAbs(0.74025307f, 1e-5));
        REQUIRE_THAT(p[0][1].item<float>(), WithinAbs(0.12987346f, 1e-5));
        REQUIRE_THAT(p[0][2].item<float>(), WithinAbs(0.12987346f, 1e-5));
        REQUIRE_THAT(p.sum().item<float>(), WithinAbs(1.0f, 1e-5));

        // The pre-fix implementation solved [z - tau]_+^2 (no (alpha - 1)
        // factor), which on this input saturates to the one-hot [1, 0, 0] --
        // i.e. it reproduced sparsemax, not 1.5-entmax.
        auto sm = sparsemax(z, /*dim=*/-1);
        REQUIRE_THAT(sm[0][0].item<float>(), WithinAbs(1.0f, 1e-5));
        REQUIRE(p[0][1].item<float>() > 0.1f);
    }

    SECTION("z = [5, 0, 0] -- saturated, support size 1") {
        // z' = ([5, 0, 0] - 5)/2 = [0, -5/2, -5/2].
        //   rho = 1: M = 0, S = 0, tau = -1, and line 6 holds:
        //            z'_[2] = -5/2 <= -1 <= 0 = z'_[1].
        // Line 7: p = [(0 + 1)^2, 0, 0] = [1, 0, 0].
        auto z = torch::tensor({{5.0f, 0.0f, 0.0f}});
        auto p = entmax15(z, /*dim=*/-1);

        REQUIRE_THAT(p[0][0].item<float>(), WithinAbs(1.0f, 1e-5));
        REQUIRE_THAT(p[0][1].item<float>(), WithinAbs(0.0f, 1e-6));
        REQUIRE_THAT(p[0][2].item<float>(), WithinAbs(0.0f, 1e-6));
        REQUIRE(support_size(p) == 1);
    }

    SECTION("z = [2, 1, 0, -1] -- partial support of size 2") {
        // z' = ([2, 1, 0, -1] - 2)/2 = [0, -1/2, -1, -3/2].
        //   rho = 2: M = -1/4, S = 1/8,
        //            tau = -1/4 - sqrt((1 - 1/8)/2) = -1/4 - sqrt(7)/4
        //                = -0.91143783
        // Line 6 holds at rho* = 2: z'_[3] = -1 <= tau <= -1/2 = z'_[2].
        //     p_1 = (0    + 0.91143783)^2 = 0.83071891
        //     p_2 = (-1/2 + 0.91143783)^2 = 0.16928109
        //     p_3 = p_4 = 0
        auto z = torch::tensor({{2.0f, 1.0f, 0.0f, -1.0f}});
        auto p = entmax15(z, /*dim=*/-1);

        REQUIRE_THAT(p[0][0].item<float>(), WithinAbs(0.83071891f, 1e-5));
        REQUIRE_THAT(p[0][1].item<float>(), WithinAbs(0.16928109f, 1e-5));
        REQUIRE_THAT(p[0][2].item<float>(), WithinAbs(0.0f, 1e-6));
        REQUIRE_THAT(p[0][3].item<float>(), WithinAbs(0.0f, 1e-6));
        REQUIRE_THAT(p.sum().item<float>(), WithinAbs(1.0f, 1e-5));
    }
}

// ----------------------------------------------------------------------------
// Simplex invariants
// ----------------------------------------------------------------------------

TEST_CASE("entmax15 output is a non-negative distribution", "[entmax]") {
    torch::manual_seed(103);

    SECTION("random batch sums to 1 along dim and is non-negative") {
        auto z = torch::randn({32, 48}) * 3.0f;
        auto p = entmax15(z, /*dim=*/-1);

        auto sums = p.sum(/*dim=*/-1);
        REQUIRE_THAT((sums - 1.0f).abs().max().item<float>(), WithinAbs(0.0f, 1e-5));
        REQUIRE(p.min().item<float>() >= 0.0f);
    }

    SECTION("single-element rows collapse to 1") {
        auto z = torch::randn({7, 1});
        auto p = entmax15(z, /*dim=*/-1);
        REQUIRE_THAT((p - 1.0f).abs().max().item<float>(), WithinAbs(0.0f, 1e-6));
    }

    SECTION("shift invariance: adding a constant to a row changes nothing") {
        auto z = torch::randn({8, 20}) * 2.0f;
        auto shift = torch::randn({8, 1}) * 5.0f;
        auto p = entmax15(z, /*dim=*/-1);
        auto p_shifted = entmax15(z + shift, /*dim=*/-1);
        REQUIRE_THAT((p - p_shifted).abs().max().item<float>(), WithinAbs(0.0f, 1e-5));
    }
}

TEST_CASE("entmax15 sits between softmax and sparsemax in sparsity", "[entmax]") {
    // The alpha-entmax support size is non-increasing in alpha, so on identical
    // logits: support(softmax, alpha=1) >= support(entmax15) >= support(sparsemax,
    // alpha=2).
    SECTION("worked example, all three strictly ordered") {
        auto z = torch::tensor({{2.0f, 1.0f, 0.0f, -1.0f}});

        auto soft = torch::softmax(z, /*dim=*/-1);
        auto ent = entmax15(z, /*dim=*/-1);
        auto sparse = sparsemax(z, /*dim=*/-1);

        REQUIRE(support_size(soft) == 4);    // softmax is never sparse
        REQUIRE(support_size(ent) == 2);     // hand-derived above
        REQUIRE(support_size(sparse) == 1);  // tau = 1 -> [1, 0, 0, 0]
    }

    SECTION("random batch, ordering holds row by row") {
        torch::manual_seed(20260806);
        auto z = torch::randn({64, 32}) * 2.5f;

        auto ent = entmax15(z, /*dim=*/-1);
        auto sparse = sparsemax(z, /*dim=*/-1);

        auto n_ent = (ent > 0).to(torch::kLong).sum(/*dim=*/-1);
        auto n_sparse = (sparse > 0).to(torch::kLong).sum(/*dim=*/-1);

        REQUIRE((n_sparse <= n_ent).all().item<bool>());
        REQUIRE((n_ent <= 32).all().item<bool>());
        // And it is a real gap on this draw, not a tie everywhere.
        REQUIRE(n_ent.sum().item<int64_t>() > n_sparse.sum().item<int64_t>());
    }
}

// ----------------------------------------------------------------------------
// Gradient
// ----------------------------------------------------------------------------

TEST_CASE("entmax15 gradient flows and is finite", "[entmax]") {
    torch::manual_seed(5);
    auto z = torch::randn({16, 24}) * 2.0f;
    z.requires_grad_(true);
    auto loss = entmax15(z, /*dim=*/-1).pow(2).sum();
    loss.backward();

    REQUIRE(z.grad().defined());
    REQUIRE(z.grad().sizes() == z.sizes());
    REQUIRE(torch::isfinite(z.grad()).all().item<bool>());
    // Rows are not gradient-dead: at least some entries move.
    REQUIRE(z.grad().abs().max().item<float>() > 0.0f);
}

TEST_CASE("entmax15 backward matches central finite differences", "[entmax]") {
    // Proposition 1 gives the analytic Jacobian; the custom autograd Function
    // implements it directly, so check it against the numerical derivative of
    // the forward. Both rows sit well away from a support boundary (row 0 has
    // support 3 of 4 with the nearest kink ~0.026 away in the rescaled scores,
    // row 1 has full support), so the piecewise operator is smooth at h = 1e-6.
    auto z0 = torch::tensor({{1.5, 0.4, -0.3, -1.1},
                             {0.8, 0.5, 0.2, -0.4}}, torch::dtype(torch::kDouble));
    auto w = torch::tensor({{0.30, -1.20, 0.70, 2.10},
                            {-0.45, 0.90, 1.60, -0.25}}, torch::dtype(torch::kDouble));

    auto z = z0.clone();
    z.requires_grad_(true);
    auto loss = (entmax15(z, /*dim=*/-1) * w).sum();
    loss.backward();
    auto analytic = z.grad().clone();

    const double h = 1e-6;
    for (int64_t r = 0; r < z0.size(0); ++r) {
        for (int64_t c = 0; c < z0.size(1); ++c) {
            auto zp = z0.clone();
            zp[r][c] += h;
            auto zm = z0.clone();
            zm[r][c] -= h;

            const double fp = (entmax15(zp, /*dim=*/-1) * w).sum().item<double>();
            const double fm = (entmax15(zm, /*dim=*/-1) * w).sum().item<double>();
            const double numeric = (fp - fm) / (2.0 * h);

            REQUIRE_THAT(analytic[r][c].item<double>(), WithinAbs(numeric, 1e-6));
        }
    }
}

TEST_CASE("entmax15 backward equals the Proposition 1 Jacobian directly", "[entmax]") {
    // dz = s . dp - (<s, dp> / ||s||_1) s, with s = sqrt(p*) at alpha = 1.5.
    torch::manual_seed(11);
    auto z0 = torch::randn({5, 9}, torch::dtype(torch::kDouble));
    auto dp = torch::randn({5, 9}, torch::dtype(torch::kDouble));

    auto z = z0.clone();
    z.requires_grad_(true);
    auto p = entmax15(z, /*dim=*/-1);
    p.backward(dp);

    auto s = entmax15(z0, /*dim=*/-1).sqrt();
    auto ds = dp * s;
    auto expected = ds - (ds.sum(-1, true) / s.sum(-1, true)) * s;

    REQUIRE_THAT((z.grad() - expected).abs().max().item<double>(),
                 WithinAbs(0.0, 1e-10));
}

// ----------------------------------------------------------------------------
// Layout: arbitrary dim, non-contiguous input
// ----------------------------------------------------------------------------

TEST_CASE("entmax15 reduces over an arbitrary dim", "[entmax]") {
    torch::manual_seed(7);
    auto z = torch::randn({4, 6, 5}) * 2.0f;

    SECTION("dim = 1 sums to 1 along dim 1") {
        auto p = entmax15(z, /*dim=*/1);
        REQUIRE(p.sizes() == z.sizes());
        auto sums = p.sum(/*dim=*/1);
        REQUIRE_THAT((sums - 1.0f).abs().max().item<float>(), WithinAbs(0.0f, 1e-5));
    }

    SECTION("negative and positive dim index agree") {
        auto a = entmax15(z, /*dim=*/-1);
        auto b = entmax15(z, /*dim=*/2);
        REQUIRE(torch::equal(a, b));
    }

    SECTION("dim = 0 matches the transposed last-dim result") {
        auto a = entmax15(z, /*dim=*/0);
        auto b = entmax15(z.permute({1, 2, 0}).contiguous(), /*dim=*/-1)
                     .permute({2, 0, 1});
        REQUIRE_THAT((a - b).abs().max().item<float>(), WithinAbs(0.0f, 1e-6));
    }

    SECTION("out-of-range dim is rejected") {
        REQUIRE_THROWS(entmax15(z, /*dim=*/3));
        REQUIRE_THROWS(entmax15(z, /*dim=*/-4));
    }
}

TEST_CASE("entmax15 handles non-contiguous input", "[entmax]") {
    torch::manual_seed(13);
    auto base = torch::randn({6, 10, 4}) * 2.0f;

    SECTION("transposed view") {
        auto view = base.transpose(0, 2);  // (4, 10, 6), non-contiguous
        REQUIRE_FALSE(view.is_contiguous());

        auto from_view = entmax15(view, /*dim=*/1);
        auto from_copy = entmax15(view.contiguous(), /*dim=*/1);
        REQUIRE_THAT((from_view - from_copy).abs().max().item<float>(),
                     WithinAbs(0.0f, 1e-6));
        auto sums = from_view.sum(/*dim=*/1);
        REQUIRE_THAT((sums - 1.0f).abs().max().item<float>(), WithinAbs(0.0f, 1e-5));
    }

    SECTION("strided slice") {
        auto view = base.slice(/*dim=*/1, 0, 10, /*step=*/2);  // (6, 5, 4)
        REQUIRE_FALSE(view.is_contiguous());

        auto from_view = entmax15(view, /*dim=*/1);
        auto from_copy = entmax15(view.contiguous(), /*dim=*/1);
        REQUIRE_THAT((from_view - from_copy).abs().max().item<float>(),
                     WithinAbs(0.0f, 1e-6));
    }
}

// ----------------------------------------------------------------------------
// TabNetConfig::use_sparsemax is a live knob
// ----------------------------------------------------------------------------

TEST_CASE("TabNetStep attentive mask follows use_sparsemax", "[entmax][attention]") {
    // Both steps are built from the same seed, so their attentive-transformer
    // weights are identical and the two masks come from the SAME logits. The
    // support ordering is then the operator-level guarantee, not an empirical
    // observation.
    const int64_t input_dim = 24, n_a = 12;

    torch::manual_seed(909);
    TabNetStep step_sparse(input_dim, 8, n_a, 2, /*use_sparsemax=*/true);
    torch::manual_seed(909);
    TabNetStep step_entmax(input_dim, 8, n_a, 2, /*use_sparsemax=*/false);

    REQUIRE(step_sparse->use_sparsemax());
    REQUIRE_FALSE(step_entmax->use_sparsemax());

    step_sparse->eval();
    step_entmax->eval();

    // Scale the attention split so the logits are well spread; sparsemax is then
    // firmly in its sparse regime and the support gap is unambiguous.
    torch::manual_seed(4);
    auto att_prev = torch::randn({16, n_a}) * 8.0f;
    auto prior_scales = torch::ones({16, input_dim});

    auto mask_sparse = step_sparse->attentive_forward(att_prev, prior_scales);
    auto mask_entmax = step_entmax->attentive_forward(att_prev, prior_scales);

    // Both are simplex projections.
    REQUIRE_THAT((mask_sparse.sum(-1) - 1.0f).abs().max().item<float>(),
                 WithinAbs(0.0f, 1e-5));
    REQUIRE_THAT((mask_entmax.sum(-1) - 1.0f).abs().max().item<float>(),
                 WithinAbs(0.0f, 1e-5));

    // The knob is live: identical weights, identical input, different masks.
    REQUIRE_FALSE(torch::allclose(mask_sparse, mask_entmax, 1e-4, 1e-6));

    auto n_sparse = (mask_sparse > 0).to(torch::kLong).sum();
    auto n_entmax = (mask_entmax > 0).to(torch::kLong).sum();
    REQUIRE(n_entmax.item<int64_t>() > n_sparse.item<int64_t>());
}

TEST_CASE("TabNetEncoder forwards and trains at both use_sparsemax settings",
          "[entmax][attention]") {
    const int64_t input_dim = 16, batch = 64;

    auto build = [&](bool use_sparsemax, int64_t n_steps) {
        torch::manual_seed(2026);
        return TabNetEncoder(input_dim, n_steps, /*n_d=*/8, /*n_a=*/8,
                             /*relaxation_factor=*/1.5f,
                             /*sparsity_coefficient=*/1e-3f, use_sparsemax);
    };

    SECTION("accessor reports the configured mapping") {
        REQUIRE(build(true, 2)->use_sparsemax());
        REQUIRE_FALSE(build(false, 2)->use_sparsemax());
    }

    SECTION("single-step feature importance is the raw mask and differs") {
        // At n_steps = 1 the returned feature_importance IS the one attentive
        // mask, produced from logits that are identical across the two settings
        // (same seeded weights, prior_scales still all ones). The support
        // ordering is therefore the operator-level guarantee.
        torch::manual_seed(31);
        auto x = torch::randn({batch, input_dim}) * 1.5f;

        auto enc_sparse = build(true, 1);
        auto enc_entmax = build(false, 1);
        enc_sparse->eval();
        enc_entmax->eval();

        auto imp_s = enc_sparse->forward(x).second;
        auto imp_e = enc_entmax->forward(x).second;

        REQUIRE(imp_s.sizes() == imp_e.sizes());
        // Each is still a per-row distribution.
        REQUIRE_THAT((imp_s.sum(-1) - 1.0f).abs().max().item<float>(),
                     WithinAbs(0.0f, 1e-4));
        REQUIRE_THAT((imp_e.sum(-1) - 1.0f).abs().max().item<float>(),
                     WithinAbs(0.0f, 1e-4));

        // The knob is live, and entmax never drops a feature sparsemax kept.
        REQUIRE_FALSE(torch::allclose(imp_s, imp_e, 1e-4, 1e-6));
        auto n_s = (imp_s > 0).to(torch::kLong).sum(-1);
        auto n_e = (imp_e > 0).to(torch::kLong).sum(-1);
        REQUIRE((n_s <= n_e).all().item<bool>());
    }

    SECTION("multi-step forwards differ between the two settings") {
        torch::manual_seed(97);
        auto x = torch::randn({batch, input_dim}) * 1.5f;

        auto enc_sparse = build(true, 3);
        auto enc_entmax = build(false, 3);
        enc_sparse->eval();
        enc_entmax->eval();

        auto out_s = enc_sparse->forward(x);
        auto out_e = enc_entmax->forward(x);

        REQUIRE(out_s.first.sizes() == out_e.first.sizes());
        REQUIRE(torch::isfinite(out_s.first).all().item<bool>());
        REQUIRE(torch::isfinite(out_e.first).all().item<bool>());
        REQUIRE_FALSE(torch::allclose(out_s.second, out_e.second, 1e-4, 1e-6));
    }

    SECTION("both settings train") {
        auto train_once = [&](bool use_sparsemax) {
            auto encoder = build(use_sparsemax, 2);
            torch::manual_seed(77);
            torch::nn::Linear head(8, 1);

            std::vector<torch::Tensor> params = encoder->parameters();
            auto head_params = head->parameters();
            params.insert(params.end(), head_params.begin(), head_params.end());
            torch::optim::Adam opt(params, torch::optim::AdamOptions(1e-2));

            torch::manual_seed(555);
            auto x = torch::randn({batch, input_dim});
            auto w = torch::randn({input_dim, 1});
            auto y = x.mm(w);

            std::vector<double> losses;
            for (int step = 0; step < 60; ++step) {
                opt.zero_grad();
                auto latent = encoder->forward(x).first;
                auto loss = torch::mse_loss(head->forward(latent), y);
                loss.backward();
                opt.step();
                losses.push_back(loss.item<double>());
            }
            REQUIRE(losses.size() == static_cast<std::size_t>(60));

            // Compare windowed means so a single noisy step cannot decide it.
            double first = 0.0, last = 0.0;
            for (std::size_t i = 0; i < 5; ++i) {
                first += losses[i];
                last += losses[losses.size() - 1 - i];
            }
            REQUIRE(std::isfinite(first));
            REQUIRE(std::isfinite(last));
            REQUIRE(last < first);
        };

        train_once(/*use_sparsemax=*/true);
        train_once(/*use_sparsemax=*/false);
    }
}

TEST_CASE("use_sparsemax = false round-trips through the checkpoint", "[entmax][checkpoint]") {
    ModelConfig cfg;
    cfg.encoder_architecture = EncoderArchitecture::TabNet;
    cfg.tabnet.n_steps = 2;
    cfg.tabnet.n_d = 8;
    cfg.tabnet.n_a = 8;
    cfg.tabnet.use_sparsemax = false;  // non-default

    const std::string path =
        (std::filesystem::temp_directory_path() / "resolve_entmax_tabnet_cfg.pt").string();
    {
        torch::serialize::OutputArchive ar;
        save_model_config(ar, cfg);
        ar.save_to(path);
    }
    torch::serialize::InputArchive ar;
    ar.load_from(path);
    ModelConfig loaded = load_model_config(ar);
    std::filesystem::remove(path);

    REQUIRE(loaded.encoder_architecture == EncoderArchitecture::TabNet);
    REQUIRE(loaded.tabnet.n_steps == 2);
    REQUIRE_FALSE(loaded.tabnet.use_sparsemax);

    // A model rebuilt from the reloaded config must actually run entmax: it
    // matches an explicitly entmax encoder built from the same seed, and does
    // not match the sparsemax one.
    const int64_t input_dim = 16;
    auto make = [&](bool use_sparsemax) {
        torch::manual_seed(1905);
        auto enc = TabNetEncoder(input_dim, loaded.tabnet.n_steps, loaded.tabnet.n_d,
                                 loaded.tabnet.n_a, loaded.tabnet.relaxation_factor,
                                 loaded.tabnet.sparsity_coefficient, use_sparsemax);
        enc->eval();
        return enc;
    };

    torch::manual_seed(1905);
    auto rebuilt = TabNetEncoder(input_dim, loaded.tabnet.n_steps, loaded.tabnet.n_d,
                                 loaded.tabnet.n_a, loaded.tabnet.relaxation_factor,
                                 loaded.tabnet.sparsity_coefficient,
                                 loaded.tabnet.use_sparsemax);
    rebuilt->eval();
    REQUIRE_FALSE(rebuilt->use_sparsemax());

    torch::manual_seed(5702);
    auto x = torch::randn({32, input_dim}) * 1.5f;

    auto imp_rebuilt = rebuilt->forward(x).second;
    auto imp_entmax = make(false)->forward(x).second;
    auto imp_sparse = make(true)->forward(x).second;

    REQUIRE(torch::allclose(imp_rebuilt, imp_entmax, 1e-5, 1e-7));
    REQUIRE_FALSE(torch::allclose(imp_rebuilt, imp_sparse, 1e-4, 1e-6));
}
