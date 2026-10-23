#include "resolve/tabnet.hpp"
#include "resolve/encoder.hpp"  // run_norm_fp32

#include <torch/autograd.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace resolve {

// =============================================================================
// Sparsemax Implementation
// =============================================================================

torch::Tensor sparsemax(torch::Tensor input, int64_t dim) {
    // Move dim to last position for easier processing
    auto original_dim = dim;
    if (dim < 0) {
        dim = input.dim() + dim;
    }

    // Transpose if needed
    if (dim != input.dim() - 1) {
        input = input.transpose(dim, -1);
    }

    auto original_shape = input.sizes().vec();
    auto n = input.size(-1);

    // Flatten to 2D: (batch, n)
    input = input.reshape({-1, n});

    // Sort in descending order
    auto [sorted, _] = input.sort(/*dim=*/-1, /*descending=*/true);

    // Compute cumulative sum
    auto cumsum = sorted.cumsum(/*dim=*/-1);

    // Compute k(z): number of elements in support
    auto range = torch::arange(1, n + 1, input.options());
    auto bound = 1.0f + range * sorted;
    auto is_in_support = (cumsum < bound).to(torch::kFloat32);
    auto k = is_in_support.sum(/*dim=*/-1, /*keepdim=*/true);

    // Compute tau (threshold). tau = (cumsum_{k} - 1) / k, where cumsum_{k} is
    // the cumulative sum AT the support boundary index k -- a single gathered
    // value, not the sum of every in-support cumsum.
    auto k_idx = (k.to(torch::kLong) - 1).clamp_min(0);
    auto cumsum_k = cumsum.gather(/*dim=*/-1, k_idx);
    auto tau = (cumsum_k - 1.0f) / k;

    // Compute output
    auto output = (input - tau).clamp_min(0.0f);

    // Reshape back
    output = output.reshape(original_shape);

    // Transpose back if needed
    if (original_dim < 0) {
        original_dim = static_cast<int64_t>(original_shape.size()) + original_dim;
    }
    if (original_dim != static_cast<int64_t>(original_shape.size()) - 1) {
        output = output.transpose(original_dim, -1);
    }

    return output;
}

// =============================================================================
// Entmax-1.5 Implementation
// =============================================================================
//
// alpha-entmax (Peters, Niculae & Martins, "Sparse Sequence-to-Sequence
// Models", ACL 2019; arXiv:1905.05702) is the family of simplex mappings
//
//     alpha-entmax(z) := argmax_{p in simplex} p^T z + H^T_alpha(p)     (Eq. 10)
//
// over the Tsallis alpha-entropies, with softmax (alpha = 1) and sparsemax
// (alpha = 2) as its endpoints. The solution has the closed form
//
//     alpha-entmax(z) = [ (alpha - 1) z - tau * 1 ]_+ ^ (1/(alpha - 1))  (Eq. 13)
//
// where tau is the unique threshold making the entries sum to one; entries with
// z_j <= tau/(alpha - 1) get exactly zero. At alpha = 1.5 this reads
// [ z/2 - tau ]_+^2. The (alpha - 1) = 1/2 rescaling of z is what separates
// 1.5-entmax from a plain squared-hinge simplex projection: dropping it runs the
// operator at a different temperature and collapses its support towards
// sparsemax's.
//
// alpha = 1.5 admits an EXACT sort-based solution -- Algorithm 2 of the paper
// (Sec. 3.3, derived in Appendix C.2) -- reproduced here verbatim:
//
//     Algorithm 2  Compute 1.5-entmax(z) exactly
//     1  Sort z, yielding z_[d] <= ... <= z_[1]; set z <- z/2
//     2  for rho in 1, ..., d do
//     3      M(rho) <- 1/rho sum_{j=1}^{rho} z_[j]
//     4      S(rho) <- sum_{j=1}^{rho} (z_[j] - M(rho))^2
//     5      tau(rho) <- M(rho) - sqrt( 1/rho (1 - S(rho)) )
//     6      if z_[rho+1] <= tau(rho) <= z_[rho] then
//     7          return p* = [z - tau * 1]_+^2
//
// which is what the reference implementation (github.com/deep-spin/entmax,
// `entmax15`) computes. The paper's Algorithm 1 (bisection) is the
// general-alpha routine (`entmax_bisect`) and is deliberately not implemented
// here: alpha = 1.5 is the only value the engine uses, and Algorithm 2 solves
// it exactly and in closed form.
//
// The backward pass uses the closed-form Jacobian of Proposition 1 (Sec. 3.4):
// with s_i = (p*_i)^(2 - alpha) on the support and 0 elsewhere -- so
// s = sqrt(p*) at alpha = 1.5 --
//
//     d alpha-entmax(z) / dz = diag(s) - (1 / ||s||_1) s s^T.
//
// It is a hand-written autograd Function rather than autograd through the
// sort/sqrt because line 5's discriminant has to be clamped for the rho past
// the support, where sqrt's derivative is not finite.

namespace {

// Algorithm 2, lines 1-6, vectorized over rho and over every slice along `dim`.
// `z_half` is (alpha - 1) * z = z / 2 (line 1). Returns tau (keepdim) and the
// support size rho*.
std::pair<torch::Tensor, torch::Tensor> entmax15_threshold(
    const torch::Tensor& z_half, int64_t dim
) {
    const int64_t n = z_half.size(dim);

    // Line 1: z_[1] >= ... >= z_[d].
    auto sorted = std::get<0>(z_half.sort(dim, /*descending=*/true));

    // rho = 1, ..., d, laid out along `dim` so it broadcasts against `sorted`.
    std::vector<int64_t> rho_shape(static_cast<size_t>(z_half.dim()), 1);
    rho_shape[static_cast<size_t>(dim)] = n;
    auto rho = torch::arange(1, n + 1, z_half.options()).view(rho_shape);

    // Lines 3-4. S(rho) = sum_{j<=rho} (z_[j] - M(rho))^2 = rho (E[z^2] - E[z]^2)
    // over the top-rho prefix.
    auto mean = sorted.cumsum(dim) / rho;
    auto mean_sq = sorted.square().cumsum(dim) / rho;
    auto ss = rho * (mean_sq - mean.square());

    // Line 5. Past the true support 1 - S(rho) turns negative; clamping keeps
    // sqrt finite, and those rho fail the line-6 test regardless.
    auto delta = ((1.0 - ss) / rho).clamp_min(0.0);
    auto tau_all = mean - delta.sqrt();

    // Line 6. tau(rho) <= z_[rho] holds exactly for rho = 1, ..., rho*, so
    // counting it recovers rho* (the paired z_[rho*+1] <= tau(rho*) then follows
    // from the descending order).
    auto support = (tau_all <= sorted).to(torch::kLong).sum(dim, /*keepdim=*/true);
    auto tau = tau_all.gather(dim, (support - 1).clamp_min(0));
    return {tau, support};
}

class Entmax15Function : public torch::autograd::Function<Entmax15Function> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor input,
        int64_t dim
    ) {
        // alpha-entmax is invariant to adding a constant to every score, so
        // centering on the row max is free and keeps the squaring in range.
        auto shifted = input - std::get<0>(input.max(dim, /*keepdim=*/true));

        // Eq. 13 at alpha = 1.5: the (alpha - 1) = 1/2 rescaling of z (line 1).
        auto z_half = shifted / 2.0;

        // Line 7.
        auto tau = entmax15_threshold(z_half, dim).first;
        auto output = (z_half - tau).clamp_min(0.0).square();

        ctx->save_for_backward({output});
        ctx->saved_data["dim"] = dim;
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        const auto saved = ctx->get_saved_variables();
        const auto& p = saved[0];
        const int64_t dim = ctx->saved_data["dim"].toInt();

        // Proposition 1: s_i = (p*_i)^(2 - alpha) = sqrt(p*_i) at alpha = 1.5,
        // and 0 off the support -- which sqrt(0) = 0 already gives. The Jacobian
        // diag(s) - s s^T / ||s||_1 applied to dY is s . dY - <s, dY>/||s||_1 s.
        auto s = p.sqrt();
        auto ds = grad_outputs[0] * s;
        auto q = ds.sum(dim, /*keepdim=*/true) / s.sum(dim, /*keepdim=*/true);
        return {ds - q * s, torch::Tensor()};
    }
};

}  // namespace

torch::Tensor entmax15(torch::Tensor input, int64_t dim) {
    const int64_t ndim = input.dim();
    TORCH_CHECK(ndim > 0, "entmax15: input must have at least 1 dimension");
    TORCH_CHECK(input.is_floating_point(),
        "entmax15: input must be a floating-point tensor, got ",
        input.scalar_type());
    TORCH_CHECK(dim >= -ndim && dim < ndim,
        "entmax15: dim ", dim, " is out of range for a ", ndim, "-D input");
    if (dim < 0) {
        dim += ndim;
    }

    // The threshold search squares the scores and differences near-equal
    // cumulative moments; run it at >= float32 and hand the result back in the
    // caller's dtype so an autocast fp16 call site stays well conditioned.
    const auto dtype = input.scalar_type();
    const bool promote = (dtype == torch::kHalf || dtype == torch::kBFloat16);

    auto output =
        Entmax15Function::apply(promote ? input.to(torch::kFloat) : input, dim);
    return promote ? output.to(dtype) : output;
}


// =============================================================================
// Ghost batch normalization
// =============================================================================

namespace {

torch::Tensor batch_norm_fp32(torch::nn::BatchNorm1d& bn, const torch::Tensor& x) {
    return run_norm_fp32([&bn](torch::Tensor t) { return bn->forward(t); }, x);
}

}  // namespace

torch::Tensor apply_ghost_batch_norm(torch::nn::BatchNorm1d bn,
                                     torch::Tensor x,
                                     int64_t virtual_batch_size) {
    const int64_t n_rows = x.size(0);
    if (virtual_batch_size <= 0 || !bn->is_training() ||
        n_rows <= virtual_batch_size) {
        return batch_norm_fp32(bn, x);
    }

    // Slice sizes, with a trailing single row folded into the slice before it so
    // no slice reaches BatchNorm1d with one row.
    std::vector<int64_t> sizes;
    for (int64_t start = 0; start < n_rows; start += virtual_batch_size) {
        sizes.push_back(std::min(virtual_batch_size, n_rows - start));
    }
    if (sizes.size() > 1 && sizes.back() == 1) {
        sizes.pop_back();
        sizes.back() += 1;
    }

    std::vector<torch::Tensor> normalized;
    normalized.reserve(sizes.size());
    int64_t at = 0;
    for (int64_t size : sizes) {
        normalized.push_back(batch_norm_fp32(bn, x.narrow(/*dim=*/0, at, size)));
        at += size;
    }
    return torch::cat(normalized, /*dim=*/0);
}

// =============================================================================
// TabNet blocks
// =============================================================================

namespace {
constexpr int64_t kTabNetNShared = 2;       // Shared feature-transformer blocks
constexpr int64_t kTabNetNIndependent = 2;  // Step-specific blocks
constexpr float kMaskEntropyEps = 1e-15f;   // log(0) guard in the mask entropy

// x + block(x), rescaled by sqrt(0.5) so the residual sum keeps its variance.
torch::Tensor scaled_residual(const torch::Tensor& x, const torch::Tensor& block_out) {
    return (x + block_out) * std::sqrt(0.5f);
}
}  // namespace

TabNetGLUBlockImpl::TabNetGLUBlockImpl(int64_t in_dim, int64_t out_dim,
                                       int64_t virtual_batch_size)
    : virtual_batch_size_(virtual_batch_size) {
    fc_ = register_module("fc", torch::nn::Linear(in_dim, 2 * out_dim));
    bn_ = register_module("bn", torch::nn::BatchNorm1d(2 * out_dim));
}

torch::Tensor TabNetGLUBlockImpl::forward(torch::Tensor x) {
    // GLU halves the last dimension: out = a * sigmoid(b) with [a,b] = FC(x).
    auto normalized = apply_ghost_batch_norm(bn_, fc_->forward(x),
                                             virtual_batch_size_);
    return torch::glu(normalized, /*dim=*/1);
}

TabNetStepImpl::TabNetStepImpl(
    int64_t input_dim, int64_t n_d, int64_t n_a, int64_t n_independent,
    bool use_sparsemax, int64_t virtual_batch_size
) : use_sparsemax_(use_sparsemax), virtual_batch_size_(virtual_batch_size) {
    // Attentive transformer: maps the previous attention split (n_a) to a
    // per-feature logit (input_dim), batch-normalized, then masked by the prior
    // scale and projected onto the simplex by `attentive_forward`.
    attention_fc_ = register_module("attention_fc", torch::nn::Linear(n_a, input_dim));
    bn_attention_ = register_module("bn_attention", torch::nn::BatchNorm1d(input_dim));

    // Step-specific feature-transformer GLU blocks (dim n_d + n_a throughout).
    independent_ = register_module("independent", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_independent; ++i) {
        independent_->push_back(
            TabNetGLUBlock(n_d + n_a, n_d + n_a, virtual_batch_size));
    }
}

torch::Tensor TabNetStepImpl::attentive_forward(
    torch::Tensor att_prev, torch::Tensor prior_scales
) {
    auto logits = apply_ghost_batch_norm(
        bn_attention_, attention_fc_->forward(att_prev), virtual_batch_size_);
    logits = logits * prior_scales;  // Prior scale down features already used.
    // TabNetConfig::use_sparsemax picks the sparse simplex mapping: sparsemax
    // (alpha = 2, the TabNet default) or 1.5-entmax, which is strictly less
    // aggressive and keeps a larger mask support.
    return use_sparsemax_ ? sparsemax(logits, /*dim=*/-1)
                          : entmax15(logits, /*dim=*/-1);
}

torch::Tensor TabNetStepImpl::feature_independent(torch::Tensor h) {
    for (const auto& block : *independent_) {
        h = scaled_residual(h, block->as<TabNetGLUBlockImpl>()->forward(h));
    }
    return h;
}

// =============================================================================
// TabNet Encoder
// =============================================================================

TabNetEncoderImpl::TabNetEncoderImpl(
    int64_t input_dim,
    int64_t n_steps,
    int64_t n_d,
    int64_t n_a,
    float relaxation_factor,
    bool use_sparsemax,
    int64_t virtual_batch_size
) : input_dim_(input_dim), n_steps_(n_steps), n_d_(n_d), n_a_(n_a),
    relaxation_factor_(relaxation_factor),
    use_sparsemax_(use_sparsemax), virtual_batch_size_(virtual_batch_size) {
    TORCH_CHECK(n_steps > 0, "TabNet needs at least one decision step, got ", n_steps);

    // Batch-normalize the raw input features. This one is full-batch: ghost
    // batch normalization applies to every block BELOW it, and the input
    // itself is normalized over the whole batch (Sec. 3.2).
    initial_bn_ = register_module("initial_bn", torch::nn::BatchNorm1d(input_dim));

    // Shared feature-transformer GLU blocks, reused across every step. The first
    // maps input_dim -> n_d + n_a; the rest keep n_d + n_a.
    shared_ = register_module("shared", torch::nn::ModuleList());
    for (int64_t i = 0; i < kTabNetNShared; ++i) {
        int64_t in_dim = (i == 0) ? input_dim : (n_d + n_a);
        shared_->push_back(TabNetGLUBlock(in_dim, n_d + n_a, virtual_batch_size));
    }

    // Decision steps (attentive transformer + independent feature-transformer).
    steps_ = register_module("steps", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_steps; ++i) {
        steps_->push_back(TabNetStep(input_dim, n_d, n_a, kTabNetNIndependent,
                                     use_sparsemax, virtual_batch_size));
    }
}

torch::Tensor TabNetEncoderImpl::run_shared(torch::Tensor x) const {
    torch::Tensor h = shared_->ptr(0)->as<TabNetGLUBlockImpl>()->forward(x);
    for (size_t i = 1; i < shared_->size(); ++i) {
        h = scaled_residual(h, shared_->ptr(i)->as<TabNetGLUBlockImpl>()->forward(h));
    }
    return h;
}

TabNetOutput TabNetEncoderImpl::forward(torch::Tensor x) {
    const int64_t batch_size = x.size(0);

    // Original (batch-normalized) features. These are what each step's mask is
    // applied to -- masking is per-step over the ORIGINAL features (M[i] * f),
    // not cumulative over a running product of masks.
    auto features = batch_norm_fp32(initial_bn_, x);

    // Seed the first attention split a[0] from an initial (shared) feature
    // transform over the full features.
    auto att = run_shared(features).slice(/*dim=*/1, n_d_, n_d_ + n_a_);

    // All features equally available at the start.
    auto prior_scales = torch::ones({batch_size, input_dim_}, features.options());
    auto aggregated_output = torch::zeros({batch_size, n_d_}, features.options());
    auto entropy = torch::zeros({batch_size}, features.options());

    for (int64_t step = 0; step < n_steps_; ++step) {
        auto step_module = steps_->ptr(step)->as<TabNetStepImpl>();

        // Attentive transformer -> sparse simplex mask over the original
        // features (sparsemax or 1.5-entmax, per use_sparsemax_).
        auto mask = step_module->attentive_forward(att, prior_scales);

        // Relaxation: features selected now are less available to later steps.
        prior_scales = prior_scales * (relaxation_factor_ - mask);

        // Feature transformer over the masked original features.
        auto ft = step_module->feature_independent(run_shared(mask * features));
        aggregated_output = aggregated_output + torch::relu(ft.slice(/*dim=*/1, 0, n_d_));
        att = ft.slice(/*dim=*/1, n_d_, n_d_ + n_a_);

        entropy = entropy + (-mask * torch::log(mask + kMaskEntropyEps)).sum(/*dim=*/-1);
    }

    return {aggregated_output, entropy.mean() / static_cast<double>(n_steps_)};
}

}  // namespace resolve
