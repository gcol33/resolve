#pragma once

// TabNet (Arik & Pfister, "TabNet: Attentive Interpretable Tabular Learning",
// AAAI 2021, arXiv:1908.07442) and the sparse simplex mappings its attentive
// transformer uses.

#include <torch/torch.h>

#include <cstdint>

namespace resolve {

// =============================================================================
// Sparsemax and Entmax
// =============================================================================

// Sparsemax: the Euclidean projection onto the probability simplex, which
// produces exact zeros.
torch::Tensor sparsemax(torch::Tensor input, int64_t dim = -1);

// Entmax-1.5: the alpha = 1.5 member of the alpha-entmax family
//     alpha-entmax(z) = [ (alpha - 1) z - tau * 1 ]_+ ^ (1/(alpha - 1))
// (Peters, Niculae & Martins, "Sparse Sequence-to-Sequence Models", ACL 2019,
// arXiv:1905.05702, Eq. 13), i.e. [ z/2 - tau ]_+^2 here. alpha = 1 is softmax
// and alpha = 2 is sparsemax, so 1.5-entmax is sparse but keeps a strictly
// larger support than sparsemax on the same scores.
//
// Computed exactly by the paper's sort-based Algorithm 2 (never by bisection,
// which is the general-alpha `entmax_bisect` routine), with the closed-form
// Proposition 1 Jacobian as the backward. Differentiable, and reduces over an
// arbitrary `dim` of an arbitrarily strided input. See cpp_src/tabnet.cpp for
// the transcribed algorithm and gradient.
torch::Tensor entmax15(torch::Tensor input, int64_t dim = -1);

// =============================================================================
// Ghost batch normalization
// =============================================================================

// Normalize each slice of `virtual_batch_size` rows with `bn` independently
// instead of the whole batch at once (Arik & Pfister, section 3.2, which
// applies it to every block of the feature and attentive transformers -- the
// initial input normalization stays full-batch). One module, so the affine
// parameters and running statistics are shared across the slices and the
// parameter names are the plain BatchNorm1d ones.
//
// A `virtual_batch_size` of 0 (or one no smaller than the batch) normalizes the
// batch in one piece, i.e. plain batch normalization. In eval mode the running
// statistics already describe the training distribution, so the split would
// change nothing and is skipped. A trailing slice of one row would make
// BatchNorm1d throw ("Expected more than 1 value per channel"), so it is folded
// into the slice before it. Every slice is normalized through run_norm_fp32,
// so the statistics stay float32 under autocast.
torch::Tensor apply_ghost_batch_norm(torch::nn::BatchNorm1d bn,
                                     torch::Tensor x,
                                     int64_t virtual_batch_size);

// =============================================================================
// TabNet blocks
// =============================================================================

// GLU block of the TabNet feature transformer: FC -> BN -> GLU. The FC produces
// 2*out_dim; the gated linear unit halves it back to out_dim (Fig. 4).
class TabNetGLUBlockImpl : public torch::nn::Module {
public:
    TabNetGLUBlockImpl(int64_t in_dim, int64_t out_dim,
                       int64_t virtual_batch_size = 0);
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t virtual_batch_size_;

    torch::nn::Linear fc_{nullptr};
    torch::nn::BatchNorm1d bn_{nullptr};
};

TORCH_MODULE(TabNetGLUBlock);

// One TabNet decision step. Holds the attentive transformer (produces a sparse
// feature mask over the ORIGINAL features from the previous step's attention
// split and the prior scale) and the step-specific ("independent") half of the
// feature transformer. The shared half is owned by the encoder and reused
// across steps.
class TabNetStepImpl : public torch::nn::Module {
public:
    TabNetStepImpl(
        int64_t input_dim,
        int64_t n_d,            // Decision layer dimension
        int64_t n_a,            // Attention layer dimension
        int64_t n_independent,  // Step-specific feature-transformer GLU blocks
        bool use_sparsemax = true,  // sparsemax (alpha=2) vs 1.5-entmax mask
        int64_t virtual_batch_size = 0  // Ghost batch norm slice; 0 = full batch
    );

    // att_prev: (batch, n_a) previous step's attention split.
    // prior_scales: (batch, input_dim) feature availability.
    // Returns the simplex mask over features: (batch, input_dim), produced by
    // sparsemax when use_sparsemax, otherwise by entmax15.
    torch::Tensor attentive_forward(torch::Tensor att_prev, torch::Tensor prior_scales);

    // Apply this step's independent GLU blocks after the encoder's shared blocks.
    // shared_out and the return are (batch, n_d + n_a).
    torch::Tensor feature_independent(torch::Tensor shared_out);

    [[nodiscard]] bool use_sparsemax() const noexcept { return use_sparsemax_; }

private:
    bool use_sparsemax_;
    int64_t virtual_batch_size_;

    torch::nn::Linear attention_fc_{nullptr};
    torch::nn::BatchNorm1d bn_attention_{nullptr};
    torch::nn::ModuleList independent_{nullptr};
};

TORCH_MODULE(TabNetStep);

// What one TabNet forward produces.
struct TabNetOutput {
    torch::Tensor output;        // (batch, n_d), the aggregated decision output
    // The entropy of a step's feature mask, summed over features and averaged
    // over rows and steps: Arik & Pfister's sparsity regularizer L_sparse,
    // which TabNetConfig::sparsity_coefficient weights into the training
    // objective.
    torch::Tensor mask_entropy;  // scalar
};

class TabNetEncoderImpl : public torch::nn::Module {
public:
    TabNetEncoderImpl(
        int64_t input_dim,
        int64_t n_steps = 3,
        int64_t n_d = 64,
        int64_t n_a = 64,
        float relaxation_factor = 1.5f,
        bool use_sparsemax = true,  // TabNetConfig::use_sparsemax
        int64_t virtual_batch_size = 0  // TabNetConfig::virtual_batch_size
    );

    // x: (batch, input_dim)
    TabNetOutput forward(torch::Tensor x);

    [[nodiscard]] int64_t output_dim() const noexcept { return n_d_; }

    // Which simplex mapping every step's attentive transformer applies.
    [[nodiscard]] bool use_sparsemax() const noexcept { return use_sparsemax_; }

    // The ghost batch norm slice every block below the initial normalization
    // runs at. 0 = plain batch normalization.
    [[nodiscard]] int64_t virtual_batch_size() const noexcept {
        return virtual_batch_size_;
    }

private:
    // Run the shared feature-transformer GLU blocks (input_dim -> n_d + n_a),
    // with sqrt(0.5) residual scaling between blocks after the first.
    torch::Tensor run_shared(torch::Tensor x) const;

    int64_t input_dim_;
    int64_t n_steps_;
    int64_t n_d_;
    int64_t n_a_;
    float relaxation_factor_;
    bool use_sparsemax_;
    int64_t virtual_batch_size_;

    torch::nn::BatchNorm1d initial_bn_{nullptr};  // Input feature batch norm
    torch::nn::ModuleList shared_{nullptr};       // Shared feature-transformer blocks
    torch::nn::ModuleList steps_{nullptr};
};

TORCH_MODULE(TabNetEncoder);

}  // namespace resolve
