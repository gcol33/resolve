#pragma once

#include "resolve/types.hpp"

namespace resolve {

// How each hidden layer of an encoder's MLP tail is built: Linear, then the
// normalization, the activation and dropout, optionally with a skip connection
// around the block. The plain MLP, the mixture's backbone and the TabM ensemble
// read it alike.
struct MLPBlockConfig {
    ActivationType activation = ActivationType::GELU;
    NormLayerType normalization = NormLayerType::BatchNorm;
    int norm_groups = kDefaultNormGroups;
    float dropout = kDefaultDropout;
    bool use_residual = false;
    float leaky_relu_slope = kDefaultLeakyReLUSlope;
    float elu_alpha = kDefaultELUAlpha;

    static MLPBlockConfig from_model_config(const ModelConfig& cfg) {
        MLPBlockConfig block;
        block.activation = cfg.activation;
        block.normalization = cfg.normalization;
        block.norm_groups = cfg.norm_groups;
        block.dropout = cfg.dropout;
        block.use_residual = cfg.use_residual;
        block.leaky_relu_slope = cfg.leaky_relu_slope;
        block.elu_alpha = cfg.elu_alpha;
        return block;
    }
};

}  // namespace resolve
