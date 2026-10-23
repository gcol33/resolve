#pragma once

// Auxiliary objective terms: what a model forward adds to the training loss
// beyond the target losses. Each term is named and already weighted by its
// coefficient, so the trainer adds the values as they come and records each
// term's per-epoch mean under its name.

#include <torch/torch.h>

#include <map>
#include <string>

namespace resolve {

// Term name -> weighted scalar. Ordered by name, so every consumer lists the
// terms in one order.
using AuxLosses = std::map<std::string, torch::Tensor>;

// The mixture-of-experts load-balancing term, weighted by
// ModelConfig::moe_aux_loss_weight.
inline constexpr const char* kMoEAuxLoss = "moe";

// TabNet's feature-mask sparsity regularizer, weighted by
// TabNetConfig::sparsity_coefficient.
inline constexpr const char* kTabNetSparsityLoss = "tabnet_sparsity";

}  // namespace resolve
