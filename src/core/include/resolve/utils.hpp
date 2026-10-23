#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

namespace resolve {

// Half-cosine ramp: 1 at progress 0, 0 at progress 1, clamped outside [0, 1].
// The shared shape behind the LR cosine-annealing schedule (Trainer) and the
// JEPA EMA-decay schedule (JEPAPretrainer). std::numbers keeps the arithmetic
// in float end to end.
inline float cosine_ramp(float progress) noexcept {
    progress = std::clamp(progress, 0.0f, 1.0f);
    return 0.5f * (1.0f + std::cos(std::numbers::pi_v<float> * progress));
}

} // namespace resolve
