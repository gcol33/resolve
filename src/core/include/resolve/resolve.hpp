#pragma once

// RESOLVE: C++ core library using libtorch
// Predict plot-level environmental variables from species composition

#include "resolve/types.hpp"
#include "resolve/species_encoder.hpp"
#include "resolve/encoder.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/predictor.hpp"
#include "resolve/loss.hpp"

namespace resolve {

constexpr const char* VERSION = "0.1.0";

} // namespace resolve
