#pragma once

#include "resolve/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

namespace resolve {

// MurmurHash3 finalizer for feature hashing
uint32_t murmur_hash(const std::string& key, uint32_t seed = 0);

// Feature hashing for species
void hash_species(
    const std::vector<std::pair<std::string, float>>& species_abundances,
    float* embedding,
    int hash_dim
);

// Select top-k species by abundance
std::vector<std::pair<std::string, float>> select_top_k(
    std::vector<std::pair<std::string, float>> species,
    int k
);

// Select bottom-k species by abundance
std::vector<std::pair<std::string, float>> select_bottom_k(
    std::vector<std::pair<std::string, float>> species,
    int k
);

// Apply selection mode to species list
std::vector<std::pair<std::string, float>> apply_selection(
    std::vector<std::pair<std::string, float>> species,
    SelectionMode mode,
    int k
);

// Apply normalization to abundance values
void apply_normalization(
    std::vector<std::pair<std::string, float>>& species,
    NormalizationMode mode
);

} // namespace resolve
