#include "resolve/species_encoding.hpp"
#include <algorithm>
#include <cmath>

namespace resolve {

uint32_t murmur_hash(const std::string& key, uint32_t seed) {
    uint32_t h = seed;
    for (char c : key) {
        h ^= static_cast<uint32_t>(c);
        h *= 0x5bd1e995;
        h ^= h >> 15;
    }
    return h;
}

void hash_species(
    const std::vector<std::pair<std::string, float>>& species_abundances,
    float* embedding,
    int hash_dim
) {
    std::fill(embedding, embedding + hash_dim, 0.0f);

    for (const auto& [species, abundance] : species_abundances) {
        uint32_t h1 = murmur_hash(species, 0);
        uint32_t h2 = murmur_hash(species, 1);

        int idx = h1 % hash_dim;
        float sign = (h2 % 2 == 0) ? 1.0f : -1.0f;
        embedding[idx] += sign * abundance;
    }
}

std::vector<std::pair<std::string, float>> select_top_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    if (static_cast<int>(species.size()) <= k) {
        return species;
    }

    std::partial_sort(
        species.begin(),
        species.begin() + k,
        species.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; }
    );

    species.resize(k);
    return species;
}

std::vector<std::pair<std::string, float>> select_bottom_k(
    std::vector<std::pair<std::string, float>> species,
    int k
) {
    if (static_cast<int>(species.size()) <= k) {
        return species;
    }

    std::partial_sort(
        species.begin(),
        species.begin() + k,
        species.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    species.resize(k);
    return species;
}

std::vector<std::pair<std::string, float>> apply_selection(
    std::vector<std::pair<std::string, float>> species,
    SelectionMode mode,
    int k
) {
    switch (mode) {
        case SelectionMode::Top:
            return select_top_k(std::move(species), k);
        case SelectionMode::Bottom:
            return select_bottom_k(std::move(species), k);
        case SelectionMode::TopBottom: {
            auto top = select_top_k(species, k);
            auto bottom = select_bottom_k(species, k);
            // Merge, avoiding duplicates
            for (const auto& s : bottom) {
                if (std::find_if(top.begin(), top.end(),
                        [&s](const auto& x) { return x.first == s.first; }) == top.end()) {
                    top.push_back(s);
                }
            }
            return top;
        }
        case SelectionMode::All:
        default:
            return species;
    }
}

void apply_normalization(
    std::vector<std::pair<std::string, float>>& species,
    NormalizationMode mode
) {
    if (mode == NormalizationMode::Norm) {
        float total = 0.0f;
        for (const auto& [sp, ab] : species) total += ab;
        if (total > 0) {
            for (auto& [sp, ab] : species) ab /= total;
        }
    } else if (mode == NormalizationMode::Log1p) {
        for (auto& [sp, ab] : species) ab = std::log1p(ab);
    }
}

} // namespace resolve
