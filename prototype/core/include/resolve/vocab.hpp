#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace resolve {

/**
 * Vocabulary mapping for species IDs.
 *
 * Index 0 is reserved for unknown/padding.
 * Provides mapping from species ID strings to integer indices for nn.Embedding.
 */
class SpeciesVocab {
public:
    SpeciesVocab() = default;

    explicit SpeciesVocab(std::unordered_map<std::string, int64_t> species_to_id)
        : species_to_id_(std::move(species_to_id)) {}

    /**
     * Number of species including unknown (index 0).
     */
    int64_t n_species() const {
        return static_cast<int64_t>(species_to_id_.size()) + 1;
    }

    /**
     * Encode species ID to integer. Returns 0 for unknown.
     */
    int64_t encode(const std::string& species_id) const {
        auto it = species_to_id_.find(species_id);
        return it != species_to_id_.end() ? it->second : 0;
    }

    /**
     * Encode batch of species IDs.
     */
    std::vector<int64_t> encode_batch(const std::vector<std::string>& species_ids) const {
        std::vector<int64_t> result;
        result.reserve(species_ids.size());
        for (const auto& id : species_ids) {
            result.push_back(encode(id));
        }
        return result;
    }

    /**
     * Build vocabulary from species data.
     *
     * @param species_ids Vector of all species IDs (with duplicates)
     * @param min_count Minimum occurrences to include in vocab (default 1 = all)
     */
    static SpeciesVocab from_species_data(
        const std::vector<std::string>& species_ids,
        int min_count = 1
    ) {
        // Count occurrences
        std::unordered_map<std::string, int> counts;
        for (const auto& id : species_ids) {
            if (!id.empty()) {
                counts[id]++;
            }
        }

        // Filter by min_count and sort alphabetically
        std::vector<std::string> filtered;
        for (const auto& [id, count] : counts) {
            if (count >= min_count) {
                filtered.push_back(id);
            }
        }
        std::sort(filtered.begin(), filtered.end());

        // Build mapping (1-indexed, 0 = unknown)
        std::unordered_map<std::string, int64_t> species_to_id;
        for (size_t i = 0; i < filtered.size(); ++i) {
            species_to_id[filtered[i]] = static_cast<int64_t>(i + 1);
        }

        return SpeciesVocab(std::move(species_to_id));
    }

    /**
     * Save vocabulary to JSON file.
     */
    void save(const std::string& path) const {
        nlohmann::json j;
        j["species_to_id"] = species_to_id_;
        std::ofstream f(path);
        if (!f) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        f << j.dump(2);
    }

    /**
     * Load vocabulary from JSON file.
     */
    static SpeciesVocab load(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("Cannot open file for reading: " + path);
        }
        nlohmann::json j;
        f >> j;
        auto species_to_id = j["species_to_id"].get<std::unordered_map<std::string, int64_t>>();
        return SpeciesVocab(std::move(species_to_id));
    }

    const std::unordered_map<std::string, int64_t>& mapping() const {
        return species_to_id_;
    }

private:
    std::unordered_map<std::string, int64_t> species_to_id_;
};


/**
 * Vocabulary mapping for genus and family names.
 *
 * Index 0 is reserved for unknown/padding.
 */
class TaxonomyVocab {
public:
    TaxonomyVocab() = default;

    TaxonomyVocab(
        std::unordered_map<std::string, int64_t> genus_to_id,
        std::unordered_map<std::string, int64_t> family_to_id
    ) : genus_to_id_(std::move(genus_to_id)),
        family_to_id_(std::move(family_to_id)) {}

    int64_t n_genera() const {
        return static_cast<int64_t>(genus_to_id_.size()) + 1;
    }

    int64_t n_families() const {
        return static_cast<int64_t>(family_to_id_.size()) + 1;
    }

    int64_t encode_genus(const std::string& genus) const {
        if (genus.empty()) return 0;
        auto it = genus_to_id_.find(genus);
        return it != genus_to_id_.end() ? it->second : 0;
    }

    int64_t encode_family(const std::string& family) const {
        if (family.empty()) return 0;
        auto it = family_to_id_.find(family);
        return it != family_to_id_.end() ? it->second : 0;
    }

    /**
     * Build vocabulary from species data.
     */
    static TaxonomyVocab from_species_data(
        const std::vector<std::string>& genera,
        const std::vector<std::string>& families
    ) {
        // Collect unique values
        std::set<std::string> unique_genera(genera.begin(), genera.end());
        std::set<std::string> unique_families(families.begin(), families.end());
        unique_genera.erase("");  // Remove empty strings
        unique_families.erase("");

        // Build mappings (alphabetically sorted, 1-indexed)
        std::unordered_map<std::string, int64_t> genus_to_id;
        std::unordered_map<std::string, int64_t> family_to_id;

        int64_t idx = 1;
        for (const auto& g : unique_genera) {
            genus_to_id[g] = idx++;
        }

        idx = 1;
        for (const auto& f : unique_families) {
            family_to_id[f] = idx++;
        }

        return TaxonomyVocab(std::move(genus_to_id), std::move(family_to_id));
    }

    void save(const std::string& path) const {
        nlohmann::json j;
        j["genus_to_id"] = genus_to_id_;
        j["family_to_id"] = family_to_id_;
        std::ofstream f(path);
        if (!f) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        f << j.dump(2);
    }

    static TaxonomyVocab load(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("Cannot open file for reading: " + path);
        }
        nlohmann::json j;
        f >> j;
        auto genus_to_id = j["genus_to_id"].get<std::unordered_map<std::string, int64_t>>();
        auto family_to_id = j["family_to_id"].get<std::unordered_map<std::string, int64_t>>();
        return TaxonomyVocab(std::move(genus_to_id), std::move(family_to_id));
    }

    const std::unordered_map<std::string, int64_t>& genus_mapping() const {
        return genus_to_id_;
    }

    const std::unordered_map<std::string, int64_t>& family_mapping() const {
        return family_to_id_;
    }

private:
    std::unordered_map<std::string, int64_t> genus_to_id_;
    std::unordered_map<std::string, int64_t> family_to_id_;
};

} // namespace resolve
