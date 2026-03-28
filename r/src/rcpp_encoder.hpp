// rcpp_encoder.hpp - RSpeciesEncoder class wrapper
#ifndef RCPP_ENCODER_HPP
#define RCPP_ENCODER_HPP

#include "rcpp_common.hpp"

// =============================================================================
// SpeciesEncoder class wrapper
// =============================================================================

class RSpeciesEncoder {
public:
    RSpeciesEncoder(
        int hash_dim = 32,
        int top_k = 3,
        std::string aggregation = "abundance",
        std::string normalization = "norm",
        bool track_unknown_count = false,
        std::string selection = "top",
        std::string representation = "abundance",
        int min_species_frequency = 1
    ) : encoder_(
            hash_dim,
            top_k,
            parse_aggregation_mode(aggregation),
            parse_normalization_mode(normalization),
            track_unknown_count,
            parse_selection_mode(selection),
            parse_representation_mode(representation),
            min_species_frequency
        ) {}

    // Fit from R data frame
    void fit(DataFrame species_df) {
        CharacterVector species_id = species_df["species_id"];
        CharacterVector genus = species_df["genus"];
        CharacterVector family = species_df["family"];
        NumericVector abundance = species_df["abundance"];
        CharacterVector plot_id = species_df["plot_id"];

        std::vector<resolve::SpeciesRecord> records;
        records.reserve(species_id.size());
        for (int i = 0; i < species_id.size(); ++i) {
            resolve::SpeciesRecord rec;
            rec.species_id = as<std::string>(species_id[i]);
            rec.genus = as<std::string>(genus[i]);
            rec.family = as<std::string>(family[i]);
            rec.abundance = abundance[i];
            rec.plot_id = as<std::string>(plot_id[i]);
            records.push_back(rec);
        }

        encoder_.fit(records);
    }

    // Transform species data
    List transform(DataFrame species_df, CharacterVector plot_ids) {
        CharacterVector species_id = species_df["species_id"];
        CharacterVector genus = species_df["genus"];
        CharacterVector family = species_df["family"];
        NumericVector abundance = species_df["abundance"];
        CharacterVector df_plot_id = species_df["plot_id"];

        std::vector<resolve::SpeciesRecord> records;
        records.reserve(species_id.size());
        for (int i = 0; i < species_id.size(); ++i) {
            resolve::SpeciesRecord rec;
            rec.species_id = as<std::string>(species_id[i]);
            rec.genus = as<std::string>(genus[i]);
            rec.family = as<std::string>(family[i]);
            rec.abundance = abundance[i];
            rec.plot_id = as<std::string>(df_plot_id[i]);
            records.push_back(rec);
        }

        std::vector<std::string> pids = as<std::vector<std::string>>(plot_ids);
        resolve::EncodedSpecies encoded = encoder_.transform(records, pids);

        List result;
        if (encoded.hash_embedding.defined()) {
            result["hash_embedding"] = tensor_to_r_mat(encoded.hash_embedding);
        }
        if (encoded.genus_ids.defined()) {
            result["genus_ids"] = tensor_to_r_mat(encoded.genus_ids.to(torch::kFloat32));
        }
        if (encoded.family_ids.defined()) {
            result["family_ids"] = tensor_to_r_mat(encoded.family_ids.to(torch::kFloat32));
        }
        if (encoded.unknown_fraction.defined()) {
            result["unknown_fraction"] = tensor_to_r_vec(encoded.unknown_fraction);
        }
        if (encoded.unknown_count.defined()) {
            result["unknown_count"] = tensor_to_r_vec(encoded.unknown_count);
        }
        if (encoded.species_vector.defined()) {
            result["species_vector"] = tensor_to_r_mat(encoded.species_vector);
        }
        if (encoded.species_ids.defined() && encoded.species_ids.numel() > 0) {
            result["species_ids"] = tensor_to_r_mat(encoded.species_ids.to(torch::kFloat32));
        }
        result["plot_ids"] = wrap(encoded.plot_ids);

        return result;
    }

    bool is_fitted() const { return encoder_.is_fitted(); }
    int hash_dim() const { return encoder_.hash_dim(); }
    int top_k() const { return encoder_.top_k(); }
    int n_genera() const { return encoder_.n_genera(); }
    int n_families() const { return encoder_.n_families(); }
    int n_taxonomy_slots() const { return encoder_.n_taxonomy_slots(); }
    bool uses_explicit_vector() const { return encoder_.uses_explicit_vector(); }
    int n_species_vector() const { return encoder_.n_species_vector(); }
    int n_known_species() const { return encoder_.n_known_species(); }

    void save(std::string path) const { encoder_.save(path); }

    static RSpeciesEncoder load(std::string path) {
        RSpeciesEncoder wrapper;
        wrapper.encoder_ = resolve::SpeciesEncoder::load(path);
        return wrapper;
    }

    resolve::SpeciesEncoder& encoder() { return encoder_; }
    const resolve::SpeciesEncoder& encoder() const { return encoder_; }

private:
    RSpeciesEncoder() : encoder_() {}  // For load()
    resolve::SpeciesEncoder encoder_;
};

#endif // RCPP_ENCODER_HPP
