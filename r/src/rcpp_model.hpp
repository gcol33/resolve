// rcpp_model.hpp - RResolveModel class wrapper
#ifndef RCPP_MODEL_HPP
#define RCPP_MODEL_HPP

#include "rcpp_common.hpp"

// =============================================================================
// ResolveModel class wrapper
// =============================================================================

class RResolveModel {
public:
    RResolveModel(List schema_list, List config_list) {
        resolve::ResolveSchema schema;
        schema.n_plots = schema_list["n_plots"];
        schema.n_species = schema_list["n_species"];
        if (schema_list.containsElementNamed("n_species_vocab")) {
            schema.n_species_vocab = schema_list["n_species_vocab"];
        }
        schema.has_coordinates = schema_list["has_coordinates"];
        schema.has_abundance = schema_list["has_abundance"];
        schema.has_taxonomy = schema_list["has_taxonomy"];
        schema.n_genera = schema_list["n_genera"];
        schema.n_families = schema_list["n_families"];
        if (schema_list.containsElementNamed("covariate_names")) {
            schema.covariate_names = as<std::vector<std::string>>(schema_list["covariate_names"]);
        }
        schema.track_unknown_fraction = schema_list["track_unknown_fraction"];
        schema.track_unknown_count = schema_list["track_unknown_count"];

        // Parse targets
        if (schema_list.containsElementNamed("targets")) {
            List targets = schema_list["targets"];
            CharacterVector target_names = targets.names();
            for (int i = 0; i < targets.size(); ++i) {
                List target_cfg = targets[i];
                resolve::TargetConfig tc;
                tc.name = as<std::string>(target_names[i]);
                tc.task = parse_task_type(as<std::string>(target_cfg["task"]));
                if (target_cfg.containsElementNamed("transform")) {
                    tc.transform = parse_transform_type(as<std::string>(target_cfg["transform"]));
                }
                if (target_cfg.containsElementNamed("num_classes")) {
                    tc.num_classes = target_cfg["num_classes"];
                }
                if (target_cfg.containsElementNamed("weight")) {
                    tc.weight = target_cfg["weight"];
                }
                schema.targets.push_back(tc);
            }
        }

        resolve::ModelConfig config;
        if (config_list.containsElementNamed("species_encoding")) {
            config.species_encoding = parse_species_encoding_mode(
                as<std::string>(config_list["species_encoding"]));
        }
        if (config_list.containsElementNamed("hash_dim")) {
            config.hash_dim = config_list["hash_dim"];
        }
        if (config_list.containsElementNamed("species_embed_dim")) {
            config.species_embed_dim = config_list["species_embed_dim"];
        }
        if (config_list.containsElementNamed("genus_emb_dim")) {
            config.genus_emb_dim = config_list["genus_emb_dim"];
        }
        if (config_list.containsElementNamed("family_emb_dim")) {
            config.family_emb_dim = config_list["family_emb_dim"];
        }
        if (config_list.containsElementNamed("top_k")) {
            config.top_k = config_list["top_k"];
        }
        if (config_list.containsElementNamed("top_k_species")) {
            config.top_k_species = config_list["top_k_species"];
        }
        if (config_list.containsElementNamed("n_taxonomy_slots")) {
            config.n_taxonomy_slots = config_list["n_taxonomy_slots"];
        }
        if (config_list.containsElementNamed("hidden_dims")) {
            config.hidden_dims = as<std::vector<int64_t>>(config_list["hidden_dims"]);
        }
        if (config_list.containsElementNamed("dropout")) {
            config.dropout = config_list["dropout"];
        }

        model_ = std::make_shared<resolve::ResolveModel>(schema, config);
    }

    List forward(
        NumericMatrix continuous,
        Nullable<NumericMatrix> genus_ids = R_NilValue,
        Nullable<NumericMatrix> family_ids = R_NilValue,
        Nullable<NumericMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }
        if (species_ids.isNotNull()) {
            species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        }
        if (species_vector.isNotNull()) {
            species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        }

        auto outputs = (*model_)->forward(cont_t, genus_t, family_t, species_id_t, species_vec_t);

        List result;
        for (const auto& [name, tensor] : outputs) {
            result[name] = tensor_to_r_vec(tensor);
        }
        return result;
    }

    NumericVector get_latent(
        NumericMatrix continuous,
        Nullable<NumericMatrix> genus_ids = R_NilValue,
        Nullable<NumericMatrix> family_ids = R_NilValue,
        Nullable<NumericMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }
        if (species_ids.isNotNull()) {
            species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        }
        if (species_vector.isNotNull()) {
            species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        }

        torch::Tensor latent = (*model_)->get_latent(cont_t, genus_t, family_t, species_id_t, species_vec_t);
        return tensor_to_r_vec(latent);
    }

    void train(bool mode = true) { (*model_)->train(mode); }
    void eval() { (*model_)->eval(); }

    void to_device(std::string device) {
        if (device == "cuda") {
            (*model_)->to(torch::kCUDA);
        } else {
            (*model_)->to(torch::kCPU);
        }
    }

    int latent_dim() const { return (*model_)->latent_dim(); }

    std::shared_ptr<resolve::ResolveModel>& model() { return model_; }

private:
    std::shared_ptr<resolve::ResolveModel> model_;
};

#endif // RCPP_MODEL_HPP
