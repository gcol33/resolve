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
                if (target_cfg.containsElementNamed("class_weights")) {
                    tc.class_weights = as<std::vector<float>>(target_cfg["class_weights"]);
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

        // RankPool / Transformer fields
        if (config_list.containsElementNamed("cover_dropout")) {
            config.cover_dropout = as<float>(config_list["cover_dropout"]);
        }
        if (config_list.containsElementNamed("d_model")) {
            config.d_model = config_list["d_model"];
        }
        if (config_list.containsElementNamed("n_heads")) {
            config.n_heads = config_list["n_heads"];
        }
        if (config_list.containsElementNamed("n_attention_layers")) {
            config.n_attention_layers = config_list["n_attention_layers"];
        }
        if (config_list.containsElementNamed("transformer_ff_dim")) {
            config.transformer_ff_dim = config_list["transformer_ff_dim"];
        }
        if (config_list.containsElementNamed("transformer_pooling")) {
            config.transformer_pooling = as<std::string>(config_list["transformer_pooling"]);
        }
        if (config_list.containsElementNamed("transformer_dropout")) {
            config.transformer_dropout = as<float>(config_list["transformer_dropout"]);
        }

        // uses_explicit_vector
        if (config_list.containsElementNamed("uses_explicit_vector")) {
            config.uses_explicit_vector = config_list["uses_explicit_vector"];
        }

        // MoE configuration
        if (config_list.containsElementNamed("moe_routing")) {
            config.moe_routing = parse_moe_routing_type(as<std::string>(config_list["moe_routing"]));
        }
        if (config_list.containsElementNamed("n_experts")) {
            config.n_experts = config_list["n_experts"];
        }
        if (config_list.containsElementNamed("expert_hidden_dims")) {
            config.expert_hidden_dims = as<std::vector<int64_t>>(config_list["expert_hidden_dims"]);
        }
        if (config_list.containsElementNamed("moe_top_k")) {
            config.moe_top_k = config_list["moe_top_k"];
        }
        if (config_list.containsElementNamed("moe_noise_std")) {
            config.moe_noise_std = as<float>(config_list["moe_noise_std"]);
        }
        if (config_list.containsElementNamed("moe_aux_loss_weight")) {
            config.moe_aux_loss_weight = as<float>(config_list["moe_aux_loss_weight"]);
        }

        // Configurable architecture
        if (config_list.containsElementNamed("activation")) {
            config.activation = parse_activation_type(as<std::string>(config_list["activation"]));
        }
        if (config_list.containsElementNamed("normalization")) {
            config.normalization = parse_norm_layer_type(as<std::string>(config_list["normalization"]));
        }
        if (config_list.containsElementNamed("norm_groups")) {
            config.norm_groups = config_list["norm_groups"];
        }
        if (config_list.containsElementNamed("use_residual")) {
            config.use_residual = config_list["use_residual"];
        }
        if (config_list.containsElementNamed("leaky_relu_slope")) {
            config.leaky_relu_slope = as<float>(config_list["leaky_relu_slope"]);
        }
        if (config_list.containsElementNamed("elu_alpha")) {
            config.elu_alpha = as<float>(config_list["elu_alpha"]);
        }

        // Multi-layer prediction heads
        if (config_list.containsElementNamed("head_hidden_dims")) {
            config.head_hidden_dims = as<std::vector<int64_t>>(config_list["head_hidden_dims"]);
        }
        if (config_list.containsElementNamed("head_activation")) {
            config.head_activation = parse_activation_type(as<std::string>(config_list["head_activation"]));
        }
        if (config_list.containsElementNamed("head_dropout")) {
            config.head_dropout = as<float>(config_list["head_dropout"]);
        }

        // Advanced architecture
        if (config_list.containsElementNamed("encoder_architecture")) {
            config.encoder_architecture = parse_encoder_architecture(as<std::string>(config_list["encoder_architecture"]));
        }

        // Architecture-specific sub-configs
        if (config_list.containsElementNamed("ft_transformer")) {
            config.ft_transformer = parse_ft_transformer_config(as<List>(config_list["ft_transformer"]));
        }
        if (config_list.containsElementNamed("tabnet")) {
            config.tabnet = parse_tabnet_config(as<List>(config_list["tabnet"]));
        }
        if (config_list.containsElementNamed("saint")) {
            config.saint = parse_saint_config(as<List>(config_list["saint"]));
        }
        if (config_list.containsElementNamed("gnn")) {
            config.gnn = parse_gnn_config(as<List>(config_list["gnn"]));
        }
        if (config_list.containsElementNamed("trait_net")) {
            config.trait_net = parse_trait_net_config(as<List>(config_list["trait_net"]));
        }
        if (config_list.containsElementNamed("excelformer")) {
            config.excelformer = parse_excelformer_config(as<List>(config_list["excelformer"]));
        }
        if (config_list.containsElementNamed("heterogeneous_gnn")) {
            config.heterogeneous_gnn = parse_heterogeneous_gnn_config(as<List>(config_list["heterogeneous_gnn"]));
        }
        if (config_list.containsElementNamed("parallel_layers")) {
            config.parallel_layers = parse_parallel_layers_config(as<List>(config_list["parallel_layers"]));
        }
        if (config_list.containsElementNamed("tabm")) {
            config.tabm = parse_tabm_config(as<List>(config_list["tabm"]));
        }

        model_ = std::make_shared<resolve::ResolveModel>(schema, config);
    }

    // Helper to unpack optional pool fields from R Nullable types
    struct PoolFields {
        torch::Tensor genus_ids, family_ids, weights, mask, has_cover;
    };

    static PoolFields unpack_pool(
        Nullable<IntegerMatrix> pool_genus_ids,
        Nullable<IntegerMatrix> pool_family_ids,
        Nullable<NumericMatrix> pool_weights,
        Nullable<IntegerMatrix> pool_mask,
        Nullable<NumericVector> pool_has_cover
    ) {
        PoolFields p;
        if (pool_genus_ids.isNotNull()) p.genus_ids = r_int_mat_to_tensor(as<IntegerMatrix>(pool_genus_ids));
        if (pool_family_ids.isNotNull()) p.family_ids = r_int_mat_to_tensor(as<IntegerMatrix>(pool_family_ids));
        if (pool_weights.isNotNull()) p.weights = r_mat_to_tensor(as<NumericMatrix>(pool_weights));
        if (pool_mask.isNotNull()) p.mask = r_int_mat_to_tensor(as<IntegerMatrix>(pool_mask)).to(torch::kBool);
        if (pool_has_cover.isNotNull()) p.has_cover = r_vec_to_tensor(as<NumericVector>(pool_has_cover));
        return p;
    }

    List forward(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> pool_genus_ids = R_NilValue,
        Nullable<IntegerMatrix> pool_family_ids = R_NilValue,
        Nullable<NumericMatrix> pool_weights = R_NilValue,
        Nullable<IntegerMatrix> pool_mask = R_NilValue,
        Nullable<NumericVector> pool_has_cover = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));

        auto pool = unpack_pool(pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover);

        auto outputs = (*model_)->forward(
            cont_t, genus_t, family_t, species_id_t, species_vec_t,
            pool.genus_ids, pool.family_ids, pool.weights, pool.mask, pool.has_cover);

        List result;
        for (const auto& [name, tensor] : outputs) {
            result[name] = tensor_to_r_vec(tensor);
        }
        return result;
    }

    NumericVector get_latent(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> pool_genus_ids = R_NilValue,
        Nullable<IntegerMatrix> pool_family_ids = R_NilValue,
        Nullable<NumericMatrix> pool_weights = R_NilValue,
        Nullable<IntegerMatrix> pool_mask = R_NilValue,
        Nullable<NumericVector> pool_has_cover = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));

        auto pool = unpack_pool(pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover);

        torch::Tensor latent = (*model_)->get_latent(
            cont_t, genus_t, family_t, species_id_t, species_vec_t,
            pool.genus_ids, pool.family_ids, pool.weights, pool.mask, pool.has_cover);
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

    // Forward with MoE auxiliary loss
    List forward_with_aux(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> pool_genus_ids = R_NilValue,
        Nullable<IntegerMatrix> pool_family_ids = R_NilValue,
        Nullable<NumericMatrix> pool_weights = R_NilValue,
        Nullable<IntegerMatrix> pool_mask = R_NilValue,
        Nullable<NumericVector> pool_has_cover = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        auto p = unpack_pool(pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover);

        auto result = (*model_)->forward_with_aux(
            cont_t, genus_t, family_t, species_id_t, species_vec_t,
            p.genus_ids, p.family_ids, p.weights, p.mask, p.has_cover);

        List outputs;
        for (const auto& [name, tensor] : result.outputs) {
            outputs[name] = tensor_to_r_vec(tensor);
        }
        List ret;
        ret["outputs"] = outputs;
        if (result.moe_aux_loss.defined()) {
            ret["moe_aux_loss"] = tensor_to_r_vec(result.moe_aux_loss);
        }
        return ret;
    }

    // Forward for single target
    NumericVector forward_single(
        const std::string& target,
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));

        auto result = (*model_)->forward_single(target, cont_t, genus_t, family_t, species_id_t, species_vec_t);
        return tensor_to_r_vec(result);
    }

    // Encode with intermediate activations (diagnostics)
    List encode_with_activations(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t;
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));

        auto [latent, activations] = (*model_)->encode_with_activations(cont_t, genus_t, family_t);
        List act_list;
        for (const auto& a : activations) {
            act_list.push_back(tensor_to_r_mat(a));
        }
        return List::create(Named("latent") = tensor_to_r_mat(latent), Named("activations") = act_list);
    }

    // MoE gate probabilities
    NumericMatrix get_gate_probs(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t;
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));

        auto result = (*model_)->get_gate_probs(cont_t, genus_t, family_t);
        return tensor_to_r_mat(result);
    }

    // Set species trait matrix (for TraitNet architecture)
    void set_traits(NumericMatrix traits) {
        torch::Tensor t = r_mat_to_tensor(traits);
        (*model_)->set_traits(t);
    }

    // Accessors
    std::string species_encoding() const {
        auto mode = (*model_)->species_encoding();
        switch (mode) {
            case resolve::SpeciesEncodingMode::Hash: return "hash";
            case resolve::SpeciesEncodingMode::Embed: return "embed";
            case resolve::SpeciesEncodingMode::Sparse: return "sparse";
            case resolve::SpeciesEncodingMode::RankPool: return "rank_pool";
            case resolve::SpeciesEncodingMode::Transformer: return "transformer";
            default: return "unknown";
        }
    }

    bool uses_explicit_vector() const { return (*model_)->uses_explicit_vector(); }
    bool uses_moe() const { return (*model_)->uses_moe(); }
    int n_experts() const { return (*model_)->n_experts(); }

    // Embedding weight extraction
    Nullable<NumericMatrix> get_genus_weights() {
        auto t = (*model_)->get_genus_weights();
        if (!t.defined()) return R_NilValue;
        return tensor_to_r_mat(t);
    }

    Nullable<NumericMatrix> get_family_weights() {
        auto t = (*model_)->get_family_weights();
        if (!t.defined()) return R_NilValue;
        return tensor_to_r_mat(t);
    }

    Nullable<NumericMatrix> get_species_weights() {
        auto t = (*model_)->get_species_weights();
        if (!t.defined()) return R_NilValue;
        return tensor_to_r_mat(t);
    }

private:
    // Private constructor for wrapping an existing model (used by load paths)
    RResolveModel(std::shared_ptr<resolve::ResolveModel> model) : model_(model) {}
    friend class RTrainer;
    friend class RPredictor;

    std::shared_ptr<resolve::ResolveModel> model_;
};

#endif // RCPP_MODEL_HPP
