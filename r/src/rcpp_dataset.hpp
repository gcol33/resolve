// rcpp_dataset.hpp - RResolveDataset class wrapper
#ifndef RCPP_DATASET_HPP
#define RCPP_DATASET_HPP

#include "rcpp_common.hpp"

// =============================================================================
// ResolveDataset class wrapper (mirrors Python ResolveDataset)
// =============================================================================

class RResolveDataset {
public:
    // Load from CSV files (matches Python ResolveDataset.from_csv)
    static RResolveDataset from_csv(
        std::string header_path,
        std::string species_path,
        List roles_list,
        List targets_list,
        List config_list = List()
    ) {
        // Build RoleMapping
        resolve::RoleMapping roles;
        if (roles_list.containsElementNamed("plot_id")) {
            roles.plot_id = as<std::string>(roles_list["plot_id"]);
        }
        if (roles_list.containsElementNamed("species_id")) {
            roles.species_id = as<std::string>(roles_list["species_id"]);
        }
        if (roles_list.containsElementNamed("abundance")) {
            roles.abundance = as<std::string>(roles_list["abundance"]);
        }
        if (roles_list.containsElementNamed("longitude")) {
            roles.longitude = as<std::string>(roles_list["longitude"]);
        }
        if (roles_list.containsElementNamed("latitude")) {
            roles.latitude = as<std::string>(roles_list["latitude"]);
        }
        if (roles_list.containsElementNamed("genus")) {
            roles.genus = as<std::string>(roles_list["genus"]);
        }
        if (roles_list.containsElementNamed("family")) {
            roles.family = as<std::string>(roles_list["family"]);
        }
        if (roles_list.containsElementNamed("covariates")) {
            roles.covariates = as<std::vector<std::string>>(roles_list["covariates"]);
        }

        // Build TargetSpecs
        std::vector<resolve::TargetSpec> targets;
        CharacterVector target_names = targets_list.names();
        for (int i = 0; i < targets_list.size(); ++i) {
            List target_cfg = targets_list[i];
            resolve::TargetSpec spec;
            spec.target_name = as<std::string>(target_names[i]);
            spec.column_name = as<std::string>(target_cfg["column"]);

            if (target_cfg.containsElementNamed("task")) {
                spec.task = parse_task_type(as<std::string>(target_cfg["task"]));
            }
            if (target_cfg.containsElementNamed("transform")) {
                spec.transform = parse_transform_type(as<std::string>(target_cfg["transform"]));
            }
            if (target_cfg.containsElementNamed("num_classes")) {
                spec.num_classes = target_cfg["num_classes"];
            }
            if (target_cfg.containsElementNamed("weight")) {
                spec.weight = target_cfg["weight"];
            }
            targets.push_back(spec);
        }

        // Build DatasetConfig
        resolve::DatasetConfig config;
        if (config_list.containsElementNamed("species_encoding")) {
            config.species_encoding = parse_species_encoding_mode(
                as<std::string>(config_list["species_encoding"]));
        }
        if (config_list.containsElementNamed("hash_dim")) {
            config.hash_dim = config_list["hash_dim"];
        }
        if (config_list.containsElementNamed("top_k")) {
            config.top_k = config_list["top_k"];
        }
        if (config_list.containsElementNamed("top_k_species")) {
            config.top_k_species = config_list["top_k_species"];
        }
        if (config_list.containsElementNamed("selection")) {
            config.selection = parse_selection_mode(as<std::string>(config_list["selection"]));
        }
        if (config_list.containsElementNamed("representation")) {
            config.representation = parse_representation_mode(
                as<std::string>(config_list["representation"]));
        }
        if (config_list.containsElementNamed("normalization")) {
            config.normalization = parse_normalization_mode(
                as<std::string>(config_list["normalization"]));
        }
        if (config_list.containsElementNamed("track_unknown_fraction")) {
            config.track_unknown_fraction = config_list["track_unknown_fraction"];
        }
        if (config_list.containsElementNamed("track_unknown_count")) {
            config.track_unknown_count = config_list["track_unknown_count"];
        }
        if (config_list.containsElementNamed("use_taxonomy")) {
            config.use_taxonomy = config_list["use_taxonomy"];
        }

        // Load dataset via C++ core
        RResolveDataset wrapper;
        wrapper.dataset_ = std::make_shared<resolve::ResolveDataset>(
            resolve::ResolveDataset::from_csv(header_path, species_path, roles, targets, config)
        );
        return wrapper;
    }

    // Load from species CSV only (matches Python ResolveDataset.from_species_csv)
    static RResolveDataset from_species_csv(
        std::string species_path,
        List roles_list,
        List targets_list,
        List config_list = List()
    ) {
        // Build RoleMapping
        resolve::RoleMapping roles;
        roles.plot_id = as<std::string>(roles_list["plot_id"]);
        roles.species_id = as<std::string>(roles_list["species_id"]);
        if (roles_list.containsElementNamed("abundance")) {
            roles.abundance = as<std::string>(roles_list["abundance"]);
        }
        if (roles_list.containsElementNamed("longitude")) {
            roles.longitude = as<std::string>(roles_list["longitude"]);
        }
        if (roles_list.containsElementNamed("latitude")) {
            roles.latitude = as<std::string>(roles_list["latitude"]);
        }
        if (roles_list.containsElementNamed("genus")) {
            roles.genus = as<std::string>(roles_list["genus"]);
        }
        if (roles_list.containsElementNamed("family")) {
            roles.family = as<std::string>(roles_list["family"]);
        }
        if (roles_list.containsElementNamed("covariates")) {
            roles.covariates = as<std::vector<std::string>>(roles_list["covariates"]);
        }

        // Build TargetSpecs
        std::vector<resolve::TargetSpec> targets;
        CharacterVector target_names = targets_list.names();
        for (int i = 0; i < targets_list.size(); ++i) {
            List spec = targets_list[i];
            resolve::TargetSpec ts;
            ts.column_name = as<std::string>(target_names[i]);
            ts.target_name = as<std::string>(target_names[i]);
            ts.task = parse_task_type(as<std::string>(spec["task"]));
            if (spec.containsElementNamed("transform")) {
                ts.transform = parse_transform_type(as<std::string>(spec["transform"]));
            }
            if (spec.containsElementNamed("num_classes")) {
                ts.num_classes = spec["num_classes"];
            }
            targets.push_back(ts);
        }

        // Build DatasetConfig
        resolve::DatasetConfig config;
        if (config_list.containsElementNamed("species_encoding")) {
            config.species_encoding = parse_species_encoding_mode(
                as<std::string>(config_list["species_encoding"]));
        }
        if (config_list.containsElementNamed("hash_dim")) config.hash_dim = config_list["hash_dim"];
        if (config_list.containsElementNamed("top_k")) config.top_k = config_list["top_k"];
        if (config_list.containsElementNamed("top_k_species")) config.top_k_species = config_list["top_k_species"];

        RResolveDataset wrapper;
        wrapper.dataset_ = std::make_shared<resolve::ResolveDataset>(
            resolve::ResolveDataset::from_species_csv(species_path, roles, targets, config));
        return wrapper;
    }

    // Accessors (return R types)
    NumericMatrix coordinates() const {
        return tensor_to_r_mat(dataset_->coordinates());
    }

    NumericMatrix covariates() const {
        return tensor_to_r_mat(dataset_->covariates());
    }

    NumericMatrix hash_embedding() const {
        return tensor_to_r_mat(dataset_->hash_embedding());
    }

    Nullable<NumericMatrix> species_ids() const {
        if (dataset_->species_ids().defined() && dataset_->species_ids().numel() > 0) {
            return tensor_to_r_mat(dataset_->species_ids().to(torch::kFloat32));
        }
        return R_NilValue;
    }

    Nullable<NumericMatrix> species_vector() const {
        if (dataset_->species_vector().defined() && dataset_->species_vector().numel() > 0) {
            return tensor_to_r_mat(dataset_->species_vector());
        }
        return R_NilValue;
    }

    Nullable<NumericMatrix> genus_ids() const {
        if (dataset_->genus_ids().defined() && dataset_->genus_ids().numel() > 0) {
            return tensor_to_r_mat(dataset_->genus_ids().to(torch::kFloat32));
        }
        return R_NilValue;
    }

    Nullable<NumericMatrix> family_ids() const {
        if (dataset_->family_ids().defined() && dataset_->family_ids().numel() > 0) {
            return tensor_to_r_mat(dataset_->family_ids().to(torch::kFloat32));
        }
        return R_NilValue;
    }

    Nullable<NumericVector> unknown_fraction() const {
        if (dataset_->unknown_fraction().defined() && dataset_->unknown_fraction().numel() > 0) {
            return tensor_to_r_vec(dataset_->unknown_fraction());
        }
        return R_NilValue;
    }

    Nullable<NumericVector> unknown_count() const {
        if (dataset_->unknown_count().defined() && dataset_->unknown_count().numel() > 0) {
            return tensor_to_r_vec(dataset_->unknown_count());
        }
        return R_NilValue;
    }

    List targets() const {
        List result;
        for (const auto& [name, tensor] : dataset_->targets()) {
            result[name] = tensor_to_r_vec(tensor);
        }
        return result;
    }

    List schema() const {
        const auto& s = dataset_->schema();
        List targets_list;
        for (const auto& tc : s.targets) {
            std::string task_str = (tc.task == resolve::TaskType::Regression) ? "regression" : "classification";
            std::string transform_str = (tc.transform == resolve::TransformType::Log1p) ? "log1p" : "none";
            targets_list[tc.name] = List::create(
                Named("task") = task_str,
                Named("transform") = transform_str,
                Named("num_classes") = tc.num_classes,
                Named("weight") = tc.weight,
                Named("class_weights") = wrap(tc.class_weights)
            );
        }

        return List::create(
            Named("n_plots") = s.n_plots,
            Named("n_species") = s.n_species,
            Named("n_species_vocab") = s.n_species_vocab,
            Named("has_coordinates") = s.has_coordinates,
            Named("has_abundance") = s.has_abundance,
            Named("has_taxonomy") = s.has_taxonomy,
            Named("n_genera") = s.n_genera,
            Named("n_families") = s.n_families,
            Named("n_genera_vocab") = s.n_genera_vocab,
            Named("n_families_vocab") = s.n_families_vocab,
            Named("covariate_names") = wrap(s.covariate_names),
            Named("targets") = targets_list,
            Named("track_unknown_fraction") = s.track_unknown_fraction,
            Named("track_unknown_count") = s.track_unknown_count
        );
    }

    CharacterVector plot_ids() const {
        return wrap(dataset_->plot_ids());
    }

    CharacterVector species_vocab() const {
        return wrap(dataset_->species_vocab());
    }

    int n_plots() const {
        return static_cast<int>(dataset_->n_plots());
    }

    List config() const {
        const auto& c = dataset_->config();
        std::string encoding_str;
        switch (c.species_encoding) {
            case resolve::SpeciesEncodingMode::Hash: encoding_str = "hash"; break;
            case resolve::SpeciesEncodingMode::Embed: encoding_str = "embed"; break;
            case resolve::SpeciesEncodingMode::Sparse: encoding_str = "sparse"; break;
            case resolve::SpeciesEncodingMode::RankPool: encoding_str = "rank_pool"; break;
            case resolve::SpeciesEncodingMode::Transformer: encoding_str = "transformer"; break;
        }
        return List::create(
            Named("species_encoding") = encoding_str,
            Named("hash_dim") = c.hash_dim,
            Named("top_k") = c.top_k,
            Named("top_k_species") = c.top_k_species,
            Named("track_unknown_fraction") = c.track_unknown_fraction,
            Named("track_unknown_count") = c.track_unknown_count,
            Named("use_taxonomy") = c.use_taxonomy
        );
    }

    bool has_raw_species_data() const { return dataset_->has_raw_species_data(); }

    Nullable<NumericVector> raw_species_ids() const {
        const auto& t = dataset_->raw_species_ids();
        if (!t.defined()) return R_NilValue;
        torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
        float* data = cpu.data_ptr<float>();
        return NumericVector(data, data + cpu.numel());
    }

    Nullable<NumericVector> raw_weights() const {
        const auto& t = dataset_->raw_weights();
        if (!t.defined()) return R_NilValue;
        return tensor_to_r_vec(t);
    }

    Nullable<NumericVector> plot_offsets() const {
        const auto& t = dataset_->plot_offsets();
        if (!t.defined()) return R_NilValue;
        torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
        float* data = cpu.data_ptr<float>();
        return NumericVector(data, data + cpu.numel());
    }

    List taxonomy_vocab() const {
        const auto& tv = dataset_->taxonomy_vocab();
        // Convert genus and family maps
        List result;
        result["n_genera"] = (int)tv.n_genera();
        result["n_families"] = (int)tv.n_families();
        return result;
    }

    // Access internal dataset for C++ functions
    std::shared_ptr<resolve::ResolveDataset>& dataset() { return dataset_; }
    const std::shared_ptr<resolve::ResolveDataset>& dataset() const { return dataset_; }

    // Default constructor for Rcpp module
    RResolveDataset() = default;

private:
    std::shared_ptr<resolve::ResolveDataset> dataset_;
};

#endif // RCPP_DATASET_HPP
