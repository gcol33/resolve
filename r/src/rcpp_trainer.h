// rcpp_trainer.h - RTrainer class wrapper
#ifndef RCPP_TRAINER_H
#define RCPP_TRAINER_H

#include "rcpp_common.h"
#include "rcpp_model.h"
#include "rcpp_dataset.h"

// =============================================================================
// Trainer class wrapper
// =============================================================================

class RTrainer {
public:
    RTrainer(RResolveModel& model, List config_list) {
        resolve::TrainConfig config;
        if (config_list.containsElementNamed("batch_size")) {
            config.batch_size = config_list["batch_size"];
        }
        if (config_list.containsElementNamed("max_epochs")) {
            config.max_epochs = config_list["max_epochs"];
        }
        if (config_list.containsElementNamed("patience")) {
            config.patience = config_list["patience"];
        }
        if (config_list.containsElementNamed("lr")) {
            config.lr = config_list["lr"];
        }
        if (config_list.containsElementNamed("weight_decay")) {
            config.weight_decay = config_list["weight_decay"];
        }
        if (config_list.containsElementNamed("device")) {
            std::string dev = as<std::string>(config_list["device"]);
            config.device = (dev == "cuda") ? torch::kCUDA : torch::kCPU;
        }
        if (config_list.containsElementNamed("loss_config")) {
            config.loss_config = parse_loss_config_mode(
                as<std::string>(config_list["loss_config"]));
        }
        if (config_list.containsElementNamed("lr_scheduler")) {
            config.lr_scheduler = parse_lr_scheduler_type(
                as<std::string>(config_list["lr_scheduler"]));
        }
        if (config_list.containsElementNamed("lr_step_size")) {
            config.lr_step_size = config_list["lr_step_size"];
        }
        if (config_list.containsElementNamed("lr_gamma")) {
            config.lr_gamma = config_list["lr_gamma"];
        }
        if (config_list.containsElementNamed("lr_min")) {
            config.lr_min = config_list["lr_min"];
        }
        // Phase boundaries
        if (config_list.containsElementNamed("phase_boundaries")) {
            IntegerVector pb = config_list["phase_boundaries"];
            if (pb.size() >= 2) {
                config.phase_boundaries = {pb[0], pb[1]};
            }
        }
        // Band thresholds
        if (config_list.containsElementNamed("band_thresholds")) {
            config.band_thresholds = as<std::vector<float>>(config_list["band_thresholds"]);
        }
        // Checkpointing
        if (config_list.containsElementNamed("checkpoint_dir")) {
            config.checkpoint_dir = as<std::string>(config_list["checkpoint_dir"]);
        }
        if (config_list.containsElementNamed("checkpoint_every")) {
            config.checkpoint_every = config_list["checkpoint_every"];
        }
        // AMP
        if (config_list.containsElementNamed("use_amp")) {
            config.use_amp = config_list["use_amp"];
        }
        if (config_list.containsElementNamed("amp_init_scale")) {
            config.amp_init_scale = as<float>(config_list["amp_init_scale"]);
        }
        if (config_list.containsElementNamed("amp_growth_factor")) {
            config.amp_growth_factor = as<float>(config_list["amp_growth_factor"]);
        }
        if (config_list.containsElementNamed("amp_backoff_factor")) {
            config.amp_backoff_factor = as<float>(config_list["amp_backoff_factor"]);
        }
        if (config_list.containsElementNamed("amp_growth_interval")) {
            config.amp_growth_interval = config_list["amp_growth_interval"];
        }
        // CUDA opts
        if (config_list.containsElementNamed("cudnn_benchmark")) {
            config.cudnn_benchmark = config_list["cudnn_benchmark"];
        }
        if (config_list.containsElementNamed("allow_tf32")) {
            config.allow_tf32 = config_list["allow_tf32"];
        }
        if (config_list.containsElementNamed("vram_fraction")) {
            config.vram_fraction = as<float>(config_list["vram_fraction"]);
        }

        trainer_ = std::make_unique<resolve::Trainer>(*(model.model()), config);
    }

    void prepare_data(
        NumericMatrix coordinates,
        NumericMatrix covariates,
        NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> species_ids,
        Nullable<NumericMatrix> species_vector,
        Nullable<IntegerMatrix> genus_ids,
        Nullable<IntegerMatrix> family_ids,
        Nullable<NumericVector> unknown_fraction,
        Nullable<NumericVector> unknown_count,
        List targets,
        double test_size = 0.2,
        int seed = 42
    ) {
        torch::Tensor coords_t = r_mat_to_tensor(coordinates);
        torch::Tensor covs_t = r_mat_to_tensor(covariates);
        torch::Tensor hash_t = r_mat_to_tensor(hash_embedding);

        torch::Tensor species_id_t, species_vec_t, genus_t, family_t, unk_frac_t, unk_cnt_t;

        if (species_ids.isNotNull()) {
            species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        }
        if (species_vector.isNotNull()) {
            species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        }
        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }
        if (unknown_fraction.isNotNull()) {
            unk_frac_t = r_vec_to_tensor(as<NumericVector>(unknown_fraction));
        }
        if (unknown_count.isNotNull()) {
            unk_cnt_t = r_vec_to_tensor(as<NumericVector>(unknown_count));
        }

        // Convert targets list to map
        std::unordered_map<std::string, torch::Tensor> target_map;
        CharacterVector target_names = targets.names();
        for (int i = 0; i < targets.size(); ++i) {
            std::string name = as<std::string>(target_names[i]);
            target_map[name] = r_vec_to_tensor(as<NumericVector>(targets[i]));
        }

        trainer_->prepare_data(
            coords_t, covs_t, hash_t, species_id_t, species_vec_t,
            genus_t, family_t, unk_frac_t, unk_cnt_t,
            target_map,
            /*pool_genus_ids=*/{}, /*pool_family_ids=*/{},
            /*pool_weights=*/{}, /*pool_mask=*/{}, /*pool_has_cover=*/{},
            static_cast<float>(test_size), seed
        );
    }

    // Prepare data with pool fields (for rank_pool / transformer modes)
    void prepare_data_pool(
        NumericMatrix continuous,
        IntegerMatrix species_ids,
        Nullable<IntegerMatrix> pool_genus_ids,
        Nullable<IntegerMatrix> pool_family_ids,
        NumericMatrix pool_weights,
        IntegerMatrix pool_mask,
        NumericVector pool_has_cover,
        Nullable<NumericVector> unknown_fraction,
        List targets,
        double test_size = 0.2,
        int seed = 42
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor sp_ids_t = r_int_mat_to_tensor(species_ids);
        torch::Tensor p_weights_t = r_mat_to_tensor(pool_weights);
        torch::Tensor p_mask_t = r_int_mat_to_tensor(pool_mask).to(torch::kBool);
        torch::Tensor p_cover_t = r_vec_to_tensor(pool_has_cover);
        torch::Tensor p_genus_t, p_family_t, unk_frac_t;

        if (pool_genus_ids.isNotNull()) p_genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(pool_genus_ids));
        if (pool_family_ids.isNotNull()) p_family_t = r_int_mat_to_tensor(as<IntegerMatrix>(pool_family_ids));
        if (unknown_fraction.isNotNull()) unk_frac_t = r_vec_to_tensor(as<NumericVector>(unknown_fraction));

        std::unordered_map<std::string, torch::Tensor> target_map;
        CharacterVector target_names = targets.names();
        for (int i = 0; i < targets.size(); ++i) {
            target_map[as<std::string>(target_names[i])] = r_vec_to_tensor(as<NumericVector>(targets[i]));
        }

        trainer_->prepare_data(
            cont_t,
            /*covariates=*/{}, /*hash_embedding=*/{},
            sp_ids_t, /*species_vector=*/{},
            /*genus_ids=*/{}, /*family_ids=*/{},
            unk_frac_t, /*unknown_count=*/{},
            target_map,
            p_genus_t, p_family_t, p_weights_t, p_mask_t, p_cover_t,
            static_cast<float>(test_size), seed
        );
    }

    // Prepare data from ResolveDataset (matches Python API)
    void prepare_data_from_dataset(
        RResolveDataset& dataset,
        double test_size = 0.2,
        int seed = 42
    ) {
        trainer_->prepare_data(*(dataset.dataset()), static_cast<float>(test_size), seed);
    }

    List fit() {
        auto result = trainer_->fit();
        return train_result_to_list(result);
    }

    void save(std::string path, Nullable<List> metadata = R_NilValue) {
        if (metadata.isNotNull()) {
            auto rm = parse_run_metadata(as<List>(metadata));
            trainer_->save(path, &rm);
        } else {
            trainer_->save(path);
        }
    }

    // Accessors
    List get_scalers() const {
        return scalers_to_list(trainer_->scalers());
    }

    List get_config() const {
        const auto& c = trainer_->config();
        return List::create(
            Named("batch_size") = c.batch_size,
            Named("max_epochs") = c.max_epochs,
            Named("patience") = c.patience,
            Named("lr") = c.lr,
            Named("weight_decay") = c.weight_decay,
            Named("device") = c.device.is_cuda() ? "cuda" : "cpu",
            Named("vram_fraction") = c.vram_fraction
        );
    }

    // Diagnostic / evaluation methods
    List compute_diagnostics() {
        auto diag = trainer_->compute_diagnostics();
        return network_diagnostics_to_list(diag);
    }

    List compute_calibration(std::string target_name, int n_bins = 10) {
        auto result = trainer_->compute_calibration(target_name, n_bins);
        return calibration_result_to_list(result);
    }

    List compute_residuals(std::string target_name) {
        auto result = trainer_->compute_residuals(target_name);
        return residual_analysis_to_list(result);
    }

    List cross_validate(int n_folds = 5, int seed = 42) {
        auto result = trainer_->cross_validate(n_folds, seed);
        return cross_validation_result_to_list(result);
    }

    List cross_validate_spatial(List spatial_config_list, int n_folds = 5, int seed = 42) {
        auto config = parse_spatial_block_config(spatial_config_list);
        auto result = trainer_->cross_validate_spatial(config, n_folds, seed);
        return cross_validation_result_to_list(result);
    }

    List predict_from_trainer(
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
        torch::Tensor pg, pf, pw, pm, ph;
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        if (pool_genus_ids.isNotNull()) pg = r_int_mat_to_tensor(as<IntegerMatrix>(pool_genus_ids));
        if (pool_family_ids.isNotNull()) pf = r_int_mat_to_tensor(as<IntegerMatrix>(pool_family_ids));
        if (pool_weights.isNotNull()) pw = r_mat_to_tensor(as<NumericMatrix>(pool_weights));
        if (pool_mask.isNotNull()) pm = r_int_mat_to_tensor(as<IntegerMatrix>(pool_mask)).to(torch::kBool);
        if (pool_has_cover.isNotNull()) ph = r_vec_to_tensor(as<NumericVector>(pool_has_cover));

        auto result = trainer_->predict(cont_t, genus_t, family_t, species_id_t, species_vec_t,
                                        pg, pf, pw, pm, ph);
        List outputs;
        for (const auto& [name, tensor] : result) {
            outputs[name] = tensor_to_r_vec(tensor);
        }
        return outputs;
    }

    resolve::Trainer& trainer() { return *trainer_; }

private:
    std::unique_ptr<resolve::Trainer> trainer_;
};

#endif // RCPP_TRAINER_HPP
