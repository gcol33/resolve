// rcpp_trainer.hpp - RTrainer class wrapper
#ifndef RCPP_TRAINER_HPP
#define RCPP_TRAINER_HPP

#include "rcpp_common.hpp"
#include "rcpp_model.hpp"
#include "rcpp_dataset.hpp"

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
        resolve::TrainResult result = trainer_->fit();

        List metrics;
        for (const auto& [target_name, target_metrics] : result.final_metrics) {
            List target_list;
            for (const auto& [metric_name, value] : target_metrics) {
                target_list[metric_name] = value;
            }
            metrics[target_name] = target_list;
        }

        return List::create(
            Named("best_epoch") = result.best_epoch,
            Named("final_metrics") = metrics,
            Named("train_loss") = wrap(result.train_loss_history),
            Named("test_loss") = wrap(result.test_loss_history),
            Named("train_time_seconds") = result.train_time_seconds,
            Named("resumed_from_epoch") = result.resumed_from_epoch
        );
    }

    void save(std::string path) { trainer_->save(path); }

    resolve::Trainer& trainer() { return *trainer_; }

private:
    std::unique_ptr<resolve::Trainer> trainer_;
};

#endif // RCPP_TRAINER_HPP
