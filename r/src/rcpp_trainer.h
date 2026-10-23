// rcpp_trainer.h - RTrainer wrapper (thin C-facade client).
#ifndef RCPP_TRAINER_H
#define RCPP_TRAINER_H

#include "rcpp_common.h"
#include "rcpp_model.h"
#include "rcpp_dataset.h"

class RTrainer {
public:
    RTrainer(RResolveModel& model, List config_list) {
        capi_require_loaded();
        ValuePtr config = owned_value(r_list_to_value_map(config_list, "config"));
        trainer_ = capi_own(resolve_trainer_create(model.handle(), config.get()),
                            resolve_trainer_free);
    }

    void prepare_data(
        NumericMatrix coordinates, NumericMatrix covariates, NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> species_ids, Nullable<NumericMatrix> species_vector,
        Nullable<IntegerMatrix> genus_ids, Nullable<IntegerMatrix> family_ids,
        Nullable<NumericVector> unknown_fraction, Nullable<NumericVector> unknown_count,
        List targets, Nullable<IntegerMatrix> categorical_ids = R_NilValue,
        double test_size = 0.2, int seed = 42) {
        ValuePtr in = owned_value(resolve_value_new_map());
        map_set_num_matrix(in.get(), "coordinates", coordinates);
        map_set_num_matrix(in.get(), "covariates", covariates);
        map_set_num_matrix(in.get(), "hash_embedding", hash_embedding);
        map_set_opt_int_matrix(in.get(), "species_ids", species_ids);
        map_set_opt_num_matrix(in.get(), "species_vector", species_vector);
        map_set_opt_int_matrix(in.get(), "genus_ids", genus_ids);
        map_set_opt_int_matrix(in.get(), "family_ids", family_ids);
        map_set_opt_num_vector(in.get(), "unknown_fraction", unknown_fraction);
        map_set_opt_num_vector(in.get(), "unknown_count", unknown_count);
        map_set_opt_int_matrix(in.get(), "categorical_ids", categorical_ids);
        set_targets(in.get(), targets);
        capi_check_status(resolve_trainer_prepare_data(trainer_.get(), in.get(), test_size, seed));
    }

    void prepare_data_pool(
        NumericMatrix continuous, IntegerMatrix species_ids,
        Nullable<IntegerMatrix> pool_genus_ids, Nullable<IntegerMatrix> pool_family_ids,
        NumericMatrix pool_weights, IntegerMatrix pool_mask, NumericVector pool_has_cover,
        Nullable<NumericVector> unknown_fraction, List targets,
        Nullable<IntegerMatrix> categorical_ids = R_NilValue,
        double test_size = 0.2, int seed = 42) {
        // The pool path feeds the continuous features through the engine's
        // "coordinates" slot (matches the original prepare_data_pool wiring).
        ValuePtr in = owned_value(resolve_value_new_map());
        map_set_num_matrix(in.get(), "coordinates", continuous);
        map_set_int_matrix(in.get(), "species_ids", species_ids);
        map_set_opt_int_matrix(in.get(), "pool_genus_ids", pool_genus_ids);
        map_set_opt_int_matrix(in.get(), "pool_family_ids", pool_family_ids);
        map_set_num_matrix(in.get(), "pool_weights", pool_weights);
        map_set_int_matrix(in.get(), "pool_mask", pool_mask);
        map_set_num_vector(in.get(), "pool_has_cover", pool_has_cover);
        map_set_opt_num_vector(in.get(), "unknown_fraction", unknown_fraction);
        map_set_opt_int_matrix(in.get(), "categorical_ids", categorical_ids);
        set_targets(in.get(), targets);
        capi_check_status(resolve_trainer_prepare_data(trainer_.get(), in.get(), test_size, seed));
    }

    void prepare_data_from_dataset(RResolveDataset& dataset, double test_size = 0.2, int seed = 42) {
        capi_check_status(resolve_trainer_prepare_data_from_dataset(
            trainer_.get(), dataset.handle(), test_size, seed));
    }

    // Holds out the plots named by `stopping_plots` (plot IDs) for early
    // stopping. Always sent as a STRING_ARRAY, so a single plot ID is not
    // read as a scalar.
    void prepare_data_stopping_fold(RResolveDataset& dataset, CharacterVector stopping_plots,
                                    int seed = 42) {
        std::vector<std::string> ids(static_cast<size_t>(stopping_plots.size()));
        std::vector<const char*> ptrs(ids.size());
        for (R_xlen_t i = 0; i < stopping_plots.size(); ++i) {
            if (stopping_plots[i] == NA_STRING) {
                stop("stoppingPlots must not contain NA");
            }
            ids[static_cast<size_t>(i)] = std::string(CHAR(stopping_plots[i]));
            ptrs[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)].c_str();
        }
        ValuePtr fold = owned_value(
            resolve_value_new_string_array(ptrs.data(), static_cast<int64_t>(ptrs.size())));
        capi_check_status(resolve_trainer_prepare_data_stopping_fold(
            trainer_.get(), dataset.handle(), fold.get(), seed));
    }

    RObject fit() { return value_to_r_owned(resolve_trainer_fit(trainer_.get())); }

    RObject resume(std::string checkpoint_path) {
        return value_to_r_owned(resolve_trainer_resume(trainer_.get(), checkpoint_path.c_str()));
    }

    void save(std::string path, Nullable<List> metadata = R_NilValue) {
        if (metadata.isNotNull()) {
            ValuePtr md = owned_value(r_list_to_value_map(as<List>(metadata), "metadata"));
            capi_check_status(resolve_trainer_save(trainer_.get(), path.c_str(), md.get()));
        } else {
            capi_check_status(resolve_trainer_save(trainer_.get(), path.c_str(), nullptr));
        }
    }

    RObject get_scalers()   const { return get("scalers"); }
    RObject get_config()    const { return get("config"); }
    RObject test_indices()  const { return get("test_indices"); }
    RObject train_indices() const { return get("train_indices"); }
    RObject test_plot_ids() const { return get("test_plot_ids"); }
    RObject train_plot_ids() const { return get("train_plot_ids"); }
    RObject effective_batch_size() const { return get("effective_batch_size"); }
    RObject run_metadata() const { return get("run_metadata"); }

    List categorical_vocab() const {
        ValuePtr v = owned_value(resolve_trainer_get(trainer_.get(), "categorical_vocab"));
        return categorical_vocab_value_to_r(v.get());
    }

    RObject compute_diagnostics() { return compute("diagnostics", nullptr, -1); }
    RObject compute_calibration(std::string target_name, int n_bins = 10) {
        return compute("calibration", target_name.c_str(), n_bins);
    }
    RObject compute_residuals(std::string target_name) {
        return compute("residuals", target_name.c_str(), -1);
    }
    RObject compute_classification_predictions(std::string target_name) {
        return compute("classification_predictions", target_name.c_str(), -1);
    }

    void load_state(std::string path, std::string device = "cpu", double vram_fraction = 1.0) {
        capi_check_status(resolve_trainer_load_state(
            trainer_.get(), path.c_str(), device.c_str(), vram_fraction));
    }

    RObject cross_validate(int n_folds = 5, int seed = 42) {
        return value_to_r_owned(resolve_trainer_cross_validate(trainer_.get(), n_folds, seed));
    }
    RObject cross_validate_spatial(List spatial_config_list, int n_folds = 5, int seed = 42) {
        ValuePtr cfg = owned_value(r_list_to_value_map(spatial_config_list, "spatialConfig"));
        return value_to_r_owned(resolve_trainer_cross_validate_spatial(
            trainer_.get(), cfg.get(), n_folds, seed));
    }

    RObject predict_from_trainer(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> pool_genus_ids = R_NilValue,
        Nullable<IntegerMatrix> pool_family_ids = R_NilValue,
        Nullable<NumericMatrix> pool_weights = R_NilValue,
        Nullable<IntegerMatrix> pool_mask = R_NilValue,
        Nullable<NumericVector> pool_has_cover = R_NilValue,
        Nullable<IntegerMatrix> categorical_ids = R_NilValue) {
        ValuePtr in = forward_inputs(continuous, genus_ids, family_ids, species_ids, species_vector,
                                     pool_genus_ids, pool_family_ids, pool_weights, pool_mask,
                                     pool_has_cover, categorical_ids);
        return value_to_r_owned(resolve_trainer_predict(trainer_.get(), in.get()));
    }

    // A checkpoint's persisted record ("train_config" or "run_metadata") as a
    // list, without loading the model.
    static List read_checkpoint(std::string what, std::string path) {
        capi_require_loaded();
        if (what == "train_config") {
            return as<List>(value_to_r_owned(resolve_load_train_config(path.c_str())));
        }
        if (what == "run_metadata") {
            return as<List>(value_to_r_owned(resolve_load_run_metadata(path.c_str())));
        }
        stop("read_checkpoint: unknown record '%s'", what.c_str());
    }

private:
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_trainer_get(trainer_.get(), what));
    }
    // One evaluator of resolve_trainer_compute; a NULL target and a negative
    // bin count leave the argument out.
    RObject compute(const char* kind, const char* target_name, int n_bins) {
        ValuePtr args = owned_value(resolve_value_new_map());
        if (target_name != nullptr) {
            capi_check_status(resolve_map_set_string(args.get(), "target_name", target_name));
        }
        if (n_bins >= 0) capi_check_status(resolve_map_set_int(args.get(), "n_bins", n_bins));
        return value_to_r_owned(resolve_trainer_compute(trainer_.get(), kind, args.get()));
    }
    static void set_targets(resolve_value_t* in, List targets) {
        if (targets.size() > 0 && Rf_isNull(targets.names())) {
            stop("set_targets: targets must be a named list of numeric vectors");
        }
        ValuePtr tmap = owned_value(resolve_value_new_map());
        CharacterVector names = targets.names();
        for (R_xlen_t i = 0; i < targets.size(); ++i) {
            NumericVector v = as<NumericVector>(targets[i]);
            std::vector<double> buf(v.begin(), v.end());
            capi_check_status(resolve_map_set_double_array(tmap.get(), r_string(names[i]).c_str(),
                                                           buf.data(),
                                                           static_cast<int64_t>(buf.size())));
        }
        map_put(in, "targets", std::move(tmap));
    }

    std::shared_ptr<resolve_trainer_t> trainer_;
};

#endif // RCPP_TRAINER_H
