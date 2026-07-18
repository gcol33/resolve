// rcpp_trainer.h - RTrainer wrapper (thin C-facade client).
#ifndef RCPP_TRAINER_H
#define RCPP_TRAINER_H

#include "rcpp_common.h"
#include "rcpp_model.h"
#include "rcpp_dataset.h"

class RTrainer {
public:
    RTrainer(RResolveModel& model, List config_list) {
        ValuePtr config(r_list_to_value_map(config_list));
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
        ValuePtr in(resolve_value_new_map());
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
        ValuePtr in(resolve_value_new_map());
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

    RObject fit() { return value_to_r_owned(resolve_trainer_fit(trainer_.get())); }

    void save(std::string path, Nullable<List> metadata = R_NilValue) {
        if (metadata.isNotNull()) {
            ValuePtr md(r_list_to_value_map(as<List>(metadata)));
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

    List categorical_vocab() const {
        ValuePtr v(resolve_trainer_get(trainer_.get(), "categorical_vocab"));
        capi_check(v.get());
        return categorical_vocab_value_to_r(v.get());
    }

    RObject compute_diagnostics() {
        ValuePtr args(resolve_value_new_map());
        return value_to_r_owned(resolve_trainer_compute(trainer_.get(), "diagnostics", args.get()));
    }
    RObject compute_calibration(std::string target_name, int n_bins = 10) {
        ValuePtr args(resolve_value_new_map());
        resolve_map_set_string(args.get(), "target_name", target_name.c_str());
        resolve_map_set_int(args.get(), "n_bins", n_bins);
        return value_to_r_owned(resolve_trainer_compute(trainer_.get(), "calibration", args.get()));
    }
    RObject compute_residuals(std::string target_name) {
        ValuePtr args(resolve_value_new_map());
        resolve_map_set_string(args.get(), "target_name", target_name.c_str());
        return value_to_r_owned(resolve_trainer_compute(trainer_.get(), "residuals", args.get()));
    }
    RObject compute_classification_predictions(std::string target_name) {
        ValuePtr args(resolve_value_new_map());
        resolve_map_set_string(args.get(), "target_name", target_name.c_str());
        return value_to_r_owned(resolve_trainer_compute(
            trainer_.get(), "classification_predictions", args.get()));
    }

    void load_state(std::string path, std::string device = "cpu", double vram_fraction = 1.0) {
        capi_check_status(resolve_trainer_load_state(
            trainer_.get(), path.c_str(), device.c_str(), vram_fraction));
    }

    RObject cross_validate(int n_folds = 5, int seed = 42) {
        return value_to_r_owned(resolve_trainer_cross_validate(trainer_.get(), n_folds, seed));
    }
    RObject cross_validate_spatial(List spatial_config_list, int n_folds = 5, int seed = 42) {
        ValuePtr cfg(r_list_to_value_map(spatial_config_list));
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
        ValuePtr in(resolve_value_new_map());
        fill_forward_inputs(in.get(), continuous, genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                            categorical_ids);
        return value_to_r_owned(resolve_trainer_predict(trainer_.get(), in.get()));
    }

    static List load_train_config(std::string path) {
        return as<List>(value_to_r_owned(resolve_load_train_config(path.c_str())));
    }
    static List load_run_metadata(std::string path) {
        return as<List>(value_to_r_owned(resolve_load_run_metadata(path.c_str())));
    }

private:
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_trainer_get(trainer_.get(), what));
    }
    static void set_targets(resolve_value_t* in, List targets) {
        if (targets.size() > 0 && Rf_isNull(targets.names())) {
            stop("set_targets: `targets` must be a named list of numeric vectors");
        }
        // Attach the map to `in` (caller owns `in` via ValuePtr) BEFORE filling
        // it, so a throw from as<NumericVector> on a non-numeric target frees the
        // partial map through `in` instead of leaking it.
        resolve_value_t* tmap = resolve_value_new_map();
        resolve_map_set_value(in, "targets", tmap);
        CharacterVector names = targets.names();
        for (R_xlen_t i = 0; i < targets.size(); ++i) {
            SEXP nm_i = STRING_ELT(names, i);
            std::string name = (nm_i == NA_STRING) ? std::string() : std::string(CHAR(nm_i));
            NumericVector v = as<NumericVector>(targets[i]);
            std::vector<double> buf(v.begin(), v.end());
            resolve_map_set_double_array(tmap, name.c_str(), buf.data(),
                                         static_cast<int64_t>(buf.size()));
        }
    }

    std::shared_ptr<resolve_trainer_t> trainer_;
};

#endif // RCPP_TRAINER_H
