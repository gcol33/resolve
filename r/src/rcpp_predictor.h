// rcpp_predictor.h - RPredictor wrapper (thin C-facade client).
#ifndef RCPP_PREDICTOR_H
#define RCPP_PREDICTOR_H

#include "rcpp_common.h"
#include "rcpp_dataset.h"

class RPredictor {
public:
    static RPredictor load(std::string path, std::string device = "cpu", double vram_fraction = 1.0) {
        RPredictor p;
        p.predictor_ = capi_own(resolve_predictor_load(
            path.c_str(), device.c_str(), vram_fraction), resolve_predictor_free);
        return p;
    }

    RObject predict(
        NumericMatrix coordinates, NumericMatrix covariates, NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<NumericVector> unknown_fraction = R_NilValue,
        Nullable<NumericVector> unknown_count = R_NilValue,
        Nullable<IntegerMatrix> pool_genus_ids = R_NilValue,
        Nullable<IntegerMatrix> pool_family_ids = R_NilValue,
        Nullable<NumericMatrix> pool_weights = R_NilValue,
        Nullable<IntegerMatrix> pool_mask = R_NilValue,
        Nullable<NumericVector> pool_has_cover = R_NilValue,
        Nullable<IntegerMatrix> categorical_ids = R_NilValue,
        bool return_latent = false) {
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
        map_set_opt_int_matrix(in.get(), "pool_genus_ids", pool_genus_ids);
        map_set_opt_int_matrix(in.get(), "pool_family_ids", pool_family_ids);
        map_set_opt_num_matrix(in.get(), "pool_weights", pool_weights);
        map_set_opt_int_matrix(in.get(), "pool_mask", pool_mask);
        map_set_opt_num_vector(in.get(), "pool_has_cover", pool_has_cover);
        map_set_opt_int_matrix(in.get(), "categorical_ids", categorical_ids);
        return value_to_r_owned(resolve_predictor_predict(
            predictor_.get(), in.get(), return_latent ? 1 : 0));
    }

    RObject get_embeddings(
        NumericMatrix coordinates, NumericMatrix covariates, NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue) {
        ValuePtr in(resolve_value_new_map());
        map_set_num_matrix(in.get(), "coordinates", coordinates);
        map_set_num_matrix(in.get(), "covariates", covariates);
        map_set_num_matrix(in.get(), "hash_embedding", hash_embedding);
        map_set_opt_int_matrix(in.get(), "genus_ids", genus_ids);
        map_set_opt_int_matrix(in.get(), "family_ids", family_ids);
        return value_to_r_owned(resolve_predictor_get_embeddings(predictor_.get(), in.get()));
    }

    RObject get_genus_embeddings()   { return get("genus_embeddings"); }
    RObject get_family_embeddings()  { return get("family_embeddings"); }
    RObject get_species_embeddings() { return get("species_embeddings"); }

    void optimize_for_inference() {
        capi_check_status(resolve_predictor_optimize_for_inference(predictor_.get()));
    }

    RObject device()      const { return get("device"); }
    RObject get_scalers() const { return get("scalers"); }

    List categorical_vocab() const {
        ValuePtr v(resolve_predictor_get(predictor_.get(), "categorical_vocab"));
        capi_check(v.get());
        return categorical_vocab_value_to_r(v.get());
    }

    RObject predict_dataset(RResolveDataset& dataset, bool return_latent = false,
                            int64_t batch_size = 4096) {
        return value_to_r_owned(resolve_predictor_predict_dataset(
            predictor_.get(), dataset.handle(), return_latent ? 1 : 0, batch_size));
    }

private:
    RPredictor() = default;
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_predictor_get(predictor_.get(), what));
    }
    std::shared_ptr<resolve_predictor_t> predictor_;
};

#endif // RCPP_PREDICTOR_H
