// rcpp_model.h - RResolveModel wrapper (thin C-facade client).
#ifndef RCPP_MODEL_H
#define RCPP_MODEL_H

#include "rcpp_common.h"

class RResolveModel {
public:
    RResolveModel(List schema_list, List config_list) {
        ValuePtr schema(r_list_to_value_map(schema_list));
        ValuePtr config(r_list_to_value_map(config_list));
        model_ = capi_own(resolve_model_create(schema.get(), config.get()), resolve_model_free);
    }

    RObject forward(
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
        return call("forward", continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                    categorical_ids);
    }

    RObject get_latent(
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
        return call("get_latent", continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                    categorical_ids);
    }

    RObject forward_with_aux(
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
        return call("forward_with_aux", continuous, genus_ids, family_ids, species_ids, species_vector,
                    pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                    categorical_ids);
    }

    RObject forward_single(
        const std::string& target,
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue,
        Nullable<IntegerMatrix> categorical_ids = R_NilValue) {
        ValuePtr in(resolve_value_new_map());
        resolve_map_set_string(in.get(), "target", target.c_str());
        map_set_num_matrix(in.get(), "continuous", continuous);
        map_set_opt_int_matrix(in.get(), "genus_ids", genus_ids);
        map_set_opt_int_matrix(in.get(), "family_ids", family_ids);
        map_set_opt_int_matrix(in.get(), "species_ids", species_ids);
        map_set_opt_num_matrix(in.get(), "species_vector", species_vector);
        map_set_opt_int_matrix(in.get(), "categorical_ids", categorical_ids);
        return value_to_r_owned(resolve_model_call(model_.get(), "forward_single", in.get()));
    }

    RObject encode_with_activations(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        Nullable<IntegerMatrix> categorical_ids = R_NilValue) {
        ValuePtr in(resolve_value_new_map());
        map_set_num_matrix(in.get(), "continuous", continuous);
        map_set_opt_int_matrix(in.get(), "genus_ids", genus_ids);
        map_set_opt_int_matrix(in.get(), "family_ids", family_ids);
        map_set_opt_int_matrix(in.get(), "categorical_ids", categorical_ids);
        return value_to_r_owned(resolve_model_call(model_.get(), "encode_with_activations", in.get()));
    }

    RObject get_gate_probs(
        NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue) {
        ValuePtr in(resolve_value_new_map());
        map_set_num_matrix(in.get(), "continuous", continuous);
        map_set_opt_int_matrix(in.get(), "genus_ids", genus_ids);
        map_set_opt_int_matrix(in.get(), "family_ids", family_ids);
        return value_to_r_owned(resolve_model_call(model_.get(), "get_gate_probs", in.get()));
    }

    void train(bool mode = true) { capi_check_status(resolve_model_set_train(model_.get(), mode ? 1 : 0)); }
    void eval() { capi_check_status(resolve_model_set_train(model_.get(), 0)); }
    void to_device(std::string device) {
        capi_check_status(resolve_model_to_device(model_.get(), device.c_str()));
    }
    void set_traits(NumericMatrix traits) {
        int nr = traits.nrow(), nc = traits.ncol();
        std::vector<double> buf(static_cast<size_t>(nr) * nc);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j) buf[static_cast<size_t>(i) * nc + j] = traits(i, j);
        ValuePtr t(resolve_value_new_double_matrix(buf.data(), nr, nc));
        capi_check_status(resolve_model_set_traits(model_.get(), t.get()));
    }

    RObject latent_dim()           const { return get("latent_dim"); }
    RObject species_encoding()     const { return get("species_encoding"); }
    RObject uses_explicit_vector() const { return get("uses_explicit_vector"); }
    RObject uses_moe()             const { return get("uses_moe"); }
    RObject n_experts()            const { return get("n_experts"); }
    RObject get_genus_weights()    const { return get("genus_weights"); }
    RObject get_family_weights()   const { return get("family_weights"); }
    RObject get_species_weights()  const { return get("species_weights"); }

    resolve_model_t* handle() const { return model_.get(); }

private:
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_model_get(model_.get(), what));
    }
    RObject call(
        const char* method, NumericMatrix continuous,
        Nullable<IntegerMatrix> genus_ids, Nullable<IntegerMatrix> family_ids,
        Nullable<IntegerMatrix> species_ids, Nullable<NumericMatrix> species_vector,
        Nullable<IntegerMatrix> pool_genus_ids, Nullable<IntegerMatrix> pool_family_ids,
        Nullable<NumericMatrix> pool_weights, Nullable<IntegerMatrix> pool_mask,
        Nullable<NumericVector> pool_has_cover, Nullable<IntegerMatrix> categorical_ids) {
        ValuePtr in(resolve_value_new_map());
        fill_forward_inputs(in.get(), continuous, genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                            categorical_ids);
        return value_to_r_owned(resolve_model_call(model_.get(), method, in.get()));
    }

    std::shared_ptr<resolve_model_t> model_;
};

#endif // RCPP_MODEL_H
