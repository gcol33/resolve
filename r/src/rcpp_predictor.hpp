// rcpp_predictor.hpp - RPredictor class wrapper
#ifndef RCPP_PREDICTOR_HPP
#define RCPP_PREDICTOR_HPP

#include "rcpp_common.hpp"
#include "rcpp_dataset.hpp"

// =============================================================================
// Predictor class wrapper
// =============================================================================

class RPredictor {
public:
    static RPredictor load(
        std::string path,
        std::string device = "cpu",
        double vram_fraction = 1.0
    ) {
        torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
        return RPredictor(resolve::Predictor::load(
            path, dev, static_cast<float>(vram_fraction)));
    }

    List predict(
        NumericMatrix coordinates,
        NumericMatrix covariates,
        NumericMatrix hash_embedding,
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
        bool return_latent = false
    ) {
        torch::Tensor coords_t = r_mat_to_tensor(coordinates);
        torch::Tensor covs_t = r_mat_to_tensor(covariates);
        torch::Tensor hash_t = r_mat_to_tensor(hash_embedding);
        torch::Tensor species_id_t, species_vec_t, genus_t, family_t, uf_t, uc_t;
        torch::Tensor pg, pf, pw, pm, ph;

        if (species_ids.isNotNull()) species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        if (species_vector.isNotNull()) species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        if (genus_ids.isNotNull()) genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        if (family_ids.isNotNull()) family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        if (unknown_fraction.isNotNull()) uf_t = r_vec_to_tensor(as<NumericVector>(unknown_fraction));
        if (unknown_count.isNotNull()) uc_t = r_vec_to_tensor(as<NumericVector>(unknown_count));
        if (pool_genus_ids.isNotNull()) pg = r_int_mat_to_tensor(as<IntegerMatrix>(pool_genus_ids));
        if (pool_family_ids.isNotNull()) pf = r_int_mat_to_tensor(as<IntegerMatrix>(pool_family_ids));
        if (pool_weights.isNotNull()) pw = r_mat_to_tensor(as<NumericMatrix>(pool_weights));
        if (pool_mask.isNotNull()) pm = r_int_mat_to_tensor(as<IntegerMatrix>(pool_mask)).to(torch::kBool);
        if (pool_has_cover.isNotNull()) ph = r_vec_to_tensor(as<NumericVector>(pool_has_cover));

        auto preds = predictor_.predict(coords_t, covs_t, hash_t, species_id_t, species_vec_t,
                                         genus_t, family_t, uf_t, uc_t,
                                         pg, pf, pw, pm, ph, return_latent);

        List result;
        List predictions;
        for (const auto& [name, tensor] : preds.predictions) {
            predictions[name] = tensor_to_r_vec(tensor);
        }
        result["predictions"] = predictions;

        List targets;
        for (const auto& [name, tensor] : preds.targets) {
            targets[name] = tensor_to_r_vec(tensor);
        }
        result["targets"] = targets;

        result["plot_ids"] = wrap(preds.plot_ids);
        if (return_latent && preds.latent.defined()) {
            result["latent"] = tensor_to_r_mat(preds.latent);
        }
        return result;
    }

    NumericMatrix get_embeddings(
        NumericMatrix coordinates,
        NumericMatrix covariates,
        NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue
    ) {
        torch::Tensor coords_t = r_mat_to_tensor(coordinates);
        torch::Tensor covs_t = r_mat_to_tensor(covariates);
        torch::Tensor hash_t = r_mat_to_tensor(hash_embedding);

        torch::Tensor genus_t, family_t;
        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }

        torch::Tensor emb = predictor_.get_embeddings(coords_t, covs_t, hash_t, genus_t, family_t);
        return tensor_to_r_mat(emb);
    }

    NumericMatrix get_genus_embeddings() {
        return tensor_to_r_mat(predictor_.get_genus_embeddings());
    }

    NumericMatrix get_family_embeddings() {
        return tensor_to_r_mat(predictor_.get_family_embeddings());
    }

    NumericMatrix get_species_embeddings() {
        auto t = predictor_.get_species_embeddings();
        return tensor_to_r_mat(t);
    }

    void optimize_for_inference() {
        predictor_.optimize_for_inference();
    }

    std::string device() const {
        return predictor_.device().is_cuda() ? "cuda" : "cpu";
    }

    List get_scalers() const {
        return scalers_to_list(predictor_.scalers());
    }

    // Predict from ResolveDataset (matches Python API)
    List predict_dataset(
        RResolveDataset& dataset,
        bool return_latent = false
    ) {
        resolve::ResolvePredictions preds = predictor_.predict(*(dataset.dataset()), return_latent);

        List result;
        List predictions;
        for (const auto& [name, tensor] : preds.predictions) {
            predictions[name] = tensor_to_r_vec(tensor);
        }
        result["predictions"] = predictions;

        List targets;
        for (const auto& [name, tensor] : preds.targets) {
            targets[name] = tensor_to_r_vec(tensor);
        }
        result["targets"] = targets;

        if (return_latent && preds.latent.defined()) {
            result["latent"] = tensor_to_r_mat(preds.latent);
        }
        result["plot_ids"] = wrap(preds.plot_ids);

        return result;
    }

private:
    RPredictor(resolve::Predictor pred) : predictor_(std::move(pred)) {}
    resolve::Predictor predictor_;
};

#endif // RCPP_PREDICTOR_HPP
