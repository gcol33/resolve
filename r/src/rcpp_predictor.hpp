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
    static RPredictor load(std::string path, std::string device = "cpu") {
        torch::Device dev = (device == "cuda") ? torch::kCUDA : torch::kCPU;
        return RPredictor(resolve::Predictor::load(path, dev));
    }

    List predict(
        NumericMatrix coordinates,
        NumericMatrix covariates,
        NumericMatrix hash_embedding,
        Nullable<IntegerMatrix> genus_ids = R_NilValue,
        Nullable<IntegerMatrix> family_ids = R_NilValue,
        bool return_latent = false
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

        resolve::ResolvePredictions preds = predictor_.predict(
            coords_t, covs_t, hash_t, genus_t, family_t, return_latent
        );

        List result;
        for (const auto& [name, tensor] : preds.predictions) {
            result[name] = tensor_to_r_vec(tensor);
        }
        if (return_latent && preds.latent.defined()) {
            result["latent"] = tensor_to_r_mat(preds.latent);
        }
        result["plot_ids"] = wrap(preds.plot_ids);

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

    // Predict from ResolveDataset (matches Python API)
    List predict_dataset(
        RResolveDataset& dataset,
        bool return_latent = false
    ) {
        resolve::ResolvePredictions preds = predictor_.predict(*(dataset.dataset()), return_latent);

        List result;
        for (const auto& [name, tensor] : preds.predictions) {
            result[name] = tensor_to_r_vec(tensor);
        }
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
