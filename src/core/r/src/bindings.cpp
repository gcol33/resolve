// RESOLVE R bindings using Rcpp
// Links to libtorch for neural network operations

#include <Rcpp.h>
#include <torch/torch.h>

// Include resolve headers
#include "resolve/types.hpp"
#include "resolve/model.hpp"
#include "resolve/trainer.hpp"
#include "resolve/predictor.hpp"
#include "resolve/loss.hpp"
#include "resolve/dataset.hpp"
#include "resolve/role_mapping.hpp"

using namespace Rcpp;

// ============================================================================
// Helper functions for R <-> Torch conversion
// ============================================================================

// Convert R numeric vector to torch tensor
torch::Tensor r_to_tensor_1d(NumericVector x) {
    std::vector<float> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(data.size())},
                           torch::kFloat32).clone();
}

// Convert R numeric matrix to torch tensor
torch::Tensor r_to_tensor_2d(NumericMatrix x) {
    int64_t rows = x.nrow();
    int64_t cols = x.ncol();
    std::vector<float> data(rows * cols);

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            data[i * cols + j] = x(i, j);
        }
    }

    return torch::from_blob(data.data(), {rows, cols}, torch::kFloat32).clone();
}

// Convert R integer matrix to torch tensor (int64)
torch::Tensor r_to_tensor_int64(IntegerMatrix x) {
    int64_t rows = x.nrow();
    int64_t cols = x.ncol();
    std::vector<int64_t> data(rows * cols);

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            data[i * cols + j] = x(i, j);
        }
    }

    return torch::from_blob(data.data(), {rows, cols}, torch::kInt64).clone();
}

// Convert torch tensor to R numeric vector
NumericVector tensor_to_r_1d(const torch::Tensor& t) {
    auto t_cpu = t.to(torch::kCPU).contiguous().to(torch::kFloat32);
    auto t_flat = t_cpu.view({-1});
    auto accessor = t_flat.accessor<float, 1>();

    NumericVector result(t_flat.size(0));
    for (int64_t i = 0; i < t_flat.size(0); ++i) {
        result[i] = accessor[i];
    }
    return result;
}

// Convert torch tensor to R numeric matrix
NumericMatrix tensor_to_r_2d(const torch::Tensor& t) {
    auto t_cpu = t.to(torch::kCPU).contiguous().to(torch::kFloat32);

    int64_t rows = t_cpu.size(0);
    int64_t cols = t_cpu.size(1);

    NumericMatrix result(rows, cols);
    auto accessor = t_cpu.accessor<float, 2>();

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            result(i, j) = accessor[i][j];
        }
    }
    return result;
}

// ============================================================================
// Metrics (simple exported functions)
// ============================================================================

//' Compute band accuracy
//' @param pred Numeric vector of predictions
//' @param target Numeric vector of targets
//' @param threshold Band threshold (default 0.25)
//' @return Band accuracy as double
//' @export
// [[Rcpp::export]]
double resolve_band_accuracy(NumericVector pred, NumericVector target,
                             double threshold = 0.25) {
    return resolve::Metrics::band_accuracy(r_to_tensor_1d(pred),
                                           r_to_tensor_1d(target),
                                           threshold);
}

//' Compute mean absolute error
//' @param pred Numeric vector of predictions
//' @param target Numeric vector of targets
//' @return MAE as double
//' @export
// [[Rcpp::export]]
double resolve_mae(NumericVector pred, NumericVector target) {
    return resolve::Metrics::mae(r_to_tensor_1d(pred), r_to_tensor_1d(target));
}

//' Compute root mean squared error
//' @param pred Numeric vector of predictions
//' @param target Numeric vector of targets
//' @return RMSE as double
//' @export
// [[Rcpp::export]]
double resolve_rmse(NumericVector pred, NumericVector target) {
    return resolve::Metrics::rmse(r_to_tensor_1d(pred), r_to_tensor_1d(target));
}

//' Compute symmetric mean absolute percentage error
//' @param pred Numeric vector of predictions
//' @param target Numeric vector of targets
//' @param eps Small value to avoid division by zero
//' @return SMAPE as double
//' @export
// [[Rcpp::export]]
double resolve_smape(NumericVector pred, NumericVector target,
                     double eps = 1e-8) {
    return resolve::Metrics::smape(r_to_tensor_1d(pred),
                                   r_to_tensor_1d(target), eps);
}

// ============================================================================
// Model creation and inference (using XPtr for class instances)
// ============================================================================

//' Create a new ResolveModel
//' @param schema_list List with schema parameters
//' @param config_list List with model config parameters
//' @return External pointer to model
//' @export
// [[Rcpp::export]]
SEXP resolve_model_new(List schema_list, List config_list) {
    // Parse schema
    resolve::ResolveSchema schema;
    schema.n_plots = as<int64_t>(schema_list["n_plots"]);
    schema.n_species = as<int64_t>(schema_list["n_species"]);

    if (schema_list.containsElementNamed("n_species_vocab")) {
        schema.n_species_vocab = as<int64_t>(schema_list["n_species_vocab"]);
    }
    if (schema_list.containsElementNamed("has_coordinates")) {
        schema.has_coordinates = as<bool>(schema_list["has_coordinates"]);
    }
    if (schema_list.containsElementNamed("has_abundance")) {
        schema.has_abundance = as<bool>(schema_list["has_abundance"]);
    }
    if (schema_list.containsElementNamed("has_taxonomy")) {
        schema.has_taxonomy = as<bool>(schema_list["has_taxonomy"]);
    }
    if (schema_list.containsElementNamed("n_genera")) {
        schema.n_genera = as<int64_t>(schema_list["n_genera"]);
    }
    if (schema_list.containsElementNamed("n_families")) {
        schema.n_families = as<int64_t>(schema_list["n_families"]);
    }

    if (schema_list.containsElementNamed("covariate_names")) {
        CharacterVector cov_names = schema_list["covariate_names"];
        for (int i = 0; i < cov_names.size(); ++i) {
            schema.covariate_names.push_back(as<std::string>(cov_names[i]));
        }
    }

    // Parse targets
    if (schema_list.containsElementNamed("targets")) {
        List targets_list = schema_list["targets"];
        for (int i = 0; i < targets_list.size(); ++i) {
            List target = targets_list[i];
            resolve::TargetConfig cfg;
            cfg.name = as<std::string>(target["name"]);
            std::string task_str = as<std::string>(target["task"]);
            cfg.task = (task_str == "regression")
                ? resolve::TaskType::Regression
                : resolve::TaskType::Classification;
            if (target.containsElementNamed("transform")) {
                std::string transform_str = as<std::string>(target["transform"]);
                cfg.transform = (transform_str == "log1p")
                    ? resolve::TransformType::Log1p
                    : resolve::TransformType::None;
            }
            if (target.containsElementNamed("num_classes")) {
                cfg.num_classes = as<int>(target["num_classes"]);
            }
            if (target.containsElementNamed("weight")) {
                cfg.weight = as<float>(target["weight"]);
            }
            schema.targets.push_back(cfg);
        }
    }

    // Parse model config
    resolve::ModelConfig config;
    if (config_list.containsElementNamed("hash_dim")) {
        config.hash_dim = as<int>(config_list["hash_dim"]);
    }
    if (config_list.containsElementNamed("genus_emb_dim")) {
        config.genus_emb_dim = as<int>(config_list["genus_emb_dim"]);
    }
    if (config_list.containsElementNamed("family_emb_dim")) {
        config.family_emb_dim = as<int>(config_list["family_emb_dim"]);
    }
    if (config_list.containsElementNamed("top_k")) {
        config.top_k = as<int>(config_list["top_k"]);
    }
    if (config_list.containsElementNamed("hidden_dims")) {
        IntegerVector hd = config_list["hidden_dims"];
        config.hidden_dims.clear();
        for (int i = 0; i < hd.size(); ++i) {
            config.hidden_dims.push_back(hd[i]);
        }
    }
    if (config_list.containsElementNamed("dropout")) {
        config.dropout = as<float>(config_list["dropout"]);
    }

    // Create model
    auto* model = new resolve::ResolveModel(schema, config);
    return XPtr<resolve::ResolveModel>(model);
}

//' Forward pass through model
//' @param model_ptr External pointer to model
//' @param continuous Numeric matrix of continuous features
//' @param genus_ids Integer matrix of genus IDs (optional)
//' @param family_ids Integer matrix of family IDs (optional)
//' @return List of predictions
//' @export
// [[Rcpp::export]]
List resolve_model_forward(SEXP model_ptr, NumericMatrix continuous,
                           Nullable<IntegerMatrix> genus_ids = R_NilValue,
                           Nullable<IntegerMatrix> family_ids = R_NilValue) {
    XPtr<resolve::ResolveModel> model(model_ptr);

    auto continuous_t = r_to_tensor_2d(continuous);
    torch::Tensor genus_ids_t, family_ids_t;

    if (genus_ids.isNotNull()) {
        genus_ids_t = r_to_tensor_int64(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_to_tensor_int64(as<IntegerMatrix>(family_ids));
    }

    auto outputs = (*model)->forward(continuous_t, genus_ids_t, family_ids_t,
                                     torch::Tensor(), torch::Tensor());

    List result;
    for (const auto& [name, tensor] : outputs) {
        result[name] = tensor_to_r_1d(tensor);
    }
    return result;
}

//' Get latent embeddings from model
//' @param model_ptr External pointer to model
//' @param continuous Numeric matrix of continuous features
//' @param genus_ids Integer matrix of genus IDs (optional)
//' @param family_ids Integer matrix of family IDs (optional)
//' @return Numeric matrix of latent embeddings
//' @export
// [[Rcpp::export]]
NumericMatrix resolve_model_get_latent(SEXP model_ptr, NumericMatrix continuous,
                                       Nullable<IntegerMatrix> genus_ids = R_NilValue,
                                       Nullable<IntegerMatrix> family_ids = R_NilValue) {
    XPtr<resolve::ResolveModel> model(model_ptr);

    auto continuous_t = r_to_tensor_2d(continuous);
    torch::Tensor genus_ids_t, family_ids_t;

    if (genus_ids.isNotNull()) {
        genus_ids_t = r_to_tensor_int64(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_to_tensor_int64(as<IntegerMatrix>(family_ids));
    }

    auto latent = (*model)->get_latent(continuous_t, genus_ids_t, family_ids_t,
                                       torch::Tensor(), torch::Tensor());
    return tensor_to_r_2d(latent);
}

// ============================================================================
// Trainer
// ============================================================================

//' Create a new Trainer
//' @param model_ptr External pointer to ResolveModel
//' @param config_list List with training config parameters
//' @return External pointer to Trainer
//' @export
// [[Rcpp::export]]
SEXP resolve_trainer_new(SEXP model_ptr, List config_list) {
    XPtr<resolve::ResolveModel> model(model_ptr);

    resolve::TrainConfig config;
    if (config_list.containsElementNamed("batch_size")) {
        config.batch_size = as<int>(config_list["batch_size"]);
    }
    if (config_list.containsElementNamed("max_epochs")) {
        config.max_epochs = as<int>(config_list["max_epochs"]);
    }
    if (config_list.containsElementNamed("patience")) {
        config.patience = as<int>(config_list["patience"]);
    }
    if (config_list.containsElementNamed("lr")) {
        config.lr = as<float>(config_list["lr"]);
    }
    if (config_list.containsElementNamed("weight_decay")) {
        config.weight_decay = as<float>(config_list["weight_decay"]);
    }

    auto* trainer = new resolve::Trainer(*model, config);
    return XPtr<resolve::Trainer>(trainer);
}

//' Prepare training data
//' @param trainer_ptr External pointer to Trainer
//' @param continuous Numeric matrix of continuous features
//' @param targets_list Named list of numeric vectors (targets)
//' @param genus_ids Integer matrix of genus IDs (optional)
//' @param family_ids Integer matrix of family IDs (optional)
//' @param test_size Fraction for test set (default 0.2)
//' @param seed Random seed
//' @export
// [[Rcpp::export]]
void resolve_trainer_prepare_data(
    SEXP trainer_ptr,
    NumericMatrix continuous,
    List targets_list,
    Nullable<IntegerMatrix> genus_ids = R_NilValue,
    Nullable<IntegerMatrix> family_ids = R_NilValue,
    double test_size = 0.2,
    int seed = 42
) {
    XPtr<resolve::Trainer> trainer(trainer_ptr);

    auto continuous_t = r_to_tensor_2d(continuous);

    torch::Tensor genus_ids_t, family_ids_t;
    if (genus_ids.isNotNull()) {
        genus_ids_t = r_to_tensor_int64(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_to_tensor_int64(as<IntegerMatrix>(family_ids));
    }

    // Convert targets list to map
    std::unordered_map<std::string, torch::Tensor> targets;
    CharacterVector names = targets_list.names();
    for (int i = 0; i < targets_list.size(); ++i) {
        std::string name = as<std::string>(names[i]);
        targets[name] = r_to_tensor_1d(as<NumericVector>(targets_list[i]));
    }

    trainer->prepare_data(
        torch::Tensor(),        // coordinates (optional)
        torch::Tensor(),        // covariates (optional)
        continuous_t,           // hash_embedding
        torch::Tensor(),        // species_ids
        torch::Tensor(),        // species_vector
        genus_ids_t,
        family_ids_t,
        torch::Tensor(),        // unknown_fraction
        torch::Tensor(),        // unknown_count
        targets,
        static_cast<float>(test_size),
        seed
    );
}

//' Train the model
//' @param trainer_ptr External pointer to Trainer
//' @return List with training results
//' @export
// [[Rcpp::export]]
List resolve_trainer_fit(SEXP trainer_ptr) {
    XPtr<resolve::Trainer> trainer(trainer_ptr);

    auto result = trainer->fit();

    // Convert metrics to R list
    List final_metrics;
    for (const auto& [target_name, target_metrics] : result.final_metrics) {
        List metrics_list;
        for (const auto& [metric_name, value] : target_metrics) {
            metrics_list[metric_name] = value;
        }
        final_metrics[target_name] = metrics_list;
    }

    return List::create(
        Named("best_epoch") = result.best_epoch,
        Named("final_metrics") = final_metrics,
        Named("train_loss_history") = wrap(result.train_loss_history),
        Named("test_loss_history") = wrap(result.test_loss_history),
        Named("train_time_seconds") = result.train_time_seconds
    );
}

//' Save trainer checkpoint
//' @param trainer_ptr External pointer to Trainer
//' @param path File path to save
//' @export
// [[Rcpp::export]]
void resolve_trainer_save(SEXP trainer_ptr, std::string path) {
    XPtr<resolve::Trainer> trainer(trainer_ptr);
    trainer->save(path);
}

//' Load trainer checkpoint
//' @param path File path to load
//' @return List with model_ptr and scalers
//' @export
// [[Rcpp::export]]
List resolve_trainer_load(std::string path) {
    auto [model, scalers] = resolve::Trainer::load(path);

    auto* model_ptr = new resolve::ResolveModel(std::move(model));
    auto* scalers_ptr = new resolve::Scalers(std::move(scalers));

    return List::create(
        Named("model") = XPtr<resolve::ResolveModel>(model_ptr),
        Named("scalers") = XPtr<resolve::Scalers>(scalers_ptr)
    );
}

// ============================================================================
// Predictor
// ============================================================================

//' Create a new Predictor
//' @param model_ptr External pointer to ResolveModel
//' @param scalers_ptr External pointer to Scalers
//' @return External pointer to Predictor
//' @export
// [[Rcpp::export]]
SEXP resolve_predictor_new(SEXP model_ptr, SEXP scalers_ptr) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    XPtr<resolve::Scalers> scalers(scalers_ptr);

    auto* predictor = new resolve::Predictor(*model, *scalers);
    return XPtr<resolve::Predictor>(predictor);
}

//' Load predictor from checkpoint
//' @param path File path to load
//' @return External pointer to Predictor
//' @export
// [[Rcpp::export]]
SEXP resolve_predictor_load(std::string path) {
    auto predictor = new resolve::Predictor(resolve::Predictor::load(path));
    return XPtr<resolve::Predictor>(predictor);
}

//' Predict with Predictor
//' @param predictor_ptr External pointer to Predictor
//' @param continuous Numeric matrix of continuous features
//' @param genus_ids Integer matrix of genus IDs (optional)
//' @param family_ids Integer matrix of family IDs (optional)
//' @param return_latent Whether to return latent embeddings
//' @return List of predictions
//' @export
// [[Rcpp::export]]
List resolve_predictor_predict(
    SEXP predictor_ptr,
    NumericMatrix continuous,
    Nullable<IntegerMatrix> genus_ids = R_NilValue,
    Nullable<IntegerMatrix> family_ids = R_NilValue,
    bool return_latent = false
) {
    XPtr<resolve::Predictor> predictor(predictor_ptr);

    auto continuous_t = r_to_tensor_2d(continuous);

    torch::Tensor genus_ids_t, family_ids_t;
    if (genus_ids.isNotNull()) {
        genus_ids_t = r_to_tensor_int64(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_to_tensor_int64(as<IntegerMatrix>(family_ids));
    }

    auto result = predictor->predict(
        torch::Tensor(),  // coordinates
        torch::Tensor(),  // covariates
        continuous_t,     // hash_embedding
        genus_ids_t,
        family_ids_t,
        return_latent
    );

    // Convert predictions to R list
    List predictions;
    for (const auto& [name, tensor] : result.predictions) {
        predictions[name] = tensor_to_r_1d(tensor);
    }

    List ret = List::create(Named("predictions") = predictions);
    if (return_latent && result.latent.defined()) {
        ret["latent"] = tensor_to_r_2d(result.latent);
    }

    return ret;
}

//' Get latent embeddings
//' @param predictor_ptr External pointer to Predictor
//' @param continuous Numeric matrix of continuous features
//' @param genus_ids Integer matrix of genus IDs (optional)
//' @param family_ids Integer matrix of family IDs (optional)
//' @return Numeric matrix of latent embeddings
//' @export
// [[Rcpp::export]]
NumericMatrix resolve_predictor_get_embeddings(
    SEXP predictor_ptr,
    NumericMatrix continuous,
    Nullable<IntegerMatrix> genus_ids = R_NilValue,
    Nullable<IntegerMatrix> family_ids = R_NilValue
) {
    XPtr<resolve::Predictor> predictor(predictor_ptr);

    auto continuous_t = r_to_tensor_2d(continuous);

    torch::Tensor genus_ids_t, family_ids_t;
    if (genus_ids.isNotNull()) {
        genus_ids_t = r_to_tensor_int64(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_to_tensor_int64(as<IntegerMatrix>(family_ids));
    }

    auto latent = predictor->get_embeddings(
        torch::Tensor(),  // coordinates
        torch::Tensor(),  // covariates
        continuous_t,     // hash_embedding
        genus_ids_t,
        family_ids_t
    );

    return tensor_to_r_2d(latent);
}

//' Get learned genus embeddings
//' @param predictor_ptr External pointer to Predictor
//' @return Numeric matrix of genus embeddings
//' @export
// [[Rcpp::export]]
NumericMatrix resolve_predictor_get_genus_embeddings(SEXP predictor_ptr) {
    XPtr<resolve::Predictor> predictor(predictor_ptr);
    return tensor_to_r_2d(predictor->get_genus_embeddings());
}

//' Get learned family embeddings
//' @param predictor_ptr External pointer to Predictor
//' @return Numeric matrix of family embeddings
//' @export
// [[Rcpp::export]]
NumericMatrix resolve_predictor_get_family_embeddings(SEXP predictor_ptr) {
    XPtr<resolve::Predictor> predictor(predictor_ptr);
    return tensor_to_r_2d(predictor->get_family_embeddings());
}

// ============================================================================
// Utilities
// ============================================================================

//' Check if CUDA is available
//' @return TRUE if CUDA is available
//' @export
// [[Rcpp::export]]
bool resolve_cuda_available() {
    return torch::cuda::is_available();
}

//' Get CUDA device count
//' @return Number of CUDA devices
//' @export
// [[Rcpp::export]]
int resolve_cuda_device_count() {
    return torch::cuda::device_count();
}

// ============================================================================
// Dataset Loading (Phase 1 API)
// ============================================================================

//' Load dataset from CSV files
//' @param header_path Path to header CSV file (plot-level data)
//' @param species_path Path to species CSV file (species occurrences)
//' @param roles_list Named list of role mappings
//' @param targets_list List of target specifications
//' @param config_list Dataset configuration
//' @return External pointer to ResolveDataset
//' @export
// [[Rcpp::export]]
SEXP resolve_dataset_from_csv(
    std::string header_path,
    std::string species_path,
    List roles_list,
    List targets_list,
    List config_list
) {
    using namespace resolve;

    // Parse role mapping
    RoleMapping roles;
    roles.plot_id = as<std::string>(roles_list["plot_id"]);
    roles.species_id = as<std::string>(roles_list["species_id"]);

    if (roles_list.containsElementNamed("abundance") && !Rf_isNull(roles_list["abundance"])) {
        roles.abundance = as<std::string>(roles_list["abundance"]);
    }
    if (roles_list.containsElementNamed("longitude") && !Rf_isNull(roles_list["longitude"])) {
        roles.longitude = as<std::string>(roles_list["longitude"]);
    }
    if (roles_list.containsElementNamed("latitude") && !Rf_isNull(roles_list["latitude"])) {
        roles.latitude = as<std::string>(roles_list["latitude"]);
    }
    if (roles_list.containsElementNamed("genus") && !Rf_isNull(roles_list["genus"])) {
        roles.genus = as<std::string>(roles_list["genus"]);
    }
    if (roles_list.containsElementNamed("family") && !Rf_isNull(roles_list["family"])) {
        roles.family = as<std::string>(roles_list["family"]);
    }

    // Parse target specifications
    std::vector<TargetSpec> targets;
    for (int i = 0; i < targets_list.size(); ++i) {
        List target = targets_list[i];
        TargetSpec spec;
        spec.column_name = as<std::string>(target["column"]);
        spec.target_name = target.containsElementNamed("name")
            ? as<std::string>(target["name"])
            : spec.column_name;

        std::string task_str = as<std::string>(target["task"]);
        spec.task = (task_str == "regression")
            ? TaskType::Regression
            : TaskType::Classification;

        if (target.containsElementNamed("transform")) {
            std::string transform_str = as<std::string>(target["transform"]);
            spec.transform = (transform_str == "log1p")
                ? TransformType::Log1p
                : TransformType::None;
        }
        if (target.containsElementNamed("num_classes")) {
            spec.num_classes = as<int>(target["num_classes"]);
        }

        targets.push_back(spec);
    }

    // Parse dataset configuration
    DatasetConfig config;
    if (config_list.containsElementNamed("species_encoding")) {
        std::string enc = as<std::string>(config_list["species_encoding"]);
        if (enc == "embed") {
            config.species_encoding = SpeciesEncodingMode::Embed;
        } else if (enc == "sparse") {
            config.species_encoding = SpeciesEncodingMode::Sparse;
        } else {
            config.species_encoding = SpeciesEncodingMode::Hash;
        }
    }
    if (config_list.containsElementNamed("hash_dim")) {
        config.hash_dim = as<int>(config_list["hash_dim"]);
    }
    if (config_list.containsElementNamed("top_k")) {
        config.top_k = as<int>(config_list["top_k"]);
    }
    if (config_list.containsElementNamed("top_k_species")) {
        config.top_k_species = as<int>(config_list["top_k_species"]);
    }

    // Load dataset
    auto* dataset = new ResolveDataset(
        header_path.empty()
            ? ResolveDataset::from_species_csv(species_path, roles, targets, config)
            : ResolveDataset::from_csv(header_path, species_path, roles, targets, config)
    );

    return XPtr<ResolveDataset>(dataset);
}

//' Load dataset from species CSV only
//' @param species_path Path to species CSV file
//' @param roles_list Named list of role mappings
//' @param targets_list List of target specifications
//' @param config_list Dataset configuration
//' @return External pointer to ResolveDataset
//' @export
// [[Rcpp::export]]
SEXP resolve_dataset_from_species_csv(
    std::string species_path,
    List roles_list,
    List targets_list,
    List config_list
) {
    return resolve_dataset_from_csv("", species_path, roles_list, targets_list, config_list);
}

//' Get dataset schema
//' @param dataset_ptr External pointer to ResolveDataset
//' @return List with schema information
//' @export
// [[Rcpp::export]]
List resolve_dataset_schema(SEXP dataset_ptr) {
    XPtr<resolve::ResolveDataset> dataset(dataset_ptr);
    const auto& schema = dataset->schema();

    List targets_list;
    for (const auto& target : schema.targets) {
        targets_list.push_back(List::create(
            Named("name") = target.name,
            Named("task") = (target.task == resolve::TaskType::Regression ? "regression" : "classification"),
            Named("transform") = (target.transform == resolve::TransformType::Log1p ? "log1p" : "none"),
            Named("num_classes") = target.num_classes,
            Named("weight") = target.weight
        ));
    }

    return List::create(
        Named("n_plots") = schema.n_plots,
        Named("n_species") = schema.n_species,
        Named("n_species_vocab") = schema.n_species_vocab,
        Named("has_coordinates") = schema.has_coordinates,
        Named("has_abundance") = schema.has_abundance,
        Named("has_taxonomy") = schema.has_taxonomy,
        Named("n_genera") = schema.n_genera,
        Named("n_families") = schema.n_families,
        Named("covariate_names") = wrap(schema.covariate_names),
        Named("targets") = targets_list
    );
}

//' Get number of plots in dataset
//' @param dataset_ptr External pointer to ResolveDataset
//' @return Number of plots
//' @export
// [[Rcpp::export]]
int64_t resolve_dataset_n_plots(SEXP dataset_ptr) {
    XPtr<resolve::ResolveDataset> dataset(dataset_ptr);
    return dataset->n_plots();
}

//' Get plot IDs from dataset
//' @param dataset_ptr External pointer to ResolveDataset
//' @return Character vector of plot IDs
//' @export
// [[Rcpp::export]]
CharacterVector resolve_dataset_plot_ids(SEXP dataset_ptr) {
    XPtr<resolve::ResolveDataset> dataset(dataset_ptr);
    return wrap(dataset->plot_ids());
}

//' Train with dataset (new API)
//' @param model_ptr External pointer to ResolveModel
//' @param dataset_ptr External pointer to ResolveDataset
//' @param config_list Training configuration
//' @param test_size Test set fraction
//' @param seed Random seed
//' @return List with training results
//' @export
// [[Rcpp::export]]
List resolve_train_with_dataset(
    SEXP model_ptr,
    SEXP dataset_ptr,
    List config_list,
    double test_size = 0.2,
    int seed = 42
) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    XPtr<resolve::ResolveDataset> dataset(dataset_ptr);

    resolve::TrainConfig config;
    if (config_list.containsElementNamed("batch_size")) {
        config.batch_size = as<int>(config_list["batch_size"]);
    }
    if (config_list.containsElementNamed("max_epochs")) {
        config.max_epochs = as<int>(config_list["max_epochs"]);
    }
    if (config_list.containsElementNamed("patience")) {
        config.patience = as<int>(config_list["patience"]);
    }
    if (config_list.containsElementNamed("lr")) {
        config.lr = as<float>(config_list["lr"]);
    }

    resolve::Trainer trainer(*model, config);
    trainer.prepare_data(*dataset, static_cast<float>(test_size), seed);

    auto result = trainer.fit();

    // Convert metrics to R list
    List final_metrics;
    for (const auto& [target_name, target_metrics] : result.final_metrics) {
        List metrics_list;
        for (const auto& [metric_name, value] : target_metrics) {
            metrics_list[metric_name] = value;
        }
        final_metrics[target_name] = metrics_list;
    }

    return List::create(
        Named("best_epoch") = result.best_epoch,
        Named("final_metrics") = final_metrics,
        Named("train_loss_history") = wrap(result.train_loss_history),
        Named("test_loss_history") = wrap(result.test_loss_history),
        Named("train_time_seconds") = result.train_time_seconds
    );
}

//' Predict with dataset (new API)
//' @param predictor_ptr External pointer to Predictor
//' @param dataset_ptr External pointer to ResolveDataset
//' @param return_latent Whether to return latent embeddings
//' @return List of predictions
//' @export
// [[Rcpp::export]]
List resolve_predict_with_dataset(
    SEXP predictor_ptr,
    SEXP dataset_ptr,
    bool return_latent = false
) {
    XPtr<resolve::Predictor> predictor(predictor_ptr);
    XPtr<resolve::ResolveDataset> dataset(dataset_ptr);

    auto result = predictor->predict(*dataset, return_latent);

    // Convert predictions to R list
    List predictions;
    for (const auto& [name, tensor] : result.predictions) {
        predictions[name] = tensor_to_r_1d(tensor);
    }

    List ret = List::create(
        Named("predictions") = predictions,
        Named("plot_ids") = wrap(result.plot_ids)
    );

    if (return_latent && result.latent.defined()) {
        ret["latent"] = tensor_to_r_2d(result.latent);
    }

    return ret;
}
