/**
 * Rcpp bindings for RESOLVE C++ core.
 *
 * This exposes the C++ implementation to R, allowing:
 *   - Full training and inference from R
 *   - Interoperability with R data frames and matrices
 *   - Same API as the Python implementation
 */

#include <Rcpp.h>
#include <RcppTorch.h>

#include <resolve/types.hpp>
#include <resolve/vocab.hpp>
#include <resolve/plot_encoder.hpp>
#include <resolve/model.hpp>
#include <resolve/loss.hpp>
#include <resolve/trainer.hpp>

using namespace Rcpp;

// ============================================================================
// Helper functions for R <-> C++ conversion
// ============================================================================

/**
 * Convert R numeric matrix to torch Tensor
 */
torch::Tensor matrix_to_tensor(NumericMatrix mat) {
    int64_t rows = mat.nrow();
    int64_t cols = mat.ncol();
    auto tensor = torch::zeros({rows, cols}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            accessor[i][j] = static_cast<float>(mat(i, j));
        }
    }
    return tensor;
}

/**
 * Convert torch Tensor to R numeric matrix
 */
NumericMatrix tensor_to_matrix(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.cpu().contiguous();
    int64_t rows = cpu_tensor.size(0);
    int64_t cols = cpu_tensor.ndimension() > 1 ? cpu_tensor.size(1) : 1;

    NumericMatrix mat(rows, cols);
    auto accessor = cpu_tensor.accessor<float, 2>();

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            mat(i, j) = accessor[i][j];
        }
    }
    return mat;
}

// ============================================================================
// PlotEncoder functions (generalized encoder)
// ============================================================================

/**
 * Convert R data frame to PlotRecord vector
 */
std::vector<resolve::PlotRecord> dataframe_to_plot_records(
    DataFrame df,
    CharacterVector categorical_cols,
    CharacterVector numeric_cols,
    const std::string& plot_id_col
) {
    int n = df.nrows();
    std::vector<resolve::PlotRecord> records;
    records.reserve(n);

    CharacterVector plot_ids = df[plot_id_col];

    for (int i = 0; i < n; ++i) {
        resolve::PlotRecord record;
        record.plot_id = as<std::string>(plot_ids[i]);

        // Extract categorical columns
        for (int j = 0; j < categorical_cols.size(); ++j) {
            std::string col_name = as<std::string>(categorical_cols[j]);
            if (df.containsElementNamed(col_name.c_str())) {
                CharacterVector col = df[col_name];
                record.categorical[col_name] = as<std::string>(col[i]);
            }
        }

        // Extract numeric columns
        for (int j = 0; j < numeric_cols.size(); ++j) {
            std::string col_name = as<std::string>(numeric_cols[j]);
            if (df.containsElementNamed(col_name.c_str())) {
                NumericVector col = df[col_name];
                record.numeric[col_name] = static_cast<float>(col[i]);
            }
        }

        records.push_back(record);
    }

    return records;
}

/**
 * Convert R data frame to ObservationRecord vector
 */
std::vector<resolve::ObservationRecord> dataframe_to_obs_records(
    DataFrame df,
    CharacterVector categorical_cols,
    CharacterVector numeric_cols,
    const std::string& plot_id_col
) {
    int n = df.nrows();
    std::vector<resolve::ObservationRecord> records;
    records.reserve(n);

    CharacterVector plot_ids = df[plot_id_col];

    for (int i = 0; i < n; ++i) {
        resolve::ObservationRecord record;
        record.plot_id = as<std::string>(plot_ids[i]);

        // Extract categorical columns
        for (int j = 0; j < categorical_cols.size(); ++j) {
            std::string col_name = as<std::string>(categorical_cols[j]);
            if (df.containsElementNamed(col_name.c_str())) {
                CharacterVector col = df[col_name];
                record.categorical[col_name] = as<std::string>(col[i]);
            }
        }

        // Extract numeric columns
        for (int j = 0; j < numeric_cols.size(); ++j) {
            std::string col_name = as<std::string>(numeric_cols[j]);
            if (df.containsElementNamed(col_name.c_str())) {
                NumericVector col = df[col_name];
                record.numeric[col_name] = static_cast<float>(col[i]);
            }
        }

        records.push_back(record);
    }

    return records;
}

/**
 * Create a new PlotEncoder
 */
// [[Rcpp::export]]
SEXP plot_encoder_new() {
    auto* encoder = new resolve::PlotEncoder();
    XPtr<resolve::PlotEncoder> ptr(encoder, true);
    return ptr;
}

/**
 * Add numeric encoding to PlotEncoder
 */
// [[Rcpp::export]]
void plot_encoder_add_numeric(
    SEXP encoder_ptr,
    std::string name,
    CharacterVector columns,
    std::string source = "plot"
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    std::vector<std::string> cols = as<std::vector<std::string>>(columns);
    resolve::DataSource src = (source == "observation")
        ? resolve::DataSource::Observation
        : resolve::DataSource::Plot;
    encoder->add_numeric(name, cols, src);
}

/**
 * Add raw encoding (no scaling) to PlotEncoder
 */
// [[Rcpp::export]]
void plot_encoder_add_raw(
    SEXP encoder_ptr,
    std::string name,
    CharacterVector columns,
    std::string source = "plot"
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    std::vector<std::string> cols = as<std::vector<std::string>>(columns);
    resolve::DataSource src = (source == "observation")
        ? resolve::DataSource::Observation
        : resolve::DataSource::Plot;
    encoder->add_raw(name, cols, src);
}

/**
 * Add hash encoding to PlotEncoder
 */
// [[Rcpp::export]]
void plot_encoder_add_hash(
    SEXP encoder_ptr,
    std::string name,
    CharacterVector columns,
    int dim = 32,
    int top_k = 0,
    int bottom_k = 0,
    std::string rank_by = "",
    std::string source = "observation"
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    std::vector<std::string> cols = as<std::vector<std::string>>(columns);
    resolve::DataSource src = (source == "observation")
        ? resolve::DataSource::Observation
        : resolve::DataSource::Plot;
    encoder->add_hash(name, cols, dim, top_k, bottom_k, rank_by, src);
}

/**
 * Add embed encoding to PlotEncoder
 */
// [[Rcpp::export]]
void plot_encoder_add_embed(
    SEXP encoder_ptr,
    std::string name,
    CharacterVector columns,
    int dim = 16,
    int top_k = 0,
    int bottom_k = 0,
    std::string rank_by = "",
    std::string source = "observation"
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    std::vector<std::string> cols = as<std::vector<std::string>>(columns);
    resolve::DataSource src = (source == "observation")
        ? resolve::DataSource::Observation
        : resolve::DataSource::Plot;
    encoder->add_embed(name, cols, dim, top_k, bottom_k, rank_by, src);
}

/**
 * Add onehot encoding to PlotEncoder
 */
// [[Rcpp::export]]
void plot_encoder_add_onehot(
    SEXP encoder_ptr,
    std::string name,
    CharacterVector columns,
    std::string source = "plot"
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    std::vector<std::string> cols = as<std::vector<std::string>>(columns);
    resolve::DataSource src = (source == "observation")
        ? resolve::DataSource::Observation
        : resolve::DataSource::Plot;
    encoder->add_onehot(name, cols, src);
}

/**
 * Fit PlotEncoder on data
 */
// [[Rcpp::export]]
void plot_encoder_fit(
    SEXP encoder_ptr,
    DataFrame plot_df,
    std::string plot_id_col,
    CharacterVector plot_categorical_cols,
    CharacterVector plot_numeric_cols,
    Nullable<DataFrame> obs_df = R_NilValue,
    std::string obs_plot_id_col = "",
    Nullable<CharacterVector> obs_categorical_cols = R_NilValue,
    Nullable<CharacterVector> obs_numeric_cols = R_NilValue
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);

    auto plot_records = dataframe_to_plot_records(
        plot_df, plot_categorical_cols, plot_numeric_cols, plot_id_col
    );

    std::vector<resolve::ObservationRecord> obs_records;
    if (obs_df.isNotNull()) {
        obs_records = dataframe_to_obs_records(
            as<DataFrame>(obs_df),
            obs_categorical_cols.isNotNull() ? as<CharacterVector>(obs_categorical_cols) : CharacterVector(),
            obs_numeric_cols.isNotNull() ? as<CharacterVector>(obs_numeric_cols) : CharacterVector(),
            obs_plot_id_col
        );
    }

    encoder->fit(plot_records, obs_records);
}

/**
 * Transform data using fitted PlotEncoder
 */
// [[Rcpp::export]]
List plot_encoder_transform(
    SEXP encoder_ptr,
    DataFrame plot_df,
    std::string plot_id_col,
    CharacterVector plot_ids,
    CharacterVector plot_categorical_cols,
    CharacterVector plot_numeric_cols,
    Nullable<DataFrame> obs_df = R_NilValue,
    std::string obs_plot_id_col = "",
    Nullable<CharacterVector> obs_categorical_cols = R_NilValue,
    Nullable<CharacterVector> obs_numeric_cols = R_NilValue
) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);

    auto plot_records = dataframe_to_plot_records(
        plot_df, plot_categorical_cols, plot_numeric_cols, plot_id_col
    );

    std::vector<resolve::ObservationRecord> obs_records;
    if (obs_df.isNotNull()) {
        obs_records = dataframe_to_obs_records(
            as<DataFrame>(obs_df),
            obs_categorical_cols.isNotNull() ? as<CharacterVector>(obs_categorical_cols) : CharacterVector(),
            obs_numeric_cols.isNotNull() ? as<CharacterVector>(obs_numeric_cols) : CharacterVector(),
            obs_plot_id_col
        );
    }

    std::vector<std::string> plot_id_vec = as<std::vector<std::string>>(plot_ids);
    auto encoded = encoder->transform(plot_records, obs_records, plot_id_vec);

    List result;

    // Continuous features
    if (encoded.continuous_features().numel() > 0) {
        result["continuous"] = tensor_to_matrix(encoded.continuous_features());
    }

    // Unknown fraction
    if (encoded.unknown_fraction.defined()) {
        result["unknown_fraction"] = tensor_to_matrix(encoded.unknown_fraction.unsqueeze(1));
    }

    // Embedding IDs (as named list)
    List embed_ids;
    for (const auto& col : encoded.columns) {
        if (col.is_embedding_ids) {
            embed_ids[col.name] = tensor_to_matrix(col.values.to(torch::kFloat32));
        }
    }
    if (embed_ids.size() > 0) {
        result["embedding_ids"] = embed_ids;
    }

    result["plot_ids"] = wrap(encoded.plot_ids);

    return result;
}

/**
 * Get PlotEncoder properties
 */
// [[Rcpp::export]]
List plot_encoder_info(SEXP encoder_ptr) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);

    // Get embedding configs
    auto embed_configs = encoder->embedding_configs();
    List embeddings;
    for (const auto& [name, vocab_size, dim, n_slots] : embed_configs) {
        embeddings.push_back(List::create(
            Named("name") = name,
            Named("vocab_size") = vocab_size,
            Named("dim") = dim,
            Named("n_slots") = n_slots
        ));
    }

    // Get specs summary
    List specs;
    for (const auto& spec : encoder->specs()) {
        std::string type_str;
        switch (spec.type) {
            case resolve::EncodingType::Numeric: type_str = "numeric"; break;
            case resolve::EncodingType::Raw: type_str = "raw"; break;
            case resolve::EncodingType::Hash: type_str = "hash"; break;
            case resolve::EncodingType::Embed: type_str = "embed"; break;
            case resolve::EncodingType::OneHot: type_str = "onehot"; break;
        }
        specs.push_back(List::create(
            Named("name") = spec.name,
            Named("type") = type_str,
            Named("columns") = wrap(spec.columns),
            Named("dim") = spec.dim,
            Named("top_k") = spec.top_k,
            Named("bottom_k") = spec.bottom_k
        ));
    }

    return List::create(
        Named("is_fitted") = encoder->is_fitted(),
        Named("continuous_dim") = encoder->continuous_dim(),
        Named("n_specs") = encoder->specs().size(),
        Named("specs") = specs,
        Named("embeddings") = embeddings
    );
}

/**
 * Save PlotEncoder to file
 */
// [[Rcpp::export]]
void plot_encoder_save(SEXP encoder_ptr, std::string path) {
    XPtr<resolve::PlotEncoder> encoder(encoder_ptr);
    encoder->save(path);
}

/**
 * Load PlotEncoder from file
 */
// [[Rcpp::export]]
SEXP plot_encoder_load(std::string path) {
    auto encoder = new resolve::PlotEncoder(resolve::PlotEncoder::load(path));
    XPtr<resolve::PlotEncoder> ptr(encoder, true);
    return ptr;
}

// ============================================================================
// Model functions
// ============================================================================

/**
 * Convert R list to TargetConfig
 */
resolve::TargetConfig list_to_target_config(List cfg) {
    resolve::TargetConfig target;
    target.name = as<std::string>(cfg["name"]);

    std::string task_str = as<std::string>(cfg["task"]);
    target.task = (task_str == "classification")
        ? resolve::TaskType::Classification
        : resolve::TaskType::Regression;

    target.n_classes = cfg.containsElementNamed("n_classes") ? as<int>(cfg["n_classes"]) : 1;

    if (cfg.containsElementNamed("transform")) {
        std::string transform_str = as<std::string>(cfg["transform"]);
        if (transform_str == "log1p") target.transform = resolve::TransformType::Log1p;
        else if (transform_str == "sqrt") target.transform = resolve::TransformType::Sqrt;
        else target.transform = resolve::TransformType::None;
    }

    return target;
}

/**
 * Convert R list to ModelConfig
 */
resolve::ModelConfig list_to_model_config(List cfg) {
    resolve::ModelConfig config;
    config.encoder_dim = cfg.containsElementNamed("encoder_dim") ? as<int>(cfg["encoder_dim"]) : 256;
    config.hidden_dim = cfg.containsElementNamed("hidden_dim") ? as<int>(cfg["hidden_dim"]) : 512;
    config.n_encoder_layers = cfg.containsElementNamed("n_encoder_layers") ? as<int>(cfg["n_encoder_layers"]) : 3;
    config.dropout = cfg.containsElementNamed("dropout") ? as<float>(cfg["dropout"]) : 0.1f;
    config.hash_dim = cfg.containsElementNamed("hash_dim") ? as<int>(cfg["hash_dim"]) : 32;
    config.genus_vocab_size = cfg.containsElementNamed("genus_vocab_size") ? as<int>(cfg["genus_vocab_size"]) : 1000;
    config.family_vocab_size = cfg.containsElementNamed("family_vocab_size") ? as<int>(cfg["family_vocab_size"]) : 200;
    config.species_vocab_size = cfg.containsElementNamed("species_vocab_size") ? as<int>(cfg["species_vocab_size"]) : 0;
    config.n_species_vector = cfg.containsElementNamed("n_species_vector") ? as<int>(cfg["n_species_vector"]) : 0;
    config.n_continuous = cfg.containsElementNamed("n_continuous") ? as<int>(cfg["n_continuous"]) : 0;
    config.top_k = cfg.containsElementNamed("top_k") ? as<int>(cfg["top_k"]) : 3;

    if (cfg.containsElementNamed("mode")) {
        std::string mode_str = as<std::string>(cfg["mode"]);
        if (mode_str == "embed") config.mode = resolve::SpeciesEncodingMode::Embed;
        else if (mode_str == "sparse") config.mode = resolve::SpeciesEncodingMode::Sparse;
        else config.mode = resolve::SpeciesEncodingMode::Hash;
    }

    return config;
}

/**
 * Create a new ResolveModel
 */
// [[Rcpp::export]]
SEXP resolve_model_new(List target_configs_list, List model_config_list) {
    // Convert target configs
    std::vector<resolve::TargetConfig> target_configs;
    for (int i = 0; i < target_configs_list.size(); ++i) {
        target_configs.push_back(list_to_target_config(target_configs_list[i]));
    }

    // Convert model config
    resolve::ModelConfig model_config = list_to_model_config(model_config_list);

    // Create model
    auto* model = new resolve::ResolveModel(model_config, target_configs);

    XPtr<resolve::ResolveModel> ptr(model, true);
    return ptr;
}

/**
 * Forward pass through model
 */
// [[Rcpp::export]]
List resolve_model_forward(
    SEXP model_ptr,
    NumericMatrix continuous,
    Nullable<NumericMatrix> genus_ids = R_NilValue,
    Nullable<NumericMatrix> family_ids = R_NilValue,
    Nullable<NumericMatrix> species_ids = R_NilValue,
    Nullable<NumericMatrix> species_vector = R_NilValue
) {
    XPtr<resolve::ResolveModel> model(model_ptr);

    torch::Tensor cont_tensor = matrix_to_tensor(continuous);

    torch::Tensor genus_tensor, family_tensor, species_id_tensor, species_vec_tensor;

    if (genus_ids.isNotNull()) {
        genus_tensor = matrix_to_tensor(as<NumericMatrix>(genus_ids)).to(torch::kInt64);
    }
    if (family_ids.isNotNull()) {
        family_tensor = matrix_to_tensor(as<NumericMatrix>(family_ids)).to(torch::kInt64);
    }
    if (species_ids.isNotNull()) {
        species_id_tensor = matrix_to_tensor(as<NumericMatrix>(species_ids)).to(torch::kInt64);
    }
    if (species_vector.isNotNull()) {
        species_vec_tensor = matrix_to_tensor(as<NumericMatrix>(species_vector));
    }

    auto outputs = (*model)->forward(
        cont_tensor, genus_tensor, family_tensor, species_id_tensor, species_vec_tensor
    );

    List result;
    for (const auto& [name, tensor] : outputs) {
        result[name] = tensor_to_matrix(tensor);
    }

    return result;
}

/**
 * Get latent representation from model
 */
// [[Rcpp::export]]
NumericMatrix resolve_model_get_latent(
    SEXP model_ptr,
    NumericMatrix continuous,
    Nullable<NumericMatrix> genus_ids = R_NilValue,
    Nullable<NumericMatrix> family_ids = R_NilValue,
    Nullable<NumericMatrix> species_ids = R_NilValue,
    Nullable<NumericMatrix> species_vector = R_NilValue
) {
    XPtr<resolve::ResolveModel> model(model_ptr);

    torch::Tensor cont_tensor = matrix_to_tensor(continuous);

    torch::Tensor genus_tensor, family_tensor, species_id_tensor, species_vec_tensor;

    if (genus_ids.isNotNull()) {
        genus_tensor = matrix_to_tensor(as<NumericMatrix>(genus_ids)).to(torch::kInt64);
    }
    if (family_ids.isNotNull()) {
        family_tensor = matrix_to_tensor(as<NumericMatrix>(family_ids)).to(torch::kInt64);
    }
    if (species_ids.isNotNull()) {
        species_id_tensor = matrix_to_tensor(as<NumericMatrix>(species_ids)).to(torch::kInt64);
    }
    if (species_vector.isNotNull()) {
        species_vec_tensor = matrix_to_tensor(as<NumericMatrix>(species_vector));
    }

    auto latent = (*model)->get_latent(
        cont_tensor, genus_tensor, family_tensor, species_id_tensor, species_vec_tensor
    );

    return tensor_to_matrix(latent);
}

/**
 * Save model to file
 */
// [[Rcpp::export]]
void resolve_model_save(SEXP model_ptr, std::string path) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    torch::serialize::OutputArchive archive;
    (*model)->save(archive);
    archive.save_to(path);
}

/**
 * Load model state from file
 */
// [[Rcpp::export]]
void resolve_model_load(SEXP model_ptr, std::string path) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    (*model)->load(archive);
}

/**
 * Set model to training mode
 */
// [[Rcpp::export]]
void resolve_model_train(SEXP model_ptr) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    (*model)->train();
}

/**
 * Set model to evaluation mode
 */
// [[Rcpp::export]]
void resolve_model_eval(SEXP model_ptr) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    (*model)->eval();
}

/**
 * Move model to device
 */
// [[Rcpp::export]]
void resolve_model_to(SEXP model_ptr, std::string device) {
    XPtr<resolve::ResolveModel> model(model_ptr);
    (*model)->to(torch::Device(device));
}

// ============================================================================
// Predictor functions
// ============================================================================

// TODO: Reimplement predictor functions using PlotEncoder

// ============================================================================
// Metrics functions
// ============================================================================

/**
 * Compute band accuracy
 */
// [[Rcpp::export]]
double metrics_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    int n = pred.size();
    if (n != target.size()) {
        Rcpp::stop("pred and target must have the same length");
    }

    int count = 0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (!NumericVector::is_na(pred[i]) && !NumericVector::is_na(target[i])) {
            valid++;
            double rel_error = std::abs(pred[i] - target[i]) / (std::abs(target[i]) + 1e-8);
            if (rel_error <= threshold) {
                count++;
            }
        }
    }

    return valid > 0 ? static_cast<double>(count) / valid : 0.0;
}

/**
 * Compute MAE
 */
// [[Rcpp::export]]
double metrics_mae(NumericVector pred, NumericVector target) {
    int n = pred.size();
    if (n != target.size()) {
        Rcpp::stop("pred and target must have the same length");
    }

    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (!NumericVector::is_na(pred[i]) && !NumericVector::is_na(target[i])) {
            sum += std::abs(pred[i] - target[i]);
            valid++;
        }
    }

    return valid > 0 ? sum / valid : 0.0;
}

/**
 * Compute RMSE
 */
// [[Rcpp::export]]
double metrics_rmse(NumericVector pred, NumericVector target) {
    int n = pred.size();
    if (n != target.size()) {
        Rcpp::stop("pred and target must have the same length");
    }

    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (!NumericVector::is_na(pred[i]) && !NumericVector::is_na(target[i])) {
            double diff = pred[i] - target[i];
            sum += diff * diff;
            valid++;
        }
    }

    return valid > 0 ? std::sqrt(sum / valid) : 0.0;
}

/**
 * Compute SMAPE
 */
// [[Rcpp::export]]
double metrics_smape(NumericVector pred, NumericVector target) {
    int n = pred.size();
    if (n != target.size()) {
        Rcpp::stop("pred and target must have the same length");
    }

    double sum = 0.0;
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        if (!NumericVector::is_na(pred[i]) && !NumericVector::is_na(target[i])) {
            double denom = (std::abs(pred[i]) + std::abs(target[i])) / 2 + 1e-8;
            sum += std::abs(pred[i] - target[i]) / denom;
            valid++;
        }
    }

    return valid > 0 ? sum / valid : 0.0;
}
