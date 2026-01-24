#include <Rcpp.h>
#include <torch/torch.h>
#include "resolve/resolve.hpp"

using namespace Rcpp;

// Helper: Convert R numeric vector to torch tensor
torch::Tensor r_to_tensor(NumericVector x) {
    std::vector<float> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(data.size())},
                           torch::kFloat32).clone();
}

// Helper: Convert R numeric matrix to torch tensor
torch::Tensor r_matrix_to_tensor(NumericMatrix x) {
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

// Helper: Convert R integer matrix to torch tensor
torch::Tensor r_int_matrix_to_tensor(IntegerMatrix x) {
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

// Helper: Convert torch tensor to R numeric vector
NumericVector tensor_to_r(const torch::Tensor& t) {
    auto t_cpu = t.to(torch::kCPU).contiguous();
    auto t_flat = t_cpu.view({-1}).to(torch::kFloat32);
    auto accessor = t_flat.accessor<float, 1>();

    NumericVector result(t_flat.size(0));
    for (int64_t i = 0; i < t_flat.size(0); ++i) {
        result[i] = accessor[i];
    }
    return result;
}

// Helper: Convert torch tensor to R numeric matrix
NumericMatrix tensor_to_r_matrix(const torch::Tensor& t) {
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
// TaxonomyVocab
// ============================================================================

// [[Rcpp::export]]
SEXP cpp_taxonomy_vocab_new() {
    auto* vocab = new resolve::TaxonomyVocab();
    return XPtr<resolve::TaxonomyVocab>(vocab);
}

// [[Rcpp::export]]
void cpp_taxonomy_vocab_fit(SEXP vocab_ptr, CharacterVector genera, CharacterVector families) {
    XPtr<resolve::TaxonomyVocab> vocab(vocab_ptr);

    std::vector<std::string> genera_vec(genera.size());
    std::vector<std::string> families_vec(families.size());

    for (int i = 0; i < genera.size(); ++i) {
        genera_vec[i] = as<std::string>(genera[i]);
    }
    for (int i = 0; i < families.size(); ++i) {
        families_vec[i] = as<std::string>(families[i]);
    }

    vocab->fit(genera_vec, families_vec);
}

// [[Rcpp::export]]
int cpp_taxonomy_vocab_encode_genus(SEXP vocab_ptr, std::string genus) {
    XPtr<resolve::TaxonomyVocab> vocab(vocab_ptr);
    return vocab->encode_genus(genus);
}

// [[Rcpp::export]]
int cpp_taxonomy_vocab_encode_family(SEXP vocab_ptr, std::string family) {
    XPtr<resolve::TaxonomyVocab> vocab(vocab_ptr);
    return vocab->encode_family(family);
}

// [[Rcpp::export]]
int cpp_taxonomy_vocab_n_genera(SEXP vocab_ptr) {
    XPtr<resolve::TaxonomyVocab> vocab(vocab_ptr);
    return vocab->n_genera();
}

// [[Rcpp::export]]
int cpp_taxonomy_vocab_n_families(SEXP vocab_ptr) {
    XPtr<resolve::TaxonomyVocab> vocab(vocab_ptr);
    return vocab->n_families();
}

// ============================================================================
// SpeciesEncoder
// ============================================================================

// [[Rcpp::export]]
SEXP cpp_species_encoder_new(int hash_dim = 32, int top_k = 3) {
    auto* encoder = new resolve::SpeciesEncoder(hash_dim, top_k);
    return XPtr<resolve::SpeciesEncoder>(encoder);
}

// [[Rcpp::export]]
void cpp_species_encoder_fit(SEXP encoder_ptr, List species_data) {
    XPtr<resolve::SpeciesEncoder> encoder(encoder_ptr);

    // Convert R list to C++ format
    // species_data is a list of data frames, each with columns: species, genus, family, abundance
    std::vector<std::vector<std::tuple<std::string, std::string, std::string, float>>> data;

    for (int plot = 0; plot < species_data.size(); ++plot) {
        DataFrame plot_df = as<DataFrame>(species_data[plot]);
        CharacterVector species = plot_df["species"];
        CharacterVector genera = plot_df["genus"];
        CharacterVector families = plot_df["family"];
        NumericVector abundance = plot_df["abundance"];

        std::vector<std::tuple<std::string, std::string, std::string, float>> plot_data;
        for (int i = 0; i < species.size(); ++i) {
            plot_data.emplace_back(
                as<std::string>(species[i]),
                as<std::string>(genera[i]),
                as<std::string>(families[i]),
                abundance[i]
            );
        }
        data.push_back(plot_data);
    }

    encoder->fit(data);
}

// [[Rcpp::export]]
List cpp_species_encoder_transform(SEXP encoder_ptr, List species_data) {
    XPtr<resolve::SpeciesEncoder> encoder(encoder_ptr);

    // Convert R list to C++ format
    std::vector<std::vector<std::tuple<std::string, std::string, std::string, float>>> data;

    for (int plot = 0; plot < species_data.size(); ++plot) {
        DataFrame plot_df = as<DataFrame>(species_data[plot]);
        CharacterVector species = plot_df["species"];
        CharacterVector genera = plot_df["genus"];
        CharacterVector families = plot_df["family"];
        NumericVector abundance = plot_df["abundance"];

        std::vector<std::tuple<std::string, std::string, std::string, float>> plot_data;
        for (int i = 0; i < species.size(); ++i) {
            plot_data.emplace_back(
                as<std::string>(species[i]),
                as<std::string>(genera[i]),
                as<std::string>(families[i]),
                abundance[i]
            );
        }
        data.push_back(plot_data);
    }

    auto result = encoder->transform(data);

    return List::create(
        Named("hash_embedding") = tensor_to_r_matrix(result.hash_embedding),
        Named("genus_ids") = tensor_to_r_matrix(result.genus_ids.to(torch::kFloat32)),
        Named("family_ids") = tensor_to_r_matrix(result.family_ids.to(torch::kFloat32))
    );
}

// ============================================================================
// SpaccModel
// ============================================================================

// [[Rcpp::export]]
SEXP cpp_resolve_model_new(List schema_list, List config_list) {
    // Parse schema
    resolve::SpaccSchema schema;
    schema.n_plots = as<int64_t>(schema_list["n_plots"]);
    schema.n_species = as<int64_t>(schema_list["n_species"]);
    schema.n_continuous = as<int64_t>(schema_list["n_continuous"]);
    schema.has_abundance = as<bool>(schema_list["has_abundance"]);
    schema.has_taxonomy = as<bool>(schema_list["has_taxonomy"]);
    schema.n_genera = as<int64_t>(schema_list["n_genera"]);
    schema.n_families = as<int64_t>(schema_list["n_families"]);

    CharacterVector cov_names = schema_list["covariate_names"];
    for (int i = 0; i < cov_names.size(); ++i) {
        schema.covariate_names.push_back(as<std::string>(cov_names[i]));
    }

    // Parse targets
    List targets_list = schema_list["targets"];
    for (int i = 0; i < targets_list.size(); ++i) {
        List target = targets_list[i];
        resolve::TargetConfig cfg;
        cfg.name = as<std::string>(target["name"]);
        cfg.task = as<std::string>(target["task"]) == "regression"
            ? resolve::TaskType::Regression : resolve::TaskType::Classification;
        cfg.transform = as<std::string>(target["transform"]) == "log1p"
            ? resolve::TransformType::Log1p : resolve::TransformType::None;
        cfg.num_classes = target.containsElementNamed("num_classes")
            ? as<int>(target["num_classes"]) : 0;
        cfg.weight = target.containsElementNamed("weight")
            ? as<float>(target["weight"]) : 1.0f;
        schema.targets.push_back(cfg);
    }

    // Parse config
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

    auto* model = new resolve::SpaccModel(schema, config);
    return XPtr<resolve::SpaccModel>(model);
}

// [[Rcpp::export]]
List cpp_resolve_model_forward(SEXP model_ptr, NumericMatrix continuous,
                             Nullable<IntegerMatrix> genus_ids = R_NilValue,
                             Nullable<IntegerMatrix> family_ids = R_NilValue) {
    XPtr<resolve::SpaccModel> model(model_ptr);

    auto continuous_t = r_matrix_to_tensor(continuous);
    torch::Tensor genus_ids_t, family_ids_t;

    if (genus_ids.isNotNull()) {
        genus_ids_t = r_int_matrix_to_tensor(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_int_matrix_to_tensor(as<IntegerMatrix>(family_ids));
    }

    auto outputs = (*model)->forward(continuous_t, genus_ids_t, family_ids_t);

    List result;
    for (const auto& [name, tensor] : outputs) {
        result[name] = tensor_to_r(tensor);
    }
    return result;
}

// [[Rcpp::export]]
NumericMatrix cpp_resolve_model_get_latent(SEXP model_ptr, NumericMatrix continuous,
                                         Nullable<IntegerMatrix> genus_ids = R_NilValue,
                                         Nullable<IntegerMatrix> family_ids = R_NilValue) {
    XPtr<resolve::SpaccModel> model(model_ptr);

    auto continuous_t = r_matrix_to_tensor(continuous);
    torch::Tensor genus_ids_t, family_ids_t;

    if (genus_ids.isNotNull()) {
        genus_ids_t = r_int_matrix_to_tensor(as<IntegerMatrix>(genus_ids));
    }
    if (family_ids.isNotNull()) {
        family_ids_t = r_int_matrix_to_tensor(as<IntegerMatrix>(family_ids));
    }

    auto latent = (*model)->get_latent(continuous_t, genus_ids_t, family_ids_t);
    return tensor_to_r_matrix(latent);
}

// ============================================================================
// Metrics
// ============================================================================

// [[Rcpp::export]]
double cpp_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    return resolve::Metrics::band_accuracy(r_to_tensor(pred), r_to_tensor(target), threshold);
}

// [[Rcpp::export]]
double cpp_mae(NumericVector pred, NumericVector target) {
    return resolve::Metrics::mae(r_to_tensor(pred), r_to_tensor(target));
}

// [[Rcpp::export]]
double cpp_rmse(NumericVector pred, NumericVector target) {
    return resolve::Metrics::rmse(r_to_tensor(pred), r_to_tensor(target));
}

// [[Rcpp::export]]
double cpp_smape(NumericVector pred, NumericVector target, double eps = 1e-8) {
    return resolve::Metrics::smape(r_to_tensor(pred), r_to_tensor(target), eps);
}
