// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
#include <torch/torch.h>
#include "resolve/resolve.hpp"

using namespace Rcpp;

// =============================================================================
// Helper functions for type conversion
// =============================================================================

// Convert R numeric vector to torch tensor
torch::Tensor r_vec_to_tensor(NumericVector x) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor t = torch::from_blob(
        x.begin(),
        {static_cast<int64_t>(x.size())},
        options
    ).clone();
    return t;
}

// Convert R numeric matrix to torch tensor
torch::Tensor r_mat_to_tensor(NumericMatrix x) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    // R uses column-major, torch uses row-major, so we transpose
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<float> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<float>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
}

// Convert R integer vector to torch tensor (int64)
torch::Tensor r_int_vec_to_tensor(IntegerVector x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(x.size())}, options).clone();
}

// Convert R integer matrix to torch tensor (int64)
torch::Tensor r_int_mat_to_tensor(IntegerMatrix x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<int64_t> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<int64_t>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
}

// Convert torch tensor to R numeric vector
NumericVector tensor_to_r_vec(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    float* data = cpu.data_ptr<float>();
    return NumericVector(data, data + cpu.numel());
}

// Convert torch tensor to R numeric matrix
NumericMatrix tensor_to_r_mat(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    int nrow = cpu.size(0);
    int ncol = cpu.size(1);
    NumericMatrix out(nrow, ncol);
    float* data = cpu.data_ptr<float>();
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            out(i, j) = data[i * ncol + j];
        }
    }
    return out;
}

// =============================================================================
// Enum conversions
// =============================================================================

resolve::SelectionMode parse_selection_mode(const std::string& s) {
    if (s == "top") return resolve::SelectionMode::Top;
    if (s == "bottom") return resolve::SelectionMode::Bottom;
    if (s == "top_bottom") return resolve::SelectionMode::TopBottom;
    if (s == "all") return resolve::SelectionMode::All;
    stop("Invalid selection mode: " + s);
}

resolve::RepresentationMode parse_representation_mode(const std::string& s) {
    if (s == "abundance") return resolve::RepresentationMode::Abundance;
    if (s == "presence_absence") return resolve::RepresentationMode::PresenceAbsence;
    stop("Invalid representation mode: " + s);
}

resolve::NormalizationMode parse_normalization_mode(const std::string& s) {
    if (s == "raw") return resolve::NormalizationMode::Raw;
    if (s == "norm") return resolve::NormalizationMode::Norm;
    if (s == "log1p") return resolve::NormalizationMode::Log1p;
    stop("Invalid normalization mode: " + s);
}

resolve::AggregationMode parse_aggregation_mode(const std::string& s) {
    if (s == "abundance") return resolve::AggregationMode::Abundance;
    if (s == "count") return resolve::AggregationMode::Count;
    stop("Invalid aggregation mode: " + s);
}

resolve::TaskType parse_task_type(const std::string& s) {
    if (s == "regression") return resolve::TaskType::Regression;
    if (s == "classification") return resolve::TaskType::Classification;
    stop("Invalid task type: " + s);
}

resolve::TransformType parse_transform_type(const std::string& s) {
    if (s == "none") return resolve::TransformType::None;
    if (s == "log1p") return resolve::TransformType::Log1p;
    stop("Invalid transform type: " + s);
}

resolve::SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    if (s == "hash") return resolve::SpeciesEncodingMode::Hash;
    if (s == "embed") return resolve::SpeciesEncodingMode::Embed;
    if (s == "sparse") return resolve::SpeciesEncodingMode::Sparse;
    stop("Invalid species encoding mode: " + s);
}

resolve::LossConfigMode parse_loss_config_mode(const std::string& s) {
    if (s == "mae") return resolve::LossConfigMode::MAE;
    if (s == "smape") return resolve::LossConfigMode::SMAPE;
    if (s == "combined") return resolve::LossConfigMode::Combined;
    stop("Invalid loss config mode: " + s);
}

resolve::LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    if (s == "none") return resolve::LRSchedulerType::None;
    if (s == "step") return resolve::LRSchedulerType::StepLR;
    if (s == "cosine") return resolve::LRSchedulerType::CosineAnnealing;
    stop("Invalid LR scheduler type: " + s);
}

// =============================================================================
// SpeciesEncoder class wrapper
// =============================================================================

class RSpeciesEncoder {
public:
    RSpeciesEncoder(
        int hash_dim = 32,
        int top_k = 3,
        std::string aggregation = "abundance",
        std::string normalization = "norm",
        bool track_unknown_count = false,
        std::string selection = "top",
        std::string representation = "abundance",
        int min_species_frequency = 1
    ) : encoder_(
            hash_dim,
            top_k,
            parse_aggregation_mode(aggregation),
            parse_normalization_mode(normalization),
            track_unknown_count,
            parse_selection_mode(selection),
            parse_representation_mode(representation),
            min_species_frequency
        ) {}

    // Fit from R data frame
    void fit(DataFrame species_df) {
        CharacterVector species_id = species_df["species_id"];
        CharacterVector genus = species_df["genus"];
        CharacterVector family = species_df["family"];
        NumericVector abundance = species_df["abundance"];
        CharacterVector plot_id = species_df["plot_id"];

        std::vector<resolve::SpeciesRecord> records;
        records.reserve(species_id.size());
        for (int i = 0; i < species_id.size(); ++i) {
            resolve::SpeciesRecord rec;
            rec.species_id = as<std::string>(species_id[i]);
            rec.genus = as<std::string>(genus[i]);
            rec.family = as<std::string>(family[i]);
            rec.abundance = abundance[i];
            rec.plot_id = as<std::string>(plot_id[i]);
            records.push_back(rec);
        }

        encoder_.fit(records);
    }

    // Transform species data
    List transform(DataFrame species_df, CharacterVector plot_ids) {
        CharacterVector species_id = species_df["species_id"];
        CharacterVector genus = species_df["genus"];
        CharacterVector family = species_df["family"];
        NumericVector abundance = species_df["abundance"];
        CharacterVector df_plot_id = species_df["plot_id"];

        std::vector<resolve::SpeciesRecord> records;
        records.reserve(species_id.size());
        for (int i = 0; i < species_id.size(); ++i) {
            resolve::SpeciesRecord rec;
            rec.species_id = as<std::string>(species_id[i]);
            rec.genus = as<std::string>(genus[i]);
            rec.family = as<std::string>(family[i]);
            rec.abundance = abundance[i];
            rec.plot_id = as<std::string>(df_plot_id[i]);
            records.push_back(rec);
        }

        std::vector<std::string> pids = as<std::vector<std::string>>(plot_ids);
        resolve::EncodedSpecies encoded = encoder_.transform(records, pids);

        List result;
        if (encoded.hash_embedding.defined()) {
            result["hash_embedding"] = tensor_to_r_mat(encoded.hash_embedding);
        }
        if (encoded.genus_ids.defined()) {
            result["genus_ids"] = tensor_to_r_mat(encoded.genus_ids.to(torch::kFloat32));
        }
        if (encoded.family_ids.defined()) {
            result["family_ids"] = tensor_to_r_mat(encoded.family_ids.to(torch::kFloat32));
        }
        if (encoded.unknown_fraction.defined()) {
            result["unknown_fraction"] = tensor_to_r_vec(encoded.unknown_fraction);
        }
        if (encoded.unknown_count.defined()) {
            result["unknown_count"] = tensor_to_r_vec(encoded.unknown_count);
        }
        if (encoded.species_vector.defined()) {
            result["species_vector"] = tensor_to_r_mat(encoded.species_vector);
        }
        result["plot_ids"] = wrap(encoded.plot_ids);

        return result;
    }

    bool is_fitted() const { return encoder_.is_fitted(); }
    int hash_dim() const { return encoder_.hash_dim(); }
    int top_k() const { return encoder_.top_k(); }
    int n_genera() const { return encoder_.n_genera(); }
    int n_families() const { return encoder_.n_families(); }
    int n_taxonomy_slots() const { return encoder_.n_taxonomy_slots(); }
    bool uses_explicit_vector() const { return encoder_.uses_explicit_vector(); }
    int n_species_vector() const { return encoder_.n_species_vector(); }
    int n_known_species() const { return encoder_.n_known_species(); }

    void save(std::string path) const { encoder_.save(path); }

    static RSpeciesEncoder load(std::string path) {
        RSpeciesEncoder wrapper;
        wrapper.encoder_ = resolve::SpeciesEncoder::load(path);
        return wrapper;
    }

    resolve::SpeciesEncoder& encoder() { return encoder_; }
    const resolve::SpeciesEncoder& encoder() const { return encoder_; }

private:
    RSpeciesEncoder() : encoder_() {}  // For load()
    resolve::SpeciesEncoder encoder_;
};

// =============================================================================
// ResolveModel class wrapper
// =============================================================================

class RResolveModel {
public:
    RResolveModel(List schema_list, List config_list) {
        resolve::ResolveSchema schema;
        schema.n_plots = schema_list["n_plots"];
        schema.n_species = schema_list["n_species"];
        if (schema_list.containsElementNamed("n_species_vocab")) {
            schema.n_species_vocab = schema_list["n_species_vocab"];
        }
        schema.has_coordinates = schema_list["has_coordinates"];
        schema.has_abundance = schema_list["has_abundance"];
        schema.has_taxonomy = schema_list["has_taxonomy"];
        schema.n_genera = schema_list["n_genera"];
        schema.n_families = schema_list["n_families"];
        if (schema_list.containsElementNamed("covariate_names")) {
            schema.covariate_names = as<std::vector<std::string>>(schema_list["covariate_names"]);
        }
        schema.track_unknown_fraction = schema_list["track_unknown_fraction"];
        schema.track_unknown_count = schema_list["track_unknown_count"];

        // Parse targets
        if (schema_list.containsElementNamed("targets")) {
            List targets = schema_list["targets"];
            CharacterVector target_names = targets.names();
            for (int i = 0; i < targets.size(); ++i) {
                List target_cfg = targets[i];
                resolve::TargetConfig tc;
                tc.name = as<std::string>(target_names[i]);
                tc.task = parse_task_type(as<std::string>(target_cfg["task"]));
                if (target_cfg.containsElementNamed("transform")) {
                    tc.transform = parse_transform_type(as<std::string>(target_cfg["transform"]));
                }
                if (target_cfg.containsElementNamed("num_classes")) {
                    tc.num_classes = target_cfg["num_classes"];
                }
                if (target_cfg.containsElementNamed("weight")) {
                    tc.weight = target_cfg["weight"];
                }
                schema.targets.push_back(tc);
            }
        }

        resolve::ModelConfig config;
        if (config_list.containsElementNamed("species_encoding")) {
            config.species_encoding = parse_species_encoding_mode(
                as<std::string>(config_list["species_encoding"]));
        }
        if (config_list.containsElementNamed("hash_dim")) {
            config.hash_dim = config_list["hash_dim"];
        }
        if (config_list.containsElementNamed("species_embed_dim")) {
            config.species_embed_dim = config_list["species_embed_dim"];
        }
        if (config_list.containsElementNamed("genus_emb_dim")) {
            config.genus_emb_dim = config_list["genus_emb_dim"];
        }
        if (config_list.containsElementNamed("family_emb_dim")) {
            config.family_emb_dim = config_list["family_emb_dim"];
        }
        if (config_list.containsElementNamed("top_k")) {
            config.top_k = config_list["top_k"];
        }
        if (config_list.containsElementNamed("top_k_species")) {
            config.top_k_species = config_list["top_k_species"];
        }
        if (config_list.containsElementNamed("n_taxonomy_slots")) {
            config.n_taxonomy_slots = config_list["n_taxonomy_slots"];
        }
        if (config_list.containsElementNamed("hidden_dims")) {
            config.hidden_dims = as<std::vector<int64_t>>(config_list["hidden_dims"]);
        }
        if (config_list.containsElementNamed("dropout")) {
            config.dropout = config_list["dropout"];
        }

        model_ = std::make_shared<resolve::ResolveModel>(schema, config);
    }

    List forward(
        NumericMatrix continuous,
        Nullable<NumericMatrix> genus_ids = R_NilValue,
        Nullable<NumericMatrix> family_ids = R_NilValue,
        Nullable<NumericMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }
        if (species_ids.isNotNull()) {
            species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        }
        if (species_vector.isNotNull()) {
            species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        }

        auto outputs = (*model_)->forward(cont_t, genus_t, family_t, species_id_t, species_vec_t);

        List result;
        for (const auto& [name, tensor] : outputs) {
            result[name] = tensor_to_r_vec(tensor);
        }
        return result;
    }

    NumericVector get_latent(
        NumericMatrix continuous,
        Nullable<NumericMatrix> genus_ids = R_NilValue,
        Nullable<NumericMatrix> family_ids = R_NilValue,
        Nullable<NumericMatrix> species_ids = R_NilValue,
        Nullable<NumericMatrix> species_vector = R_NilValue
    ) {
        torch::Tensor cont_t = r_mat_to_tensor(continuous);
        torch::Tensor genus_t, family_t, species_id_t, species_vec_t;

        if (genus_ids.isNotNull()) {
            genus_t = r_int_mat_to_tensor(as<IntegerMatrix>(genus_ids));
        }
        if (family_ids.isNotNull()) {
            family_t = r_int_mat_to_tensor(as<IntegerMatrix>(family_ids));
        }
        if (species_ids.isNotNull()) {
            species_id_t = r_int_mat_to_tensor(as<IntegerMatrix>(species_ids));
        }
        if (species_vector.isNotNull()) {
            species_vec_t = r_mat_to_tensor(as<NumericMatrix>(species_vector));
        }

        torch::Tensor latent = (*model_)->get_latent(cont_t, genus_t, family_t, species_id_t, species_vec_t);
        return tensor_to_r_vec(latent);
    }

    void train(bool mode = true) { (*model_)->train(mode); }
    void eval() { (*model_)->eval(); }

    void to_device(std::string device) {
        if (device == "cuda") {
            (*model_)->to(torch::kCUDA);
        } else {
            (*model_)->to(torch::kCPU);
        }
    }

    int latent_dim() const { return (*model_)->latent_dim(); }

    std::shared_ptr<resolve::ResolveModel>& model() { return model_; }

private:
    std::shared_ptr<resolve::ResolveModel> model_;
};

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
            target_map, static_cast<float>(test_size), seed
        );
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

        resolve::SpaccPredictions preds = predictor_.predict(
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

private:
    RPredictor(resolve::Predictor pred) : predictor_(std::move(pred)) {}
    resolve::Predictor predictor_;
};

// =============================================================================
// Metrics (static functions)
// =============================================================================

// [[Rcpp::export]]
double resolve_band_accuracy(NumericVector pred, NumericVector target, double threshold = 0.25) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::band_accuracy(pred_t, target_t, static_cast<float>(threshold));
}

// [[Rcpp::export]]
double resolve_mae(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::mae(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_rmse(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::rmse(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_smape(NumericVector pred, NumericVector target, double eps = 1e-8) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::smape(pred_t, target_t, static_cast<float>(eps));
}

// [[Rcpp::export]]
double resolve_accuracy(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::accuracy(pred_t, target_t);
}

// [[Rcpp::export]]
double resolve_r_squared(NumericVector pred, NumericVector target) {
    torch::Tensor pred_t = r_vec_to_tensor(pred);
    torch::Tensor target_t = r_vec_to_tensor(target);
    return resolve::Metrics::r_squared(pred_t, target_t);
}

// =============================================================================
// Module exports via Rcpp modules
// =============================================================================

RCPP_MODULE(resolve_module) {
    class_<RSpeciesEncoder>("SpeciesEncoder")
        .constructor<int, int, std::string, std::string, bool, std::string, std::string, int>(
            "Create a SpeciesEncoder")
        .method("fit", &RSpeciesEncoder::fit, "Fit encoder on species data")
        .method("transform", &RSpeciesEncoder::transform, "Transform species data")
        .method("is_fitted", &RSpeciesEncoder::is_fitted, "Check if encoder is fitted")
        .method("hash_dim", &RSpeciesEncoder::hash_dim, "Get hash dimension")
        .method("top_k", &RSpeciesEncoder::top_k, "Get top-k value")
        .method("n_genera", &RSpeciesEncoder::n_genera, "Get number of genera")
        .method("n_families", &RSpeciesEncoder::n_families, "Get number of families")
        .method("n_taxonomy_slots", &RSpeciesEncoder::n_taxonomy_slots, "Get taxonomy slot count")
        .method("uses_explicit_vector", &RSpeciesEncoder::uses_explicit_vector, "Check if using explicit vector")
        .method("n_species_vector", &RSpeciesEncoder::n_species_vector, "Get species vector size")
        .method("n_known_species", &RSpeciesEncoder::n_known_species, "Get known species count")
        .method("save", &RSpeciesEncoder::save, "Save encoder to file")
        ;

    function("SpeciesEncoder_load", &RSpeciesEncoder::load, "Load encoder from file");

    class_<RResolveModel>("ResolveModel")
        .constructor<List, List>("Create a ResolveModel")
        .method("forward", &RResolveModel::forward, "Forward pass")
        .method("get_latent", &RResolveModel::get_latent, "Get latent representations")
        .method("train", &RResolveModel::train, "Set training mode")
        .method("eval", &RResolveModel::eval, "Set evaluation mode")
        .method("to_device", &RResolveModel::to_device, "Move model to device")
        .method("latent_dim", &RResolveModel::latent_dim, "Get latent dimension")
        ;

    class_<RTrainer>("Trainer")
        .constructor<RResolveModel&, List>("Create a Trainer")
        .method("prepare_data", &RTrainer::prepare_data, "Prepare training data")
        .method("fit", &RTrainer::fit, "Train the model")
        .method("save", &RTrainer::save, "Save model checkpoint")
        ;

    class_<RPredictor>("Predictor")
        .method("predict", &RPredictor::predict, "Make predictions")
        .method("get_embeddings", &RPredictor::get_embeddings, "Get latent embeddings")
        .method("get_genus_embeddings", &RPredictor::get_genus_embeddings, "Get genus embeddings")
        .method("get_family_embeddings", &RPredictor::get_family_embeddings, "Get family embeddings")
        ;

    function("Predictor_load", &RPredictor::load, "Load predictor from checkpoint");
}

// =============================================================================
// Package initialization
// =============================================================================

// [[Rcpp::export]]
std::string resolve_version() {
    return resolve::VERSION;
}
