// rcpp_common.h - Shared type conversions and utilities for R bindings
#ifndef RCPP_COMMON_H
#define RCPP_COMMON_H

#include <Rcpp.h>
#include <torch/torch.h>
#include <limits>
#include "resolve/resolve.hpp"

using namespace Rcpp;

// =============================================================================
// Type conversion: R -> Torch
// =============================================================================

// R's NumericVector stores values as 64-bit doubles. torch::from_blob does NOT
// convert the underlying memory — passing a double* with a kFloat32 dtype option
// reinterprets the byte buffer as float32, producing garbage values. We must
// convert element-by-element via static_cast<float>(...) before constructing
// the tensor, matching the pattern used by r_mat_to_tensor_impl below.
inline torch::Tensor r_vec_to_tensor(NumericVector x) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    std::vector<float> data(x.size());
    for (R_xlen_t i = 0; i < x.size(); ++i) {
        data[i] = static_cast<float>(x[i]);
    }
    return torch::from_blob(
        data.data(),
        {static_cast<int64_t>(x.size())},
        options
    ).clone();
}

// Generic matrix-to-tensor: works for NumericMatrix (float) and IntegerMatrix (int64)
template <typename MatT, typename ScalarT, torch::ScalarType DType>
inline torch::Tensor r_mat_to_tensor_impl(MatT x) {
    auto options = torch::TensorOptions().dtype(DType);
    int nrow = x.nrow();
    int ncol = x.ncol();
    std::vector<ScalarT> data(nrow * ncol);
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            data[i * ncol + j] = static_cast<ScalarT>(x(i, j));
        }
    }
    return torch::from_blob(data.data(), {nrow, ncol}, options).clone();
}

inline torch::Tensor r_mat_to_tensor(NumericMatrix x) {
    return r_mat_to_tensor_impl<NumericMatrix, float, torch::kFloat32>(x);
}

inline torch::Tensor r_int_vec_to_tensor(IntegerVector x) {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    std::vector<int64_t> data(x.begin(), x.end());
    return torch::from_blob(data.data(), {static_cast<int64_t>(x.size())}, options).clone();
}

inline torch::Tensor r_int_mat_to_tensor(IntegerMatrix x) {
    return r_mat_to_tensor_impl<IntegerMatrix, int64_t, torch::kInt64>(x);
}

// =============================================================================
// Type conversion: Torch -> R
// =============================================================================

inline NumericVector tensor_to_r_vec(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    float* data = cpu.data_ptr<float>();
    return NumericVector(data, data + cpu.numel());
}

inline NumericMatrix tensor_to_r_mat(const torch::Tensor& t) {
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

// Integer matrix conversion — for int64 tensors like categorical_ids.
// R's IntegerMatrix is 32-bit; we narrow but warn if any value overflows.
inline IntegerMatrix tensor_to_r_imat(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kInt64);
    int nrow = cpu.size(0);
    int ncol = cpu.size(1);
    IntegerMatrix out(nrow, ncol);
    int64_t* data = cpu.data_ptr<int64_t>();
    for (int i = 0; i < nrow; ++i) {
        for (int j = 0; j < ncol; ++j) {
            const int64_t v = data[i * ncol + j];
            if (v > std::numeric_limits<int>::max() ||
                v < std::numeric_limits<int>::min()) {
                Rcpp::warning("tensor_to_r_imat: value out of int range, "
                              "narrowed: %lld", static_cast<long long>(v));
            }
            out(i, j) = static_cast<int>(v);
        }
    }
    return out;
}

// =============================================================================
// Enum conversions
// =============================================================================

// Generic string-to-enum parser. Avoids duplicating the same if/stop pattern
// for every enum type. Entries is an initializer_list of {string, EnumValue}.
template <typename EnumT>
inline EnumT parse_enum(
    const std::string& s,
    std::initializer_list<std::pair<const char*, EnumT>> entries,
    const char* type_name
) {
    for (const auto& [key, val] : entries) {
        if (s == key) return val;
    }
    stop("Invalid " + std::string(type_name) + ": " + s);
}

inline resolve::SelectionMode parse_selection_mode(const std::string& s) {
    return parse_enum<resolve::SelectionMode>(s, {
        {"top", resolve::SelectionMode::Top},
        {"bottom", resolve::SelectionMode::Bottom},
        {"top_bottom", resolve::SelectionMode::TopBottom},
        {"all", resolve::SelectionMode::All},
    }, "selection mode");
}

inline resolve::RepresentationMode parse_representation_mode(const std::string& s) {
    return parse_enum<resolve::RepresentationMode>(s, {
        {"abundance", resolve::RepresentationMode::Abundance},
        {"presence_absence", resolve::RepresentationMode::PresenceAbsence},
    }, "representation mode");
}

inline resolve::NormalizationMode parse_normalization_mode(const std::string& s) {
    return parse_enum<resolve::NormalizationMode>(s, {
        {"raw", resolve::NormalizationMode::Raw},
        {"norm", resolve::NormalizationMode::Norm},
        {"log1p", resolve::NormalizationMode::Log1p},
    }, "normalization mode");
}

inline resolve::AggregationMode parse_aggregation_mode(const std::string& s) {
    return parse_enum<resolve::AggregationMode>(s, {
        {"abundance", resolve::AggregationMode::Abundance},
        {"count", resolve::AggregationMode::Count},
    }, "aggregation mode");
}

inline resolve::PoolWeighting parse_pool_weighting(const std::string& s) {
    return parse_enum<resolve::PoolWeighting>(s, {
        {"binary", resolve::PoolWeighting::Binary},
        {"abundance", resolve::PoolWeighting::Abundance},
        {"log1p", resolve::PoolWeighting::Log1p},
        {"norm", resolve::PoolWeighting::Norm},
        {"rank", resolve::PoolWeighting::Rank},
    }, "pool weighting");
}

inline resolve::TaskType parse_task_type(const std::string& s) {
    return parse_enum<resolve::TaskType>(s, {
        {"regression", resolve::TaskType::Regression},
        {"classification", resolve::TaskType::Classification},
    }, "task type");
}

inline resolve::TransformType parse_transform_type(const std::string& s) {
    return parse_enum<resolve::TransformType>(s, {
        {"none", resolve::TransformType::None},
        {"log1p", resolve::TransformType::Log1p},
    }, "transform type");
}

inline resolve::SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    return parse_enum<resolve::SpeciesEncodingMode>(s, {
        {"hash", resolve::SpeciesEncodingMode::Hash},
        {"embed", resolve::SpeciesEncodingMode::Embed},
        {"sparse", resolve::SpeciesEncodingMode::Sparse},
        {"rank_pool", resolve::SpeciesEncodingMode::RankPool},
        {"transformer", resolve::SpeciesEncodingMode::Transformer},
    }, "species encoding mode");
}

inline resolve::LossConfigMode parse_loss_config_mode(const std::string& s) {
    return parse_enum<resolve::LossConfigMode>(s, {
        {"mae", resolve::LossConfigMode::MAE},
        {"smape", resolve::LossConfigMode::SMAPE},
        {"combined", resolve::LossConfigMode::Combined},
        {"nca", resolve::LossConfigMode::NCA},
    }, "loss config mode");
}

inline resolve::LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    return parse_enum<resolve::LRSchedulerType>(s, {
        {"none", resolve::LRSchedulerType::None},
        {"step", resolve::LRSchedulerType::StepLR},
        {"cosine", resolve::LRSchedulerType::CosineAnnealing},
    }, "LR scheduler type");
}

// =============================================================================
// Enum parsers: new enums for full R binding parity
// =============================================================================

inline resolve::MoERoutingType parse_moe_routing_type(const std::string& s) {
    return parse_enum<resolve::MoERoutingType>(s, {
        {"none", resolve::MoERoutingType::None},
        {"soft", resolve::MoERoutingType::Soft},
        {"topk", resolve::MoERoutingType::TopK},
    }, "MoE routing type");
}

inline resolve::ActivationType parse_activation_type(const std::string& s) {
    return parse_enum<resolve::ActivationType>(s, {
        {"relu", resolve::ActivationType::ReLU},
        {"leaky_relu", resolve::ActivationType::LeakyReLU},
        {"gelu", resolve::ActivationType::GELU},
        {"silu", resolve::ActivationType::SiLU},
        {"tanh", resolve::ActivationType::Tanh},
        {"mish", resolve::ActivationType::Mish},
        {"elu", resolve::ActivationType::ELU},
        {"selu", resolve::ActivationType::SELU},
        {"softplus", resolve::ActivationType::Softplus},
        {"prelu", resolve::ActivationType::PReLU},
    }, "activation type");
}

inline resolve::NormLayerType parse_norm_layer_type(const std::string& s) {
    return parse_enum<resolve::NormLayerType>(s, {
        {"batch_norm", resolve::NormLayerType::BatchNorm},
        {"layer_norm", resolve::NormLayerType::LayerNorm},
        {"group_norm", resolve::NormLayerType::GroupNorm},
        {"rms_norm", resolve::NormLayerType::RMSNorm},
        {"none", resolve::NormLayerType::None},
    }, "normalization layer type");
}

inline resolve::EncoderArchitecture parse_encoder_architecture(const std::string& s) {
    return parse_enum<resolve::EncoderArchitecture>(s, {
        {"mlp", resolve::EncoderArchitecture::MLP},
        {"ft_transformer", resolve::EncoderArchitecture::FTTransformer},
        {"tabnet", resolve::EncoderArchitecture::TabNet},
        {"saint", resolve::EncoderArchitecture::SAINT},
        {"trait_net", resolve::EncoderArchitecture::TraitNet},
        {"gnn", resolve::EncoderArchitecture::GNN},
        {"excelformer", resolve::EncoderArchitecture::ExcelFormer},
        {"heterogeneous_gnn", resolve::EncoderArchitecture::HeterogeneousGNN},
    }, "encoder architecture");
}

inline resolve::GNNType parse_gnn_type(const std::string& s) {
    return parse_enum<resolve::GNNType>(s, {
        {"gcn", resolve::GNNType::GCN},
        {"gat", resolve::GNNType::GAT},
        {"graphsage", resolve::GNNType::GraphSAGE},
    }, "GNN type");
}

inline resolve::GraphConstructionMode parse_graph_construction_mode(const std::string& s) {
    return parse_enum<resolve::GraphConstructionMode>(s, {
        {"spatial", resolve::GraphConstructionMode::Spatial},
        {"taxonomic", resolve::GraphConstructionMode::Taxonomic},
        {"cooccurrence", resolve::GraphConstructionMode::CoOccurrence},
    }, "graph construction mode");
}

inline resolve::TraitInteractionMode parse_trait_interaction_mode(const std::string& s) {
    return parse_enum<resolve::TraitInteractionMode>(s, {
        {"bilinear", resolve::TraitInteractionMode::Bilinear},
        {"mlp", resolve::TraitInteractionMode::MLP},
        {"attention", resolve::TraitInteractionMode::Attention},
    }, "trait interaction mode");
}

inline resolve::ParallelAggregation parse_parallel_aggregation(const std::string& s) {
    return parse_enum<resolve::ParallelAggregation>(s, {
        {"concat", resolve::ParallelAggregation::Concat},
        {"sum", resolve::ParallelAggregation::Sum},
        {"mean", resolve::ParallelAggregation::Mean},
        {"attention", resolve::ParallelAggregation::Attention},
        {"gated", resolve::ParallelAggregation::Gated},
    }, "parallel aggregation");
}

// =============================================================================
// Sub-config parsers: R List -> C++ config struct
// =============================================================================

inline resolve::FTTransformerConfig parse_ft_transformer_config(List cfg) {
    resolve::FTTransformerConfig c;
    if (cfg.containsElementNamed("d_model")) c.d_model = cfg["d_model"];
    if (cfg.containsElementNamed("n_heads")) c.n_heads = cfg["n_heads"];
    if (cfg.containsElementNamed("n_layers")) c.n_layers = cfg["n_layers"];
    if (cfg.containsElementNamed("attention_dropout")) c.attention_dropout = cfg["attention_dropout"];
    if (cfg.containsElementNamed("ffn_dropout")) c.ffn_dropout = cfg["ffn_dropout"];
    if (cfg.containsElementNamed("ffn_multiplier")) c.ffn_multiplier = cfg["ffn_multiplier"];
    if (cfg.containsElementNamed("pre_norm")) c.pre_norm = cfg["pre_norm"];
    return c;
}

inline resolve::TabNetConfig parse_tabnet_config(List cfg) {
    resolve::TabNetConfig c;
    if (cfg.containsElementNamed("n_steps")) c.n_steps = cfg["n_steps"];
    if (cfg.containsElementNamed("n_d")) c.n_d = cfg["n_d"];
    if (cfg.containsElementNamed("n_a")) c.n_a = cfg["n_a"];
    if (cfg.containsElementNamed("relaxation_factor")) c.relaxation_factor = cfg["relaxation_factor"];
    if (cfg.containsElementNamed("sparsity_coefficient")) c.sparsity_coefficient = cfg["sparsity_coefficient"];
    if (cfg.containsElementNamed("virtual_batch_size")) c.virtual_batch_size = cfg["virtual_batch_size"];
    if (cfg.containsElementNamed("use_sparsemax")) c.use_sparsemax = cfg["use_sparsemax"];
    return c;
}

inline resolve::SAINTConfig parse_saint_config(List cfg) {
    resolve::SAINTConfig c;
    if (cfg.containsElementNamed("d_model")) c.d_model = cfg["d_model"];
    if (cfg.containsElementNamed("n_heads")) c.n_heads = cfg["n_heads"];
    if (cfg.containsElementNamed("n_layers")) c.n_layers = cfg["n_layers"];
    if (cfg.containsElementNamed("attention_dropout")) c.attention_dropout = cfg["attention_dropout"];
    if (cfg.containsElementNamed("use_row_attention")) c.use_row_attention = cfg["use_row_attention"];
    if (cfg.containsElementNamed("use_contrastive_pretrain")) c.use_contrastive_pretrain = cfg["use_contrastive_pretrain"];
    if (cfg.containsElementNamed("mixup_alpha")) c.mixup_alpha = cfg["mixup_alpha"];
    return c;
}

inline resolve::GNNConfig parse_gnn_config(List cfg) {
    resolve::GNNConfig c;
    if (cfg.containsElementNamed("gnn_type")) c.gnn_type = parse_gnn_type(as<std::string>(cfg["gnn_type"]));
    if (cfg.containsElementNamed("n_layers")) c.n_layers = cfg["n_layers"];
    if (cfg.containsElementNamed("hidden_dim")) c.hidden_dim = cfg["hidden_dim"];
    if (cfg.containsElementNamed("n_heads")) c.n_heads = cfg["n_heads"];
    if (cfg.containsElementNamed("k_neighbors")) c.k_neighbors = cfg["k_neighbors"];
    if (cfg.containsElementNamed("graph_mode")) c.graph_mode = parse_graph_construction_mode(as<std::string>(cfg["graph_mode"]));
    if (cfg.containsElementNamed("edge_dropout")) c.edge_dropout = cfg["edge_dropout"];
    if (cfg.containsElementNamed("use_edge_features")) c.use_edge_features = cfg["use_edge_features"];
    return c;
}

inline resolve::TraitNetConfig parse_trait_net_config(List cfg) {
    resolve::TraitNetConfig c;
    if (cfg.containsElementNamed("env_dim")) c.env_dim = cfg["env_dim"];
    if (cfg.containsElementNamed("trait_dim")) c.trait_dim = cfg["trait_dim"];
    if (cfg.containsElementNamed("interaction_dim")) c.interaction_dim = cfg["interaction_dim"];
    if (cfg.containsElementNamed("interaction")) c.interaction = parse_trait_interaction_mode(as<std::string>(cfg["interaction"]));
    if (cfg.containsElementNamed("shared_trait_encoder")) c.shared_trait_encoder = cfg["shared_trait_encoder"];
    return c;
}

inline resolve::ExcelFormerConfig parse_excelformer_config(List cfg) {
    resolve::ExcelFormerConfig c;
    if (cfg.containsElementNamed("d_model")) c.d_model = cfg["d_model"];
    if (cfg.containsElementNamed("n_heads")) c.n_heads = cfg["n_heads"];
    if (cfg.containsElementNamed("n_layers")) c.n_layers = cfg["n_layers"];
    if (cfg.containsElementNamed("attention_dropout")) c.attention_dropout = cfg["attention_dropout"];
    if (cfg.containsElementNamed("ffn_multiplier")) c.ffn_multiplier = cfg["ffn_multiplier"];
    if (cfg.containsElementNamed("importance_threshold")) c.importance_threshold = cfg["importance_threshold"];
    if (cfg.containsElementNamed("pre_norm")) c.pre_norm = cfg["pre_norm"];
    return c;
}

inline resolve::HeterogeneousGNNConfig parse_heterogeneous_gnn_config(List cfg) {
    resolve::HeterogeneousGNNConfig c;
    if (cfg.containsElementNamed("hidden_dim")) c.hidden_dim = cfg["hidden_dim"];
    if (cfg.containsElementNamed("output_dim")) c.output_dim = cfg["output_dim"];
    if (cfg.containsElementNamed("n_layers")) c.n_layers = cfg["n_layers"];
    if (cfg.containsElementNamed("n_edge_types")) c.n_edge_types = cfg["n_edge_types"];
    if (cfg.containsElementNamed("n_heads")) c.n_heads = cfg["n_heads"];
    if (cfg.containsElementNamed("dropout")) c.dropout = cfg["dropout"];
    if (cfg.containsElementNamed("k_cooccurrence")) c.k_cooccurrence = cfg["k_cooccurrence"];
    if (cfg.containsElementNamed("cooccurrence_threshold")) c.cooccurrence_threshold = cfg["cooccurrence_threshold"];
    if (cfg.containsElementNamed("use_taxonomic_edges")) c.use_taxonomic_edges = cfg["use_taxonomic_edges"];
    if (cfg.containsElementNamed("use_cooccurrence_edges")) c.use_cooccurrence_edges = cfg["use_cooccurrence_edges"];
    return c;
}

inline resolve::TabMConfig parse_tabm_config(List cfg) {
    resolve::TabMConfig c;
    if (cfg.containsElementNamed("enabled")) c.enabled = cfg["enabled"];
    if (cfg.containsElementNamed("n_ensembles")) c.n_ensembles = cfg["n_ensembles"];
    if (cfg.containsElementNamed("aggregation")) c.aggregation = as<std::string>(cfg["aggregation"]);
    return c;
}

inline resolve::ParallelBranchConfig parse_parallel_branch_config(List cfg) {
    resolve::ParallelBranchConfig c;
    if (cfg.containsElementNamed("hidden_dims")) c.hidden_dims = as<std::vector<int64_t>>(cfg["hidden_dims"]);
    if (cfg.containsElementNamed("activation")) c.activation = parse_activation_type(as<std::string>(cfg["activation"]));
    if (cfg.containsElementNamed("normalization")) c.normalization = parse_norm_layer_type(as<std::string>(cfg["normalization"]));
    if (cfg.containsElementNamed("dropout")) c.dropout = cfg["dropout"];
    if (cfg.containsElementNamed("branch_weight")) c.branch_weight = cfg["branch_weight"];
    return c;
}

inline resolve::ParallelLayersConfig parse_parallel_layers_config(List cfg) {
    resolve::ParallelLayersConfig c;
    if (cfg.containsElementNamed("enabled")) c.enabled = cfg["enabled"];
    if (cfg.containsElementNamed("branches")) {
        List branches = cfg["branches"];
        for (int i = 0; i < branches.size(); ++i) {
            c.branches.push_back(parse_parallel_branch_config(as<List>(branches[i])));
        }
    }
    if (cfg.containsElementNamed("aggregation")) c.aggregation = parse_parallel_aggregation(as<std::string>(cfg["aggregation"]));
    if (cfg.containsElementNamed("attention_heads")) c.attention_heads = cfg["attention_heads"];
    if (cfg.containsElementNamed("use_residual")) c.use_residual = cfg["use_residual"];
    return c;
}

// =============================================================================
// Result converters: C++ struct -> R List
// =============================================================================

inline List baseline_metrics_to_list(const resolve::BaselineMetrics& bm) {
    return List::create(
        Named("baseline_mse") = bm.baseline_mse,
        Named("baseline_mae") = bm.baseline_mae,
        Named("model_mse") = bm.model_mse,
        Named("model_mae") = bm.model_mae,
        Named("skill_score") = bm.skill_score,
        Named("r_squared") = bm.r_squared,
        Named("baseline_accuracy") = bm.baseline_accuracy,
        Named("model_accuracy") = bm.model_accuracy,
        Named("accuracy_lift") = bm.accuracy_lift,
        Named("training_mean") = bm.training_mean,
        Named("training_mode") = bm.training_mode
    );
}

inline List layer_diagnostics_to_list(const resolve::LayerDiagnostics& ld) {
    return List::create(
        Named("name") = ld.name,
        Named("n_neurons") = (int)ld.n_neurons,
        Named("n_dead") = (int)ld.n_dead,
        Named("n_saturated") = (int)ld.n_saturated,
        Named("dead_fraction") = ld.dead_fraction,
        Named("saturated_fraction") = ld.saturated_fraction,
        Named("mean_activation") = ld.mean_activation,
        Named("std_activation") = ld.std_activation,
        Named("sparsity") = ld.sparsity
    );
}

inline List network_diagnostics_to_list(const resolve::NetworkDiagnostics& nd) {
    List layers_list;
    for (const auto& ld : nd.layers) {
        layers_list.push_back(layer_diagnostics_to_list(ld));
    }
    return List::create(
        Named("layers") = layers_list,
        Named("total_neurons") = (int)nd.total_neurons,
        Named("total_dead") = (int)nd.total_dead,
        Named("total_saturated") = (int)nd.total_saturated,
        Named("overall_dead_fraction") = nd.overall_dead_fraction,
        Named("overall_saturated_fraction") = nd.overall_saturated_fraction,
        Named("has_issues") = nd.has_issues,
        Named("summary") = nd.summary
    );
}

inline List nested_metrics_to_list(
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
) {
    List result;
    for (const auto& [target, metric_map] : metrics) {
        List inner;
        for (const auto& [metric, value] : metric_map) {
            inner[metric] = value;
        }
        result[target] = inner;
    }
    return result;
}

inline List train_result_to_list(const resolve::TrainResult& tr) {
    // Baselines
    List baselines;
    for (const auto& [target, bm] : tr.baselines) {
        baselines[target] = baseline_metrics_to_list(bm);
    }

    return List::create(
        Named("best_epoch") = tr.best_epoch,
        Named("final_metrics") = nested_metrics_to_list(tr.final_metrics),
        Named("train_loss") = wrap(tr.train_loss_history),
        Named("test_loss") = wrap(tr.test_loss_history),
        Named("train_time_seconds") = tr.train_time_seconds,
        Named("resumed_from_epoch") = tr.resumed_from_epoch,
        Named("baselines") = baselines,
        Named("diagnostics") = network_diagnostics_to_list(tr.diagnostics)
    );
}

inline List calibration_bin_to_list(const resolve::CalibrationBin& cb) {
    return List::create(
        Named("bin_start") = cb.bin_start,
        Named("bin_end") = cb.bin_end,
        Named("mean_predicted_prob") = cb.mean_predicted_prob,
        Named("actual_frequency") = cb.actual_frequency,
        Named("count") = (int)cb.count
    );
}

inline List calibration_result_to_list(const resolve::CalibrationResult& cr) {
    List bins;
    for (const auto& b : cr.bins) {
        bins.push_back(calibration_bin_to_list(b));
    }
    return List::create(
        Named("target_name") = cr.target_name,
        Named("class_idx") = cr.class_idx,
        Named("bins") = bins,
        Named("expected_calibration_error") = cr.expected_calibration_error,
        Named("max_calibration_error") = cr.max_calibration_error
    );
}

inline List residual_analysis_to_list(const resolve::ResidualAnalysis& ra) {
    return List::create(
        Named("target_name") = ra.target_name,
        Named("predictions") = wrap(ra.predictions),
        Named("actuals") = wrap(ra.actuals),
        Named("residuals") = wrap(ra.residuals),
        Named("mean_residual") = ra.mean_residual,
        Named("std_residual") = ra.std_residual,
        Named("skewness") = ra.skewness,
        Named("kurtosis") = ra.kurtosis,
        Named("q05") = ra.q05,
        Named("q25") = ra.q25,
        Named("q50") = ra.q50,
        Named("q75") = ra.q75,
        Named("q95") = ra.q95
    );
}

inline List cross_validation_result_to_list(const resolve::CrossValidationResult& cvr) {
    List fold_results;
    for (const auto& fr : cvr.fold_results) {
        fold_results.push_back(train_result_to_list(fr));
    }
    return List::create(
        Named("n_folds") = cvr.n_folds,
        Named("mean_metrics") = nested_metrics_to_list(cvr.mean_metrics),
        Named("std_metrics") = nested_metrics_to_list(cvr.std_metrics),
        Named("fold_results") = fold_results,
        Named("total_time_seconds") = cvr.total_time_seconds
    );
}

inline List scalers_to_list(const resolve::Scalers& s) {
    List result;
    if (s.continuous_mean.defined()) {
        result["continuous_mean"] = tensor_to_r_vec(s.continuous_mean);
    }
    if (s.continuous_scale.defined()) {
        result["continuous_scale"] = tensor_to_r_vec(s.continuous_scale);
    }
    return result;
}

inline List run_metadata_to_list(const resolve::RunMetadata& rm) {
    return List::create(
        Named("resolve_version") = rm.resolve_version,
        Named("created_at") = rm.created_at,
        Named("completed_at") = rm.completed_at,
        Named("train_time_seconds") = rm.train_time_seconds,
        Named("n_plots_train") = (int)rm.n_plots_train,
        Named("n_plots_test") = (int)rm.n_plots_test,
        Named("best_epoch") = rm.best_epoch,
        Named("total_epochs") = rm.total_epochs,
        Named("final_metrics") = nested_metrics_to_list(rm.final_metrics)
    );
}

inline resolve::RunMetadata parse_run_metadata(List cfg) {
    resolve::RunMetadata rm;
    if (cfg.containsElementNamed("created_at")) rm.created_at = as<std::string>(cfg["created_at"]);
    if (cfg.containsElementNamed("completed_at")) rm.completed_at = as<std::string>(cfg["completed_at"]);
    if (cfg.containsElementNamed("train_time_seconds")) rm.train_time_seconds = cfg["train_time_seconds"];
    if (cfg.containsElementNamed("n_plots_train")) rm.n_plots_train = cfg["n_plots_train"];
    if (cfg.containsElementNamed("n_plots_test")) rm.n_plots_test = cfg["n_plots_test"];
    if (cfg.containsElementNamed("best_epoch")) rm.best_epoch = cfg["best_epoch"];
    if (cfg.containsElementNamed("total_epochs")) rm.total_epochs = cfg["total_epochs"];
    return rm;
}

inline resolve::SpatialBlockConfig parse_spatial_block_config(List cfg) {
    resolve::SpatialBlockConfig c;
    if (cfg.containsElementNamed("lat_size")) c.lat_size = cfg["lat_size"];
    if (cfg.containsElementNamed("lon_size")) c.lon_size = cfg["lon_size"];
    if (cfg.containsElementNamed("balance")) c.balance = cfg["balance"];
    return c;
}

#endif // RCPP_COMMON_HPP
