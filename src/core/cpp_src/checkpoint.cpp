#include "resolve/checkpoint.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>

namespace resolve {

namespace {
// Escape a string for embedding in a JSON string literal. Target names and
// class labels come from user CSV headers and may contain " or \; emitting them
// raw produces invalid JSON that a monitoring parser chokes on.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}
}  // namespace

void write_progress_file(
    const std::string& checkpoint_dir,
    int epoch,
    int max_epochs,
    int best_epoch,
    float best_loss,
    int epochs_without_improvement,
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics
) {
    namespace fs = std::filesystem;
    fs::create_directories(checkpoint_dir);

    std::string progress_path = checkpoint_dir + "/progress.json";
    std::ofstream file(progress_path);
    if (!file.is_open()) return;

    file << "{\n";
    file << "  \"epoch\": " << epoch << ",\n";
    file << "  \"max_epochs\": " << max_epochs << ",\n";
    file << "  \"best_epoch\": " << best_epoch << ",\n";
    file << "  \"best_loss\": " << best_loss << ",\n";
    file << "  \"epochs_without_improvement\": " << epochs_without_improvement << ",\n";
    file << "  \"progress_pct\": "
         << (max_epochs > 0 ? 100.0f * epoch / max_epochs : 0.0f) << ",\n";

    // Write a deterministic "best metric": the lexicographically-first band
    // accuracy of the alphabetically-first target. `metrics` is an unordered_map,
    // so select via sorted keys rather than iteration order (which is unspecified
    // and made the reported value vary across runs/builds). Ordering is
    // lexicographic on the band name, which equals numeric order only for
    // equal-width thresholds (band_25/50/75); a band_100 would sort before band_25.
    float best_metric = 0.0f;
    std::string best_metric_name;
    {
        std::vector<std::string> target_names;
        target_names.reserve(metrics.size());
        for (const auto& [name, unused] : metrics) target_names.push_back(name);
        std::sort(target_names.begin(), target_names.end());
        for (const auto& tname : target_names) {
            const auto& tm = metrics.at(tname);
            std::vector<std::string> band_names;
            for (const auto& [mname, unused] : tm) {
                if (mname.rfind("band_", 0) == 0) band_names.push_back(mname);
            }
            if (!band_names.empty()) {
                std::sort(band_names.begin(), band_names.end());
                best_metric = tm.at(band_names.front());
                best_metric_name = tname + "/" + band_names.front();
                break;
            }
        }
    }
    file << "  \"best_metric\": " << best_metric << ",\n";
    file << "  \"best_metric_name\": \"" << json_escape(best_metric_name) << "\"\n";
    file << "}\n";
}

void save_model_config(
    torch::serialize::OutputArchive& archive,
    const ModelConfig& config
) {
    archive.write("species_encoding", torch::tensor(static_cast<int>(config.species_encoding)));
    archive.write("uses_explicit_vector", torch::tensor(static_cast<int>(config.uses_explicit_vector)));
    archive.write("hash_dim", torch::tensor(config.hash_dim));
    archive.write("species_embed_dim", torch::tensor(config.species_embed_dim));
    archive.write("genus_emb_dim", torch::tensor(config.genus_emb_dim));
    archive.write("family_emb_dim", torch::tensor(config.family_emb_dim));
    archive.write("categorical_embed_dim", torch::tensor(config.categorical_embed_dim));
    archive.write("top_k", torch::tensor(config.top_k));
    archive.write("top_k_species", torch::tensor(config.top_k_species));
    archive.write("n_taxonomy_slots", torch::tensor(config.n_taxonomy_slots));
    archive.write("dropout", torch::tensor(config.dropout));

    // Save hidden dims
    std::vector<int64_t> hidden_dims_vec(config.hidden_dims);
    archive.write("hidden_dims", torch::tensor(hidden_dims_vec));

    // Save MoE configuration
    archive.write("moe_routing", torch::tensor(static_cast<int>(config.moe_routing)));
    archive.write("n_experts", torch::tensor(config.n_experts));
    archive.write("moe_top_k", torch::tensor(config.moe_top_k));
    archive.write("moe_noise_std", torch::tensor(config.moe_noise_std));
    archive.write("moe_aux_loss_weight", torch::tensor(config.moe_aux_loss_weight));

    // Save expert hidden dims
    std::vector<int64_t> expert_dims_vec(config.expert_hidden_dims);
    archive.write("expert_hidden_dims", torch::tensor(expert_dims_vec));

    // Save configurable architecture fields
    archive.write("activation", torch::tensor(static_cast<int>(config.activation)));
    archive.write("normalization", torch::tensor(static_cast<int>(config.normalization)));
    archive.write("norm_groups", torch::tensor(config.norm_groups));
    archive.write("use_residual", torch::tensor(static_cast<int>(config.use_residual)));
    archive.write("leaky_relu_slope", torch::tensor(config.leaky_relu_slope));
    archive.write("elu_alpha", torch::tensor(config.elu_alpha));

    // Save head architecture
    std::vector<int64_t> head_dims_vec(config.head_hidden_dims);
    archive.write("head_hidden_dims", torch::tensor(head_dims_vec));
    archive.write("head_activation", torch::tensor(static_cast<int>(config.head_activation)));
    archive.write("head_dropout", torch::tensor(config.head_dropout));

    // Save encoder architecture
    archive.write("encoder_architecture", torch::tensor(static_cast<int>(config.encoder_architecture)));

    // Save TabM configuration
    archive.write("tabm_enabled", torch::tensor(static_cast<int>(config.tabm.enabled)));
    archive.write("tabm_n_ensembles", torch::tensor(config.tabm.n_ensembles));
    // Save aggregation as bytes
    std::vector<uint8_t> tabm_agg_bytes(config.tabm.aggregation.begin(), config.tabm.aggregation.end());
    archive.write("tabm_aggregation_len", torch::tensor(static_cast<int64_t>(tabm_agg_bytes.size())));
    if (!tabm_agg_bytes.empty()) {
        archive.write("tabm_aggregation", torch::from_blob(
            tabm_agg_bytes.data(), {static_cast<int64_t>(tabm_agg_bytes.size())}, torch::kUInt8).clone());
    }

    // RankPool / Transformer encoder fields. These shape the embedding
    // tables and (for transformer) the attention stack, so the loaded
    // ModelConfig must carry them through Predictor.load -- otherwise the
    // reconstructed model is sized differently from the saved one and the
    // first matmul blows up. Mirrors pitfall #7 in port.md
    // (the same trap categorical_embed_dim hit during the cat port).
    archive.write("cover_dropout", torch::tensor(config.cover_dropout));
    archive.write("d_model", torch::tensor(config.d_model));
    archive.write("n_heads", torch::tensor(config.n_heads));
    archive.write("n_attention_layers", torch::tensor(config.n_attention_layers));
    archive.write("transformer_ff_dim", torch::tensor(config.transformer_ff_dim));
    archive.write("transformer_dropout", torch::tensor(config.transformer_dropout));
    std::vector<uint8_t> tx_pool_bytes(
        config.transformer_pooling.begin(), config.transformer_pooling.end());
    archive.write("transformer_pooling_len",
                  torch::tensor(static_cast<int64_t>(tx_pool_bytes.size())));
    if (!tx_pool_bytes.empty()) {
        archive.write("transformer_pooling", torch::from_blob(
            tx_pool_bytes.data(),
            {static_cast<int64_t>(tx_pool_bytes.size())},
            torch::kUInt8).clone());
    }

    // Architecture-specific sub-configs (issue #37). These size the layers built
    // by TabularAdapter / TraitNetEncoder, so a checkpoint of a non-MLP encoder
    // must carry them or Predictor::load rebuilds default-sized modules whose
    // weights then mismatch or fail to load. Written unconditionally; the loader
    // reads each with try_read so pre-#37 checkpoints keep the struct defaults.
    // FT-Transformer
    archive.write("ft_d_model", torch::tensor(config.ft_transformer.d_model));
    archive.write("ft_n_heads", torch::tensor(config.ft_transformer.n_heads));
    archive.write("ft_n_layers", torch::tensor(config.ft_transformer.n_layers));
    archive.write("ft_attention_dropout", torch::tensor(config.ft_transformer.attention_dropout));
    archive.write("ft_ffn_dropout", torch::tensor(config.ft_transformer.ffn_dropout));
    archive.write("ft_ffn_multiplier", torch::tensor(config.ft_transformer.ffn_multiplier));
    archive.write("ft_pre_norm", torch::tensor(static_cast<int>(config.ft_transformer.pre_norm)));
    // TabNet
    archive.write("tabnet_n_steps", torch::tensor(config.tabnet.n_steps));
    archive.write("tabnet_n_d", torch::tensor(config.tabnet.n_d));
    archive.write("tabnet_n_a", torch::tensor(config.tabnet.n_a));
    archive.write("tabnet_relaxation_factor", torch::tensor(config.tabnet.relaxation_factor));
    archive.write("tabnet_sparsity_coefficient", torch::tensor(config.tabnet.sparsity_coefficient));
    archive.write("tabnet_virtual_batch_size", torch::tensor(config.tabnet.virtual_batch_size));
    archive.write("tabnet_use_sparsemax", torch::tensor(static_cast<int>(config.tabnet.use_sparsemax)));
    // SAINT
    archive.write("saint_d_model", torch::tensor(config.saint.d_model));
    archive.write("saint_n_heads", torch::tensor(config.saint.n_heads));
    archive.write("saint_n_layers", torch::tensor(config.saint.n_layers));
    archive.write("saint_attention_dropout", torch::tensor(config.saint.attention_dropout));
    archive.write("saint_use_row_attention", torch::tensor(static_cast<int>(config.saint.use_row_attention)));
    archive.write("saint_use_contrastive_pretrain", torch::tensor(static_cast<int>(config.saint.use_contrastive_pretrain)));
    archive.write("saint_mixup_alpha", torch::tensor(config.saint.mixup_alpha));
    // GNN
    archive.write("gnn_type", torch::tensor(static_cast<int>(config.gnn.gnn_type)));
    archive.write("gnn_n_layers", torch::tensor(config.gnn.n_layers));
    archive.write("gnn_hidden_dim", torch::tensor(config.gnn.hidden_dim));
    archive.write("gnn_n_heads", torch::tensor(config.gnn.n_heads));
    archive.write("gnn_k_neighbors", torch::tensor(config.gnn.k_neighbors));
    archive.write("gnn_graph_mode", torch::tensor(static_cast<int>(config.gnn.graph_mode)));
    archive.write("gnn_edge_dropout", torch::tensor(config.gnn.edge_dropout));
    archive.write("gnn_use_edge_features", torch::tensor(static_cast<int>(config.gnn.use_edge_features)));
    // TraitNet
    archive.write("trait_env_dim", torch::tensor(config.trait_net.env_dim));
    archive.write("trait_trait_dim", torch::tensor(config.trait_net.trait_dim));
    archive.write("trait_interaction_dim", torch::tensor(config.trait_net.interaction_dim));
    archive.write("trait_interaction", torch::tensor(static_cast<int>(config.trait_net.interaction)));
    archive.write("trait_shared_trait_encoder", torch::tensor(static_cast<int>(config.trait_net.shared_trait_encoder)));
    // ExcelFormer
    archive.write("excel_d_model", torch::tensor(config.excelformer.d_model));
    archive.write("excel_n_heads", torch::tensor(config.excelformer.n_heads));
    archive.write("excel_n_layers", torch::tensor(config.excelformer.n_layers));
    archive.write("excel_attention_dropout", torch::tensor(config.excelformer.attention_dropout));
    archive.write("excel_ffn_multiplier", torch::tensor(config.excelformer.ffn_multiplier));
    archive.write("excel_importance_threshold", torch::tensor(config.excelformer.importance_threshold));
    archive.write("excel_pre_norm", torch::tensor(static_cast<int>(config.excelformer.pre_norm)));
    // Heterogeneous GNN
    archive.write("hgnn_hidden_dim", torch::tensor(config.heterogeneous_gnn.hidden_dim));
    archive.write("hgnn_output_dim", torch::tensor(config.heterogeneous_gnn.output_dim));
    archive.write("hgnn_n_layers", torch::tensor(config.heterogeneous_gnn.n_layers));
    archive.write("hgnn_n_edge_types", torch::tensor(config.heterogeneous_gnn.n_edge_types));
    archive.write("hgnn_n_heads", torch::tensor(config.heterogeneous_gnn.n_heads));
    archive.write("hgnn_dropout", torch::tensor(config.heterogeneous_gnn.dropout));
    archive.write("hgnn_k_cooccurrence", torch::tensor(config.heterogeneous_gnn.k_cooccurrence));
    archive.write("hgnn_cooccurrence_threshold", torch::tensor(config.heterogeneous_gnn.cooccurrence_threshold));
    archive.write("hgnn_use_taxonomic_edges", torch::tensor(static_cast<int>(config.heterogeneous_gnn.use_taxonomic_edges)));
    archive.write("hgnn_use_cooccurrence_edges", torch::tensor(static_cast<int>(config.heterogeneous_gnn.use_cooccurrence_edges)));
    // Parallel layers (variable number of branches)
    archive.write("parallel_enabled", torch::tensor(static_cast<int>(config.parallel_layers.enabled)));
    archive.write("parallel_aggregation", torch::tensor(static_cast<int>(config.parallel_layers.aggregation)));
    archive.write("parallel_attention_heads", torch::tensor(config.parallel_layers.attention_heads));
    archive.write("parallel_use_residual", torch::tensor(static_cast<int>(config.parallel_layers.use_residual)));
    archive.write("parallel_n_branches",
                  torch::tensor(static_cast<int64_t>(config.parallel_layers.branches.size())));
    for (size_t i = 0; i < config.parallel_layers.branches.size(); ++i) {
        const auto& b = config.parallel_layers.branches[i];
        const std::string p = "parallel_branch_" + std::to_string(i) + "_";
        std::vector<int64_t> hd(b.hidden_dims);
        archive.write(p + "hidden_dims", torch::tensor(hd));
        archive.write(p + "activation", torch::tensor(static_cast<int>(b.activation)));
        archive.write(p + "normalization", torch::tensor(static_cast<int>(b.normalization)));
        archive.write(p + "dropout", torch::tensor(b.dropout));
        archive.write(p + "branch_weight", torch::tensor(b.branch_weight));
    }
}

ModelConfig load_model_config(
    torch::serialize::InputArchive& archive
) {
    torch::Tensor species_encoding_t, uses_explicit_vector_t;
    torch::Tensor hash_dim_t, species_embed_dim_t;
    torch::Tensor genus_emb_dim_t, family_emb_dim_t;
    torch::Tensor top_k_t, top_k_species_t, n_taxonomy_slots_t;
    torch::Tensor dropout_t, hidden_dims_t;

    archive.read("species_encoding", species_encoding_t);
    archive.read("uses_explicit_vector", uses_explicit_vector_t);
    archive.read("hash_dim", hash_dim_t);
    archive.read("species_embed_dim", species_embed_dim_t);
    archive.read("genus_emb_dim", genus_emb_dim_t);
    archive.read("family_emb_dim", family_emb_dim_t);
    archive.read("top_k", top_k_t);
    archive.read("top_k_species", top_k_species_t);
    archive.read("n_taxonomy_slots", n_taxonomy_slots_t);
    archive.read("dropout", dropout_t);
    archive.read("hidden_dims", hidden_dims_t);

    ModelConfig config;
    config.species_encoding = static_cast<SpeciesEncodingMode>(species_encoding_t.item<int>());
    config.uses_explicit_vector = uses_explicit_vector_t.item<int>() != 0;
    config.hash_dim = hash_dim_t.item<int>();
    config.species_embed_dim = species_embed_dim_t.item<int>();
    config.genus_emb_dim = genus_emb_dim_t.item<int>();
    config.family_emb_dim = family_emb_dim_t.item<int>();
    // Back-compat: pre-categorical-port checkpoints didn't save this key,
    // so try_read keeps the ModelConfig default (8) for older models.
    torch::Tensor cat_embed_dim_t;
    if (archive.try_read("categorical_embed_dim", cat_embed_dim_t)) {
        config.categorical_embed_dim = cat_embed_dim_t.item<int>();
    }
    config.top_k = top_k_t.item<int>();
    config.top_k_species = top_k_species_t.item<int>();
    config.n_taxonomy_slots = n_taxonomy_slots_t.item<int>();
    config.dropout = dropout_t.item<float>();

    std::vector<int64_t> hidden_dims(hidden_dims_t.size(0));
    for (int i = 0; i < hidden_dims_t.size(0); ++i) {
        hidden_dims[i] = hidden_dims_t[i].item<int64_t>();
    }
    config.hidden_dims = hidden_dims;

    // Load MoE configuration (with backward compatibility)
    try {
        torch::Tensor moe_routing_t, n_experts_t, moe_top_k_t, moe_noise_std_t, moe_aux_loss_weight_t;
        torch::Tensor expert_hidden_dims_t;

        archive.read("moe_routing", moe_routing_t);
        config.moe_routing = static_cast<MoERoutingType>(moe_routing_t.item<int>());

        archive.read("n_experts", n_experts_t);
        config.n_experts = n_experts_t.item<int>();

        archive.read("moe_top_k", moe_top_k_t);
        config.moe_top_k = moe_top_k_t.item<int>();

        archive.read("moe_noise_std", moe_noise_std_t);
        config.moe_noise_std = moe_noise_std_t.item<float>();

        archive.read("moe_aux_loss_weight", moe_aux_loss_weight_t);
        config.moe_aux_loss_weight = moe_aux_loss_weight_t.item<float>();

        archive.read("expert_hidden_dims", expert_hidden_dims_t);
        config.expert_hidden_dims.clear();
        for (int i = 0; i < expert_hidden_dims_t.size(0); ++i) {
            config.expert_hidden_dims.push_back(expert_hidden_dims_t[i].item<int64_t>());
        }
    } catch (...) {
        // MoE config may not be present in older checkpoints - use defaults
        config.moe_routing = MoERoutingType::None;
    }

    // Load configurable architecture fields (with backward compatibility)
    try {
        torch::Tensor activation_t, normalization_t, norm_groups_t, use_residual_t;
        torch::Tensor leaky_relu_slope_t, elu_alpha_t;
        torch::Tensor head_hidden_dims_t, head_activation_t, head_dropout_t;

        archive.read("activation", activation_t);
        config.activation = static_cast<ActivationType>(activation_t.item<int>());

        archive.read("normalization", normalization_t);
        config.normalization = static_cast<NormLayerType>(normalization_t.item<int>());

        archive.read("norm_groups", norm_groups_t);
        config.norm_groups = norm_groups_t.item<int>();

        archive.read("use_residual", use_residual_t);
        config.use_residual = use_residual_t.item<int>() != 0;

        archive.read("leaky_relu_slope", leaky_relu_slope_t);
        config.leaky_relu_slope = leaky_relu_slope_t.item<float>();

        archive.read("elu_alpha", elu_alpha_t);
        config.elu_alpha = elu_alpha_t.item<float>();

        // Load head architecture
        archive.read("head_hidden_dims", head_hidden_dims_t);
        config.head_hidden_dims.clear();
        for (int i = 0; i < head_hidden_dims_t.size(0); ++i) {
            config.head_hidden_dims.push_back(head_hidden_dims_t[i].item<int64_t>());
        }

        archive.read("head_activation", head_activation_t);
        config.head_activation = static_cast<ActivationType>(head_activation_t.item<int>());

        archive.read("head_dropout", head_dropout_t);
        config.head_dropout = head_dropout_t.item<float>();
    } catch (...) {
        // Architecture config may not be present in older checkpoints - use defaults
        // Defaults match the legacy behavior (GELU + BatchNorm + no residual)
        config.activation = ActivationType::GELU;
        config.normalization = NormLayerType::BatchNorm;
        config.norm_groups = kDefaultNormGroups;
        config.use_residual = false;
        config.leaky_relu_slope = kDefaultLeakyReLUSlope;
        config.elu_alpha = kDefaultELUAlpha;
        config.head_hidden_dims = {};
        config.head_activation = ActivationType::GELU;
        config.head_dropout = 0.0f;
    }

    // Load encoder architecture (with backward compatibility)
    try {
        torch::Tensor encoder_arch_t;
        archive.read("encoder_architecture", encoder_arch_t);
        config.encoder_architecture = static_cast<EncoderArchitecture>(encoder_arch_t.item<int>());
    } catch (...) {
        config.encoder_architecture = EncoderArchitecture::MLP;  // Default for older checkpoints
    }

    // Load TabM configuration (with backward compatibility)
    try {
        torch::Tensor tabm_enabled_t, tabm_n_ensembles_t, tabm_agg_len_t;

        archive.read("tabm_enabled", tabm_enabled_t);
        config.tabm.enabled = tabm_enabled_t.item<int>() != 0;

        archive.read("tabm_n_ensembles", tabm_n_ensembles_t);
        config.tabm.n_ensembles = tabm_n_ensembles_t.item<int>();

        archive.read("tabm_aggregation_len", tabm_agg_len_t);
        int64_t agg_len = tabm_agg_len_t.item<int64_t>();
        if (agg_len > 0) {
            torch::Tensor tabm_agg_t;
            archive.read("tabm_aggregation", tabm_agg_t);
            auto ptr = tabm_agg_t.data_ptr<uint8_t>();
            config.tabm.aggregation = std::string(reinterpret_cast<const char*>(ptr), agg_len);
        }
    } catch (...) {
        // TabM config not present in older checkpoints - disabled by default
        config.tabm = TabMConfig{};
    }

    // RankPool / Transformer encoder fields (back-compat: pre-rank-pool-port
    // checkpoints didn't save these; try_read keeps the ModelConfig defaults
    // for older models, which is fine because hash/embed/sparse models never
    // read these fields).
    torch::Tensor cover_dropout_t;
    if (archive.try_read("cover_dropout", cover_dropout_t)) {
        config.cover_dropout = cover_dropout_t.item<float>();
    }
    torch::Tensor d_model_t;
    if (archive.try_read("d_model", d_model_t)) {
        config.d_model = d_model_t.item<int>();
    }
    torch::Tensor n_heads_t;
    if (archive.try_read("n_heads", n_heads_t)) {
        config.n_heads = n_heads_t.item<int>();
    }
    torch::Tensor n_attn_t;
    if (archive.try_read("n_attention_layers", n_attn_t)) {
        config.n_attention_layers = n_attn_t.item<int>();
    }
    torch::Tensor tx_ff_t;
    if (archive.try_read("transformer_ff_dim", tx_ff_t)) {
        config.transformer_ff_dim = tx_ff_t.item<int>();
    }
    torch::Tensor tx_dropout_t;
    if (archive.try_read("transformer_dropout", tx_dropout_t)) {
        config.transformer_dropout = tx_dropout_t.item<float>();
    }
    torch::Tensor tx_pool_len_t;
    if (archive.try_read("transformer_pooling_len", tx_pool_len_t)) {
        int64_t pool_len = tx_pool_len_t.item<int64_t>();
        if (pool_len > 0) {
            torch::Tensor tx_pool_t;
            if (archive.try_read("transformer_pooling", tx_pool_t)) {
                auto ptr = tx_pool_t.data_ptr<uint8_t>();
                config.transformer_pooling = std::string(
                    reinterpret_cast<const char*>(ptr), pool_len);
            }
        }
    }

    // Architecture-specific sub-configs (issue #37). Each read uses a FRESH
    // tensor (InputArchive::read copies into the passed tensor, so reusing one
    // across differing dtypes/sizes trips a setStorage mismatch). try_read keeps
    // the struct default for any key a pre-#37 checkpoint never wrote.
    auto rd_int = [&](const char* key, int& dst) {
        torch::Tensor t; if (archive.try_read(key, t)) dst = t.item<int>();
    };
    auto rd_flt = [&](const char* key, float& dst) {
        torch::Tensor t; if (archive.try_read(key, t)) dst = t.item<float>();
    };
    auto rd_bool = [&](const char* key, bool& dst) {
        torch::Tensor t; if (archive.try_read(key, t)) dst = t.item<int>() != 0;
    };
    auto rd_enum = [&](const char* key, auto& dst) {
        torch::Tensor t;
        if (archive.try_read(key, t)) {
            dst = static_cast<std::decay_t<decltype(dst)>>(t.item<int>());
        }
    };

    // FT-Transformer
    rd_int("ft_d_model", config.ft_transformer.d_model);
    rd_int("ft_n_heads", config.ft_transformer.n_heads);
    rd_int("ft_n_layers", config.ft_transformer.n_layers);
    rd_flt("ft_attention_dropout", config.ft_transformer.attention_dropout);
    rd_flt("ft_ffn_dropout", config.ft_transformer.ffn_dropout);
    rd_int("ft_ffn_multiplier", config.ft_transformer.ffn_multiplier);
    rd_bool("ft_pre_norm", config.ft_transformer.pre_norm);
    // TabNet
    rd_int("tabnet_n_steps", config.tabnet.n_steps);
    rd_int("tabnet_n_d", config.tabnet.n_d);
    rd_int("tabnet_n_a", config.tabnet.n_a);
    rd_flt("tabnet_relaxation_factor", config.tabnet.relaxation_factor);
    rd_flt("tabnet_sparsity_coefficient", config.tabnet.sparsity_coefficient);
    rd_int("tabnet_virtual_batch_size", config.tabnet.virtual_batch_size);
    rd_bool("tabnet_use_sparsemax", config.tabnet.use_sparsemax);
    // SAINT
    rd_int("saint_d_model", config.saint.d_model);
    rd_int("saint_n_heads", config.saint.n_heads);
    rd_int("saint_n_layers", config.saint.n_layers);
    rd_flt("saint_attention_dropout", config.saint.attention_dropout);
    rd_bool("saint_use_row_attention", config.saint.use_row_attention);
    rd_bool("saint_use_contrastive_pretrain", config.saint.use_contrastive_pretrain);
    rd_flt("saint_mixup_alpha", config.saint.mixup_alpha);
    // GNN
    rd_enum("gnn_type", config.gnn.gnn_type);
    rd_int("gnn_n_layers", config.gnn.n_layers);
    rd_int("gnn_hidden_dim", config.gnn.hidden_dim);
    rd_int("gnn_n_heads", config.gnn.n_heads);
    rd_int("gnn_k_neighbors", config.gnn.k_neighbors);
    rd_enum("gnn_graph_mode", config.gnn.graph_mode);
    rd_flt("gnn_edge_dropout", config.gnn.edge_dropout);
    rd_bool("gnn_use_edge_features", config.gnn.use_edge_features);
    // TraitNet
    rd_int("trait_env_dim", config.trait_net.env_dim);
    rd_int("trait_trait_dim", config.trait_net.trait_dim);
    rd_int("trait_interaction_dim", config.trait_net.interaction_dim);
    rd_enum("trait_interaction", config.trait_net.interaction);
    rd_bool("trait_shared_trait_encoder", config.trait_net.shared_trait_encoder);
    // ExcelFormer
    rd_int("excel_d_model", config.excelformer.d_model);
    rd_int("excel_n_heads", config.excelformer.n_heads);
    rd_int("excel_n_layers", config.excelformer.n_layers);
    rd_flt("excel_attention_dropout", config.excelformer.attention_dropout);
    rd_int("excel_ffn_multiplier", config.excelformer.ffn_multiplier);
    rd_flt("excel_importance_threshold", config.excelformer.importance_threshold);
    rd_bool("excel_pre_norm", config.excelformer.pre_norm);
    // Heterogeneous GNN
    rd_int("hgnn_hidden_dim", config.heterogeneous_gnn.hidden_dim);
    rd_int("hgnn_output_dim", config.heterogeneous_gnn.output_dim);
    rd_int("hgnn_n_layers", config.heterogeneous_gnn.n_layers);
    rd_int("hgnn_n_edge_types", config.heterogeneous_gnn.n_edge_types);
    rd_int("hgnn_n_heads", config.heterogeneous_gnn.n_heads);
    rd_flt("hgnn_dropout", config.heterogeneous_gnn.dropout);
    rd_int("hgnn_k_cooccurrence", config.heterogeneous_gnn.k_cooccurrence);
    rd_flt("hgnn_cooccurrence_threshold", config.heterogeneous_gnn.cooccurrence_threshold);
    rd_bool("hgnn_use_taxonomic_edges", config.heterogeneous_gnn.use_taxonomic_edges);
    rd_bool("hgnn_use_cooccurrence_edges", config.heterogeneous_gnn.use_cooccurrence_edges);
    // Parallel layers
    rd_bool("parallel_enabled", config.parallel_layers.enabled);
    rd_enum("parallel_aggregation", config.parallel_layers.aggregation);
    rd_int("parallel_attention_heads", config.parallel_layers.attention_heads);
    rd_bool("parallel_use_residual", config.parallel_layers.use_residual);
    torch::Tensor n_branches_t;
    if (archive.try_read("parallel_n_branches", n_branches_t)) {
        int64_t n_branches = n_branches_t.item<int64_t>();
        config.parallel_layers.branches.clear();
        for (int64_t i = 0; i < n_branches; ++i) {
            const std::string p = "parallel_branch_" + std::to_string(i) + "_";
            ParallelBranchConfig b;
            torch::Tensor hd_t;
            if (archive.try_read(p + "hidden_dims", hd_t)) {
                b.hidden_dims.resize(hd_t.size(0));
                for (int64_t j = 0; j < hd_t.size(0); ++j) {
                    b.hidden_dims[j] = hd_t[j].item<int64_t>();
                }
            }
            rd_enum((p + "activation").c_str(), b.activation);
            rd_enum((p + "normalization").c_str(), b.normalization);
            rd_flt((p + "dropout").c_str(), b.dropout);
            rd_flt((p + "branch_weight").c_str(), b.branch_weight);
            config.parallel_layers.branches.push_back(std::move(b));
        }
    }

    return config;
}

// Why: libtorch's archive API stores tensors, not strings. We need a
// length-prefixed UInt8 tensor pattern to round-trip a std::string (same
// approach save_run_metadata uses for resolve_version/timestamps).
static void write_string_to_archive(
    torch::serialize::OutputArchive& archive,
    const std::string& prefix,
    const std::string& value
) {
    archive.write(prefix + "_len", torch::tensor(static_cast<int64_t>(value.size())));
    if (!value.empty()) {
        std::vector<uint8_t> bytes(value.begin(), value.end());
        archive.write(prefix, torch::from_blob(
            bytes.data(), {static_cast<int64_t>(bytes.size())}, torch::kUInt8).clone());
    }
}

void save_scalers(
    torch::serialize::OutputArchive& archive,
    const Scalers& scalers
) {
    // Why: load_scalers previously ate exceptions silently. If the
    // "continuous_mean" archive.read failed for any reason, scalers came
    // back with undefined tensors and downstream predict crashed in
    // (continuous - undefined_tensor). Make presence explicit with a
    // boolean flag so the load path has a clean signal.
    int has_continuous = scalers.continuous_mean.defined() ? 1 : 0;
    archive.write("scalers_has_continuous", torch::tensor(has_continuous));
    if (has_continuous) {
        archive.write("continuous_mean", scalers.continuous_mean);
        archive.write("continuous_scale", scalers.continuous_scale);
    }

    // Save target scalers. Why: the previous format wrote only mean/scale
    // and dropped the target name, so load_scalers couldn't rebuild the
    // {name -> (mean, scale)} map and the loaded Predictor produced
    // predictions in scaled (mean=0, std=1) space instead of the original
    // target scale. Write name alongside each entry.
    archive.write("n_target_scalers", torch::tensor(static_cast<int64_t>(scalers.target_scalers.size())));
    int idx = 0;
    for (const auto& [name, scaler] : scalers.target_scalers) {
        std::string prefix = "target_scaler_" + std::to_string(idx) + "_";
        write_string_to_archive(archive, prefix + "name", name);
        archive.write(prefix + "mean", scaler.first);
        archive.write(prefix + "scale", scaler.second);
        idx++;
    }
}

Scalers load_scalers(
    torch::serialize::InputArchive& archive
) {
    Scalers scalers;

    // Continuous scalers. Prefer the explicit presence flag from the new
    // save format; fall back to try_read on bare keys for older checkpoints.
    torch::Tensor has_t;
    bool has_continuous = false;
    if (archive.try_read("scalers_has_continuous", has_t)) {
        has_continuous = (has_t.item<int>() != 0);
    } else {
        // Legacy checkpoint: just attempt the bare reads.
        has_continuous = true;
    }
    if (has_continuous) {
        // try_read avoids the silent-catch-everything anti-pattern below.
        archive.try_read("continuous_mean", scalers.continuous_mean);
        archive.try_read("continuous_scale", scalers.continuous_scale);
    }

    auto read_string_pair = [&](const std::string& prefix) -> std::string {
        torch::Tensor len_t;
        if (!archive.try_read(prefix + "_len", len_t)) return std::string();
        int64_t len = len_t.item<int64_t>();
        if (len <= 0) return std::string();
        torch::Tensor t;
        if (!archive.try_read(prefix, t)) return std::string();
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), len);
    };

    // Target scalers: restore the {name -> (mean, scale)} map. Try the
    // new naming first; fall back to legacy keys so older checkpoints
    // still partially load (without names, target_scalers stays empty
    // and predictions come back in scaled space — better than crashing).
    torch::Tensor n_target_scalers_t;
    if (archive.try_read("n_target_scalers", n_target_scalers_t)) {
        int64_t n_scalers = n_target_scalers_t.item<int64_t>();
        for (int64_t i = 0; i < n_scalers; ++i) {
            std::string idx_s = std::to_string(i);
            std::string prefix = "target_scaler_" + idx_s + "_";
            std::string name = read_string_pair(prefix + "name");

            torch::Tensor mean, scale;
            if (!archive.try_read(prefix + "mean", mean)) {
                archive.try_read("target_scaler_mean_" + idx_s, mean);
            }
            if (!archive.try_read(prefix + "scale", scale)) {
                archive.try_read("target_scaler_scale_" + idx_s, scale);
            }
            if (!name.empty() && mean.defined() && scale.defined()) {
                scalers.target_scalers[name] = {mean, scale};
            }
        }
    }

    return scalers;
}

void save_schema(
    torch::serialize::OutputArchive& archive,
    const ResolveSchema& schema
) {
    archive.write("schema_n_plots", torch::tensor(schema.n_plots));
    archive.write("schema_n_species", torch::tensor(schema.n_species));
    archive.write("schema_n_species_vocab", torch::tensor(schema.n_species_vocab));
    archive.write("schema_has_coordinates", torch::tensor(static_cast<int>(schema.has_coordinates)));
    archive.write("schema_has_abundance", torch::tensor(static_cast<int>(schema.has_abundance)));
    archive.write("schema_has_taxonomy", torch::tensor(static_cast<int>(schema.has_taxonomy)));
    archive.write("schema_n_genera", torch::tensor(schema.n_genera));
    archive.write("schema_n_families", torch::tensor(schema.n_families));
    archive.write("schema_n_genera_vocab", torch::tensor(schema.n_genera_vocab));
    archive.write("schema_n_families_vocab", torch::tensor(schema.n_families_vocab));
    archive.write("schema_track_unknown_fraction", torch::tensor(static_cast<int>(schema.track_unknown_fraction)));
    archive.write("schema_track_unknown_count", torch::tensor(static_cast<int>(schema.track_unknown_count)));
    archive.write("schema_n_covariates", torch::tensor(static_cast<int64_t>(schema.covariate_names.size())));
    for (size_t i = 0; i < schema.covariate_names.size(); ++i) {
        write_string_to_archive(archive, "schema_covariate_" + std::to_string(i),
                                schema.covariate_names[i]);
    }
    archive.write("schema_n_targets", torch::tensor(static_cast<int64_t>(schema.targets.size())));
    for (size_t i = 0; i < schema.targets.size(); ++i) {
        const auto& target = schema.targets[i];
        std::string prefix = "schema_target_" + std::to_string(i) + "_";
        write_string_to_archive(archive, prefix + "name", target.name);
        archive.write(prefix + "task", torch::tensor(static_cast<int>(target.task)));
        archive.write(prefix + "transform", torch::tensor(static_cast<int>(target.transform)));
        archive.write(prefix + "num_classes", torch::tensor(target.num_classes));
        archive.write(prefix + "weight", torch::tensor(target.weight));

        // Ordered class vocabulary for classification targets. Empty (count
        // 0) for regression. Empty for already-integer-encoded
        // classification targets that the loader didn't auto-factorize.
        // Per-class strings are serialized via the same length-prefix +
        // UInt8 bytes scheme used elsewhere. Back-compat: pre-classification
        // checkpoints won't have this key; the load path treats absent ==
        // empty (see schema load below).
        archive.write(prefix + "n_class_names",
                      torch::tensor(static_cast<int64_t>(target.class_names.size())));
        for (size_t j = 0; j < target.class_names.size(); ++j) {
            write_string_to_archive(archive,
                prefix + "class_" + std::to_string(j),
                target.class_names[j]);
        }
    }

    // Categorical covariates: column count + per-column name + per-column
    // vocab size + shared embed_dim. Vocab sizes include the reserved UNK
    // slot at code 0 (so the column's embedding table is size K+1).
    archive.write("schema_n_categoricals",
                  torch::tensor(static_cast<int64_t>(schema.categorical_names.size())));
    archive.write("schema_categorical_embed_dim",
                  torch::tensor(schema.categorical_embed_dim));
    for (size_t i = 0; i < schema.categorical_names.size(); ++i) {
        const std::string prefix = "schema_categorical_" + std::to_string(i) + "_";
        write_string_to_archive(archive, prefix + "name", schema.categorical_names[i]);
        archive.write(prefix + "vocab_size",
                      torch::tensor(schema.categorical_vocab_sizes[i]));
    }

    // Rank-pool / transformer pooling scheme + resolved species cap (issue #38),
    // so the predict side rebuilds the same DatasetConfig instead of defaulting
    // to Log1p.
    archive.write("schema_pool_weighting", torch::tensor(schema.pool_weighting));
    archive.write("schema_pool_species_cap", torch::tensor(schema.pool_species_cap));
}

ResolveSchema load_schema(
    torch::serialize::InputArchive& archive
) {
    // Why: each archive.read(key, t) reuses the destination tensor's
    // storage rather than allocating fresh. Reading heterogeneous dtypes
    // (int64 / int32 / float32) into the same tensor then triggers a
    // storage-size mismatch at libtorch's set_storage_offset (storage of
    // size 4 used to satisfy itemsize 8, or vice versa).
    // How to apply: every read uses a fresh local tensor.
    auto read_i64 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<int64_t>();
    };
    auto read_i32 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<int>();
    };
    auto read_bool = [&](const std::string& key) -> bool {
        return read_i32(key) != 0;
    };
    auto read_f32 = [&](const std::string& key) {
        torch::Tensor t;
        archive.read(key, t);
        return t.item<float>();
    };
    auto read_string = [&](const std::string& prefix) -> std::string {
        int64_t len = read_i64(prefix + "_len");
        if (len <= 0) return std::string();
        torch::Tensor t;
        archive.read(prefix, t);
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), len);
    };

    ResolveSchema schema;
    schema.n_plots = read_i64("schema_n_plots");
    schema.n_species = read_i64("schema_n_species");
    schema.n_species_vocab = read_i64("schema_n_species_vocab");
    schema.has_coordinates = read_bool("schema_has_coordinates");
    schema.has_abundance = read_bool("schema_has_abundance");
    schema.has_taxonomy = read_bool("schema_has_taxonomy");
    schema.n_genera = read_i64("schema_n_genera");
    schema.n_families = read_i64("schema_n_families");
    schema.n_genera_vocab = read_i64("schema_n_genera_vocab");
    schema.n_families_vocab = read_i64("schema_n_families_vocab");
    schema.track_unknown_fraction = read_bool("schema_track_unknown_fraction");
    schema.track_unknown_count = read_bool("schema_track_unknown_count");
    int64_t n_covariates = read_i64("schema_n_covariates");
    schema.covariate_names.resize(n_covariates);
    for (int64_t i = 0; i < n_covariates; ++i) {
        // Back-compat: older checkpoints didn't save covariate names.
        // try_read returns false silently when the key is absent, leaving
        // the existing empty string in place. Names aren't load-bearing
        // for model construction (model indexes by count, not name), so
        // empty-string fallback is safe.
        torch::Tensor len_t;
        if (archive.try_read("schema_covariate_" + std::to_string(i) + "_len", len_t)) {
            int64_t len = len_t.item<int64_t>();
            if (len > 0) {
                torch::Tensor name_t;
                archive.read("schema_covariate_" + std::to_string(i), name_t);
                auto ptr = name_t.data_ptr<uint8_t>();
                schema.covariate_names[i] = std::string(reinterpret_cast<const char*>(ptr), len);
            }
        }
    }
    int64_t n_targets = read_i64("schema_n_targets");
    schema.targets.resize(n_targets);
    for (int64_t i = 0; i < n_targets; ++i) {
        std::string prefix = "schema_target_" + std::to_string(i) + "_";
        // Back-compat: older checkpoints didn't save target names. Missing
        // names would collide on register_module("head_") for all targets,
        // so synthesize a fallback name when absent.
        torch::Tensor name_len_t;
        if (archive.try_read(prefix + "name_len", name_len_t)) {
            schema.targets[i].name = read_string(prefix + "name");
        }
        if (schema.targets[i].name.empty()) {
            schema.targets[i].name = "target_" + std::to_string(i);
        }
        schema.targets[i].task = static_cast<TaskType>(read_i32(prefix + "task"));
        schema.targets[i].transform = static_cast<TransformType>(read_i32(prefix + "transform"));
        schema.targets[i].num_classes = read_i32(prefix + "num_classes");
        schema.targets[i].weight = read_f32(prefix + "weight");

        // Class names (back-compat: pre-classification checkpoints omit
        // these keys; treat absent as no class vocab, which matches the
        // original behaviour where classification just used raw int codes).
        torch::Tensor n_cn_t;
        if (archive.try_read(prefix + "n_class_names", n_cn_t)) {
            int64_t n_cn = n_cn_t.item<int64_t>();
            schema.targets[i].class_names.resize(static_cast<size_t>(n_cn));
            for (int64_t j = 0; j < n_cn; ++j) {
                schema.targets[i].class_names[j] =
                    read_string(prefix + "class_" + std::to_string(j));
            }
        }
    }

    // Categorical covariates (back-compat: pre-categorical-port checkpoints
    // won't have any of these keys; treat as schema with zero categoricals).
    torch::Tensor n_cat_t;
    if (archive.try_read("schema_n_categoricals", n_cat_t)) {
        int64_t n_cat = n_cat_t.item<int64_t>();
        torch::Tensor embed_dim_t;
        if (archive.try_read("schema_categorical_embed_dim", embed_dim_t)) {
            schema.categorical_embed_dim = embed_dim_t.item<int64_t>();
        }
        schema.categorical_names.resize(n_cat);
        schema.categorical_vocab_sizes.resize(n_cat);
        for (int64_t i = 0; i < n_cat; ++i) {
            const std::string prefix = "schema_categorical_" + std::to_string(i) + "_";
            schema.categorical_names[i] = read_string(prefix + "name");
            schema.categorical_vocab_sizes[i] = read_i64(prefix + "vocab_size");
        }
    }

    // Pool weighting scheme + species cap (back-compat: pre-issue-#38
    // checkpoints keep the schema defaults, Log1p / auto).
    torch::Tensor pool_w_t, pool_cap_t;
    if (archive.try_read("schema_pool_weighting", pool_w_t)) {
        schema.pool_weighting = pool_w_t.item<int>();
    }
    if (archive.try_read("schema_pool_species_cap", pool_cap_t)) {
        schema.pool_species_cap = pool_cap_t.item<int>();
    }
    return schema;
}

void save_train_config(
    torch::serialize::OutputArchive& archive,
    const TrainConfig& config,
    int requested_batch_size
) {
    // batch_size semantics in the checkpoint:
    // ---------------------------------------
    // Trainer::fit mutates `config_.batch_size` in place when the CUDA
    // auto-halve-on-OOM loop fires, so `config.batch_size` at save time is the
    // EFFECTIVE batch size that actually trained the model. `train_batch_size`
    // records the value the caller REQUESTED (passed here, restored by
    // load_train_config), and `train_effective_batch_size` records the effective
    // value, so a fallback run is detectable when the two diverge. On a clean run
    // they are equal. A -1 requested value means "no separate request known".
    const int requested = requested_batch_size >= 0 ? requested_batch_size : config.batch_size;
    archive.write("train_batch_size", torch::tensor(requested));
    archive.write("train_effective_batch_size", torch::tensor(config.batch_size));
    archive.write("train_batch_size_floor", torch::tensor(config.batch_size_floor));

    archive.write("train_max_epochs", torch::tensor(config.max_epochs));
    archive.write("train_patience", torch::tensor(config.patience));
    archive.write("train_lr", torch::tensor(config.lr));
    archive.write("train_weight_decay", torch::tensor(config.weight_decay));
    archive.write("train_phase_boundary_1", torch::tensor(config.phase_boundaries.first));
    archive.write("train_phase_boundary_2", torch::tensor(config.phase_boundaries.second));
    archive.write("train_loss_config", torch::tensor(static_cast<int>(config.loss_config)));
    archive.write("train_lr_scheduler", torch::tensor(static_cast<int>(config.lr_scheduler)));
    archive.write("train_lr_step_size", torch::tensor(config.lr_step_size));
    archive.write("train_lr_gamma", torch::tensor(config.lr_gamma));
    archive.write("train_lr_min", torch::tensor(config.lr_min));

    // Save VRAM cap used during training (informational; the cap is applied
    // at Trainer::fit / Predictor::load time, not reloaded from the
    // checkpoint, but recording it lets `resolve info` show what the model
    // was trained under).
    archive.write("train_vram_fraction", torch::tensor(config.vram_fraction));

    // Save band thresholds
    std::vector<float> thresholds(config.band_thresholds);
    archive.write("train_band_thresholds", torch::tensor(thresholds));
}

void save_run_metadata(
    torch::serialize::OutputArchive& archive,
    const RunMetadata& metadata
) {
    // Save version as bytes
    std::vector<uint8_t> version_bytes(metadata.resolve_version.begin(), metadata.resolve_version.end());
    archive.write("meta_version_len", torch::tensor(static_cast<int64_t>(version_bytes.size())));
    if (!version_bytes.empty()) {
        archive.write("meta_version", torch::from_blob(
            version_bytes.data(), {static_cast<int64_t>(version_bytes.size())}, torch::kUInt8).clone());
    }

    // Save timestamps as bytes
    std::vector<uint8_t> created_bytes(metadata.created_at.begin(), metadata.created_at.end());
    archive.write("meta_created_len", torch::tensor(static_cast<int64_t>(created_bytes.size())));
    if (!created_bytes.empty()) {
        archive.write("meta_created", torch::from_blob(
            created_bytes.data(), {static_cast<int64_t>(created_bytes.size())}, torch::kUInt8).clone());
    }

    std::vector<uint8_t> completed_bytes(metadata.completed_at.begin(), metadata.completed_at.end());
    archive.write("meta_completed_len", torch::tensor(static_cast<int64_t>(completed_bytes.size())));
    if (!completed_bytes.empty()) {
        archive.write("meta_completed", torch::from_blob(
            completed_bytes.data(), {static_cast<int64_t>(completed_bytes.size())}, torch::kUInt8).clone());
    }

    // Save numeric fields
    archive.write("meta_train_time", torch::tensor(metadata.train_time_seconds));
    archive.write("meta_n_plots_train", torch::tensor(metadata.n_plots_train));
    archive.write("meta_n_plots_test", torch::tensor(metadata.n_plots_test));
    archive.write("meta_best_epoch", torch::tensor(metadata.best_epoch));
    archive.write("meta_total_epochs", torch::tensor(metadata.total_epochs));

    // Save final metrics as flattened tensors
    int64_t n_targets = static_cast<int64_t>(metadata.final_metrics.size());
    archive.write("meta_n_targets", torch::tensor(n_targets));

    int target_idx = 0;
    for (const auto& [target_name, metrics] : metadata.final_metrics) {
        std::string prefix = "meta_target_" + std::to_string(target_idx) + "_";

        // Save target name
        std::vector<uint8_t> name_bytes(target_name.begin(), target_name.end());
        archive.write(prefix + "name_len", torch::tensor(static_cast<int64_t>(name_bytes.size())));
        if (!name_bytes.empty()) {
            archive.write(prefix + "name", torch::from_blob(
                name_bytes.data(), {static_cast<int64_t>(name_bytes.size())}, torch::kUInt8).clone());
        }

        // Save metrics for this target
        int64_t n_metrics = static_cast<int64_t>(metrics.size());
        archive.write(prefix + "n_metrics", torch::tensor(n_metrics));

        int metric_idx = 0;
        for (const auto& [metric_name, value] : metrics) {
            std::string m_prefix = prefix + "metric_" + std::to_string(metric_idx) + "_";

            std::vector<uint8_t> m_name_bytes(metric_name.begin(), metric_name.end());
            archive.write(m_prefix + "name_len", torch::tensor(static_cast<int64_t>(m_name_bytes.size())));
            if (!m_name_bytes.empty()) {
                archive.write(m_prefix + "name", torch::from_blob(
                    m_name_bytes.data(), {static_cast<int64_t>(m_name_bytes.size())}, torch::kUInt8).clone());
            }
            archive.write(m_prefix + "value", torch::tensor(value));
            metric_idx++;
        }
        target_idx++;
    }
}

TrainConfig load_train_config(
    torch::serialize::InputArchive& archive
) {
    // Inverse of save_train_config. Recovers the persisted training
    // hyperparameters; fields not written by save_train_config (device,
    // checkpoint_dir, AMP/cuDNN flags, log callback, ...) keep their
    // TrainConfig defaults — the caller sets those for the run. Every read is
    // try_read so a checkpoint missing a key falls back to the default instead
    // of throwing (forward/backward compatibility across schema additions).
    TrainConfig config;
    // Each read uses a fresh tensor: InputArchive::read copies into the passed
    // tensor, so reusing one across reads of different dtype/size trips a
    // setStorage size-mismatch (e.g. int32 storage then an int64 read).
    auto rd_int = [&](const char* key, int& dst) {
        torch::Tensor x;
        if (archive.try_read(key, x)) dst = x.item<int>();
    };
    auto rd_float = [&](const char* key, float& dst) {
        torch::Tensor x;
        if (archive.try_read(key, x)) dst = x.item<float>();
    };
    auto rd_enum_int = [&](const char* key) -> std::optional<int> {
        torch::Tensor x;
        if (archive.try_read(key, x)) return x.item<int>();
        return std::nullopt;
    };

    rd_int("train_batch_size", config.batch_size);
    rd_int("train_batch_size_floor", config.batch_size_floor);
    rd_int("train_max_epochs", config.max_epochs);
    rd_int("train_patience", config.patience);
    rd_float("train_lr", config.lr);
    rd_float("train_weight_decay", config.weight_decay);

    torch::Tensor pb1_t, pb2_t;
    const bool has_pb1 = archive.try_read("train_phase_boundary_1", pb1_t);
    const bool has_pb2 = archive.try_read("train_phase_boundary_2", pb2_t);
    if (has_pb1 && has_pb2) {
        config.phase_boundaries = {pb1_t.item<int>(), pb2_t.item<int>()};
    }

    if (auto lc = rd_enum_int("train_loss_config"))
        config.loss_config = static_cast<LossConfigMode>(*lc);
    if (auto ls = rd_enum_int("train_lr_scheduler"))
        config.lr_scheduler = static_cast<LRSchedulerType>(*ls);
    rd_int("train_lr_step_size", config.lr_step_size);
    rd_float("train_lr_gamma", config.lr_gamma);
    rd_float("train_lr_min", config.lr_min);
    rd_float("train_vram_fraction", config.vram_fraction);

    torch::Tensor bt_t;
    if (archive.try_read("train_band_thresholds", bt_t)) {
        std::vector<float> bt(static_cast<size_t>(bt_t.size(0)));
        for (int64_t i = 0; i < bt_t.size(0); ++i) bt[static_cast<size_t>(i)] = bt_t[i].item<float>();
        config.band_thresholds = std::move(bt);
    }
    return config;
}

RunMetadata load_run_metadata(
    torch::serialize::InputArchive& archive
) {
    // Inverse of save_run_metadata. Decodes the byte-stored strings and the
    // flattened per-target metric tree. Missing keys fall back to defaults.
    RunMetadata meta;

    auto read_string_pair = [&](const std::string& prefix) -> std::string {
        torch::Tensor len_t;
        if (!archive.try_read(prefix + "_len", len_t)) return std::string();
        int64_t len = len_t.item<int64_t>();
        if (len <= 0) return std::string();
        torch::Tensor t;
        if (!archive.try_read(prefix, t)) return std::string();
        auto ptr = t.data_ptr<uint8_t>();
        return std::string(reinterpret_cast<const char*>(ptr), static_cast<size_t>(len));
    };

    std::string version = read_string_pair("meta_version");
    if (!version.empty()) meta.resolve_version = version;
    meta.created_at = read_string_pair("meta_created");
    meta.completed_at = read_string_pair("meta_completed");

    // Fresh tensor per read (see load_train_config note on InputArchive::read).
    { torch::Tensor x; if (archive.try_read("meta_train_time", x)) meta.train_time_seconds = x.item<float>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_train", x)) meta.n_plots_train = x.item<int64_t>(); }
    { torch::Tensor x; if (archive.try_read("meta_n_plots_test", x)) meta.n_plots_test = x.item<int64_t>(); }
    { torch::Tensor x; if (archive.try_read("meta_best_epoch", x)) meta.best_epoch = x.item<int>(); }
    { torch::Tensor x; if (archive.try_read("meta_total_epochs", x)) meta.total_epochs = x.item<int>(); }

    torch::Tensor n_targets_t;
    if (archive.try_read("meta_n_targets", n_targets_t)) {
        int64_t n_targets = n_targets_t.item<int64_t>();
        for (int64_t ti = 0; ti < n_targets; ++ti) {
            std::string prefix = "meta_target_" + std::to_string(ti) + "_";
            std::string target_name = read_string_pair(prefix + "name");
            torch::Tensor n_metrics_t;
            if (!archive.try_read(prefix + "n_metrics", n_metrics_t)) continue;
            int64_t n_metrics = n_metrics_t.item<int64_t>();
            std::unordered_map<std::string, float> metrics;
            for (int64_t mi = 0; mi < n_metrics; ++mi) {
                std::string m_prefix = prefix + "metric_" + std::to_string(mi) + "_";
                std::string metric_name = read_string_pair(m_prefix + "name");
                torch::Tensor val_t;
                if (archive.try_read(m_prefix + "value", val_t))
                    metrics[metric_name] = val_t.item<float>();
            }
            meta.final_metrics[target_name] = std::move(metrics);
        }
    }
    return meta;
}

void write_metadata_json(
    const std::string& checkpoint_path,
    const ModelConfig& model_config,
    const TrainConfig& train_config,
    const RunMetadata& metadata,
    const ResolveSchema& schema,
    int requested_batch_size
) {
    // Replace .pt extension with .json
    std::string json_path = checkpoint_path;
    if (json_path.size() >= 3 && json_path.substr(json_path.size() - 3) == ".pt") {
        json_path = json_path.substr(0, json_path.size() - 3) + ".json";
    } else {
        json_path += ".json";
    }

    std::ofstream file(json_path);
    if (!file.is_open()) return;

    file << "{\n";

    // Run metadata
    file << "  \"resolve_version\": \"" << json_escape(metadata.resolve_version) << "\",\n";
    file << "  \"created_at\": \"" << json_escape(metadata.created_at) << "\",\n";
    file << "  \"completed_at\": \"" << json_escape(metadata.completed_at) << "\",\n";
    file << "  \"train_time_seconds\": " << metadata.train_time_seconds << ",\n";
    file << "  \"n_plots_train\": " << metadata.n_plots_train << ",\n";
    file << "  \"n_plots_test\": " << metadata.n_plots_test << ",\n";
    file << "  \"best_epoch\": " << metadata.best_epoch << ",\n";
    file << "  \"total_epochs\": " << metadata.total_epochs << ",\n";

    // Model config
    file << "  \"model_config\": {\n";
    file << "    \"species_encoding\": " << static_cast<int>(model_config.species_encoding) << ",\n";
    file << "    \"hash_dim\": " << model_config.hash_dim << ",\n";
    file << "    \"species_embed_dim\": " << model_config.species_embed_dim << ",\n";
    file << "    \"genus_emb_dim\": " << model_config.genus_emb_dim << ",\n";
    file << "    \"family_emb_dim\": " << model_config.family_emb_dim << ",\n";
    file << "    \"top_k\": " << model_config.top_k << ",\n";
    file << "    \"dropout\": " << model_config.dropout << ",\n";
    file << "    \"hidden_dims\": [";
    for (size_t i = 0; i < model_config.hidden_dims.size(); ++i) {
        file << model_config.hidden_dims[i];
        if (i < model_config.hidden_dims.size() - 1) file << ", ";
    }
    file << "]\n";
    file << "  },\n";

    // Train config. `batch_size` is the value the caller REQUESTED; the CUDA OOM
    // auto-halve retry may have shrunk the value that actually trained the model,
    // reported under `effective_batch_size`. `batch_size_floor` is the retry lower
    // bound. When the two batch sizes diverge, the run fell back on OOM (issue #86).
    const int req_bs = requested_batch_size >= 0 ? requested_batch_size : train_config.batch_size;
    file << "  \"train_config\": {\n";
    file << "    \"batch_size\": " << req_bs << ",\n";
    file << "    \"effective_batch_size\": " << train_config.batch_size << ",\n";
    file << "    \"batch_size_floor\": " << train_config.batch_size_floor << ",\n";
    file << "    \"max_epochs\": " << train_config.max_epochs << ",\n";
    file << "    \"patience\": " << train_config.patience << ",\n";
    file << "    \"lr\": " << train_config.lr << ",\n";
    file << "    \"weight_decay\": " << train_config.weight_decay << ",\n";
    file << "    \"loss_config\": " << static_cast<int>(train_config.loss_config) << ",\n";
    file << "    \"lr_scheduler\": " << static_cast<int>(train_config.lr_scheduler) << "\n";
    file << "  },\n";

    // Schema
    file << "  \"schema\": {\n";
    file << "    \"n_plots\": " << schema.n_plots << ",\n";
    file << "    \"n_species\": " << schema.n_species << ",\n";
    file << "    \"has_coordinates\": " << (schema.has_coordinates ? "true" : "false") << ",\n";
    file << "    \"has_taxonomy\": " << (schema.has_taxonomy ? "true" : "false") << ",\n";
    file << "    \"n_genera\": " << schema.n_genera << ",\n";
    file << "    \"n_families\": " << schema.n_families << ",\n";
    file << "    \"n_covariates\": " << schema.covariate_names.size() << ",\n";
    file << "    \"n_targets\": " << schema.targets.size() << "\n";
    file << "  },\n";

    // Final metrics
    file << "  \"final_metrics\": {\n";
    bool first_target = true;
    for (const auto& [target_name, metrics] : metadata.final_metrics) {
        if (!first_target) file << ",\n";
        first_target = false;
        file << "    \"" << json_escape(target_name) << "\": {\n";
        bool first_metric = true;
        for (const auto& [metric_name, value] : metrics) {
            if (!first_metric) file << ",\n";
            first_metric = false;
            file << "      \"" << json_escape(metric_name) << "\": " << value;
        }
        file << "\n    }";
    }
    file << "\n  }\n";

    file << "}\n";
}

} // namespace resolve
