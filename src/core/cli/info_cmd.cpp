// RESOLVE CLI - Info command implementation
//
// Prints everything that defines a checkpoint's architecture, so `info` alone
// answers "what model is this?" for every encoder family, not just the MLP.

#include <cstddef>
#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "resolve/resolve.hpp"
#include "resolve/config_registry.hpp"

#include "arg_parser.hpp"

using resolve_cli::ParsedArgs;

namespace {

void print_name_list(const char* label, const std::vector<std::string>& names) {
    if (names.empty()) return;
    std::cout << label;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << names[i];
    }
    std::cout << std::endl;
}

void print_dims(const char* label, const std::vector<int64_t>& dims) {
    std::cout << label << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << dims[i];
    }
    std::cout << "]" << std::endl;
}

// Prints one field-registry row as "<indent><label> <field>: <value>". A whole
// sub-config block therefore follows the struct: a hyperparameter added to it
// appears in `resolve info` in the same edit that adds it, instead of waiting
// for someone to notice this file. Enum values print as the names the CLI
// accepts on its flags, from the shared tables in enum_names.hpp.
struct ConsoleFieldWriter {
    std::string label;
    std::string indent;
    // When true, a row whose registry key is empty is skipped. Those fields are
    // not in the archive, so they read back as struct defaults, and printing
    // them would report a recipe the run never used.
    bool persisted_only = false;

    std::ostream& open_line(const char* name) const {
        std::cout << indent << label << " " << name << ": ";
        return std::cout;
    }

    template <typename Seq>
    void print_values(const Seq& values) const {
        std::cout << "[";
        bool first = true;
        for (const auto& v : values) {
            if (!first) std::cout << ", ";
            first = false;
            std::cout << v;
        }
        std::cout << "]" << std::endl;
    }

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        if (persisted_only && !resolve::has_checkpoint_key(key)) return;
        if constexpr (std::is_same_v<T, resolve::LogCallback>) {
            (void)name;  // a callback has nothing to report
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            open_line(name) << (value.is_cuda() ? "cuda" : "cpu") << std::endl;
        } else if constexpr (resolve::is_registered_config_v<T>) {
            std::cout << indent << label << " " << name << ":" << std::endl;
            resolve::for_each_field(
                value, ConsoleFieldWriter{label, indent + "  ", persisted_only});
        } else if constexpr (std::is_same_v<T, std::vector<resolve::ParallelBranchConfig>>) {
            open_line(name) << value.size() << std::endl;
            for (std::size_t i = 0; i < value.size(); ++i) {
                std::cout << indent << "  " << label << " branch " << i << ":" << std::endl;
                resolve::for_each_field(
                    value[i], ConsoleFieldWriter{label, indent + "    ", persisted_only});
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            open_line(name) << (value ? "yes" : "no") << std::endl;
        } else if constexpr (std::is_enum_v<T>) {
            open_line(name) << resolve::enum_to_name(value) << std::endl;
        } else if constexpr (std::is_same_v<T, std::string>) {
            open_line(name) << value << std::endl;
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
            open_line(name) << value << std::endl;
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            open_line(name);
            print_values(value);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            open_line(name) << value.first << ", " << value.second << std::endl;
        } else {
            static_assert(resolve::registry_detail::always_false<T>,
                          "config field type has no console representation; "
                          "add a branch to ConsoleFieldWriter");
        }
    }
};

// Print every field of one config struct under a common label.
template <typename Cfg>
void print_config_block(const char* label, const Cfg& config, bool persisted_only = false) {
    resolve::for_each_field(config, ConsoleFieldWriter{label, "  ", persisted_only});
}

// Architecture-specific hyperparameters. Only the block that actually shaped
// this checkpoint is printed: the other sub-configs are carried at their
// defaults and reporting them would read as configuration the model uses.
void print_architecture_config(const resolve::ModelConfig& config) {
    using namespace resolve;

    switch (config.encoder_architecture) {
        case EncoderArchitecture::FTTransformer:
            print_config_block("FT-Transformer", config.ft_transformer);
            break;
        case EncoderArchitecture::TabNet:
            print_config_block("TabNet", config.tabnet);
            break;
        case EncoderArchitecture::SAINT:
            print_config_block("SAINT", config.saint);
            break;
        case EncoderArchitecture::GNN:
            print_config_block("GNN", config.gnn);
            break;
        case EncoderArchitecture::TraitNet:
            print_config_block("TraitNet", config.trait_net);
            break;
        case EncoderArchitecture::ExcelFormer:
            print_config_block("ExcelFormer", config.excelformer);
            break;
        case EncoderArchitecture::HeterogeneousGNN:
            print_config_block("HeteroGNN", config.heterogeneous_gnn);
            break;
        case EncoderArchitecture::MLP:
        default:
            std::cout << "  Activation: " << activation_type_to_string(config.activation)
                      << std::endl;
            std::cout << "  Normalization: "
                      << norm_layer_type_to_string(config.normalization) << std::endl;
            std::cout << "  Residual connections: "
                      << (config.use_residual ? "yes" : "no") << std::endl;
            break;
    }

    if (config.moe_routing != MoERoutingType::None) {
        std::cout << "  MoE routing: " << moe_routing_type_to_string(config.moe_routing)
                  << std::endl;
        std::cout << "  MoE placement: " << moe_placement_to_string(config.moe_placement)
                  << std::endl;
        std::cout << "  MoE experts: " << config.n_experts << std::endl;
        print_dims("  MoE expert hidden dims: ", config.expert_hidden_dims);
        std::cout << "  MoE top-k: " << config.moe_top_k << std::endl;
        std::cout << "  MoE noise std: " << config.moe_noise_std << std::endl;
        std::cout << "  MoE aux loss weight: " << config.moe_aux_loss_weight << std::endl;
    }

    if (config.tabm.enabled) {
        print_config_block("TabM", config.tabm);
    }

    if (config.parallel_layers.enabled) {
        print_config_block("Parallel", config.parallel_layers);
    }

    if (!config.head_hidden_dims.empty()) {
        print_dims("  Head hidden dims: ", config.head_hidden_dims);
        std::cout << "  Head activation: "
                  << activation_type_to_string(config.head_activation) << std::endl;
        std::cout << "  Head dropout: " << config.head_dropout << std::endl;
    }
}

}  // namespace

int info_command(const ParsedArgs& args) {
    using namespace resolve;

    const std::string model_path = args.get("--model");
    if (model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        return 1;
    }

    std::cout << "RESOLVE Model Information" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    try {
        // Load model. Trainer::load returns (model, scalers, vocab); `info`
        // only reports schema/config/parameter counts, so the vocab is unused.
        auto [model, scalers, vocab] = Trainer::load(model_path, torch::kCPU);
        (void)vocab;
        const auto& schema = model->schema();
        const auto& config = model->config();

        // Print schema information
        std::cout << "\nSchema:" << std::endl;
        std::cout << "  Plots: " << schema.n_plots << std::endl;
        std::cout << "  Species: " << schema.n_species << std::endl;
        std::cout << "  Species vocab: " << schema.n_species_vocab << std::endl;
        std::cout << "  Has coordinates: " << (schema.has_coordinates ? "yes" : "no") << std::endl;
        std::cout << "  Has abundance: " << (schema.has_abundance ? "yes" : "no") << std::endl;
        std::cout << "  Has taxonomy: " << (schema.has_taxonomy ? "yes" : "no") << std::endl;

        if (schema.has_taxonomy) {
            std::cout << "  Genera: " << schema.n_genera << " (vocab: " << schema.n_genera_vocab << ")" << std::endl;
            std::cout << "  Families: " << schema.n_families << " (vocab: " << schema.n_families_vocab << ")" << std::endl;
        }

        print_name_list("  Covariates: ", schema.covariate_names);
        print_name_list("  Categoricals: ", schema.categorical_names);
        if (schema.has_categoricals()) {
            std::cout << "  Categorical vocab sizes: ";
            for (size_t i = 0; i < schema.categorical_vocab_sizes.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << schema.categorical_vocab_sizes[i];
            }
            std::cout << std::endl;
            std::cout << "  Categorical embed dim: " << schema.categorical_embed_dim
                      << std::endl;
        }

        // Print targets
        std::cout << "\nTargets:" << std::endl;
        for (const auto& target : schema.targets) {
            std::cout << "  " << target.name << ": ";
            if (target.task == TaskType::Classification) {
                std::cout << "classification (" << target.num_classes << " classes)";
            } else {
                std::cout << "regression";
                if (target.transform == TransformType::Log1p) {
                    std::cout << " [log1p]";
                }
            }
            std::cout << std::endl;
        }

        // Print model configuration
        std::cout << "\nModel Configuration:" << std::endl;
        std::cout << "  Species encoding: "
                  << species_encoding_to_string(config.species_encoding) << std::endl;
        std::cout << "  Encoder architecture: "
                  << encoder_architecture_to_string(config.encoder_architecture)
                  << std::endl;
        std::cout << "  Hash dim: " << config.hash_dim << std::endl;
        std::cout << "  Species embed dim: " << config.species_embed_dim << std::endl;
        std::cout << "  Genus embed dim: " << config.genus_emb_dim << std::endl;
        std::cout << "  Family embed dim: " << config.family_emb_dim << std::endl;
        std::cout << "  Top-k: " << config.top_k << std::endl;
        std::cout << "  Top-k species: " << config.top_k_species << std::endl;
        std::cout << "  Taxonomy slots: " << config.n_taxonomy_slots << std::endl;
        std::cout << "  Dropout: " << config.dropout << std::endl;
        print_dims("  Hidden dims: ", config.hidden_dims);

        // Encoder-specific hyperparameters. cover_dropout is shared by the
        // rank-pool and transformer encoders; the transformer block defines the
        // rest of a transformer checkpoint's architecture.
        if (config.species_encoding == SpeciesEncodingMode::RankPool ||
            config.species_encoding == SpeciesEncodingMode::Transformer) {
            std::cout << "  Cover dropout: " << config.cover_dropout << std::endl;
        }
        if (config.species_encoding == SpeciesEncodingMode::Transformer) {
            std::cout << "  d_model: " << config.d_model << std::endl;
            std::cout << "  Attention heads: " << config.n_heads << std::endl;
            std::cout << "  Attention layers: " << config.n_attention_layers << std::endl;
            std::cout << "  Transformer FF dim: " << config.transformer_ff_dim << std::endl;
            std::cout << "  Transformer pooling: " << config.transformer_pooling << std::endl;
            std::cout << "  Transformer dropout: " << config.transformer_dropout << std::endl;
        }

        print_architecture_config(config);

        std::cout << "  Latent dim: " << model->latent_dim() << std::endl;

        // The loading-side DatasetConfig this checkpoint implies -- the same one
        // `resolve predict` rebuilds to encode new data. Driven by the shared
        // field registry, so a loader knob added to DatasetConfig is reported
        // here in the edit that adds it. `selection` is the selection the run
        // APPLIED, not the one it was configured with (issue #113).
        std::cout << "\nData Encoding:" << std::endl;
        print_config_block("Dataset", dataset_config_from_checkpoint(schema, config));

        // The training recipe the checkpoint records. Driven by the same field
        // registry as the archive itself, so a hyperparameter added to
        // TrainConfig is reported here in the edit that adds it; only the rows
        // save_train_config actually persists are shown, since the rest read
        // back as struct defaults rather than as what this run used.
        std::cout << "\nTraining Configuration:" << std::endl;
        print_config_block("Train", Trainer::load_train_config(model_path),
                           /*persisted_only=*/true);

        // Print parameter count
        int64_t total_params = 0;
        for (const auto& param : model->parameters()) {
            total_params += param.numel();
        }
        std::cout << "\nTotal parameters: " << total_params << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
