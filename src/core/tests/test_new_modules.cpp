#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "resolve/tabm.hpp"
#include "resolve/adapter.hpp"
#include "resolve/pretraining.hpp"
#include "resolve/vae.hpp"
#include "resolve/loss.hpp"
#include "resolve/attention.hpp"
#include "resolve/model.hpp"

using namespace resolve;
using namespace Catch::Matchers;

// ============================================================================
// Phase 2: TabM — BatchEnsembleLinear
// ============================================================================

TEST_CASE("BatchEnsembleLinear forward shape", "[tabm]") {
    BatchEnsembleLinear layer(32, 64, 8);
    auto x = torch::randn({16, 32});
    auto out = layer->forward(x);

    REQUIRE(out.size(0) == 16);      // batch
    REQUIRE(out.size(1) == 8);       // n_ensembles
    REQUIRE(out.size(2) == 64);      // out_features
}

TEST_CASE("BatchEnsembleLinear accessors", "[tabm]") {
    BatchEnsembleLinear layer(10, 20, 4);

    REQUIRE(layer->in_features() == 10);
    REQUIRE(layer->out_features() == 20);
    REQUIRE(layer->n_ensembles() == 4);
}

TEST_CASE("BatchEnsembleLinear gradient flow", "[tabm]") {
    BatchEnsembleLinear layer(16, 32, 4);
    auto x = torch::randn({8, 16}, torch::requires_grad(true));
    auto out = layer->forward(x);
    auto loss = out.sum();
    loss.backward();

    REQUIRE(x.grad().defined());
    REQUIRE(x.grad().numel() > 0);
}

TEST_CASE("BatchEnsembleLinear ensemble diversity", "[tabm]") {
    BatchEnsembleLinear layer(16, 32, 4);
    auto x = torch::randn({8, 16});
    auto out = layer->forward(x);

    // Each ensemble member should produce different outputs (due to r/s perturbations)
    auto member0 = out.select(1, 0);  // (8, 32)
    auto member1 = out.select(1, 1);  // (8, 32)
    auto diff = (member0 - member1).abs().mean().item<float>();
    REQUIRE(diff > 1e-6f);  // Members should differ
}

// ============================================================================
// Phase 2: TabM — TabMEncoder
// ============================================================================

TEST_CASE("TabMEncoder forward shape", "[tabm]") {
    std::vector<int64_t> dims = {64, 32};
    TabMEncoder encoder(32, dims, 8, 0.1f, std::string("mean"));
    auto x = torch::randn({16, 32});
    auto out = encoder->forward(x);

    REQUIRE(out.size(0) == 16);      // batch
    REQUIRE(out.size(1) == 32);      // last hidden dim
}

TEST_CASE("TabMEncoder forward_all returns per-ensemble", "[tabm]") {
    std::vector<int64_t> dims = {64, 32};
    TabMEncoder encoder(32, dims, 8, 0.0f, std::string("mean"));
    auto x = torch::randn({16, 32});
    auto all = encoder->forward_all(x);

    REQUIRE(all.size(0) == 16);      // batch
    REQUIRE(all.size(1) == 8);       // n_ensembles
    REQUIRE(all.size(2) == 32);      // output_dim
}

TEST_CASE("TabMEncoder aggregation matches forward_all mean", "[tabm]") {
    std::vector<int64_t> dims = {32, 16};
    TabMEncoder encoder(16, dims, 4, 0.0f, std::string("mean"));
    encoder->eval();  // Disable dropout
    auto x = torch::randn({8, 16});

    auto aggregated = encoder->forward(x);
    auto all = encoder->forward_all(x);
    auto manual_mean = all.mean(1);  // Mean across ensemble dim

    auto diff = (aggregated - manual_mean).abs().max().item<float>();
    REQUIRE(diff < 1e-5f);
}

TEST_CASE("TabMEncoder accessors", "[tabm]") {
    std::vector<int64_t> dims = {64, 16};
    TabMEncoder encoder(32, dims, 12);

    REQUIRE(encoder->output_dim() == 16);
    REQUIRE(encoder->n_ensembles() == 12);
}

// ============================================================================
// Phase 3: T-JEPA — FeatureMasker
// ============================================================================

TEST_CASE("FeatureMasker create_mask shape", "[pretraining]") {
    FeatureMasker masker(100, 0.3f);
    auto mask = masker->create_mask(16);

    REQUIRE(mask.size(0) == 16);
    REQUIRE(mask.size(1) == 100);
}

TEST_CASE("FeatureMasker mask ratio is approximate", "[pretraining]") {
    FeatureMasker masker(200, 0.3f, MaskStrategy::Random);
    auto mask = masker->create_mask(64);

    // Mean of mask should be ~0.7 (70% kept)
    float keep_ratio = mask.to(torch::kFloat).mean().item<float>();
    REQUIRE_THAT(keep_ratio, WithinAbs(0.7f, 0.1f));
}

TEST_CASE("FeatureMasker apply_mask preserves shape", "[pretraining]") {
    FeatureMasker masker(50, 0.3f);
    auto features = torch::randn({8, 50});
    auto mask = masker->create_mask(8);
    auto masked = masker->apply_mask(features, mask);

    REQUIRE(masked.size(0) == 8);
    REQUIRE(masked.size(1) == 50);
}

TEST_CASE("FeatureMasker accessors", "[pretraining]") {
    FeatureMasker masker(64, 0.4f);
    REQUIRE(masker->n_features() == 64);
    REQUIRE_THAT(masker->mask_ratio(), WithinAbs(0.4f, 1e-6f));
}

// ============================================================================
// Phase 3: T-JEPA — JEPAPredictor
// ============================================================================

TEST_CASE("JEPAPredictor forward shape", "[pretraining]") {
    JEPAPredictor predictor(64, 128, 2, 0.1f);
    auto context = torch::randn({16, 64});
    auto predicted = predictor->forward(context);

    REQUIRE(predicted.size(0) == 16);
    REQUIRE(predicted.size(1) == 64);  // Same as latent_dim
}

TEST_CASE("JEPAPredictor gradient flow", "[pretraining]") {
    JEPAPredictor predictor(32, 64, 2, 0.0f);
    auto x = torch::randn({8, 32}, torch::requires_grad(true));
    auto out = predictor->forward(x);
    out.sum().backward();

    REQUIRE(x.grad().defined());
}

// ============================================================================
// Phase 4: SCARF — SCARFCorruptor
// ============================================================================

TEST_CASE("SCARFCorruptor corrupt preserves shape", "[pretraining]") {
    SCARFCorruptor corruptor(50, 0.6f);
    auto features = torch::randn({16, 50});
    auto corrupted = corruptor->corrupt(features);

    REQUIRE(corrupted.size(0) == 16);
    REQUIRE(corrupted.size(1) == 50);
}

TEST_CASE("SCARFCorruptor actually modifies features", "[pretraining]") {
    SCARFCorruptor corruptor(50, 0.6f);
    auto features = torch::randn({16, 50});
    auto corrupted = corruptor->corrupt(features);

    // With 60% corruption, the outputs should differ
    auto diff = (features - corrupted).abs().sum().item<float>();
    REQUIRE(diff > 0.0f);
}

TEST_CASE("SCARFCorruptor accessors", "[pretraining]") {
    SCARFCorruptor corruptor(100, 0.5f);
    REQUIRE(corruptor->n_features() == 100);
    REQUIRE_THAT(corruptor->corruption_rate(), WithinAbs(0.5f, 1e-6f));
}

// ============================================================================
// Phase 4: SCARF — ProjectionHead
// ============================================================================

TEST_CASE("ProjectionHead forward shape", "[pretraining]") {
    ProjectionHead head(64, 128);
    auto x = torch::randn({16, 64});
    auto out = head->forward(x);

    REQUIRE(out.size(0) == 16);
    REQUIRE(out.size(1) == 128);
}

// ============================================================================
// Phase 6: NCA Loss
// ============================================================================

TEST_CASE("NCALoss forward returns scalar", "[nca]") {
    NCALoss nca(32, 5, 0.1f, 16);

    auto latent = torch::randn({16, 32});
    auto targets = torch::randint(0, 5, {16});

    // Must first populate reference set
    nca->update_references(latent, targets);

    auto loss = nca->forward(latent, targets);
    REQUIRE(loss.dim() == 0);  // Scalar
    REQUIRE(loss.item<float>() > 0.0f);
}

TEST_CASE("NCALoss predict returns class probabilities", "[nca]") {
    NCALoss nca(32, 5, 0.1f, 16);

    auto ref_latent = torch::randn({32, 32});
    auto ref_targets = torch::randint(0, 5, {32});
    nca->update_references(ref_latent, ref_targets);

    auto query = torch::randn({8, 32});
    auto probs = nca->predict(query);

    REQUIRE(probs.size(0) == 8);
    REQUIRE(probs.size(1) == 5);

    // Probabilities should sum to ~1
    auto sums = probs.sum(1);
    for (int i = 0; i < 8; ++i) {
        REQUIRE_THAT(sums[i].item<float>(), WithinAbs(1.0f, 1e-4f));
    }
}

TEST_CASE("NCALoss update_references", "[nca]") {
    NCALoss nca(16, 3, 0.1f, 8);

    auto latent1 = torch::randn({10, 16});
    auto targets1 = torch::randint(0, 3, {10});
    nca->update_references(latent1, targets1);

    // Should not crash on forward after update
    auto loss = nca->forward(latent1, targets1);
    REQUIRE(loss.item<float>() >= 0.0f);
}

// ============================================================================
// Phase 7: ExcelFormer Encoder
// ============================================================================

TEST_CASE("ExcelFormerEncoder forward shape", "[excelformer]") {
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(10, no_cats, 64, 4, 2, 0, 0.1f, 0.5f, true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("ExcelFormerEncoder with categoricals", "[excelformer]") {
    std::vector<int64_t> cat_cards = {5, 10};
    ExcelFormerEncoder encoder(8, cat_cards, 64, 4, 2, 0, 0.1f, 0.5f, true);

    auto numerical = torch::randn({8, 8});
    std::vector<torch::Tensor> categoricals = {
        torch::randint(0, 5, {8}),
        torch::randint(0, 10, {8})
    };
    auto out = encoder->forward(numerical, categoricals);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);
}

TEST_CASE("ExcelFormerEncoder feature_importance", "[excelformer]") {
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(10, no_cats, 32, 4, 2);
    auto importance = encoder->feature_importance();

    // Should be sigmoid of learnable logits, so in [0, 1]
    REQUIRE(importance.min().item<float>() >= 0.0f);
    REQUIRE(importance.max().item<float>() <= 1.0f);
}

TEST_CASE("ExcelFormerEncoder importance_logits receive gradient", "[excelformer]") {
    // The semi-permeable mask must be differentiable so the learnable
    // importance parameter actually trains. A hard boolean mask leaves it
    // gradient-dead (importance stuck at its init, mask permanently open).
    std::vector<int64_t> no_cats;
    ExcelFormerEncoder encoder(10, no_cats, 32, 4, 2, 0, 0.1f, 0.5f, true);

    torch::Tensor importance_logits;
    for (const auto& named : encoder->named_parameters()) {
        if (named.key().find("importance_logits") != std::string::npos) {
            importance_logits = named.value();
        }
    }
    REQUIRE(importance_logits.defined());

    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);
    auto loss = out.pow(2).mean();
    loss.backward();

    REQUIRE(importance_logits.grad().defined());
    REQUIRE(importance_logits.grad().abs().sum().item<float>() > 0.0f);
}

// ============================================================================
// Phase 8: VAE — SpeciesVAE
// ============================================================================

TEST_CASE("SpeciesVAE forward returns triple", "[vae]") {
    VAEConfig config;
    config.latent_dim = 32;
    config.encoder_dims = {64, 32};
    config.decoder_dims = {32, 64};

    SpeciesVAE vae(100, config);
    auto species = torch::randn({8, 100});
    auto [recon, mu, logvar] = vae->forward(species);

    REQUIRE(recon.size(0) == 8);
    REQUIRE(recon.size(1) == 100);   // Reconstructed species vector
    REQUIRE(mu.size(0) == 8);
    REQUIRE(mu.size(1) == 32);       // Latent dim
    REQUIRE(logvar.size(0) == 8);
    REQUIRE(logvar.size(1) == 32);
}

TEST_CASE("SpeciesVAE encode", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};

    SpeciesVAE vae(50, config);
    auto species = torch::randn({8, 50});
    auto [mu, logvar] = vae->encode(species);

    REQUIRE(mu.size(0) == 8);
    REQUIRE(mu.size(1) == 16);
    REQUIRE(logvar.size(0) == 8);
    REQUIRE(logvar.size(1) == 16);
}

TEST_CASE("SpeciesVAE decode", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};

    SpeciesVAE vae(50, config);
    auto z = torch::randn({8, 16});
    auto decoded = vae->decode(z);

    REQUIRE(decoded.size(0) == 8);
    REQUIRE(decoded.size(1) == 50);
}

TEST_CASE("SpeciesVAE reparameterize", "[vae]") {
    auto mu = torch::zeros({8, 16});
    auto logvar = torch::zeros({8, 16});  // var = 1, so z ~ N(0, 1)

    auto z = SpeciesVAEImpl::reparameterize(mu, logvar);
    REQUIRE(z.size(0) == 8);
    REQUIRE(z.size(1) == 16);

    // Mean should be ~0 for many samples
    auto mean = z.mean().item<float>();
    REQUIRE_THAT(mean, WithinAbs(0.0f, 0.5f));
}

TEST_CASE("SpeciesVAE vae_loss", "[vae]") {
    auto recon = torch::randn({8, 50});
    auto input = torch::randn({8, 50});
    auto mu = torch::randn({8, 16});
    auto logvar = torch::randn({8, 16});

    auto loss = SpeciesVAEImpl::vae_loss(recon, input, mu, logvar, 1.0f);
    REQUIRE(loss.dim() == 0);  // Scalar
    REQUIRE(loss.item<float>() > 0.0f);
}

TEST_CASE("SpeciesVAE get_projection_weights", "[vae]") {
    VAEConfig config;
    config.latent_dim = 32;
    config.encoder_dims = {64};
    config.decoder_dims = {64};

    SpeciesVAE vae(100, config);
    auto weights = vae->get_projection_weights();

    // Should be (input_dim, latent_dim) for initializing species_projection
    // Note: actual shape depends on implementation (may be transposed)
    REQUIRE(weights.defined());
    REQUIRE(weights.numel() > 0);
}

TEST_CASE("SpeciesVAE accessors", "[vae]") {
    VAEConfig config;
    config.latent_dim = 32;
    config.encoder_dims = {64};
    config.decoder_dims = {64};

    SpeciesVAE vae(100, config);
    REQUIRE(vae->input_dim() == 100);
    REQUIRE(vae->latent_dim() == 32);
}

// ============================================================================
// Phase 9: Heterogeneous GNN
// ============================================================================

TEST_CASE("TypedMessagePassingLayer forward shape", "[hetero_gnn]") {
    TypedMessagePassingLayer layer(32, 32, 3, 4, 0.0f);

    int n_nodes = 20;
    int n_edges = 40;
    auto node_features = torch::randn({n_nodes, 32});

    // Random edges: source and target node indices
    auto edge_index = torch::randint(0, n_nodes, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto out = layer->forward(node_features, edge_index, edge_type);

    REQUIRE(out.size(0) == n_nodes);
    REQUIRE(out.size(1) == 32);
}

TEST_CASE("HeterogeneousGNNEncoder forward shape", "[hetero_gnn]") {
    int n_species = 50;
    HeterogeneousGNNEncoder encoder(n_species, 64, 32, 2, 3, 4, 0.0f);

    int n_edges = 100;
    auto edge_index = torch::randint(0, n_species, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto embeddings = encoder->forward(edge_index, edge_type);

    REQUIRE(embeddings.size(0) == n_species);
    REQUIRE(embeddings.size(1) == 32);  // output_dim
}

TEST_CASE("HeterogeneousGNNEncoder aggregate_for_plots", "[hetero_gnn]") {
    int n_species = 50;
    int batch_size = 8;
    int output_dim = 32;

    // Simulate species embeddings and abundance vectors
    auto species_emb = torch::randn({n_species, output_dim});
    auto species_vector = torch::randn({batch_size, n_species}).abs();  // Positive abundances

    auto plot_features = HeterogeneousGNNEncoderImpl::aggregate_for_plots(
        species_emb, species_vector);

    REQUIRE(plot_features.size(0) == batch_size);
    REQUIRE(plot_features.size(1) == output_dim);
}

TEST_CASE("HeterogeneousGNNEncoder accessors", "[hetero_gnn]") {
    HeterogeneousGNNEncoder encoder(100, 128, 64, 3, 3, 4, 0.1f);
    REQUIRE(encoder->output_dim() == 64);
    REQUIRE(encoder->n_species() == 100);
}

TEST_CASE("HeterogeneousGNNEncoder gradient flow", "[hetero_gnn]") {
    int n_species = 20;
    HeterogeneousGNNEncoder encoder(n_species, 32, 16, 2, 3, 4, 0.0f);

    int n_edges = 30;
    auto edge_index = torch::randint(0, n_species, {2, n_edges});
    auto edge_type = torch::randint(0, 3, {n_edges});

    auto embeddings = encoder->forward(edge_index, edge_type);
    auto loss = embeddings.sum();
    loss.backward();

    // Check that learnable parameters got gradients
    bool has_grad = false;
    for (auto& param : encoder->parameters()) {
        if (param.grad().defined() && param.grad().abs().sum().item<float>() > 0) {
            has_grad = true;
            break;
        }
    }
    REQUIRE(has_grad);
}

// ============================================================================
// Phase 1: TabularAdapter with FT-Transformer
// ============================================================================

TEST_CASE("TabularAdapter FTTransformer forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;
    schema.covariate_names = {};
    schema.track_unknown_fraction = true;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.hidden_dims = {64, 32};
    config.encoder_architecture = EncoderArchitecture::FTTransformer;
    config.ft_transformer.d_model = 64;
    config.ft_transformer.n_heads = 4;
    config.ft_transformer.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // n_continuous = 2 (coords) + 1 (unknown_fraction) + 32 (hash) = 35
    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);  // Latent dimension depends on config
}

TEST_CASE("TabularAdapter ExcelFormer forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = false;
    schema.covariate_names = {};
    schema.track_unknown_fraction = false;
    schema.targets.push_back({"area", TaskType::Regression, TransformType::None, 0, 1.0f});

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Hash;
    config.hash_dim = 32;
    config.hidden_dims = {64, 32};
    config.encoder_architecture = EncoderArchitecture::ExcelFormer;
    config.excelformer.d_model = 64;
    config.excelformer.n_heads = 4;
    config.excelformer.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // n_continuous = 2 (coords) + 32 (hash) = 34
    auto continuous = torch::randn({8, 34});

    auto out = adapter->forward(continuous);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter TabNet forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::TabNet;
    config.tabnet.n_steps = 3;
    config.tabnet.n_d = 16;
    config.tabnet.n_a = 16;

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter SAINT forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::SAINT;
    config.saint.d_model = 64;
    config.saint.n_heads = 4;
    config.saint.n_layers = 2;

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter GNN forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.hash_dim = 32;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::GNN;
    config.gnn.hidden_dim = 64;
    config.gnn.n_layers = 2;
    config.gnn.k_neighbors = 4;  // Must be < batch_size

    TabularAdapter adapter(schema, config);

    auto continuous = torch::randn({8, 35});
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});

    auto out = adapter->forward(continuous, genus_ids, family_ids);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

TEST_CASE("TabularAdapter HeterogeneousGNN forward", "[adapter]") {
    ResolveSchema schema;
    schema.n_plots = 100;
    schema.n_species = 50;
    schema.n_species_vocab = 50;
    schema.has_coordinates = true;
    schema.has_abundance = true;
    schema.has_taxonomy = true;
    schema.n_genera = 20;
    schema.n_families = 10;

    ModelConfig config;
    config.species_encoding = SpeciesEncodingMode::Sparse;
    config.uses_explicit_vector = true;
    config.species_embed_dim = 16;
    config.n_taxonomy_slots = 3;
    config.genus_emb_dim = 4;
    config.family_emb_dim = 4;
    config.encoder_architecture = EncoderArchitecture::HeterogeneousGNN;
    config.heterogeneous_gnn.hidden_dim = 64;
    config.heterogeneous_gnn.n_layers = 2;

    TabularAdapter adapter(schema, config);

    // Build a simple species graph: 50 species, some co-occurrence edges
    auto edge_index = torch::tensor({{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 0}}, torch::kInt64);
    auto edge_type = torch::zeros({6}, torch::kInt64);  // All type 0
    adapter->set_species_graph(edge_index, edge_type);

    // n_continuous = 2 (coords) + 1 (unknown_fraction) = 3
    auto continuous = torch::randn({8, 3});
    auto genus_ids = torch::randint(0, 21, {8, 3});
    auto family_ids = torch::randint(0, 11, {8, 3});
    auto species_vector = torch::rand({8, 50});  // Required for HeterogeneousGNN

    auto out = adapter->forward(continuous, genus_ids, family_ids, {}, species_vector);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) > 0);
}

// ============================================================================
// Standalone attention encoder tests
// ============================================================================

TEST_CASE("FTTransformerEncoder forward shape", "[attention]") {
    std::vector<int64_t> no_cats;
    FTTransformerEncoder encoder(10, no_cats, 64, 4, 2, 0, 0.1f, true, true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("TabNetEncoder forward shape", "[attention]") {
    TabNetEncoder encoder(32, 3, 16, 16, 1.5f, 1e-3f);
    auto x = torch::randn({8, 32});
    auto [out, importance] = encoder->forward(x);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 16);  // n_d
    REQUIRE(importance.size(0) == 8);
    REQUIRE(importance.size(1) == 32);  // input_dim
}

TEST_CASE("SAINTEncoder forward shape", "[attention]") {
    std::vector<int64_t> no_cats;
    SAINTEncoder encoder(10, no_cats, 64, 4, 2, 0, 0.1f, true, true);
    auto numerical = torch::randn({8, 10});
    auto out = encoder->forward(numerical);

    REQUIRE(out.size(0) == 8);
    REQUIRE(out.size(1) == 64);  // d_model
}

TEST_CASE("GNNEncoder forward shape", "[attention]") {
    GNNEncoder encoder(16, 32, 16, 2, GNNEncoderImpl::GNNType::GCN, 4, 0.0f);

    int n_nodes = 10;
    auto x = torch::randn({n_nodes, 16});
    auto adj = torch::eye(n_nodes) + torch::randn({n_nodes, n_nodes}).abs() * 0.1f;

    auto out = encoder->forward(x, adj);

    REQUIRE(out.size(0) == n_nodes);
    REQUIRE(out.size(1) == 16);  // out_features
}

// ============================================================================
// VAEPretrainer (quick smoke test)
// ============================================================================

TEST_CASE("VAEPretrainer construction", "[vae]") {
    VAEConfig config;
    config.latent_dim = 16;
    config.encoder_dims = {32};
    config.decoder_dims = {32};
    config.pretrain_epochs = 2;
    config.batch_size = 16;

    VAEPretrainer pretrainer(50, config);

    auto& vae = pretrainer.vae();
    REQUIRE(vae->input_dim() == 50);
    REQUIRE(vae->latent_dim() == 16);
}

TEST_CASE("VAEPretrainer pretrain runs", "[vae]") {
    VAEConfig config;
    config.latent_dim = 8;
    config.encoder_dims = {16};
    config.decoder_dims = {16};
    config.pretrain_epochs = 2;
    config.batch_size = 8;

    VAEPretrainer pretrainer(20, config);
    auto species = torch::randn({32, 20}).abs();

    auto result = pretrainer.pretrain(species);

    REQUIRE(result.epochs_completed == 2);
    REQUIRE(result.loss_history.size() == 2);
    REQUIRE(result.recon_loss_history.size() == 2);
    REQUIRE(result.kl_loss_history.size() == 2);
    REQUIRE(result.total_time_seconds > 0.0f);
}
