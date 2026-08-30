#pragma once

#include <string>

// Single source of truth for the checkpoint archive keys that serialize a
// ResolveSchema. save_schema (write) and load_schema (read) in checkpoint.cpp
// are a matched pair: a key written by one but read under a different spelling
// by the other silently drops a field on load (this is the mechanism behind the
// class_weights checkpoint drop, issue #91 / #98). Referencing these constants
// from both sides makes a rename or a new field a one-place change and removes
// the hand-synced string-literal duplication.
namespace resolve {
namespace ckpt_schema_keys {

// Scalar schema fields.
inline constexpr const char* kNPlots            = "schema_n_plots";
inline constexpr const char* kNSpecies          = "schema_n_species";
inline constexpr const char* kNSpeciesVocab     = "schema_n_species_vocab";
inline constexpr const char* kHasCoordinates    = "schema_has_coordinates";
inline constexpr const char* kHasAbundance      = "schema_has_abundance";
inline constexpr const char* kHasTaxonomy       = "schema_has_taxonomy";
inline constexpr const char* kNGenera           = "schema_n_genera";
inline constexpr const char* kNFamilies         = "schema_n_families";
inline constexpr const char* kNGeneraVocab      = "schema_n_genera_vocab";
inline constexpr const char* kNFamiliesVocab    = "schema_n_families_vocab";
inline constexpr const char* kTrackUnknownFrac  = "schema_track_unknown_fraction";
inline constexpr const char* kTrackUnknownCount = "schema_track_unknown_count";
inline constexpr const char* kNCovariates       = "schema_n_covariates";
inline constexpr const char* kNTargets          = "schema_n_targets";
inline constexpr const char* kNCategoricals     = "schema_n_categoricals";
inline constexpr const char* kCategoricalEmbedDim = "schema_categorical_embed_dim";
inline constexpr const char* kPoolWeighting     = "schema_pool_weighting";
inline constexpr const char* kPoolSpeciesCap    = "schema_pool_species_cap";

// Remaining DatasetConfig knobs (issue #102). Absent on a pre-fix checkpoint;
// the load path then keeps the ResolveSchema defaults, which are the
// DatasetConfig defaults, i.e. exactly today's behaviour.
inline constexpr const char* kTopKSpecies       = "schema_top_k_species";
inline constexpr const char* kSelection         = "schema_selection";
// Per-plot species budget for the pooled / sparse encodings (issue #113).
// Absent on a pre-fix checkpoint, which then keeps the default 0 = no budget --
// exactly what those encodings did before the knob existed.
inline constexpr const char* kSpeciesBudget     = "schema_species_budget";
inline constexpr const char* kRepresentation    = "schema_representation";
inline constexpr const char* kNormalization     = "schema_normalization";
inline constexpr const char* kAggregation       = "schema_aggregation";
inline constexpr const char* kUseTaxonomy       = "schema_use_taxonomy";

// Fitted species / genus / family vocabularies (issue #102). Each is one
// string list written under the shared "<prefix>_lengths" (int64) +
// "<prefix>_bytes" (uint8) layout that TaxonomyVocab::save and
// CategoricalVocab::save already use -- two archive entries per list, not one
// per name, so a 30k-species vocab does not become 30k zip members.
inline constexpr const char* kSpeciesVocab      = "schema_species_vocab";
inline constexpr const char* kGenusVocab        = "schema_genus_vocab";
inline constexpr const char* kFamilyVocab       = "schema_family_vocab";

// Indexed / prefixed keys. The per-target and per-categorical blocks build a
// prefix once, then suffix each field; keep the suffixes here so save/load
// cannot disagree on a spelling.
inline std::string covariate(int64_t i) {
    return "schema_covariate_" + std::to_string(i);
}
inline std::string target_prefix(int64_t i) {
    return "schema_target_" + std::to_string(i) + "_";
}
inline std::string categorical_prefix(int64_t i) {
    return "schema_categorical_" + std::to_string(i) + "_";
}

// Target sub-field suffixes (appended to target_prefix(i)).
inline constexpr const char* kTargetName         = "name";
inline constexpr const char* kTargetTask         = "task";
inline constexpr const char* kTargetTransform    = "transform";
inline constexpr const char* kTargetNumClasses   = "num_classes";
inline constexpr const char* kTargetWeight       = "weight";
inline constexpr const char* kTargetNClassNames  = "n_class_names";
inline constexpr const char* kTargetClassPrefix  = "class_";        // + j
inline constexpr const char* kTargetNClassWeights = "n_class_weights";
inline constexpr const char* kTargetClassWeights = "class_weights";

// Categorical sub-field suffixes (appended to categorical_prefix(i)).
inline constexpr const char* kCategoricalName      = "name";
inline constexpr const char* kCategoricalVocabSize = "vocab_size";

}  // namespace ckpt_schema_keys
}  // namespace resolve
