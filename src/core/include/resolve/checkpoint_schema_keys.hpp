#pragma once

#include <cstdint>
#include <string>

// The checkpoint archive keys that serialize a ResolveSchema's own fields.
// save_schema (write) and load_schema (read) in checkpoint.cpp are a matched
// pair that both reference these constants, so a key cannot be written under
// one spelling and read under another. The loader settings
// (ResolveSchema::dataset_config) carry their keys in the DatasetConfig field
// registry (config_registry.hpp).
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
inline constexpr const char* kNCovariates       = "schema_n_covariates";
inline constexpr const char* kNTargets          = "schema_n_targets";
inline constexpr const char* kNCategoricals     = "schema_n_categoricals";
inline constexpr const char* kCategoricalEmbedDim = "schema_categorical_embed_dim";

// Fitted species / genus / family vocabularies. Each is one string list
// (string_list_archive.hpp): two archive entries per list, not one per name,
// so a 30k-species vocabulary does not become 30k zip members.
inline constexpr const char* kSpeciesVocab      = "schema_species_vocab";
inline constexpr const char* kGenusVocab        = "schema_genus_vocab";
inline constexpr const char* kFamilyVocab       = "schema_family_vocab";

// Indexed / prefixed keys. The per-target and per-categorical blocks build a
// prefix once, then suffix each field.
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
// TargetMissingPolicy as int; a checkpoint without it reads as Drop.
inline constexpr const char* kTargetMissing      = "missing";

// Categorical sub-field suffixes (appended to categorical_prefix(i)).
inline constexpr const char* kCategoricalName      = "name";
inline constexpr const char* kCategoricalVocabSize = "vocab_size";

}  // namespace ckpt_schema_keys
}  // namespace resolve
