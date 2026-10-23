// rcpp_dataset.h - RResolveDataset wrapper (thin C-facade client).
//
// Holds an opaque resolve_dataset_t handle and marshals everything through the
// resolve_c C ABI.
#ifndef RCPP_DATASET_H
#define RCPP_DATASET_H

#include "rcpp_common.h"

// The wrap() / as() traits that route through the module's S4 class, declared
// before the loader converts a schemaSource argument.
class RResolveDataset;
RCPP_EXPOSED_CLASS_NODECL(RResolveDataset)

class RResolveDataset {
public:
    RResolveDataset() = default;

    // The one loader (resolve_dataset_load). `header` and `species` are each
    // NULL (header only: a single species table), a CSV path, or a named list
    // of character columns; `schema_source` (an RResolveDataset) or `vocabs`
    // (a vocabulary list) encodes the dataset in a trained model's namespace.
    static RResolveDataset load(SEXP header, SEXP species, List roles_list, List targets_list,
                                List config_list, SEXP schema_source, SEXP vocabs_list) {
        capi_require_loaded();
        const resolve_dataset_t* source = nullptr;
        if (!Rf_isNull(schema_source)) {
            source = as<RResolveDataset*>(schema_source)->handle();
            if (source == nullptr) stop("schemaSource is not a loaded dataset");
        }
        ValuePtr header_value = owned_value(r_table_to_value(header, "header"));
        ValuePtr species_value = owned_value(r_table_to_value(species, "species"));
        ValuePtr roles = owned_value(r_list_to_value_map(roles_list, "roles"));
        ValuePtr targets = owned_value(r_list_to_value_map(targets_list, "targets"));
        ValuePtr config = owned_value(r_list_to_value_map(config_list, "config"));
        ValuePtr vocabs = Rf_isNull(vocabs_list)
                              ? owned_value(resolve_value_new_null())
                              : owned_value(r_list_to_value_map(vocabs_list, "vocabs"));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_load(header_value.get(), species_value.get(), roles.get(),
                                              targets.get(), config.get(), source, vocabs.get()),
                         resolve_dataset_free);
        return w;
    }

    // Accessors: each dispatches by name to resolve_dataset_get and converts the
    // returned value tree. Optional tensors come back as a NULL-kind value ->
    // R NULL.
    RObject coordinates()      const { return get("coordinates"); }
    RObject covariates()       const { return get("covariates"); }
    RObject hash_embedding()   const { return get("hash_embedding"); }
    RObject species_ids()      const { return get("species_ids"); }
    // The sparse encoding's species vector as its sparse rows: a list of
    // row_offsets, species, values and n_species (plot p holds
    // species[row_offsets[p] .. row_offsets[p + 1]) with their values, 0-based).
    RObject species_vector()   const { return get("species_vector"); }
    // The same as the dense (n_plots, n_species) matrix.
    RObject species_vector_dense() const { return get("species_vector_dense"); }
    RObject genus_ids()        const { return get("genus_ids"); }
    RObject family_ids()       const { return get("family_ids"); }
    // The taxonomy of the VOCABULARY: index = species code, value = its genus
    // / family code. genus_ids / family_ids above are per-plot slots.
    RObject species_genus_ids()  const { return get("species_genus_ids"); }
    RObject species_family_ids() const { return get("species_family_ids"); }
    RObject unknown_fraction() const { return get("unknown_fraction"); }
    RObject unknown_count()    const { return get("unknown_count"); }
    RObject categorical_ids()  const { return get("categorical_ids"); }
    // Rank-pool / transformer encoder tensors.
    RObject pool_genus_ids()   const { return get("pool_genus_ids"); }
    RObject pool_family_ids()  const { return get("pool_family_ids"); }
    RObject pool_weights()     const { return get("pool_weights"); }
    RObject pool_mask()        const { return get("pool_mask"); }
    RObject pool_has_cover()   const { return get("pool_has_cover"); }
    RObject has_pool_data()    const { return get("has_pool_data"); }
    RObject targets()          const { return get("targets"); }
    RObject schema()           const { return get("schema"); }
    RObject plot_ids()         const { return get("plot_ids"); }
    RObject species_vocab()    const { return get("species_vocab"); }
    RObject n_plots()          const { return get("n_plots"); }
    RObject config()           const { return get("config"); }
    RObject has_raw_species_data() const { return get("has_raw_species_data"); }
    RObject raw_species_ids()  const { return get("raw_species_ids"); }
    RObject raw_weights()      const { return get("raw_weights"); }
    RObject plot_offsets()     const { return get("plot_offsets"); }
    RObject taxonomy_vocab()   const { return get("taxonomy_vocab"); }
    // Every vocabulary this dataset fitted, in the form the loader's `vocabs`
    // takes.
    RObject vocabs()           const { return get("vocabs"); }

    // The typed species graph a HeterogeneousGNN passes messages on, built
    // from this dataset's taxonomy and co-occurrence. `config` is a
    // HeterogeneousGNNConfig list; an empty list takes the defaults.
    List species_graph(List config) const {
        ValuePtr cfg = owned_value(r_list_to_value_map(config, "config"));
        return as<List>(value_to_r_owned(resolve_build_species_graph(ds_.get(), cfg.get())));
    }

    // Train a variational autoencoder on this dataset's species vectors.
    List pretrain_vae(List config) const {
        ValuePtr cfg = owned_value(r_list_to_value_map(config, "config"));
        return as<List>(value_to_r_owned(resolve_vae_pretrain(ds_.get(), cfg.get())));
    }

    // Per-column vocabulary: named list of named integer vectors.
    List categorical_vocab() const {
        ValuePtr v = owned_value(resolve_dataset_get(ds_.get(), "categorical_vocab"));
        return categorical_vocab_value_to_r(v.get());
    }

    // Internal handle access for model / trainer / predictor.
    resolve_dataset_t* handle() const { return ds_.get(); }

private:
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_dataset_get(ds_.get(), what));
    }
    std::shared_ptr<resolve_dataset_t> ds_;
};

#endif // RCPP_DATASET_H
