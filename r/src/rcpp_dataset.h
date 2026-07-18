// rcpp_dataset.h - RResolveDataset wrapper (thin C-facade client).
//
// Holds an opaque resolve_dataset_t handle and marshals everything through the
// resolve_c C ABI. The Rcpp module surface (method names / R argument types /
// return shapes) is unchanged from the libtorch-linked version; only the
// implementation moved behind the C facade.
#ifndef RCPP_DATASET_H
#define RCPP_DATASET_H

#include "rcpp_common.h"

class RResolveDataset {
public:
    RResolveDataset() = default;

    static RResolveDataset from_csv(
        std::string header_path, std::string species_path,
        List roles_list, List targets_list, List config_list = List()) {
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_csv(
            header_path.c_str(), species_path.c_str(),
            roles.get(), targets.get(), config.get()), resolve_dataset_free);
        return w;
    }

    static RResolveDataset from_csv_with_schema(
        std::string header_path, std::string species_path,
        List roles_list, List targets_list, RResolveDataset schema_source,
        List config_list = List()) {
        if (!schema_source.ds_) stop("from_csv_with_schema: schema_source is not a loaded dataset");
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_csv_with_schema(
            header_path.c_str(), species_path.c_str(),
            roles.get(), targets.get(), schema_source.ds_.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    static RResolveDataset from_species_csv(
        std::string species_path, List roles_list, List targets_list,
        List config_list = List()) {
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_species_csv(
            species_path.c_str(), roles.get(), targets.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    // --- In-memory (DataFrame) loaders (issue #22). header_cols / species_cols
    // are named lists of character vectors (the R verb coerces every column to
    // character with NA -> "" before calling). ---
    static RResolveDataset from_dataframe(
        List header_cols, List species_cols,
        List roles_list, List targets_list, List config_list = List()) {
        ValuePtr header(r_charlist_to_value_map(header_cols));
        ValuePtr species(r_charlist_to_value_map(species_cols));
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_dataframe(
            header.get(), species.get(), roles.get(), targets.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    static RResolveDataset from_dataframe_header(
        List header_cols, std::string species_path,
        List roles_list, List targets_list, List config_list = List()) {
        ValuePtr header(r_charlist_to_value_map(header_cols));
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_dataframe_header(
            header.get(), species_path.c_str(),
            roles.get(), targets.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    static RResolveDataset from_species_dataframe(
        List species_cols, List roles_list, List targets_list,
        List config_list = List()) {
        ValuePtr species(r_charlist_to_value_map(species_cols));
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_species_dataframe(
            species.get(), roles.get(), targets.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    static RResolveDataset from_dataframe_with_schema(
        List header_cols, List species_cols,
        List roles_list, List targets_list, RResolveDataset schema_source,
        List config_list = List()) {
        if (!schema_source.ds_) stop("from_dataframe_with_schema: schema_source is not a loaded dataset");
        ValuePtr header(r_charlist_to_value_map(header_cols));
        ValuePtr species(r_charlist_to_value_map(species_cols));
        ValuePtr roles(r_list_to_value_map(roles_list));
        ValuePtr targets(r_list_to_value_map(targets_list));
        ValuePtr config(r_list_to_value_map(config_list));
        RResolveDataset w;
        w.ds_ = capi_own(resolve_dataset_from_dataframe_with_schema(
            header.get(), species.get(), roles.get(), targets.get(),
            schema_source.ds_.get(), config.get()),
            resolve_dataset_free);
        return w;
    }

    // Accessors: each dispatches by name to resolve_dataset_get and converts the
    // returned value tree. Optional tensors come back as a NULL-kind value ->
    // R NULL, matching the old Nullable<...> accessors.
    RObject coordinates()      const { return get("coordinates"); }
    RObject covariates()       const { return get("covariates"); }
    RObject hash_embedding()   const { return get("hash_embedding"); }
    RObject species_ids()      const { return get("species_ids"); }
    RObject species_vector()   const { return get("species_vector"); }
    RObject genus_ids()        const { return get("genus_ids"); }
    RObject family_ids()       const { return get("family_ids"); }
    RObject unknown_fraction() const { return get("unknown_fraction"); }
    RObject unknown_count()    const { return get("unknown_count"); }
    RObject categorical_ids()  const { return get("categorical_ids"); }
    // Rank-pool / transformer encoder tensors (parity with Python pool_* accessors).
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

    // Per-column vocabulary: named list of named integer vectors.
    List categorical_vocab() const {
        ValuePtr v(resolve_dataset_get(ds_.get(), "categorical_vocab"));
        capi_check(v.get());
        return categorical_vocab_value_to_r(v.get());
    }

    // Internal handle access for trainer / predictor.
    resolve_dataset_t* handle() const { return ds_.get(); }
    const std::shared_ptr<resolve_dataset_t>& shared() const { return ds_; }

private:
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_dataset_get(ds_.get(), what));
    }
    std::shared_ptr<resolve_dataset_t> ds_;
};

#endif // RCPP_DATASET_H
