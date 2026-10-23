// rcpp_fuzzy.h - RFuzzyIndex wrapper (thin C-facade client over resolve_fuzzy_*).
#ifndef RCPP_FUZZY_H
#define RCPP_FUZZY_H

#include "rcpp_common.h"

class RFuzzyIndex {
public:
    // A Damerau-Levenshtein index over `entries`; `options` is a list of
    // max_edit_distance / case_insensitive / damerau / buckets.
    static RFuzzyIndex build(CharacterVector entries, List options) {
        capi_require_loaded();
        ValuePtr names = owned_value(r_to_value(entries));
        ValuePtr opts = owned_value(r_list_to_value_map(options, "options"));
        RFuzzyIndex index;
        index.index_ = capi_own(resolve_fuzzy_build(names.get(), opts.get()), resolve_fuzzy_free);
        return index;
    }

    // The best matches of each needle, one row per match: query (0-based
    // needle position), entry, id (0-based entry position), distance.
    RObject query(CharacterVector needles, List options) const {
        ValuePtr queries = owned_value(r_to_value(needles));
        ValuePtr opts = owned_value(r_list_to_value_map(options, "options"));
        return value_to_r_owned(resolve_fuzzy_query(index_.get(), queries.get(), opts.get()));
    }

    RObject size() const { return get("size"); }
    RObject bucket_count() const { return get("bucket_count"); }
    RObject max_supported_distance() const { return get("max_supported_distance"); }

private:
    RFuzzyIndex() = default;
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_fuzzy_get(index_.get(), what));
    }
    std::shared_ptr<resolve_fuzzy_t> index_;
};

#endif // RCPP_FUZZY_H
