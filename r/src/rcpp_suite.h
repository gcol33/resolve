// rcpp_suite.h - RSuite wrapper (thin C-facade client over resolve_suite_*).
#ifndef RCPP_SUITE_H
#define RCPP_SUITE_H

#include "rcpp_common.h"

class RSuite {
public:
    static RSuite load(std::string dir, List options) {
        ValuePtr opts(r_list_to_value_map(options, "options"));
        RSuite s;
        s.suite_ = capi_own(resolve_suite_load(dir.c_str(), opts.get()), resolve_suite_free);
        return s;
    }

    RObject manifest() const { return get("manifest"); }
    RObject targets() const { return get("targets"); }
    RObject n_encodings() const { return get("n_encodings"); }
    RObject directory() const { return get("directory"); }

    // `header_cols` may be an empty list when the suite reads no header column.
    RObject predict_frame(List header_cols, List species_cols, List options) {
        ValuePtr species(r_charlist_to_value_map(species_cols));
        ValuePtr header(header_cols.size() > 0 ? r_charlist_to_value_map(header_cols)
                                               : resolve_value_new_null());
        ValuePtr opts(r_list_to_value_map(options, "options"));
        return value_to_r_owned(resolve_suite_predict_dataframe(
            suite_.get(), header.get(), species.get(), opts.get()));
    }

    RObject predict_csv(std::string header_path, std::string species_path, List options) {
        ValuePtr opts(r_list_to_value_map(options, "options"));
        return value_to_r_owned(resolve_suite_predict_csv(
            suite_.get(), header_path.c_str(), species_path.c_str(), opts.get()));
    }

    static RObject verify(std::string dir) {
        return value_to_r_owned(resolve_suite_verify(dir.c_str()));
    }

    static RObject seal(std::string dir) {
        return value_to_r_owned(resolve_suite_seal(dir.c_str()));
    }

private:
    RSuite() = default;
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_suite_get(suite_.get(), what));
    }
    std::shared_ptr<resolve_suite_t> suite_;
};

#endif // RCPP_SUITE_H
