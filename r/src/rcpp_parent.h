// rcpp_parent.h - RParent wrapper (thin C-facade client over resolve_parent_*):
// the trained model a fine-tuned model starts from.
#ifndef RCPP_PARENT_H
#define RCPP_PARENT_H

#include "rcpp_common.h"

class RParent {
public:
    static RParent open(std::string source, List config) {
        capi_require_loaded();
        ValuePtr cfg = owned_value(r_list_to_value_map(config, "config"));
        RParent p;
        p.parent_ = capi_own(resolve_parent_open(source.c_str(), cfg.get()), resolve_parent_free);
        return p;
    }

    RObject vocabs() const { return get("vocabs"); }
    RObject dataset_config() const { return get("dataset_config"); }
    RObject model_config() const { return get("model_config"); }
    RObject schema() const { return get("schema"); }
    RObject scalers() const { return get("scalers"); }
    RObject record() const { return get("record"); }

    const resolve_parent_t* handle() const { return parent_.get(); }

private:
    RParent() = default;
    RObject get(const char* what) const {
        return value_to_r_owned(resolve_parent_get(parent_.get(), what));
    }
    std::shared_ptr<resolve_parent_t> parent_;
};

#endif  // RCPP_PARENT_H
