// rcpp_common.h - R <-> C ABI marshaling for the resolve package.
//
// Issue #17: the R package no longer touches libtorch's C++ ABI. It reaches the
// `resolve_c` shared library through the flat C facade in resolve/resolve_capi.h
// and marshals everything across the boundary as a `resolve_value_t` tree. This
// is the ONLY engine header included under r/src/; there is no `#include
// <torch/...>` anywhere in the package sources.
//
// resolve_c is bound at RUNTIME (resolve_capi_dynload.h), not linked at build
// time, so the package installs and R CMD checks with no backend present (the
// mlverse/torch model). The dynload header supplies the same `resolve_*` symbols
// as forwarders, so every call site below is unchanged; a call made before the
// backend is loaded would hit a NULL forwarder, so engine entry points guard
// with capi_require_loaded() first (the R verbs also gate on resolve.available()).
#ifndef RCPP_COMMON_H
#define RCPP_COMMON_H

#include <Rcpp.h>
#include "resolve_capi_dynload.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Rcpp;

// =============================================================================
// Error checking: turn a C-facade failure into an R error
// =============================================================================

// Raise a clean R error (never a NULL-pointer call) when the resolve_c backend
// has not been loaded. Call at the top of any function that reaches a forwarder
// without going through an R verb that already gated on resolve.available().
inline void capi_require_loaded() {
    if (!resolve_capi_available()) {
        stop("The resolve_c backend is not installed. Install it with "
             "resolve.install_backend(), or set RESOLVE_C_HOME to a directory "
             "containing the resolve_c shared library.");
    }
}

inline void capi_check(const void* p) {
    if (p == nullptr) stop(resolve_last_error());
}
inline void capi_check_status(int rc) {
    if (rc != 0) stop(resolve_last_error());
}

// =============================================================================
// RAII for a resolve_value_t we own (input trees we build, result trees we get)
// =============================================================================

class ValuePtr {
public:
    ValuePtr() : v_(nullptr) {}
    explicit ValuePtr(resolve_value_t* v) : v_(v) {}
    ~ValuePtr() { if (v_) resolve_value_free(v_); }
    ValuePtr(const ValuePtr&) = delete;
    ValuePtr& operator=(const ValuePtr&) = delete;
    ValuePtr(ValuePtr&& o) noexcept : v_(o.v_) { o.v_ = nullptr; }
    resolve_value_t* get() const { return v_; }
private:
    resolve_value_t* v_;
};

// Custom-deleter shared handle factory: mirrors the old shared_ptr<engine>
// ownership so Rcpp's value-returning factories and module storage can copy a
// wrapper without double-freeing the underlying C handle.
template <typename T>
inline std::shared_ptr<T> capi_own(T* handle, void (*deleter)(T*)) {
    capi_check(handle);
    return std::shared_ptr<T>(handle, deleter);
}

// =============================================================================
// value tree -> R object (recursive). Single source of truth for every result
// accessor: each C op returns a resolve_value_t and this turns it into the same
// R shape the old Rcpp converters produced.
// =============================================================================

inline RObject value_to_r(const resolve_value_t* v);

// int64 scalar -> R integer when it fits, else double (R has no native int64).
inline RObject int_scalar_to_r(int64_t x) {
    if (x > std::numeric_limits<int>::max() || x < std::numeric_limits<int>::min()) {
        return wrap(static_cast<double>(x));
    }
    return wrap(static_cast<int>(x));
}

inline IntegerVector int_array_to_r(const int64_t* data, int64_t n) {
    IntegerVector out(n);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t x = data ? data[i] : 0;
        if (x > std::numeric_limits<int>::max() || x < std::numeric_limits<int>::min()) {
            Rcpp::warning("resolve: int64 value out of int range, narrowed: %lld",
                          static_cast<long long>(x));
        }
        out[i] = static_cast<int>(x);
    }
    return out;
}

inline RObject value_to_r(const resolve_value_t* v) {
    if (v == nullptr) return R_NilValue;
    switch (resolve_value_kind(v)) {
        case RESOLVE_VALUE_NULL:
            return R_NilValue;
        case RESOLVE_VALUE_BOOL:
            return wrap(static_cast<bool>(resolve_value_as_bool(v)));
        case RESOLVE_VALUE_INT:
            return int_scalar_to_r(resolve_value_as_int(v));
        case RESOLVE_VALUE_DOUBLE:
            return wrap(resolve_value_as_double(v));
        case RESOLVE_VALUE_STRING:
            return wrap(std::string(resolve_value_as_string(v)));
        case RESOLVE_VALUE_INT_ARRAY: {
            int64_t n = 0;
            const int64_t* d = resolve_value_as_int_array(v, &n);
            return int_array_to_r(d, n);
        }
        case RESOLVE_VALUE_DOUBLE_ARRAY: {
            int64_t n = 0;
            const double* d = resolve_value_as_double_array(v, &n);
            return NumericVector(d, d + n);
        }
        case RESOLVE_VALUE_STRING_ARRAY: {
            int64_t n = resolve_value_string_array_size(v);
            CharacterVector out(n);
            for (int64_t i = 0; i < n; ++i) out[i] = resolve_value_string_at(v, i);
            return out;
        }
        case RESOLVE_VALUE_DOUBLE_MATRIX: {
            int64_t nr = 0, nc = 0;
            const double* d = resolve_value_as_double_matrix(v, &nr, &nc);
            NumericMatrix out(nr, nc);
            for (int64_t i = 0; i < nr; ++i)
                for (int64_t j = 0; j < nc; ++j)
                    out(i, j) = d ? d[i * nc + j] : 0.0;
            return out;
        }
        case RESOLVE_VALUE_INT_MATRIX: {
            int64_t nr = 0, nc = 0;
            const int64_t* d = resolve_value_as_int_matrix(v, &nr, &nc);
            IntegerMatrix out(nr, nc);
            for (int64_t i = 0; i < nr; ++i)
                for (int64_t j = 0; j < nc; ++j) {
                    const int64_t x = d ? d[i * nc + j] : 0;
                    if (x > std::numeric_limits<int>::max() || x < std::numeric_limits<int>::min()) {
                        Rcpp::warning("resolve: int64 value out of int range, narrowed: %lld",
                                      static_cast<long long>(x));
                    }
                    out(i, j) = static_cast<int>(x);
                }
            return out;
        }
        case RESOLVE_VALUE_MAP: {
            int64_t n = resolve_map_size(v);
            List out(n);
            CharacterVector names(n);
            for (int64_t i = 0; i < n; ++i) {
                names[i] = resolve_map_key_at(v, i);
                out[i] = value_to_r(resolve_map_value_at(v, i));
            }
            out.attr("names") = names;
            return out;
        }
        case RESOLVE_VALUE_LIST: {
            int64_t n = resolve_list_size(v);
            List out(n);
            for (int64_t i = 0; i < n; ++i) out[i] = value_to_r(resolve_list_at(v, i));
            return out;
        }
    }
    return R_NilValue;
}

// Take ownership of a C-returned value tree: stop on NULL (error), convert, free.
inline RObject value_to_r_owned(resolve_value_t* v) {
    capi_check(v);
    ValuePtr guard(v);
    return value_to_r(v);
}

// =============================================================================
// R object -> value tree (recursive). Used for config / roles / targets /
// schema lists. Matrices never appear inside these, so only scalars / atomic
// vectors / (named) lists are handled; method-argument matrices are set
// explicitly via the matrix helpers below.
// =============================================================================

inline resolve_value_t* r_to_value(SEXP x) {
    if (Rf_isNull(x)) return resolve_value_new_null();

    // A NAMED atomic vector (e.g. a classification class_mapping written as
    // c(forest = 0L, grass = 1L)) is semantically a map. Promote it to a map so
    // the C-ABI parsers that only read maps (parse_targets' class_mapping)
    // receive the names instead of silently getting a nameless array -- which
    // would drop the mapping and let the loader re-factorize with a different
    // code order.
    const int atype = TYPEOF(x);
    if (atype == INTSXP || atype == REALSXP || atype == STRSXP || atype == LGLSXP) {
        SEXP nm = Rf_getAttrib(x, R_NamesSymbol);
        if (nm != R_NilValue && Rf_xlength(x) > 0) {
            resolve_value_t* map = resolve_value_new_map();
            for (R_xlen_t i = 0; i < Rf_xlength(x); ++i) {
                SEXP nm_i = STRING_ELT(nm, i);
                std::string key = (nm_i == NA_STRING) ? std::string() : std::string(CHAR(nm_i));
                resolve_value_t* val;
                switch (atype) {
                    case INTSXP:  val = resolve_value_new_int(INTEGER(x)[i]); break;
                    case REALSXP: val = resolve_value_new_double(REAL(x)[i]); break;
                    case LGLSXP:  val = resolve_value_new_bool(LOGICAL(x)[i] == TRUE ? 1 : 0); break;
                    default: {
                        SEXP e = STRING_ELT(x, i);
                        val = resolve_value_new_string(e == NA_STRING ? "" : CHAR(e));
                    }
                }
                resolve_map_set_value(map, key.c_str(), val);
            }
            return map;
        }
    }

    switch (TYPEOF(x)) {
        case LGLSXP: {
            LogicalVector v(x);
            if (v.size() == 1) return resolve_value_new_bool(v[0] == TRUE ? 1 : 0);
            std::vector<int64_t> a(v.size());
            for (R_xlen_t i = 0; i < v.size(); ++i) a[i] = (v[i] == TRUE) ? 1 : 0;
            return resolve_value_new_int_array(a.data(), a.size());
        }
        case INTSXP: {
            IntegerVector v(x);
            if (v.size() == 1) return resolve_value_new_int(v[0]);
            std::vector<int64_t> a(v.begin(), v.end());
            return resolve_value_new_int_array(a.data(), a.size());
        }
        case REALSXP: {
            NumericVector v(x);
            if (v.size() == 1) return resolve_value_new_double(v[0]);
            std::vector<double> a(v.begin(), v.end());
            return resolve_value_new_double_array(a.data(), a.size());
        }
        case STRSXP: {
            CharacterVector v(x);
            if (v.size() == 1) return resolve_value_new_string(
                v[0] == NA_STRING ? "" : (const char*)CHAR(v[0]));
            std::vector<std::string> s(v.size());
            std::vector<const char*> p(v.size());
            for (R_xlen_t i = 0; i < v.size(); ++i) {
                s[i] = (v[i] == NA_STRING) ? "" : std::string(CHAR(v[i]));
                p[i] = s[i].c_str();
            }
            return resolve_value_new_string_array(p.data(), p.size());
        }
        case VECSXP: {
            List v(x);
            bool named = v.hasAttribute("names");
            if (named) {
                resolve_value_t* map = resolve_value_new_map();
                CharacterVector names = v.names();
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    std::string key = (names[i] == NA_STRING) ? std::string()
                                                              : std::string(CHAR(names[i]));
                    resolve_map_set_value(map, key.c_str(), r_to_value(v[i]));
                }
                return map;
            }
            resolve_value_t* lst = resolve_value_new_list();
            for (R_xlen_t i = 0; i < v.size(); ++i)
                resolve_list_append_value(lst, r_to_value(v[i]));
            return lst;
        }
        default:
            stop("r_to_value: unsupported R type (SEXPTYPE %d)", TYPEOF(x));
    }
    return resolve_value_new_null();
}

// Build a value MAP from a (possibly empty / R_NilValue) R list argument.
//
// A named list is the carrier for every structured argument (roles / targets /
// config / schema / vocabs): each element's NAME is the key the engine reads,
// so an UNNAMED list carries no keys and names nothing. Only the genuinely
// empty cases -- NULL and a zero-length list -- normalize to an empty map; a
// non-empty list that produced no keys is a caller error and says so, naming
// the argument. Treating that case as "empty" instead discarded the whole
// argument in silence: a fully specified `targets = list(list(column = "area"))`
// reached the engine as zero targets and surfaced only much later, as an
// autograd error inside fit().
inline resolve_value_t* r_list_to_value_map(SEXP x, const char* what) {
    if (Rf_isNull(x) || Rf_xlength(x) == 0) return resolve_value_new_map();
    resolve_value_t* v = r_to_value(x);
    if (resolve_value_kind(v) != RESOLVE_VALUE_MAP) {
        resolve_value_free(v);
        Rcpp::stop("%s must be a NAMED list -- every element needs a name, "
                   "which is the key the engine reads. An unnamed list carries "
                   "no keys, so nothing in it would reach the engine.", what);
    }
    return v;
}

// Build a value MAP carrying an in-memory DataFrame (issue #22): an ordered
// (column name -> STRING_ARRAY) mapping. `cols` is a named list of character
// vectors (the caller coerces every column to character with NA -> "" so the
// cell semantics match a CSV). Unlike r_to_value, a length-1 column still
// becomes a STRING_ARRAY (never a STRING scalar), which ColumnTable requires.
inline resolve_value_t* r_charlist_to_value_map(List cols) {
    resolve_value_t* map = resolve_value_new_map();
    CharacterVector names = cols.names();
    for (R_xlen_t c = 0; c < cols.size(); ++c) {
        CharacterVector col = as<CharacterVector>(cols[c]);
        std::vector<std::string> s(static_cast<size_t>(col.size()));
        std::vector<const char*> p(static_cast<size_t>(col.size()));
        for (R_xlen_t i = 0; i < col.size(); ++i) {
            s[static_cast<size_t>(i)] =
                (col[i] == NA_STRING) ? std::string() : std::string(CHAR(col[i]));
            p[static_cast<size_t>(i)] = s[static_cast<size_t>(i)].c_str();
        }
        std::string key = (names.size() > c && names[c] != NA_STRING)
            ? std::string(CHAR(names[c])) : std::string();
        resolve_map_set_string_array(map, key.c_str(), p.data(),
                                     static_cast<int64_t>(p.size()));
    }
    return map;
}

// =============================================================================
// Method-argument tensor helpers: set R matrices / vectors into an input map.
// Float tensors are passed as double matrices/arrays (the engine casts to
// float32); integer id tensors as int64 matrices/arrays.
// =============================================================================

inline void map_set_num_matrix(resolve_value_t* map, const char* key, NumericMatrix x) {
    int nr = x.nrow(), nc = x.ncol();
    std::vector<double> buf(static_cast<size_t>(nr) * nc);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j) buf[static_cast<size_t>(i) * nc + j] = x(i, j);
    resolve_map_set_double_matrix(map, key, buf.data(), nr, nc);
}
inline void map_set_int_matrix(resolve_value_t* map, const char* key, IntegerMatrix x) {
    int nr = x.nrow(), nc = x.ncol();
    std::vector<int64_t> buf(static_cast<size_t>(nr) * nc);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j) buf[static_cast<size_t>(i) * nc + j] = x(i, j);
    resolve_map_set_int_matrix(map, key, buf.data(), nr, nc);
}
inline void map_set_num_vector(resolve_value_t* map, const char* key, NumericVector x) {
    std::vector<double> buf(x.begin(), x.end());
    resolve_map_set_double_array(map, key, buf.data(), static_cast<int64_t>(buf.size()));
}

// Optional-argument variants: only set the key when the argument is non-null.
inline void map_set_opt_num_matrix(resolve_value_t* map, const char* key, Nullable<NumericMatrix> x) {
    if (x.isNotNull()) map_set_num_matrix(map, key, as<NumericMatrix>(x));
}
inline void map_set_opt_int_matrix(resolve_value_t* map, const char* key, Nullable<IntegerMatrix> x) {
    if (x.isNotNull()) map_set_int_matrix(map, key, as<IntegerMatrix>(x));
}
inline void map_set_opt_num_vector(resolve_value_t* map, const char* key, Nullable<NumericVector> x) {
    if (x.isNotNull()) map_set_num_vector(map, key, as<NumericVector>(x));
}

// Fill the standard forward/predict input map (continuous required; the rest
// optional). Shared by model.forward / get_latent / forward_with_aux,
// trainer.predict, and the species/pool part of predictor.predict.
inline void fill_forward_inputs(
    resolve_value_t* in,
    NumericMatrix continuous,
    Nullable<IntegerMatrix> genus_ids, Nullable<IntegerMatrix> family_ids,
    Nullable<IntegerMatrix> species_ids, Nullable<NumericMatrix> species_vector,
    Nullable<IntegerMatrix> pool_genus_ids, Nullable<IntegerMatrix> pool_family_ids,
    Nullable<NumericMatrix> pool_weights, Nullable<IntegerMatrix> pool_mask,
    Nullable<NumericVector> pool_has_cover, Nullable<IntegerMatrix> categorical_ids) {
    map_set_num_matrix(in, "continuous", continuous);
    map_set_opt_int_matrix(in, "genus_ids", genus_ids);
    map_set_opt_int_matrix(in, "family_ids", family_ids);
    map_set_opt_int_matrix(in, "species_ids", species_ids);
    map_set_opt_num_matrix(in, "species_vector", species_vector);
    map_set_opt_int_matrix(in, "pool_genus_ids", pool_genus_ids);
    map_set_opt_int_matrix(in, "pool_family_ids", pool_family_ids);
    map_set_opt_num_matrix(in, "pool_weights", pool_weights);
    map_set_opt_int_matrix(in, "pool_mask", pool_mask);
    map_set_opt_num_vector(in, "pool_has_cover", pool_has_cover);
    map_set_opt_int_matrix(in, "categorical_ids", categorical_ids);
}

// =============================================================================
// categorical_vocab: the C facade returns a MAP { column -> MAP{string->int} }.
// Reshape each inner map into a NAMED IntegerVector (codes named by the source
// string), matching the old accessor's return shape.
// =============================================================================

inline List categorical_vocab_value_to_r(const resolve_value_t* v) {
    List out;
    if (v == nullptr || resolve_value_kind(v) != RESOLVE_VALUE_MAP) return out;
    int64_t ncol = resolve_map_size(v);
    for (int64_t c = 0; c < ncol; ++c) {
        const char* col = resolve_map_key_at(v, c);
        const resolve_value_t* inner = resolve_map_value_at(v, c);
        int64_t k = resolve_map_size(inner);
        IntegerVector codes(k);
        CharacterVector keys(k);
        for (int64_t i = 0; i < k; ++i) {
            keys[i] = resolve_map_key_at(inner, i);
            codes[i] = static_cast<int>(resolve_value_as_int(resolve_map_value_at(inner, i)));
        }
        codes.attr("names") = keys;
        out[col] = codes;
    }
    return out;
}

#endif // RCPP_COMMON_H
