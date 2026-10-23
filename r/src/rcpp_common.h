// rcpp_common.h - R <-> C ABI marshaling for the resolveR package.
//
// The R package reaches the engine through the flat C ABI in
// resolve/resolve_capi.h and marshals everything across the boundary as a
// `resolve_value_t` tree. This is the ONLY engine header included under r/src/;
// there is no `#include <torch/...>` anywhere in the package sources.
//
// resolve_c is bound at RUNTIME (resolve_capi_dynload.h), not linked at build
// time, so the package installs and R CMD checks with no backend present (the
// mlverse/torch model). The dynload header supplies the `resolve_*` symbols as
// forwarders; a call made before the backend is loaded would hit a NULL
// forwarder, so every constructor and factory reaching one first calls
// capi_require_loaded().
#ifndef RCPP_COMMON_H
#define RCPP_COMMON_H

#include <Rcpp.h>
#include "resolve_capi_dynload.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Rcpp;

// =============================================================================
// Error checking: turn a C-facade failure into an R error
// =============================================================================

// Raise a clean R error (never a NULL-pointer call) when the resolve_c backend
// has not been loaded.
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
    // Hand the tree over (to a map or list that takes ownership, or to a
    // caller), leaving this empty.
    resolve_value_t* release() {
        resolve_value_t* v = v_;
        v_ = nullptr;
        return v;
    }
private:
    resolve_value_t* v_;
};

// A freshly built value node, owned; stops with the engine's message when the
// builder failed.
inline ValuePtr owned_value(resolve_value_t* v) {
    capi_check(v);
    return ValuePtr(v);
}

// Put `child` into `map` under `key`, or append it to `list`; the container
// takes ownership whatever happens.
inline void map_put(resolve_value_t* map, const std::string& key, ValuePtr child) {
    capi_check_status(resolve_map_set_value(map, key.c_str(), child.release()));
}
inline void list_put(resolve_value_t* list, ValuePtr child) {
    capi_check_status(resolve_list_append_value(list, child.release()));
}

// Custom-deleter shared handle factory, so Rcpp's value-returning factories and
// module storage can copy a wrapper without double-freeing the C handle.
template <typename T>
inline std::shared_ptr<T> capi_own(T* handle, void (*deleter)(T*)) {
    capi_check(handle);
    return std::shared_ptr<T>(handle, deleter);
}

// =============================================================================
// value tree -> R object (recursive). Every result accessor goes through this.
// =============================================================================

inline RObject value_to_r(const resolve_value_t* v);

// An int64 value as R stores integers: R's int, whose smallest value is NA.
inline bool fits_r_integer(int64_t x) {
    return x > std::numeric_limits<int>::min() && x <= std::numeric_limits<int>::max();
}

// Integers past 2^53 have no exact double.
constexpr int64_t kExactDoubleLimit = int64_t{1} << 53;

// An int64 array or row-major matrix as an R integer vector / matrix when
// every value fits R's integer range, and as a double vector / matrix
// otherwise (exact up to 2^53: hash keys and record offsets keep their
// values). One warning, not one per element, when a value is past 2^53.
inline RObject int64_to_r(const int64_t* data, int64_t nrow, int64_t ncol, bool matrix) {
    const int64_t n = matrix ? nrow * ncol : nrow;
    bool fits = true;
    int64_t inexact = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!fits_r_integer(data[i])) fits = false;
        if (data[i] > kExactDoubleLimit || data[i] < -kExactDoubleLimit) ++inexact;
    }
    if (inexact > 0) {
        Rcpp::warning("resolve: %lld integer value(s) beyond 2^53 are rounded as doubles",
                      static_cast<long long>(inexact));
    }
    auto fill = [&](auto out) {
        for (int64_t i = 0; i < nrow; ++i) {
            for (int64_t j = 0; j < (matrix ? ncol : 1); ++j) {
                const int64_t source = matrix ? i * ncol + j : i;
                const R_xlen_t target = static_cast<R_xlen_t>(matrix ? j * nrow + i : i);
                out[target] = static_cast<typename decltype(out)::stored_type>(data[source]);
            }
        }
        return RObject(out);
    };
    if (matrix) {
        return fits ? fill(IntegerMatrix(static_cast<int>(nrow), static_cast<int>(ncol)))
                    : fill(NumericMatrix(static_cast<int>(nrow), static_cast<int>(ncol)));
    }
    return fits ? fill(IntegerVector(static_cast<R_xlen_t>(n)))
                : fill(NumericVector(static_cast<R_xlen_t>(n)));
}

inline RObject value_to_r(const resolve_value_t* v) {
    if (v == nullptr) return R_NilValue;
    switch (resolve_value_kind(v)) {
        case RESOLVE_VALUE_NULL:
            return R_NilValue;
        case RESOLVE_VALUE_BOOL:
            return wrap(static_cast<bool>(resolve_value_as_bool(v)));
        case RESOLVE_VALUE_INT: {
            const int64_t x = resolve_value_as_int(v);
            return int64_to_r(&x, 1, 1, false);
        }
        case RESOLVE_VALUE_DOUBLE:
            return wrap(resolve_value_as_double(v));
        case RESOLVE_VALUE_STRING:
            return wrap(std::string(resolve_value_as_string(v)));
        case RESOLVE_VALUE_INT_ARRAY: {
            int64_t n = 0;
            const int64_t* d = resolve_value_as_int_array(v, &n);
            return int64_to_r(d, n, 1, false);
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
                for (int64_t j = 0; j < nc; ++j) out(i, j) = d[i * nc + j];
            return out;
        }
        case RESOLVE_VALUE_INT_MATRIX: {
            int64_t nr = 0, nc = 0;
            const int64_t* d = resolve_value_as_int_matrix(v, &nr, &nc);
            return int64_to_r(d, nr, nc, true);
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
    const ValuePtr guard = owned_value(v);
    return value_to_r(guard.get());
}

// =============================================================================
// R object -> value tree (recursive), for config / roles / targets / schema
// lists. Every node is owned until it is handed to its parent, so an R error
// part-way (an unsupported element type) frees what was built.
// =============================================================================

inline std::string r_string(SEXP s) {
    return s == NA_STRING ? std::string() : std::string(CHAR(s));
}

// A numeric, logical or character element of an atomic vector as a scalar node.
inline ValuePtr r_element_to_value(SEXP x, R_xlen_t i) {
    switch (TYPEOF(x)) {
        case INTSXP: return owned_value(resolve_value_new_int(INTEGER(x)[i]));
        case REALSXP: return owned_value(resolve_value_new_double(REAL(x)[i]));
        case LGLSXP: return owned_value(resolve_value_new_bool(LOGICAL(x)[i] == TRUE ? 1 : 0));
        default: return owned_value(resolve_value_new_string(r_string(STRING_ELT(x, i)).c_str()));
    }
}

inline resolve_value_t* r_to_value(SEXP x) {
    if (Rf_isNull(x)) return resolve_value_new_null();

    // A NAMED atomic vector (a classification class_mapping written as
    // c(forest = 0L, grass = 1L)) is semantically a map: the C-ABI parsers
    // read a mapping from a map, so it travels as one.
    const int atype = TYPEOF(x);
    if (atype == INTSXP || atype == REALSXP || atype == STRSXP || atype == LGLSXP) {
        SEXP nm = Rf_getAttrib(x, R_NamesSymbol);
        if (nm != R_NilValue && Rf_xlength(x) > 0) {
            ValuePtr map = owned_value(resolve_value_new_map());
            for (R_xlen_t i = 0; i < Rf_xlength(x); ++i) {
                map_put(map.get(), r_string(STRING_ELT(nm, i)), r_element_to_value(x, i));
            }
            return map.release();
        }
    }

    switch (atype) {
        case LGLSXP: {
            LogicalVector v(x);
            if (v.size() == 1) return resolve_value_new_bool(v[0] == TRUE ? 1 : 0);
            std::vector<int64_t> a(v.size());
            for (R_xlen_t i = 0; i < v.size(); ++i) a[i] = (v[i] == TRUE) ? 1 : 0;
            return resolve_value_new_int_array(a.data(), static_cast<int64_t>(a.size()));
        }
        case INTSXP: {
            IntegerVector v(x);
            if (v.size() == 1) return resolve_value_new_int(v[0]);
            std::vector<int64_t> a(v.begin(), v.end());
            return resolve_value_new_int_array(a.data(), static_cast<int64_t>(a.size()));
        }
        case REALSXP: {
            NumericVector v(x);
            if (v.size() == 1) return resolve_value_new_double(v[0]);
            std::vector<double> a(v.begin(), v.end());
            return resolve_value_new_double_array(a.data(), static_cast<int64_t>(a.size()));
        }
        case STRSXP: {
            CharacterVector v(x);
            if (v.size() == 1) return resolve_value_new_string(r_string(v[0]).c_str());
            std::vector<std::string> s(v.size());
            std::vector<const char*> p(v.size());
            for (R_xlen_t i = 0; i < v.size(); ++i) {
                s[i] = r_string(v[i]);
                p[i] = s[i].c_str();
            }
            return resolve_value_new_string_array(p.data(), static_cast<int64_t>(p.size()));
        }
        case VECSXP: {
            List v(x);
            if (v.hasAttribute("names")) {
                ValuePtr map = owned_value(resolve_value_new_map());
                CharacterVector names = v.names();
                for (R_xlen_t i = 0; i < v.size(); ++i) {
                    map_put(map.get(), r_string(names[i]), owned_value(r_to_value(v[i])));
                }
                return map.release();
            }
            ValuePtr list = owned_value(resolve_value_new_list());
            for (R_xlen_t i = 0; i < v.size(); ++i) list_put(list.get(), owned_value(r_to_value(v[i])));
            return list.release();
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
// the argument, rather than reaching the engine as nothing.
inline resolve_value_t* r_list_to_value_map(SEXP x, const char* what) {
    if (Rf_isNull(x) || Rf_xlength(x) == 0) return resolve_value_new_map();
    ValuePtr v = owned_value(r_to_value(x));
    if (resolve_value_kind(v.get()) != RESOLVE_VALUE_MAP) {
        Rcpp::stop("%s must be a NAMED list -- every element needs a name, "
                   "which is the key the engine reads. An unnamed list carries "
                   "no keys, so nothing in it would reach the engine.", what);
    }
    return v.release();
}

// A table argument of the dataset loader: NULL (none), a single string (the
// path of a CSV file), or a named list of character vectors (an in-memory
// table, every cell a string with NA -> "", the cell semantics of a CSV).
inline resolve_value_t* r_table_to_value(SEXP x, const char* what) {
    if (Rf_isNull(x)) return resolve_value_new_null();
    if (TYPEOF(x) == STRSXP && Rf_xlength(x) == 1) {
        return resolve_value_new_string(r_string(STRING_ELT(x, 0)).c_str());
    }
    if (TYPEOF(x) != VECSXP) {
        Rcpp::stop("%s must be a CSV path or a list of character columns", what);
    }
    List cols(x);
    CharacterVector names = cols.names();
    ValuePtr map = owned_value(resolve_value_new_map());
    for (R_xlen_t c = 0; c < cols.size(); ++c) {
        CharacterVector col = as<CharacterVector>(cols[c]);
        std::vector<std::string> s(static_cast<size_t>(col.size()));
        std::vector<const char*> p(static_cast<size_t>(col.size()));
        for (R_xlen_t i = 0; i < col.size(); ++i) {
            s[static_cast<size_t>(i)] = r_string(col[i]);
            p[static_cast<size_t>(i)] = s[static_cast<size_t>(i)].c_str();
        }
        const std::string key = names.size() > c ? r_string(names[c]) : std::string();
        capi_check_status(resolve_map_set_string_array(map.get(), key.c_str(), p.data(),
                                                       static_cast<int64_t>(p.size())));
    }
    return map.release();
}

// =============================================================================
// Method-argument tensors: R matrices / vectors into an input map. Float
// tensors travel as double matrices / arrays (the engine casts to float32),
// integer ids as int64 matrices / arrays.
// =============================================================================

// An R matrix as a row-major buffer of `Elem`.
template <typename Elem, typename RMatrix>
std::vector<Elem> row_major(const RMatrix& x) {
    const int nr = x.nrow(), nc = x.ncol();
    std::vector<Elem> buf(static_cast<size_t>(nr) * nc);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j) buf[static_cast<size_t>(i) * nc + j] = static_cast<Elem>(x(i, j));
    return buf;
}

inline void map_set_num_matrix(resolve_value_t* map, const char* key, NumericMatrix x) {
    const std::vector<double> buf = row_major<double>(x);
    capi_check_status(resolve_map_set_double_matrix(map, key, buf.data(), x.nrow(), x.ncol()));
}
inline void map_set_int_matrix(resolve_value_t* map, const char* key, IntegerMatrix x) {
    const std::vector<int64_t> buf = row_major<int64_t>(x);
    capi_check_status(resolve_map_set_int_matrix(map, key, buf.data(), x.nrow(), x.ncol()));
}
inline void map_set_num_vector(resolve_value_t* map, const char* key, NumericVector x) {
    std::vector<double> buf(x.begin(), x.end());
    capi_check_status(resolve_map_set_double_array(map, key, buf.data(),
                                                   static_cast<int64_t>(buf.size())));
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

// Fill the composition and categorical part of a plot-input map (every key
// optional): the inputs every model, trainer and predictor entry point reads
// besides its continuous block.
inline void fill_composition_inputs(
    resolve_value_t* in, Nullable<IntegerMatrix> genus_ids, Nullable<IntegerMatrix> family_ids,
    Nullable<IntegerMatrix> species_ids, Nullable<NumericMatrix> species_vector,
    Nullable<IntegerMatrix> pool_genus_ids, Nullable<IntegerMatrix> pool_family_ids,
    Nullable<NumericMatrix> pool_weights, Nullable<IntegerMatrix> pool_mask,
    Nullable<NumericVector> pool_has_cover, Nullable<IntegerMatrix> categorical_ids) {
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

// The model / trainer input map: the assembled continuous block plus the
// composition inputs.
inline ValuePtr forward_inputs(
    NumericMatrix continuous, Nullable<IntegerMatrix> genus_ids, Nullable<IntegerMatrix> family_ids,
    Nullable<IntegerMatrix> species_ids, Nullable<NumericMatrix> species_vector,
    Nullable<IntegerMatrix> pool_genus_ids, Nullable<IntegerMatrix> pool_family_ids,
    Nullable<NumericMatrix> pool_weights, Nullable<IntegerMatrix> pool_mask,
    Nullable<NumericVector> pool_has_cover, Nullable<IntegerMatrix> categorical_ids) {
    ValuePtr in = owned_value(resolve_value_new_map());
    map_set_num_matrix(in.get(), "continuous", continuous);
    fill_composition_inputs(in.get(), genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask,
                            pool_has_cover, categorical_ids);
    return in;
}

// The predictor input map: the unscaled continuous parts the predictor
// standardises with the checkpoint's scalers (each optional: a dataset without
// coordinates, covariates or a hash encoding returns NULL for that part), the
// unknown-species columns, and the composition inputs.
inline ValuePtr predictor_inputs(
    Nullable<NumericMatrix> coordinates, Nullable<NumericMatrix> covariates,
    Nullable<NumericMatrix> hash_embedding, Nullable<IntegerMatrix> species_ids,
    Nullable<NumericMatrix> species_vector, Nullable<IntegerMatrix> genus_ids,
    Nullable<IntegerMatrix> family_ids, Nullable<NumericVector> unknown_fraction,
    Nullable<NumericVector> unknown_count, Nullable<IntegerMatrix> pool_genus_ids,
    Nullable<IntegerMatrix> pool_family_ids, Nullable<NumericMatrix> pool_weights,
    Nullable<IntegerMatrix> pool_mask, Nullable<NumericVector> pool_has_cover,
    Nullable<IntegerMatrix> categorical_ids) {
    ValuePtr in = owned_value(resolve_value_new_map());
    map_set_opt_num_matrix(in.get(), "coordinates", coordinates);
    map_set_opt_num_matrix(in.get(), "covariates", covariates);
    map_set_opt_num_matrix(in.get(), "hash_embedding", hash_embedding);
    map_set_opt_num_vector(in.get(), "unknown_fraction", unknown_fraction);
    map_set_opt_num_vector(in.get(), "unknown_count", unknown_count);
    fill_composition_inputs(in.get(), genus_ids, family_ids, species_ids, species_vector,
                            pool_genus_ids, pool_family_ids, pool_weights, pool_mask,
                            pool_has_cover, categorical_ids);
    return in;
}

// =============================================================================
// categorical_vocab: the C facade returns a MAP { column -> MAP{string->int} }.
// Each inner map becomes a NAMED IntegerVector (codes named by the source
// string).
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
