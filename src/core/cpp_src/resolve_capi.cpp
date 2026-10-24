// resolve_capi.cpp - the flat C ABI over the engine (resolve_capi.h).
//
// The one translation unit between the engine's C++ ABI and the C ABI the R
// package (and any other client built with another C++ toolchain) calls. All
// tensor <-> raw-buffer marshaling, all config / enum parsing and all
// result-struct -> value-tree conversion live here, compiled with the engine.

#define RESOLVE_CAPI_BUILD 1
#include "resolve/resolve_capi.h"
#include "resolve/resolve.hpp"
#include "resolve/config_registry.hpp"
#include "resolve/fuzzy.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace resolve;

// ============================================================================
// Error handling: thread-local last-error + exception -> C boundary translation
// ============================================================================

namespace {

thread_local std::string g_last_error;

void set_error(const char* msg) { g_last_error = msg ? msg : "unknown error"; }

}  // namespace

// Wrap a body that returns a pointer (NULL on error). Variadic so the body may
// contain top-level commas (e.g. multi-variable declarations) without being
// split into multiple macro arguments.
#define CAPI_BODY_PTR(...)                                                   \
    try {                                                                    \
        __VA_ARGS__                                                          \
    } catch (const std::exception& e) {                                      \
        set_error(e.what());                                                 \
        return nullptr;                                                      \
    } catch (...) {                                                          \
        set_error("unknown C++ exception");                                  \
        return nullptr;                                                      \
    }

// Wrap a body that returns int (0 ok, -1 error).
#define CAPI_BODY_INT(...)                                                   \
    try {                                                                    \
        __VA_ARGS__                                                          \
    } catch (const std::exception& e) {                                      \
        set_error(e.what());                                                 \
        return -1;                                                           \
    } catch (...) {                                                          \
        set_error("unknown C++ exception");                                  \
        return -1;                                                           \
    }

extern "C" const char* resolve_last_error(void) { return g_last_error.c_str(); }

// ============================================================================
// resolve_value: the tagged marshaling tree
// ============================================================================

struct resolve_value {
    resolve_value_kind_t kind = RESOLVE_VALUE_NULL;

    bool b = false;
    int64_t i = 0;
    double d = 0.0;
    std::string s;

    std::vector<int64_t> iarr;        // INT_ARRAY / INT_MATRIX data (row-major)
    std::vector<double> darr;         // DOUBLE_ARRAY / DOUBLE_MATRIX data
    std::vector<std::string> sarr;    // STRING_ARRAY
    int64_t nrow = 0, ncol = 0;       // matrix dims

    std::vector<std::string> keys;            // MAP keys
    std::vector<resolve_value*> vals;         // MAP values (owned)
    std::vector<resolve_value*> items;        // LIST items (owned)

    ~resolve_value() {
        for (auto* v : vals) delete v;
        for (auto* v : items) delete v;
    }
};

// ----------------------------------------------------------------------------
// Internal builders (assemble result trees on the C++ side)
// ----------------------------------------------------------------------------
//
// The builders throw like any allocation (std::bad_alloc) or on a malformed
// argument (std::invalid_argument). Every caller runs inside a CAPI_BODY_*
// wrapper, so the exception becomes a NULL / -1 return with its message in
// resolve_last_error() and never crosses the boundary.

namespace {

// Owns a value tree until release(): a converter that throws part-way frees
// the tree it was building instead of leaking it.
struct ValueGuard {
    resolve_value* p;
    explicit ValueGuard(resolve_value* v) noexcept : p(v) {}
    ~ValueGuard() { delete p; }
    ValueGuard(const ValueGuard&) = delete;
    ValueGuard& operator=(const ValueGuard&) = delete;
    resolve_value* release() noexcept {
        resolve_value* q = p;
        p = nullptr;
        return q;
    }
};

resolve_value* v_new(resolve_value_kind_t k) {
    auto* v = new resolve_value();
    v->kind = k;
    return v;
}
resolve_value* v_null() { return v_new(RESOLVE_VALUE_NULL); }
resolve_value* v_bool(bool x) { auto* v = v_new(RESOLVE_VALUE_BOOL); v->b = x; return v; }
resolve_value* v_int(int64_t x) { auto* v = v_new(RESOLVE_VALUE_INT); v->i = x; return v; }
resolve_value* v_double(double x) { auto* v = v_new(RESOLVE_VALUE_DOUBLE); v->d = x; return v; }
resolve_value* v_string(std::string x) {
    ValueGuard g(v_new(RESOLVE_VALUE_STRING));
    g.p->s = std::move(x);
    return g.release();
}
resolve_value* v_string_array(std::vector<std::string> x) {
    ValueGuard g(v_new(RESOLVE_VALUE_STRING_ARRAY));
    g.p->sarr = std::move(x);
    return g.release();
}
// A string array copied from `n` C strings (a NULL entry reads as "").
resolve_value* v_string_array(const char* const* v, int64_t n) {
    if (n < 0) throw std::invalid_argument("a value node cannot have a negative size");
    if (n > 0 && v == nullptr) throw std::invalid_argument("a string array was given no data");
    std::vector<std::string> strings;
    strings.reserve(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) strings.emplace_back(v[j] ? v[j] : "");
    return v_string_array(std::move(strings));
}
resolve_value* v_map() { return v_new(RESOLVE_VALUE_MAP); }
resolve_value* v_list() { return v_new(RESOLVE_VALUE_LIST); }

// The element storage of a numeric node: int64 for the INT kinds, double for
// the DOUBLE kinds.
template <typename T>
std::vector<T>& numeric_storage(resolve_value* v);
template <>
std::vector<int64_t>& numeric_storage<int64_t>(resolve_value* v) { return v->iarr; }
template <>
std::vector<double>& numeric_storage<double>(resolve_value* v) { return v->darr; }

// A numeric node of `kind` holding `n` elements copied from `data` (converted
// to the node's element type), with matrix dimensions nrow x ncol (0 x 0 for an
// array). The caller's buffer stays the caller's.
template <typename Elem, typename Src>
resolve_value* v_numeric(resolve_value_kind_t kind, const Src* data, int64_t n,
                         int64_t nrow = 0, int64_t ncol = 0) {
    if (n < 0) throw std::invalid_argument("a value node cannot have a negative size");
    if (n > 0 && data == nullptr) {
        throw std::invalid_argument("a value node of " + std::to_string(n) +
                                    " elements was given no data");
    }
    ValueGuard g(v_new(kind));
    g.p->nrow = nrow;
    g.p->ncol = ncol;
    numeric_storage<Elem>(g.p).assign(data, data + n);
    return g.release();
}

resolve_value* v_int_array(const std::vector<int64_t>& x) {
    return v_numeric<int64_t>(RESOLVE_VALUE_INT_ARRAY, x.data(), static_cast<int64_t>(x.size()));
}
// Every engine-side float vector reaching the boundary is std::vector<float>;
// the DOUBLE_ARRAY kind is the wire representation, not the source type.
resolve_value* v_double_array(const std::vector<float>& x) {
    return v_numeric<double>(RESOLVE_VALUE_DOUBLE_ARRAY, x.data(), static_cast<int64_t>(x.size()));
}

// Rows times columns of a matrix node, refusing a negative dimension and a
// product that overflows int64.
int64_t matrix_size(int64_t nrow, int64_t ncol) {
    if (nrow < 0 || ncol < 0) throw std::invalid_argument("a matrix cannot have a negative dimension");
    if (ncol > 0 && nrow > std::numeric_limits<int64_t>::max() / ncol) {
        throw std::invalid_argument("a matrix of that many elements cannot be represented");
    }
    return nrow * ncol;
}

// Put `child` into map `m` under `key`, taking ownership of it: on any failure
// the child is freed and the map unchanged.
void v_put(resolve_value* m, std::string key, resolve_value* child) {
    ValueGuard owned(child);
    m->vals.reserve(m->vals.size() + 1);
    m->keys.push_back(std::move(key));
    m->vals.push_back(owned.release());
}
void v_append(resolve_value* l, resolve_value* child) {
    ValueGuard owned(child);
    l->items.push_back(owned.release());
}

// `v` is a node of `kind`; `what` names the call in the error otherwise.
void require_kind(const resolve_value* v, resolve_value_kind_t kind, const char* what) {
    if (v == nullptr) throw std::invalid_argument(std::string(what) + ": the value is NULL");
    if (v->kind != kind) {
        throw std::invalid_argument(std::string(what) + ": the value is not a " +
                                    (kind == RESOLVE_VALUE_MAP ? "map" : "list"));
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// C API: value construction
// ----------------------------------------------------------------------------

namespace {

// Put the node `make()` builds into `map` under `key`: 0, or -1 with the
// message in resolve_last_error().
template <typename Make>
int map_set(resolve_value* map, const char* key, Make&& make) {
    CAPI_BODY_INT({
        require_kind(map, RESOLVE_VALUE_MAP, "map setter");
        if (key == nullptr) throw std::invalid_argument("map setter: the key is NULL");
        v_put(map, key, make());
        return 0;
    })
}

}  // namespace

extern "C" {

resolve_value_t* resolve_value_new_null(void) { CAPI_BODY_PTR({ return v_null(); }) }
resolve_value_t* resolve_value_new_map(void) { CAPI_BODY_PTR({ return v_map(); }) }
resolve_value_t* resolve_value_new_list(void) { CAPI_BODY_PTR({ return v_list(); }) }
void resolve_value_free(resolve_value_t* v) { delete v; }

resolve_value_t* resolve_value_new_bool(int v) { CAPI_BODY_PTR({ return v_bool(v != 0); }) }
resolve_value_t* resolve_value_new_int(int64_t v) { CAPI_BODY_PTR({ return v_int(v); }) }
resolve_value_t* resolve_value_new_double(double v) { CAPI_BODY_PTR({ return v_double(v); }) }
resolve_value_t* resolve_value_new_string(const char* v) {
    CAPI_BODY_PTR({ return v_string(v ? v : ""); })
}
resolve_value_t* resolve_value_new_int_array(const int64_t* v, int64_t n) {
    CAPI_BODY_PTR({ return v_numeric<int64_t>(RESOLVE_VALUE_INT_ARRAY, v, n); })
}
resolve_value_t* resolve_value_new_double_array(const double* v, int64_t n) {
    CAPI_BODY_PTR({ return v_numeric<double>(RESOLVE_VALUE_DOUBLE_ARRAY, v, n); })
}
resolve_value_t* resolve_value_new_string_array(const char* const* v, int64_t n) {
    CAPI_BODY_PTR({ return v_string_array(v, n); })
}
resolve_value_t* resolve_value_new_double_matrix(const double* v, int64_t nrow, int64_t ncol) {
    CAPI_BODY_PTR({
        return v_numeric<double>(RESOLVE_VALUE_DOUBLE_MATRIX, v, matrix_size(nrow, ncol), nrow, ncol);
    })
}
resolve_value_t* resolve_value_new_int_matrix(const int64_t* v, int64_t nrow, int64_t ncol) {
    CAPI_BODY_PTR({
        return v_numeric<int64_t>(RESOLVE_VALUE_INT_MATRIX, v, matrix_size(nrow, ncol), nrow, ncol);
    })
}

int resolve_map_set_null(resolve_value_t* m, const char* k) {
    return map_set(m, k, [] { return v_null(); });
}
int resolve_map_set_bool(resolve_value_t* m, const char* k, int v) {
    return map_set(m, k, [&] { return v_bool(v != 0); });
}
int resolve_map_set_int(resolve_value_t* m, const char* k, int64_t v) {
    return map_set(m, k, [&] { return v_int(v); });
}
int resolve_map_set_double(resolve_value_t* m, const char* k, double v) {
    return map_set(m, k, [&] { return v_double(v); });
}
int resolve_map_set_string(resolve_value_t* m, const char* k, const char* v) {
    return map_set(m, k, [&] { return v_string(v ? v : ""); });
}
int resolve_map_set_int_array(resolve_value_t* m, const char* k, const int64_t* v, int64_t n) {
    return map_set(m, k, [&] { return v_numeric<int64_t>(RESOLVE_VALUE_INT_ARRAY, v, n); });
}
int resolve_map_set_double_array(resolve_value_t* m, const char* k, const double* v, int64_t n) {
    return map_set(m, k, [&] { return v_numeric<double>(RESOLVE_VALUE_DOUBLE_ARRAY, v, n); });
}
int resolve_map_set_string_array(resolve_value_t* m, const char* k, const char* const* v, int64_t n) {
    return map_set(m, k, [&] { return v_string_array(v, n); });
}
int resolve_map_set_double_matrix(resolve_value_t* m, const char* k, const double* v,
                                  int64_t nrow, int64_t ncol) {
    return map_set(m, k, [&] {
        return v_numeric<double>(RESOLVE_VALUE_DOUBLE_MATRIX, v, matrix_size(nrow, ncol), nrow, ncol);
    });
}
int resolve_map_set_int_matrix(resolve_value_t* m, const char* k, const int64_t* v,
                               int64_t nrow, int64_t ncol) {
    return map_set(m, k, [&] {
        return v_numeric<int64_t>(RESOLVE_VALUE_INT_MATRIX, v, matrix_size(nrow, ncol), nrow, ncol);
    });
}
int resolve_map_set_value(resolve_value_t* m, const char* k, resolve_value_t* child) {
    ValueGuard owned(child);
    return map_set(m, k, [&] {
        if (owned.p == nullptr) throw std::invalid_argument("resolve_map_set_value: the child is NULL");
        return owned.release();
    });
}
int resolve_list_append_value(resolve_value_t* l, resolve_value_t* child) {
    ValueGuard owned(child);
    CAPI_BODY_INT({
        require_kind(l, RESOLVE_VALUE_LIST, "resolve_list_append_value");
        if (owned.p == nullptr) throw std::invalid_argument("resolve_list_append_value: the child is NULL");
        v_append(l, owned.release());
        return 0;
    })
}

// ----------------------------------------------------------------------------
// C API: value reading
// ----------------------------------------------------------------------------

resolve_value_kind_t resolve_value_kind(const resolve_value_t* v) {
    return v ? v->kind : RESOLVE_VALUE_NULL;
}
int         resolve_value_as_bool  (const resolve_value_t* v) { return v && v->b ? 1 : 0; }
int64_t     resolve_value_as_int   (const resolve_value_t* v) { return v ? v->i : 0; }
double      resolve_value_as_double(const resolve_value_t* v) { return v ? v->d : 0.0; }
const char* resolve_value_as_string(const resolve_value_t* v) { return v ? v->s.c_str() : ""; }

const int64_t* resolve_value_as_int_array(const resolve_value_t* v, int64_t* n) {
    if (n) *n = v ? static_cast<int64_t>(v->iarr.size()) : 0;
    return (v && !v->iarr.empty()) ? v->iarr.data() : nullptr;
}
const double* resolve_value_as_double_array(const resolve_value_t* v, int64_t* n) {
    if (n) *n = v ? static_cast<int64_t>(v->darr.size()) : 0;
    return (v && !v->darr.empty()) ? v->darr.data() : nullptr;
}
const double* resolve_value_as_double_matrix(const resolve_value_t* v, int64_t* nrow, int64_t* ncol) {
    if (nrow) *nrow = v ? v->nrow : 0;
    if (ncol) *ncol = v ? v->ncol : 0;
    return (v && !v->darr.empty()) ? v->darr.data() : nullptr;
}
const int64_t* resolve_value_as_int_matrix(const resolve_value_t* v, int64_t* nrow, int64_t* ncol) {
    if (nrow) *nrow = v ? v->nrow : 0;
    if (ncol) *ncol = v ? v->ncol : 0;
    return (v && !v->iarr.empty()) ? v->iarr.data() : nullptr;
}
int64_t resolve_value_string_array_size(const resolve_value_t* v) {
    return v ? static_cast<int64_t>(v->sarr.size()) : 0;
}
const char* resolve_value_string_at(const resolve_value_t* v, int64_t i) {
    if (!v || i < 0 || i >= static_cast<int64_t>(v->sarr.size())) return "";
    return v->sarr[i].c_str();
}

int64_t resolve_map_size(const resolve_value_t* m) {
    return m ? static_cast<int64_t>(m->keys.size()) : 0;
}
const char* resolve_map_key_at(const resolve_value_t* m, int64_t i) {
    if (!m || i < 0 || i >= static_cast<int64_t>(m->keys.size())) return "";
    return m->keys[i].c_str();
}
const resolve_value_t* resolve_map_value_at(const resolve_value_t* m, int64_t i) {
    if (!m || i < 0 || i >= static_cast<int64_t>(m->vals.size())) return nullptr;
    return m->vals[i];
}
const resolve_value_t* resolve_map_get(const resolve_value_t* m, const char* key) {
    if (!m || !key) return nullptr;
    for (size_t j = 0; j < m->keys.size(); ++j) {
        if (m->keys[j] == key) return m->vals[j];
    }
    return nullptr;
}
int64_t resolve_list_size(const resolve_value_t* l) {
    return l ? static_cast<int64_t>(l->items.size()) : 0;
}
const resolve_value_t* resolve_list_at(const resolve_value_t* l, int64_t i) {
    if (!l || i < 0 || i >= static_cast<int64_t>(l->items.size())) return nullptr;
    return l->items[i];
}

}  // extern "C"

// ============================================================================
// Internal helpers: value-map readers
// ============================================================================
//
// An R or Python caller hands whatever its language has -- R has no int64, so
// a count arrives as a double, and a float weight arrives as a double -- so the
// numeric readers convert between the numeric kinds. A node of any other kind
// where a number, a string or a list is read is a caller error and throws,
// naming `what`.

namespace {

const resolve_value* vget(const resolve_value* m, const char* k) {
    return resolve_map_get(m, k);
}
bool vhas(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return v != nullptr && v->kind != RESOLVE_VALUE_NULL;
}

[[noreturn]] void wrong_kind(const std::string& what, const char* expected) {
    throw std::invalid_argument(what + ": expected " + expected);
}

// A numeric or boolean scalar node as T.
template <typename T>
T vnum(const resolve_value* v, const std::string& what) {
    if (v != nullptr) {
        switch (v->kind) {
            case RESOLVE_VALUE_INT: return static_cast<T>(v->i);
            case RESOLVE_VALUE_DOUBLE: return static_cast<T>(v->d);
            case RESOLVE_VALUE_BOOL: return static_cast<T>(v->b ? 1 : 0);
            default: break;
        }
    }
    wrong_kind(what, "a number");
}
template <typename T>
T vnum(const resolve_value* m, const char* k) { return vnum<T>(vget(m, k), std::string(k)); }

bool vbool(const resolve_value* v, const std::string& what) { return vnum<int64_t>(v, what) != 0; }
bool vbool(const resolve_value* m, const char* k) { return vbool(vget(m, k), std::string(k)); }

const std::string& vstr(const resolve_value* v, const std::string& what) {
    if (v == nullptr || v->kind != RESOLVE_VALUE_STRING) wrong_kind(what, "a string");
    return v->s;
}
std::string vstr(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return v == nullptr ? std::string() : vstr(v, std::string(k));
}

// A numeric array node (or one numeric scalar, or a list of numeric scalars)
// as a vector of T.
template <typename T>
std::vector<T> vnum_vec(const resolve_value* v, const std::string& what) {
    std::vector<T> out;
    if (v == nullptr || v->kind == RESOLVE_VALUE_NULL) return out;
    switch (v->kind) {
        case RESOLVE_VALUE_INT_ARRAY:
            out.reserve(v->iarr.size());
            for (int64_t x : v->iarr) out.push_back(static_cast<T>(x));
            break;
        case RESOLVE_VALUE_DOUBLE_ARRAY:
            out.reserve(v->darr.size());
            for (double x : v->darr) out.push_back(static_cast<T>(x));
            break;
        case RESOLVE_VALUE_LIST:
            out.reserve(v->items.size());
            for (const resolve_value* item : v->items) out.push_back(vnum<T>(item, what));
            break;
        default: out.push_back(vnum<T>(v, what)); break;
    }
    return out;
}
template <typename T>
std::vector<T> vnum_vec(const resolve_value* m, const char* k) {
    return vnum_vec<T>(vget(m, k), std::string(k));
}

// A list of names: a string array, a single string, or a list of strings (an R
// list or a JSON array).
std::vector<std::string> vstr_vec(const resolve_value* v, const std::string& what) {
    std::vector<std::string> out;
    if (!v || v->kind == RESOLVE_VALUE_NULL) return out;
    if (v->kind == RESOLVE_VALUE_STRING_ARRAY) return v->sarr;
    if (v->kind == RESOLVE_VALUE_STRING) {
        out.push_back(v->s);
        return out;
    }
    if (v->kind == RESOLVE_VALUE_LIST) {
        for (const resolve_value* item : v->items) out.push_back(vstr(item, what));
        return out;
    }
    wrong_kind(what, "a string or an array of strings");
}
std::vector<std::string> vstr_vec(const resolve_value* m, const char* k) {
    return vstr_vec(vget(m, k), std::string(k));
}

// MAP (ordered column name -> STRING_ARRAY) -> ColumnTable. The in-memory
// DataFrame carrier: a data.frame with every cell stringified.
// ColumnTable's constructor validates equal-length columns and unique names.
ColumnTable value_to_column_table(const resolve_value* v, const char* what) {
    if (!v || v->kind != RESOLVE_VALUE_MAP) {
        throw std::invalid_argument(std::string(what) +
            " must be a MAP value tree (ordered column name -> string array)");
    }
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> cols;
    names.reserve(v->keys.size());
    cols.reserve(v->vals.size());
    for (size_t c = 0; c < v->keys.size(); ++c) {
        const resolve_value* colv = v->vals[c];
        if (!colv || colv->kind != RESOLVE_VALUE_STRING_ARRAY) {
            throw std::invalid_argument(std::string(what) + " column '" +
                v->keys[c] + "' must be a string array");
        }
        names.push_back(v->keys[c]);
        cols.push_back(colv->sarr);
    }
    return ColumnTable(std::move(names), std::move(cols));
}

// Throws std::invalid_argument for a key of map `v` that is not in
// `accepted`: an unrecognized key is a caller typo (`name` / `type` where the
// parser reads `column` / `task`, `species_encodng` for `species_encoding`), and
// ignoring it means the value the caller believed they set never reaches the
// engine. A key given twice (an R list concatenated with c()) is refused too,
// since only one of its values could be read.
template <typename Names>
void reject_unknown_keys(const resolve_value* v, const std::string& what, const Names& accepted) {
    if (!v || v->kind != RESOLVE_VALUE_MAP) return;
    for (size_t i = 0; i < v->keys.size(); ++i) {
        const auto first = v->keys.begin() + static_cast<std::ptrdiff_t>(i);
        if (std::find(v->keys.begin(), first, v->keys[i]) != first) {
            throw std::invalid_argument(what + ": the key '" + v->keys[i] + "' is given twice");
        }
    }
    for (const std::string& key : v->keys) {
        const bool known = std::any_of(std::begin(accepted), std::end(accepted),
                                       [&](const auto& a) { return key == a; });
        if (known) continue;
        std::string msg = what + ": unknown key '" + key + "'. Accepted keys: ";
        bool first = true;
        for (const auto& a : accepted) {
            if (!first) msg += ", ";
            msg += a;
            first = false;
        }
        throw std::invalid_argument(msg);
    }
}
void reject_unknown_keys(const resolve_value* v, const std::string& what,
                         std::initializer_list<const char*> accepted) {
    reject_unknown_keys<std::initializer_list<const char*>>(v, what, accepted);
}

}  // namespace

// ============================================================================
// Internal helpers: value <-> torch::Tensor
// ============================================================================

namespace {

// A numeric array or matrix node as a tensor of `dtype` (a matrix node gives
// (nrow, ncol), an array node 1-D), converted element by element; an absent or
// NULL-kind node gives an undefined tensor.
torch::Tensor value_to_tensor(const resolve_value* v, torch::Dtype dtype, const std::string& what) {
    if (!v || v->kind == RESOLVE_VALUE_NULL) return torch::Tensor();
    const bool matrix = v->kind == RESOLVE_VALUE_DOUBLE_MATRIX || v->kind == RESOLVE_VALUE_INT_MATRIX;
    const bool integer = v->kind == RESOLVE_VALUE_INT_MATRIX || v->kind == RESOLVE_VALUE_INT_ARRAY;
    if (!matrix && !integer && v->kind != RESOLVE_VALUE_DOUBLE_ARRAY) {
        wrong_kind(what, "a numeric array or matrix");
    }
    const int64_t n = integer ? static_cast<int64_t>(v->iarr.size())
                              : static_cast<int64_t>(v->darr.size());
    const std::vector<int64_t> shape = matrix ? std::vector<int64_t>{v->nrow, v->ncol}
                                              : std::vector<int64_t>{n};
    if (n == 0) return torch::empty(shape, torch::TensorOptions().dtype(dtype));
    void* data = integer ? static_cast<void*>(const_cast<int64_t*>(v->iarr.data()))
                         : static_cast<void*>(const_cast<double*>(v->darr.data()));
    return torch::from_blob(data, shape, integer ? torch::kInt64 : torch::kFloat64)
        .to(dtype, /*non_blocking=*/false, /*copy=*/true);
}

// The tensor a map carries under `k`, or an undefined tensor when it carries
// none.
torch::Tensor opt_tensor(const resolve_value* m, const char* k, torch::Dtype dtype) {
    return value_to_tensor(vget(m, k), dtype, k);
}

// A caller's buffer of `n` doubles as a float32 tensor (the metric entry
// points take their inputs this way; R has no float vector).
torch::Tensor f32_tensor_from_doubles(const double* src, int64_t n, const char* what) {
    if (n < 0) throw std::invalid_argument(std::string(what) + ": negative length");
    if (n > 0 && src == nullptr) throw std::invalid_argument(std::string(what) + " is NULL");
    if (n == 0) return torch::empty({0}, torch::kFloat32);
    return torch::from_blob(const_cast<double*>(src), {n}, torch::kFloat64)
        .to(torch::kFloat32, /*non_blocking=*/false, /*copy=*/true);
}

// A tensor as a numeric node of `kind`: (rows, cols) for a matrix kind (a 1-D
// tensor one column), flattened for an array kind; int64 for the INT kinds and
// double otherwise.
resolve_value* tensor_to_value(const torch::Tensor& t, resolve_value_kind_t kind) {
    const bool integer = kind == RESOLVE_VALUE_INT_ARRAY || kind == RESOLVE_VALUE_INT_MATRIX;
    const bool matrix = kind == RESOLVE_VALUE_INT_MATRIX || kind == RESOLVE_VALUE_DOUBLE_MATRIX;
    const torch::Tensor host =
        t.detach().to(torch::kCPU, integer ? torch::kInt64 : torch::kFloat64).contiguous();
    const int64_t nrow = matrix ? (host.dim() >= 1 ? host.size(0) : 1) : 0;
    const int64_t ncol = matrix ? (host.dim() >= 2 ? host.size(1) : 1) : 0;
    return integer ? v_numeric<int64_t>(kind, host.data_ptr<int64_t>(), host.numel(), nrow, ncol)
                   : v_numeric<double>(kind, host.data_ptr<double>(), host.numel(), nrow, ncol);
}
resolve_value* tensor_to_mat(const torch::Tensor& t) { return tensor_to_value(t, RESOLVE_VALUE_DOUBLE_MATRIX); }
resolve_value* tensor_to_imat(const torch::Tensor& t) { return tensor_to_value(t, RESOLVE_VALUE_INT_MATRIX); }
resolve_value* tensor_to_vec(const torch::Tensor& t) { return tensor_to_value(t, RESOLVE_VALUE_DOUBLE_ARRAY); }
resolve_value* tensor_to_ivec(const torch::Tensor& t) { return tensor_to_value(t, RESOLVE_VALUE_INT_ARRAY); }

// A tensor as a node of `kind`, or a NULL-kind node when it is undefined or
// empty: the rule every optional-tensor accessor follows.
resolve_value* tensor_or_null(const torch::Tensor& t, resolve_value_kind_t kind) {
    return (t.defined() && t.numel() > 0) ? tensor_to_value(t, kind) : v_null();
}

}  // namespace

// ============================================================================
// Config <-> value tree, driven by the field registry
// ============================================================================
//
// A registry row's member name IS its value-tree key, so one visitor reads a
// map into a config struct and its inverse writes the struct back out, both
// walking the list in resolve/config_registry.hpp. The reader accepts exactly
// those names: a key no row names is refused, in a nested sub-config too, so a
// misspelled setting fails instead of training with the default. Enum and
// device spellings come from enum_names.hpp and parse_device, which the CLI and
// `resolve info` read too.

namespace {

template <typename Cfg>
void read_config(Cfg& cfg, const resolve_value* map, const std::string& what);

// The value `parse()` reads from a name, with `field` in front of the message
// of the exception it throws for a name it does not know.
template <typename Parse>
auto named(const std::string& field, Parse&& parse) -> decltype(parse()) {
    try {
        return parse();
    } catch (const std::exception& e) {
        throw std::invalid_argument(field + ": " + e.what());
    }
}

// Value map -> config struct. An absent key leaves the member's value in
// place, which is what makes every entry of an R config list optional.
struct ValueFieldReader {
    const resolve_value* map;
    std::string what;

    template <typename T>
    void operator()(const char* name, const char*, T& value) const {
        const resolve_value* v = vget(map, name);
        if (v == nullptr || v->kind == RESOLVE_VALUE_NULL) return;
        const std::string field = what + "." + name;
        if constexpr (std::is_same_v<T, LogCallback>) {
            throw std::invalid_argument(field + ": a callback cannot be passed as a value");
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            const std::string& name_text = vstr(v, field);
            value = named(field, [&] { return parse_device(name_text); });
        } else if constexpr (is_registered_config_v<T>) {
            read_config(value, v, field);
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            if (v->kind != RESOLVE_VALUE_LIST) wrong_kind(field, "a list of branch maps");
            value.clear();
            for (size_t i = 0; i < v->items.size(); ++i) {
                ParallelBranchConfig branch;
                read_config(branch, v->items[i], field + "[" + std::to_string(i) + "]");
                value.push_back(std::move(branch));
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            value = vbool(v, field);
        } else if constexpr (std::is_enum_v<T>) {
            const std::string& name_text = vstr(v, field);
            value = named(field, [&] { return enum_from_name<T>(name_text); });
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t> ||
                             std::is_same_v<T, float>) {
            value = vnum<T>(v, field);
        } else if constexpr (std::is_same_v<T, std::string>) {
            value = vstr(v, field);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            value = vnum_vec<typename T::value_type>(v, field);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            const std::vector<int> pair_values = vnum_vec<int>(v, field);
            if (pair_values.size() != 2) wrong_kind(field, "two numbers");
            value = {pair_values[0], pair_values[1]};
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no value-tree representation; "
                          "add a branch here and to ValueFieldWriter");
        }
    }
};

// Reads the registered struct `cfg` from map `map`; an absent or NULL-kind map
// leaves it as it is. `what` names the struct in errors (a nested one as
// parent.member).
template <typename Cfg>
void read_config(Cfg& cfg, const resolve_value* map, const std::string& what) {
    if (!map || map->kind == RESOLVE_VALUE_NULL) return;
    if (map->kind != RESOLVE_VALUE_MAP) wrong_kind(what, "a map of field name -> value");
    reject_unknown_keys(map, what, field_names<Cfg>());
    for_each_field(cfg, ValueFieldReader{map, what});
}

template <typename Cfg>
Cfg parse_config(const resolve_value* map, const std::string& what) {
    Cfg cfg{};
    read_config(cfg, map, what);
    return cfg;
}

// Config struct -> value map. A child node is attached to its parent before it
// is filled, so a throw part-way through leaves it owned (and freed) by the
// root.
struct ValueFieldWriter {
    resolve_value* map;

    template <typename T>
    void operator()(const char* name, const char*, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback>) {
            (void)name;
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            v_put(map, name, v_string(device_name(value)));
        } else if constexpr (is_registered_config_v<T>) {
            auto* child = v_map();
            v_put(map, name, child);
            for_each_field(value, ValueFieldWriter{child});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            auto* list = v_list();
            v_put(map, name, list);
            for (const auto& branch : value) {
                auto* item = v_map();
                v_append(list, item);
                for_each_field(branch, ValueFieldWriter{item});
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            v_put(map, name, v_bool(value));
        } else if constexpr (std::is_enum_v<T>) {
            v_put(map, name, v_string(enum_to_name(value)));
        } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
            v_put(map, name, v_int(value));
        } else if constexpr (std::is_same_v<T, float>) {
            v_put(map, name, v_double(value));
        } else if constexpr (std::is_same_v<T, std::string>) {
            v_put(map, name, v_string(value));
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            v_put(map, name, v_int_array(value));
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            v_put(map, name, v_double_array(value));
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            v_put(map, name, v_int_array(std::vector<int64_t>{
                static_cast<int64_t>(value.first),
                static_cast<int64_t>(value.second)}));
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no value-tree representation; "
                          "add a branch here and to ValueFieldReader");
        }
    }
};

// A registered struct as a value map: every field under its member name.
template <typename Cfg>
resolve_value* config_to_value(const Cfg& cfg) {
    ValueGuard g(v_map());
    for_each_field(cfg, ValueFieldWriter{g.p});
    return g.release();
}

}  // namespace

// ============================================================================
// Top-level parsers: roles / targets / dataset config / model config / schema
// ============================================================================

namespace {

RoleMapping parse_roles(const resolve_value* r) {
    reject_unknown_keys(r, "roles", {"plot_id", "species_id", "abundance",
                                     "longitude", "latitude", "genus", "family",
                                     "covariates", "categoricals"});
    RoleMapping roles;
    if (vhas(r, "plot_id")) roles.plot_id = vstr(r, "plot_id");
    if (vhas(r, "species_id")) roles.species_id = vstr(r, "species_id");
    if (vhas(r, "abundance")) roles.abundance = vstr(r, "abundance");
    if (vhas(r, "longitude")) roles.longitude = vstr(r, "longitude");
    if (vhas(r, "latitude")) roles.latitude = vstr(r, "latitude");
    if (vhas(r, "genus")) roles.genus = vstr(r, "genus");
    if (vhas(r, "family")) roles.family = vstr(r, "family");
    if (vhas(r, "covariates")) roles.covariates = vstr_vec(r, "covariates");
    if (vhas(r, "categoricals")) roles.categoricals = vstr_vec(r, "categoricals");
    return roles;
}

// Target list -> TargetSpec vector. The map is keyed by target name; each entry
// is a sub-map carrying task/transform/num_classes/weight, and optionally
// "column" (absent: the target's name names its column).
std::vector<TargetSpec> parse_targets(const resolve_value* targets) {
    std::vector<TargetSpec> out;
    // Absent or explicitly null is legal and means "no targets" (an inference
    // dataset). Any other non-map kind is a malformed argument: the map's KEYS
    // are the target names, so a keyless tree carries no targets at all, and
    // returning an empty vector for it discards everything the caller passed.
    if (!targets || targets->kind == RESOLVE_VALUE_NULL) return out;
    if (targets->kind != RESOLVE_VALUE_MAP) {
        throw std::invalid_argument(
            "targets must be a map of target name -> specification; got a "
            "keyless value, which names no targets. Every entry needs a name "
            "(in R, a named list: list(area = list(column = \"area\", "
            "task = \"regression\"))).");
    }
    for (size_t i = 0; i < targets->keys.size(); ++i) {
        const std::string& name = targets->keys[i];
        const resolve_value* spec = targets->vals[i];
        if (name.empty()) {
            throw std::invalid_argument(
                "targets: an entry has an empty name. The name identifies the "
                "target and, when no \"column\" is given, names its column.");
        }
        if (std::find(targets->keys.begin(), targets->keys.begin() + static_cast<std::ptrdiff_t>(i),
                      name) != targets->keys.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw std::invalid_argument("targets: the name '" + name +
                                        "' is given twice; each target needs a distinct name");
        }
        if (!spec || spec->kind != RESOLVE_VALUE_MAP) {
            throw std::invalid_argument(
                "target '" + name + "': the specification must be a map of "
                "keys (column / task / transform / num_classes / weight / "
                "class_mapping / missing), not a bare value.");
        }
        reject_unknown_keys(spec, "target '" + name + "'",
                            {"column", "task", "transform", "num_classes",
                             "weight", "class_mapping", "missing"});
        TargetSpec ts;
        ts.target_name = name;
        ts.column_name = vhas(spec, "column") ? vstr(spec, "column") : name;
        if (vhas(spec, "task")) ts.task = enum_from_name<TaskType>(vstr(spec, "task"));
        if (vhas(spec, "transform")) ts.transform = enum_from_name<TransformType>(vstr(spec, "transform"));
        if (vhas(spec, "num_classes")) ts.num_classes = vnum<int>(spec, "num_classes");
        if (vhas(spec, "weight")) ts.weight = vnum<float>(spec, "weight");
        if (vhas(spec, "missing")) ts.missing = enum_from_name<TargetMissingPolicy>(vstr(spec, "missing"));
        // Optional explicit class -> integer-code mapping (a sub-map of
        // name->int), so a held-out set can be pinned to the same codes as the
        // training set instead of auto-fitting a possibly-different order.
        const resolve_value* cm = vget(spec, "class_mapping");
        if (cm && cm->kind != RESOLVE_VALUE_NULL) {
            if (cm->kind != RESOLVE_VALUE_MAP) wrong_kind("target '" + name + "'.class_mapping",
                                                         "a map of class name -> code");
            for (size_t j = 0; j < cm->keys.size(); ++j) {
                ts.class_mapping[cm->keys[j]] = vnum<int64_t>(cm->vals[j], "class_mapping." + cm->keys[j]);
            }
        }
        out.push_back(ts);
    }
    return out;
}

DatasetConfig parse_dataset_config(const resolve_value* c) {
    return parse_config<DatasetConfig>(c, "dataset config");
}
ModelConfig parse_model_config(const resolve_value* c) {
    return parse_config<ModelConfig>(c, "model config");
}
TrainConfig parse_train_config(const resolve_value* c) {
    return parse_config<TrainConfig>(c, "train config");
}

// Single source of truth for the value-tree schema keys, consumed by both
// parse_schema (read) and schema_to_value (write) below. A field emitted under
// one spelling but read under another would drop on the round-trip, so both
// sites reference these constants.
namespace schema_tree_keys {
inline constexpr const char* kNPlots             = "n_plots";
inline constexpr const char* kNSpecies           = "n_species";
inline constexpr const char* kNSpeciesVocab      = "n_species_vocab";
inline constexpr const char* kHasCoordinates     = "has_coordinates";
inline constexpr const char* kHasAbundance       = "has_abundance";
inline constexpr const char* kHasTaxonomy        = "has_taxonomy";
inline constexpr const char* kNGenera            = "n_genera";
inline constexpr const char* kNFamilies          = "n_families";
inline constexpr const char* kCovariateNames     = "covariate_names";
inline constexpr const char* kTargets            = "targets";
inline constexpr const char* kCategoricalNames   = "categorical_names";
inline constexpr const char* kCategoricalVocabSizes = "categorical_vocab_sizes";
inline constexpr const char* kCategoricalEmbedDim   = "categorical_embed_dim";
// The loader settings: a DatasetConfig map, through the field registry.
inline constexpr const char* kDatasetConfig      = "dataset_config";
inline constexpr const char* kUnrecordedSettings = "unrecorded_dataset_settings";
inline constexpr const char* kSpeciesVocab       = "species_vocab";
inline constexpr const char* kGenusVocab         = "genus_vocab";
inline constexpr const char* kFamilyVocab        = "family_vocab";
// Categorical string -> code maps. Not part of ResolveSchema (which carries
// only column names + sizes); carried alongside it in the "vocabs" tree the
// vocabulary-reusing dataset loaders take.
inline constexpr const char* kCategoricalVocab   = "categorical_vocab";
// Per-target sub-map keys.
inline constexpr const char* kTargetTask         = "task";
inline constexpr const char* kTargetTransform    = "transform";
inline constexpr const char* kTargetNumClasses   = "num_classes";
inline constexpr const char* kTargetWeight       = "weight";
inline constexpr const char* kTargetClassWeights = "class_weights";
inline constexpr const char* kTargetClassNames   = "class_names";
inline constexpr const char* kTargetMissing      = "missing";

inline constexpr const char* kSchemaKeys[] = {
    kNPlots, kNSpecies, kNSpeciesVocab, kHasCoordinates, kHasAbundance, kHasTaxonomy,
    kNGenera, kNFamilies, kCovariateNames, kTargets, kCategoricalNames,
    kCategoricalVocabSizes, kCategoricalEmbedDim, kDatasetConfig, kUnrecordedSettings,
    kSpeciesVocab, kGenusVocab, kFamilyVocab,
};
inline constexpr const char* kVocabsKeys[] = {
    kNPlots, kNSpecies, kNSpeciesVocab, kHasCoordinates, kHasAbundance, kHasTaxonomy,
    kNGenera, kNFamilies, kCovariateNames, kTargets, kCategoricalNames,
    kCategoricalVocabSizes, kCategoricalEmbedDim, kDatasetConfig, kUnrecordedSettings,
    kSpeciesVocab, kGenusVocab, kFamilyVocab, kCategoricalVocab,
};
inline constexpr const char* kTargetConfigKeys[] = {
    kTargetTask, kTargetTransform, kTargetNumClasses, kTargetWeight,
    kTargetClassWeights, kTargetClassNames, kTargetMissing,
};
}  // namespace schema_tree_keys

// Schema map -> ResolveSchema: the full set schema_to_value emits, including
// the categorical columns, so a categorical dataset's schema round-trips into a
// categorical-aware model. `what` names the tree in errors; a key outside the
// schema's (or, for a vocabulary tree, the schema's plus categorical_vocab) is
// refused.
template <typename Keys>
ResolveSchema parse_schema(const resolve_value* s, const std::string& what, const Keys& accepted) {
    namespace k = schema_tree_keys;
    ResolveSchema schema;
    if (!s || s->kind == RESOLVE_VALUE_NULL) return schema;
    if (s->kind != RESOLVE_VALUE_MAP) wrong_kind(what, "a map");
    reject_unknown_keys(s, what, accepted);
    if (vhas(s, k::kNPlots)) schema.n_plots = vnum<int64_t>(s, k::kNPlots);
    if (vhas(s, k::kNSpecies)) schema.n_species = vnum<int64_t>(s, k::kNSpecies);
    if (vhas(s, k::kNSpeciesVocab)) schema.n_species_vocab = vnum<int64_t>(s, k::kNSpeciesVocab);
    if (vhas(s, k::kHasCoordinates)) schema.has_coordinates = vbool(s, k::kHasCoordinates);
    if (vhas(s, k::kHasAbundance)) schema.has_abundance = vbool(s, k::kHasAbundance);
    if (vhas(s, k::kHasTaxonomy)) schema.has_taxonomy = vbool(s, k::kHasTaxonomy);
    if (vhas(s, k::kNGenera)) schema.n_genera = vnum<int64_t>(s, k::kNGenera);
    if (vhas(s, k::kNFamilies)) schema.n_families = vnum<int64_t>(s, k::kNFamilies);
    if (vhas(s, k::kCovariateNames)) schema.covariate_names = vstr_vec(s, k::kCovariateNames);
    if (vhas(s, k::kCategoricalNames)) schema.categorical_names = vstr_vec(s, k::kCategoricalNames);
    if (vhas(s, k::kCategoricalVocabSizes)) {
        schema.categorical_vocab_sizes = vnum_vec<int64_t>(s, k::kCategoricalVocabSizes);
    }
    if (vhas(s, k::kCategoricalEmbedDim)) {
        schema.categorical_embed_dim = vnum<int64_t>(s, k::kCategoricalEmbedDim);
    }
    read_config(schema.dataset_config, vget(s, k::kDatasetConfig), what + "." + k::kDatasetConfig);
    if (vhas(s, k::kUnrecordedSettings)) {
        schema.unrecorded_dataset_settings = vstr_vec(s, k::kUnrecordedSettings);
    }
    if (vhas(s, k::kSpeciesVocab)) schema.species_vocab = vstr_vec(s, k::kSpeciesVocab);
    if (vhas(s, k::kGenusVocab)) schema.genus_vocab = vstr_vec(s, k::kGenusVocab);
    if (vhas(s, k::kFamilyVocab)) schema.family_vocab = vstr_vec(s, k::kFamilyVocab);

    const resolve_value* targets = vget(s, k::kTargets);
    if (targets && targets->kind != RESOLVE_VALUE_NULL) {
        if (targets->kind != RESOLVE_VALUE_MAP) wrong_kind(what + ".targets", "a map");
        for (size_t i = 0; i < targets->keys.size(); ++i) {
            const resolve_value* tc_v = targets->vals[i];
            const std::string target_what = what + ".targets." + targets->keys[i];
            if (!tc_v || tc_v->kind != RESOLVE_VALUE_MAP) wrong_kind(target_what, "a map");
            reject_unknown_keys(tc_v, target_what, k::kTargetConfigKeys);
            TargetConfig tc;
            tc.name = targets->keys[i];
            if (vhas(tc_v, k::kTargetTask)) tc.task = enum_from_name<TaskType>(vstr(tc_v, k::kTargetTask));
            if (vhas(tc_v, k::kTargetTransform)) tc.transform = enum_from_name<TransformType>(vstr(tc_v, k::kTargetTransform));
            if (vhas(tc_v, k::kTargetNumClasses)) tc.num_classes = vnum<int>(tc_v, k::kTargetNumClasses);
            if (vhas(tc_v, k::kTargetWeight)) tc.weight = vnum<float>(tc_v, k::kTargetWeight);
            if (vhas(tc_v, k::kTargetClassWeights)) tc.class_weights = vnum_vec<float>(tc_v, k::kTargetClassWeights);
            if (vhas(tc_v, k::kTargetClassNames)) tc.class_names = vstr_vec(tc_v, k::kTargetClassNames);
            if (vhas(tc_v, k::kTargetMissing)) tc.missing = enum_from_name<TargetMissingPolicy>(vstr(tc_v, k::kTargetMissing));
            schema.targets.push_back(tc);
        }
    }
    return schema;
}

// Inverse of nested_metrics_to_value: a map(target -> map(metric -> number)).
std::unordered_map<std::string, std::unordered_map<std::string, float>>
value_to_nested_metrics(const resolve_value* v) {
    std::unordered_map<std::string, std::unordered_map<std::string, float>> out;
    if (!v || v->kind == RESOLVE_VALUE_NULL) return out;
    if (v->kind != RESOLVE_VALUE_MAP) wrong_kind("final_metrics", "a map of target -> metrics");
    for (size_t i = 0; i < v->keys.size(); ++i) {
        const resolve_value* metrics = v->vals[i];
        if (!metrics || metrics->kind != RESOLVE_VALUE_MAP) {
            wrong_kind("final_metrics." + v->keys[i], "a map of metric -> number");
        }
        auto& inner = out[v->keys[i]];
        for (size_t j = 0; j < metrics->keys.size(); ++j) {
            inner[metrics->keys[j]] = vnum<float>(metrics->vals[j], metrics->keys[j]);
        }
    }
    return out;
}

// Inverse of run_metadata_to_value: reads every key it writes, so a metadata
// map handed back from R round-trips.
RunMetadata parse_run_metadata(const resolve_value* c) {
    reject_unknown_keys(c, "run metadata",
                        {"resolve_version", "created_at", "completed_at", "train_time_seconds",
                         "n_plots_train", "n_plots_test", "hold_out", "test_size", "split_seed",
                         "best_epoch", "total_epochs", "final_metrics"});
    RunMetadata rm;
    if (vhas(c, "resolve_version")) rm.resolve_version = vstr(c, "resolve_version");
    if (vhas(c, "created_at")) rm.created_at = vstr(c, "created_at");
    if (vhas(c, "completed_at")) rm.completed_at = vstr(c, "completed_at");
    if (vhas(c, "train_time_seconds")) rm.train_time_seconds = vnum<float>(c, "train_time_seconds");
    if (vhas(c, "n_plots_train")) rm.n_plots_train = vnum<int64_t>(c, "n_plots_train");
    if (vhas(c, "n_plots_test")) rm.n_plots_test = vnum<int64_t>(c, "n_plots_test");
    if (vhas(c, "hold_out")) rm.hold_out = enum_from_name<HoldOutMode>(vstr(c, "hold_out"));
    if (vhas(c, "test_size")) rm.test_size = vnum<float>(c, "test_size");
    if (vhas(c, "split_seed")) rm.split_seed = vnum<int>(c, "split_seed");
    if (vhas(c, "best_epoch")) rm.best_epoch = vnum<int>(c, "best_epoch");
    if (vhas(c, "total_epochs")) rm.total_epochs = vnum<int>(c, "total_epochs");
    if (vhas(c, "final_metrics")) rm.final_metrics = value_to_nested_metrics(vget(c, "final_metrics"));
    return rm;
}

// ============================================================================
// Result-struct -> value-tree converters
// ============================================================================
//
// Each converter builds its tree under a ValueGuard, so a throw part-way
// (an allocation, a tensor conversion) frees the partial tree.

resolve_value* baseline_metrics_to_value(const BaselineMetrics& bm) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "baseline_mse", v_double(bm.baseline_mse));
    v_put(m, "baseline_mae", v_double(bm.baseline_mae));
    v_put(m, "model_mse", v_double(bm.model_mse));
    v_put(m, "model_mae", v_double(bm.model_mae));
    v_put(m, "skill_score", v_double(bm.skill_score));
    v_put(m, "r_squared", v_double(bm.r_squared));
    v_put(m, "baseline_accuracy", v_double(bm.baseline_accuracy));
    v_put(m, "model_accuracy", v_double(bm.model_accuracy));
    v_put(m, "accuracy_lift", v_double(bm.accuracy_lift));
    v_put(m, "training_mean", v_double(bm.training_mean));
    v_put(m, "training_mode", v_int(bm.training_mode));
    return g.release();
}
resolve_value* layer_diagnostics_to_value(const LayerDiagnostics& ld) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "name", v_string(ld.name));
    v_put(m, "n_neurons", v_int(ld.n_neurons));
    v_put(m, "n_dead", v_int(ld.n_dead));
    v_put(m, "n_saturated", v_int(ld.n_saturated));
    v_put(m, "dead_fraction", v_double(ld.dead_fraction));
    v_put(m, "saturated_fraction", v_double(ld.saturated_fraction));
    v_put(m, "mean_activation", v_double(ld.mean_activation));
    v_put(m, "std_activation", v_double(ld.std_activation));
    v_put(m, "sparsity", v_double(ld.sparsity));
    return g.release();
}
resolve_value* network_diagnostics_to_value(const NetworkDiagnostics& nd) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    auto* layers = v_list();
    v_put(m, "layers", layers);
    for (const auto& ld : nd.layers) v_append(layers, layer_diagnostics_to_value(ld));
    v_put(m, "total_neurons", v_int(nd.total_neurons));
    v_put(m, "total_dead", v_int(nd.total_dead));
    v_put(m, "total_saturated", v_int(nd.total_saturated));
    v_put(m, "overall_dead_fraction", v_double(nd.overall_dead_fraction));
    v_put(m, "overall_saturated_fraction", v_double(nd.overall_saturated_fraction));
    v_put(m, "has_issues", v_bool(nd.has_issues));
    v_put(m, "summary", v_string(nd.summary));
    return g.release();
}
// The keys of a hash map in sorted order, so a map emitted twice (a run and
// the same run resumed from its checkpoint) reads the same.
template <typename Map>
std::vector<std::string> sorted_keys(const Map& map) {
    std::vector<std::string> keys;
    keys.reserve(map.size());
    for (const auto& entry : map) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());
    return keys;
}

resolve_value* nested_metrics_to_value(
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics) {
    ValueGuard g(v_map());
    for (const auto& target : sorted_keys(metrics)) {
        const auto& metric_map = metrics.at(target);
        auto* inner = v_map();
        v_put(g.p, target, inner);
        for (const auto& metric : sorted_keys(metric_map)) {
            v_put(inner, metric, v_double(metric_map.at(metric)));
        }
    }
    return g.release();
}
resolve_value* named_histories_to_value(
    const std::unordered_map<std::string, std::vector<float>>& histories) {
    ValueGuard g(v_map());
    for (const auto& name : sorted_keys(histories)) v_put(g.p, name, v_double_array(histories.at(name)));
    return g.release();
}
resolve_value* train_result_to_value(const TrainResult& tr) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "best_epoch", v_int(tr.best_epoch));
    v_put(m, "final_metrics", nested_metrics_to_value(tr.final_metrics));
    v_put(m, "train_loss", v_double_array(tr.train_loss_history));
    v_put(m, "test_loss", v_double_array(tr.test_loss_history));
    v_put(m, "lr", v_double_array(tr.lr_history));
    v_put(m, "selection", v_double_array(tr.selection_history));
    v_put(m, "train_loss_by_target", named_histories_to_value(tr.train_loss_history_by_target));
    v_put(m, "test_loss_by_target", named_histories_to_value(tr.test_loss_history_by_target));
    v_put(m, "train_aux_loss", named_histories_to_value(tr.train_aux_loss_history));
    v_put(m, "train_time_seconds", v_double(tr.train_time_seconds));
    v_put(m, "resumed_from_epoch", v_int(tr.resumed_from_epoch));
    v_put(m, "effective_batch_size", v_int(tr.effective_batch_size));
    auto* baselines = v_map();
    v_put(m, "baselines", baselines);
    for (const auto& target : sorted_keys(tr.baselines)) {
        v_put(baselines, target, baseline_metrics_to_value(tr.baselines.at(target)));
    }
    v_put(m, "diagnostics", network_diagnostics_to_value(tr.diagnostics));
    return g.release();
}
resolve_value* calibration_bin_to_value(const CalibrationBin& cb) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "bin_start", v_double(cb.bin_start));
    v_put(m, "bin_end", v_double(cb.bin_end));
    v_put(m, "mean_predicted_prob", v_double(cb.mean_predicted_prob));
    v_put(m, "actual_frequency", v_double(cb.actual_frequency));
    v_put(m, "count", v_int(cb.count));
    return g.release();
}
resolve_value* calibration_result_to_value(const CalibrationResult& cr) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "target_name", v_string(cr.target_name));
    v_put(m, "class_idx", v_int(cr.class_idx));
    auto* bins = v_list();
    v_put(m, "bins", bins);
    for (const auto& b : cr.bins) v_append(bins, calibration_bin_to_value(b));
    v_put(m, "expected_calibration_error", v_double(cr.expected_calibration_error));
    v_put(m, "max_calibration_error", v_double(cr.max_calibration_error));
    return g.release();
}
resolve_value* residual_analysis_to_value(const ResidualAnalysis& ra) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "target_name", v_string(ra.target_name));
    v_put(m, "predictions", v_double_array(ra.predictions));
    v_put(m, "actuals", v_double_array(ra.actuals));
    v_put(m, "residuals", v_double_array(ra.residuals));
    v_put(m, "test_rows", v_int_array(ra.test_rows));
    v_put(m, "mean_residual", v_double(ra.mean_residual));
    v_put(m, "std_residual", v_double(ra.std_residual));
    v_put(m, "skewness", v_double(ra.skewness));
    v_put(m, "kurtosis", v_double(ra.kurtosis));
    v_put(m, "q05", v_double(ra.q05));
    v_put(m, "q25", v_double(ra.q25));
    v_put(m, "q50", v_double(ra.q50));
    v_put(m, "q75", v_double(ra.q75));
    v_put(m, "q95", v_double(ra.q95));
    return g.release();
}
// An int64 tensor as an INT_ARRAY, empty when it is undefined.
resolve_value* ivec_or_empty(const torch::Tensor& t) {
    return t.defined() ? tensor_to_ivec(t) : v_numeric<int64_t>(RESOLVE_VALUE_INT_ARRAY, static_cast<const int64_t*>(nullptr), 0);
}
resolve_value* classification_predictions_to_value(const ClassificationPredictions& cp) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "target_name", v_string(cp.target_name));
    v_put(m, "class_names", v_string_array(cp.class_names));
    v_put(m, "predicted_classes", ivec_or_empty(cp.predicted_classes));
    v_put(m, "actuals", ivec_or_empty(cp.actuals));
    v_put(m, "test_rows", ivec_or_empty(cp.test_rows));
    v_put(m, "probabilities",
          cp.probabilities.defined()
              ? tensor_to_mat(cp.probabilities)
              : v_numeric<double>(RESOLVE_VALUE_DOUBLE_MATRIX, static_cast<const double*>(nullptr), 0));
    return g.release();
}
resolve_value* cross_validation_result_to_value(const CrossValidationResult& cvr) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "n_folds", v_int(cvr.n_folds));
    v_put(m, "mean_metrics", nested_metrics_to_value(cvr.mean_metrics));
    v_put(m, "std_metrics", nested_metrics_to_value(cvr.std_metrics));
    auto* folds = v_list();
    v_put(m, "fold_results", folds);
    for (const auto& fr : cvr.fold_results) v_append(folds, train_result_to_value(fr));
    v_put(m, "total_time_seconds", v_double(cvr.total_time_seconds));
    return g.release();
}
resolve_value* scalers_to_value(const Scalers& s) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    if (s.continuous_mean.defined()) v_put(m, "continuous_mean", tensor_to_vec(s.continuous_mean));
    if (s.continuous_scale.defined()) v_put(m, "continuous_scale", tensor_to_vec(s.continuous_scale));
    if (s.continuous_fill.defined()) v_put(m, "continuous_fill", tensor_to_vec(s.continuous_fill));
    // Per-target regression scaling { name -> {mean, scale} }.
    if (!s.target_scalers.empty()) {
        auto* ts = v_map();
        v_put(m, "target_scalers", ts);
        for (const auto& name : sorted_keys(s.target_scalers)) {
            const auto& ms = s.target_scalers.at(name);
            auto* pair_m = v_map();
            v_put(ts, name, pair_m);
            v_put(pair_m, "mean", v_double(ms.first.defined() ? ms.first.item<double>() : 0.0));
            v_put(pair_m, "scale", v_double(ms.second.defined() ? ms.second.item<double>() : 0.0));
        }
    }
    return g.release();
}
// One entry per column: a sub-map { name -> code }, which R turns into a named
// integer vector.
resolve_value* categorical_vocab_to_value(const CategoricalVocab& vocab) {
    ValueGuard g(v_map());
    for (const auto& col : vocab.column_names()) {
        auto* inner = v_map();
        v_put(g.p, col, inner);
        for (const auto& [k, code] : vocab.column_map(col)) v_put(inner, k, v_int(code));
    }
    return g.release();
}

// Inverse of categorical_vocab_to_value. Restores the per-column string -> code
// maps verbatim from the tree the vocabulary-reusing dataset loaders receive:
// the codes come from the checkpoint, so they are assigned, not re-derived.
CategoricalVocab value_to_categorical_vocab(const resolve_value* v) {
    CategoricalVocab vocab;
    if (!v || v->kind == RESOLVE_VALUE_NULL) return vocab;
    if (v->kind != RESOLVE_VALUE_MAP) wrong_kind("categorical_vocab", "a map of column -> codes");
    for (size_t i = 0; i < v->keys.size(); ++i) {
        const resolve_value* inner = v->vals[i];
        if (!inner || inner->kind != RESOLVE_VALUE_MAP) {
            wrong_kind("categorical_vocab." + v->keys[i], "a map of value -> code");
        }
        std::unordered_map<std::string, int64_t> cmap;
        cmap.reserve(inner->keys.size());
        for (size_t j = 0; j < inner->keys.size(); ++j) {
            cmap[inner->keys[j]] = vnum<int64_t>(inner->vals[j], inner->keys[j]);
        }
        vocab.set_column_map(v->keys[i], cmap);
    }
    return vocab;
}

// ResolveSchema -> MAP tree. Single source of truth for the emitted schema,
// shared by the dataset "schema" accessor and the predictor "schema"/"vocabs"
// accessors -- a field emitted by one but not the other is exactly
// how the vocab would go missing on one binding path.
resolve_value* schema_to_value(const ResolveSchema& s) {
    namespace k = schema_tree_keys;
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, k::kNPlots, v_int(s.n_plots));
    v_put(m, k::kNSpecies, v_int(s.n_species));
    v_put(m, k::kNSpeciesVocab, v_int(s.n_species_vocab));
    v_put(m, k::kHasCoordinates, v_bool(s.has_coordinates));
    v_put(m, k::kHasAbundance, v_bool(s.has_abundance));
    v_put(m, k::kHasTaxonomy, v_bool(s.has_taxonomy));
    v_put(m, k::kNGenera, v_int(s.n_genera));
    v_put(m, k::kNFamilies, v_int(s.n_families));
    v_put(m, k::kCovariateNames, v_string_array(s.covariate_names));
    auto* targets_m = v_map();
    v_put(m, k::kTargets, targets_m);
    for (const auto& tc : s.targets) {
        auto* tm = v_map();
        v_put(targets_m, tc.name, tm);
        v_put(tm, k::kTargetTask, v_string(enum_to_name(tc.task)));
        v_put(tm, k::kTargetTransform, v_string(enum_to_name(tc.transform)));
        v_put(tm, k::kTargetNumClasses, v_int(tc.num_classes));
        v_put(tm, k::kTargetWeight, v_double(tc.weight));
        v_put(tm, k::kTargetClassWeights, v_double_array(tc.class_weights));
        // Per-class label vocabulary (class_names[code] == label).
        v_put(tm, k::kTargetClassNames, v_string_array(tc.class_names));
        v_put(tm, k::kTargetMissing, v_string(enum_to_name(tc.missing)));
    }
    v_put(m, k::kCategoricalNames, v_string_array(s.categorical_names));
    v_put(m, k::kCategoricalVocabSizes, v_int_array(s.categorical_vocab_sizes));
    v_put(m, k::kCategoricalEmbedDim, v_int(s.categorical_embed_dim));
    v_put(m, k::kDatasetConfig, config_to_value(s.dataset_config));
    v_put(m, k::kUnrecordedSettings, v_string_array(s.unrecorded_dataset_settings));
    v_put(m, k::kSpeciesVocab, v_string_array(s.species_vocab));
    v_put(m, k::kGenusVocab, v_string_array(s.genus_vocab));
    v_put(m, k::kFamilyVocab, v_string_array(s.family_vocab));
    return g.release();
}

ResolveSchema value_to_schema(const resolve_value* s) {
    return parse_schema(s, "schema", schema_tree_keys::kSchemaKeys);
}

// ExternalVocabs <-> MAP tree. The tree is a schema tree (which already carries
// species/genus/family + the target class vocabularies) plus the categorical
// string -> code maps, which live on the Predictor rather than the schema.
resolve_value* external_vocabs_to_value(const ExternalVocabs& v) {
    ResolveSchema s;
    s.species_vocab = v.species_vocab;
    s.genus_vocab = v.taxonomy.genus_names();
    s.family_vocab = v.taxonomy.family_names();
    s.targets = v.targets;
    s.dataset_config.pool_species_cap = v.pool_species_cap;
    ValueGuard g(schema_to_value(s));
    v_put(g.p, schema_tree_keys::kCategoricalVocab, categorical_vocab_to_value(v.categorical));
    return g.release();
}

ExternalVocabs value_to_external_vocabs(const resolve_value* v) {
    ExternalVocabs out =
        external_vocabs_from_schema(parse_schema(v, "vocabs", schema_tree_keys::kVocabsKeys));
    out.categorical = value_to_categorical_vocab(vget(v, schema_tree_keys::kCategoricalVocab));
    return out;
}

resolve_value* run_metadata_to_value(const RunMetadata& m0) {
    ValueGuard g(v_map());
    resolve_value* m = g.p;
    v_put(m, "resolve_version", v_string(m0.resolve_version));
    v_put(m, "created_at", v_string(m0.created_at));
    v_put(m, "completed_at", v_string(m0.completed_at));
    v_put(m, "train_time_seconds", v_double(m0.train_time_seconds));
    v_put(m, "n_plots_train", v_int(m0.n_plots_train));
    v_put(m, "n_plots_test", v_int(m0.n_plots_test));
    v_put(m, "hold_out", v_string(enum_to_name(m0.hold_out)));
    v_put(m, "test_size", v_double(m0.test_size));
    v_put(m, "split_seed", v_int(m0.split_seed));
    v_put(m, "best_epoch", v_int(m0.best_epoch));
    v_put(m, "total_epochs", v_int(m0.total_epochs));
    v_put(m, "final_metrics", nested_metrics_to_value(m0.final_metrics));
    return g.release();
}

// A tensor map as a value map of `kind` nodes: name -> double array for
// per-target values (dataset targets, forward outputs, predictions), name ->
// (n_plots, n_classes) matrix for class probabilities.
resolve_value* tensor_map_to_value(const TensorMap& m0, resolve_value_kind_t kind) {
    ValueGuard g(v_map());
    for (const auto& name : sorted_keys(m0)) v_put(g.p, name, tensor_to_value(m0.at(name), kind));
    return g.release();
}

}  // namespace

// ============================================================================
// Handle structs
// ============================================================================

struct resolve_dataset   { resolve::ResolveDataset ds; };
struct resolve_model     { resolve::ResolveModel model; };
struct resolve_trainer   { std::unique_ptr<resolve::Trainer> trainer; };
struct resolve_predictor { resolve::Predictor predictor; };
struct resolve_suite     { resolve::SuitePredictor suite; };
struct resolve_fuzzy     { resolve::fuzzy::FuzzyIndex index; };

namespace {

// The text of a device argument; NULL is a caller error, not the CPU.
torch::Device device_arg(const char* device) {
    if (device == nullptr) throw std::invalid_argument("device is NULL");
    return parse_device(device);
}

// The default value of every field of a registered struct, by the name a
// caller asks resolve_default_config for.
struct DefaultConfig {
    const char* name;
    resolve_value* (*defaults)();
};
template <typename Cfg>
resolve_value* default_config_value() {
    return config_to_value(Cfg{});
}
constexpr DefaultConfig kDefaultConfigs[] = {
    {"dataset", &default_config_value<DatasetConfig>},
    {"model", &default_config_value<ModelConfig>},
    {"train", &default_config_value<TrainConfig>},
    {"spatial_block", &default_config_value<SpatialBlockConfig>},
    {"retrieval", &default_config_value<RetrievalConfig>},
    {"pretrain", &default_config_value<PretrainConfig>},
    {"masked_species", &default_config_value<MLMPretrainConfig>},
    {"vae", &default_config_value<VAEConfig>},
};

}  // namespace

// ============================================================================
// Free functions + metrics
// ============================================================================

extern "C" {

const char* resolve_capi_version(void) { return resolve::VERSION; }

int resolve_capi_set_vram_fraction(double fraction, int device_index) {
    CAPI_BODY_INT({
        resolve::set_vram_fraction(fraction, device_index);
        return 0;
    })
}

int resolve_capi_set_thread_pools(int intraop_threads, int interop_threads) {
    CAPI_BODY_INT({
        resolve::set_thread_pools(intraop_threads, interop_threads);
        return 0;
    })
}

int resolve_capi_install_crash_handler(int shutdown_exit_code) {
    CAPI_BODY_INT({
        resolve::install_crash_handler(shutdown_exit_code);
        return 0;
    })
}

int resolve_capi_signal_work_complete(void) {
    CAPI_BODY_INT({
        resolve::signal_work_complete();
        return 0;
    })
}

resolve_value_t* resolve_capi_configure_cuda_allocator(int force) {
    CAPI_BODY_PTR({
        return v_string(resolve::configure_cuda_allocator(force != 0));
    })
}

resolve_value_t* resolve_default_config(const char* which) {
    CAPI_BODY_PTR({
        const std::string name = which ? which : "";
        for (const DefaultConfig& entry : kDefaultConfigs) {
            if (name == entry.name) return entry.defaults();
        }
        std::string known;
        for (const DefaultConfig& entry : kDefaultConfigs) {
            known += known.empty() ? "" : ", ";
            known += entry.name;
        }
        throw std::invalid_argument("default_config: unknown config '" + name +
                                    "'. Known: " + known);
    })
}

}  // extern "C"

namespace {

// One metric over two caller buffers of `n` doubles, written to `out`.
template <typename Metric>
int metric(const double* pred, const double* target, int64_t n, double* out, Metric&& compute) {
    CAPI_BODY_INT({
        if (out == nullptr) throw std::invalid_argument("metric: out is NULL");
        *out = compute(f32_tensor_from_doubles(pred, n, "pred"),
                       f32_tensor_from_doubles(target, n, "target"));
        return 0;
    })
}

}  // namespace

extern "C" {

int resolve_metric_band_accuracy(const double* pred, const double* target, int64_t n, double threshold, double* out) {
    return metric(pred, target, n, out, [&](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::band_accuracy(p, t, static_cast<float>(threshold));
    });
}
int resolve_metric_mae(const double* pred, const double* target, int64_t n, double* out) {
    return metric(pred, target, n, out, [](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::mae(p, t);
    });
}
int resolve_metric_rmse(const double* pred, const double* target, int64_t n, double* out) {
    return metric(pred, target, n, out, [](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::rmse(p, t);
    });
}
int resolve_metric_smape(const double* pred, const double* target, int64_t n, double eps, double* out) {
    return metric(pred, target, n, out, [&](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::smape(p, t, static_cast<float>(eps));
    });
}
int resolve_metric_accuracy(const double* pred, const double* target, int64_t n, double* out) {
    return metric(pred, target, n, out, [](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::accuracy(p, t);
    });
}
int resolve_metric_r_squared(const double* pred, const double* target, int64_t n, double* out) {
    return metric(pred, target, n, out, [](const torch::Tensor& p, const torch::Tensor& t) {
        return resolve::Metrics::r_squared(p, t);
    });
}

}  // extern "C"

// ============================================================================
// Dataset
// ============================================================================

namespace {

// A dataset table argument: a STRING names a CSV file, a MAP is an in-memory
// column table (decoded into `storage`, which must outlive the source).
std::optional<TableSource> table_source(const resolve_value* v, const char* what,
                                        std::optional<ColumnTable>& storage) {
    if (v == nullptr || v->kind == RESOLVE_VALUE_NULL) return std::nullopt;
    if (v->kind == RESOLVE_VALUE_STRING) {
        if (v->s.empty()) return std::nullopt;
        return TableSource::csv(v->s);
    }
    if (v->kind == RESOLVE_VALUE_MAP) {
        storage.emplace(value_to_column_table(v, what));
        return TableSource::table(*storage);
    }
    wrong_kind(what, "a CSV path (string) or a column table (map of column -> strings)");
}

}  // namespace

extern "C" {

resolve_dataset_t* resolve_dataset_load(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config,
    const resolve_dataset_t* schema_source, const resolve_value_t* vocabs) {
    CAPI_BODY_PTR({
        std::optional<ColumnTable> header_table;
        std::optional<ColumnTable> species_table;
        const std::optional<TableSource> header_source = table_source(header, "header", header_table);
        const std::optional<TableSource> species_source = table_source(species, "species", species_table);
        if (!species_source) throw std::invalid_argument("dataset_load: a species table is required");
        const bool has_vocabs = vocabs != nullptr && vocabs->kind != RESOLVE_VALUE_NULL;
        if (schema_source != nullptr && has_vocabs) {
            throw std::invalid_argument("dataset_load: pass schema_source or vocabs, not both");
        }
        std::optional<ExternalVocabs> adopted;
        if (schema_source != nullptr) adopted = schema_source->ds.external_vocabs();
        if (has_vocabs) adopted = value_to_external_vocabs(vocabs);
        return new resolve_dataset{ResolveDataset::load(
            header_source, *species_source, parse_roles(roles), parse_targets(targets),
            parse_dataset_config(config), adopted ? &*adopted : nullptr)};
    })
}

void resolve_dataset_free(resolve_dataset_t* ds) { delete ds; }

resolve_value_t* resolve_dataset_get(const resolve_dataset_t* ds, const char* what) {
    CAPI_BODY_PTR({
        if (!ds) throw std::invalid_argument("dataset_get: null handle");
        const ResolveDataset& d = ds->ds;
        const std::string w = what ? what : "";

        if (w == "coordinates") return tensor_or_null(d.coordinates(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "covariates") return tensor_or_null(d.covariates(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "hash_embedding") return tensor_or_null(d.hash_embedding(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "species_ids") return tensor_or_null(d.species_ids(), RESOLVE_VALUE_INT_MATRIX);
        // The sparse encoding's species vector as its sparse rows: plot p's
        // species codes are species[row_offsets[p] .. row_offsets[p + 1]),
        // each with its value. The dense matrix only on request.
        if (w == "species_vector") {
            const auto& rows = d.species_vector();
            if (!is_sparse_rows(rows)) return v_null();
            ValueGuard g(v_map());
            v_put(g.p, "row_offsets", tensor_to_ivec(rows.crow_indices()));
            v_put(g.p, "species", tensor_to_ivec(rows.col_indices()));
            v_put(g.p, "values", tensor_to_vec(rows.values()));
            v_put(g.p, "n_species", v_int(rows.size(1)));
            return g.release();
        }
        if (w == "species_vector_dense") {
            const auto& rows = d.species_vector();
            return is_sparse_rows(rows) ? tensor_to_mat(sparse_rows_to_dense(rows)) : v_null();
        }
        if (w == "genus_ids") return tensor_or_null(d.genus_ids(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "family_ids") return tensor_or_null(d.family_ids(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "unknown_fraction") return tensor_or_null(d.unknown_fraction(), RESOLVE_VALUE_DOUBLE_ARRAY);
        if (w == "unknown_count") return tensor_or_null(d.unknown_count(), RESOLVE_VALUE_DOUBLE_ARRAY);
        if (w == "categorical_ids") return tensor_or_null(d.categorical_ids(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "pool_genus_ids") return tensor_or_null(d.pool_genus_ids(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "pool_family_ids") return tensor_or_null(d.pool_family_ids(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "pool_weights") return tensor_or_null(d.pool_weights(), RESOLVE_VALUE_DOUBLE_MATRIX);
        // A boolean mask travels as an integer matrix (the value tree has no
        // boolean matrix).
        if (w == "pool_mask") return tensor_or_null(d.pool_mask(), RESOLVE_VALUE_INT_MATRIX);
        if (w == "pool_has_cover") return tensor_or_null(d.pool_has_cover(), RESOLVE_VALUE_DOUBLE_ARRAY);
        if (w == "has_pool_data") return v_bool(d.has_pool_data());
        if (w == "categorical_vocab") return categorical_vocab_to_value(d.categorical_vocab());
        if (w == "targets") return tensor_map_to_value(d.targets(), RESOLVE_VALUE_DOUBLE_ARRAY);
        if (w == "plot_ids") return v_string_array(d.plot_ids());
        if (w == "species_vocab") return v_string_array(d.species_vocab());
        if (w == "n_plots") return v_int(d.n_plots());
        if (w == "has_raw_species_data") return v_bool(d.has_raw_species_data());
        // int64 on the wire: the raw species IDs are 64-bit hash keys and the
        // CSR offsets index every record, so neither may pass through float.
        if (w == "raw_species_ids") return tensor_or_null(d.raw_species_ids(), RESOLVE_VALUE_INT_ARRAY);
        if (w == "raw_weights") return tensor_or_null(d.raw_weights(), RESOLVE_VALUE_DOUBLE_ARRAY);
        if (w == "plot_offsets") return tensor_or_null(d.plot_offsets(), RESOLVE_VALUE_INT_ARRAY);
        // The taxonomy of the vocabulary: index = species code, value = its
        // genus / family code (species_graph.hpp builds its edges from these).
        if (w == "species_genus_ids") return tensor_or_null(d.species_genus_ids(), RESOLVE_VALUE_INT_ARRAY);
        if (w == "species_family_ids") return tensor_or_null(d.species_family_ids(), RESOLVE_VALUE_INT_ARRAY);
        if (w == "taxonomy_vocab") {
            const auto& tv = d.taxonomy_vocab();
            ValueGuard g(v_map());
            v_put(g.p, "n_genera", v_int(tv.n_genera()));
            v_put(g.p, "n_families", v_int(tv.n_families()));
            return g.release();
        }
        if (w == "config") return config_to_value(d.config());
        if (w == "schema") return schema_to_value(d.schema());
        // Every vocabulary this dataset fitted, in the carrier the
        // vocabulary-reusing loaders take.
        if (w == "vocabs") return external_vocabs_to_value(d.external_vocabs());
        throw std::invalid_argument("dataset_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"

// ============================================================================
// Model
// ============================================================================

namespace {

// The PlotInputs fields an input map carries, each under its field name, read
// as float32 for a float field and through int64 for an integer or boolean one
// (the value tree has no boolean matrix), then put in the field's dtype and
// layout (a dense species_vector matrix becomes sparse rows).
PlotInputs extract_model_inputs(const resolve_value* in) {
    PlotInputs x;
    x.for_each([&](const PlotInputField& field, torch::Tensor& t) {
        t = conform_to_field(field, opt_tensor(in, field.name, field.dtype == torch::kFloat32
                                                                   ? torch::kFloat32
                                                                   : torch::kInt64));
    });
    return x;
}

// The ContinuousInputs parts an input map carries, each under its field name.
ContinuousInputs extract_continuous_parts(const resolve_value* in) {
    ContinuousInputs parts;
    parts.for_each([&](const char* name, torch::Tensor& t) { t = opt_tensor(in, name, torch::kFloat32); });
    return parts;
}

resolve_value* pretrain_result_to_value(const PretrainResult& r) {
    ValueGuard g(v_map());
    v_put(g.p, "loss_history", v_double_array(r.loss_history));
    v_put(g.p, "total_time_seconds", v_double(r.total_time_seconds));
    v_put(g.p, "epochs_completed", v_int(r.epochs_completed));
    v_put(g.p, "effective_batch_size", v_int(r.effective_batch_size));
    return g.release();
}

}  // namespace

extern "C" {

resolve_model_t* resolve_model_create(const resolve_value_t* schema, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        return new resolve_model{ResolveModel(value_to_schema(schema), parse_model_config(config))};
    })
}

resolve_model_t* resolve_model_create_for_dataset(const resolve_dataset_t* ds,
                                                  const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!ds) throw std::invalid_argument("model_create_for_dataset: null dataset handle");
        return new resolve_model{ResolveModel(
            ds->ds.schema(), model_config_for_dataset(ds->ds, parse_model_config(config)))};
    })
}

void resolve_model_free(resolve_model_t* m) { delete m; }

resolve_value_t* resolve_model_call(resolve_model_t* m, const char* method, const resolve_value_t* inputs) {
    CAPI_BODY_PTR({
        if (!m) throw std::invalid_argument("model_call: null handle");
        const std::string mt = method ? method : "";
        const PlotInputs in = extract_model_inputs(inputs);

        if (mt == "forward") {
            return tensor_map_to_value(m->model->forward(in), RESOLVE_VALUE_DOUBLE_ARRAY);
        }
        if (mt == "get_latent") {
            // (batch, latent_dim) as a matrix, one row per plot.
            return tensor_to_mat(m->model->get_latent(in));
        }
        if (mt == "forward_with_aux") {
            auto result = m->model->forward_with_aux(in);
            ValueGuard g(v_map());
            v_put(g.p, "outputs", tensor_map_to_value(result.outputs, RESOLVE_VALUE_DOUBLE_ARRAY));
            auto* aux = v_map();
            v_put(g.p, "aux_losses", aux);
            for (const auto& [name, term] : result.aux_losses) {
                if (term.defined()) v_put(aux, name, v_double(term.detach().item<double>()));
            }
            return g.release();
        }
        if (mt == "forward_single") {
            return tensor_to_vec(m->model->forward_single(vstr(inputs, "target"), in));
        }
        if (mt == "encode_with_activations") {
            auto [latent, activations] = m->model->encode_with_activations(in);
            ValueGuard g(v_map());
            v_put(g.p, "latent", tensor_to_mat(latent));
            auto* acts = v_list();
            v_put(g.p, "activations", acts);
            for (const auto& a : activations) v_append(acts, tensor_to_mat(a));
            return g.release();
        }
        if (mt == "get_gate_probs") {
            return tensor_to_mat(m->model->get_gate_probs(in));
        }
        throw std::invalid_argument("model_call: unknown method '" + mt + "'");
    })
}

resolve_value_t* resolve_model_get(const resolve_model_t* m, const char* what) {
    CAPI_BODY_PTR({
        if (!m) throw std::invalid_argument("model_get: null handle");
        const std::string w = what ? what : "";
        if (w == "latent_dim") return v_int(m->model->latent_dim());
        if (w == "uses_explicit_vector") return v_bool(m->model->uses_explicit_vector());
        if (w == "uses_moe") return v_bool(m->model->uses_moe());
        if (w == "n_experts") return v_int(m->model->n_experts());
        if (w == "species_encoding") return v_string(enum_to_name(m->model->species_encoding()));
        if (w == "config") return config_to_value(m->model->config());
        if (w == "genus_weights") return tensor_or_null(m->model->get_genus_weights(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "family_weights") return tensor_or_null(m->model->get_family_weights(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "species_weights") return tensor_or_null(m->model->get_species_weights(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "composition_parameters") {
            ValueGuard g(v_list());
            for (const auto& t : m->model->composition_parameters()) v_append(g.p, tensor_to_mat(t));
            return g.release();
        }
        if (w == "aux_loss_names") return v_string_array(m->model->aux_loss_names());
        if (w == "row_reference_rows") return v_int(m->model->row_reference_rows());
        // The species graph a HeterogeneousGNN passes messages on.
        if (w == "requires_species_graph") return v_bool(m->model->requires_species_graph());
        if (w == "has_species_graph") return v_bool(m->model->has_species_graph());
        if (w == "species_graph_edge_index") {
            return tensor_or_null(m->model->species_graph_edge_index(), RESOLVE_VALUE_INT_MATRIX);
        }
        if (w == "species_graph_edge_type") {
            return tensor_or_null(m->model->species_graph_edge_type(), RESOLVE_VALUE_INT_ARRAY);
        }
        throw std::invalid_argument("model_get: unknown accessor '" + w + "'");
    })
}

int resolve_model_set_train(resolve_model_t* m, int mode) {
    CAPI_BODY_INT({
        if (!m) throw std::invalid_argument("model_set_train: null handle");
        m->model->train(mode != 0);
        return 0;
    })
}
int resolve_model_to_device(resolve_model_t* m, const char* device) {
    CAPI_BODY_INT({
        if (!m) throw std::invalid_argument("model_to_device: null handle");
        m->model->to(device_arg(device));
        return 0;
    })
}
int resolve_model_set_traits(resolve_model_t* m, const resolve_value_t* traits) {
    CAPI_BODY_INT({
        if (!m) throw std::invalid_argument("model_set_traits: null handle");
        m->model->set_traits(value_to_tensor(traits, torch::kFloat32, "traits"));
        return 0;
    })
}
int resolve_model_set_species_graph(resolve_model_t* m,
                                    const resolve_value_t* edge_index,
                                    const resolve_value_t* edge_type) {
    CAPI_BODY_INT({
        if (!m) throw std::invalid_argument("model_set_species_graph: null handle");
        m->model->set_species_graph(value_to_tensor(edge_index, torch::kInt64, "edge_index"),
                                    value_to_tensor(edge_type, torch::kInt64, "edge_type").reshape({-1}));
        return 0;
    })
}
resolve_value_t* resolve_build_species_graph(const resolve_dataset_t* ds,
                                             const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!ds) throw std::invalid_argument("build_species_graph: null handle");
        const auto graph = build_species_graph(
            ds->ds, parse_config<HeterogeneousGNNConfig>(config, "heterogeneous_gnn config"));
        ValueGuard g(v_map());
        v_put(g.p, "edge_index", tensor_to_imat(graph.edge_index));
        v_put(g.p, "edge_type", tensor_to_ivec(graph.edge_type));
        v_put(g.p, "n_species", v_int(graph.n_species));
        v_put(g.p, "n_edges", v_int(graph.n_edges()));
        return g.release();
    })
}

resolve_value_t* resolve_model_pretrain(resolve_model_t* m, const char* method,
                                        const resolve_dataset_t* ds,
                                        const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!m || !ds) throw std::invalid_argument("model_pretrain: null handle");
        const std::string name = method ? method : "";
        const PlotInputs inputs = pretraining_inputs(ds->ds, m->model);
        if (name == "jepa") {
            JEPAPretrainer pretrainer(m->model, parse_config<PretrainConfig>(config, "pretrain config"));
            return pretrain_result_to_value(pretrainer.pretrain(inputs));
        }
        if (name == "scarf") {
            SCARFPretrainer pretrainer(m->model, parse_config<PretrainConfig>(config, "pretrain config"));
            return pretrain_result_to_value(pretrainer.pretrain(inputs));
        }
        if (name == "masked_species") {
            MaskedSpeciesPretrainer pretrainer(
                m->model, parse_config<MLMPretrainConfig>(config, "masked_species config"));
            return pretrain_result_to_value(pretrainer.pretrain(inputs));
        }
        throw std::invalid_argument("model_pretrain: unknown method '" + name +
                                    "'. Known: jepa, scarf, masked_species");
    })
}

resolve_value_t* resolve_vae_pretrain(const resolve_dataset_t* ds, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!ds) throw std::invalid_argument("vae_pretrain: null handle");
        const torch::Tensor& species_vector = ds->ds.species_vector();
        if (!is_sparse_rows(species_vector)) {
            throw std::invalid_argument(
                "vae_pretrain: the dataset carries no species vector; load it with "
                "species_encoding = \"sparse\"");
        }
        VAEPretrainer pretrainer(species_vector.size(1),
                                 parse_config<VAEConfig>(config, "vae config"));
        PlotInputs inputs;
        inputs.species_vector = species_vector;
        const VAEPretrainResult result = pretrainer.pretrain(inputs);
        ValueGuard g(v_map());
        v_put(g.p, "loss_history", v_double_array(result.loss_history));
        v_put(g.p, "recon_loss_history", v_double_array(result.recon_loss_history));
        v_put(g.p, "kl_loss_history", v_double_array(result.kl_loss_history));
        v_put(g.p, "total_time_seconds", v_double(result.total_time_seconds));
        v_put(g.p, "epochs_completed", v_int(result.epochs_completed));
        v_put(g.p, "effective_batch_size", v_int(result.effective_batch_size));
        v_put(g.p, "latent_dim", v_int(pretrainer.vae()->latent_dim()));
        v_put(g.p, "projection_weights", tensor_to_mat(pretrainer.vae()->get_projection_weights()));
        return g.release();
    })
}

}  // extern "C"

// ============================================================================
// Trainer
// ============================================================================

extern "C" {

resolve_trainer_t* resolve_trainer_create(resolve_model_t* model, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!model) throw std::invalid_argument("trainer_create: null model handle");
        return new resolve_trainer{
            std::make_unique<resolve::Trainer>(model->model, parse_train_config(config))};
    })
}

void resolve_trainer_free(resolve_trainer_t* t) { delete t; }

int resolve_trainer_prepare_data(resolve_trainer_t* t, const resolve_value_t* in, double test_size, int seed) {
    CAPI_BODY_INT({
        if (!t) throw std::invalid_argument("prepare_data: null handle");
        TensorMap target_map;
        const resolve_value* targets = vget(in, "targets");
        if (targets && targets->kind != RESOLVE_VALUE_NULL) {
            if (targets->kind != RESOLVE_VALUE_MAP) wrong_kind("targets", "a map of name -> values");
            for (size_t i = 0; i < targets->keys.size(); ++i) {
                target_map[targets->keys[i]] =
                    value_to_tensor(targets->vals[i], torch::kFloat32, targets->keys[i]);
            }
        }
        t->trainer->prepare_data(extract_continuous_parts(in), extract_model_inputs(in),
                                 target_map, static_cast<float>(test_size), seed);
        return 0;
    })
}

int resolve_trainer_prepare_data_from_dataset(resolve_trainer_t* t, const resolve_dataset_t* ds, double test_size, int seed) {
    CAPI_BODY_INT({
        if (!t || !ds) throw std::invalid_argument("prepare_data_from_dataset: null handle");
        t->trainer->prepare_data(ds->ds, static_cast<float>(test_size), seed);
        return 0;
    })
}

int resolve_trainer_prepare_data_stopping_fold(resolve_trainer_t* t, const resolve_dataset_t* ds,
                                               const resolve_value_t* stopping, int seed) {
    CAPI_BODY_INT({
        if (!t || !ds) throw std::invalid_argument("prepare_data_stopping_fold: null handle");
        std::vector<int64_t> indices;
        if (stopping && stopping->kind == RESOLVE_VALUE_INT_ARRAY) {
            indices = stopping->iarr;
        } else if (stopping && stopping->kind == RESOLVE_VALUE_STRING_ARRAY) {
            indices = ds->ds.plot_indices(stopping->sarr);
        } else {
            throw std::invalid_argument(
                "prepare_data_stopping_fold: the stopping fold is an INT_ARRAY "
                "of plot positions or a STRING_ARRAY of plot IDs");
        }
        t->trainer->prepare_data(ds->ds, indices, seed);
        return 0;
    })
}

resolve_value_t* resolve_trainer_fit(resolve_trainer_t* t) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("fit: null handle");
        return train_result_to_value(t->trainer->fit());
    })
}

resolve_value_t* resolve_trainer_resume(resolve_trainer_t* t, const char* checkpoint_path) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("resume: null handle");
        if (!checkpoint_path) throw std::invalid_argument("resume: null checkpoint path");
        return train_result_to_value(t->trainer->resume(checkpoint_path));
    })
}

int resolve_trainer_save(resolve_trainer_t* t, const char* path, const resolve_value_t* metadata) {
    CAPI_BODY_INT({
        if (!t) throw std::invalid_argument("save: null handle");
        if (!path) throw std::invalid_argument("save: null path");
        if (metadata && metadata->kind != RESOLVE_VALUE_NULL) {
            if (metadata->kind != RESOLVE_VALUE_MAP) wrong_kind("run metadata", "a map");
            const RunMetadata rm = parse_run_metadata(metadata);
            t->trainer->save(path, &rm);
        } else {
            t->trainer->save(path);
        }
        return 0;
    })
}

int resolve_trainer_load_state(resolve_trainer_t* t, const char* path, const char* device, double vram_fraction) {
    CAPI_BODY_INT({
        if (!t) throw std::invalid_argument("load_state: null handle");
        if (!path) throw std::invalid_argument("load_state: null path");
        t->trainer->load_state(path, device_arg(device), static_cast<float>(vram_fraction));
        return 0;
    })
}

resolve_value_t* resolve_trainer_get(const resolve_trainer_t* t, const char* what) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("trainer_get: null handle");
        const std::string w = what ? what : "";
        const resolve::Trainer& tr = *t->trainer;
        if (w == "scalers") return scalers_to_value(tr.scalers());
        if (w == "categorical_vocab") return categorical_vocab_to_value(tr.categorical_vocab());
        if (w == "test_indices") return ivec_or_empty(tr.test_indices());
        if (w == "train_indices") return ivec_or_empty(tr.train_indices());
        if (w == "test_plot_ids") return v_string_array(tr.test_plot_ids());
        if (w == "train_plot_ids") return v_string_array(tr.train_plot_ids());
        if (w == "effective_batch_size") return v_int(tr.effective_batch_size());
        if (w == "run_metadata") return run_metadata_to_value(tr.run_metadata());
        // Every TrainConfig field, the ones a checkpoint does not persist
        // (device, checkpoint destination, AMP / cuDNN switches) included.
        if (w == "config") return config_to_value(tr.config());
        throw std::invalid_argument("trainer_get: unknown accessor '" + w + "'");
    })
}

resolve_value_t* resolve_trainer_compute(resolve_trainer_t* t, const char* kind, const resolve_value_t* args) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("compute: null handle");
        const std::string k = kind ? kind : "";
        resolve::Trainer& tr = *t->trainer;
        if (k == "diagnostics") {
            reject_unknown_keys(args, "compute diagnostics", {});
            return network_diagnostics_to_value(tr.compute_diagnostics());
        }
        if (k == "calibration") {
            reject_unknown_keys(args, "compute calibration", {"target_name", "n_bins"});
            const int n_bins = vhas(args, "n_bins") ? vnum<int>(args, "n_bins") : 10;
            return calibration_result_to_value(tr.compute_calibration(vstr(args, "target_name"), n_bins));
        }
        if (k == "residuals") {
            reject_unknown_keys(args, "compute residuals", {"target_name"});
            return residual_analysis_to_value(tr.compute_residuals(vstr(args, "target_name")));
        }
        if (k == "classification_predictions") {
            reject_unknown_keys(args, "compute classification_predictions", {"target_name"});
            return classification_predictions_to_value(
                tr.compute_classification_predictions(vstr(args, "target_name")));
        }
        throw std::invalid_argument("compute: unknown kind '" + k + "'");
    })
}

resolve_value_t* resolve_trainer_cross_validate(resolve_trainer_t* t, int n_folds, int seed) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("cross_validate: null handle");
        return cross_validation_result_to_value(t->trainer->cross_validate(n_folds, seed));
    })
}

resolve_value_t* resolve_trainer_cross_validate_spatial(resolve_trainer_t* t, const resolve_value_t* spatial_cfg, int n_folds, int seed) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("cross_validate_spatial: null handle");
        return cross_validation_result_to_value(t->trainer->cross_validate_spatial(
            parse_config<SpatialBlockConfig>(spatial_cfg, "spatial block config"), n_folds, seed));
    })
}

resolve_value_t* resolve_trainer_predict(resolve_trainer_t* t, const resolve_value_t* in) {
    CAPI_BODY_PTR({
        if (!t) throw std::invalid_argument("predict: null handle");
        return tensor_map_to_value(t->trainer->predict(extract_model_inputs(in)),
                                   RESOLVE_VALUE_DOUBLE_ARRAY);
    })
}

resolve_value_t* resolve_load_train_config(const char* path) {
    CAPI_BODY_PTR({
        if (!path) throw std::invalid_argument("load_train_config: null path");
        return config_to_value(resolve::Trainer::load_train_config(path));
    })
}
resolve_value_t* resolve_load_run_metadata(const char* path) {
    CAPI_BODY_PTR({
        if (!path) throw std::invalid_argument("load_run_metadata: null path");
        return run_metadata_to_value(resolve::Trainer::load_run_metadata(path));
    })
}

}  // extern "C"

// ============================================================================
// Predictor
// ============================================================================

namespace {

// The standard predictions result (predictions / probabilities / targets /
// plot_ids / optional latent) shared by both predict paths.
resolve_value* predictions_to_value(const ResolvePredictions& preds, bool return_latent) {
    ValueGuard g(v_map());
    v_put(g.p, "predictions", tensor_map_to_value(preds.predictions, RESOLVE_VALUE_DOUBLE_ARRAY));
    v_put(g.p, "probabilities", tensor_map_to_value(preds.probabilities, RESOLVE_VALUE_DOUBLE_MATRIX));
    v_put(g.p, "targets", tensor_map_to_value(preds.targets, RESOLVE_VALUE_DOUBLE_ARRAY));
    if (return_latent && preds.latent.defined()) v_put(g.p, "latent", tensor_to_mat(preds.latent));
    v_put(g.p, "plot_ids", v_string_array(preds.plot_ids));
    return g.release();
}

}  // namespace

extern "C" {

resolve_predictor_t* resolve_predictor_load(const char* path, const char* device, double vram_fraction) {
    CAPI_BODY_PTR({
        if (!path) throw std::invalid_argument("predictor_load: null path");
        return new resolve_predictor{
            resolve::Predictor::load(path, device_arg(device), static_cast<float>(vram_fraction))};
    })
}

void resolve_predictor_free(resolve_predictor_t* p) { delete p; }

resolve_value_t* resolve_predictor_predict(resolve_predictor_t* p, const resolve_value_t* in, int return_latent) {
    CAPI_BODY_PTR({
        if (!p) throw std::invalid_argument("predict: null handle");
        const auto preds = p->predictor.predict(extract_continuous_parts(in),
                                                extract_model_inputs(in), return_latent != 0);
        return predictions_to_value(preds, return_latent != 0);
    })
}

resolve_value_t* resolve_predictor_predict_dataset(resolve_predictor_t* p, const resolve_dataset_t* ds, int return_latent, int64_t batch_size) {
    CAPI_BODY_PTR({
        if (!p || !ds) throw std::invalid_argument("predict_dataset: null handle");
        const ResolvePredictions preds = p->predictor.predict(ds->ds, return_latent != 0, batch_size);
        return predictions_to_value(preds, return_latent != 0);
    })
}

resolve_value_t* resolve_predictor_get_embeddings(resolve_predictor_t* p, const resolve_value_t* in) {
    CAPI_BODY_PTR({
        if (!p) throw std::invalid_argument("get_embeddings: null handle");
        return tensor_to_mat(p->predictor.get_embeddings(extract_continuous_parts(in),
                                                         extract_model_inputs(in)));
    })
}

int resolve_predictor_optimize_for_inference(resolve_predictor_t* p) {
    CAPI_BODY_INT({
        if (!p) throw std::invalid_argument("optimize_for_inference: null handle");
        p->predictor.optimize_for_inference();
        return 0;
    })
}

int resolve_predictor_set_retrieval(resolve_predictor_t* p, const resolve_dataset_t* reference,
                                    const resolve_value_t* config, int64_t batch_size) {
    CAPI_BODY_INT({
        if (!p || !reference) throw std::invalid_argument("set_retrieval: null handle");
        p->predictor.set_retrieval(reference->ds,
                                   parse_config<RetrievalConfig>(config, "retrieval config"),
                                   batch_size);
        return 0;
    })
}

int resolve_predictor_clear_retrieval(resolve_predictor_t* p) {
    CAPI_BODY_INT({
        if (!p) throw std::invalid_argument("clear_retrieval: null handle");
        p->predictor.clear_retrieval();
        return 0;
    })
}

resolve_value_t* resolve_predictor_get(const resolve_predictor_t* p, const char* what) {
    CAPI_BODY_PTR({
        if (!p) throw std::invalid_argument("predictor_get: null handle");
        const std::string w = what ? what : "";
        const resolve::Predictor& pr = p->predictor;
        if (w == "device") return v_string(device_name(pr.device()));
        if (w == "scalers") return scalers_to_value(pr.scalers());
        if (w == "categorical_vocab") return categorical_vocab_to_value(pr.categorical_vocab());
        // The checkpoint's schema (loader settings + fitted species / taxonomy
        // vocabularies) and the complete vocabulary carrier the
        // vocabulary-reusing dataset loaders take. `dataset_config` is the
        // DatasetConfig the checkpoint implies, ready to pass to a loader.
        if (w == "schema") return schema_to_value(pr.schema());
        if (w == "vocabs") return external_vocabs_to_value(pr.external_vocabs());
        if (w == "species_vocab") return v_string_array(pr.species_vocab());
        if (w == "genus_vocab") return v_string_array(pr.genus_vocab());
        if (w == "family_vocab") return v_string_array(pr.family_vocab());
        if (w == "dataset_config") {
            return config_to_value(dataset_config_from_checkpoint(pr.schema(), pr.model()->config()));
        }
        if (w == "model_config") return config_to_value(pr.model()->config());
        if (w == "genus_embeddings") return tensor_or_null(pr.get_genus_embeddings(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "family_embeddings") return tensor_or_null(pr.get_family_embeddings(), RESOLVE_VALUE_DOUBLE_MATRIX);
        if (w == "species_embeddings") return tensor_or_null(pr.get_species_embeddings(), RESOLVE_VALUE_DOUBLE_MATRIX);
        // The retrieval predict() applies: its settings and reference size,
        // or NULL-kind when none is set.
        if (w == "retrieval") {
            const RetrievalIndex* retrieval = pr.retrieval();
            if (retrieval == nullptr) return v_null();
            ValueGuard g(config_to_value(retrieval->config()));
            v_put(g.p, "n_reference", v_int(retrieval->n_reference()));
            return g.release();
        }
        throw std::invalid_argument("predictor_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"

// ============================================================================
// Model suite
// ============================================================================

namespace {

// A JSON document as a value tree: objects keep their member order.
resolve_value* json_to_value(const json::Value& j) {
    switch (j.kind()) {
        case json::Value::Kind::Null: return v_null();
        case json::Value::Kind::Bool: return v_bool(j.as_bool());
        case json::Value::Kind::Number: return v_double(j.as_number());
        case json::Value::Kind::String: return v_string(j.as_string());
        case json::Value::Kind::Array: {
            ValueGuard g(v_list());
            for (const auto& item : j.items()) v_append(g.p, json_to_value(item));
            return g.release();
        }
        case json::Value::Kind::Object: {
            ValueGuard g(v_map());
            for (const auto& [key, value] : j.members()) v_put(g.p, key, json_to_value(value));
            return g.release();
        }
    }
    return v_null();
}

SuiteLoadOptions parse_suite_load_options(const resolve_value* o) {
    SuiteLoadOptions out;
    if (!o || o->kind == RESOLVE_VALUE_NULL) return out;
    reject_unknown_keys(o, "suite load options", {"device", "vram_fraction", "verify", "targets"});
    if (vhas(o, "device")) out.device = parse_device(vstr(o, "device"));
    if (vhas(o, "vram_fraction")) out.vram_fraction = vnum<float>(o, "vram_fraction");
    if (vhas(o, "verify")) out.verify_checksums = vbool(o, "verify");
    if (vhas(o, "targets")) out.targets = vstr_vec(o, "targets");
    return out;
}

SuitePredictOptions parse_suite_predict_options(const resolve_value* o) {
    SuitePredictOptions out;
    if (!o || o->kind == RESOLVE_VALUE_NULL) return out;
    reject_unknown_keys(o, "suite predict options", {"columns", "batch_size", "keep_members"});
    if (vhas(o, "batch_size")) out.batch_size = vnum<int64_t>(o, "batch_size");
    if (vhas(o, "keep_members")) out.keep_members = vbool(o, "keep_members");
    const resolve_value* c = vget(o, "columns");
    if (c && c->kind != RESOLVE_VALUE_NULL) {
        reject_unknown_keys(c, "suite columns",
                            {"plot_id", "species_id", "abundance", "longitude", "latitude",
                             "genus", "family", "covariates", "categoricals"});
        SuiteColumns& cols = out.columns;
        if (vhas(c, "plot_id")) cols.plot_id = vstr(c, "plot_id");
        if (vhas(c, "species_id")) cols.species_id = vstr(c, "species_id");
        if (vhas(c, "abundance")) cols.abundance = vstr(c, "abundance");
        if (vhas(c, "longitude")) cols.longitude = vstr(c, "longitude");
        if (vhas(c, "latitude")) cols.latitude = vstr(c, "latitude");
        if (vhas(c, "genus")) cols.genus = vstr(c, "genus");
        if (vhas(c, "family")) cols.family = vstr(c, "family");
        if (vhas(c, "covariates")) cols.covariates = vstr_vec(c, "covariates");
        if (vhas(c, "categoricals")) cols.categoricals = vstr_vec(c, "categoricals");
    }
    return out;
}

resolve_value* suite_predictions_to_value(const SuitePredictions& p) {
    ValueGuard g(v_map());
    resolve_value* result = g.p;
    v_put(result, "plot_ids", v_string_array(p.plot_ids));
    auto* targets = v_map();
    v_put(result, "targets", targets);
    auto defined_or_null = [](const torch::Tensor& t, resolve_value_kind_t kind) -> resolve_value* {
        return t.defined() ? tensor_to_value(t, kind) : v_null();
    };
    for (const auto& t : p.targets) {
        const bool vote = t.combine == SuiteCombine::Vote;
        const bool spherical = t.combine == SuiteCombine::SphericalMean;
        auto* m = v_map();
        v_put(targets, t.name, m);
        v_put(m, "name", v_string(t.name));
        v_put(m, "task", v_string(enum_to_name(t.task)));
        v_put(m, "combine", v_string(enum_to_name(t.combine)));
        v_put(m, "status", v_string(enum_to_name(t.status)));
        v_put(m, "limit", v_string(t.limit));
        v_put(m, "units", v_string(t.units));
        v_put(m, "dispersion_units", v_string(t.dispersion_units));
        v_put(m, "class_names", v_string_array(t.class_names));
        v_put(m, "member_seeds", v_int_array(t.member_seeds));
        v_put(m, "value", vote        ? tensor_to_ivec(t.value)
                          : spherical ? tensor_to_mat(t.value)
                                      : tensor_to_vec(t.value));
        v_put(m, "agreement", defined_or_null(t.agreement, RESOLVE_VALUE_DOUBLE_ARRAY));
        v_put(m, "probabilities", defined_or_null(t.probabilities, RESOLVE_VALUE_DOUBLE_MATRIX));
        v_put(m, "dispersion", defined_or_null(t.dispersion, RESOLVE_VALUE_DOUBLE_ARRAY));
        if (!t.members.defined()) {
            v_put(m, "members", v_null());
        } else if (spherical) {
            auto* coords = v_map();
            v_put(m, "members", coords);
            v_put(coords, "latitude", tensor_to_mat(t.members.select(2, 0)));
            v_put(coords, "longitude", tensor_to_mat(t.members.select(2, 1)));
        } else {
            v_put(m, "members", vote ? tensor_to_imat(t.members) : tensor_to_mat(t.members));
        }
        v_put(m, "n_species", defined_or_null(t.recognition.n_species, RESOLVE_VALUE_DOUBLE_ARRAY));
        v_put(m, "n_recognised", defined_or_null(t.recognition.n_recognised, RESOLVE_VALUE_DOUBLE_ARRAY));
        v_put(m, "recognised_share", defined_or_null(t.recognition.count_share, RESOLVE_VALUE_DOUBLE_ARRAY));
        v_put(m, "recognised_abundance_share",
              defined_or_null(t.recognition.abundance_share, RESOLVE_VALUE_DOUBLE_ARRAY));
    }
    // The predictions as one row per plot (suite_prediction_columns), the class
    // probabilities included and named, so a binding's data frame holds the
    // columns the CLI writes.
    auto* columns = v_map();
    v_put(result, "columns", columns);
    std::vector<std::string> probability_columns;
    for (const auto& column : suite_prediction_columns(p, /*probabilities=*/true)) {
        switch (column.kind) {
            case SuiteColumn::Kind::Text:
                v_put(columns, column.name, v_string_array(column.text));
                break;
            case SuiteColumn::Kind::Integer:
                v_put(columns, column.name, v_int_array(column.integers));
                break;
            case SuiteColumn::Kind::Number:
                v_put(columns, column.name, v_double_array(column.numbers));
                break;
        }
        if (column.probability) probability_columns.push_back(column.name);
    }
    v_put(result, "probability_columns", v_string_array(probability_columns));
    return g.release();
}

}  // namespace

extern "C" {

resolve_suite_t* resolve_suite_load(const char* dir, const resolve_value_t* options) {
    CAPI_BODY_PTR({
        if (!dir) throw std::invalid_argument("suite_load: null directory");
        return new resolve_suite{SuitePredictor::load(dir, parse_suite_load_options(options))};
    })
}

void resolve_suite_free(resolve_suite_t* s) { delete s; }

resolve_value_t* resolve_suite_get(const resolve_suite_t* s, const char* what) {
    CAPI_BODY_PTR({
        if (!s) throw std::invalid_argument("suite_get: null handle");
        const std::string w = what ? what : "";
        if (w == "manifest") return json_to_value(s->suite.manifest().to_json());
        if (w == "targets") return v_string_array(s->suite.target_names());
        if (w == "n_encodings") return v_int(static_cast<int64_t>(s->suite.n_encodings()));
        if (w == "directory") return v_string(s->suite.directory());
        throw std::invalid_argument("suite_get: unknown accessor '" + w + "'");
    })
}

resolve_value_t* resolve_suite_predict_dataframe(resolve_suite_t* s, const resolve_value_t* header,
                                                 const resolve_value_t* species,
                                                 const resolve_value_t* options) {
    CAPI_BODY_PTR({
        if (!s) throw std::invalid_argument("suite_predict: null handle");
        const bool has_header = header && header->kind != RESOLVE_VALUE_NULL;
        ColumnTable h = has_header ? value_to_column_table(header, "header") : ColumnTable();
        ColumnTable sp = value_to_column_table(species, "species");
        return suite_predictions_to_value(s->suite.predict(
            SuiteInput::tables(has_header ? &h : nullptr, sp), parse_suite_predict_options(options)));
    })
}

resolve_value_t* resolve_suite_predict_csv(resolve_suite_t* s, const char* header_path,
                                           const char* species_path,
                                           const resolve_value_t* options) {
    CAPI_BODY_PTR({
        if (!s) throw std::invalid_argument("suite_predict: null handle");
        if (!species_path) throw std::invalid_argument("suite_predict: a species CSV is required");
        return suite_predictions_to_value(s->suite.predict(
            SuiteInput::csv(header_path ? header_path : "", species_path),
            parse_suite_predict_options(options)));
    })
}

resolve_value_t* resolve_suite_verify(const char* dir) {
    CAPI_BODY_PTR({
        if (!dir) throw std::invalid_argument("suite_verify: null directory");
        const SuiteManifest manifest = SuiteManifest::read(dir);
        ValueGuard g(v_map());
        v_put(g.p, "manifest", json_to_value(manifest.to_json()));
        v_put(g.p, "problems", v_string_array(manifest.verify(dir)));
        return g.release();
    })
}

resolve_value_t* resolve_suite_seal(const char* dir) {
    CAPI_BODY_PTR({
        if (!dir) throw std::invalid_argument("suite_seal: null directory");
        const std::string path = (std::filesystem::path(dir) / kSuiteManifestFile).string();
        SuiteManifest manifest = SuiteManifest::from_json(json::read_file(path));
        manifest.seal(dir);
        manifest.write(dir);
        return json_to_value(manifest.to_json());
    })
}

}  // extern "C"

// ============================================================================
// Fuzzy string index
// ============================================================================

namespace {

// A lookup from each key to the value given beside it: the per-entry buckets of
// a build, the per-needle bucket hints of a query. A key given twice with two
// different values has no one value and is refused.
std::unordered_map<std::string, std::string> keyed_values(const std::vector<std::string>& keys,
                                                          const std::vector<std::string>& values,
                                                          const char* what) {
    if (values.size() != keys.size()) {
        throw std::invalid_argument(std::string(what) + ": " + std::to_string(values.size()) +
                                    " values for " + std::to_string(keys.size()) + " strings");
    }
    std::unordered_map<std::string, std::string> lookup;
    lookup.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto [it, inserted] = lookup.emplace(keys[i], values[i]);
        if (!inserted && it->second != values[i]) {
            throw std::invalid_argument(std::string(what) + ": '" + keys[i] +
                                        "' is given two different values");
        }
    }
    return lookup;
}

// The value `lookup` holds for `key`, as a callback the index calls per string.
std::function<std::string(std::string_view)> lookup_fn(
    const std::unordered_map<std::string, std::string>& lookup) {
    return [&lookup](std::string_view key) {
        const auto it = lookup.find(std::string(key));
        return it == lookup.end() ? std::string() : it->second;
    };
}

resolve_value* matches_to_value(const std::vector<std::vector<fuzzy::Match>>& matches) {
    std::vector<int64_t> query;
    std::vector<std::string> entry;
    std::vector<int64_t> id;
    std::vector<int64_t> distance;
    for (size_t q = 0; q < matches.size(); ++q) {
        for (const auto& match : matches[q]) {
            query.push_back(static_cast<int64_t>(q));
            entry.push_back(match.entry);
            id.push_back(match.id);
            distance.push_back(match.distance);
        }
    }
    ValueGuard g(v_map());
    v_put(g.p, "query", v_int_array(query));
    v_put(g.p, "entry", v_string_array(std::move(entry)));
    v_put(g.p, "id", v_int_array(id));
    v_put(g.p, "distance", v_int_array(distance));
    return g.release();
}

}  // namespace

extern "C" {

resolve_fuzzy_t* resolve_fuzzy_build(const resolve_value_t* entries, const resolve_value_t* options) {
    CAPI_BODY_PTR({
        const std::vector<std::string> names = vstr_vec(entries, std::string("entries"));
        reject_unknown_keys(options, "fuzzy build options",
                            {"max_edit_distance", "case_insensitive", "damerau", "buckets"});
        fuzzy::BuildOptions opts;
        if (vhas(options, "max_edit_distance")) opts.max_edit_distance = vnum<int>(options, "max_edit_distance");
        if (vhas(options, "case_insensitive")) opts.case_insensitive = vbool(options, "case_insensitive");
        if (vhas(options, "damerau")) opts.damerau = vbool(options, "damerau");
        std::unordered_map<std::string, std::string> buckets;
        if (vhas(options, "buckets")) {
            buckets = keyed_values(names, vstr_vec(options, "buckets"), "fuzzy build buckets");
            opts.bucket_fn = lookup_fn(buckets);
        }
        return new resolve_fuzzy{fuzzy::FuzzyIndex::build(names, std::move(opts))};
    })
}

void resolve_fuzzy_free(resolve_fuzzy_t* f) { delete f; }

resolve_value_t* resolve_fuzzy_query(const resolve_fuzzy_t* f, const resolve_value_t* needles,
                                     const resolve_value_t* options) {
    CAPI_BODY_PTR({
        if (!f) throw std::invalid_argument("fuzzy_query: null handle");
        const std::vector<std::string> queries = vstr_vec(needles, std::string("needles"));
        reject_unknown_keys(options, "fuzzy query options",
                            {"max_edit_distance", "top_n", "bucket_hint", "bucket_hints"});
        fuzzy::QueryOptions q;
        if (vhas(options, "max_edit_distance")) q.max_edit_distance = vnum<int>(options, "max_edit_distance");
        if (vhas(options, "top_n")) q.top_n = vnum<int>(options, "top_n");
        if (vhas(options, "bucket_hint") && vhas(options, "bucket_hints")) {
            throw std::invalid_argument("fuzzy_query: pass bucket_hint or bucket_hints, not both");
        }
        if (vhas(options, "bucket_hint")) q.bucket_hint = vstr(options, "bucket_hint");
        std::unordered_map<std::string, std::string> hints;
        std::function<std::string(std::string_view)> hint_fn;
        if (vhas(options, "bucket_hints")) {
            hints = keyed_values(queries, vstr_vec(options, "bucket_hints"), "fuzzy query bucket_hints");
            hint_fn = lookup_fn(hints);
        }
        return matches_to_value(f->index.query_batch(queries, q, hint_fn));
    })
}

resolve_value_t* resolve_fuzzy_get(const resolve_fuzzy_t* f, const char* what) {
    CAPI_BODY_PTR({
        if (!f) throw std::invalid_argument("fuzzy_get: null handle");
        const std::string w = what ? what : "";
        if (w == "size") return v_int(static_cast<int64_t>(f->index.size()));
        if (w == "bucket_count") return v_int(static_cast<int64_t>(f->index.bucket_count()));
        if (w == "max_supported_distance") return v_int(f->index.max_supported_distance());
        throw std::invalid_argument("fuzzy_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"
