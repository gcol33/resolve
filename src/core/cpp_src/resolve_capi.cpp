// resolve_capi.cpp - MSVC-side implementation of the flat C ABI facade.
//
// This is the ONLY translation unit that bridges the engine's C++ ABI to the
// flat C ABI consumed by the R package (and any other mingw / non-MSVC client).
// All tensor <-> raw-buffer marshaling, all config/enum parsing, and all
// result-struct -> value-tree conversion live here, on the MSVC side of the
// boundary. See include/resolve/resolve_capi.h and
// dev_notes/issue17_c_abi_facade_design.md.

#define RESOLVE_CAPI_BUILD 1
#include "resolve/resolve_capi.h"
#include "resolve/resolve.hpp"
#include "resolve/config_registry.hpp"

#include <torch/torch.h>

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <string>
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
// Internal builders (used to assemble result trees on the C++ side)
// ----------------------------------------------------------------------------

namespace {

// Every value-tree builder is marked noexcept. The header promises "C++
// exceptions never cross the boundary" (resolve_capi.h), but the C header cannot
// carry a C++ noexcept spec, so it is enforced here at the single choke point
// through which all allocations flow: a std::bad_alloc (the only realistic
// failure) terminates deterministically instead of unwinding into a possibly
// foreign-ABI caller. Every extern "C" builder / setter below delegates to
// these, so none of them can throw across the boundary either.
resolve_value* v_new(resolve_value_kind_t k) noexcept {
    auto* v = new resolve_value();
    v->kind = k;
    return v;
}
resolve_value* v_null() noexcept            { return v_new(RESOLVE_VALUE_NULL); }
resolve_value* v_bool(bool x) noexcept      { auto* v = v_new(RESOLVE_VALUE_BOOL);   v->b = x; return v; }
resolve_value* v_int(int64_t x) noexcept    { auto* v = v_new(RESOLVE_VALUE_INT);    v->i = x; return v; }
resolve_value* v_double(double x) noexcept  { auto* v = v_new(RESOLVE_VALUE_DOUBLE); v->d = x; return v; }
resolve_value* v_string(const std::string& x) noexcept { auto* v = v_new(RESOLVE_VALUE_STRING); v->s = x; return v; }
resolve_value* v_string(const char* x) noexcept { auto* v = v_new(RESOLVE_VALUE_STRING); if (x) v->s = x; return v; }

resolve_value* v_int_array(const std::vector<int64_t>& x) noexcept {
    auto* v = v_new(RESOLVE_VALUE_INT_ARRAY); v->iarr = x; return v;
}
// Every engine-side float vector reaching the boundary is std::vector<float>;
// the DOUBLE_ARRAY kind is the wire representation, not the source type.
resolve_value* v_double_array(const std::vector<float>& x) noexcept {
    auto* v = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    v->darr.assign(x.begin(), x.end());
    return v;
}
resolve_value* v_string_array(const std::vector<std::string>& x) noexcept {
    auto* v = v_new(RESOLVE_VALUE_STRING_ARRAY); v->sarr = x; return v;
}
// Raw-buffer overloads used by the extern "C" entry points (copy the caller's
// buffer; the caller keeps ownership of its own memory).
resolve_value* v_int_array(const int64_t* v, int64_t n) noexcept {
    auto* node = v_new(RESOLVE_VALUE_INT_ARRAY);
    node->iarr.assign(v, v + (n > 0 ? n : 0));
    return node;
}
resolve_value* v_double_array(const double* v, int64_t n) noexcept {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    node->darr.assign(v, v + (n > 0 ? n : 0));
    return node;
}
resolve_value* v_string_array(const char* const* v, int64_t n) noexcept {
    auto* node = v_new(RESOLVE_VALUE_STRING_ARRAY);
    node->sarr.reserve(n > 0 ? static_cast<size_t>(n) : 0);
    for (int64_t j = 0; j < n; ++j) node->sarr.emplace_back(v[j] ? v[j] : "");
    return node;
}
resolve_value* v_double_matrix(const double* v, int64_t nrow, int64_t ncol) noexcept {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->darr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    return node;
}
resolve_value* v_int_matrix(const int64_t* v, int64_t nrow, int64_t ncol) noexcept {
    auto* node = v_new(RESOLVE_VALUE_INT_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->iarr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    return node;
}
resolve_value* v_map() noexcept  { return v_new(RESOLVE_VALUE_MAP); }
resolve_value* v_list() noexcept { return v_new(RESOLVE_VALUE_LIST); }

// Put a child into a map (takes ownership). Used by C++ result assembly.
void v_put(resolve_value* m, const std::string& key, resolve_value* child) noexcept {
    m->keys.push_back(key);
    m->vals.push_back(child);
}
void v_append(resolve_value* l, resolve_value* child) noexcept { l->items.push_back(child); }

}  // namespace

// ----------------------------------------------------------------------------
// C API: value construction
// ----------------------------------------------------------------------------

extern "C" {

resolve_value_t* resolve_value_new_null(void) { return v_null(); }
resolve_value_t* resolve_value_new_map(void)  { return v_map(); }
resolve_value_t* resolve_value_new_list(void) { return v_list(); }
void resolve_value_free(resolve_value_t* v) { delete v; }

resolve_value_t* resolve_value_new_bool(int v)        { return v_bool(v != 0); }
resolve_value_t* resolve_value_new_int(int64_t v)     { return v_int(v); }
resolve_value_t* resolve_value_new_double(double v)   { return v_double(v); }
resolve_value_t* resolve_value_new_string(const char* v) { return v_string(v ? v : ""); }
resolve_value_t* resolve_value_new_int_array(const int64_t* v, int64_t n) {
    return v_int_array(v, n);
}
resolve_value_t* resolve_value_new_double_array(const double* v, int64_t n) {
    return v_double_array(v, n);
}
resolve_value_t* resolve_value_new_string_array(const char* const* v, int64_t n) {
    return v_string_array(v, n);
}
resolve_value_t* resolve_value_new_double_matrix(const double* v, int64_t nrow, int64_t ncol) {
    return v_double_matrix(v, nrow, ncol);
}
resolve_value_t* resolve_value_new_int_matrix(const int64_t* v, int64_t nrow, int64_t ncol) {
    return v_int_matrix(v, nrow, ncol);
}

void resolve_map_set_null  (resolve_value_t* m, const char* k) { v_put(m, k, v_null()); }
void resolve_map_set_bool  (resolve_value_t* m, const char* k, int v) { v_put(m, k, v_bool(v != 0)); }
void resolve_map_set_int   (resolve_value_t* m, const char* k, int64_t v) { v_put(m, k, v_int(v)); }
void resolve_map_set_double(resolve_value_t* m, const char* k, double v) { v_put(m, k, v_double(v)); }
void resolve_map_set_string(resolve_value_t* m, const char* k, const char* v) {
    v_put(m, k, v_string(v ? v : ""));
}
void resolve_map_set_int_array(resolve_value_t* m, const char* k, const int64_t* v, int64_t n) {
    v_put(m, k, v_int_array(v, n));
}
void resolve_map_set_double_array(resolve_value_t* m, const char* k, const double* v, int64_t n) {
    v_put(m, k, v_double_array(v, n));
}
void resolve_map_set_string_array(resolve_value_t* m, const char* k, const char* const* v, int64_t n) {
    v_put(m, k, v_string_array(v, n));
}
void resolve_map_set_double_matrix(resolve_value_t* m, const char* k, const double* v, int64_t nrow, int64_t ncol) {
    v_put(m, k, v_double_matrix(v, nrow, ncol));
}
void resolve_map_set_int_matrix(resolve_value_t* m, const char* k, const int64_t* v, int64_t nrow, int64_t ncol) {
    v_put(m, k, v_int_matrix(v, nrow, ncol));
}
void resolve_map_set_value(resolve_value_t* m, const char* k, resolve_value_t* child) {
    v_put(m, k, child);
}
void resolve_list_append_value(resolve_value_t* l, resolve_value_t* child) { v_append(l, child); }

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
// Internal helpers: value-map readers (coercion-tolerant)
// ============================================================================

namespace {

const resolve_value* vget(const resolve_value* m, const char* k) {
    return resolve_map_get(m, k);
}
bool vhas(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return v != nullptr && v->kind != RESOLVE_VALUE_NULL;
}
std::string vstr(const resolve_value* v) {
    return v ? v->s : std::string();
}
std::string vstr(const resolve_value* m, const char* k) { return vstr(vget(m, k)); }
int64_t vint(const resolve_value* v) {
    if (!v) return 0;
    switch (v->kind) {
        case RESOLVE_VALUE_INT:    return v->i;
        case RESOLVE_VALUE_DOUBLE: return static_cast<int64_t>(v->d);
        case RESOLVE_VALUE_BOOL:   return v->b ? 1 : 0;
        default:                   return 0;
    }
}
int64_t vint(const resolve_value* m, const char* k) { return vint(vget(m, k)); }
double vdbl(const resolve_value* v) {
    if (!v) return 0.0;
    switch (v->kind) {
        case RESOLVE_VALUE_DOUBLE: return v->d;
        case RESOLVE_VALUE_INT:    return static_cast<double>(v->i);
        case RESOLVE_VALUE_BOOL:   return v->b ? 1.0 : 0.0;
        default:                   return 0.0;
    }
}
double vdbl(const resolve_value* m, const char* k) { return vdbl(vget(m, k)); }
bool vbool(const resolve_value* v) {
    if (!v) return false;
    switch (v->kind) {
        case RESOLVE_VALUE_BOOL:   return v->b;
        case RESOLVE_VALUE_INT:    return v->i != 0;
        case RESOLVE_VALUE_DOUBLE: return v->d != 0.0;
        default:                   return false;
    }
}
bool vbool(const resolve_value* m, const char* k) { return vbool(vget(m, k)); }

// Narrowing conversions at the C ABI boundary are deliberate: an R or Python
// caller hands whatever numeric vector its language has (R has no int64, so a
// count arrives as double; a float weight arrives as double), and the engine
// wants the target type. Spelling the cast out per element keeps that intent
// visible instead of letting a vector range-construct convert implicitly.
template <typename Out, typename In>
std::vector<Out> narrow_vec(const std::vector<In>& src) {
    std::vector<Out> out;
    out.reserve(src.size());
    for (const In& x : src) out.push_back(static_cast<Out>(x));
    return out;
}

std::vector<int64_t> vint_vec(const resolve_value* v) {
    std::vector<int64_t> out;
    if (!v) return out;
    if (v->kind == RESOLVE_VALUE_INT_ARRAY) return v->iarr;
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        out = narrow_vec<int64_t>(v->darr);
    } else if (v->kind == RESOLVE_VALUE_INT) {
        out.push_back(v->i);
    } else if (v->kind == RESOLVE_VALUE_DOUBLE) {
        out.push_back(static_cast<int64_t>(v->d));
    }
    return out;
}
std::vector<int64_t> vint_vec(const resolve_value* m, const char* k) { return vint_vec(vget(m, k)); }

std::vector<float> vfloat_vec(const resolve_value* v) {
    std::vector<float> out;
    if (!v) return out;
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        out = narrow_vec<float>(v->darr);
    } else if (v->kind == RESOLVE_VALUE_INT_ARRAY) {
        out = narrow_vec<float>(v->iarr);
    } else if (v->kind == RESOLVE_VALUE_DOUBLE) {
        out.push_back(static_cast<float>(v->d));
    } else if (v->kind == RESOLVE_VALUE_INT) {
        out.push_back(static_cast<float>(v->i));
    }
    return out;
}
std::vector<float> vfloat_vec(const resolve_value* m, const char* k) { return vfloat_vec(vget(m, k)); }

std::vector<std::string> vstr_vec(const resolve_value* v) {
    if (v && v->kind == RESOLVE_VALUE_STRING_ARRAY) return v->sarr;
    std::vector<std::string> out;
    if (v && v->kind == RESOLVE_VALUE_STRING) out.push_back(v->s);
    return out;
}
std::vector<std::string> vstr_vec(const resolve_value* m, const char* k) { return vstr_vec(vget(m, k)); }

// MAP (ordered column name -> STRING_ARRAY) -> ColumnTable. The in-memory
// DataFrame carrier for issue #22: a data.frame with every cell stringified.
// ColumnTable's constructor validates equal-length columns and unique names.
ColumnTable value_to_column_table(const resolve_value* v, const char* what) {
    if (!v || v->kind != RESOLVE_VALUE_MAP) {
        throw std::runtime_error(std::string(what) +
            " must be a MAP value tree (ordered column name -> string array)");
    }
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> cols;
    names.reserve(v->keys.size());
    cols.reserve(v->vals.size());
    for (size_t c = 0; c < v->keys.size(); ++c) {
        const resolve_value* colv = v->vals[c];
        if (!colv || colv->kind != RESOLVE_VALUE_STRING_ARRAY) {
            throw std::runtime_error(std::string(what) + " column '" +
                v->keys[c] + "' must be a string array");
        }
        names.push_back(v->keys[c]);
        cols.push_back(colv->sarr);
    }
    return ColumnTable(std::move(names), std::move(cols));
}

// ============================================================================
// Internal helpers: value <-> torch::Tensor
// ============================================================================

// DOUBLE_MATRIX / DOUBLE_ARRAY -> float32 tensor. Element-by-element cast (R
// doubles are 64-bit; reinterpreting the bytes as float32 would be garbage).
torch::Tensor value_to_f32(const resolve_value* v) {
    if (!v || v->kind == RESOLVE_VALUE_NULL) return torch::Tensor();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    if (v->kind == RESOLVE_VALUE_DOUBLE_MATRIX) {
        std::vector<float> data = narrow_vec<float>(v->darr);
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        std::vector<float> data = narrow_vec<float>(v->darr);
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_INT_MATRIX) {
        std::vector<float> data = narrow_vec<float>(v->iarr);
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_INT_ARRAY) {
        std::vector<float> data = narrow_vec<float>(v->iarr);
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    throw std::runtime_error("value_to_f32: unsupported value kind");
}

// Raw caller-owned double buffer -> float32 tensor. The metric entry points
// all take their inputs this way (R has no float vector, and the C ABI carries
// doubles), so the per-element narrowing lives here rather than being repeated
// as an implicit vector range-construct in each one.
torch::Tensor f32_tensor_from_doubles(const double* src, int64_t n) {
    std::vector<float> data;
    data.reserve(static_cast<size_t>(n < 0 ? 0 : n));
    for (int64_t i = 0; i < n; ++i) data.push_back(static_cast<float>(src[i]));
    return torch::from_blob(data.data(), {n}, torch::kFloat32).clone();
}

// INT_MATRIX / INT_ARRAY -> int64 tensor.
torch::Tensor value_to_i64(const resolve_value* v) {
    if (!v || v->kind == RESOLVE_VALUE_NULL) return torch::Tensor();
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    if (v->kind == RESOLVE_VALUE_INT_MATRIX) {
        std::vector<int64_t> data = v->iarr;
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_INT_ARRAY) {
        std::vector<int64_t> data = v->iarr;
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_DOUBLE_MATRIX) {
        std::vector<int64_t> data = narrow_vec<int64_t>(v->darr);
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        std::vector<int64_t> data = narrow_vec<int64_t>(v->darr);
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    throw std::runtime_error("value_to_i64: unsupported value kind");
}

// Optional tensor from a map key. Returns an undefined tensor when the key is
// absent or NULL-kind (mirrors the old Nullable<...> "absent" path).
torch::Tensor opt_f32(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return (v && v->kind != RESOLVE_VALUE_NULL) ? value_to_f32(v) : torch::Tensor();
}
torch::Tensor opt_i64(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return (v && v->kind != RESOLVE_VALUE_NULL) ? value_to_i64(v) : torch::Tensor();
}

// float32 (2-D) tensor -> DOUBLE_MATRIX value. Mirrors tensor_to_r_mat.
resolve_value* tensor_to_mat(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    auto* v = v_new(RESOLVE_VALUE_DOUBLE_MATRIX);
    v->nrow = cpu.size(0);
    v->ncol = cpu.dim() >= 2 ? cpu.size(1) : 1;
    const float* p = cpu.data_ptr<float>();
    v->darr.assign(p, p + cpu.numel());
    return v;
}
// int64 (2-D) tensor -> INT_MATRIX value. Mirrors tensor_to_r_imat.
resolve_value* tensor_to_imat(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kInt64);
    auto* v = v_new(RESOLVE_VALUE_INT_MATRIX);
    v->nrow = cpu.size(0);
    v->ncol = cpu.dim() >= 2 ? cpu.size(1) : 1;
    const int64_t* p = cpu.data_ptr<int64_t>();
    v->iarr.assign(p, p + cpu.numel());
    return v;
}
// tensor (any shape) -> flat DOUBLE_ARRAY. Mirrors tensor_to_r_vec.
resolve_value* tensor_to_vec(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kFloat32);
    auto* v = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    const float* p = cpu.data_ptr<float>();
    v->darr.assign(p, p + cpu.numel());
    return v;
}
// int64 tensor (any shape) -> flat INT_ARRAY. Mirrors tensor_to_r_ivec.
resolve_value* tensor_to_ivec(const torch::Tensor& t) {
    torch::Tensor cpu = t.cpu().contiguous().to(torch::kInt64);
    auto* v = v_new(RESOLVE_VALUE_INT_ARRAY);
    const int64_t* p = cpu.data_ptr<int64_t>();
    v->iarr.assign(p, p + cpu.numel());
    return v;
}

}  // namespace

// ============================================================================
// Enum parsers / emitters
// ============================================================================
//
// The string <-> enum tables these call sites use live in
// include/resolve/enum_names.hpp, one table per enum, read by both the parser
// and the emitter. They are shared with the CLI, which parses the same
// spellings off its flags and prints them back in `resolve info`. The
// file-level `using namespace resolve` brings parse_selection_mode /
// selection_mode_to_string / ... into scope unqualified.

namespace {

// ============================================================================
// Config <-> value tree, driven by the field registry
// ============================================================================
//
// A registry row's member name IS its value-tree key, so one visitor reads a map
// into a config struct and its inverse writes the struct back out, both walking
// the list in resolve/config_registry.hpp. Neither side can carry a field the
// other drops, and the enum spellings come from enum_names.hpp, which the CLI
// and `resolve info` read too (issue #108).

// Value map -> config struct. An absent key leaves the struct default in place,
// which is what makes every entry of an R config list optional.
struct ValueFieldReader {
    const resolve_value* map;

    template <typename T>
    void operator()(const char* name, const char*, T& value) const {
        if constexpr (std::is_same_v<T, LogCallback>) {
            // A callback cannot cross the C boundary.
            (void)name;
            (void)value;
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            if (vhas(map, name)) {
                value = (vstr(map, name) == "cuda") ? torch::kCUDA : torch::kCPU;
            }
        } else if constexpr (is_registered_config_v<T>) {
            if (vhas(map, name)) for_each_field(value, ValueFieldReader{vget(map, name)});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            const resolve_value* list = vget(map, name);
            if (list && list->kind == RESOLVE_VALUE_LIST) {
                value.clear();
                for (auto* item : list->items) {
                    ParallelBranchConfig branch;
                    for_each_field(branch, ValueFieldReader{item});
                    value.push_back(std::move(branch));
                }
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            if (vhas(map, name)) value = vbool(map, name);
        } else if constexpr (std::is_enum_v<T>) {
            if (vhas(map, name)) value = enum_from_name<T>(vstr(map, name));
        } else if constexpr (std::is_same_v<T, int>) {
            if (vhas(map, name)) value = static_cast<int>(vint(map, name));
        } else if constexpr (std::is_same_v<T, float>) {
            if (vhas(map, name)) value = static_cast<float>(vdbl(map, name));
        } else if constexpr (std::is_same_v<T, std::string>) {
            if (vhas(map, name)) value = vstr(map, name);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            if (vhas(map, name)) value = vint_vec(map, name);
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            if (vhas(map, name)) value = vfloat_vec(map, name);
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            if (vhas(map, name)) {
                const std::vector<int64_t> pair_values = vint_vec(map, name);
                if (pair_values.size() >= 2) {
                    value = {static_cast<int>(pair_values[0]),
                             static_cast<int>(pair_values[1])};
                }
            }
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no value-tree representation; "
                          "add a branch here and to ValueFieldWriter");
        }
    }
};

// Config struct -> value map. A child node is attached to its parent before it is
// filled, so a throw part-way through leaves it owned (and freed) by the root.
struct ValueFieldWriter {
    resolve_value* map;

    template <typename T>
    void operator()(const char* name, const char*, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback>) {
            (void)name;
        } else if constexpr (std::is_same_v<T, torch::Device>) {
            v_put(map, name, v_string(value.is_cuda() ? "cuda" : "cpu"));
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
        } else if constexpr (std::is_same_v<T, int>) {
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

// SpatialBlockConfig is a cross-validation argument rather than a persisted
// config, so it has no registry: it is parsed here and never written back.
SpatialBlockConfig parse_spatial_block_config(const resolve_value* c) {
    SpatialBlockConfig x;
    if (vhas(c, "lat_size")) x.lat_size = (float)vdbl(c, "lat_size");
    if (vhas(c, "lon_size")) x.lon_size = (float)vdbl(c, "lon_size");
    if (vhas(c, "balance")) x.balance = vbool(c, "balance");
    return x;
}

// ============================================================================
// Top-level parsers: roles / targets / dataset config / model config / schema
// ============================================================================

RoleMapping parse_roles(const resolve_value* r) {
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
// "column" (used by from_csv; absent for from_species_csv where the key names
// the column directly). One parser covers both call sites.
std::vector<TargetSpec> parse_targets(const resolve_value* targets) {
    std::vector<TargetSpec> out;
    if (!targets || targets->kind != RESOLVE_VALUE_MAP) return out;
    for (size_t i = 0; i < targets->keys.size(); ++i) {
        const std::string& name = targets->keys[i];
        const resolve_value* spec = targets->vals[i];
        TargetSpec ts;
        ts.target_name = name;
        ts.column_name = vhas(spec, "column") ? vstr(spec, "column") : name;
        if (vhas(spec, "task")) ts.task = parse_task_type(vstr(spec, "task"));
        if (vhas(spec, "transform")) ts.transform = parse_transform_type(vstr(spec, "transform"));
        if (vhas(spec, "num_classes")) ts.num_classes = (int)vint(spec, "num_classes");
        if (vhas(spec, "weight")) ts.weight = (float)vdbl(spec, "weight");
        // Optional explicit class -> integer-code mapping (a sub-map of
        // name->int), so a held-out set can be pinned to the same codes as the
        // training set instead of auto-fitting a possibly-different order.
        const resolve_value* cm = vget(spec, "class_mapping");
        if (cm && cm->kind == RESOLVE_VALUE_MAP) {
            for (size_t j = 0; j < cm->keys.size(); ++j) {
                ts.class_mapping[cm->keys[j]] = vint(cm->vals[j]);
            }
        }
        out.push_back(ts);
    }
    return out;
}

DatasetConfig parse_dataset_config(const resolve_value* c) {
    DatasetConfig config;
    for_each_field(config, ValueFieldReader{c});
    return config;
}

ModelConfig parse_model_config(const resolve_value* c) {
    ModelConfig config;
    for_each_field(config, ValueFieldReader{c});
    return config;
}

// Single source of truth for the value-tree schema keys, consumed by both
// parse_schema (read) and the dataset "schema" emitter (write) below. A field
// emitted under one spelling but read under another silently drops on the
// round-trip — the same field-drop mechanism behind issue #91 — so both sites
// reference these constants (issue #98).
namespace schema_tree_keys {
inline constexpr const char* kNPlots             = "n_plots";
inline constexpr const char* kNSpecies           = "n_species";
inline constexpr const char* kNSpeciesVocab      = "n_species_vocab";
inline constexpr const char* kHasCoordinates     = "has_coordinates";
inline constexpr const char* kHasAbundance       = "has_abundance";
inline constexpr const char* kHasTaxonomy        = "has_taxonomy";
inline constexpr const char* kNGenera            = "n_genera";
inline constexpr const char* kNFamilies          = "n_families";
inline constexpr const char* kNGeneraVocab       = "n_genera_vocab";
inline constexpr const char* kNFamiliesVocab     = "n_families_vocab";
inline constexpr const char* kCovariateNames     = "covariate_names";
inline constexpr const char* kTargets            = "targets";
inline constexpr const char* kTrackUnknownFrac   = "track_unknown_fraction";
inline constexpr const char* kTrackUnknownCount  = "track_unknown_count";
inline constexpr const char* kCategoricalNames   = "categorical_names";
inline constexpr const char* kCategoricalVocabSizes = "categorical_vocab_sizes";
inline constexpr const char* kCategoricalEmbedDim   = "categorical_embed_dim";
inline constexpr const char* kPoolWeighting      = "pool_weighting";
inline constexpr const char* kPoolSpeciesCap     = "pool_species_cap";
// Remaining DatasetConfig knobs + the fitted vocabularies (issue #102).
inline constexpr const char* kTopKSpecies        = "top_k_species";
inline constexpr const char* kSelection          = "selection";
inline constexpr const char* kRepresentation     = "representation";
inline constexpr const char* kNormalization      = "normalization";
inline constexpr const char* kAggregation        = "aggregation";
inline constexpr const char* kUseTaxonomy        = "use_taxonomy";
inline constexpr const char* kSpeciesVocab       = "species_vocab";
inline constexpr const char* kGenusVocab         = "genus_vocab";
inline constexpr const char* kFamilyVocab        = "family_vocab";
// Categorical string -> code maps. Not part of ResolveSchema (which carries
// only column names + sizes); carried alongside it in the "vocabs" tree the
// *_with_vocabs dataset loaders take.
inline constexpr const char* kCategoricalVocab   = "categorical_vocab";
// Per-target sub-map keys.
inline constexpr const char* kTargetTask         = "task";
inline constexpr const char* kTargetTransform    = "transform";
inline constexpr const char* kTargetNumClasses   = "num_classes";
inline constexpr const char* kTargetWeight       = "weight";
inline constexpr const char* kTargetClassWeights = "class_weights";
inline constexpr const char* kTargetClassNames   = "class_names";
}  // namespace schema_tree_keys

// Schema map -> ResolveSchema. Reads the full set the dataset's schema()
// accessor emits, INCLUDING categorical_names / categorical_vocab_sizes /
// categorical_embed_dim, so a categorical dataset's schema round-trips into a
// categorical-aware model (matches the nanobind path; harmless when absent).
ResolveSchema parse_schema(const resolve_value* s) {
    namespace k = schema_tree_keys;
    ResolveSchema schema;
    if (vhas(s, k::kNPlots)) schema.n_plots = vint(s, k::kNPlots);
    if (vhas(s, k::kNSpecies)) schema.n_species = vint(s, k::kNSpecies);
    if (vhas(s, k::kNSpeciesVocab)) schema.n_species_vocab = vint(s, k::kNSpeciesVocab);
    if (vhas(s, k::kHasCoordinates)) schema.has_coordinates = vbool(s, k::kHasCoordinates);
    if (vhas(s, k::kHasAbundance)) schema.has_abundance = vbool(s, k::kHasAbundance);
    if (vhas(s, k::kHasTaxonomy)) schema.has_taxonomy = vbool(s, k::kHasTaxonomy);
    if (vhas(s, k::kNGenera)) schema.n_genera = vint(s, k::kNGenera);
    if (vhas(s, k::kNFamilies)) schema.n_families = vint(s, k::kNFamilies);
    if (vhas(s, k::kNGeneraVocab)) schema.n_genera_vocab = vint(s, k::kNGeneraVocab);
    if (vhas(s, k::kNFamiliesVocab)) schema.n_families_vocab = vint(s, k::kNFamiliesVocab);
    if (vhas(s, k::kCovariateNames)) schema.covariate_names = vstr_vec(s, k::kCovariateNames);
    if (vhas(s, k::kTrackUnknownFrac)) schema.track_unknown_fraction = vbool(s, k::kTrackUnknownFrac);
    if (vhas(s, k::kTrackUnknownCount)) schema.track_unknown_count = vbool(s, k::kTrackUnknownCount);
    if (vhas(s, k::kCategoricalNames)) schema.categorical_names = vstr_vec(s, k::kCategoricalNames);
    if (vhas(s, k::kCategoricalVocabSizes)) schema.categorical_vocab_sizes = vint_vec(s, k::kCategoricalVocabSizes);
    if (vhas(s, k::kCategoricalEmbedDim)) schema.categorical_embed_dim = vint(s, k::kCategoricalEmbedDim);
    if (vhas(s, k::kPoolWeighting)) schema.pool_weighting = (int)vint(s, k::kPoolWeighting);
    if (vhas(s, k::kPoolSpeciesCap)) schema.pool_species_cap = (int)vint(s, k::kPoolSpeciesCap);
    // Remaining loader knobs + the fitted vocabularies (issue #102). The enum
    // fields reuse the DatasetConfig string spellings so one parser serves both
    // trees.
    if (vhas(s, k::kTopKSpecies)) schema.top_k_species = (int)vint(s, k::kTopKSpecies);
    if (vhas(s, k::kSelection)) schema.selection = parse_selection_mode(vstr(s, k::kSelection));
    if (vhas(s, k::kRepresentation)) schema.representation = parse_representation_mode(vstr(s, k::kRepresentation));
    if (vhas(s, k::kNormalization)) schema.normalization = parse_normalization_mode(vstr(s, k::kNormalization));
    if (vhas(s, k::kAggregation)) schema.aggregation = parse_aggregation_mode(vstr(s, k::kAggregation));
    if (vhas(s, k::kUseTaxonomy)) schema.use_taxonomy = vbool(s, k::kUseTaxonomy);
    if (vhas(s, k::kSpeciesVocab)) schema.species_vocab = vstr_vec(s, k::kSpeciesVocab);
    if (vhas(s, k::kGenusVocab)) schema.genus_vocab = vstr_vec(s, k::kGenusVocab);
    if (vhas(s, k::kFamilyVocab)) schema.family_vocab = vstr_vec(s, k::kFamilyVocab);

    const resolve_value* targets = vget(s, k::kTargets);
    if (targets && targets->kind == RESOLVE_VALUE_MAP) {
        for (size_t i = 0; i < targets->keys.size(); ++i) {
            const resolve_value* tc_v = targets->vals[i];
            TargetConfig tc;
            tc.name = targets->keys[i];
            if (vhas(tc_v, k::kTargetTask)) tc.task = parse_task_type(vstr(tc_v, k::kTargetTask));
            if (vhas(tc_v, k::kTargetTransform)) tc.transform = parse_transform_type(vstr(tc_v, k::kTargetTransform));
            if (vhas(tc_v, k::kTargetNumClasses)) tc.num_classes = (int)vint(tc_v, k::kTargetNumClasses);
            if (vhas(tc_v, k::kTargetWeight)) tc.weight = (float)vdbl(tc_v, k::kTargetWeight);
            if (vhas(tc_v, k::kTargetClassWeights)) tc.class_weights = vfloat_vec(tc_v, k::kTargetClassWeights);
            if (vhas(tc_v, k::kTargetClassNames)) tc.class_names = vstr_vec(tc_v, k::kTargetClassNames);
            schema.targets.push_back(tc);
        }
    }
    return schema;
}

TrainConfig parse_train_config(const resolve_value* c) {
    TrainConfig config;
    for_each_field(config, ValueFieldReader{c});
    return config;
}

// Inverse of nested_metrics_to_value: a map(target -> map(metric -> number)).
std::unordered_map<std::string, std::unordered_map<std::string, float>>
value_to_nested_metrics(const resolve_value* v) {
    std::unordered_map<std::string, std::unordered_map<std::string, float>> out;
    if (!v || v->kind != RESOLVE_VALUE_MAP) return out;
    const int64_t n_targets = resolve_map_size(v);
    for (int64_t i = 0; i < n_targets; ++i) {
        const char* tk = resolve_map_key_at(v, i);
        if (!tk) continue;
        const resolve_value* tm = vget(v, tk);
        if (!tm || tm->kind != RESOLVE_VALUE_MAP) continue;
        auto& inner = out[tk];
        const int64_t n_metrics = resolve_map_size(tm);
        for (int64_t j = 0; j < n_metrics; ++j) {
            const char* mk = resolve_map_key_at(tm, j);
            if (!mk) continue;
            inner[mk] = static_cast<float>(vdbl(vget(tm, mk)));
        }
    }
    return out;
}

RunMetadata parse_run_metadata(const resolve_value* c) {
    RunMetadata rm;
    // Inverse of run_metadata_to_value: read every key it writes, so a metadata
    // map handed in from R/save round-trips instead of dropping resolve_version
    // and final_metrics.
    if (vhas(c, "resolve_version")) rm.resolve_version = vstr(c, "resolve_version");
    if (vhas(c, "created_at")) rm.created_at = vstr(c, "created_at");
    if (vhas(c, "completed_at")) rm.completed_at = vstr(c, "completed_at");
    if (vhas(c, "train_time_seconds")) rm.train_time_seconds = (float)vdbl(c, "train_time_seconds");
    if (vhas(c, "n_plots_train")) rm.n_plots_train = vint(c, "n_plots_train");
    if (vhas(c, "n_plots_test")) rm.n_plots_test = vint(c, "n_plots_test");
    if (vhas(c, "best_epoch")) rm.best_epoch = (int)vint(c, "best_epoch");
    if (vhas(c, "total_epochs")) rm.total_epochs = (int)vint(c, "total_epochs");
    if (vhas(c, "final_metrics")) rm.final_metrics = value_to_nested_metrics(vget(c, "final_metrics"));
    return rm;
}

// ============================================================================
// Result-struct -> value-tree converters. One-for-one with r/src/rcpp_common.h.
// ============================================================================

// RAII guard: frees a partially-built value tree if a converter throws
// mid-build (e.g. a tensor->value conversion faults on a device error),
// instead of leaking it -- CAPI_BODY_PTR catches the exception and returns
// NULL, so the local root would otherwise have no owner. release() hands
// ownership back to the caller on success.
struct ValueGuard {
    resolve_value* p;
    explicit ValueGuard(resolve_value* v) noexcept : p(v) {}
    ~ValueGuard() { if (p) resolve_value_free(p); }
    ValueGuard(const ValueGuard&) = delete;
    ValueGuard& operator=(const ValueGuard&) = delete;
    resolve_value* release() noexcept { auto* q = p; p = nullptr; return q; }
};

resolve_value* baseline_metrics_to_value(const BaselineMetrics& bm) {
    auto* m = v_map();
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
    return m;
}
resolve_value* layer_diagnostics_to_value(const LayerDiagnostics& ld) {
    auto* m = v_map();
    v_put(m, "name", v_string(ld.name));
    v_put(m, "n_neurons", v_int(ld.n_neurons));
    v_put(m, "n_dead", v_int(ld.n_dead));
    v_put(m, "n_saturated", v_int(ld.n_saturated));
    v_put(m, "dead_fraction", v_double(ld.dead_fraction));
    v_put(m, "saturated_fraction", v_double(ld.saturated_fraction));
    v_put(m, "mean_activation", v_double(ld.mean_activation));
    v_put(m, "std_activation", v_double(ld.std_activation));
    v_put(m, "sparsity", v_double(ld.sparsity));
    return m;
}
resolve_value* network_diagnostics_to_value(const NetworkDiagnostics& nd) {
    auto* m = v_map();
    auto* layers = v_list();
    for (const auto& ld : nd.layers) v_append(layers, layer_diagnostics_to_value(ld));
    v_put(m, "layers", layers);
    v_put(m, "total_neurons", v_int(nd.total_neurons));
    v_put(m, "total_dead", v_int(nd.total_dead));
    v_put(m, "total_saturated", v_int(nd.total_saturated));
    v_put(m, "overall_dead_fraction", v_double(nd.overall_dead_fraction));
    v_put(m, "overall_saturated_fraction", v_double(nd.overall_saturated_fraction));
    v_put(m, "has_issues", v_bool(nd.has_issues));
    v_put(m, "summary", v_string(nd.summary));
    return m;
}
resolve_value* nested_metrics_to_value(
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& metrics) {
    auto* result = v_map();
    for (const auto& [target, metric_map] : metrics) {
        auto* inner = v_map();
        for (const auto& [metric, value] : metric_map) v_put(inner, metric, v_double(value));
        v_put(result, target, inner);
    }
    return result;
}
resolve_value* train_result_to_value(const TrainResult& tr) {
    auto* m = v_map();
    v_put(m, "best_epoch", v_int(tr.best_epoch));
    v_put(m, "final_metrics", nested_metrics_to_value(tr.final_metrics));
    v_put(m, "train_loss", v_double_array(tr.train_loss_history));
    v_put(m, "test_loss", v_double_array(tr.test_loss_history));
    v_put(m, "train_time_seconds", v_double(tr.train_time_seconds));
    v_put(m, "resumed_from_epoch", v_int(tr.resumed_from_epoch));
    v_put(m, "effective_batch_size", v_int(tr.effective_batch_size));
    auto* baselines = v_map();
    for (const auto& [target, bm] : tr.baselines) v_put(baselines, target, baseline_metrics_to_value(bm));
    v_put(m, "baselines", baselines);
    v_put(m, "diagnostics", network_diagnostics_to_value(tr.diagnostics));
    return m;
}
resolve_value* calibration_bin_to_value(const CalibrationBin& cb) {
    auto* m = v_map();
    v_put(m, "bin_start", v_double(cb.bin_start));
    v_put(m, "bin_end", v_double(cb.bin_end));
    v_put(m, "mean_predicted_prob", v_double(cb.mean_predicted_prob));
    v_put(m, "actual_frequency", v_double(cb.actual_frequency));
    v_put(m, "count", v_int(cb.count));
    return m;
}
resolve_value* calibration_result_to_value(const CalibrationResult& cr) {
    auto* m = v_map();
    v_put(m, "target_name", v_string(cr.target_name));
    v_put(m, "class_idx", v_int(cr.class_idx));
    auto* bins = v_list();
    for (const auto& b : cr.bins) v_append(bins, calibration_bin_to_value(b));
    v_put(m, "bins", bins);
    v_put(m, "expected_calibration_error", v_double(cr.expected_calibration_error));
    v_put(m, "max_calibration_error", v_double(cr.max_calibration_error));
    return m;
}
resolve_value* residual_analysis_to_value(const ResidualAnalysis& ra) {
    auto* m = v_map();
    v_put(m, "target_name", v_string(ra.target_name));
    v_put(m, "predictions", v_double_array(ra.predictions));
    v_put(m, "actuals", v_double_array(ra.actuals));
    v_put(m, "residuals", v_double_array(ra.residuals));
    v_put(m, "mean_residual", v_double(ra.mean_residual));
    v_put(m, "std_residual", v_double(ra.std_residual));
    v_put(m, "skewness", v_double(ra.skewness));
    v_put(m, "kurtosis", v_double(ra.kurtosis));
    v_put(m, "q05", v_double(ra.q05));
    v_put(m, "q25", v_double(ra.q25));
    v_put(m, "q50", v_double(ra.q50));
    v_put(m, "q75", v_double(ra.q75));
    v_put(m, "q95", v_double(ra.q95));
    return m;
}
resolve_value* classification_predictions_to_value(const ClassificationPredictions& cp) {
    auto* m = v_map();
    ValueGuard g(m);
    v_put(m, "target_name", v_string(cp.target_name));
    v_put(m, "class_names", v_string_array(cp.class_names));
    // Key name matches the nanobind field (ClassificationPredictions
    // .predicted_classes) so code reads the same across R and Python.
    v_put(m, "predicted_classes",
          (cp.predicted_classes.defined() && cp.predicted_classes.numel() > 0)
              ? tensor_to_ivec(cp.predicted_classes) : v_int_array({}));
    v_put(m, "actuals",
          (cp.actuals.defined() && cp.actuals.numel() > 0)
              ? tensor_to_ivec(cp.actuals) : v_int_array({}));
    if (cp.probabilities.defined() && cp.probabilities.numel() > 0) {
        v_put(m, "probabilities", tensor_to_mat(cp.probabilities));
    } else {
        auto* empty = v_new(RESOLVE_VALUE_DOUBLE_MATRIX);  // 0x0
        v_put(m, "probabilities", empty);
    }
    return g.release();
}
resolve_value* cross_validation_result_to_value(const CrossValidationResult& cvr) {
    auto* m = v_map();
    v_put(m, "n_folds", v_int(cvr.n_folds));
    v_put(m, "mean_metrics", nested_metrics_to_value(cvr.mean_metrics));
    v_put(m, "std_metrics", nested_metrics_to_value(cvr.std_metrics));
    auto* folds = v_list();
    for (const auto& fr : cvr.fold_results) v_append(folds, train_result_to_value(fr));
    v_put(m, "fold_results", folds);
    v_put(m, "total_time_seconds", v_double(cvr.total_time_seconds));
    return m;
}
resolve_value* scalers_to_value(const Scalers& s) {
    auto* m = v_map();
    ValueGuard g(m);
    if (s.continuous_mean.defined()) v_put(m, "continuous_mean", tensor_to_vec(s.continuous_mean));
    if (s.continuous_scale.defined()) v_put(m, "continuous_scale", tensor_to_vec(s.continuous_scale));
    // Per-target regression scaling { name -> {mean, scale} }. save_scalers /
    // load_scalers persist these in the checkpoint, but the accessor marshal
    // previously dropped them, so trainer$get("scalers") / predictor$get("scalers")
    // from R (and the nanobind Scalers view) could not see target scaling.
    if (!s.target_scalers.empty()) {
        auto* ts = v_map();
        // Deterministic key order (unordered_map iteration is not reproducible).
        std::vector<std::string> names;
        names.reserve(s.target_scalers.size());
        for (const auto& kv : s.target_scalers) names.push_back(kv.first);
        std::sort(names.begin(), names.end());
        for (const auto& name : names) {
            const auto& ms = s.target_scalers.at(name);
            auto* pair_m = v_map();
            v_put(pair_m, "mean", v_double(ms.first.defined() ? ms.first.item<double>() : 0.0));
            v_put(pair_m, "scale", v_double(ms.second.defined() ? ms.second.item<double>() : 0.0));
            v_put(ts, name, pair_m);
        }
        v_put(m, "target_scalers", ts);
    }
    return g.release();
}
resolve_value* categorical_vocab_to_value(const CategoricalVocab& vocab) {
    // One entry per column: a sub-map { name -> code } so the R side can rebuild
    // a named integer vector. (The Rcpp version returned a named IntegerVector;
    // the value tree carries the same key/value pairs as a string->int map.)
    auto* out = v_map();
    for (const auto& col : vocab.column_names()) {
        const auto& cmap = vocab.column_map(col);
        auto* inner = v_map();
        for (const auto& [k, code] : cmap) v_put(inner, k, v_int(code));
        v_put(out, col, inner);
    }
    return out;
}

// Inverse of categorical_vocab_to_value. Restores the per-column string -> code
// maps verbatim from the tree the *_with_vocabs dataset loaders receive
// (issue #102): the codes come from the checkpoint, so they are assigned, not
// re-derived.
CategoricalVocab value_to_categorical_vocab(const resolve_value* v) {
    CategoricalVocab vocab;
    if (!v || v->kind != RESOLVE_VALUE_MAP) return vocab;
    for (size_t i = 0; i < v->keys.size(); ++i) {
        const resolve_value* inner = v->vals[i];
        if (!inner || inner->kind != RESOLVE_VALUE_MAP) continue;
        std::unordered_map<std::string, int64_t> cmap;
        cmap.reserve(inner->keys.size());
        for (size_t j = 0; j < inner->keys.size(); ++j) {
            cmap[inner->keys[j]] = vint(inner->vals[j]);
        }
        vocab.set_column_map(v->keys[i], cmap);
    }
    return vocab;
}

// ResolveSchema -> MAP tree. Single source of truth for the emitted schema,
// shared by the dataset "schema" accessor and the predictor "schema"/"vocabs"
// accessors (issue #102) -- a field emitted by one but not the other is exactly
// how the vocab would go missing on one binding path.
resolve_value* schema_to_value(const ResolveSchema& s) {
    namespace k = schema_tree_keys;
    auto* targets_m = v_map();
    for (const auto& tc : s.targets) {
        std::string task_str = (tc.task == TaskType::Regression) ? "regression" : "classification";
        std::string transform_str = (tc.transform == TransformType::Log1p) ? "log1p" : "none";
        auto* tm = v_map();
        v_put(tm, k::kTargetTask, v_string(task_str));
        v_put(tm, k::kTargetTransform, v_string(transform_str));
        v_put(tm, k::kTargetNumClasses, v_int(tc.num_classes));
        v_put(tm, k::kTargetWeight, v_double(tc.weight));
        v_put(tm, k::kTargetClassWeights, v_double_array(tc.class_weights));
        // Per-class label vocabulary (class_names[code] == label), so
        // R callers can recover the code->label mapping like Python's
        // TargetConfig.class_names (issue #76).
        v_put(tm, k::kTargetClassNames, v_string_array(tc.class_names));
        v_put(targets_m, tc.name, tm);
    }
    auto* m = v_map();
    v_put(m, k::kNPlots, v_int(s.n_plots));
    v_put(m, k::kNSpecies, v_int(s.n_species));
    v_put(m, k::kNSpeciesVocab, v_int(s.n_species_vocab));
    v_put(m, k::kHasCoordinates, v_bool(s.has_coordinates));
    v_put(m, k::kHasAbundance, v_bool(s.has_abundance));
    v_put(m, k::kHasTaxonomy, v_bool(s.has_taxonomy));
    v_put(m, k::kNGenera, v_int(s.n_genera));
    v_put(m, k::kNFamilies, v_int(s.n_families));
    v_put(m, k::kNGeneraVocab, v_int(s.n_genera_vocab));
    v_put(m, k::kNFamiliesVocab, v_int(s.n_families_vocab));
    v_put(m, k::kCovariateNames, v_string_array(s.covariate_names));
    v_put(m, k::kTargets, targets_m);
    v_put(m, k::kTrackUnknownFrac, v_bool(s.track_unknown_fraction));
    v_put(m, k::kTrackUnknownCount, v_bool(s.track_unknown_count));
    v_put(m, k::kCategoricalNames, v_string_array(s.categorical_names));
    v_put(m, k::kCategoricalVocabSizes, v_int_array(s.categorical_vocab_sizes));
    v_put(m, k::kCategoricalEmbedDim, v_int(s.categorical_embed_dim));
    v_put(m, k::kPoolWeighting, v_int(s.pool_weighting));
    v_put(m, k::kPoolSpeciesCap, v_int(s.pool_species_cap));
    // Remaining loader knobs + the fitted vocabularies (issue #102).
    v_put(m, k::kTopKSpecies, v_int(s.top_k_species));
    v_put(m, k::kSelection, v_string(selection_mode_to_string(s.selection)));
    v_put(m, k::kRepresentation, v_string(representation_mode_to_string(s.representation)));
    v_put(m, k::kNormalization, v_string(normalization_mode_to_string(s.normalization)));
    v_put(m, k::kAggregation, v_string(aggregation_mode_to_string(s.aggregation)));
    v_put(m, k::kUseTaxonomy, v_bool(s.use_taxonomy));
    v_put(m, k::kSpeciesVocab, v_string_array(s.species_vocab));
    v_put(m, k::kGenusVocab, v_string_array(s.genus_vocab));
    v_put(m, k::kFamilyVocab, v_string_array(s.family_vocab));
    return m;
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
    auto* m = schema_to_value(s);
    v_put(m, schema_tree_keys::kCategoricalVocab, categorical_vocab_to_value(v.categorical));
    return m;
}

ExternalVocabs value_to_external_vocabs(const resolve_value* v) {
    ExternalVocabs out = external_vocabs_from_schema(parse_schema(v));
    if (vhas(v, schema_tree_keys::kCategoricalVocab)) {
        out.categorical = value_to_categorical_vocab(
            vget(v, schema_tree_keys::kCategoricalVocab));
    }
    return out;
}

// TrainConfig -> value map. Emits every field the registry carries, so a live
// trainer's get_config() and a checkpoint's load_train_config() expose the same
// names and parse_train_config reads either straight back (issue #87). Enums
// travel as their names; the device, the checkpoint destination and the AMP /
// cuDNN switches are part of a live config even though save_train_config leaves
// them out of the archive.
resolve_value* train_config_to_value(const TrainConfig& c) {
    auto* m = v_map();
    ValueGuard g(m);
    for_each_field(c, ValueFieldWriter{m});
    return g.release();
}

// DatasetConfig -> value map. One emitter for both callers -- the dataset's own
// `config` accessor and the predictor's `dataset_config` (the loading config a
// checkpoint implies) -- which were two hand-written blocks that had to agree on
// every key and every enum spelling.
resolve_value* dataset_config_to_value(const DatasetConfig& c) {
    auto* m = v_map();
    ValueGuard g(m);
    for_each_field(c, ValueFieldWriter{m});
    return g.release();
}

resolve_value* run_metadata_to_value(const RunMetadata& m0) {
    auto* m = v_map();
    v_put(m, "resolve_version", v_string(m0.resolve_version));
    v_put(m, "created_at", v_string(m0.created_at));
    v_put(m, "completed_at", v_string(m0.completed_at));
    v_put(m, "train_time_seconds", v_double(m0.train_time_seconds));
    v_put(m, "n_plots_train", v_int(m0.n_plots_train));
    v_put(m, "n_plots_test", v_int(m0.n_plots_test));
    v_put(m, "best_epoch", v_int(m0.best_epoch));
    v_put(m, "total_epochs", v_int(m0.total_epochs));
    v_put(m, "final_metrics", nested_metrics_to_value(m0.final_metrics));
    return m;
}

// target map -> value map (name -> double array). Used by dataset.targets(),
// model.forward(), trainer.predict(), predictor.predict().
resolve_value* target_map_to_value(const std::unordered_map<std::string, torch::Tensor>& m0) {
    auto* m = v_map();
    ValueGuard g(m);
    for (const auto& [name, tensor] : m0) v_put(m, name, tensor_to_vec(tensor));
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

int resolve_metric_band_accuracy(const double* pred, const double* target, int64_t n, double threshold, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::band_accuracy(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n),
            static_cast<float>(threshold));
        return 0;
    })
}
int resolve_metric_mae(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::mae(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n));
        return 0;
    })
}
int resolve_metric_rmse(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::rmse(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n));
        return 0;
    })
}
int resolve_metric_smape(const double* pred, const double* target, int64_t n, double eps, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::smape(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n),
            static_cast<float>(eps));
        return 0;
    })
}
int resolve_metric_accuracy(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::accuracy(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n));
        return 0;
    })
}
int resolve_metric_r_squared(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        *out = resolve::Metrics::r_squared(
            f32_tensor_from_doubles(pred, n), f32_tensor_from_doubles(target, n));
        return 0;
    })
}

}  // extern "C"

// ============================================================================
// Dataset
// ============================================================================

extern "C" {

resolve_dataset_t* resolve_dataset_from_csv(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config) {
    CAPI_BODY_PTR({
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* h = new resolve_dataset{ResolveDataset::from_csv(header_path, species_path, r, t, c)};
        return h;
    })
}

resolve_dataset_t* resolve_dataset_from_csv_with_schema(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_dataset_t* schema_source, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!schema_source) throw std::runtime_error("from_csv_with_schema: schema_source is null");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* h = new resolve_dataset{ResolveDataset::from_csv_with_schema(
            header_path, species_path, r, t, schema_source->ds, c)};
        return h;
    })
}

resolve_dataset_t* resolve_dataset_from_csv_with_vocabs(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        ExternalVocabs v = value_to_external_vocabs(vocabs);
        auto* h = new resolve_dataset{ResolveDataset::from_csv_with_vocabs(
            header_path, species_path, r, t, v, c)};
        return h;
    })
}

resolve_dataset_t* resolve_dataset_from_species_csv(
    const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config) {
    CAPI_BODY_PTR({
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* h = new resolve_dataset{ResolveDataset::from_species_csv(species_path, r, t, c)};
        return h;
    })
}

resolve_dataset_t* resolve_dataset_from_species_csv_with_vocabs(
    const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        ExternalVocabs v = value_to_external_vocabs(vocabs);
        auto* h = new resolve_dataset{ResolveDataset::from_species_csv_with_vocabs(
            species_path, r, t, v, c)};
        return h;
    })
}

resolve_dataset_t* resolve_dataset_from_dataframe(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ColumnTable h = value_to_column_table(header, "header");
        ColumnTable s = value_to_column_table(species, "species");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* d = new resolve_dataset{ResolveDataset::from_dataframe(h, s, r, t, c)};
        return d;
    })
}

resolve_dataset_t* resolve_dataset_from_dataframe_header(
    const resolve_value_t* header, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ColumnTable h = value_to_column_table(header, "header");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* d = new resolve_dataset{ResolveDataset::from_dataframe_header(
            h, species_path, r, t, c)};
        return d;
    })
}

resolve_dataset_t* resolve_dataset_from_species_dataframe(
    const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ColumnTable s = value_to_column_table(species, "species");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* d = new resolve_dataset{ResolveDataset::from_species_dataframe(s, r, t, c)};
        return d;
    })
}

resolve_dataset_t* resolve_dataset_from_dataframe_with_schema(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_dataset_t* schema_source, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!schema_source) throw std::runtime_error(
            "from_dataframe_with_schema: schema_source is null");
        ColumnTable h = value_to_column_table(header, "header");
        ColumnTable s = value_to_column_table(species, "species");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        auto* d = new resolve_dataset{ResolveDataset::from_dataframe_with_schema(
            h, s, r, t, schema_source->ds, c)};
        return d;
    })
}

resolve_dataset_t* resolve_dataset_from_dataframe_with_vocabs(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ColumnTable h = value_to_column_table(header, "header");
        ColumnTable s = value_to_column_table(species, "species");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        ExternalVocabs v = value_to_external_vocabs(vocabs);
        auto* d = new resolve_dataset{ResolveDataset::from_dataframe_with_vocabs(
            h, s, r, t, v, c)};
        return d;
    })
}

resolve_dataset_t* resolve_dataset_from_species_dataframe_with_vocabs(
    const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ColumnTable s = value_to_column_table(species, "species");
        RoleMapping r = parse_roles(roles);
        std::vector<TargetSpec> t = parse_targets(targets);
        DatasetConfig c = parse_dataset_config(config);
        ExternalVocabs v = value_to_external_vocabs(vocabs);
        auto* d = new resolve_dataset{ResolveDataset::from_species_dataframe_with_vocabs(
            s, r, t, v, c)};
        return d;
    })
}

void resolve_dataset_free(resolve_dataset_t* ds) { delete ds; }

resolve_value_t* resolve_dataset_get(const resolve_dataset_t* ds, const char* what) {
    CAPI_BODY_PTR({
        if (!ds) throw std::runtime_error("dataset_get: null handle");
        const ResolveDataset& d = ds->ds;
        std::string w = what ? what : "";

        auto mat_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return (t.defined() && t.numel() > 0) ? tensor_to_mat(t) : v_null();
        };
        auto imat_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return (t.defined() && t.numel() > 0) ? tensor_to_imat(t) : v_null();
        };
        auto vec_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return (t.defined() && t.numel() > 0) ? tensor_to_vec(t) : v_null();
        };
        // int64 flat accessor: keep exact values. Routing an int64 tensor through
        // vec_or_null (float32) silently rounds every value above 2^24, so the CSR
        // offsets and raw MurmurHash species IDs must use this at ASAAS scale.
        auto ivec_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return (t.defined() && t.numel() > 0) ? tensor_to_ivec(t) : v_null();
        };

        if (w == "coordinates") return mat_or_null(d.coordinates());
        if (w == "covariates") return mat_or_null(d.covariates());
        if (w == "hash_embedding") return mat_or_null(d.hash_embedding());
        if (w == "species_ids") return imat_or_null(d.species_ids());
        if (w == "species_vector") return mat_or_null(d.species_vector());
        if (w == "genus_ids") return imat_or_null(d.genus_ids());
        if (w == "family_ids") return imat_or_null(d.family_ids());
        if (w == "unknown_fraction") return vec_or_null(d.unknown_fraction());
        if (w == "unknown_count") return vec_or_null(d.unknown_count());
        if (w == "categorical_ids") return imat_or_null(d.categorical_ids());
        // Rank-pool / transformer encoder tensors (parity with the Python
        // ResolveDataset pool_* accessors). pool_mask is bool -> emit as int.
        if (w == "pool_genus_ids") return imat_or_null(d.pool_genus_ids());
        if (w == "pool_family_ids") return imat_or_null(d.pool_family_ids());
        if (w == "pool_weights") return mat_or_null(d.pool_weights());
        if (w == "pool_mask") return d.pool_mask().defined()
                ? imat_or_null(d.pool_mask().to(torch::kLong))
                : resolve_value_new_null();
        if (w == "pool_has_cover") return vec_or_null(d.pool_has_cover());
        if (w == "has_pool_data") return v_bool(d.has_pool_data());
        if (w == "categorical_vocab") return categorical_vocab_to_value(d.categorical_vocab());
        if (w == "targets") return target_map_to_value(d.targets());
        if (w == "plot_ids") return v_string_array(d.plot_ids());
        if (w == "species_vocab") return v_string_array(d.species_vocab());
        if (w == "n_plots") return v_int(d.n_plots());
        if (w == "has_raw_species_data") return v_bool(d.has_raw_species_data());
        if (w == "raw_species_ids") return ivec_or_null(d.raw_species_ids());
        if (w == "raw_weights") return vec_or_null(d.raw_weights());
        if (w == "plot_offsets") return ivec_or_null(d.plot_offsets());

        if (w == "taxonomy_vocab") {
            const auto& tv = d.taxonomy_vocab();
            auto* m = v_map();
            v_put(m, "n_genera", v_int(tv.n_genera()));
            v_put(m, "n_families", v_int(tv.n_families()));
            return m;
        }

        if (w == "config") return dataset_config_to_value(d.config());

        if (w == "schema") return schema_to_value(d.schema());
        // Every vocabulary this dataset fitted, in the carrier the
        // *_with_vocabs loaders take (issue #102).
        if (w == "vocabs") return external_vocabs_to_value(d.external_vocabs());

        throw std::runtime_error("dataset_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"

// ============================================================================
// Model
// ============================================================================

namespace {

// Extract the standard model-input tensors from an input map (forward family).
struct ModelInputs {
    torch::Tensor continuous, genus_ids, family_ids, species_ids, species_vector,
                  pool_genus_ids, pool_family_ids, pool_weights, pool_mask, pool_has_cover,
                  categorical_ids;
};
ModelInputs extract_model_inputs(const resolve_value* in) {
    ModelInputs x;
    x.continuous     = opt_f32(in, "continuous");
    x.genus_ids      = opt_i64(in, "genus_ids");
    x.family_ids     = opt_i64(in, "family_ids");
    x.species_ids    = opt_i64(in, "species_ids");
    x.species_vector = opt_f32(in, "species_vector");
    x.pool_genus_ids  = opt_i64(in, "pool_genus_ids");
    x.pool_family_ids = opt_i64(in, "pool_family_ids");
    x.pool_weights    = opt_f32(in, "pool_weights");
    torch::Tensor pm  = opt_i64(in, "pool_mask");
    x.pool_mask       = pm.defined() ? pm.to(torch::kBool) : pm;
    x.pool_has_cover  = opt_f32(in, "pool_has_cover");
    x.categorical_ids = opt_i64(in, "categorical_ids");
    return x;
}

}  // namespace

extern "C" {

resolve_model_t* resolve_model_create(const resolve_value_t* schema, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        ResolveSchema s = parse_schema(schema);
        ModelConfig c = parse_model_config(config);
        auto* h = new resolve_model{ResolveModel(s, c)};
        return h;
    })
}

void resolve_model_free(resolve_model_t* m) { delete m; }

resolve_value_t* resolve_model_call(resolve_model_t* m, const char* method, const resolve_value_t* inputs) {
    CAPI_BODY_PTR({
        if (!m) throw std::runtime_error("model_call: null handle");
        std::string mt = method ? method : "";
        ModelInputs in = extract_model_inputs(inputs);

        if (mt == "forward") {
            auto outputs = m->model->forward(
                in.continuous, in.genus_ids, in.family_ids, in.species_ids, in.species_vector,
                in.pool_genus_ids, in.pool_family_ids, in.pool_weights, in.pool_mask,
                in.pool_has_cover, in.categorical_ids);
            return target_map_to_value(outputs);
        }
        if (mt == "get_latent") {
            torch::Tensor latent = m->model->get_latent(
                in.continuous, in.genus_ids, in.family_ids, in.species_ids, in.species_vector,
                in.pool_genus_ids, in.pool_family_ids, in.pool_weights, in.pool_mask,
                in.pool_has_cover, in.categorical_ids);
            // latent is [batch, latent_dim]; return a matrix so R gets per-plot
            // rows (matching Python's 2-D get_latent and encode_with_activations
            // below), not a flat length batch*latent_dim vector with no dims.
            return tensor_to_mat(latent);
        }
        if (mt == "forward_with_aux") {
            auto result = m->model->forward_with_aux(
                in.continuous, in.genus_ids, in.family_ids, in.species_ids, in.species_vector,
                in.pool_genus_ids, in.pool_family_ids, in.pool_weights, in.pool_mask,
                in.pool_has_cover, in.categorical_ids);
            auto* ret = v_map();
            ValueGuard g(ret);  // frees the partial tree if a conversion throws
            v_put(ret, "outputs", target_map_to_value(result.outputs));
            if (result.moe_aux_loss.defined()) v_put(ret, "moe_aux_loss", tensor_to_vec(result.moe_aux_loss));
            return g.release();
        }
        if (mt == "forward_single") {
            std::string target = vstr(inputs, "target");
            auto result = m->model->forward_single(
                target, in.continuous, in.genus_ids, in.family_ids, in.species_ids,
                in.species_vector, in.categorical_ids);
            return tensor_to_vec(result);
        }
        if (mt == "encode_with_activations") {
            auto [latent, activations] = m->model->encode_with_activations(
                in.continuous, in.genus_ids, in.family_ids, in.categorical_ids);
            auto* ret = v_map();
            ValueGuard g(ret);  // frees the partial tree if a conversion throws
            v_put(ret, "latent", tensor_to_mat(latent));
            auto* acts = v_list();
            v_put(ret, "activations", acts);  // attach before filling so acts is owned
            for (const auto& a : activations) v_append(acts, tensor_to_mat(a));
            return g.release();
        }
        if (mt == "get_gate_probs") {
            auto result = m->model->get_gate_probs(in.continuous, in.genus_ids, in.family_ids);
            return tensor_to_mat(result);
        }
        throw std::runtime_error("model_call: unknown method '" + mt + "'");
    })
}

resolve_value_t* resolve_model_get(const resolve_model_t* m, const char* what) {
    CAPI_BODY_PTR({
        if (!m) throw std::runtime_error("model_get: null handle");
        std::string w = what ? what : "";
        if (w == "latent_dim") return v_int(m->model->latent_dim());
        if (w == "uses_explicit_vector") return v_bool(m->model->uses_explicit_vector());
        if (w == "uses_moe") return v_bool(m->model->uses_moe());
        if (w == "n_experts") return v_int(m->model->n_experts());
        if (w == "species_encoding") {
            return v_string(species_encoding_to_string(m->model->species_encoding()));
        }
        auto weights_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return t.defined() ? tensor_to_mat(t) : v_null();
        };
        if (w == "genus_weights") return weights_or_null(m->model->get_genus_weights());
        if (w == "family_weights") return weights_or_null(m->model->get_family_weights());
        if (w == "species_weights") return weights_or_null(m->model->get_species_weights());
        throw std::runtime_error("model_get: unknown accessor '" + w + "'");
    })
}

int resolve_model_set_train(resolve_model_t* m, int mode) {
    CAPI_BODY_INT({
        if (!m) throw std::runtime_error("model_set_train: null handle");
        m->model->train(mode != 0);
        return 0;
    })
}
int resolve_model_to_device(resolve_model_t* m, const char* device) {
    CAPI_BODY_INT({
        if (!m) throw std::runtime_error("model_to_device: null handle");
        std::string dev = device ? device : "cpu";
        m->model->to(dev == "cuda" ? torch::kCUDA : torch::kCPU);
        return 0;
    })
}
int resolve_model_set_traits(resolve_model_t* m, const resolve_value_t* traits) {
    CAPI_BODY_INT({
        if (!m) throw std::runtime_error("model_set_traits: null handle");
        m->model->set_traits(value_to_f32(traits));
        return 0;
    })
}

}  // extern "C"

// ============================================================================
// Trainer
// ============================================================================

extern "C" {

resolve_trainer_t* resolve_trainer_create(resolve_model_t* model, const resolve_value_t* config) {
    CAPI_BODY_PTR({
        if (!model) throw std::runtime_error("trainer_create: null model handle");
        TrainConfig c = parse_train_config(config);
        auto* h = new resolve_trainer{std::make_unique<resolve::Trainer>(model->model, c)};
        return h;
    })
}

void resolve_trainer_free(resolve_trainer_t* t) { delete t; }

int resolve_trainer_prepare_data(resolve_trainer_t* t, const resolve_value_t* in, double test_size, int seed) {
    CAPI_BODY_INT({
        if (!t) throw std::runtime_error("prepare_data: null handle");
        ModelInputs x = extract_model_inputs(in);
        torch::Tensor coordinates = opt_f32(in, "coordinates");
        torch::Tensor covariates = opt_f32(in, "covariates");
        torch::Tensor hash_embedding = opt_f32(in, "hash_embedding");
        torch::Tensor unknown_fraction = opt_f32(in, "unknown_fraction");
        torch::Tensor unknown_count = opt_f32(in, "unknown_count");

        std::unordered_map<std::string, torch::Tensor> target_map;
        const resolve_value* targets = vget(in, "targets");
        if (targets && targets->kind == RESOLVE_VALUE_MAP) {
            for (size_t i = 0; i < targets->keys.size(); ++i) {
                target_map[targets->keys[i]] = value_to_f32(targets->vals[i]);
            }
        }

        t->trainer->prepare_data(
            coordinates, covariates, hash_embedding, x.species_ids, x.species_vector,
            x.genus_ids, x.family_ids, unknown_fraction, unknown_count, target_map,
            x.pool_genus_ids, x.pool_family_ids, x.pool_weights, x.pool_mask, x.pool_has_cover,
            x.categorical_ids, (float)test_size, seed);
        return 0;
    })
}

int resolve_trainer_prepare_data_from_dataset(resolve_trainer_t* t, const resolve_dataset_t* ds, double test_size, int seed) {
    CAPI_BODY_INT({
        if (!t || !ds) throw std::runtime_error("prepare_data_from_dataset: null handle");
        t->trainer->prepare_data(ds->ds, (float)test_size, seed);
        return 0;
    })
}

resolve_value_t* resolve_trainer_fit(resolve_trainer_t* t) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("fit: null handle");
        return train_result_to_value(t->trainer->fit());
    })
}

int resolve_trainer_save(resolve_trainer_t* t, const char* path, const resolve_value_t* metadata) {
    CAPI_BODY_INT({
        if (!t) throw std::runtime_error("save: null handle");
        if (metadata && metadata->kind == RESOLVE_VALUE_MAP) {
            RunMetadata rm = parse_run_metadata(metadata);
            t->trainer->save(path, &rm);
        } else {
            t->trainer->save(path);
        }
        return 0;
    })
}

int resolve_trainer_load_state(resolve_trainer_t* t, const char* path, const char* device, double vram_fraction) {
    CAPI_BODY_INT({
        if (!t) throw std::runtime_error("load_state: null handle");
        std::string dev = device ? device : "cpu";
        t->trainer->load_state(path, dev == "cuda" ? torch::kCUDA : torch::kCPU, (float)vram_fraction);
        return 0;
    })
}

resolve_value_t* resolve_trainer_get(const resolve_trainer_t* t, const char* what) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("trainer_get: null handle");
        std::string w = what ? what : "";
        resolve::Trainer& tr = *t->trainer;
        if (w == "scalers") return scalers_to_value(tr.scalers());
        if (w == "categorical_vocab") return categorical_vocab_to_value(tr.categorical_vocab());
        if (w == "test_indices") {
            auto x = tr.test_indices();
            return (x.defined() && x.numel() > 0) ? tensor_to_ivec(x) : v_int_array({});
        }
        if (w == "train_indices") {
            auto x = tr.train_indices();
            return (x.defined() && x.numel() > 0) ? tensor_to_ivec(x) : v_int_array({});
        }
        if (w == "test_plot_ids") return v_string_array(tr.test_plot_ids());
        if (w == "train_plot_ids") return v_string_array(tr.train_plot_ids());
        if (w == "effective_batch_size") return v_int(tr.effective_batch_size());
        if (w == "config") {
            // Full TrainConfig serializer (parity with the Python Trainer.config
            // surface and resolve.load_train_config). Device / checkpoint fields
            // are now emitted by train_config_to_value itself.
            return train_config_to_value(tr.config());
        }
        throw std::runtime_error("trainer_get: unknown accessor '" + w + "'");
    })
}

resolve_value_t* resolve_trainer_compute(resolve_trainer_t* t, const char* kind, const resolve_value_t* args) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("compute: null handle");
        std::string k = kind ? kind : "";
        resolve::Trainer& tr = *t->trainer;
        if (k == "diagnostics") return network_diagnostics_to_value(tr.compute_diagnostics());
        std::string target = vstr(args, "target_name");
        if (k == "calibration") {
            int n_bins = vhas(args, "n_bins") ? (int)vint(args, "n_bins") : 10;
            return calibration_result_to_value(tr.compute_calibration(target, n_bins));
        }
        if (k == "residuals") return residual_analysis_to_value(tr.compute_residuals(target));
        if (k == "classification_predictions")
            return classification_predictions_to_value(tr.compute_classification_predictions(target));
        throw std::runtime_error("compute: unknown kind '" + k + "'");
    })
}

resolve_value_t* resolve_trainer_cross_validate(resolve_trainer_t* t, int n_folds, int seed) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("cross_validate: null handle");
        return cross_validation_result_to_value(t->trainer->cross_validate(n_folds, seed));
    })
}

resolve_value_t* resolve_trainer_cross_validate_spatial(resolve_trainer_t* t, const resolve_value_t* spatial_cfg, int n_folds, int seed) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("cross_validate_spatial: null handle");
        SpatialBlockConfig cfg = parse_spatial_block_config(spatial_cfg);
        return cross_validation_result_to_value(t->trainer->cross_validate_spatial(cfg, n_folds, seed));
    })
}

resolve_value_t* resolve_trainer_predict(resolve_trainer_t* t, const resolve_value_t* in) {
    CAPI_BODY_PTR({
        if (!t) throw std::runtime_error("predict: null handle");
        ModelInputs x = extract_model_inputs(in);
        auto result = t->trainer->predict(
            x.continuous, x.genus_ids, x.family_ids, x.species_ids, x.species_vector,
            x.pool_genus_ids, x.pool_family_ids, x.pool_weights, x.pool_mask, x.pool_has_cover,
            x.categorical_ids);
        return target_map_to_value(result);
    })
}

resolve_value_t* resolve_load_train_config(const char* path) {
    CAPI_BODY_PTR({ return train_config_to_value(resolve::Trainer::load_train_config(path)); })
}
resolve_value_t* resolve_load_run_metadata(const char* path) {
    CAPI_BODY_PTR({ return run_metadata_to_value(resolve::Trainer::load_run_metadata(path)); })
}

}  // extern "C"

// ============================================================================
// Predictor
// ============================================================================

namespace {

// Assemble the standard predictions result (predictions / targets / plot_ids /
// optional latent) shared by both predict paths.
resolve_value* predictions_to_value(const ResolvePredictions& preds, bool return_latent) {
    auto* result = v_map();
    ValueGuard g(result);  // frees the partial tree if a conversion throws
    v_put(result, "predictions", target_map_to_value(preds.predictions));
    v_put(result, "targets", target_map_to_value(preds.targets));
    if (return_latent && preds.latent.defined()) v_put(result, "latent", tensor_to_mat(preds.latent));
    v_put(result, "plot_ids", v_string_array(preds.plot_ids));
    return g.release();
}

}  // namespace

extern "C" {

resolve_predictor_t* resolve_predictor_load(const char* path, const char* device, double vram_fraction) {
    CAPI_BODY_PTR({
        std::string dev = device ? device : "cpu";
        auto* h = new resolve_predictor{
            resolve::Predictor::load(path, dev == "cuda" ? torch::kCUDA : torch::kCPU, (float)vram_fraction)};
        return h;
    })
}

void resolve_predictor_free(resolve_predictor_t* p) { delete p; }

resolve_value_t* resolve_predictor_predict(resolve_predictor_t* p, const resolve_value_t* in, int return_latent) {
    CAPI_BODY_PTR({
        if (!p) throw std::runtime_error("predict: null handle");
        ModelInputs x = extract_model_inputs(in);
        torch::Tensor coordinates = opt_f32(in, "coordinates");
        torch::Tensor covariates = opt_f32(in, "covariates");
        torch::Tensor hash_embedding = opt_f32(in, "hash_embedding");
        torch::Tensor unknown_fraction = opt_f32(in, "unknown_fraction");
        torch::Tensor unknown_count = opt_f32(in, "unknown_count");

        auto preds = p->predictor.predict(
            coordinates, covariates, hash_embedding, x.species_ids, x.species_vector,
            x.genus_ids, x.family_ids, unknown_fraction, unknown_count,
            x.pool_genus_ids, x.pool_family_ids, x.pool_weights, x.pool_mask, x.pool_has_cover,
            x.categorical_ids, return_latent != 0);
        return predictions_to_value(preds, return_latent != 0);
    })
}

resolve_value_t* resolve_predictor_predict_dataset(resolve_predictor_t* p, const resolve_dataset_t* ds, int return_latent, int64_t batch_size) {
    CAPI_BODY_PTR({
        if (!p || !ds) throw std::runtime_error("predict_dataset: null handle");
        ResolvePredictions preds = p->predictor.predict(ds->ds, return_latent != 0, batch_size);
        return predictions_to_value(preds, return_latent != 0);
    })
}

resolve_value_t* resolve_predictor_get_embeddings(resolve_predictor_t* p, const resolve_value_t* in) {
    CAPI_BODY_PTR({
        if (!p) throw std::runtime_error("get_embeddings: null handle");
        torch::Tensor coordinates = opt_f32(in, "coordinates");
        torch::Tensor covariates = opt_f32(in, "covariates");
        torch::Tensor hash_embedding = opt_f32(in, "hash_embedding");
        torch::Tensor genus_ids = opt_i64(in, "genus_ids");
        torch::Tensor family_ids = opt_i64(in, "family_ids");
        torch::Tensor emb = p->predictor.get_embeddings(coordinates, covariates, hash_embedding, genus_ids, family_ids);
        return tensor_to_mat(emb);
    })
}

int resolve_predictor_optimize_for_inference(resolve_predictor_t* p) {
    CAPI_BODY_INT({
        if (!p) throw std::runtime_error("optimize_for_inference: null handle");
        p->predictor.optimize_for_inference();
        return 0;
    })
}

resolve_value_t* resolve_predictor_get(const resolve_predictor_t* p, const char* what) {
    CAPI_BODY_PTR({
        if (!p) throw std::runtime_error("predictor_get: null handle");
        std::string w = what ? what : "";
        const resolve::Predictor& pr = p->predictor;
        if (w == "device") return v_string(pr.device().is_cuda() ? "cuda" : "cpu");
        if (w == "scalers") return scalers_to_value(pr.scalers());
        if (w == "categorical_vocab") return categorical_vocab_to_value(pr.categorical_vocab());
        // Issue #102: the checkpoint's schema (loader knobs + fitted species /
        // taxonomy vocabularies) and the complete vocabulary carrier the
        // *_with_vocabs dataset loaders take. `dataset_config` is the
        // DatasetConfig the checkpoint implies, ready to pass straight to a
        // loader.
        if (w == "schema") return schema_to_value(pr.schema());
        if (w == "vocabs") return external_vocabs_to_value(pr.external_vocabs());
        if (w == "species_vocab") return v_string_array(pr.species_vocab());
        if (w == "genus_vocab") return v_string_array(pr.genus_vocab());
        if (w == "family_vocab") return v_string_array(pr.family_vocab());
        if (w == "dataset_config") {
            return dataset_config_to_value(dataset_config_from_checkpoint(
                pr.schema(), pr.model()->config()));
        }
        // Guard undefined tensors (an encoder without a given table returns an
        // empty tensor) so R gets NULL rather than a throw from tensor_to_mat's
        // .cpu(), matching the model weight accessors and Python (which returns
        // None).
        auto mat_or_null = [](const torch::Tensor& t) -> resolve_value* {
            return (t.defined() && t.numel() > 0) ? tensor_to_mat(t) : v_null();
        };
        if (w == "genus_embeddings") return mat_or_null(pr.get_genus_embeddings());
        if (w == "family_embeddings") return mat_or_null(pr.get_family_embeddings());
        if (w == "species_embeddings") return mat_or_null(pr.get_species_embeddings());
        throw std::runtime_error("predictor_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"
