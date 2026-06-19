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

#include <torch/torch.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
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
void set_error(const std::string& msg) { g_last_error = msg; }

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

resolve_value* v_new(resolve_value_kind_t k) {
    auto* v = new resolve_value();
    v->kind = k;
    return v;
}
resolve_value* v_null()            { return v_new(RESOLVE_VALUE_NULL); }
resolve_value* v_bool(bool x)      { auto* v = v_new(RESOLVE_VALUE_BOOL);   v->b = x; return v; }
resolve_value* v_int(int64_t x)    { auto* v = v_new(RESOLVE_VALUE_INT);    v->i = x; return v; }
resolve_value* v_double(double x)  { auto* v = v_new(RESOLVE_VALUE_DOUBLE); v->d = x; return v; }
resolve_value* v_string(const std::string& x) { auto* v = v_new(RESOLVE_VALUE_STRING); v->s = x; return v; }

resolve_value* v_int_array(const std::vector<int64_t>& x) {
    auto* v = v_new(RESOLVE_VALUE_INT_ARRAY); v->iarr = x; return v;
}
resolve_value* v_double_array(const std::vector<double>& x) {
    auto* v = v_new(RESOLVE_VALUE_DOUBLE_ARRAY); v->darr = x; return v;
}
resolve_value* v_double_array(const std::vector<float>& x) {
    auto* v = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    v->darr.assign(x.begin(), x.end());
    return v;
}
resolve_value* v_string_array(const std::vector<std::string>& x) {
    auto* v = v_new(RESOLVE_VALUE_STRING_ARRAY); v->sarr = x; return v;
}
resolve_value* v_map()  { return v_new(RESOLVE_VALUE_MAP); }
resolve_value* v_list() { return v_new(RESOLVE_VALUE_LIST); }

// Put a child into a map (takes ownership). Used by C++ result assembly.
void v_put(resolve_value* m, const std::string& key, resolve_value* child) {
    m->keys.push_back(key);
    m->vals.push_back(child);
}
void v_append(resolve_value* l, resolve_value* child) { l->items.push_back(child); }

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
    auto* node = v_new(RESOLVE_VALUE_INT_ARRAY);
    node->iarr.assign(v, v + (n > 0 ? n : 0));
    return node;
}
resolve_value_t* resolve_value_new_double_array(const double* v, int64_t n) {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    node->darr.assign(v, v + (n > 0 ? n : 0));
    return node;
}
resolve_value_t* resolve_value_new_string_array(const char* const* v, int64_t n) {
    auto* node = v_new(RESOLVE_VALUE_STRING_ARRAY);
    node->sarr.reserve(n > 0 ? n : 0);
    for (int64_t j = 0; j < n; ++j) node->sarr.emplace_back(v[j] ? v[j] : "");
    return node;
}
resolve_value_t* resolve_value_new_double_matrix(const double* v, int64_t nrow, int64_t ncol) {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->darr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    return node;
}
resolve_value_t* resolve_value_new_int_matrix(const int64_t* v, int64_t nrow, int64_t ncol) {
    auto* node = v_new(RESOLVE_VALUE_INT_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->iarr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    return node;
}

void resolve_map_set_null  (resolve_value_t* m, const char* k) { v_put(m, k, v_null()); }
void resolve_map_set_bool  (resolve_value_t* m, const char* k, int v) { v_put(m, k, v_bool(v != 0)); }
void resolve_map_set_int   (resolve_value_t* m, const char* k, int64_t v) { v_put(m, k, v_int(v)); }
void resolve_map_set_double(resolve_value_t* m, const char* k, double v) { v_put(m, k, v_double(v)); }
void resolve_map_set_string(resolve_value_t* m, const char* k, const char* v) {
    v_put(m, k, v_string(v ? v : ""));
}
void resolve_map_set_int_array(resolve_value_t* m, const char* k, const int64_t* v, int64_t n) {
    auto* node = v_new(RESOLVE_VALUE_INT_ARRAY);
    node->iarr.assign(v, v + (n > 0 ? n : 0));
    v_put(m, k, node);
}
void resolve_map_set_double_array(resolve_value_t* m, const char* k, const double* v, int64_t n) {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_ARRAY);
    node->darr.assign(v, v + (n > 0 ? n : 0));
    v_put(m, k, node);
}
void resolve_map_set_string_array(resolve_value_t* m, const char* k, const char* const* v, int64_t n) {
    auto* node = v_new(RESOLVE_VALUE_STRING_ARRAY);
    node->sarr.reserve(n > 0 ? n : 0);
    for (int64_t j = 0; j < n; ++j) node->sarr.emplace_back(v[j] ? v[j] : "");
    v_put(m, k, node);
}
void resolve_map_set_double_matrix(resolve_value_t* m, const char* k, const double* v, int64_t nrow, int64_t ncol) {
    auto* node = v_new(RESOLVE_VALUE_DOUBLE_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->darr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    v_put(m, k, node);
}
void resolve_map_set_int_matrix(resolve_value_t* m, const char* k, const int64_t* v, int64_t nrow, int64_t ncol) {
    auto* node = v_new(RESOLVE_VALUE_INT_MATRIX);
    node->nrow = nrow; node->ncol = ncol;
    node->iarr.assign(v, v + (nrow * ncol > 0 ? nrow * ncol : 0));
    v_put(m, k, node);
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
std::string vstr(const resolve_value* m, const char* k) {
    const resolve_value* v = vget(m, k);
    return v ? v->s : std::string();
}
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

std::vector<int64_t> vint_vec(const resolve_value* v) {
    std::vector<int64_t> out;
    if (!v) return out;
    if (v->kind == RESOLVE_VALUE_INT_ARRAY) return v->iarr;
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        out.reserve(v->darr.size());
        for (double x : v->darr) out.push_back(static_cast<int64_t>(x));
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
        out.assign(v->darr.begin(), v->darr.end());
    } else if (v->kind == RESOLVE_VALUE_INT_ARRAY) {
        out.assign(v->iarr.begin(), v->iarr.end());
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
        std::vector<float> data(v->darr.begin(), v->darr.end());
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        std::vector<float> data(v->darr.begin(), v->darr.end());
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_INT_MATRIX) {
        std::vector<float> data(v->iarr.begin(), v->iarr.end());
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_INT_ARRAY) {
        std::vector<float> data(v->iarr.begin(), v->iarr.end());
        return torch::from_blob(data.data(),
            {static_cast<int64_t>(data.size())}, opts).clone();
    }
    throw std::runtime_error("value_to_f32: unsupported value kind");
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
        std::vector<int64_t> data(v->darr.begin(), v->darr.end());
        return torch::from_blob(data.data(), {v->nrow, v->ncol}, opts).clone();
    }
    if (v->kind == RESOLVE_VALUE_DOUBLE_ARRAY) {
        std::vector<int64_t> data(v->darr.begin(), v->darr.end());
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
// Enum parsers (string -> engine enum). Moved from r/src/rcpp_common.h.
// ============================================================================

namespace {

template <typename EnumT>
EnumT parse_enum(const std::string& s,
                 std::initializer_list<std::pair<const char*, EnumT>> entries,
                 const char* type_name) {
    for (const auto& [key, val] : entries) {
        if (s == key) return val;
    }
    throw std::runtime_error(std::string("Invalid ") + type_name + ": " + s);
}

SelectionMode parse_selection_mode(const std::string& s) {
    return parse_enum<SelectionMode>(s, {
        {"top", SelectionMode::Top}, {"bottom", SelectionMode::Bottom},
        {"top_bottom", SelectionMode::TopBottom}, {"all", SelectionMode::All},
    }, "selection mode");
}
RepresentationMode parse_representation_mode(const std::string& s) {
    return parse_enum<RepresentationMode>(s, {
        {"abundance", RepresentationMode::Abundance},
        {"presence_absence", RepresentationMode::PresenceAbsence},
    }, "representation mode");
}
NormalizationMode parse_normalization_mode(const std::string& s) {
    return parse_enum<NormalizationMode>(s, {
        {"raw", NormalizationMode::Raw}, {"norm", NormalizationMode::Norm},
        {"log1p", NormalizationMode::Log1p},
    }, "normalization mode");
}
PoolWeighting parse_pool_weighting(const std::string& s) {
    return parse_enum<PoolWeighting>(s, {
        {"binary", PoolWeighting::Binary}, {"abundance", PoolWeighting::Abundance},
        {"log1p", PoolWeighting::Log1p}, {"norm", PoolWeighting::Norm},
        {"rank", PoolWeighting::Rank},
    }, "pool weighting");
}
TaskType parse_task_type(const std::string& s) {
    return parse_enum<TaskType>(s, {
        {"regression", TaskType::Regression},
        {"classification", TaskType::Classification},
    }, "task type");
}
TransformType parse_transform_type(const std::string& s) {
    return parse_enum<TransformType>(s, {
        {"none", TransformType::None}, {"log1p", TransformType::Log1p},
    }, "transform type");
}
SpeciesEncodingMode parse_species_encoding_mode(const std::string& s) {
    return parse_enum<SpeciesEncodingMode>(s, {
        {"hash", SpeciesEncodingMode::Hash}, {"embed", SpeciesEncodingMode::Embed},
        {"sparse", SpeciesEncodingMode::Sparse},
        {"rank_pool", SpeciesEncodingMode::RankPool},
        {"transformer", SpeciesEncodingMode::Transformer},
    }, "species encoding mode");
}
LossConfigMode parse_loss_config_mode(const std::string& s) {
    return parse_enum<LossConfigMode>(s, {
        {"mae", LossConfigMode::MAE}, {"smape", LossConfigMode::SMAPE},
        {"combined", LossConfigMode::Combined}, {"nca", LossConfigMode::NCA},
    }, "loss config mode");
}
LRSchedulerType parse_lr_scheduler_type(const std::string& s) {
    return parse_enum<LRSchedulerType>(s, {
        {"none", LRSchedulerType::None}, {"step", LRSchedulerType::StepLR},
        {"cosine", LRSchedulerType::CosineAnnealing},
    }, "LR scheduler type");
}
MoERoutingType parse_moe_routing_type(const std::string& s) {
    return parse_enum<MoERoutingType>(s, {
        {"none", MoERoutingType::None}, {"soft", MoERoutingType::Soft},
        {"topk", MoERoutingType::TopK},
    }, "MoE routing type");
}
ActivationType parse_activation_type(const std::string& s) {
    return parse_enum<ActivationType>(s, {
        {"relu", ActivationType::ReLU}, {"leaky_relu", ActivationType::LeakyReLU},
        {"gelu", ActivationType::GELU}, {"silu", ActivationType::SiLU},
        {"tanh", ActivationType::Tanh}, {"mish", ActivationType::Mish},
        {"elu", ActivationType::ELU}, {"selu", ActivationType::SELU},
        {"softplus", ActivationType::Softplus}, {"prelu", ActivationType::PReLU},
    }, "activation type");
}
NormLayerType parse_norm_layer_type(const std::string& s) {
    return parse_enum<NormLayerType>(s, {
        {"batch_norm", NormLayerType::BatchNorm}, {"layer_norm", NormLayerType::LayerNorm},
        {"group_norm", NormLayerType::GroupNorm}, {"rms_norm", NormLayerType::RMSNorm},
        {"none", NormLayerType::None},
    }, "normalization layer type");
}
EncoderArchitecture parse_encoder_architecture(const std::string& s) {
    return parse_enum<EncoderArchitecture>(s, {
        {"mlp", EncoderArchitecture::MLP}, {"ft_transformer", EncoderArchitecture::FTTransformer},
        {"tabnet", EncoderArchitecture::TabNet}, {"saint", EncoderArchitecture::SAINT},
        {"trait_net", EncoderArchitecture::TraitNet}, {"gnn", EncoderArchitecture::GNN},
        {"excelformer", EncoderArchitecture::ExcelFormer},
        {"heterogeneous_gnn", EncoderArchitecture::HeterogeneousGNN},
    }, "encoder architecture");
}
GNNType parse_gnn_type(const std::string& s) {
    return parse_enum<GNNType>(s, {
        {"gcn", GNNType::GCN}, {"gat", GNNType::GAT}, {"graphsage", GNNType::GraphSAGE},
    }, "GNN type");
}
GraphConstructionMode parse_graph_construction_mode(const std::string& s) {
    return parse_enum<GraphConstructionMode>(s, {
        {"spatial", GraphConstructionMode::Spatial},
        {"taxonomic", GraphConstructionMode::Taxonomic},
        {"cooccurrence", GraphConstructionMode::CoOccurrence},
    }, "graph construction mode");
}
TraitInteractionMode parse_trait_interaction_mode(const std::string& s) {
    return parse_enum<TraitInteractionMode>(s, {
        {"bilinear", TraitInteractionMode::Bilinear}, {"mlp", TraitInteractionMode::MLP},
        {"attention", TraitInteractionMode::Attention},
    }, "trait interaction mode");
}
ParallelAggregation parse_parallel_aggregation(const std::string& s) {
    return parse_enum<ParallelAggregation>(s, {
        {"concat", ParallelAggregation::Concat}, {"sum", ParallelAggregation::Sum},
        {"mean", ParallelAggregation::Mean}, {"attention", ParallelAggregation::Attention},
        {"gated", ParallelAggregation::Gated},
    }, "parallel aggregation");
}

// ============================================================================
// Sub-config parsers (value map -> engine config struct)
// ============================================================================

FTTransformerConfig parse_ft_transformer_config(const resolve_value* c) {
    FTTransformerConfig x;
    if (vhas(c, "d_model")) x.d_model = (int)vint(c, "d_model");
    if (vhas(c, "n_heads")) x.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "n_layers")) x.n_layers = (int)vint(c, "n_layers");
    if (vhas(c, "attention_dropout")) x.attention_dropout = (float)vdbl(c, "attention_dropout");
    if (vhas(c, "ffn_dropout")) x.ffn_dropout = (float)vdbl(c, "ffn_dropout");
    if (vhas(c, "ffn_multiplier")) x.ffn_multiplier = (int)vint(c, "ffn_multiplier");
    if (vhas(c, "pre_norm")) x.pre_norm = vbool(c, "pre_norm");
    return x;
}
TabNetConfig parse_tabnet_config(const resolve_value* c) {
    TabNetConfig x;
    if (vhas(c, "n_steps")) x.n_steps = (int)vint(c, "n_steps");
    if (vhas(c, "n_d")) x.n_d = (int)vint(c, "n_d");
    if (vhas(c, "n_a")) x.n_a = (int)vint(c, "n_a");
    if (vhas(c, "relaxation_factor")) x.relaxation_factor = (float)vdbl(c, "relaxation_factor");
    if (vhas(c, "sparsity_coefficient")) x.sparsity_coefficient = (float)vdbl(c, "sparsity_coefficient");
    if (vhas(c, "virtual_batch_size")) x.virtual_batch_size = (int)vint(c, "virtual_batch_size");
    if (vhas(c, "use_sparsemax")) x.use_sparsemax = vbool(c, "use_sparsemax");
    return x;
}
SAINTConfig parse_saint_config(const resolve_value* c) {
    SAINTConfig x;
    if (vhas(c, "d_model")) x.d_model = (int)vint(c, "d_model");
    if (vhas(c, "n_heads")) x.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "n_layers")) x.n_layers = (int)vint(c, "n_layers");
    if (vhas(c, "attention_dropout")) x.attention_dropout = (float)vdbl(c, "attention_dropout");
    if (vhas(c, "use_row_attention")) x.use_row_attention = vbool(c, "use_row_attention");
    if (vhas(c, "use_contrastive_pretrain")) x.use_contrastive_pretrain = vbool(c, "use_contrastive_pretrain");
    if (vhas(c, "mixup_alpha")) x.mixup_alpha = (float)vdbl(c, "mixup_alpha");
    return x;
}
GNNConfig parse_gnn_config(const resolve_value* c) {
    GNNConfig x;
    if (vhas(c, "gnn_type")) x.gnn_type = parse_gnn_type(vstr(c, "gnn_type"));
    if (vhas(c, "n_layers")) x.n_layers = (int)vint(c, "n_layers");
    if (vhas(c, "hidden_dim")) x.hidden_dim = (int)vint(c, "hidden_dim");
    if (vhas(c, "n_heads")) x.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "k_neighbors")) x.k_neighbors = (int)vint(c, "k_neighbors");
    if (vhas(c, "graph_mode")) x.graph_mode = parse_graph_construction_mode(vstr(c, "graph_mode"));
    if (vhas(c, "edge_dropout")) x.edge_dropout = (float)vdbl(c, "edge_dropout");
    if (vhas(c, "use_edge_features")) x.use_edge_features = vbool(c, "use_edge_features");
    return x;
}
TraitNetConfig parse_trait_net_config(const resolve_value* c) {
    TraitNetConfig x;
    if (vhas(c, "env_dim")) x.env_dim = (int)vint(c, "env_dim");
    if (vhas(c, "trait_dim")) x.trait_dim = (int)vint(c, "trait_dim");
    if (vhas(c, "interaction_dim")) x.interaction_dim = (int)vint(c, "interaction_dim");
    if (vhas(c, "interaction")) x.interaction = parse_trait_interaction_mode(vstr(c, "interaction"));
    if (vhas(c, "shared_trait_encoder")) x.shared_trait_encoder = vbool(c, "shared_trait_encoder");
    return x;
}
ExcelFormerConfig parse_excelformer_config(const resolve_value* c) {
    ExcelFormerConfig x;
    if (vhas(c, "d_model")) x.d_model = (int)vint(c, "d_model");
    if (vhas(c, "n_heads")) x.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "n_layers")) x.n_layers = (int)vint(c, "n_layers");
    if (vhas(c, "attention_dropout")) x.attention_dropout = (float)vdbl(c, "attention_dropout");
    if (vhas(c, "ffn_multiplier")) x.ffn_multiplier = (int)vint(c, "ffn_multiplier");
    if (vhas(c, "importance_threshold")) x.importance_threshold = (float)vdbl(c, "importance_threshold");
    if (vhas(c, "pre_norm")) x.pre_norm = vbool(c, "pre_norm");
    return x;
}
HeterogeneousGNNConfig parse_heterogeneous_gnn_config(const resolve_value* c) {
    HeterogeneousGNNConfig x;
    if (vhas(c, "hidden_dim")) x.hidden_dim = (int)vint(c, "hidden_dim");
    if (vhas(c, "output_dim")) x.output_dim = (int)vint(c, "output_dim");
    if (vhas(c, "n_layers")) x.n_layers = (int)vint(c, "n_layers");
    if (vhas(c, "n_edge_types")) x.n_edge_types = (int)vint(c, "n_edge_types");
    if (vhas(c, "n_heads")) x.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "dropout")) x.dropout = (float)vdbl(c, "dropout");
    if (vhas(c, "k_cooccurrence")) x.k_cooccurrence = (int)vint(c, "k_cooccurrence");
    if (vhas(c, "cooccurrence_threshold")) x.cooccurrence_threshold = (float)vdbl(c, "cooccurrence_threshold");
    if (vhas(c, "use_taxonomic_edges")) x.use_taxonomic_edges = vbool(c, "use_taxonomic_edges");
    if (vhas(c, "use_cooccurrence_edges")) x.use_cooccurrence_edges = vbool(c, "use_cooccurrence_edges");
    return x;
}
TabMConfig parse_tabm_config(const resolve_value* c) {
    TabMConfig x;
    if (vhas(c, "enabled")) x.enabled = vbool(c, "enabled");
    if (vhas(c, "n_ensembles")) x.n_ensembles = (int)vint(c, "n_ensembles");
    if (vhas(c, "aggregation")) x.aggregation = vstr(c, "aggregation");
    return x;
}
ParallelBranchConfig parse_parallel_branch_config(const resolve_value* c) {
    ParallelBranchConfig x;
    if (vhas(c, "hidden_dims")) x.hidden_dims = vint_vec(c, "hidden_dims");
    if (vhas(c, "activation")) x.activation = parse_activation_type(vstr(c, "activation"));
    if (vhas(c, "normalization")) x.normalization = parse_norm_layer_type(vstr(c, "normalization"));
    if (vhas(c, "dropout")) x.dropout = (float)vdbl(c, "dropout");
    if (vhas(c, "branch_weight")) x.branch_weight = (float)vdbl(c, "branch_weight");
    return x;
}
ParallelLayersConfig parse_parallel_layers_config(const resolve_value* c) {
    ParallelLayersConfig x;
    if (vhas(c, "enabled")) x.enabled = vbool(c, "enabled");
    const resolve_value* branches = vget(c, "branches");
    if (branches && branches->kind == RESOLVE_VALUE_LIST) {
        for (auto* b : branches->items) x.branches.push_back(parse_parallel_branch_config(b));
    }
    if (vhas(c, "aggregation")) x.aggregation = parse_parallel_aggregation(vstr(c, "aggregation"));
    if (vhas(c, "attention_heads")) x.attention_heads = (int)vint(c, "attention_heads");
    if (vhas(c, "use_residual")) x.use_residual = vbool(c, "use_residual");
    return x;
}
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
        out.push_back(ts);
    }
    return out;
}

DatasetConfig parse_dataset_config(const resolve_value* c) {
    DatasetConfig config;
    if (vhas(c, "species_encoding")) config.species_encoding = parse_species_encoding_mode(vstr(c, "species_encoding"));
    if (vhas(c, "hash_dim")) config.hash_dim = (int)vint(c, "hash_dim");
    if (vhas(c, "top_k")) config.top_k = (int)vint(c, "top_k");
    if (vhas(c, "top_k_species")) config.top_k_species = (int)vint(c, "top_k_species");
    if (vhas(c, "selection")) config.selection = parse_selection_mode(vstr(c, "selection"));
    if (vhas(c, "representation")) config.representation = parse_representation_mode(vstr(c, "representation"));
    if (vhas(c, "normalization")) config.normalization = parse_normalization_mode(vstr(c, "normalization"));
    if (vhas(c, "track_unknown_fraction")) config.track_unknown_fraction = vbool(c, "track_unknown_fraction");
    if (vhas(c, "track_unknown_count")) config.track_unknown_count = vbool(c, "track_unknown_count");
    if (vhas(c, "use_taxonomy")) config.use_taxonomy = vbool(c, "use_taxonomy");
    if (vhas(c, "pool_weighting")) config.pool_weighting = parse_pool_weighting(vstr(c, "pool_weighting"));
    if (vhas(c, "pool_species_cap")) config.pool_species_cap = (int)vint(c, "pool_species_cap");
    return config;
}

ModelConfig parse_model_config(const resolve_value* c) {
    ModelConfig config;
    if (vhas(c, "species_encoding")) config.species_encoding = parse_species_encoding_mode(vstr(c, "species_encoding"));
    if (vhas(c, "hash_dim")) config.hash_dim = (int)vint(c, "hash_dim");
    if (vhas(c, "species_embed_dim")) config.species_embed_dim = (int)vint(c, "species_embed_dim");
    if (vhas(c, "genus_emb_dim")) config.genus_emb_dim = (int)vint(c, "genus_emb_dim");
    if (vhas(c, "family_emb_dim")) config.family_emb_dim = (int)vint(c, "family_emb_dim");
    if (vhas(c, "categorical_embed_dim")) config.categorical_embed_dim = (int)vint(c, "categorical_embed_dim");
    if (vhas(c, "top_k")) config.top_k = (int)vint(c, "top_k");
    if (vhas(c, "top_k_species")) config.top_k_species = (int)vint(c, "top_k_species");
    if (vhas(c, "n_taxonomy_slots")) config.n_taxonomy_slots = (int)vint(c, "n_taxonomy_slots");
    if (vhas(c, "hidden_dims")) config.hidden_dims = vint_vec(c, "hidden_dims");
    if (vhas(c, "dropout")) config.dropout = (float)vdbl(c, "dropout");

    if (vhas(c, "cover_dropout")) config.cover_dropout = (float)vdbl(c, "cover_dropout");
    if (vhas(c, "d_model")) config.d_model = (int)vint(c, "d_model");
    if (vhas(c, "n_heads")) config.n_heads = (int)vint(c, "n_heads");
    if (vhas(c, "n_attention_layers")) config.n_attention_layers = (int)vint(c, "n_attention_layers");
    if (vhas(c, "transformer_ff_dim")) config.transformer_ff_dim = (int)vint(c, "transformer_ff_dim");
    if (vhas(c, "transformer_pooling")) config.transformer_pooling = vstr(c, "transformer_pooling");
    if (vhas(c, "transformer_dropout")) config.transformer_dropout = (float)vdbl(c, "transformer_dropout");

    if (vhas(c, "uses_explicit_vector")) config.uses_explicit_vector = vbool(c, "uses_explicit_vector");

    if (vhas(c, "moe_routing")) config.moe_routing = parse_moe_routing_type(vstr(c, "moe_routing"));
    if (vhas(c, "n_experts")) config.n_experts = (int)vint(c, "n_experts");
    if (vhas(c, "expert_hidden_dims")) config.expert_hidden_dims = vint_vec(c, "expert_hidden_dims");
    if (vhas(c, "moe_top_k")) config.moe_top_k = (int)vint(c, "moe_top_k");
    if (vhas(c, "moe_noise_std")) config.moe_noise_std = (float)vdbl(c, "moe_noise_std");
    if (vhas(c, "moe_aux_loss_weight")) config.moe_aux_loss_weight = (float)vdbl(c, "moe_aux_loss_weight");

    if (vhas(c, "activation")) config.activation = parse_activation_type(vstr(c, "activation"));
    if (vhas(c, "normalization")) config.normalization = parse_norm_layer_type(vstr(c, "normalization"));
    if (vhas(c, "norm_groups")) config.norm_groups = (int)vint(c, "norm_groups");
    if (vhas(c, "use_residual")) config.use_residual = vbool(c, "use_residual");
    if (vhas(c, "leaky_relu_slope")) config.leaky_relu_slope = (float)vdbl(c, "leaky_relu_slope");
    if (vhas(c, "elu_alpha")) config.elu_alpha = (float)vdbl(c, "elu_alpha");

    if (vhas(c, "head_hidden_dims")) config.head_hidden_dims = vint_vec(c, "head_hidden_dims");
    if (vhas(c, "head_activation")) config.head_activation = parse_activation_type(vstr(c, "head_activation"));
    if (vhas(c, "head_dropout")) config.head_dropout = (float)vdbl(c, "head_dropout");

    if (vhas(c, "encoder_architecture")) config.encoder_architecture = parse_encoder_architecture(vstr(c, "encoder_architecture"));

    if (vhas(c, "ft_transformer")) config.ft_transformer = parse_ft_transformer_config(vget(c, "ft_transformer"));
    if (vhas(c, "tabnet")) config.tabnet = parse_tabnet_config(vget(c, "tabnet"));
    if (vhas(c, "saint")) config.saint = parse_saint_config(vget(c, "saint"));
    if (vhas(c, "gnn")) config.gnn = parse_gnn_config(vget(c, "gnn"));
    if (vhas(c, "trait_net")) config.trait_net = parse_trait_net_config(vget(c, "trait_net"));
    if (vhas(c, "excelformer")) config.excelformer = parse_excelformer_config(vget(c, "excelformer"));
    if (vhas(c, "heterogeneous_gnn")) config.heterogeneous_gnn = parse_heterogeneous_gnn_config(vget(c, "heterogeneous_gnn"));
    if (vhas(c, "parallel_layers")) config.parallel_layers = parse_parallel_layers_config(vget(c, "parallel_layers"));
    if (vhas(c, "tabm")) config.tabm = parse_tabm_config(vget(c, "tabm"));
    return config;
}

// Schema map -> ResolveSchema. Reads the full set the dataset's schema()
// accessor emits, INCLUDING categorical_names / categorical_vocab_sizes /
// categorical_embed_dim, so a categorical dataset's schema round-trips into a
// categorical-aware model (matches the nanobind path; harmless when absent).
ResolveSchema parse_schema(const resolve_value* s) {
    ResolveSchema schema;
    if (vhas(s, "n_plots")) schema.n_plots = vint(s, "n_plots");
    if (vhas(s, "n_species")) schema.n_species = vint(s, "n_species");
    if (vhas(s, "n_species_vocab")) schema.n_species_vocab = vint(s, "n_species_vocab");
    if (vhas(s, "has_coordinates")) schema.has_coordinates = vbool(s, "has_coordinates");
    if (vhas(s, "has_abundance")) schema.has_abundance = vbool(s, "has_abundance");
    if (vhas(s, "has_taxonomy")) schema.has_taxonomy = vbool(s, "has_taxonomy");
    if (vhas(s, "n_genera")) schema.n_genera = vint(s, "n_genera");
    if (vhas(s, "n_families")) schema.n_families = vint(s, "n_families");
    if (vhas(s, "n_genera_vocab")) schema.n_genera_vocab = vint(s, "n_genera_vocab");
    if (vhas(s, "n_families_vocab")) schema.n_families_vocab = vint(s, "n_families_vocab");
    if (vhas(s, "covariate_names")) schema.covariate_names = vstr_vec(s, "covariate_names");
    if (vhas(s, "track_unknown_fraction")) schema.track_unknown_fraction = vbool(s, "track_unknown_fraction");
    if (vhas(s, "track_unknown_count")) schema.track_unknown_count = vbool(s, "track_unknown_count");
    if (vhas(s, "categorical_names")) schema.categorical_names = vstr_vec(s, "categorical_names");
    if (vhas(s, "categorical_vocab_sizes")) schema.categorical_vocab_sizes = vint_vec(s, "categorical_vocab_sizes");
    if (vhas(s, "categorical_embed_dim")) schema.categorical_embed_dim = vint(s, "categorical_embed_dim");

    const resolve_value* targets = vget(s, "targets");
    if (targets && targets->kind == RESOLVE_VALUE_MAP) {
        for (size_t i = 0; i < targets->keys.size(); ++i) {
            const resolve_value* tc_v = targets->vals[i];
            TargetConfig tc;
            tc.name = targets->keys[i];
            if (vhas(tc_v, "task")) tc.task = parse_task_type(vstr(tc_v, "task"));
            if (vhas(tc_v, "transform")) tc.transform = parse_transform_type(vstr(tc_v, "transform"));
            if (vhas(tc_v, "num_classes")) tc.num_classes = (int)vint(tc_v, "num_classes");
            if (vhas(tc_v, "weight")) tc.weight = (float)vdbl(tc_v, "weight");
            if (vhas(tc_v, "class_weights")) tc.class_weights = vfloat_vec(tc_v, "class_weights");
            schema.targets.push_back(tc);
        }
    }
    return schema;
}

TrainConfig parse_train_config(const resolve_value* c) {
    TrainConfig config;
    if (vhas(c, "batch_size")) config.batch_size = (int)vint(c, "batch_size");
    if (vhas(c, "max_epochs")) config.max_epochs = (int)vint(c, "max_epochs");
    if (vhas(c, "patience")) config.patience = (int)vint(c, "patience");
    if (vhas(c, "lr")) config.lr = (float)vdbl(c, "lr");
    if (vhas(c, "weight_decay")) config.weight_decay = (float)vdbl(c, "weight_decay");
    if (vhas(c, "device")) config.device = (vstr(c, "device") == "cuda") ? torch::kCUDA : torch::kCPU;
    if (vhas(c, "loss_config")) config.loss_config = parse_loss_config_mode(vstr(c, "loss_config"));
    if (vhas(c, "lr_scheduler")) config.lr_scheduler = parse_lr_scheduler_type(vstr(c, "lr_scheduler"));
    if (vhas(c, "lr_step_size")) config.lr_step_size = (int)vint(c, "lr_step_size");
    if (vhas(c, "lr_gamma")) config.lr_gamma = (float)vdbl(c, "lr_gamma");
    if (vhas(c, "lr_min")) config.lr_min = (float)vdbl(c, "lr_min");
    if (vhas(c, "phase_boundaries")) {
        auto pb = vint_vec(c, "phase_boundaries");
        if (pb.size() >= 2) config.phase_boundaries = {(int)pb[0], (int)pb[1]};
    }
    if (vhas(c, "band_thresholds")) config.band_thresholds = vfloat_vec(c, "band_thresholds");
    if (vhas(c, "checkpoint_dir")) config.checkpoint_dir = vstr(c, "checkpoint_dir");
    if (vhas(c, "checkpoint_every")) config.checkpoint_every = (int)vint(c, "checkpoint_every");
    if (vhas(c, "use_amp")) config.use_amp = vbool(c, "use_amp");
    if (vhas(c, "amp_init_scale")) config.amp_init_scale = (float)vdbl(c, "amp_init_scale");
    if (vhas(c, "amp_growth_factor")) config.amp_growth_factor = (float)vdbl(c, "amp_growth_factor");
    if (vhas(c, "amp_backoff_factor")) config.amp_backoff_factor = (float)vdbl(c, "amp_backoff_factor");
    if (vhas(c, "amp_growth_interval")) config.amp_growth_interval = (int)vint(c, "amp_growth_interval");
    if (vhas(c, "cudnn_benchmark")) config.cudnn_benchmark = vbool(c, "cudnn_benchmark");
    if (vhas(c, "allow_tf32")) config.allow_tf32 = vbool(c, "allow_tf32");
    if (vhas(c, "vram_fraction")) config.vram_fraction = (float)vdbl(c, "vram_fraction");
    if (vhas(c, "batch_size_floor")) config.batch_size_floor = (int)vint(c, "batch_size_floor");
    return config;
}

RunMetadata parse_run_metadata(const resolve_value* c) {
    RunMetadata rm;
    if (vhas(c, "created_at")) rm.created_at = vstr(c, "created_at");
    if (vhas(c, "completed_at")) rm.completed_at = vstr(c, "completed_at");
    if (vhas(c, "train_time_seconds")) rm.train_time_seconds = (float)vdbl(c, "train_time_seconds");
    if (vhas(c, "n_plots_train")) rm.n_plots_train = vint(c, "n_plots_train");
    if (vhas(c, "n_plots_test")) rm.n_plots_test = vint(c, "n_plots_test");
    if (vhas(c, "best_epoch")) rm.best_epoch = (int)vint(c, "best_epoch");
    if (vhas(c, "total_epochs")) rm.total_epochs = (int)vint(c, "total_epochs");
    return rm;
}

// ============================================================================
// Result-struct -> value-tree converters. One-for-one with r/src/rcpp_common.h.
// ============================================================================

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
    v_put(m, "target_name", v_string(cp.target_name));
    v_put(m, "class_names", v_string_array(cp.class_names));
    v_put(m, "predictions",
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
    return m;
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
    if (s.continuous_mean.defined()) v_put(m, "continuous_mean", tensor_to_vec(s.continuous_mean));
    if (s.continuous_scale.defined()) v_put(m, "continuous_scale", tensor_to_vec(s.continuous_scale));
    return m;
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
resolve_value* train_config_to_value(const TrainConfig& c) {
    auto* m = v_map();
    v_put(m, "batch_size", v_int(c.batch_size));
    v_put(m, "batch_size_floor", v_int(c.batch_size_floor));
    v_put(m, "max_epochs", v_int(c.max_epochs));
    v_put(m, "patience", v_int(c.patience));
    v_put(m, "lr", v_double(c.lr));
    v_put(m, "weight_decay", v_double(c.weight_decay));
    v_put(m, "phase_boundaries",
          v_int_array({c.phase_boundaries.first, c.phase_boundaries.second}));
    v_put(m, "loss_config", v_int(static_cast<int>(c.loss_config)));
    v_put(m, "lr_scheduler", v_int(static_cast<int>(c.lr_scheduler)));
    v_put(m, "lr_step_size", v_int(c.lr_step_size));
    v_put(m, "lr_gamma", v_double(c.lr_gamma));
    v_put(m, "lr_min", v_double(c.lr_min));
    v_put(m, "vram_fraction", v_double(c.vram_fraction));
    v_put(m, "band_thresholds", v_double_array(c.band_thresholds));
    return m;
}
resolve_value* run_metadata_to_value(const RunMetadata& m0) {
    auto* m = v_map();
    v_put(m, "resolve_version", v_string(m0.resolve_version));
    v_put(m, "created_at", v_string(m0.created_at));
    v_put(m, "completed_at", v_string(m0.completed_at));
    v_put(m, "train_time_seconds", v_double(m0.train_time_seconds));
    v_put(m, "n_plots_train", v_double(static_cast<double>(m0.n_plots_train)));
    v_put(m, "n_plots_test", v_double(static_cast<double>(m0.n_plots_test)));
    v_put(m, "best_epoch", v_int(m0.best_epoch));
    v_put(m, "total_epochs", v_int(m0.total_epochs));
    v_put(m, "final_metrics", nested_metrics_to_value(m0.final_metrics));
    return m;
}

// target map -> value map (name -> double array). Used by dataset.targets(),
// model.forward(), trainer.predict(), predictor.predict().
resolve_value* target_map_to_value(const std::unordered_map<std::string, torch::Tensor>& m0) {
    auto* m = v_map();
    for (const auto& [name, tensor] : m0) v_put(m, name, tensor_to_vec(tensor));
    return m;
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

namespace {
// Platform-specific bits kept OUT of the CAPI_BODY_* macro bodies: the C
// preprocessor does not allow #if/#endif inside a macro argument.
std::string default_cuda_alloc_conf() {
    std::string base = "garbage_collection_threshold:0.8,max_split_size_mb:256";
#if !defined(_WIN32)
    base = "expandable_segments:True," + base;
#endif
    return base;
}
void set_env_var(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
}  // namespace

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
        std::string base = default_cuda_alloc_conf();
        const char* existing = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
        if (force || existing == nullptr || existing[0] == '\0') {
            set_env_var("PYTORCH_CUDA_ALLOC_CONF", base.c_str());
            return v_string(base);
        }
        return v_string(std::string(existing));
    })
}

int resolve_metric_band_accuracy(const double* pred, const double* target, int64_t n, double threshold, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::band_accuracy(pt, tt, (float)threshold);
        return 0;
    })
}
int resolve_metric_mae(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::mae(pt, tt);
        return 0;
    })
}
int resolve_metric_rmse(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::rmse(pt, tt);
        return 0;
    })
}
int resolve_metric_smape(const double* pred, const double* target, int64_t n, double eps, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::smape(pt, tt, (float)eps);
        return 0;
    })
}
int resolve_metric_accuracy(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::accuracy(pt, tt);
        return 0;
    })
}
int resolve_metric_r_squared(const double* pred, const double* target, int64_t n, double* out) {
    CAPI_BODY_INT({
        std::vector<float> p(pred, pred + n), t(target, target + n);
        auto pt = torch::from_blob(p.data(), {n}, torch::kFloat32).clone();
        auto tt = torch::from_blob(t.data(), {n}, torch::kFloat32).clone();
        *out = resolve::Metrics::r_squared(pt, tt);
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
        if (w == "categorical_vocab") return categorical_vocab_to_value(d.categorical_vocab());
        if (w == "targets") return target_map_to_value(d.targets());
        if (w == "plot_ids") return v_string_array(d.plot_ids());
        if (w == "species_vocab") return v_string_array(d.species_vocab());
        if (w == "n_plots") return v_int(d.n_plots());
        if (w == "has_raw_species_data") return v_bool(d.has_raw_species_data());
        if (w == "raw_species_ids") return vec_or_null(d.raw_species_ids());
        if (w == "raw_weights") return vec_or_null(d.raw_weights());
        if (w == "plot_offsets") return vec_or_null(d.plot_offsets());

        if (w == "taxonomy_vocab") {
            const auto& tv = d.taxonomy_vocab();
            auto* m = v_map();
            v_put(m, "n_genera", v_int(tv.n_genera()));
            v_put(m, "n_families", v_int(tv.n_families()));
            return m;
        }

        if (w == "config") {
            const auto& c = d.config();
            std::string enc;
            switch (c.species_encoding) {
                case SpeciesEncodingMode::Hash: enc = "hash"; break;
                case SpeciesEncodingMode::Embed: enc = "embed"; break;
                case SpeciesEncodingMode::Sparse: enc = "sparse"; break;
                case SpeciesEncodingMode::RankPool: enc = "rank_pool"; break;
                case SpeciesEncodingMode::Transformer: enc = "transformer"; break;
            }
            auto* m = v_map();
            v_put(m, "species_encoding", v_string(enc));
            v_put(m, "hash_dim", v_int(c.hash_dim));
            v_put(m, "top_k", v_int(c.top_k));
            v_put(m, "top_k_species", v_int(c.top_k_species));
            v_put(m, "track_unknown_fraction", v_bool(c.track_unknown_fraction));
            v_put(m, "track_unknown_count", v_bool(c.track_unknown_count));
            v_put(m, "use_taxonomy", v_bool(c.use_taxonomy));
            return m;
        }

        if (w == "schema") {
            const auto& s = d.schema();
            auto* targets_m = v_map();
            for (const auto& tc : s.targets) {
                std::string task_str = (tc.task == TaskType::Regression) ? "regression" : "classification";
                std::string transform_str = (tc.transform == TransformType::Log1p) ? "log1p" : "none";
                auto* tm = v_map();
                v_put(tm, "task", v_string(task_str));
                v_put(tm, "transform", v_string(transform_str));
                v_put(tm, "num_classes", v_int(tc.num_classes));
                v_put(tm, "weight", v_double(tc.weight));
                v_put(tm, "class_weights", v_double_array(tc.class_weights));
                v_put(targets_m, tc.name, tm);
            }
            auto* m = v_map();
            v_put(m, "n_plots", v_int(s.n_plots));
            v_put(m, "n_species", v_int(s.n_species));
            v_put(m, "n_species_vocab", v_int(s.n_species_vocab));
            v_put(m, "has_coordinates", v_bool(s.has_coordinates));
            v_put(m, "has_abundance", v_bool(s.has_abundance));
            v_put(m, "has_taxonomy", v_bool(s.has_taxonomy));
            v_put(m, "n_genera", v_int(s.n_genera));
            v_put(m, "n_families", v_int(s.n_families));
            v_put(m, "n_genera_vocab", v_int(s.n_genera_vocab));
            v_put(m, "n_families_vocab", v_int(s.n_families_vocab));
            v_put(m, "covariate_names", v_string_array(s.covariate_names));
            v_put(m, "targets", targets_m);
            v_put(m, "track_unknown_fraction", v_bool(s.track_unknown_fraction));
            v_put(m, "track_unknown_count", v_bool(s.track_unknown_count));
            v_put(m, "categorical_names", v_string_array(s.categorical_names));
            v_put(m, "categorical_vocab_sizes", v_int_array(s.categorical_vocab_sizes));
            v_put(m, "categorical_embed_dim", v_int(s.categorical_embed_dim));
            return m;
        }

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
            return tensor_to_vec(latent);
        }
        if (mt == "forward_with_aux") {
            auto result = m->model->forward_with_aux(
                in.continuous, in.genus_ids, in.family_ids, in.species_ids, in.species_vector,
                in.pool_genus_ids, in.pool_family_ids, in.pool_weights, in.pool_mask,
                in.pool_has_cover, in.categorical_ids);
            auto* ret = v_map();
            v_put(ret, "outputs", target_map_to_value(result.outputs));
            if (result.moe_aux_loss.defined()) v_put(ret, "moe_aux_loss", tensor_to_vec(result.moe_aux_loss));
            return ret;
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
            v_put(ret, "latent", tensor_to_mat(latent));
            auto* acts = v_list();
            for (const auto& a : activations) v_append(acts, tensor_to_mat(a));
            v_put(ret, "activations", acts);
            return ret;
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
            switch (m->model->species_encoding()) {
                case SpeciesEncodingMode::Hash: return v_string("hash");
                case SpeciesEncodingMode::Embed: return v_string("embed");
                case SpeciesEncodingMode::Sparse: return v_string("sparse");
                case SpeciesEncodingMode::RankPool: return v_string("rank_pool");
                case SpeciesEncodingMode::Transformer: return v_string("transformer");
                default: return v_string("unknown");
            }
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
        if (w == "config") {
            const auto& c = tr.config();
            auto* m = v_map();
            v_put(m, "batch_size", v_int(c.batch_size));
            v_put(m, "batch_size_floor", v_int(c.batch_size_floor));
            v_put(m, "max_epochs", v_int(c.max_epochs));
            v_put(m, "patience", v_int(c.patience));
            v_put(m, "lr", v_double(c.lr));
            v_put(m, "weight_decay", v_double(c.weight_decay));
            v_put(m, "device", v_string(c.device.is_cuda() ? "cuda" : "cpu"));
            v_put(m, "vram_fraction", v_double(c.vram_fraction));
            return m;
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
    v_put(result, "predictions", target_map_to_value(preds.predictions));
    v_put(result, "targets", target_map_to_value(preds.targets));
    if (return_latent && preds.latent.defined()) v_put(result, "latent", tensor_to_mat(preds.latent));
    v_put(result, "plot_ids", v_string_array(preds.plot_ids));
    return result;
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
        if (w == "genus_embeddings") return tensor_to_mat(pr.get_genus_embeddings());
        if (w == "family_embeddings") return tensor_to_mat(pr.get_family_embeddings());
        if (w == "species_embeddings") return tensor_to_mat(pr.get_species_embeddings());
        throw std::runtime_error("predictor_get: unknown accessor '" + w + "'");
    })
}

}  // extern "C"
