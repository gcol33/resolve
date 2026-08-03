// resolve_capi_dynload.h - runtime binding of the resolve_c C ABI.
//
// The R package used to LINK the resolve_c import library at build time
// (Makevars: -lresolve_c). That makes resolve.dll/.so hard-depend on resolve_c
// + libtorch at load time, so the package cannot build or load where those
// binaries are absent -- which is exactly CRAN's check machines. This header
// removes the link-time dependency: it declares a function-pointer table over
// every resolve_c symbol, and identically-named `static inline` forwarders that
// dispatch through the table. `resolve_capi_load()` binds the table at runtime
// via LoadLibrary/dlopen (the "lantern" pattern the mlverse/torch R package
// uses for libtorch). Every existing call site in rcpp_common.h / rcpp_*.h stays
// byte-identical: `resolve_value_new_map()` still compiles and now resolves to
// the forwarder instead of an imported symbol.
//
// Contract: `resolve_capi_load()` MUST run (once) before any forwarder is
// called. The R package calls it from .onLoad (zzz.R) after locating the
// installed resolve_c; the R front-door verbs gate on `resolve_capi_available()`
// so a missing backend yields a clean R error, never a NULL-pointer call.
#ifndef RESOLVE_CAPI_DYNLOAD_H
#define RESOLVE_CAPI_DYNLOAD_H

// Suppress the function PROTOTYPES in resolve_capi.h: this header supplies its
// own identically-named forwarders instead. The opaque handle typedefs and the
// resolve_value_kind_t enum are outside that guard and still come in.
#define RESOLVE_CAPI_DYNLOAD 1
#include "resolve/resolve_capi.h"

#ifdef __cplusplus
extern "C" {
#endif

// Function-pointer table: one member per resolve_c symbol, same name and
// signature as the C ABI. Generated from the single-source symbol list.
typedef struct ResolveCApiTable {
#define RESOLVE_SYM(ret, name, params, args)  ret (*name) params;
#define RESOLVE_SYM_VOID(name, params, args)  void (*name) params;
#include "resolve/resolve_capi_symbols.inc"
#undef RESOLVE_SYM
#undef RESOLVE_SYM_VOID
} ResolveCApiTable;

// The one instance (defined in resolve_capi_dynload.cpp). All members are NULL
// until resolve_capi_load() succeeds.
extern ResolveCApiTable g_resolve_capi;

// Load resolve_c from `path` and bind every symbol into g_resolve_capi.
// Returns 0 on success, -1 on failure (message via resolve_capi_dynload_error()).
// Idempotent: a second call while already loaded is a no-op returning 0. On a
// partial library (any symbol missing) it binds nothing, closes the handle, and
// fails -- the table is never left half-populated.
int resolve_capi_load(const char* path);

// 1 once a load has succeeded, else 0. Cheap; safe to call before any load.
int resolve_capi_available(void);

// Human-readable message for the most recent resolve_capi_load() failure
// (empty string if none). Distinct from resolve_last_error(), which reports
// engine-side failures once the library is bound.
const char* resolve_capi_dynload_error(void);

#ifdef __cplusplus
}  // extern "C"
#endif

// Forwarders: identical names and signatures to the resolve_c C ABI, each
// dispatching through the table. `static inline` so every TU gets its own copy
// with no link-time symbol and no collision (the real prototypes are suppressed
// above). This is what keeps the marshaling code in rcpp_common.h and the
// rcpp_*.h clients unchanged.
#define RESOLVE_SYM(ret, name, params, args) \
    static inline ret name params { return g_resolve_capi.name args; }
#define RESOLVE_SYM_VOID(name, params, args) \
    static inline void name params { g_resolve_capi.name args; }
#include "resolve/resolve_capi_symbols.inc"
#undef RESOLVE_SYM
#undef RESOLVE_SYM_VOID

#endif  // RESOLVE_CAPI_DYNLOAD_H
