// resolve_capi_dynload.cpp - the resolve_c runtime loader.
//
// Binds every symbol of the resolve_c C ABI into g_resolve_capi via
// LoadLibrary/dlopen + GetProcAddress/dlsym. The symbol list is the single
// source in resolve/resolve_capi_symbols.inc, so this file, the table struct,
// and the forwarders can never drift apart. See resolve_capi_dynload.h.
#include "resolve_capi_dynload.h"

#include <string>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

// The single definition of the table (declared extern in the header).
ResolveCApiTable g_resolve_capi = {};

namespace {

// Message slot for the most recent load failure. A plain function-local static
// keeps it out of the header and gives it a stable address for the C accessor.
std::string& dynload_error_slot() {
    static std::string slot;
    return slot;
}

#if defined(_WIN32)
using lib_handle_t = HMODULE;

lib_handle_t dl_open(const char* path) {
    // LOAD_WITH_ALTERED_SEARCH_PATH: resolve resolve_c's sibling libtorch DLLs
    // from resolve_c's own directory instead of the process working directory,
    // so the caller only has to know where resolve_c.dll lives.
    return ::LoadLibraryExA(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
}
void* dl_sym(lib_handle_t h, const char* name) {
    return reinterpret_cast<void*>(::GetProcAddress(h, name));
}
void dl_close(lib_handle_t h) { ::FreeLibrary(h); }
std::string dl_error() {
    return "WinError " + std::to_string(static_cast<long>(::GetLastError()));
}
#else
using lib_handle_t = void*;

lib_handle_t dl_open(const char* path) { return ::dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void* dl_sym(lib_handle_t h, const char* name) { return ::dlsym(h, name); }
void dl_close(lib_handle_t h) { ::dlclose(h); }
std::string dl_error() {
    const char* e = ::dlerror();
    return e ? std::string(e) : std::string("dlopen/dlsym failed");
}
#endif

lib_handle_t g_handle = nullptr;

}  // namespace

extern "C" int resolve_capi_load(const char* path) {
    dynload_error_slot().clear();
    if (g_handle != nullptr) return 0;  // already loaded; idempotent
    if (path == nullptr || path[0] == '\0') {
        dynload_error_slot() = "resolve_capi_load: empty library path";
        return -1;
    }

    lib_handle_t h = dl_open(path);
    if (h == nullptr) {
        dynload_error_slot() = "resolve_capi_load: could not open '" +
                               std::string(path) + "' (" + dl_error() + ")";
        return -1;
    }

    // Bind into a local table first; only commit if every symbol resolved, so a
    // stale / mismatched library never leaves g_resolve_capi half-populated.
    ResolveCApiTable t = {};
    std::string missing;

#define RESOLVE_SYM(ret, name, params, args)                    \
    {                                                           \
        void* s = dl_sym(h, #name);                             \
        if (s == nullptr) { missing += ' '; missing += #name; } \
        t.name = reinterpret_cast<ret (*) params>(s);           \
    }
#define RESOLVE_SYM_VOID(name, params, args)                    \
    {                                                           \
        void* s = dl_sym(h, #name);                             \
        if (s == nullptr) { missing += ' '; missing += #name; } \
        t.name = reinterpret_cast<void (*) params>(s);          \
    }
#include "resolve/resolve_capi_symbols.inc"
#undef RESOLVE_SYM
#undef RESOLVE_SYM_VOID

    if (!missing.empty()) {
        dl_close(h);
        dynload_error_slot() = "resolve_capi_load: '" + std::string(path) +
                               "' is missing symbols:" + missing;
        return -1;
    }

    g_resolve_capi = t;
    g_handle = h;
    return 0;
}

extern "C" int resolve_capi_available(void) {
    return g_handle != nullptr ? 1 : 0;
}

extern "C" const char* resolve_capi_dynload_error(void) {
    return dynload_error_slot().c_str();
}
