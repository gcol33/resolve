#pragma once

// The single point at which RESOLVE reads or writes a process environment
// variable. Torch-free, so the light headers (io_retry.hpp) can use it.

#include <cstdlib>
#include <optional>
#include <string>

namespace resolve {

// Read an environment variable into an owned string; nullopt when unset.
//
// One accessor rather than a std::getenv per call site: getenv hands back a
// pointer into the process environment block that a concurrent putenv/setenv
// can invalidate, which is the substance of MSVC's C4996 deprecation. Copying
// at the single point of access settles the lifetime question everywhere, and
// on MSVC the copy comes from _dupenv_s, the thread-safe form the CRT asks for.
inline std::optional<std::string> get_env(const char* name) {
    if (name == nullptr) {
        return std::nullopt;
    }
#if defined(_MSC_VER)
    char* buf = nullptr;
    size_t len = 0;
    if (::_dupenv_s(&buf, &len, name) != 0 || buf == nullptr) {
        std::free(buf);
        return std::nullopt;
    }
    std::string value(buf);
    std::free(buf);
    return value;
#else
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return std::nullopt;
    }
    return std::string(value);
#endif
}

// Set an environment variable in the current process.
inline void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// RESOLVE's debug/feature switches share one convention: set to anything other
// than the literal "0" to enable, unset or "0" to disable.
inline bool env_flag_enabled(const char* name) {
    const auto value = get_env(name);
    return value.has_value() && *value != "0";
}

// The same convention read the other way round, for switches that default on.
inline bool env_flag_disabled(const char* name) {
    const auto value = get_env(name);
    return value.has_value() && *value == "0";
}

}  // namespace resolve
