#pragma once

// Compile-time member count of an aggregate. A struct that keeps a list of its
// members next to its declaration (the configuration field registries, the
// PlotInputs field list) static_asserts that the list and the struct agree, so
// a member added without a row fails the build.

#include <cstddef>
#include <type_traits>
#include <utility>

namespace resolve::registry_detail {

// Converts to any type, so a brace-init probe can supply one initializer per
// aggregate member without naming the member types. Declared and never defined:
// every use sits inside decltype, which is unevaluated.
struct AnyInit {
    template <typename T,
              typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, AnyInit>>>
    constexpr operator T() const;  // NOLINT(google-explicit-constructor)
};

template <std::size_t>
using AnyInitAt = AnyInit;

template <typename T, typename... A>
constexpr auto brace_init_ok(int) -> decltype(T{std::declval<A>()...}, true) {
    return true;
}
template <typename T, typename... A>
constexpr bool brace_init_ok(...) {
    return false;
}

template <typename T, std::size_t... I>
constexpr bool brace_init_with(std::index_sequence<I...>) {
    return brace_init_ok<T, AnyInitAt<I>...>(0);
}

// Number of direct members of an aggregate. Probes brace initialization with an
// increasing number of universal initializers; the first count that fails to
// compile is one past the member count. Each member accepts an AnyInit directly
// (it converts to anything), so brace elision never folds a nested aggregate's
// members into the outer count -- the number is the DIRECT member count.
template <typename T, std::size_t N = 0>
constexpr std::size_t aggregate_field_count() {
    if constexpr (brace_init_with<T>(std::make_index_sequence<N + 1>{})) {
        return aggregate_field_count<T, N + 1>();
    } else {
        return N;
    }
}

}  // namespace resolve::registry_detail
