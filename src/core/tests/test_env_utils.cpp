// Contract for the small shared helpers extracted while enabling -Wall/-Wextra
// (issue #109): the single environment-variable accessor that replaced seven
// std::getenv call sites, the cosine ramp shared by the LR schedule and the
// JEPA EMA schedule, and the allocator-config entry point that the nanobind and
// C-ABI shims each used to carry a copy of.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "resolve/env.hpp"
#include "resolve/gpu.hpp"
#include "resolve/io_retry.hpp"
#include "resolve/utils.hpp"

#include <cmath>
#include <string>

using namespace resolve;
using Catch::Matchers::WithinAbs;

namespace {

// Restores a variable to whatever it held on entry, so a test never leaks a
// setting into the ones that follow (several read their variable once into a
// function-local static).
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name), saved_(get_env(name)) {
        if (value == nullptr) {
            clear();
        } else {
            set_env(name, value);
        }
    }
    ~ScopedEnv() {
        if (saved_.has_value()) {
            set_env(name_, saved_->c_str());
        } else {
            clear();
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    void clear() { unset_env(name_); }

    const char* name_;
    std::optional<std::string> saved_;
};

constexpr const char* kProbe = "RESOLVE_TEST_ENV_PROBE";

}  // namespace

TEST_CASE("get_env returns an owned copy, or nullopt when unset", "[env]") {
    SECTION("a set variable comes back by value") {
        ScopedEnv guard(kProbe, "hello");
        const auto value = get_env(kProbe);
        REQUIRE(value.has_value());
        CHECK(*value == "hello");
    }

    SECTION("an unset variable is nullopt") {
        ScopedEnv guard(kProbe, nullptr);
        CHECK_FALSE(get_env(kProbe).has_value());
    }

    SECTION("unset_env removes rather than emptying") {
        // The reason unset_env exists: _putenv_s(name, "") deletes on Windows
        // while setenv(name, "", 1) keeps the name defined with an empty value,
        // so a test that spelled removal as set_env(name, "") saw "unset" on
        // one platform and "present but empty" on the other.
        set_env(kProbe, "value");
        REQUIRE(get_env(kProbe).has_value());
        unset_env(kProbe);
        CHECK_FALSE(get_env(kProbe).has_value());
    }

    SECTION("a null name is nullopt rather than a crash") {
        CHECK_FALSE(get_env(nullptr).has_value());
    }

    SECTION("the returned string outlives a later set_env") {
        ScopedEnv guard(kProbe, "first");
        const auto first = get_env(kProbe);
        REQUIRE(first.has_value());
        set_env(kProbe, "second");
        // The point of copying at the accessor: the earlier read is still
        // valid and still reads "first", where a raw getenv pointer could not
        // be relied on.
        CHECK(*first == "first");
        CHECK(*get_env(kProbe) == "second");
    }
}

TEST_CASE("env flags follow the anything-but-0 convention", "[env]") {
    SECTION("unset") {
        ScopedEnv guard(kProbe, nullptr);
        CHECK_FALSE(env_flag_enabled(kProbe));
        CHECK_FALSE(env_flag_disabled(kProbe));
    }

    SECTION("set to 0") {
        ScopedEnv guard(kProbe, "0");
        CHECK_FALSE(env_flag_enabled(kProbe));
        CHECK(env_flag_disabled(kProbe));
    }

    SECTION("set to anything else") {
        ScopedEnv guard(kProbe, "1");
        CHECK(env_flag_enabled(kProbe));
        CHECK_FALSE(env_flag_disabled(kProbe));
    }
}

TEST_CASE("io retry defaults read the environment", "[env][io_retry]") {
    SECTION("defaults with no override") {
        ScopedEnv attempts("RESOLVE_IO_RETRY_ATTEMPTS", nullptr);
        ScopedEnv backoff("RESOLVE_IO_RETRY_BACKOFF_MS", nullptr);
        CHECK(io::default_retry_attempts() == 3);
        CHECK(io::default_retry_backoff_ms() == 100);
    }

    SECTION("a valid override wins") {
        ScopedEnv attempts("RESOLVE_IO_RETRY_ATTEMPTS", "7");
        ScopedEnv backoff("RESOLVE_IO_RETRY_BACKOFF_MS", "250");
        CHECK(io::default_retry_attempts() == 7);
        CHECK(io::default_retry_backoff_ms() == 250);
    }

    SECTION("an unusable override falls back to the default") {
        ScopedEnv attempts("RESOLVE_IO_RETRY_ATTEMPTS", "not_a_number");
        ScopedEnv backoff("RESOLVE_IO_RETRY_BACKOFF_MS", "-5");
        CHECK(io::default_retry_attempts() == 3);
        CHECK(io::default_retry_backoff_ms() == 100);
    }
}

TEST_CASE("cosine_ramp is a clamped half cosine from 1 to 0", "[utils]") {
    CHECK_THAT(cosine_ramp(0.0f), WithinAbs(1.0f, 1e-6f));
    CHECK_THAT(cosine_ramp(0.5f), WithinAbs(0.5f, 1e-6f));
    CHECK_THAT(cosine_ramp(1.0f), WithinAbs(0.0f, 1e-6f));

    // Monotonically decreasing across the interval.
    float previous = cosine_ramp(0.0f);
    for (int i = 1; i <= 20; ++i) {
        const float value = cosine_ramp(static_cast<float>(i) / 20.0f);
        CHECK(value <= previous + 1e-6f);
        previous = value;
    }

    // Clamped, so an out-of-range progress cannot swing the schedule back up.
    CHECK_THAT(cosine_ramp(-0.5f), WithinAbs(1.0f, 1e-6f));
    CHECK_THAT(cosine_ramp(2.0f), WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("configure_cuda_allocator respects an existing setting", "[env][gpu]") {
    const std::string expected_default = default_cuda_alloc_conf();

    SECTION("sets the default when unset") {
        ScopedEnv guard("PYTORCH_CUDA_ALLOC_CONF", nullptr);
        CHECK(configure_cuda_allocator(/*force=*/false) == expected_default);
        CHECK(*get_env("PYTORCH_CUDA_ALLOC_CONF") == expected_default);
    }

    SECTION("leaves a user's own setting alone") {
        ScopedEnv guard("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64");
        CHECK(configure_cuda_allocator(/*force=*/false) == "max_split_size_mb:64");
        CHECK(*get_env("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:64");
    }

    SECTION("force overwrites it") {
        ScopedEnv guard("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64");
        CHECK(configure_cuda_allocator(/*force=*/true) == expected_default);
        CHECK(*get_env("PYTORCH_CUDA_ALLOC_CONF") == expected_default);
    }
}
