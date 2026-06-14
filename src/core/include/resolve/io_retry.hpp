#pragma once

#include "resolve/types.hpp"  // LogCallback, default_log

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace resolve {
namespace io {

// A transient, retryable storage read/write failure (issue #20): a file that
// could not be opened, or a stream that errored mid-read/-write. The explicit
// I/O paths (CSV reader) throw this specifically so with_retry can retry a
// flaky-disk hiccup while letting non-I/O errors (a CSV parse error, a corrupt
// checkpoint) propagate immediately -- so a permanent fault never triggers a
// re-read of a multi-GB dataset.
class IOError : public std::runtime_error {
public:
    explicit IOError(const std::string& msg) : std::runtime_error(msg) {}
};

// Default retry policy, read once from the environment so a user on slow/flaky
// storage can widen it without a rebuild:
//   RESOLVE_IO_RETRY_ATTEMPTS   total tries (default 3)
//   RESOLVE_IO_RETRY_BACKOFF_MS base backoff in ms (default 100; exp, capped)
inline int default_retry_attempts() {
    if (const char* e = std::getenv("RESOLVE_IO_RETRY_ATTEMPTS")) {
        const int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 3;
}
inline int default_retry_backoff_ms() {
    if (const char* e = std::getenv("RESOLVE_IO_RETRY_BACKOFF_MS")) {
        const int v = std::atoi(e);
        if (v >= 0) return v;
    }
    return 100;
}

// Bounded retry-with-backoff for idempotent, explicit file I/O (issue #20).
//
// Runs fn() and, on a thrown exception assignable to Exc, logs and retries up
// to `attempts` total tries with exponential backoff (`backoff_ms << (k-1)`,
// capped at 5 s), then rethrows the final exception. An exception NOT derived
// from Exc is never caught -- it propagates on the first try. Pass
// Exc = io::IOError to retry only transient storage faults (e.g. the dataset
// load, where re-reading on a permanent parse error would be expensive); the
// default Exc = std::exception retries any failure (e.g. a small checkpoint
// load, where a re-read is cheap). attempts / backoff_ms < 0 take the env
// defaults above.
//
// Scope note (issue #20): only RESOLVE's explicit stream reads/writes are
// retryable here. mmap-backed page-ins (torch::load mmap) and DLL code-page
// faults surface as OS structured exceptions at an arbitrary instruction and
// cannot be resumed; those are issue #19's domain (fail fast, do not hang).
template <typename Exc = std::exception, typename Fn>
auto with_retry(Fn&& fn,
                const char* what,
                int attempts = -1,
                int backoff_ms = -1,
                LogCallback log = default_log) -> decltype(fn()) {
    if (attempts < 0) attempts = default_retry_attempts();
    if (backoff_ms < 0) backoff_ms = default_retry_backoff_ms();
    if (attempts < 1) attempts = 1;
    constexpr int kMaxBackoffMs = 5000;

    int tried = 0;
    for (;;) {
        try {
            return fn();
        } catch (const Exc& e) {
            ++tried;
            if (tried >= attempts) throw;

            int delay = backoff_ms;
            for (int k = 1; k < tried && delay < kMaxBackoffMs; ++k) delay *= 2;
            if (delay > kMaxBackoffMs) delay = kMaxBackoffMs;

            log(std::string("io retry ") + std::to_string(tried) + "/" +
                std::to_string(attempts - 1) + " for " + (what ? what : "io") +
                " after error: " + e.what() + " (waiting " +
                std::to_string(delay) + " ms)");
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
    }
}

}  // namespace io
}  // namespace resolve
