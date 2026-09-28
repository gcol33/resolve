#pragma once

// SHA-256 (FIPS 180-4), for the checksums a model suite records for its weight
// files. A released suite is downloaded, copied and unpacked on machines the
// engine has never seen, and a truncated or swapped checkpoint loads without
// complaint as long as its tensors have the right shapes; the digest is what
// tells the two apart before a single prediction is made.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace resolve {

class Sha256 {
public:
    Sha256() noexcept { reset(); }

    void reset() noexcept;
    void update(const void* data, std::size_t size) noexcept;
    void update(std::string_view bytes) noexcept { update(bytes.data(), bytes.size()); }
    // Lower-case hexadecimal digest. Finishes the hash; call reset() to reuse.
    [[nodiscard]] std::string hex_digest();

private:
    void compress(const uint8_t* block) noexcept;

    uint32_t state_[8] = {};
    uint8_t buffer_[64] = {};
    std::size_t buffered_ = 0;
    uint64_t total_bytes_ = 0;
};

// Digest of an in-memory byte string.
[[nodiscard]] std::string sha256_hex(std::string_view bytes);

// Digest of a file's bytes, read in blocks. Throws std::runtime_error naming
// the path when it cannot be opened or read to the end.
[[nodiscard]] std::string sha256_file(const std::string& path);

}  // namespace resolve
