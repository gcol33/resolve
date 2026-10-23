#pragma once

// A list of strings through a torch::serialize archive, which stores tensors
// only. The list travels as two entries: "<key>_lengths", an int64 vector of
// byte lengths, and "<key>_bytes", the concatenated UTF-8 bytes. Two entries
// per list rather than per string, so a 30k-species vocabulary stays two zip
// members. Every reader of a persisted string list goes through
// try_read_string_list, which bounds-checks the lengths against the byte
// buffer, so a truncated or corrupt archive throws instead of reading out of
// bounds.

#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolve {

inline void write_string_list(torch::serialize::OutputArchive& archive,
                              const std::string& key,
                              const std::vector<std::string>& values) {
    std::vector<int64_t> lengths;
    lengths.reserve(values.size());
    size_t total = 0;
    for (const auto& s : values) {
        lengths.push_back(static_cast<int64_t>(s.size()));
        total += s.size();
    }
    std::vector<uint8_t> bytes;
    bytes.reserve(total);
    for (const auto& s : values) bytes.insert(bytes.end(), s.begin(), s.end());

    archive.write(key + "_lengths",
                  lengths.empty() ? torch::empty({0}, torch::kInt64)
                                  : torch::tensor(lengths, torch::kInt64));
    archive.write(key + "_bytes",
                  bytes.empty() ? torch::empty({0}, torch::kUInt8)
                                : torch::from_blob(bytes.data(),
                                                   {static_cast<int64_t>(bytes.size())},
                                                   torch::kUInt8).clone());
}

// Reads the list stored under `key` into `out`. Returns false, leaving `out`
// untouched, when either entry is absent; throws when the entries are present
// but inconsistent.
inline bool try_read_string_list(torch::serialize::InputArchive& archive,
                                 const std::string& key,
                                 std::vector<std::string>& out) {
    torch::Tensor lengths_t;
    if (!archive.try_read(key + "_lengths", lengths_t)) return false;
    torch::Tensor bytes_t;
    if (!archive.try_read(key + "_bytes", bytes_t)) return false;

    if (lengths_t.dim() != 1 || bytes_t.dim() != 1) {
        throw std::runtime_error("string list '" + key + "' is corrupt: its lengths "
                                 "and bytes must be one-dimensional");
    }
    lengths_t = lengths_t.to(torch::kLong).contiguous();
    bytes_t = bytes_t.to(torch::kUInt8).contiguous();
    const auto lengths = lengths_t.accessor<int64_t, 1>();
    const auto* data = bytes_t.numel() > 0 ? bytes_t.data_ptr<uint8_t>() : nullptr;
    const int64_t n_bytes = bytes_t.numel();

    std::vector<std::string> values;
    values.reserve(static_cast<size_t>(lengths_t.size(0)));
    int64_t offset = 0;
    for (int64_t i = 0; i < lengths_t.size(0); ++i) {
        const int64_t len = lengths[i];
        if (len < 0 || len > n_bytes - offset) {
            throw std::runtime_error("string list '" + key + "' is corrupt: entry " +
                                     std::to_string(i) + " runs past the byte buffer");
        }
        if (len == 0) {
            values.emplace_back();
        } else {
            values.emplace_back(reinterpret_cast<const char*>(data + offset),
                                static_cast<size_t>(len));
        }
        offset += len;
    }
    out = std::move(values);
    return true;
}

// As try_read_string_list, for a list the archive must hold.
inline std::vector<std::string> read_string_list(torch::serialize::InputArchive& archive,
                                                 const std::string& key) {
    std::vector<std::string> values;
    if (!try_read_string_list(archive, key, values)) {
        throw std::runtime_error("archive holds no string list '" + key + "'");
    }
    return values;
}

}  // namespace resolve
