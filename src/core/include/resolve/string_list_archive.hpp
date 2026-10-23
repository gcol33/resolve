#pragma once

// Strings through a torch::serialize archive, which stores tensors only.
//
// A single string travels as "<key>_len", an int64 scalar, and "<key>", its
// UTF-8 bytes (absent for the empty string). A list travels as two entries:
// "<key>_lengths", an int64 vector of byte lengths, and "<key>_bytes", the
// concatenated bytes. Two entries per list rather than per string, so a
// 30k-species vocabulary stays two zip members.
//
// Every reader of a persisted string goes through try_read_string or
// try_read_string_list, which check the stored lengths against the byte
// buffer, so a truncated or corrupt archive throws instead of reading out of
// bounds. Both read through a CPU copy: an archive loaded onto a device holds
// its tensors there.

#include <torch/torch.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace resolve {

inline void write_string(torch::serialize::OutputArchive& archive, const std::string& key,
                         const std::string& value) {
    archive.write(key + "_len", torch::tensor(static_cast<int64_t>(value.size())));
    if (!value.empty()) {
        archive.write(key, torch::from_blob(const_cast<char*>(value.data()),
                                            {static_cast<int64_t>(value.size())},
                                            torch::kUInt8)
                               .clone());
    }
}

// Reads the string stored under `key` into `out`. Returns false, leaving `out`
// untouched, when the length entry is absent; an empty string reads as "".
// Throws when the entries are present but inconsistent.
inline bool try_read_string(torch::serialize::InputArchive& archive, const std::string& key,
                            std::string& out) {
    torch::Tensor len_t;
    if (!archive.try_read(key + "_len", len_t)) return false;
    if (len_t.numel() != 1) {
        throw std::runtime_error("string '" + key + "' is corrupt: its length is not a scalar");
    }
    const int64_t len = len_t.to(torch::kCPU).to(torch::kLong).item<int64_t>();
    if (len < 0) {
        throw std::runtime_error("string '" + key + "' is corrupt: negative length");
    }
    if (len == 0) {
        out.clear();
        return true;
    }
    torch::Tensor bytes_t;
    if (!archive.try_read(key, bytes_t)) {
        throw std::runtime_error("string '" + key + "' is corrupt: its bytes are missing");
    }
    bytes_t = bytes_t.to(torch::kCPU).to(torch::kUInt8).contiguous();
    if (bytes_t.dim() != 1 || bytes_t.numel() < len) {
        throw std::runtime_error("string '" + key + "' is corrupt: " + std::to_string(len) +
                                 " bytes recorded, " + std::to_string(bytes_t.numel()) +
                                 " stored");
    }
    out.assign(reinterpret_cast<const char*>(bytes_t.data_ptr<uint8_t>()),
               static_cast<size_t>(len));
    return true;
}

// As try_read_string, for a string the archive must hold.
inline std::string read_string(torch::serialize::InputArchive& archive, const std::string& key) {
    std::string value;
    if (!try_read_string(archive, key, value)) {
        throw std::runtime_error("archive holds no string '" + key + "'");
    }
    return value;
}

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
    // An archive loaded onto a device holds its tensors there; the accessors
    // below read host memory.
    lengths_t = lengths_t.to(torch::kCPU).to(torch::kLong).contiguous();
    bytes_t = bytes_t.to(torch::kCPU).to(torch::kUInt8).contiguous();
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
