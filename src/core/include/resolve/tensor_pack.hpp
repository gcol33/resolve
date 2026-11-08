#pragma once

// A file of named tensors behind a JSON header, read tensor by tensor.
//
// Layout (little-endian):
//
//   12 bytes  magic "RESOLVEPACK\0"
//    4 bytes  format version (uint32, 1)
//    8 bytes  header length in bytes (uint64)
//    header   UTF-8 JSON: {"meta": <anything>, "tensors": [{"name", "dtype",
//             "shape", "offset", "bytes"}, ...]}
//    padding  to a multiple of 64 bytes from the start of the file
//    data     each tensor's bytes, C order, at its offset from the start of
//             the data region, each offset a multiple of 64
//
// dtype is one of "float16", "float32", "float64", "int16", "int32",
// "int64", "uint8". The terrain field pack and the terrain correction file
// are tensor packs; a reader loads the header once and then only the tensors
// it is asked for, so a pack of many tiles is opened without reading them all.

#include "resolve/json.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

inline constexpr char kTensorPackMagic[12] = {'R', 'E', 'S', 'O', 'L', 'V', 'E', 'P', 'A', 'C', 'K', '\0'};
inline constexpr uint32_t kTensorPackVersion = 1;

// Write `tensors` (moved to the CPU, made contiguous) under `meta`. Throws
// std::invalid_argument for a repeated name or an unsupported dtype and
// io::IOError when the file cannot be written.
void write_tensor_pack(const std::string& path, const json::Value& meta,
                       const std::vector<std::pair<std::string, torch::Tensor>>& tensors);

class TensorPackReader {
public:
    // Reads and checks the header. Throws io::IOError when the file cannot be
    // read and std::runtime_error when it is not a tensor pack or its index
    // does not fit the file.
    explicit TensorPackReader(std::string path);

    [[nodiscard]] const json::Value& meta() const noexcept { return meta_; }
    [[nodiscard]] bool contains(const std::string& name) const;
    // Every tensor name, in file order.
    [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }
    // One tensor, read from disk on each call (CPU). Thread-safe: each read
    // opens the file on its own, so reads from several threads run at once
    // (and a disk can reorder them). Throws
    // std::out_of_range for a name the pack does not hold.
    [[nodiscard]] torch::Tensor read(const std::string& name) const;
    // Bytes [first, first + count) of tensor `name`'s data into `out`, so a
    // caller can read part of a large tensor. Thread-safe. Throws
    // std::out_of_range for a name the pack does not hold or a range past the
    // tensor's end.
    void read_bytes(const std::string& name, uint64_t first, uint64_t count, void* out) const;
    // Several tensors, in the order named, read in file order: tensors less
    // than `merge_gap` bytes apart in the file come in one request. Thread-safe.
    // Throws std::out_of_range for a name the pack does not hold.
    [[nodiscard]] std::vector<torch::Tensor> read_many(const std::vector<std::string>& names,
                                                       uint64_t merge_gap) const;
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    void read_at(uint64_t offset, uint64_t count, void* out, const std::string& what) const;

    struct Entry {
        torch::ScalarType dtype;
        std::vector<int64_t> shape;
        uint64_t offset;
        uint64_t bytes;
    };
    std::string path_;
    json::Value meta_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, Entry> entries_;
    uint64_t data_start_ = 0;
};

}  // namespace resolve
