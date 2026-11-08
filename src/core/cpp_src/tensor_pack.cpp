#include "resolve/tensor_pack.hpp"

#include "resolve/io_retry.hpp"

#include <array>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>

namespace resolve {

namespace {

constexpr uint64_t kAlign = 64;

uint64_t aligned(uint64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

const std::array<std::pair<const char*, torch::ScalarType>, 7>& dtype_names() {
    static const std::array<std::pair<const char*, torch::ScalarType>, 7> names = {{
        {"float16", torch::kFloat16},
        {"float32", torch::kFloat32},
        {"float64", torch::kFloat64},
        {"int16", torch::kInt16},
        {"int32", torch::kInt32},
        {"int64", torch::kInt64},
        {"uint8", torch::kUInt8},
    }};
    return names;
}

std::string dtype_name(torch::ScalarType t) {
    for (const auto& [name, type] : dtype_names()) {
        if (type == t) return name;
    }
    throw std::invalid_argument(std::string("tensor pack: unsupported dtype ") + c10::toString(t));
}

torch::ScalarType dtype_from(const std::string& name) {
    for (const auto& [n, type] : dtype_names()) {
        if (name == n) return type;
    }
    throw std::runtime_error("tensor pack: unknown dtype '" + name + "'");
}

void put_le(std::string& out, uint64_t v, int bytes) {
    for (int k = 0; k < bytes; ++k) out.push_back(static_cast<char>((v >> (8 * k)) & 0xff));
}

uint64_t get_le(const char* p, int bytes) {
    uint64_t v = 0;
    for (int k = 0; k < bytes; ++k) v |= static_cast<uint64_t>(static_cast<unsigned char>(p[k])) << (8 * k);
    return v;
}

}  // namespace

void write_tensor_pack(const std::string& path, const json::Value& meta,
                       const std::vector<std::pair<std::string, torch::Tensor>>& tensors) {
    std::vector<torch::Tensor> data;
    json::Value index = json::Value::array();
    std::unordered_set<std::string> seen;
    uint64_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
        if (!seen.insert(name).second) {
            throw std::invalid_argument("write_tensor_pack: tensor '" + name + "' named twice");
        }
        torch::Tensor t = tensor.detach().to(torch::kCPU).contiguous();
        const uint64_t bytes = static_cast<uint64_t>(t.numel()) * t.element_size();
        json::Value e = json::Value::object();
        e.set("name", name);
        e.set("dtype", dtype_name(t.scalar_type()));
        json::Value shape = json::Value::array();
        for (const int64_t d : t.sizes()) shape.push_back(d);
        e.set("shape", std::move(shape));
        e.set("offset", static_cast<double>(offset));
        e.set("bytes", static_cast<double>(bytes));
        index.push_back(std::move(e));
        offset = aligned(offset + bytes);
        data.push_back(std::move(t));
    }
    json::Value header = json::Value::object();
    header.set("meta", meta);
    header.set("tensors", std::move(index));
    const std::string text = json::dump(header, 0);

    std::string head(kTensorPackMagic, sizeof(kTensorPackMagic));
    put_le(head, kTensorPackVersion, 4);
    put_le(head, text.size(), 8);
    head += text;
    head.append(aligned(head.size()) - head.size(), '\0');

    const std::filesystem::path target(path);
    const std::filesystem::path tmp = target.string() + ".part";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw io::IOError("cannot write '" + tmp.string() + "'");
        out.write(head.data(), static_cast<std::streamsize>(head.size()));
        uint64_t written = 0;
        static const char zeros[kAlign] = {};
        for (const auto& t : data) {
            const uint64_t bytes = static_cast<uint64_t>(t.numel()) * t.element_size();
            out.write(static_cast<const char*>(t.data_ptr()), static_cast<std::streamsize>(bytes));
            written += bytes;
            const uint64_t pad = aligned(written) - written;
            out.write(zeros, static_cast<std::streamsize>(pad));
            written += pad;
        }
        if (!out) throw io::IOError("cannot write '" + tmp.string() + "'");
    }
    std::error_code ec;
    std::filesystem::rename(tmp, target, ec);
    if (ec) throw io::IOError("cannot move '" + tmp.string() + "' to '" + path + "': " + ec.message());
}

TensorPackReader::TensorPackReader(std::string path) : path_(std::move(path)) {
    std::ifstream in(path_, std::ios::binary);
    if (!in) throw io::IOError("cannot open '" + path_ + "'");
    char fixed[24];
    in.read(fixed, sizeof(fixed));
    if (!in || std::memcmp(fixed, kTensorPackMagic, sizeof(kTensorPackMagic)) != 0) {
        throw std::runtime_error("'" + path_ + "' is not a resolve tensor pack");
    }
    const auto version = static_cast<uint32_t>(get_le(fixed + 12, 4));
    if (version != kTensorPackVersion) {
        throw std::runtime_error("'" + path_ + "' is tensor pack version " + std::to_string(version) +
                                 ", this engine reads version " + std::to_string(kTensorPackVersion));
    }
    const uint64_t header_len = get_le(fixed + 16, 8);
    const uint64_t file_size = std::filesystem::file_size(path_);
    if (24 + header_len > file_size) throw std::runtime_error("'" + path_ + "' is truncated");
    std::string text(header_len, '\0');
    in.read(text.data(), static_cast<std::streamsize>(header_len));
    if (!in) throw io::IOError("cannot read '" + path_ + "'");
    const json::Value header = json::parse(text);
    meta_ = header.at("meta");
    data_start_ = aligned(24 + header_len);
    for (const auto& e : header.at("tensors").items()) {
        Entry entry;
        entry.dtype = dtype_from(e.at("dtype").as_string());
        for (const auto& d : e.at("shape").items()) entry.shape.push_back(d.as_int());
        entry.offset = static_cast<uint64_t>(e.at("offset").as_int());
        entry.bytes = static_cast<uint64_t>(e.at("bytes").as_int());
        int64_t numel = 1;
        for (const int64_t d : entry.shape) numel *= d;
        if (entry.bytes != static_cast<uint64_t>(numel) * c10::elementSize(entry.dtype) ||
            data_start_ + entry.offset + entry.bytes > file_size) {
            throw std::runtime_error("'" + path_ + "': tensor '" + e.at("name").as_string() +
                                     "' does not fit its index entry or the file");
        }
        const std::string name = e.at("name").as_string();
        if (!entries_.emplace(name, std::move(entry)).second) {
            throw std::runtime_error("'" + path_ + "' names tensor '" + name + "' twice");
        }
        names_.push_back(name);
    }
}

bool TensorPackReader::contains(const std::string& name) const { return entries_.count(name) != 0; }

void TensorPackReader::read_at(uint64_t offset, uint64_t count, void* out, const std::string& what) const {
    io::with_retry<io::IOError>([&] {
        std::ifstream in(path_, std::ios::binary);
        if (!in) throw io::IOError("cannot open '" + path_ + "'");
        in.seekg(static_cast<std::streamoff>(data_start_ + offset));
        in.read(static_cast<char*>(out), static_cast<std::streamsize>(count));
        if (!in) throw io::IOError("cannot read " + what + " from '" + path_ + "'");
        return 0;
    }, "tensor pack read");
}

torch::Tensor TensorPackReader::read(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) throw std::out_of_range("'" + path_ + "' holds no tensor '" + name + "'");
    const Entry& e = it->second;
    torch::Tensor out = torch::empty(e.shape, torch::TensorOptions().dtype(e.dtype));
    read_at(e.offset, e.bytes, out.data_ptr(), "tensor '" + name + "'");
    return out;
}

std::vector<torch::Tensor> TensorPackReader::read_many(const std::vector<std::string>& names,
                                                       uint64_t merge_gap) const {
    std::vector<std::pair<size_t, const Entry*>> items;
    items.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        const auto it = entries_.find(names[i]);
        if (it == entries_.end()) throw std::out_of_range("'" + path_ + "' holds no tensor '" + names[i] + "'");
        items.emplace_back(i, &it->second);
    }
    std::sort(items.begin(), items.end(),
              [](const auto& x, const auto& y) { return x.second->offset < y.second->offset; });
    std::vector<torch::Tensor> out(names.size());
    std::vector<uint8_t> buffer;
    for (size_t i = 0; i < items.size();) {
        const uint64_t begin = items[i].second->offset;
        uint64_t end = begin + items[i].second->bytes;
        size_t last = i;
        while (last + 1 < items.size() && items[last + 1].second->offset < end + merge_gap) {
            ++last;
            end = std::max(end, items[last].second->offset + items[last].second->bytes);
        }
        buffer.resize(static_cast<size_t>(end - begin));
        if (end > begin) read_at(begin, end - begin, buffer.data(), "tensors");
        for (size_t k = i; k <= last; ++k) {
            const Entry& e = *items[k].second;
            torch::Tensor t = torch::empty(e.shape, torch::TensorOptions().dtype(e.dtype));
            if (e.bytes > 0) std::memcpy(t.data_ptr(), buffer.data() + (e.offset - begin), e.bytes);
            out[items[k].first] = std::move(t);
        }
        i = last + 1;
    }
    return out;
}

void TensorPackReader::read_bytes(const std::string& name, uint64_t first, uint64_t count, void* out) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) throw std::out_of_range("'" + path_ + "' holds no tensor '" + name + "'");
    const Entry& e = it->second;
    if (first > e.bytes || count > e.bytes - first) {
        throw std::out_of_range("'" + path_ + "': bytes " + std::to_string(first) + " + " + std::to_string(count) +
                                " pass the end of tensor '" + name + "'");
    }
    if (count == 0) return;
    read_at(e.offset + first, count, out, "bytes of tensor '" + name + "'");
}

}  // namespace resolve
