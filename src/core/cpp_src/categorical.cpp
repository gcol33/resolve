// Implementation of CategoricalVocab + CategoricalEmbedder.
// See include/resolve/categorical.hpp for design notes.

#include "resolve/categorical.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace resolve {

// =============================================================================
// is_na_string
// =============================================================================
//
// Mirrors the Python POC's src/resolve/data/dataset.py::_NA_STRINGS, matched
// case-insensitively. Single source of truth for NA/missing-cell detection
// across the whole engine: the dataset target/covariate loaders and the
// categorical factorizer both call this, so the same raw cell is classified
// identically regardless of the column's role (previously the target path used
// a case-insensitive matcher and the categorical path an exact set, so e.g.
// "NAN" was missing in a target column but a real category in a covariate).
bool is_na_string(const std::string& s) noexcept {
    if (s.empty() || s == "." || s == "-") return true;
    auto eq_ci = [&](const char* tok) noexcept {
        const size_t n = std::strlen(tok);
        if (s.size() != n) return false;
        for (size_t i = 0; i < n; ++i) {
            if (std::tolower(static_cast<unsigned char>(s[i])) !=
                std::tolower(static_cast<unsigned char>(tok[i]))) return false;
        }
        return true;
    };
    return eq_ci("NA") || eq_ci("N/A") || eq_ci("NaN") ||
           eq_ci("NULL") || eq_ci("None");
}

// =============================================================================
// CategoricalVocab
// =============================================================================

void CategoricalVocab::fit_column(const std::string& column_name,
                                  const std::vector<std::string>& raw_values) {
    // Collect unique non-NA values. std::set gives sorted-unique in one pass,
    // mirroring Python's `sorted(set(...))`.
    std::set<std::string> uniques;
    for (const auto& v : raw_values) {
        if (!is_na_string(v)) {
            uniques.insert(v);
        }
    }

    // Assign codes 1..K (code 0 is implicit UNK).
    std::unordered_map<std::string, int64_t> mapping;
    mapping.reserve(uniques.size());
    int64_t next_code = 1;
    for (const auto& v : uniques) {
        mapping[v] = next_code++;
    }

    // Insert into vocab. If the column was already fit, overwrite (this is
    // refit semantics — we don't merge with existing codes).
    auto existing = maps_.find(column_name);
    if (existing == maps_.end()) {
        column_order_.push_back(column_name);
    }
    maps_[column_name] = std::move(mapping);
}

void CategoricalVocab::fit(
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<std::string>>& raw_values_per_column) {
    if (column_names.size() != raw_values_per_column.size()) {
        throw std::invalid_argument(
            "CategoricalVocab::fit: column_names and raw_values_per_column "
            "must have the same length");
    }
    for (size_t i = 0; i < column_names.size(); ++i) {
        fit_column(column_names[i], raw_values_per_column[i]);
    }
}

int64_t CategoricalVocab::encode(const std::string& column_name,
                                 const std::string& raw_value) const noexcept {
    if (is_na_string(raw_value)) {
        return 0;
    }
    auto col_it = maps_.find(column_name);
    if (col_it == maps_.end()) {
        return 0;
    }
    auto val_it = col_it->second.find(raw_value);
    if (val_it == col_it->second.end()) {
        return 0;
    }
    return val_it->second;
}

torch::Tensor CategoricalVocab::encode_batch(
    const std::vector<std::string>& column_names,
    const std::vector<std::vector<std::string>>& raw_values_per_column) const {
    if (column_names.size() != raw_values_per_column.size()) {
        throw std::invalid_argument(
            "CategoricalVocab::encode_batch: column_names and "
            "raw_values_per_column must have the same length");
    }
    const int64_t n_cols = static_cast<int64_t>(column_names.size());
    if (n_cols == 0) {
        // Match the (n_rows, 0) shape convention used elsewhere in the encoder.
        return torch::empty({0, 0}, torch::kInt64);
    }
    const int64_t n_rows = static_cast<int64_t>(raw_values_per_column[0].size());
    for (const auto& col : raw_values_per_column) {
        if (static_cast<int64_t>(col.size()) != n_rows) {
            throw std::invalid_argument(
                "CategoricalVocab::encode_batch: every inner vector must "
                "have the same length");
        }
    }

    auto tensor = torch::empty({n_rows, n_cols}, torch::kInt64);
    auto acc = tensor.accessor<int64_t, 2>();
    for (int64_t c = 0; c < n_cols; ++c) {
        const auto& col_name = column_names[c];
        const auto& col_values = raw_values_per_column[c];
        for (int64_t r = 0; r < n_rows; ++r) {
            acc[r][c] = encode(col_name, col_values[r]);
        }
    }
    return tensor;
}

int64_t CategoricalVocab::vocab_size(const std::string& column_name)
    const noexcept {
    auto it = maps_.find(column_name);
    if (it == maps_.end()) {
        return 1;  // Just the UNK slot.
    }
    return static_cast<int64_t>(it->second.size()) + 1;  // +1 for UNK.
}

bool CategoricalVocab::has_column(const std::string& column_name)
    const noexcept {
    return maps_.count(column_name) > 0;
}

std::vector<int64_t> CategoricalVocab::vocab_sizes() const {
    std::vector<int64_t> sizes;
    sizes.reserve(column_order_.size());
    for (const auto& name : column_order_) {
        sizes.push_back(vocab_size(name));
    }
    return sizes;
}

const std::unordered_map<std::string, int64_t>& CategoricalVocab::column_map(
    const std::string& column_name) const {
    auto it = maps_.find(column_name);
    if (it == maps_.end()) {
        throw std::out_of_range(
            "CategoricalVocab::column_map: column '" + column_name +
            "' has not been fit");
    }
    return it->second;
}

// =============================================================================
// CategoricalVocab serialization
// =============================================================================
//
// Layout under `prefix`:
//   {prefix}n_columns                      : int64 scalar tensor
//   {prefix}column_name_lengths            : int64[N]
//   {prefix}column_name_bytes              : uint8[sum(lengths)]
//   For each column i in 0..N-1:
//     {prefix}col_{i}_n_values             : int64 scalar
//     {prefix}col_{i}_value_lengths        : int64[K_i]
//     {prefix}col_{i}_value_bytes          : uint8[sum(value_lengths)]
//     {prefix}col_{i}_codes                : int64[K_i]  (assigned codes 1..K_i)
//
// Codes are written explicitly (not derived from sort order at load time)
// because future versions may relax the sort invariant; explicit codes make
// the format self-describing.

namespace {

torch::Tensor strings_to_lengths_tensor(const std::vector<std::string>& strs) {
    std::vector<int64_t> lengths;
    lengths.reserve(strs.size());
    for (const auto& s : strs) {
        lengths.push_back(static_cast<int64_t>(s.size()));
    }
    if (lengths.empty()) {
        return torch::empty({0}, torch::kInt64);
    }
    return torch::tensor(lengths, torch::kInt64);
}

torch::Tensor strings_to_bytes_tensor(const std::vector<std::string>& strs) {
    std::vector<uint8_t> bytes;
    size_t total = 0;
    for (const auto& s : strs) total += s.size();
    bytes.reserve(total);
    for (const auto& s : strs) {
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    if (bytes.empty()) {
        return torch::empty({0}, torch::kUInt8);
    }
    // from_blob view; clone to detach from local storage.
    return torch::from_blob(bytes.data(),
                            {static_cast<int64_t>(bytes.size())},
                            torch::kUInt8)
        .clone();
}

std::vector<std::string> split_bytes_by_lengths(
    const torch::Tensor& lengths_t, const torch::Tensor& bytes_t) {
    std::vector<std::string> out;
    if (lengths_t.numel() == 0) return out;

    auto lengths_acc = lengths_t.accessor<int64_t, 1>();
    auto bytes_ptr = bytes_t.numel() > 0 ? bytes_t.data_ptr<uint8_t>() : nullptr;
    int64_t offset = 0;
    out.reserve(lengths_t.size(0));
    for (int64_t i = 0; i < lengths_t.size(0); ++i) {
        const int64_t len = lengths_acc[i];
        if (len > 0 && bytes_ptr == nullptr) {
            throw std::runtime_error(
                "CategoricalVocab::load: byte tensor is empty but lengths "
                "report a non-zero string — corrupt archive");
        }
        out.emplace_back(reinterpret_cast<const char*>(bytes_ptr + offset),
                         static_cast<size_t>(len));
        offset += len;
    }
    return out;
}

}  // namespace

void CategoricalVocab::save(torch::serialize::OutputArchive& archive,
                            const std::string& prefix) const {
    const int64_t n_cols = static_cast<int64_t>(column_order_.size());
    archive.write(prefix + "n_columns", torch::tensor(n_cols, torch::kInt64));

    archive.write(prefix + "column_name_lengths",
                  strings_to_lengths_tensor(column_order_));
    archive.write(prefix + "column_name_bytes",
                  strings_to_bytes_tensor(column_order_));

    for (int64_t i = 0; i < n_cols; ++i) {
        const auto& col_name = column_order_[i];
        const auto& col_map = maps_.at(col_name);

        // Sort values by their assigned codes so the saved order is stable
        // (independent of the map's bucket order).
        std::vector<std::pair<int64_t, std::string>> by_code;
        by_code.reserve(col_map.size());
        for (const auto& [val, code] : col_map) {
            by_code.emplace_back(code, val);
        }
        std::sort(by_code.begin(), by_code.end());

        std::vector<std::string> values;
        std::vector<int64_t> codes;
        values.reserve(by_code.size());
        codes.reserve(by_code.size());
        for (const auto& [code, val] : by_code) {
            values.push_back(val);
            codes.push_back(code);
        }

        const std::string col_prefix = prefix + "col_" + std::to_string(i) + "_";
        archive.write(col_prefix + "n_values",
                      torch::tensor(static_cast<int64_t>(values.size()),
                                    torch::kInt64));
        archive.write(col_prefix + "value_lengths",
                      strings_to_lengths_tensor(values));
        archive.write(col_prefix + "value_bytes",
                      strings_to_bytes_tensor(values));
        archive.write(col_prefix + "codes",
                      codes.empty() ? torch::empty({0}, torch::kInt64)
                                    : torch::tensor(codes, torch::kInt64));
    }
}

CategoricalVocab CategoricalVocab::load(torch::serialize::InputArchive& archive,
                                        const std::string& prefix) {
    CategoricalVocab vocab;

    // Back-compat: if the archive predates categorical support, the key
    // won't exist. Treat that as an empty vocab.
    torch::Tensor n_cols_t;
    try {
        archive.read(prefix + "n_columns", n_cols_t);
    } catch (const std::exception&) {
        return vocab;
    }
    const int64_t n_cols = n_cols_t.item<int64_t>();
    if (n_cols == 0) {
        return vocab;
    }

    torch::Tensor name_lengths, name_bytes;
    archive.read(prefix + "column_name_lengths", name_lengths);
    archive.read(prefix + "column_name_bytes", name_bytes);
    auto column_names = split_bytes_by_lengths(name_lengths, name_bytes);
    if (static_cast<int64_t>(column_names.size()) != n_cols) {
        throw std::runtime_error(
            "CategoricalVocab::load: column_name count mismatch with "
            "n_columns header (corrupt archive)");
    }

    for (int64_t i = 0; i < n_cols; ++i) {
        const std::string col_prefix = prefix + "col_" + std::to_string(i) + "_";

        torch::Tensor n_values_t, value_lengths, value_bytes, codes_t;
        archive.read(col_prefix + "n_values", n_values_t);
        archive.read(col_prefix + "value_lengths", value_lengths);
        archive.read(col_prefix + "value_bytes", value_bytes);
        archive.read(col_prefix + "codes", codes_t);

        auto values = split_bytes_by_lengths(value_lengths, value_bytes);
        const int64_t n_values = n_values_t.item<int64_t>();
        if (static_cast<int64_t>(values.size()) != n_values) {
            throw std::runtime_error(
                "CategoricalVocab::load: value count mismatch for column '"
                + column_names[i] + "' (corrupt archive)");
        }
        if (codes_t.numel() != n_values) {
            throw std::runtime_error(
                "CategoricalVocab::load: code count mismatch for column '"
                + column_names[i] + "' (corrupt archive)");
        }

        std::unordered_map<std::string, int64_t> col_map;
        col_map.reserve(static_cast<size_t>(n_values));
        if (n_values > 0) {
            auto codes_acc = codes_t.accessor<int64_t, 1>();
            for (int64_t j = 0; j < n_values; ++j) {
                col_map[values[j]] = codes_acc[j];
            }
        }
        vocab.column_order_.push_back(column_names[i]);
        vocab.maps_[column_names[i]] = std::move(col_map);
    }
    return vocab;
}

// =============================================================================
// CategoricalEmbedderImpl
// =============================================================================

CategoricalEmbedderImpl::CategoricalEmbedderImpl(
    const std::vector<int64_t>& vocab_sizes, int64_t embed_dim)
    : embed_dim_(embed_dim),
      output_dim_(static_cast<int64_t>(vocab_sizes.size()) * embed_dim) {
    if (embed_dim <= 0) {
        throw std::invalid_argument(
            "CategoricalEmbedder: embed_dim must be > 0 (got " +
            std::to_string(embed_dim) + ")");
    }
    tables_.reserve(vocab_sizes.size());
    for (size_t i = 0; i < vocab_sizes.size(); ++i) {
        const int64_t vsize = vocab_sizes[i];
        if (vsize < 1) {
            throw std::invalid_argument(
                "CategoricalEmbedder: vocab_size must be >= 1 (got " +
                std::to_string(vsize) + " for column " + std::to_string(i) +
                "). Code 0 is reserved for UNK so size must include that "
                "slot.");
        }
        auto opts = torch::nn::EmbeddingOptions(vsize, embed_dim);
        auto emb = torch::nn::Embedding(opts);
        // Match libtorch's default init for nn::Embedding (N(0, 1)). Scaled
        // down lightly for stable training in concat-with-continuous-features.
        torch::nn::init::normal_(emb->weight, /*mean=*/0.0, /*std=*/0.02);
        tables_.push_back(emb);
        register_module("cat_" + std::to_string(i), emb);
    }
}

torch::Tensor CategoricalEmbedderImpl::forward(torch::Tensor ids) {
    if (tables_.empty()) {
        // No categoricals — return (B, 0). Use ids if defined for batch size,
        // else default to a 0x0 CPU tensor (caller is responsible for not
        // trying to concat with mismatched batch).
        if (ids.defined() && ids.dim() >= 1) {
            return torch::empty({ids.size(0), 0},
                                torch::TensorOptions().dtype(torch::kFloat32)
                                    .device(ids.device()));
        }
        return torch::empty({0, 0}, torch::kFloat32);
    }
    if (!ids.defined()) {
        throw std::invalid_argument(
            "CategoricalEmbedder::forward: ids tensor is undefined but "
            "the embedder holds " +
            std::to_string(tables_.size()) +
            " categorical column(s); pass an (B, N) int64 tensor.");
    }
    if (ids.dim() != 2) {
        throw std::invalid_argument(
            "CategoricalEmbedder::forward: ids must be 2-D (got dim=" +
            std::to_string(ids.dim()) + ")");
    }
    const int64_t expected_cols = static_cast<int64_t>(tables_.size());
    if (ids.size(1) != expected_cols) {
        throw std::invalid_argument(
            "CategoricalEmbedder::forward: ids has " +
            std::to_string(ids.size(1)) +
            " columns but embedder was constructed for " +
            std::to_string(expected_cols) + " columns");
    }
    if (ids.scalar_type() != torch::kInt64) {
        ids = ids.to(torch::kInt64);
    }

    // Per-column lookups then concat along the feature axis.
    // tables_[c]->forward(ids.select(1, c)) returns (B, embed_dim).
    std::vector<torch::Tensor> parts;
    parts.reserve(tables_.size());
    for (int64_t c = 0; c < expected_cols; ++c) {
        parts.push_back(tables_[c]->forward(ids.select(1, c)));
    }
    return torch::cat(parts, /*dim=*/1);
}

torch::Tensor CategoricalEmbedderImpl::get_table_weights(int64_t column_idx)
    const {
    if (column_idx < 0 ||
        column_idx >= static_cast<int64_t>(tables_.size())) {
        throw std::out_of_range(
            "CategoricalEmbedder::get_table_weights: column_idx " +
            std::to_string(column_idx) + " out of range [0, " +
            std::to_string(tables_.size()) + ")");
    }
    return tables_[column_idx]->weight.detach().clone();
}

}  // namespace resolve
