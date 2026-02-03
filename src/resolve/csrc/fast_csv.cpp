/**
 * Fast CSV loader for PyTorch using memory-mapped files.
 *
 * Compiles via torch.utils.cpp_extension - no manual build needed.
 * Uses memory mapping for zero-copy reading of large files.
 */

#include <torch/extension.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <chrono>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

// MurmurHash3 finalizer for string hashing
inline int64_t murmur_hash_string(const std::string& s) {
    uint64_t h = 0;
    for (char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
    }
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return static_cast<int64_t>(h);
}

// Memory-mapped file wrapper
class MappedFile {
public:
    MappedFile(const std::string& path) : data_(nullptr), size_(0) {
#ifdef _WIN32
        file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        LARGE_INTEGER file_size;
        GetFileSizeEx(file_handle_, &file_size);
        size_ = static_cast<size_t>(file_size.QuadPart);

        mapping_handle_ = CreateFileMappingA(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle_ == nullptr) {
            CloseHandle(file_handle_);
            throw std::runtime_error("Cannot create file mapping: " + path);
        }

        data_ = static_cast<const char*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            throw std::runtime_error("Cannot map file: " + path);
        }
#else
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        struct stat st;
        fstat(fd, &st);
        size_ = st.st_size;

        data_ = static_cast<const char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);

        if (data_ == MAP_FAILED) {
            throw std::runtime_error("Cannot map file: " + path);
        }
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_) UnmapViewOfFile(data_);
        if (mapping_handle_) CloseHandle(mapping_handle_);
        if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
#else
        if (data_ && data_ != MAP_FAILED) munmap(const_cast<char*>(data_), size_);
#endif
    }

    const char* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const char* data_;
    size_t size_;
#ifdef _WIN32
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle_ = nullptr;
#endif
};

// Skip UTF-8 BOM if present (EF BB BF)
// Returns offset to start reading from (0 or 3)
inline size_t skip_bom(const char* data, size_t size) {
    if (size >= 3 &&
        static_cast<unsigned char>(data[0]) == 0xEF &&
        static_cast<unsigned char>(data[1]) == 0xBB &&
        static_cast<unsigned char>(data[2]) == 0xBF) {
        return 3;
    }
    return 0;
}

// Check if a string represents NA/missing value (readr-compatible)
// Handles: "", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "None", ".", "-"
inline bool is_na_string(const std::string& s) {
    if (s.empty()) return true;
    if (s.size() == 1) {
        return s[0] == '.' || s[0] == '-';
    }
    if (s.size() == 2) {
        return (s == "NA" || s == "na");
    }
    if (s.size() == 3) {
        return (s == "N/A" || s == "n/a" || s == "NaN" || s == "nan");
    }
    if (s.size() == 4) {
        return (s == "NULL" || s == "null" || s == "None" || s == "none");
    }
    return false;
}

// Parse float with NA detection - returns NaN for NA values
inline float parse_float_or_nan(const std::string& s) {
    if (is_na_string(s)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    try {
        return std::stof(s);
    } catch (...) {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

// Find line boundaries in a chunk of data
std::vector<size_t> find_line_starts(const char* data, size_t start, size_t end) {
    std::vector<size_t> starts;
    starts.push_back(start);
    for (size_t i = start; i < end; ++i) {
        if (data[i] == '\n' && i + 1 < end) {
            starts.push_back(i + 1);
        }
    }
    return starts;
}

// Fast field parsing - uses memchr for delimiter scanning
// Returns string_view when no quotes, only allocates when needed
// If hit_newline is provided, sets it to true if the field ended at a newline
std::string parse_field_fast(const char* data, size_t& pos, size_t end, char delim = ',', bool* hit_newline = nullptr) {
    const size_t start = pos;
    if (hit_newline) *hit_newline = false;

    // Quick path: no quotes - just scan for delimiter
    if (pos < end && data[pos] != '"') {
        // Use memchr for fast scanning
        const char* field_start = data + pos;
        const char* scan_end = data + end;

        // Scan until we hit delimiter, newline, or end
        while (pos < end) {
            char c = data[pos];
            if (c == delim || c == '\n' || c == '\r') break;
            pos++;
        }

        std::string result(field_start, data + pos);

        // Skip delimiter/newline and track what we hit
        if (pos < end && data[pos] == delim) {
            pos++;
        } else {
            if (hit_newline) *hit_newline = true;
            if (pos < end && data[pos] == '\r') pos++;
            if (pos < end && data[pos] == '\n') pos++;
        }

        return result;
    }

    // Slow path: quoted field
    std::string result;
    result.reserve(64);  // Pre-allocate for typical field size
    bool in_quotes = false;

    while (pos < end) {
        char c = data[pos];

        if (c == '"') {
            if (in_quotes && pos + 1 < end && data[pos + 1] == '"') {
                result += '"';
                pos += 2;
                continue;
            }
            in_quotes = !in_quotes;
            pos++;
            continue;
        }

        if (!in_quotes && (c == delim || c == '\n' || c == '\r')) {
            break;
        }

        result += c;
        pos++;
    }

    // Skip delimiter and track newline
    if (pos < end && data[pos] == delim) {
        pos++;
    } else {
        if (hit_newline) *hit_newline = true;
        if (pos < end && data[pos] == '\r') pos++;
        if (pos < end && data[pos] == '\n') pos++;
    }

    return result;
}

// Parse header row
std::vector<std::string> parse_header(const char* data, size_t size) {
    std::vector<std::string> columns;
    size_t pos = 0;
    bool hit_newline = false;

    while (pos < size && !hit_newline) {
        std::string field = parse_field_fast(data, pos, size, ',', &hit_newline);
        columns.push_back(std::move(field));
    }

    return columns;
}

} // anonymous namespace

/**
 * Load grouped CSV to mixed Python dict (numeric tensors + string lists).
 *
 * Generic version: accepts arbitrary columns to load.
 * Memory-optimized: uses hash-based group ID mapping, memory-mapped file reading.
 *
 * Args:
 *   path: Path to CSV file
 *   group_id_col: Column containing group IDs (e.g., plot IDs, patient IDs, session IDs)
 *   numeric_cols: Numeric columns to load as float32 tensors
 *   string_cols: String columns to load (hashed to int64 if hash_string_cols=true)
 *   hash_string_cols: If true, hash strings to int64; if false, return as Python lists
 *   verbose: Print progress
 *
 * Returns dict with:
 *   - group_indices: int64 tensor of group indices (0-based)
 *   - group_offsets: int64 tensor of CSR offsets for batch access
 *   - numeric columns: float32 tensors
 *   - string columns: int64 tensors (if hashed) or Python lists
 *   - "_n_records": total record count
 *   - "_n_groups": unique group count
 */
py::dict load_grouped_csv(
    const std::string& path,
    const std::string& group_id_col,
    const std::vector<std::string>& numeric_cols,
    const std::vector<std::string>& string_cols,
    bool hash_string_cols,
    bool verbose
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "  Loading: " << path << std::flush;
    }

    // Memory map the file
    MappedFile file(path);
    const char* data = file.data();
    size_t size = file.size();

    // Skip UTF-8 BOM if present
    size_t bom_offset = skip_bom(data, size);
    data += bom_offset;
    size -= bom_offset;

    if (verbose) {
        std::cout << " (" << (size / 1024 / 1024) << " MB)..." << std::flush;
    }

    // Parse header
    auto columns = parse_header(data, size);

    // Find column indices
    std::unordered_map<std::string, int> col_indices;
    int group_idx = -1;
    for (size_t i = 0; i < columns.size(); ++i) {
        col_indices[columns[i]] = i;
        if (columns[i] == group_id_col) group_idx = i;
    }

    if (group_idx < 0) throw std::runtime_error("Group ID column not found: " + group_id_col);

    // Find numeric column indices
    std::vector<int> numeric_indices;
    std::vector<std::string> found_numeric;
    for (const auto& col : numeric_cols) {
        auto it = col_indices.find(col);
        if (it != col_indices.end()) {
            numeric_indices.push_back(it->second);
            found_numeric.push_back(col);
        }
    }

    // Find string column indices
    std::vector<int> string_indices;
    std::vector<std::string> found_string;
    for (const auto& col : string_cols) {
        auto it = col_indices.find(col);
        if (it != col_indices.end()) {
            string_indices.push_back(it->second);
            found_string.push_back(col);
        }
    }

    // Find start of data (after header)
    size_t data_start = 0;
    while (data_start < size && data[data_start] != '\n') data_start++;
    if (data_start < size) data_start++;

    // Use hash-based group ID mapping (int64 hash -> int64 index)
    std::unordered_map<int64_t, int64_t> group_hash_to_idx;
    group_hash_to_idx.reserve(2000000);  // Pre-allocate for ~2M groups

    // Count lines for pre-allocation
    size_t n_lines = 0;
    for (size_t i = data_start; i < size; ++i) {
        if (data[i] == '\n') n_lines++;
    }
    if (size > 0 && data[size-1] != '\n') n_lines++;

    if (verbose) {
        std::cout << " (~" << n_lines/1000000 << "M lines)..." << std::flush;
    }

    // Allocate tensors
    auto options_i64 = torch::TensorOptions().dtype(torch::kInt64);
    auto options_f32 = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor group_indices_t = torch::empty({static_cast<int64_t>(n_lines)}, options_i64);
    int64_t* group_indices_ptr = group_indices_t.data_ptr<int64_t>();

    // Pre-allocate numeric column storage
    std::vector<std::vector<float>> numeric_data(found_numeric.size());
    for (auto& v : numeric_data) v.reserve(n_lines);

    // Pre-allocate string column storage (either as hashes or strings)
    std::vector<std::vector<int64_t>> string_hashes(found_string.size());
    std::vector<std::vector<std::string>> string_values(found_string.size());
    if (hash_string_cols) {
        for (auto& v : string_hashes) v.reserve(n_lines);
    } else {
        for (auto& v : string_values) v.reserve(n_lines);
    }

    // Parse data rows
    size_t pos = data_start;
    int64_t n_file_cols = static_cast<int64_t>(columns.size());
    int64_t record_idx = 0;
    int64_t next_group_idx = 0;

    while (pos < size) {
        // Skip empty lines
        if (data[pos] == '\n' || data[pos] == '\r') {
            pos++;
            continue;
        }

        // Parse row
        int64_t group_hash = 0;
        bool has_group = false;
        bool end_of_line = false;

        std::vector<float> row_numeric(found_numeric.size(), std::numeric_limits<float>::quiet_NaN());
        std::vector<int64_t> row_string_hashes(found_string.size(), 0);
        std::vector<std::string> row_string_values(found_string.size());

        for (int64_t col = 0; col < n_file_cols && pos < size && !end_of_line; ++col) {
            // Check if this is the group ID column
            if (col == group_idx) {
                // Parse and hash group ID directly
                const size_t field_start = pos;
                while (pos < size && data[pos] != ',' && data[pos] != '\n' && data[pos] != '\r') pos++;

                // Hash the field in-place
                uint64_t h = 0;
                for (size_t i = field_start; i < pos; ++i) {
                    h ^= static_cast<uint64_t>(data[i]);
                    h *= 0xff51afd7ed558ccdULL;
                    h ^= h >> 33;
                }
                h *= 0xc4ceb9fe1a85ec53ULL;
                h ^= h >> 33;
                group_hash = static_cast<int64_t>(h);
                has_group = (pos > field_start);

                // Skip delimiter
                if (pos < size && data[pos] == ',') {
                    pos++;
                } else {
                    end_of_line = true;
                    if (pos < size && data[pos] == '\r') pos++;
                    if (pos < size && data[pos] == '\n') pos++;
                }
                continue;
            }

            // Check if this is a numeric column
            bool is_numeric = false;
            for (size_t ni = 0; ni < numeric_indices.size(); ++ni) {
                if (col == numeric_indices[ni]) {
                    is_numeric = true;
                    const size_t field_start = pos;
                    while (pos < size && data[pos] != ',' && data[pos] != '\n' && data[pos] != '\r') pos++;

                    if (pos > field_start) {
                        char* end;
                        float val = std::strtof(data + field_start, &end);
                        if (end != data + field_start) {
                            row_numeric[ni] = val;
                        }
                    }

                    if (pos < size && data[pos] == ',') {
                        pos++;
                    } else {
                        end_of_line = true;
                        if (pos < size && data[pos] == '\r') pos++;
                        if (pos < size && data[pos] == '\n') pos++;
                    }
                    break;
                }
            }
            if (is_numeric) continue;

            // Check if this is a string column
            bool is_string = false;
            for (size_t si = 0; si < string_indices.size(); ++si) {
                if (col == string_indices[si]) {
                    is_string = true;
                    const size_t field_start = pos;
                    while (pos < size && data[pos] != ',' && data[pos] != '\n' && data[pos] != '\r') pos++;

                    if (hash_string_cols) {
                        // Hash the string in-place
                        uint64_t h = 0;
                        for (size_t i = field_start; i < pos; ++i) {
                            h ^= static_cast<uint64_t>(data[i]);
                            h *= 0xff51afd7ed558ccdULL;
                            h ^= h >> 33;
                        }
                        h *= 0xc4ceb9fe1a85ec53ULL;
                        h ^= h >> 33;
                        row_string_hashes[si] = static_cast<int64_t>(h);
                    } else {
                        row_string_values[si] = std::string(data + field_start, pos - field_start);
                    }

                    if (pos < size && data[pos] == ',') {
                        pos++;
                    } else {
                        end_of_line = true;
                        if (pos < size && data[pos] == '\r') pos++;
                        if (pos < size && data[pos] == '\n') pos++;
                    }
                    break;
                }
            }
            if (is_string) continue;

            // Skip field we don't need
            while (pos < size && data[pos] != ',' && data[pos] != '\n' && data[pos] != '\r') pos++;
            if (pos < size && data[pos] == ',') {
                pos++;
            } else {
                end_of_line = true;
                if (pos < size && data[pos] == '\r') pos++;
                if (pos < size && data[pos] == '\n') pos++;
            }
        }

        // Skip if missing group ID
        if (!has_group) continue;

        // Map group hash to index
        auto it = group_hash_to_idx.find(group_hash);
        int64_t group_index;
        if (it == group_hash_to_idx.end()) {
            group_index = next_group_idx++;
            group_hash_to_idx[group_hash] = group_index;
        } else {
            group_index = it->second;
        }

        // Store data
        group_indices_ptr[record_idx] = group_index;

        for (size_t ni = 0; ni < found_numeric.size(); ++ni) {
            numeric_data[ni].push_back(row_numeric[ni]);
        }
        if (hash_string_cols) {
            for (size_t si = 0; si < found_string.size(); ++si) {
                string_hashes[si].push_back(row_string_hashes[si]);
            }
        } else {
            for (size_t si = 0; si < found_string.size(); ++si) {
                string_values[si].push_back(std::move(row_string_values[si]));
            }
        }

        record_idx++;
    }

    // Trim group_indices tensor
    int64_t n_records = record_idx;
    int64_t n_groups = next_group_idx;
    group_indices_t = group_indices_t.slice(0, 0, n_records).contiguous();

    // Build CSR-style offsets for efficient batch access
    torch::Tensor offsets_t = torch::zeros({n_groups + 1}, options_i64);
    int64_t* offsets_ptr = offsets_t.data_ptr<int64_t>();

    // Get trimmed pointer
    group_indices_ptr = group_indices_t.data_ptr<int64_t>();

    // Count records per group
    for (int64_t i = 0; i < n_records; ++i) {
        offsets_ptr[group_indices_ptr[i] + 1]++;
    }
    // Cumulative sum
    for (int64_t i = 1; i <= n_groups; ++i) {
        offsets_ptr[i] += offsets_ptr[i - 1];
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (verbose) {
        std::cout << " done (" << n_records << " records, "
                  << n_groups << " groups, "
                  << found_numeric.size() << " numeric + "
                  << found_string.size() << " string cols, "
                  << duration.count() / 1000.0 << "s)" << std::endl;
    }

    // Build result dict
    py::dict result;

    // Always include group indices and offsets
    result["group_indices"] = group_indices_t;
    result["group_offsets"] = offsets_t;

    // Add numeric columns as tensors
    for (size_t i = 0; i < found_numeric.size(); ++i) {
        result[py::cast(found_numeric[i])] = torch::from_blob(
            numeric_data[i].data(), {static_cast<int64_t>(numeric_data[i].size())}, options_f32
        ).clone();
    }

    // Add string columns (as hashes or strings)
    if (hash_string_cols) {
        for (size_t i = 0; i < found_string.size(); ++i) {
            result[py::cast(found_string[i])] = torch::from_blob(
                string_hashes[i].data(), {static_cast<int64_t>(string_hashes[i].size())}, options_i64
            ).clone();
        }
    } else {
        for (size_t i = 0; i < found_string.size(); ++i) {
            result[py::cast(found_string[i])] = py::cast(string_values[i]);
        }
    }

    // Add metadata
    result["_n_records"] = n_records;
    result["_n_groups"] = n_groups;

    return result;
}

/**
 * Legacy wrapper for backwards compatibility.
 * Loads species CSV with hardcoded columns (plot_id, species_id, weight).
 */
std::unordered_map<std::string, torch::Tensor> load_species_csv(
    const std::string& path,
    const std::string& plot_id_col,
    const std::string& species_id_col,
    const std::string& weight_col,
    bool verbose
) {
    // Call generic version with species_id as string col (hashed) and weight as numeric
    py::dict generic_result = load_grouped_csv(
        path,
        plot_id_col,
        {weight_col},        // numeric_cols
        {species_id_col},    // string_cols (will be hashed)
        true,                // hash_string_cols
        verbose
    );

    // Convert to legacy format (keep "plot_" naming for backwards compat)
    std::unordered_map<std::string, torch::Tensor> result;
    result["plot_indices"] = generic_result["group_indices"].cast<torch::Tensor>();
    result["plot_offsets"] = generic_result["group_offsets"].cast<torch::Tensor>();
    result["species_ids"] = generic_result[py::cast(species_id_col)].cast<torch::Tensor>();
    result["weights"] = generic_result[py::cast(weight_col)].cast<torch::Tensor>();

    return result;
}

/**
 * Load header CSV to mixed Python dict (numeric tensors + string lists).
 *
 * Returns dict with:
 *   - numeric columns: float32 tensors
 *   - string columns: Python lists of strings
 *   - "_n_rows": int64 tensor with row count
 */
py::dict load_header_csv_full(
    const std::string& path,
    const std::vector<std::string>& numeric_cols,
    const std::vector<std::string>& string_cols,
    bool verbose
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "  Loading: " << path << std::flush;
    }

    // Memory map the file
    MappedFile file(path);
    const char* data = file.data();
    size_t size = file.size();

    // Skip UTF-8 BOM if present
    size_t bom_offset = skip_bom(data, size);
    data += bom_offset;
    size -= bom_offset;

    if (verbose) {
        std::cout << " (" << (size / 1024 / 1024) << " MB)..." << std::flush;
    }

    // Parse header
    auto columns = parse_header(data, size);

    // Find column indices
    std::unordered_map<std::string, int> col_indices;
    for (size_t i = 0; i < columns.size(); ++i) {
        col_indices[columns[i]] = i;
    }

    // Validate and collect numeric column indices
    std::vector<int> numeric_indices;
    std::vector<std::string> found_numeric;
    for (const auto& col : numeric_cols) {
        auto it = col_indices.find(col);
        if (it != col_indices.end()) {
            numeric_indices.push_back(it->second);
            found_numeric.push_back(col);
        }
    }

    // Validate and collect string column indices
    std::vector<int> string_indices;
    std::vector<std::string> found_string;
    for (const auto& col : string_cols) {
        auto it = col_indices.find(col);
        if (it != col_indices.end()) {
            string_indices.push_back(it->second);
            found_string.push_back(col);
        }
    }

    // Find start of data
    size_t data_start = 0;
    while (data_start < size && data[data_start] != '\n') data_start++;
    if (data_start < size) data_start++;

    // Count lines for pre-allocation
    size_t n_lines = 0;
    for (size_t i = data_start; i < size; ++i) {
        if (data[i] == '\n') n_lines++;
    }
    if (size > 0 && data[size-1] != '\n') n_lines++;

    // Pre-allocate storage
    std::vector<std::vector<float>> numeric_data(found_numeric.size());
    for (auto& v : numeric_data) v.reserve(n_lines);

    std::vector<std::vector<std::string>> string_data(found_string.size());
    for (auto& v : string_data) v.reserve(n_lines);

    // Parse data
    size_t pos = data_start;
    int64_t n_cols = static_cast<int64_t>(columns.size());
    size_t row_count = 0;

    while (pos < size) {
        // Skip empty lines
        if (data[pos] == '\n' || data[pos] == '\r') {
            pos++;
            continue;
        }

        std::vector<float> row_numeric(found_numeric.size(), 0.0f);
        std::vector<std::string> row_string(found_string.size());
        bool hit_newline = false;

        for (int64_t col = 0; col < n_cols && pos < size && !hit_newline; ++col) {
            std::string field = parse_field_fast(data, pos, size, ',', &hit_newline);

            // Check if this is a numeric column
            for (size_t ni = 0; ni < numeric_indices.size(); ++ni) {
                if (col == numeric_indices[ni]) {
                    row_numeric[ni] = parse_float_or_nan(field);
                    break;
                }
            }

            // Check if this is a string column
            for (size_t si = 0; si < string_indices.size(); ++si) {
                if (col == string_indices[si]) {
                    row_string[si] = std::move(field);
                    break;
                }
            }
        }

        // Store the row data
        for (size_t ni = 0; ni < found_numeric.size(); ++ni) {
            numeric_data[ni].push_back(row_numeric[ni]);
        }
        for (size_t si = 0; si < found_string.size(); ++si) {
            string_data[si].push_back(std::move(row_string[si]));
        }
        row_count++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (verbose) {
        std::cout << " done (" << row_count << " rows, "
                  << found_numeric.size() << " numeric + "
                  << found_string.size() << " string cols, "
                  << duration.count() / 1000.0 << "s)" << std::endl;
    }

    // Build result dict
    py::dict result;

    // Add numeric columns as tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    for (size_t i = 0; i < found_numeric.size(); ++i) {
        result[py::cast(found_numeric[i])] = torch::from_blob(
            numeric_data[i].data(), {static_cast<int64_t>(numeric_data[i].size())}, options
        ).clone();
    }

    // Add string columns as Python lists
    for (size_t i = 0; i < found_string.size(); ++i) {
        result[py::cast(found_string[i])] = py::cast(string_data[i]);
    }

    // Add row count
    result["_n_rows"] = row_count;

    return result;
}


/**
 * Load header CSV to tensors (numeric columns only).
 *
 * Returns dict of column_name -> float32 tensor
 */
std::unordered_map<std::string, torch::Tensor> load_header_csv(
    const std::string& path,
    const std::vector<std::string>& numeric_cols,
    const std::string& plot_id_col,
    bool verbose
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "  Loading: " << path << std::flush;
    }

    // Memory map the file
    MappedFile file(path);
    const char* data = file.data();
    size_t size = file.size();

    // Skip UTF-8 BOM if present
    size_t bom_offset = skip_bom(data, size);
    data += bom_offset;
    size -= bom_offset;

    if (verbose) {
        std::cout << " (" << (size / 1024 / 1024) << " MB)..." << std::flush;
    }

    // Parse header
    auto columns = parse_header(data, size);

    // Find column indices
    std::unordered_map<std::string, int> col_indices;
    int plot_idx = -1;
    for (size_t i = 0; i < columns.size(); ++i) {
        col_indices[columns[i]] = i;
        if (columns[i] == plot_id_col) plot_idx = i;
    }

    // Validate requested columns
    std::vector<int> target_indices;
    std::vector<std::string> found_cols;
    for (const auto& col : numeric_cols) {
        auto it = col_indices.find(col);
        if (it != col_indices.end()) {
            target_indices.push_back(it->second);
            found_cols.push_back(col);
        }
    }

    // Find start of data
    size_t data_start = 0;
    while (data_start < size && data[data_start] != '\n') data_start++;
    if (data_start < size) data_start++;

    // Count lines
    size_t n_lines = 0;
    for (size_t i = data_start; i < size; ++i) {
        if (data[i] == '\n') n_lines++;
    }
    if (size > 0 && data[size-1] != '\n') n_lines++;

    // Pre-allocate
    std::vector<std::vector<float>> col_data(found_cols.size());
    for (auto& v : col_data) v.reserve(n_lines);
    std::vector<std::string> plot_ids;
    plot_ids.reserve(n_lines);

    // Parse data
    size_t pos = data_start;
    int64_t n_cols = static_cast<int64_t>(columns.size());

    while (pos < size) {
        // Skip empty lines
        if (data[pos] == '\n' || data[pos] == '\r') {
            pos++;
            continue;
        }

        std::vector<float> row_values(found_cols.size(), 0.0f);
        std::string plot_id;
        bool hit_newline = false;

        for (int64_t col = 0; col < n_cols && pos < size && !hit_newline; ++col) {
            std::string field = parse_field_fast(data, pos, size, ',', &hit_newline);

            if (col == plot_idx) {
                plot_id = std::move(field);
            } else {
                // Check if this is a target column
                for (size_t ti = 0; ti < target_indices.size(); ++ti) {
                    if (col == target_indices[ti]) {
                        row_values[ti] = parse_float_or_nan(field);
                        break;
                    }
                }
            }
        }

        // Store the row data
        for (size_t ti = 0; ti < found_cols.size(); ++ti) {
            col_data[ti].push_back(row_values[ti]);
        }
        if (!plot_id.empty()) {
            plot_ids.push_back(plot_id);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (verbose) {
        std::cout << " done (" << plot_ids.size() << " rows, "
                  << found_cols.size() << " cols, "
                  << duration.count() / 1000.0 << "s)" << std::endl;
    }

    // Convert to tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    std::unordered_map<std::string, torch::Tensor> result;

    for (size_t i = 0; i < found_cols.size(); ++i) {
        result[found_cols[i]] = torch::from_blob(
            col_data[i].data(), {static_cast<int64_t>(col_data[i].size())}, options
        ).clone();
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_species_csv", &load_species_csv,
          "Load species CSV to PyTorch tensors (legacy API, memory-mapped)",
          py::arg("path"),
          py::arg("plot_id_col") = "PlotObservationID",
          py::arg("species_id_col") = "WFO_TAXON",
          py::arg("weight_col") = "Cover %",
          py::arg("verbose") = true);

    m.def("load_grouped_csv", &load_grouped_csv,
          "Load grouped CSV with arbitrary columns (memory-mapped)",
          py::arg("path"),
          py::arg("group_id_col"),
          py::arg("numeric_cols"),
          py::arg("string_cols"),
          py::arg("hash_string_cols") = true,
          py::arg("verbose") = true);

    m.def("load_header_csv", &load_header_csv,
          "Load header CSV numeric columns to tensors",
          py::arg("path"),
          py::arg("numeric_cols"),
          py::arg("plot_id_col") = "PlotObservationID",
          py::arg("verbose") = true);

    m.def("load_header_csv_full", &load_header_csv_full,
          "Load header CSV with both numeric tensors and string lists",
          py::arg("path"),
          py::arg("numeric_cols"),
          py::arg("string_cols"),
          py::arg("verbose") = true);
}
