#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace resolve {

// Interned names: every distinct name is stored once and coded 0, 1, 2, ... in
// the order it was first seen. The loaders intern species, genus, family and
// plot names as they read rows, so a record is a handful of integers instead
// of a set of strings, and grouping by a name is grouping by an integer.
class NamePool {
public:
    // The code of `name`, and whether this call added it.
    std::pair<int32_t, bool> insert(std::string_view name) {
        auto it = codes_.find(name);
        if (it != codes_.end()) return {it->second, false};
        if (names_.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::length_error("NamePool: more distinct names than an int32 code can hold");
        }
        const auto code = static_cast<int32_t>(names_.size());
        // A deque never moves its elements, so the view used as the key stays
        // valid as the pool grows.
        names_.emplace_back(name);
        codes_.emplace(names_.back(), code);
        return {code, true};
    }

    int32_t intern(std::string_view name) { return insert(name).first; }

    // The code of `name`, or -1 when the pool does not hold it.
    [[nodiscard]] int32_t find(std::string_view name) const {
        auto it = codes_.find(name);
        return it != codes_.end() ? it->second : -1;
    }

    [[nodiscard]] const std::string& name(int32_t code) const {
        return names_[static_cast<size_t>(code)];
    }

    [[nodiscard]] int32_t size() const noexcept { return static_cast<int32_t>(names_.size()); }

    // The names in code order, moved out; the pool is empty afterwards.
    std::vector<std::string> release() {
        codes_.clear();
        std::vector<std::string> out;
        out.reserve(names_.size());
        for (auto& name : names_) out.push_back(std::move(name));
        names_.clear();
        return out;
    }

private:
    std::deque<std::string> names_;
    std::unordered_map<std::string_view, int32_t> codes_;
};

}  // namespace resolve
