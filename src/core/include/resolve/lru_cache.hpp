#pragma once

// A bounded, thread-safe least-recently-used cache from int64 keys to values.
//
// get_or_make(key, make) returns the cached value, or builds it with make()
// outside the lock (so a slow build does not block readers of other keys),
// stores it and evicts the least recently used entries beyond the capacity;
// find() and put() are its two halves, for a caller that builds many values
// at once. When two threads build the same key at once, the first stored
// value wins and both return it. Values are copied out, so they should be cheap to copy
// (tensors and torch module holders are reference counted).

#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace resolve {

template <typename Value>
class LruCache {
public:
    explicit LruCache(int64_t capacity) : capacity_(capacity) {
        if (capacity_ < 1) throw std::invalid_argument("LruCache: capacity must be >= 1");
    }

    [[nodiscard]] int64_t capacity() const noexcept { return capacity_; }

    // The cached value of `key`, marked most recently used, or nothing.
    [[nodiscard]] std::optional<Value> find(int64_t key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it == entries_.end()) return std::nullopt;
        order_.splice(order_.begin(), order_, it->second.second);
        return it->second.first;
    }

    // Store `value` under `key` unless a value is already stored there, and
    // return the stored one.
    Value put(int64_t key, Value value) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it != entries_.end()) return it->second.first;
        order_.push_front(key);
        entries_.emplace(key, std::make_pair(value, order_.begin()));
        while (static_cast<int64_t>(entries_.size()) > capacity_) {
            entries_.erase(order_.back());
            order_.pop_back();
        }
        return value;
    }

    template <typename Make>
    Value get_or_make(int64_t key, Make&& make) const {
        if (std::optional<Value> hit = find(key)) return *std::move(hit);
        return put(key, make());
    }

private:
    int64_t capacity_;
    mutable std::mutex mutex_;
    mutable std::list<int64_t> order_;
    mutable std::unordered_map<int64_t, std::pair<Value, std::list<int64_t>::iterator>> entries_;
};

}  // namespace resolve
