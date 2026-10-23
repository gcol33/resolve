#pragma once

// A registered config struct as a JSON object, one member per field-registry
// row the checkpoint carries (config_registry.hpp), and back. A row with an
// empty key is left out: a checkpoint reads it back as a struct default, not as
// the value the run used. Enums appear as the names the CLI accepts, nested
// sub-configs as nested objects, and a non-finite float as null. The checkpoint
// description (checkpoint.hpp) is built from it, so the JSON sidecar and
// `resolve info` show the same fields, and a sidecar's configs read back.

#include "resolve/config_registry.hpp"
#include "resolve/json.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace resolve {

struct JsonFieldWriter {
    json::Value& object;

    template <typename T>
    void operator()(const char* name, const char* key, const T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            // Neither has a checkpoint key: the callback has no value, and the
            // device names the machine a run happened on.
            (void)name;
            (void)key;
            (void)value;
        } else {
            if (!has_checkpoint_key(key)) return;
            object.set(name, to_json(value));
        }
    }

    template <typename T>
    static json::Value to_json(const T& value) {
        if constexpr (is_registered_config_v<T>) {
            json::Value out = json::Value::object();
            for_each_field(value, JsonFieldWriter{out});
            return out;
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            json::Value out = json::Value::array();
            for (const auto& branch : value) out.push_back(to_json(branch));
            return out;
        } else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> ||
                             std::is_same_v<T, float> || std::is_same_v<T, std::string>) {
            return json::Value(value);
        } else if constexpr (std::is_enum_v<T>) {
            return json::Value(enum_to_name(value));
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                             std::is_same_v<T, std::vector<float>>) {
            json::Value out = json::Value::array();
            for (const auto v : value) out.push_back(json::Value(v));
            return out;
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            json::Value out = json::Value::array();
            out.push_back(value.first);
            out.push_back(value.second);
            return out;
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no JSON representation; "
                          "add a branch to JsonFieldWriter");
        }
    }
};

// Inverse of JsonFieldWriter: a member the object holds is read onto the field
// of that name, and a field the object lacks keeps the config's value. A member
// of the wrong kind throws, naming the field.
struct JsonFieldReader {
    const json::Value& object;

    template <typename T>
    void operator()(const char* name, const char* key, T& value) const {
        if constexpr (std::is_same_v<T, LogCallback> || std::is_same_v<T, torch::Device>) {
            (void)name;
            (void)key;
            (void)value;
        } else {
            if (!has_checkpoint_key(key)) return;
            const json::Value* member = object.find(name);
            if (member == nullptr) return;
            try {
                from_json(*member, value);
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("config field '") + name + "': " + e.what());
            }
        }
    }

    static float number_or_nan(const json::Value& v) {
        return v.is_null() ? std::numeric_limits<float>::quiet_NaN()
                           : static_cast<float>(v.as_number());
    }

    template <typename T>
    static void from_json(const json::Value& v, T& value) {
        if constexpr (is_registered_config_v<T>) {
            for_each_field(value, JsonFieldReader{v});
        } else if constexpr (std::is_same_v<T, std::vector<ParallelBranchConfig>>) {
            value.clear();
            for (const auto& item : v.items()) {
                ParallelBranchConfig branch;
                from_json(item, branch);
                value.push_back(std::move(branch));
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            value = v.as_bool();
        } else if constexpr (std::is_same_v<T, int>) {
            value = static_cast<int>(v.as_int());
        } else if constexpr (std::is_same_v<T, float>) {
            value = number_or_nan(v);
        } else if constexpr (std::is_same_v<T, std::string>) {
            value = v.as_string();
        } else if constexpr (std::is_enum_v<T>) {
            value = enum_from_name<T>(v.as_string());
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
            value.clear();
            for (const auto& item : v.items()) value.push_back(item.as_int());
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
            value.clear();
            for (const auto& item : v.items()) value.push_back(number_or_nan(item));
        } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
            if (v.items().size() != 2) throw std::runtime_error("expected two integers");
            value = {static_cast<int>(v.items()[0].as_int()),
                     static_cast<int>(v.items()[1].as_int())};
        } else {
            static_assert(registry_detail::always_false<T>,
                          "config field type has no JSON representation; "
                          "add a branch to JsonFieldReader");
        }
    }
};

template <typename Cfg>
json::Value config_to_json(const Cfg& config) {
    static_assert(is_registered_config_v<Cfg>, "config_to_json takes a registered config");
    return JsonFieldWriter::to_json(config);
}

// The config `object` describes, read onto `config` (by default the struct's
// defaults).
template <typename Cfg>
Cfg config_from_json(const json::Value& object, Cfg config = Cfg{}) {
    static_assert(is_registered_config_v<Cfg>, "config_from_json reads a registered config");
    JsonFieldReader::from_json(object, config);
    return config;
}

}  // namespace resolve
