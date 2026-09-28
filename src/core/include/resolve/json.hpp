#pragma once

// A JSON value, its reader and its writer.
//
// The engine reads and writes small structured documents -- the model-suite
// manifest (suite.hpp) above all -- and a document another program wrote has to
// be read back exactly, which a line-oriented ad-hoc reader cannot promise.
// This is the whole of RFC 8259 the engine needs: the six value kinds, objects
// that keep their member order (so a written manifest diffs cleanly and reads
// in the order it was authored), \u escapes with surrogate pairs, and errors
// that name the line and column where the text stopped being JSON.
//
// Numbers are held as double. Every integer a manifest carries (a seed, a byte
// count, a format version) is far inside 2^53, where a double is exact, and
// `as_int` refuses a number that is not integral rather than truncating it.

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace resolve::json {

class Value {
public:
    enum class Kind { Null, Bool, Number, String, Array, Object };

    using Members = std::vector<std::pair<std::string, Value>>;
    using Items = std::vector<Value>;

    Value() = default;
    Value(std::nullptr_t) {}  // NOLINT(google-explicit-constructor)
    Value(bool b) : kind_(Kind::Bool), bool_(b) {}  // NOLINT(google-explicit-constructor)
    Value(double n) : kind_(Kind::Number), number_(n) {}  // NOLINT(google-explicit-constructor)
    Value(int n) : Value(static_cast<double>(n)) {}  // NOLINT(google-explicit-constructor)
    Value(int64_t n) : Value(static_cast<double>(n)) {}  // NOLINT(google-explicit-constructor)
    Value(std::string s) : kind_(Kind::String), string_(std::move(s)) {}  // NOLINT(google-explicit-constructor)
    Value(const char* s) : Value(std::string(s)) {}  // NOLINT(google-explicit-constructor)

    static Value array() { Value v; v.kind_ = Kind::Array; return v; }
    static Value object() { Value v; v.kind_ = Kind::Object; return v; }
    static Value array_of(const std::vector<std::string>& strings);
    static Value array_of(const std::vector<double>& numbers);

    [[nodiscard]] Kind kind() const noexcept { return kind_; }
    [[nodiscard]] bool is_null() const noexcept { return kind_ == Kind::Null; }
    [[nodiscard]] bool is_bool() const noexcept { return kind_ == Kind::Bool; }
    [[nodiscard]] bool is_number() const noexcept { return kind_ == Kind::Number; }
    [[nodiscard]] bool is_string() const noexcept { return kind_ == Kind::String; }
    [[nodiscard]] bool is_array() const noexcept { return kind_ == Kind::Array; }
    [[nodiscard]] bool is_object() const noexcept { return kind_ == Kind::Object; }

    // Typed reads. Each throws std::runtime_error naming the kind it found when
    // the value is of another kind.
    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_number() const;
    // The number as an integer; throws when it has a fractional part or lies
    // outside the range a double represents exactly.
    [[nodiscard]] int64_t as_int() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Items& items() const;
    [[nodiscard]] Items& items();
    [[nodiscard]] const Members& members() const;
    [[nodiscard]] std::vector<std::string> as_string_array() const;

    // Object access. `find` returns null when the key is absent; `at` throws
    // naming the key. `set` replaces an existing member in place, so a member's
    // position is the one it was first given.
    [[nodiscard]] const Value* find(std::string_view key) const;
    [[nodiscard]] const Value& at(std::string_view key) const;
    [[nodiscard]] bool contains(std::string_view key) const { return find(key) != nullptr; }
    Value& set(const std::string& key, Value value);

    // Array append.
    void push_back(Value value);

    [[nodiscard]] std::size_t size() const;

    friend bool operator==(const Value& a, const Value& b);
    friend bool operator!=(const Value& a, const Value& b) { return !(a == b); }

private:
    Kind kind_ = Kind::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    Items items_;
    Members members_;
};

[[nodiscard]] const char* kind_name(Value::Kind kind) noexcept;

// Parse a complete JSON text. Throws std::runtime_error carrying the line and
// column of the first character that is not valid JSON, and rejects trailing
// content after the value.
[[nodiscard]] Value parse(std::string_view text);

// Serialize. `indent` > 0 pretty-prints with that many spaces per level and a
// newline between members; 0 writes the compact form. A number that is
// integral and within 2^53 is written without a fractional part; any other
// finite number with 17 significant digits, which reads back to the same
// double. A non-finite number has no JSON spelling and throws.
[[nodiscard]] std::string dump(const Value& value, int indent = 2);

// Whole-file helpers. Both throw std::runtime_error naming the path.
[[nodiscard]] Value read_file(const std::string& path);
void write_file(const std::string& path, const Value& value, int indent = 2);

}  // namespace resolve::json
