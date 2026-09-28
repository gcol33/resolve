#include "resolve/json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace resolve::json {

namespace {

// 2^53: every integer of smaller magnitude is exactly representable as a double.
constexpr double kExactIntegerLimit = 9007199254740992.0;

[[noreturn]] void wrong_kind(const char* wanted, Value::Kind found) {
    throw std::runtime_error(std::string("json: expected ") + wanted + ", found " +
                             kind_name(found));
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_document() {
        skip_whitespace();
        Value value = parse_value(0);
        skip_whitespace();
        if (pos_ != text_.size()) fail("unexpected content after the JSON value");
        return value;
    }

private:
    // Nesting bound: a manifest is a few levels deep, and a hostile or corrupt
    // file must not be able to exhaust the stack through recursion.
    static constexpr int kMaxDepth = 256;

    [[noreturn]] void fail(const std::string& what) const {
        std::size_t line = 1, column = 1;
        for (std::size_t i = 0; i < pos_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') { ++line; column = 1; } else { ++column; }
        }
        throw std::runtime_error("json: " + what + " at line " + std::to_string(line) +
                                 ", column " + std::to_string(column));
    }

    [[nodiscard]] bool at_end() const noexcept { return pos_ >= text_.size(); }
    [[nodiscard]] char peek() const noexcept { return at_end() ? '\0' : text_[pos_]; }

    void skip_whitespace() noexcept {
        while (!at_end()) {
            const char c = text_[pos_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++pos_;
        }
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    void expect_literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) {
            fail("invalid literal");
        }
        pos_ += word.size();
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) fail("nesting deeper than " + std::to_string(kMaxDepth));
        switch (peek()) {
            case '{': return parse_object(depth);
            case '[': return parse_array(depth);
            case '"': return Value(parse_string());
            case 't': expect_literal("true"); return Value(true);
            case 'f': expect_literal("false"); return Value(false);
            case 'n': expect_literal("null"); return Value();
            default: break;
        }
        const char c = peek();
        if (c == '-' || (c >= '0' && c <= '9')) return Value(parse_number());
        if (at_end()) fail("unexpected end of input");
        fail(std::string("unexpected character '") + c + "'");
    }

    Value parse_object(int depth) {
        expect('{');
        Value object = Value::object();
        skip_whitespace();
        if (peek() == '}') { ++pos_; return object; }
        while (true) {
            skip_whitespace();
            if (peek() != '"') fail("expected a member name");
            std::string key = parse_string();
            if (object.contains(key)) fail("duplicate member '" + key + "'");
            skip_whitespace();
            expect(':');
            skip_whitespace();
            object.set(key, parse_value(depth + 1));
            skip_whitespace();
            if (peek() == ',') { ++pos_; continue; }
            expect('}');
            return object;
        }
    }

    Value parse_array(int depth) {
        expect('[');
        Value array = Value::array();
        skip_whitespace();
        if (peek() == ']') { ++pos_; return array; }
        while (true) {
            skip_whitespace();
            array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (peek() == ',') { ++pos_; continue; }
            expect(']');
            return array;
        }
    }

    double parse_number() {
        const std::size_t start = pos_;
        if (peek() == '-') ++pos_;
        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            fail("invalid number");
        }
        if (peek() == '.') {
            ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (peek() == 'e' || peek() == 'E') {
            ++pos_;
            if (peek() == '+' || peek() == '-') ++pos_;
            if (!(peek() >= '0' && peek() <= '9')) fail("invalid number");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        // strtod reads the C locale's decimal point, and the grammar above has
        // already confirmed the span is a JSON number.
        const std::string span(text_.substr(start, pos_ - start));
        char* end = nullptr;
        const double value = std::strtod(span.c_str(), &end);
        if (end != span.c_str() + span.size()) fail("invalid number");
        if (!std::isfinite(value)) fail("number out of range");
        return value;
    }

    uint32_t parse_hex4() {
        if (pos_ + 4 > text_.size()) fail("truncated \\u escape");
        uint32_t code = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_++];
            code <<= 4;
            if (c >= '0' && c <= '9') code |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') code |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("invalid \\u escape");
        }
        return code;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (at_end()) fail("unterminated string");
            const char c = text_[pos_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) {
                --pos_;
                fail("control character in string");
            }
            if (c != '\\') { out += c; continue; }
            if (at_end()) fail("unterminated escape");
            const char e = text_[pos_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = parse_hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (text_.substr(pos_, 2) != "\\u") fail("unpaired surrogate");
                        pos_ += 2;
                        const uint32_t low = parse_hex4();
                        if (low < 0xDC00 || low > 0xDFFF) fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        fail("unpaired surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    --pos_;
                    fail(std::string("invalid escape '\\") + e + "'");
            }
        }
    }

    std::string_view text_;
    std::size_t pos_ = 0;
};

void dump_string(std::string& out, const std::string& s) {
    out += '"';
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

void dump_number(std::string& out, double n) {
    if (!std::isfinite(n)) {
        throw std::runtime_error("json: a non-finite number has no JSON spelling");
    }
    char buf[32];
    if (n == std::floor(n) && std::fabs(n) < kExactIntegerLimit) {
        std::snprintf(buf, sizeof(buf), "%.0f", n);
        // "-0" reads back as 0.0 either way; write the plain spelling.
        if (std::string_view(buf) == "-0") std::snprintf(buf, sizeof(buf), "0");
    } else {
        std::snprintf(buf, sizeof(buf), "%.17g", n);
    }
    out += buf;
}

void newline(std::string& out, int indent, int level) {
    if (indent <= 0) return;
    out += '\n';
    out.append(static_cast<std::size_t>(indent * level), ' ');
}

void dump_value(std::string& out, const Value& v, int indent, int level) {
    switch (v.kind()) {
        case Value::Kind::Null: out += "null"; return;
        case Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; return;
        case Value::Kind::Number: dump_number(out, v.as_number()); return;
        case Value::Kind::String: dump_string(out, v.as_string()); return;
        case Value::Kind::Array: {
            const auto& items = v.items();
            if (items.empty()) { out += "[]"; return; }
            out += '[';
            for (std::size_t i = 0; i < items.size(); ++i) {
                if (i) out += ',';
                newline(out, indent, level + 1);
                dump_value(out, items[i], indent, level + 1);
            }
            newline(out, indent, level);
            out += ']';
            return;
        }
        case Value::Kind::Object: {
            const auto& members = v.members();
            if (members.empty()) { out += "{}"; return; }
            out += '{';
            for (std::size_t i = 0; i < members.size(); ++i) {
                if (i) out += ',';
                newline(out, indent, level + 1);
                dump_string(out, members[i].first);
                out += indent > 0 ? ": " : ":";
                dump_value(out, members[i].second, indent, level + 1);
            }
            newline(out, indent, level);
            out += '}';
            return;
        }
    }
}

}  // namespace

const char* kind_name(Value::Kind kind) noexcept {
    switch (kind) {
        case Value::Kind::Null: return "null";
        case Value::Kind::Bool: return "a boolean";
        case Value::Kind::Number: return "a number";
        case Value::Kind::String: return "a string";
        case Value::Kind::Array: return "an array";
        case Value::Kind::Object: return "an object";
    }
    return "an unknown kind";
}

Value Value::array_of(const std::vector<std::string>& strings) {
    Value v = array();
    v.items_.reserve(strings.size());
    for (const auto& s : strings) v.items_.emplace_back(s);
    return v;
}

Value Value::array_of(const std::vector<double>& numbers) {
    Value v = array();
    v.items_.reserve(numbers.size());
    for (const double n : numbers) v.items_.emplace_back(n);
    return v;
}

bool Value::as_bool() const {
    if (kind_ != Kind::Bool) wrong_kind("a boolean", kind_);
    return bool_;
}

double Value::as_number() const {
    if (kind_ != Kind::Number) wrong_kind("a number", kind_);
    return number_;
}

int64_t Value::as_int() const {
    const double n = as_number();
    if (n != std::floor(n) || std::fabs(n) >= kExactIntegerLimit) {
        throw std::runtime_error("json: expected an integer, found " + dump(*this, 0));
    }
    return static_cast<int64_t>(n);
}

const std::string& Value::as_string() const {
    if (kind_ != Kind::String) wrong_kind("a string", kind_);
    return string_;
}

const Value::Items& Value::items() const {
    if (kind_ != Kind::Array) wrong_kind("an array", kind_);
    return items_;
}

Value::Items& Value::items() {
    if (kind_ != Kind::Array) wrong_kind("an array", kind_);
    return items_;
}

const Value::Members& Value::members() const {
    if (kind_ != Kind::Object) wrong_kind("an object", kind_);
    return members_;
}

std::vector<std::string> Value::as_string_array() const {
    std::vector<std::string> out;
    out.reserve(items().size());
    for (const auto& item : items()) out.push_back(item.as_string());
    return out;
}

const Value* Value::find(std::string_view key) const {
    for (const auto& [name, value] : members()) {
        if (name == key) return &value;
    }
    return nullptr;
}

const Value& Value::at(std::string_view key) const {
    const Value* v = find(key);
    if (!v) throw std::runtime_error("json: no member '" + std::string(key) + "'");
    return *v;
}

Value& Value::set(const std::string& key, Value value) {
    if (kind_ != Kind::Object) wrong_kind("an object", kind_);
    for (auto& [name, existing] : members_) {
        if (name == key) {
            existing = std::move(value);
            return existing;
        }
    }
    members_.emplace_back(key, std::move(value));
    return members_.back().second;
}

void Value::push_back(Value value) {
    if (kind_ != Kind::Array) wrong_kind("an array", kind_);
    items_.push_back(std::move(value));
}

std::size_t Value::size() const {
    if (kind_ == Kind::Array) return items_.size();
    if (kind_ == Kind::Object) return members_.size();
    wrong_kind("an array or an object", kind_);
}

bool operator==(const Value& a, const Value& b) {
    if (a.kind_ != b.kind_) return false;
    switch (a.kind_) {
        case Value::Kind::Null: return true;
        case Value::Kind::Bool: return a.bool_ == b.bool_;
        case Value::Kind::Number: return a.number_ == b.number_;
        case Value::Kind::String: return a.string_ == b.string_;
        case Value::Kind::Array: return a.items_ == b.items_;
        case Value::Kind::Object: return a.members_ == b.members_;
    }
    return false;
}

Value parse(std::string_view text) {
    // A UTF-8 byte-order mark is not JSON, but editors on Windows write one.
    if (text.substr(0, 3) == "\xEF\xBB\xBF") text.remove_prefix(3);
    return Parser(text).parse_document();
}

std::string dump(const Value& value, int indent) {
    std::string out;
    dump_value(out, value, indent, 0);
    return out;
}

Value read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("json: cannot open '" + path + "'");
    std::ostringstream buffer;
    buffer << in.rdbuf();
    try {
        return parse(buffer.str());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string(e.what()) + " in '" + path + "'");
    }
}

void write_file(const std::string& path, const Value& value, int indent) {
    const std::string text = dump(value, indent) + "\n";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("json: cannot write '" + path + "'");
    out << text;
    if (!out) throw std::runtime_error("json: writing '" + path + "' failed");
}

}  // namespace resolve::json
