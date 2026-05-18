// Damerau-Levenshtein automaton driven by row-state DP.
// See fuzzy_internal.hpp for the public surface and references.

#include "fuzzy_internal.hpp"

#include <algorithm>
#include <utility>

namespace resolve::fuzzy::detail {

// ---------------------------------------------------------------------------
// UTF-8 decoding
// ---------------------------------------------------------------------------

std::vector<uint32_t> utf8_to_codepoints(std::string_view s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());

    std::size_t i = 0;
    while (i < s.size()) {
        uint8_t b = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0;
        int extra = 0;

        if (b < 0x80) {
            cp = b;
            extra = 0;
        } else if ((b & 0xE0) == 0xC0) {
            cp = b & 0x1Fu;
            extra = 1;
        } else if ((b & 0xF0) == 0xE0) {
            cp = b & 0x0Fu;
            extra = 2;
        } else if ((b & 0xF8) == 0xF0) {
            cp = b & 0x07u;
            extra = 3;
        } else {
            // Stray continuation or 5/6-byte form: emit replacement, resync.
            out.push_back(0xFFFDu);
            ++i;
            continue;
        }
        ++i;

        bool ok = true;
        for (int k = 0; k < extra; ++k) {
            if (i >= s.size()) { ok = false; break; }
            uint8_t cb = static_cast<uint8_t>(s[i]);
            if ((cb & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cb & 0x3Fu);
            ++i;
        }
        out.push_back(ok ? cp : 0xFFFDu);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Lowercase
// ---------------------------------------------------------------------------

uint32_t to_lower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp + 32u;
    // Latin-1 Supplement: A0-DE -> E0-FE, excluding multiplication sign (D7).
    if (cp >= 0x00C0u && cp <= 0x00DEu && cp != 0x00D7u) return cp + 0x20u;
    return cp;
}

void to_lower_inplace(std::vector<uint32_t>& cps) {
    for (auto& c : cps) c = to_lower_cp(c);
}

// ---------------------------------------------------------------------------
// LevenshteinAutomaton
// ---------------------------------------------------------------------------

LevenshteinAutomaton::LevenshteinAutomaton(std::vector<uint32_t> needle,
                                           int max_k,
                                           bool damerau)
    : needle_(std::move(needle)),
      max_k_(max_k),
      damerau_(damerau),
      m_(static_cast<int>(needle_.size()))
{}

void LevenshteinAutomaton::initial_row(std::vector<int>& row) const {
    row.resize(static_cast<std::size_t>(m_ + 1));
    for (int i = 0; i <= m_; ++i) row[static_cast<std::size_t>(i)] = i;
}

bool LevenshteinAutomaton::step(const std::vector<int>& parent_row,
                                const std::vector<int>* grandparent_row,
                                uint32_t parent_char,
                                uint32_t c,
                                int effective_k,
                                std::vector<int>& new_row) const
{
    new_row.resize(static_cast<std::size_t>(m_ + 1));

    // new_row[0] = parent_row[0] + 1 (one extra insertion vs. previous depth).
    new_row[0] = parent_row[0] + 1;
    int row_min = new_row[0];
    const int cap = effective_k + 1;  // Anything strictly greater is dead.
    if (new_row[0] > cap) new_row[0] = cap;

    for (int i = 1; i <= m_; ++i) {
        const uint32_t needle_char = needle_[static_cast<std::size_t>(i - 1)];
        int sub_cost = (needle_char == c) ? 0 : 1;

        int v = parent_row[static_cast<std::size_t>(i - 1)] + sub_cost;           // substitute / match
        int del = parent_row[static_cast<std::size_t>(i)] + 1;                     // delete from needle
        if (del < v) v = del;
        int ins = new_row[static_cast<std::size_t>(i - 1)] + 1;                    // insert into needle
        if (ins < v) v = ins;

        // Damerau transposition: needle[i-1] matches the previously consumed
        // candidate char, and needle[i-2] matches the current candidate char.
        if (damerau_ && grandparent_row != nullptr && i >= 2 &&
            needle_[static_cast<std::size_t>(i - 1)] == parent_char &&
            needle_[static_cast<std::size_t>(i - 2)] == c)
        {
            int trans = (*grandparent_row)[static_cast<std::size_t>(i - 2)] + 1;
            if (trans < v) v = trans;
        }

        if (v > cap) v = cap;
        new_row[static_cast<std::size_t>(i)] = v;
        if (v < row_min) row_min = v;
    }

    return row_min <= effective_k;
}

bool LevenshteinAutomaton::accepts(const std::vector<int>& row, int& distance) const {
    int d = row[static_cast<std::size_t>(m_)];
    if (d <= max_k_) {
        distance = d;
        return true;
    }
    return false;
}

} // namespace resolve::fuzzy::detail
