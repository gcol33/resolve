# Match Strings Against a Fuzzy Index

The entries of a
[`resolve.fuzzy_index()`](https://gillescolling.com/resolve/reference/resolve.fuzzy_index.md)
within `maxEditDistance` edits of each needle, nearest first (ties by
entry position), at most `topN` per needle.

## Usage

``` r
resolve.fuzzy_match(
  index,
  needles,
  maxEditDistance = NULL,
  topN = NULL,
  bucketHint = NULL,
  bucketHints = NULL
)
```

## Arguments

- index:

  A `FuzzyIndex` from
  [`resolve.fuzzy_index()`](https://gillescolling.com/resolve/reference/resolve.fuzzy_index.md).

- needles:

  Character vector of strings to match.

- maxEditDistance:

  Largest edit distance of a match (at most the index's
  `maxEditDistance`).

- topN:

  Matches kept per needle.

- bucketHint:

  One bucket to search first for every needle.

- bucketHints:

  A bucket per needle to search first (a character vector as long as
  `needles`); not combined with `bucketHint`.

## Value

A data.frame with one row per match: `needle` (its position in
`needles`), `query` (the needle), `entry` (the matched entry), `id` (its
position in the index's entries) and `distance`. A needle with no match
has no row.

## Details

Settings left `NULL` take the engine's defaults.

## Examples

``` r
index <- resolve.fuzzy_index(c("Quercus robur", "Fagus sylvatica"))
resolve.fuzzy_match(index, "Quercus robr", topN = 1)
```
