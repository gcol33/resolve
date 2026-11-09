# Fuzzy String Index

An index over a vocabulary of strings (species names, say) that finds
the entries within a few edits of a query: Damerau-Levenshtein distance,
where an insertion, a deletion, a substitution and, with `damerau`, a
swap of two adjacent characters each cost one edit. Query it with
[`resolve.fuzzy_match()`](https://gillescolling.com/resolve/reference/resolve.fuzzy_match.md).

## Usage

``` r
resolve.fuzzy_index(
  entries,
  maxEditDistance = NULL,
  caseInsensitive = NULL,
  damerau = NULL,
  buckets = NULL
)
```

## Arguments

- entries:

  Character vector, the vocabulary.

- maxEditDistance:

  The largest edit distance a query may ask for.

- caseInsensitive:

  Fold case of entries and queries before matching.

- damerau:

  Count a swap of two adjacent characters as one edit.

- buckets:

  Optional character vector, one bucket per entry (the genus of a
  species name, say): a query given a bucket hint searches that bucket
  first and falls back to the whole vocabulary when it finds nothing
  there.

## Value

A `FuzzyIndex` object (an Rcpp module class) with methods `size()`,
`bucket_count()` and `max_supported_distance()`.

## Details

Settings left `NULL` take the engine's defaults.

## Examples

``` r
index <- resolve.fuzzy_index(c("Quercus robur", "Quercus petraea", "Fagus sylvatica"))
resolve.fuzzy_match(index, c("Quercus robr", "Fagus silvatica"))
```
