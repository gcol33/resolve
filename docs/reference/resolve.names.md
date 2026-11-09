# Resolve Species Names Against a Model's Vocabulary

A model knows each species by the name its training records were
resolved to. A table names a species as it was entered in the field, by
a current accepted name, or misspelled, and none of these need be the
name the model learnt. Each name is taken, first hit wins:

## Usage

``` r
resolve.names(names, model, target = NULL, table = NULL, maxEditDistance = 1L)
```

## Arguments

- names:

  Character vector of species names as written.

- model:

  Where the vocabulary comes from: a `Suite` from
  [`resolve.load_suite()`](https://gillescolling.com/resolve/reference/resolve.load_suite.md)
  (with `target`), a `Predictor` from
  [`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md),
  or a character vector ordered as a checkpoint orders it (`"<UNK>"`
  first).

- target:

  The suite target whose vocabulary is used.

- table:

  The path of the release's `names.json`, or `NULL` to resolve against
  the vocabulary alone.

- maxEditDistance:

  The edits a misspelled name may be away from a known name (default 1;
  0 turns the spelling step off; 2 can also rename species the models
  never saw).

## Value

A data.frame, one row per name in order: `name`, `token` (the name the
model reads), `route`, `source`, `via`, `distance`.

## Details

- as written, where the vocabulary holds it (`exact`);

- through the release's name table, which maps EVA's entered names and
  the synonymy of the plant backbones to the names the models learnt
  (`alias`, with the `source` that linked them);

- as the nearest known name or table name within `maxEditDistance`
  edits, letter case ignored, searched within the name's genus first,
  and only when every nearest name stands for one species (`spelling`,
  with the name it was read `via` and the `distance`);

and otherwise left as written (`unresolved`), where it counts as a
species the model does not know.
[`resolve.predict.suite()`](https://gillescolling.com/resolve/reference/resolve.predict.suite.md)
and
[`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md)
apply the same steps through their `names` argument.

## Examples

``` r
vocabulary <- c("<UNK>", "Quercus robur", "Anemone altaica")
resolve.names(c("Quercus robur", "Quercus robru", "Poa annua"), vocabulary)
```
