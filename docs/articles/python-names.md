# Python: species names

Reading a table’s species names as the names a model learnt: the
release’s name table, the resolver, the setting a load or a suite
imputation takes, and the report of the names not found as written.
[Species names a model knows by another
name](https://gillescolling.com/resolve/articles/prediction.html#species-names)
walks through it.

[All of the Python
reference](https://gillescolling.com/resolve/articles/python-reference.md)

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

## `resolve_names()`

``` python
resolve_names(names, model, target=None, table=True, max_edit_distance=1)
```

Resolves a list of species names against a model’s vocabulary and
returns a `pandas.DataFrame`, one row per name in order: `name`, `token`
(the name the model reads), `route`, `source`, `via`, `distance`.
`model` is a `SuitePredictor` with `target`, a `Predictor`, or a
vocabulary as a checkpoint orders it (`"<UNK>"` first). `table` is the
path of the release’s `names.json` or a `NameTable`; `None` or `True`
resolves against the vocabulary alone. Each name takes the first route
that reaches a token:

- `exact`: the vocabulary holds the name as written;
- `alias`: the name table maps the name to a token the vocabulary holds,
  `source` naming what linked them (`eva` for the names EVA’s records
  were entered under, else the backbone, e.g. `wfo`);
- `spelling`: within `max_edit_distance` edits (Damerau-Levenshtein,
  letter case ignored) of a token or a table name whose token the
  vocabulary holds, searched within the name’s genus first and across
  every name only when the genus holds none, and used only when every
  nearest name stands for one species; `via` is the name it was read
  through and `distance` the edits;
- `unresolved`: none of these; the name stays as written and counts as a
  species the model does not know.

At one edit, misspelled copies of the example plots’ names resolved
99.5% to the right species and none to a wrong one; two edits also
renamed some species the models never saw, so 1 is the default.

``` python
import resolve_core as rc

vocabulary = ["<UNK>", "Quercus robur", "Anemone altaica"]
table = rc.NameTable.from_columns(["Anemone nemorosa"], ["Anemone altaica"], ["eva"])
rc.resolve_names(["Quercus robru", "Anemone nemorosa", "Poa annua"], vocabulary, table=table)
```

## `NameTable`

``` python
NameTable.open(path)
NameTable.read(path)
NameTable.from_columns(names, tokens, sources)
```

The release’s name table: each name a user may write for a species, the
token it stands for, and the source that linked them. `open` reads a
`names.json` (`{"source": ..., "aliases": {name: [token, source]}}`)
once per process and hands later callers the same table while the file
keeps its size and modification time; `read` reads it again every time.
`find(name)` returns `(token, source)` or `None`; `len(table)`,
`name in table`, `description` and `sha256` (of the file read) describe
it.

## `SpeciesNames`

``` python
SpeciesNames(table=None, max_edit_distance=1)
```

How a load or a suite imputation resolves species names: the `NameTable`
(`None`: the vocabulary alone) and the edits a misspelling may be away
(0 turns the spelling step off). `SuitePredictor.predict(names=...)`
builds one from a path, a `NameTable` or `True`; the `*_with_vocabs`
loaders of `ResolveDataset` take one as `names`.

## `NameResolver`

``` python
NameResolver(vocabulary, table=None, max_edit_distance=1)
```

Resolves names against one vocabulary. `resolve(name)` returns a
`ResolvedName`, `resolve_batch(names)` a list of them. The spelling
index is built on the first name that needs it.

## `ResolvedName`

What the resolver gives for one name: `token`, `route` (a `NameRoute`),
`source`, `via` and `distance`.

## `NameRoute`

`Exact`, `Alias`, `Spelling` or `Unresolved`.

## `SpeciesNameReport`

The species names a load or a suite target did not find as written, one
row per name in the order they first appear: `name`, `token`, `route`
(as text), `source`, `via`, `distance` and `n_records`, the records that
carried the name. `ResolveDataset.species_names` and
`SuiteTargetPrediction.species_names` return one; `to_pandas()` gives it
as a data frame.
