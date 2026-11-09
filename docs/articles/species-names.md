# Species names and the name table

The released models were trained on species names in the European
Vegetation Archive (EVA) standardized to the World Flora Online (WFO)
taxonomic backbone, v.2.1 of 22 January 2024. A table of new plots names
its species as they were entered in the field, by a current accepted
name, or by a name from another checklist, and the models read a name
only when it is exactly one they learnt. The name table, `names.json`,
maps the names a user may write to the names the models learnt. This
article explains where its links come from, which taxonomic backbones it
reads, and how a name is matched through it. Reading a table’s names
through it during imputation is covered in [Imputing new
plots](https://gillescolling.com/resolve/articles/prediction.html#species-names);
installation in [Installing the
engine](https://gillescolling.com/resolve/articles/installation.md).

## Why a name can differ from the model’s name

EVA’s contributing databases enter each species under their own
checklist’s name, and EVA standardized every record to the WFO backbone
v.2.1 of 22 January 2024. The two need not agree. EVA resolved its
71,014 records entered as *Anemone nemorosa* to *Anemone altaica*, so a
model knows wood anemone only under that name. A table written today may
also use a newer accepted name: the WFO release the name table reads
through taxify (2026.06, of 21 June 2026) accepts *Anemonoides
nemorosa*, and an older table may call pedunculate oak *Quercus
pedunculata*, which WFO lists as a synonym of *Quercus robur*. Without
the table, all three names count as species the model never saw, and
their records are left out of the imputation.

## How the table is built

Each link in the table maps one name to one token, a name some released
model holds in its vocabulary. The links come from two sources, read in
order:

1.  **EVA’s records.** Each name a vascular plant record was entered
    under is linked to the token most of that name’s records were
    resolved to. This gives *Anemone nemorosa* → *Anemone altaica*.
2.  **The synonymy of the taxonomic backbones.** In each backbone, every
    entered name and every token is looked up; its accepted name and all
    synonyms of that accepted name are linked to the token those records
    were resolved to. This gives *Anemonoides nemorosa* → *Anemone
    altaica* and *Quercus pedunculata* → *Quercus robur*. A backbone
    spanning several kingdoms is read for plants only.

A name that several sources link to different tokens takes the link from
the earliest source (EVA, then the backbones in the order of the table
below), then the token with most records. A name that is itself a token
is never in the table, so a name a model holds is always read as
written. A model renames a name only when its own vocabulary lacks the
name and holds the token; the released models differ in vocabulary, so
the same table can rename a name for one model and not for another.

The released table holds 299,549 names linked to tokens of the 17,890
species the released models hold between them. For each name it also
records which sources agree on its token: 62,492 names are linked by one
source alone, the others by two to twelve. For 54,744 names some source
links another token; the file keeps those alternatives beside the chosen
link (`support` and `alternatives`), so a disputed name can be checked.

## The taxonomic backbones

The backbones are read through the taxify package, each at the release
recorded in `names.json`. World Flora Online comes first because EVA
resolved its records to it.

| Order | Backbone | Release (upstream date) | Names whose link it gives | Names it links to the same token |
|---:|----|----|---:|---:|
| 1 | World Flora Online (WFO) | 2026.06 (2026-06-21) | 159,032 | 166,218 |
| 2 | Euro+Med PlantBase | 2026.08 (2026-07-11) | 14,613 | 59,166 |
| 3 | World Checklist of Vascular Plants (WCVP) | 2026.06 (2026-06-04) | 40,880 | 181,843 |
| 4 | Leipzig Catalogue of Vascular Plants (LCVP) | 2026.08 (2022-11-07) | 27,263 | 145,072 |
| 5 | Catalogue of Life (COL) | 2026.09 (2026-09-11) | 19,914 | 182,953 |
| 6 | GBIF Backbone Taxonomy | 2023.08 (2023-08-28) | 18,347 | 204,942 |
| 7 | Integrated Taxonomic Information System (ITIS) | 2026.09 (2026-08-27) | 2,207 | 11,532 |
| 8 | NCBI Taxonomy | 2026.09 (2026-09) | 710 | 11,745 |
| 9 | Open Tree of Life | 2026.09 (2025-12-20) | 355 | 3,150 |
| 10 | Catalogue of Life Extended Release (COL XR) | 2026.09 (2026-09-25) | 5,084 | 208,842 |
| 11 | World Register of Marine Species (WoRMS) | 2026.09 (2026-09-01) | 284 | 20,043 |

The fourth column counts the names whose chosen link came from that
backbone; the fifth counts every name the backbone links to the chosen
token, whether or not an earlier source gave the link first. EVA’s
records give the remaining 10,860 links. The eight other backbones
taxify holds (Index Fungorum, AlgaeBase, FishBase, SeaLifeBase, the
Reptile Database, the Mammal Diversity Database, AviList and LPSN) were
read as well and give no link: they hold no vascular plants, or none
that an earlier source had not already linked.

## Matching a name through the table

A name is read in steps, and the first that reaches a token wins: as
written when the model’s vocabulary holds it (`exact`), through the
table (`alias`, with the `source` of the link), as the nearest name
within one edit of a token or a table name (`spelling`, with the name it
was read `via`), and otherwise left as written (`unresolved`). The
example writes a table of three real links in the format of `names.json`
and resolves six names against a vocabulary of two species.

``` r

table <- tempfile(fileext = ".json")
writeLines('{"source": "example",
  "aliases": {"Anemone nemorosa": ["Anemone altaica", "eva"],
              "Anemonoides nemorosa": ["Anemone altaica", "wfo"],
              "Quercus pedunculata": ["Quercus robur", "wfo"]}}', table)

vocabulary <- c("<UNK>", "Anemone altaica", "Quercus robur")
resolve.names(c("Anemone altaica", "Anemone nemorosa", "Anemonoides nemorosa",
                "Quercus pedunculata", "Quercus robru", "Carex firma"),
              vocabulary, table = table)
#>                   name           token      route source           via distance
#> 1      Anemone altaica Anemone altaica      exact                             0
#> 2     Anemone nemorosa Anemone altaica      alias    eva                      0
#> 3 Anemonoides nemorosa Anemone altaica      alias    wfo                      0
#> 4  Quercus pedunculata   Quercus robur      alias    wfo                      0
#> 5        Quercus robru   Quercus robur   spelling        Quercus robur        1
#> 6          Carex firma     Carex firma unresolved                             0
```

``` python
import json, tempfile
import resolve_core as rc

table = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
json.dump({"source": "example",
           "aliases": {"Anemone nemorosa": ["Anemone altaica", "eva"],
                       "Anemonoides nemorosa": ["Anemone altaica", "wfo"],
                       "Quercus pedunculata": ["Quercus robur", "wfo"]}}, table)
table.close()

vocabulary = ["<UNK>", "Anemone altaica", "Quercus robur"]
rc.resolve_names(["Anemone altaica", "Anemone nemorosa", "Anemonoides nemorosa",
                  "Quercus pedunculata", "Quercus robru", "Carex firma"],
                 vocabulary, table=table.name)
#>                    name            token  ...            via distance
#> 0       Anemone altaica  Anemone altaica  ...                       0
#> 1      Anemone nemorosa  Anemone altaica  ...                       0
#> 2  Anemonoides nemorosa  Anemone altaica  ...                       0
#> 3   Quercus pedunculata    Quercus robur  ...                       0
#> 4         Quercus robru    Quercus robur  ...  Quercus robur        1
#> 5           Carex firma      Carex firma  ...                       0
#>
#> [6 rows x 6 columns]
```

*Anemone altaica* is the model’s own name. The entered name *Anemone
nemorosa* reaches it through EVA’s records, the accepted name
*Anemonoides nemorosa* and the synonym *Quercus pedunculata* through
WFO. *Quercus robru* is one edit from *Quercus robur*. *Carex firma* is
in neither the vocabulary nor the table and stays unresolved; its
records would be left out.

The released `names.json` ships with the models in the [Zenodo record of
the browser app](https://doi.org/10.5281/zenodo.24077609), inside
`models.tar`. Its path is the `table` argument of
[`resolve.names()`](https://gillescolling.com/resolve/reference/resolve.names.md)
in R and `resolve_names()` in Python, and the `names` argument of the
loaders and of suite imputation.

## Where next

- [Imputing new
  plots](https://gillescolling.com/resolve/articles/prediction.html#species-names):
  reading a table’s names through `names` during imputation, and the
  report of the names not found as written.
- [Python: species
  names](https://gillescolling.com/resolve/articles/python-names.md):
  the name table, resolver and report classes in Python.
- [The browser app](https://gillescolling.com/resolve/articles/app.md):
  imputes plots in the browser and lists the names it renamed and the
  ones it did not recognise.
