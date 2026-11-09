# Build the Species Graph a Heterogeneous GNN Passes Messages On

Joins the species of a dataset's vocabulary by the relations
`heterogeneous_gnn` reads: a same-genus and a same-family edge for
species sharing a taxon, and a co-occurrence edge between species
recorded in the same plot often enough.
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
builds this itself for a heterogeneous-GNN model
(`modelConfig = list(encoder_architecture = "heterogeneous_gnn")`) and
the checkpoint carries the result, so call it directly only to inspect
the graph or to build a variant by hand.

## Usage

``` r
resolve.species_graph(
  dataset,
  kCooccurrence = NULL,
  cooccurrenceThreshold = NULL,
  taxonomicEdges = NULL,
  cooccurrenceEdges = NULL
)
```

## Arguments

- dataset:

  A `ResolveDataset`, as returned by
  [`resolve.dataset.csv()`](https://gillescolling.com/resolve/reference/resolve.dataset.csv.md).
  Co-occurrence edges need the per-plot species vector, so
  `species_encoding = "sparse"`; taxonomic edges need genus/family
  roles.

- kCooccurrence:

  How many co-occurrence partners to keep per species.

- cooccurrenceThreshold:

  Smallest share of plots two species must share for a co-occurrence
  edge.

- taxonomicEdges:

  Join species sharing a genus or a family.

- cooccurrenceEdges:

  Join species recorded together.

## Value

A list with `edge_index` (a 2-row matrix of edge endpoints, source in
row 1 and target in row 2, as 0-based species codes), `edge_type` (0
same genus, 1 same family, 2 co-occurrence), `n_species` and `n_edges`.

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets,
                               config = list(species_encoding = "sparse"))
graph <- resolve.species_graph(dataset)
graph$n_edges
```
