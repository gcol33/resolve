# Latent and species embeddings

A trained RESOLVE model holds three kinds of learned vectors, called
*embeddings*: one per plot, one per species, and one per genus and per
family. The plot vector is the model’s summary of a plot, from which it
predicts every target. The species, genus and family vectors are the
rows of the tables the model looks species up in.

This article covers getting each kind out of a trained model, projecting
the plot vectors onto two axes to look at them, comparing species and
genera by their vectors, which encodings keep the effect of each species
separate, finding similar and unusual plots and reading gradients from
the vectors, and computing plot vectors from the encoded inputs
directly.

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

## A model to read

The examples use the survey of 150 plots in the Alps that ships with the
R package: `plots` with one row per plot (coordinates, elevation in
metres, habitat) and `species` with one row per species recorded on a
plot (genus, family, cover in percent). The model predicts elevation and
habitat from the species alone.

The *encoding* is how the model turns a plot’s species list into
numbers. The rank-pool encoding used here gives every species its own
row in one table and averages the rows of a plot’s species, weighted by
their cover, so it has a species table to read. [Encoding the species
set](https://gillescolling.com/resolve/articles/encoding.md) compares
the encodings.

``` r

plots <- read.csv(system.file("extdata", "plots.csv", package = "resolveR"))
species <- read.csv(system.file("extdata", "species.csv", package = "resolveR"))

dataset <- resolve.dataset.frame(
  plots, species,
  roles = list(plot_id = "plot_id", species_id = "species",
               abundance = "cover", genus = "genus", family = "family"),
  targets = list(
    elevation = list(column = "elevation", task = "regression"),
    habitat = list(column = "habitat", task = "classification")
  ),
  config = list(species_encoding = "rank_pool")
)
path <- file.path(tempdir(), "rank_pool.pt")
fit <- resolve.train.dataset(dataset, maxEpochs = 100L,
                             savePath = path, verbose = FALSE)
predictor <- resolve.load(path)
```

``` python
import os
import tempfile

import numpy as np
import pandas as pd
import resolve_core as rc

data = "https://raw.githubusercontent.com/gcol33/resolve/master/r/inst/extdata/"
plots = pd.read_csv(data + "plots.csv")
species = pd.read_csv(data + "species.csv")

roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"
targets = [rc.TargetSpec.regression("elevation"),
           rc.TargetSpec.classification("habitat", 4)]
config = rc.DatasetConfig()
config.species_encoding = rc.SpeciesEncodingMode.RankPool
dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets, config)

model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
train_config = rc.TrainConfig()
train_config.max_epochs = 100
trainer = rc.Trainer(model, train_config)
trainer.prepare_data(dataset)
result = trainer.fit()

path = os.path.join(tempfile.gettempdir(), "rank_pool.pt")
trainer.save(path)
predictor = rc.Predictor.load(path)
```

## Plot embeddings

The model’s *encoder* turns a plot’s species composition, and its
coordinates and covariates when it has them, into one vector, the plot’s
*latent representation*. Each target has a *head*, a small network that
reads its prediction off that vector. Predicting with the latent
representation switched on returns it, one row per plot, in the
dataset’s plot order.

``` r

out <- resolve.predict.dataset(predictor, dataset, returnLatent = TRUE)
latent <- out$latent
dim(latent)
#> [1] 150  64
tail(predictor$model_config()$hidden_dims, 1)
#> [1] 64
```

``` python
out = predictor.predict_dataset(dataset, return_latent=True)
latent = out.latent.numpy()
latent.shape
#> (150, 64)
predictor.model.latent_dim
#> 64
```

The vector is as wide as the encoder’s last hidden layer. Every head
reads the same vector, so two plots close together in the latent space
are plots the model expects to have similar values on every target at
once.

## Projecting the latent space

Sixty-four numbers per plot cannot be drawn directly. Principal
component analysis finds the directions along which the plot vectors
vary most; the first two give a map of the plots in which distances
approximate those in the latent space. The habitat is taken from `plots`
by plot identifier, since the result follows the dataset’s plot order.

``` r

pcs <- prcomp(latent)
habitat <- factor(plots$habitat[match(out$plot_ids, plots$plot_id)])
colours <- c("#E69F00", "#0072B2", "#009E73", "#CC79A7")
shapes <- c(16, 17, 15, 18)

par(mar = c(4.5, 4.5, 3, 1))
plot(pcs$x[, 1], pcs$x[, 2], col = colours[habitat], pch = shapes[habitat],
     xlab = "Principal component 1", ylab = "Principal component 2")
legend("top", legend = levels(habitat), col = colours, pch = shapes,
       horiz = TRUE, bty = "n", inset = c(0, -0.14), xpd = TRUE)
```

![Scatter plot of the 150 plots on the first two principal components of
their latent vectors, with one colour and point shape per
habitat.](embeddings_files/figure-html/pca-1.svg)

``` python
import matplotlib.pyplot as plt

centred = latent - latent.mean(axis=0)
u, s, vt = np.linalg.svd(centred, full_matrices=False)
pcs = centred @ vt.T
habitat = pd.Series(out.plot_ids).map(plots.set_index("plot_id").habitat)

colours = ["#E69F00", "#0072B2", "#009E73", "#CC79A7"]
shapes = ["o", "^", "s", "D"]
plt.rcParams["font.size"] = 13
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, colour, shape in zip(sorted(habitat.unique()), colours, shapes):
    rows = (habitat == name).to_numpy()
    ax.scatter(pcs[rows, 0], pcs[rows, 1], c=colour, marker=shape, label=name)
ax.set_xlabel("Principal component 1");
ax.set_ylabel("Principal component 2");
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False);
fig.tight_layout()
```

![Scatter plot of the 150 plots on the first two principal components of
their latent vectors, with one colour and point shape per
habitat.](embeddings_files/py-figure-1.svg)

Plots of one habitat mostly fall together, because the habitat head is
trained to tell them apart from the same vectors. The signs of the
components are arbitrary, so the map can come out mirrored between runs.
Non-linear projections such as UMAP (uniform manifold approximation and
projection) or t-SNE (t-distributed stochastic neighbour embedding) take
the same matrix as input.

## Species embeddings

The species table has one row per species in the vocabulary, the list of
species the model was trained with, in the same order. Row 1 in R and
row 0 in Python is the reserved unknown entry: every species outside the
vocabulary is looked up there.

``` r

species_table <- predictor$get_species_embeddings()
species_names <- predictor$species_vocab()
dim(species_table)
#> [1] 26 32
head(species_names, 4)
#> [1] "<UNK>"               "Acer pseudoplatanus" "Silene acaulis"     
#> [4] "Abies alba"
```

``` python
species_table = predictor.get_species_embeddings().numpy()
species_names = list(predictor.species_vocab)
species_table.shape
#> (26, 32)
species_names[:4]
#> ['<UNK>', 'Acer pseudoplatanus', 'Silene acaulis', 'Abies alba']
```

The encodings that keep a row per species return a table: embed, rank
pool and transformer. Under the embed encoding each rank position, from
the most to the least abundant species of a plot, has its own region of
the table, and the vector returned for a species is the average over the
positions. The hash encoding computes a species’ contribution from its
name and has no table; the sparse encoding gives a species its own
column in the weights of the first layer. Both return `NULL` in R and
`None` in Python.

## Taxonomy embeddings

With a genus or family column among the roles, every encoding also
learns a table per taxonomic level. Both tables reserve their first row
for the unknown entry too.

``` r

genus_table <- predictor$get_genus_embeddings()
genus_names <- predictor$genus_vocab()
dim(genus_table)
#> [1] 23  8
dim(predictor$get_family_embeddings())
#> [1] 11  8
head(genus_names, 4)
#> [1] "<UNK>"         "Abies"         "Acer"          "Arrhenatherum"
```

``` python
genus_table = predictor.get_genus_embeddings().numpy()
genus_names = list(predictor.genus_vocab)
genus_table.shape
#> (23, 8)
predictor.get_family_embeddings().shape
#> torch.Size([11, 8])
genus_names[:4]
#> ['<UNK>', 'Abies', 'Acer', 'Arrhenatherum']
```

A dataset carries the same vocabularies, so the names are available
without a checkpoint.

``` r

head(dataset$schema()$genus_vocab, 4)
#> [1] "<UNK>"         "Abies"         "Acer"          "Arrhenatherum"
head(dataset$species_vocab(), 4)
#> [1] "<UNK>"               "Acer pseudoplatanus" "Silene acaulis"     
#> [4] "Abies alba"
```

``` python
dataset.schema.genus_vocab[:4]
#> ['<UNK>', 'Abies', 'Acer', 'Arrhenatherum']
dataset.species_vocab[:4]
#> ['<UNK>', 'Acer pseudoplatanus', 'Silene acaulis', 'Abies alba']
```

## Comparing species and genera

The cosine similarity of two vectors is 1 when they point the same way,
0 when they are unrelated and -1 when they point in opposite directions.
Computed between all rows of the genus table, leaving out the unknown
entry, it lists the genera whose vectors are closest to a chosen one.

``` r

cosine <- function(x) {
  unit <- x / sqrt(rowSums(x^2))
  unit %*% t(unit)
}
genera <- genus_table[-1, ]
rownames(genera) <- genus_names[-1]
similarity <- cosine(genera)
round(sort(similarity["Quercus", ], decreasing = TRUE)[2:6], 3)
#>     Trifolium Arrhenatherum       Festuca         Abies         Picea 
#>         0.731         0.668         0.575         0.487         0.480
```

``` python
def cosine(x):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    return unit @ unit.T

genera = pd.DataFrame(genus_table[1:], index=genus_names[1:])
similarity = pd.DataFrame(cosine(genera.to_numpy()),
                          index=genera.index, columns=genera.index)
similarity["Quercus"].drop("Quercus").sort_values(ascending=False).head(5).round(3)
#> Trifolium        0.731
#> Arrhenatherum    0.665
#> Festuca          0.577
#> Picea            0.487
#> Abies            0.478
#> Name: Quercus, dtype: float32
```

Two genera end up close when the model found them interchangeable for
predicting its targets on this dataset. That can match ecological or
phylogenetic similarity, and it measures neither.

The same table clusters hierarchically. Ward’s method joins, at each
step, the two groups whose merger adds the least within-group variance.

``` r

tree <- hclust(dist(genera), method = "ward.D2")
par(mar = c(4.5, 1, 1, 9))
plot(as.dendrogram(tree), horiz = TRUE, xlab = "Height")
```

![Dendrogram of the 22 genera of the survey, clustered by Ward's method
on their embedding
vectors.](embeddings_files/figure-html/dendrogram-1.svg)

``` python
from scipy.cluster.hierarchy import dendrogram, linkage

tree = linkage(genera.to_numpy(), method="ward")
fig, ax = plt.subplots(figsize=(6.5, 6))
dendrogram(tree, labels=list(genera.index), orientation="left", ax=ax,
           leaf_font_size=13,
           color_threshold=0, above_threshold_color="black");
ax.set_xlabel("Height");
fig.tight_layout()
```

![Dendrogram of the 22 genera of the survey, clustered by Ward's method
on their embedding vectors.](embeddings_files/py-figure-2.svg)

The species table compares the same way.

``` r

known <- species_table[-1, ]
rownames(known) <- species_names[-1]
round(sort(cosine(known)["Carex curvula", ], decreasing = TRUE)[2:4], 3)
#>   Vaccinium myrtillus Arrhenatherum elatius        Silene acaulis 
#>                 0.273                 0.221                 0.204
```

``` python
known = species_table[1:]
names = pd.Index(species_names[1:])
row = cosine(known)[names.get_loc("Carex curvula")]
pd.Series(row, index=names).drop("Carex curvula").nlargest(3).round(3)
#> Vaccinium myrtillus      0.271
#> Arrhenatherum elatius    0.220
#> Silene acaulis           0.204
#> dtype: float32
```

## Linear pooling and what stays readable

Several encodings combine the species of a plot linearly before any
non-linear layer mixes them:

- **hash** adds each species’ cover, with a sign of +1 or -1, to the
  column its name hashes to, so a plot’s hashed vector is a signed sum
  of one term per species;
- **rank pool** takes the cover-weighted mean of the species’ rows;
- **sparse** passes the plot’s vector of species covers to a linear
  layer, which adds one column of weights per species.

In these three, the input to the encoder is a sum of one term per
species, so it splits into the contribution of each species. The split
holds up to the first non-linear layer; after it, the encoder mixes the
terms. The transformer encoding gives up the split on purpose: attention
lets the effect of a species depend on the other species of the plot,
which is what it is used for.

## Practical uses

### Similar plots

The plots most similar to a given one are those whose latent vectors
have the highest cosine similarity to its vector.

``` r

rownames(latent) <- out$plot_ids
plot_similarity <- cosine(latent)
round(sort(plot_similarity["A001", ], decreasing = TRUE)[2:6], 3)
#>  A015  A108  A149  A098  A063 
#> 0.992 0.991 0.987 0.987 0.986
```

``` python
plot_similarity = pd.DataFrame(cosine(latent), index=out.plot_ids,
                               columns=out.plot_ids)
plot_similarity["A001"].drop("A001").nlargest(5).round(3)
#> A050    0.996
#> A063    0.996
#> A108    0.995
#> A099    0.995
#> A045    0.995
#> Name: A001, dtype: float32
```

[Predicting new
plots](https://gillescolling.com/resolve/articles/prediction.md) shows
retrieval, which blends a prediction with the recorded values of the
most similar training plots.

### Unusual plots

A plot whose nearest neighbours are all far from it has a composition
unlike the others the model saw. The mean similarity to its five nearest
neighbours ranks the plots by how isolated they are.

``` r

nearest <- apply(plot_similarity, 1, function(s) mean(sort(s, decreasing = TRUE)[2:6]))
round(sort(nearest)[1:5], 3)
#>  A128  A092  A067  A140  A030 
#> 0.703 0.773 0.866 0.880 0.901
```

``` python
nearest = plot_similarity.apply(lambda s: s.nlargest(6).iloc[1:].mean())
nearest.nsmallest(5).round(3)
#> A067    0.599
#> A092    0.763
#> A107    0.906
#> A148    0.910
#> A140    0.922
#> dtype: float32
```

Read an isolated plot next to its prediction error before calling it an
outlier: it may be rare in the survey and still well predicted.

### Gradients

The share of the variance each principal component explains shows how
many directions the latent space mainly uses. A component that
correlates with a target is a gradient along that target.

``` r

round(summary(pcs)$importance["Proportion of Variance", 1:3], 3)
#>   PC1   PC2   PC3 
#> 0.318 0.289 0.229
elevation <- plots$elevation[match(out$plot_ids, plots$plot_id)]
round(cor(pcs$x[, 1:3], elevation)[, 1], 3)
#>    PC1    PC2    PC3 
#> -0.686 -0.334  0.282
```

``` python
np.round(s[:3] ** 2 / np.sum(s ** 2), 3)
#> array([0.329, 0.276, 0.24 ], dtype=float32)
elevation = pd.Series(out.plot_ids).map(plots.set_index("plot_id").elevation)
[round(np.corrcoef(pcs[:, i], elevation)[0, 1], 3) for i in range(3)]
#> [np.float64(-0.428), np.float64(-0.678), np.float64(-0.086)]
```

## Embeddings from the encoded inputs

A predictor also computes the latent vectors from the encoded inputs
directly, the tensors a dataset holds, which is what a custom analysis
on modified inputs needs. It assembles and standardises the inputs as
training did, so the result equals the latent representation from
prediction.

``` python
inputs = dict(unknown_fraction=dataset.unknown_fraction,
              species_ids=dataset.species_ids,
              pool_genus_ids=dataset.pool_genus_ids,
              pool_family_ids=dataset.pool_family_ids,
              pool_weights=dataset.pool_weights,
              pool_mask=dataset.pool_mask,
              pool_has_cover=dataset.pool_has_cover)
direct = predictor.get_embeddings(**inputs)
direct.equal(out.latent)
```

Each input is a keyword, and inputs the encoding does not read are left
out. `ResolveModel.get_latent()` runs the encoder of a model you hold,
such as `trainer.model` in a custom training loop, on the same inputs,
with the continuous block (coordinates, covariates, the unknown-species
columns and, under the hash encoding, the hash vector) already assembled
and standardised. The model’s tables are read the same way, before or
after training:

``` python
trainer.model.get_species_weights().shape
trainer.model.get_genus_weights().shape
```

In R, `predictor$get_embeddings()` takes the encoded inputs as
positional arguments only, and a fit does not return its model, so this
route is shown in Python.

## Next steps

- [Predicting new
  plots](https://gillescolling.com/resolve/articles/prediction.md):
  scoring new plots, class probabilities and retrieval.
- [Encoding the species
  set](https://gillescolling.com/resolve/articles/encoding.md): which
  encodings build which tables.
- [Training and reading the
  fit](https://gillescolling.com/resolve/articles/training.md): the
  settings of the encoder these vectors come from.
