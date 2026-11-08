# Encoding the species set

A plot’s species list has no fixed length: one plot records three
species, the next forty. A neural network reads inputs of a fixed width,
so the list has to be turned into numbers first. This step is the
*species encoding*. RESOLVE has five encodings; they differ in how many
of a plot’s species they keep, whether two species can end up in the
same input, and how much of the representation is learned.

This article covers the five encodings and their settings, the settings
that choose which species of a plot enter the encoding, the encoder
architectures that read the encoded plot, and a short comparison of the
encodings on a small survey.

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

## The survey

The examples use the survey of 150 plots in the Alps that ships with the
R package, with its 25 species. [Preparing the two
tables](https://gillescolling.com/resolve/articles/data-preparation.md)
describes the tables and the roles.

``` r

plots <- read.csv(system.file("extdata", "plots.csv", package = "resolveR"))
species <- read.csv(system.file("extdata", "species.csv", package = "resolveR"))

roles <- list(plot_id = "plot_id", species_id = "species",
              abundance = "cover", genus = "genus", family = "family")
targets <- list(elevation = list(task = "regression"),
                habitat = list(task = "classification"))

load_with <- function(config) {
  resolve.dataset.frame(plots, species, roles = roles,
                        targets = targets, config = config)
}
```

``` python
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
           rc.TargetSpec.classification("habitat", 0)]


def load_with(**settings):
    config = rc.DatasetConfig()
    for name, value in settings.items():
        setattr(config, name, value)
    return rc.ResolveDataset.from_pandas(plots, species, roles, targets, config)
```

## Choosing the encoding

The encoding is a dataset setting, `species_encoding`, because the
loader writes the tensors the encoding needs. The model takes the
encoding from the dataset it is built for, together with the widths that
follow from it.

[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
builds the model from the dataset, so the encoding is set once, in
`config`.

`rc.model_config_for_dataset(dataset, model_config)` copies the
encoding, `hash_dim`, `top_k`, `top_k_species` and the taxonomy slots
from the dataset into a model configuration. Build every model as
`rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset, model_config))`.

| Encoding | Model input | Learned | Species the model never saw |
|----|----|----|----|
| `hash` | a fixed-width vector of hashed species | no | hashed like any other name |
| `embed` | the embeddings of the most abundant species, side by side | yes | the unknown embedding |
| `sparse` | one column per species in the vocabulary | in the first layer | no column |
| `rank_pool` | a weighted sum of species, genus and family embeddings | yes | the unknown embedding |
| `transformer` | species tokens, optionally attending to each other, then pooled | yes | the unknown embedding |

An *embedding* is a learned vector that stands for one species, genus or
family. The *vocabulary* is the list of species the training data holds,
with one integer code each; code 0 is reserved for a species outside it.
With the genus and family roles set, a species outside the vocabulary
still carries its genus and family embeddings.

## Hash

The hash encoding maps each species name to one of `hash_dim` positions
and to a sign, and adds the species’ weight there. Two species can share
a position; the random sign makes such collisions cancel on average
instead of adding up. Nothing about a species is learned, and there is
no embedding table, so memory does not grow with the number of species.
It is the default encoding, and the only one with a dedicated CUDA
kernel for the graphics card.

``` r

hashed <- load_with(list(species_encoding = "hash", hash_dim = 64L,
                         selection = "all", normalization = "log1p"))
dim(hashed$hash_embedding())
#> [1] 150  64
```

``` python
hashed = load_with(species_encoding=rc.SpeciesEncodingMode.Hash, hash_dim=64,
                   selection=rc.SelectionMode.All,
                   normalization=rc.NormalizationMode.Log1p)
hashed.hash_embedding.shape
#> torch.Size([150, 64])
```

`top_k` (default 3) is the number of species hashed per plot under the
default `selection = "top"`, so the default hash encoding reads a plot’s
three most abundant species. `selection = "all"` hashes the whole list.
`top_k` also sets the number of genus and family slots, the fixed
positions that hold a plot’s leading genera and families, so raising it
widens the taxonomy input too.

`normalization` rescales the weights before hashing: `"raw"` (default)
uses the abundance as recorded, `"norm"` divides it by the plot total,
and `"log1p"` takes `log(1 + abundance)`.

Use the hash encoding for a first run on a new dataset, for vocabularies
too large for an embedding table, and during a hyperparameter search
where the time per run matters.

## Embed

The embed encoding ranks a plot’s species by abundance, looks up the
embedding of each of the first `top_k_species` (default 10), and places
them side by side. The input width is `top_k_species` times the
embedding width `species_embed_dim` (default 32). A plot with fewer
species fills the remaining slots with the unknown embedding. The slots
keep their rank, so the model can treat the most abundant species
differently from the fourth.

``` r

embedded <- load_with(list(species_encoding = "embed", top_k_species = 5L))
dim(embedded$species_ids())
#> [1] 150   5
```

``` python
embedded = load_with(species_encoding=rc.SpeciesEncodingMode.Embed,
                     top_k_species=5)
embedded.species_ids.shape
#> torch.Size([150, 5])
```

Every species past rank `top_k_species` is left out, which is most of a
species-rich plot. The genus and family go into `top_k` fixed slots,
filled with the plot’s leading genera and families.

Use the embed encoding when a few dominant species carry the signal, and
when you want one vector per species to inspect afterwards.

## Sparse

The sparse encoding writes each plot as one row of a community matrix:
one column per species in the vocabulary, holding its abundance, or 1
under `representation = "presence_absence"`. Nothing is left out and no
two species share a column. The first layer holds one weight vector per
species, so its weights can be read species by species.

``` r

community <- load_with(list(species_encoding = "sparse"))
dim(community$species_vector_dense())
#> [1] 150  26
```

``` python
community = load_with(species_encoding=rc.SpeciesEncodingMode.Sparse)
community.species_vector.shape
#> torch.Size([150, 26])
```

The width is the size of the vocabulary, with its reserved slot, so the
first layer grows with the number of species and most entries of a row
are zero. A species absent from the training vocabulary has no column.

Use the sparse encoding on small to moderate vocabularies, and when the
model input should match a community matrix column by column. The
heterogeneous graph network below needs it.

## Rank pool

The rank-pool encoding keeps every species of a plot. The loader writes,
per plot, the species, genus and family codes of each species, a weight
and a mask, padded to the longest plot. The model sums each table’s
embeddings over the plot’s species, each weighted by `pool_weighting`.

| `pool_weighting`    | Weight of a species                        |
|---------------------|--------------------------------------------|
| `"binary"`          | 1 for every species present                |
| `"abundance"`       | the recorded abundance                     |
| `"log1p"` (default) | `log(1 + abundance)`                       |
| `"norm"`            | the abundance divided by the plot total    |
| `"rank"`            | `1 / rank`, ranked by descending abundance |

`pool_species_cap` bounds the padded width: 0 (default) pads to the
longest plot, a positive value keeps that many species, and `-x` keeps
the (100 - x)th percentile of species counts, so `-1` cuts at the 99th
percentile and reports what it left out.

``` r

pooled <- load_with(list(species_encoding = "rank_pool",
                         pool_weighting = "log1p", pool_species_cap = -1L))
dim(pooled$pool_weights())
#> [1] 150   8
```

``` python
pooled = load_with(species_encoding=rc.SpeciesEncodingMode.RankPool,
                   pool_weighting=rc.PoolWeighting.Log1p, pool_species_cap=-1)
pooled.pool_weights.shape
#> torch.Size([150, 8])
```

*Cover dropout* (`coverDropout` in R, `cover_dropout` in Python, default
0) replaces a training plot’s weights with plain presence at the given
probability, each epoch. The model also receives a flag saying whether a
plot’s weights carry cover, so a model trained with cover dropout learns
to predict plots that come without abundances.

*Plot totals* (`plotTotals` in R, `plot_totals` in Python, default off)
add two numbers per plot to the pooled features: the combined cover of
all its species, taken as overlapping at random, and the logarithm of
one plus its species count. The weighted mean of the rank-pool encoding
carries neither, and the attention pooling of the transformer not the
count, although both can matter: the EUNIS expert system, for one,
assigns most sparsely vegetated habitats only below 30% total cover.
With *totals dropout* (`plotTotalsDropout`, `plot_totals_dropout`) a
training plot’s totals are withheld at the given probability, and the
model learns to predict without them as well;
`resolve.plot_totals(predictor, FALSE)` (R) or
`predictor.plot_totals = False` (Python) then withholds them at
prediction, so that a prediction does not depend on how many species a
plot records, as an analysis of species richness may require.

A sum ignores the order of the species, so the encoding holds which
species are present and how strongly, and nothing about how they
combine. Use it when species richness varies widely, when rare species
matter, and when taxonomy is available.

## Transformer

The transformer encoding reads the same per-species tensors as rank
pooling. It adds each species’ species, genus and family embeddings into
one *token*, a vector of width `d_model`, passes the tokens through
`n_attention_layers` self-attention layers (default 0), and pools them
into one vector. In a self-attention layer each token is updated from
the other tokens of the plot, so a species’ contribution can depend on
the species it grows with.

| Setting (R argument) | Default | Meaning |
|----|----|----|
| `d_model` (`dModel`) | 128 | Token width |
| `n_heads` (`nHeads`) | 4 | Attention heads per layer |
| `n_attention_layers` (`nAttentionLayers`) | 0 | Self-attention layers |
| `transformer_ff_dim` (`transformerFfDim`) | 256 | Width of the feed-forward step in each layer |
| `transformer_pooling` (`transformerPooling`) | `"attention"` | How the tokens are pooled |
| `transformer_dropout` (`transformerDropout`) | 0.1 | Dropout inside the layers |

Attention pooling learns a weight for each token and takes the weighted
sum. `"cls"` pooling adds one learned token and reads its output, so it
needs at least one attention layer; with none it is refused. With no
attention layers the encoding is a rank pool whose weights are learned.

``` r

attended <- load_with(list(species_encoding = "transformer"))
fit <- resolve.train.dataset(attended, dModel = 64L, nHeads = 4L,
                             nAttentionLayers = 2L, transformerFfDim = 128L,
                             lr = 3e-4, maxEpochs = 100L, verbose = FALSE)
fit$result$best_epoch
#> [1] 99
```

``` python
attended = load_with(species_encoding=rc.SpeciesEncodingMode.Transformer)
model_config = rc.ModelConfig()
model_config.d_model = 64
model_config.n_heads = 4
model_config.n_attention_layers = 2
model_config.transformer_ff_dim = 128
model = rc.ResolveModel(attended.schema,
                        rc.model_config_for_dataset(attended, model_config))

train_config = rc.TrainConfig()
train_config.lr = 3e-4
train_config.max_epochs = 100
trainer = rc.Trainer(model, train_config)
trainer.prepare_data(attended)
trainer.fit().best_epoch
#> 90
```

`best_epoch` is the epoch whose weights the fit kept. The cost of
attention grows with the square of the number of species per plot, so
`pool_species_cap` matters more here, and each epoch takes longer than
under the other encodings. Attention layers are sensitive to the
learning rate and to the reduced range of 16-bit arithmetic: start from
a lower learning rate, as above, and leave mixed precision (`use_amp`,
off by default) off when the loss turns into NaN.

Use the transformer encoding when interactions between species plausibly
carry signal, on data large enough to fit attention weights, and after a
simpler encoding has given a number to compare against.

## Which species enter the encoding

Each encoding has a per-plot *budget*, the number of species it keeps,
set by the setting that also fixes its width: `top_k` for hash,
`top_k_species` for embed, and `species_budget` for sparse, rank pool
and transformer. `species_budget` defaults to 0, no budget, so those
three encode every species a plot records. `selection` decides which
species fill the budget:

| `selection`       | Species kept per plot                    |
|-------------------|------------------------------------------|
| `"top"` (default) | the most abundant                        |
| `"bottom"`        | the least abundant                       |
| `"top_bottom"`    | both ends, half each, without duplicates |
| `"all"`           | every species                            |

The embed encoding has a fixed number of ranked slots, so it refuses
`"all"`. Its slots follow the ranking: slot 0 holds the most abundant
species under `"top"` and the least abundant under `"bottom"`. A species
recorded more than once in a plot, for example once per vegetation
layer, enters every encoding once, at its summed abundance.

A budget on a pooled encoding gives a species ablation, a fit that sees
only part of each plot. Below, rank pooling reads each plot’s three
least abundant species.

``` r

rarest <- load_with(list(species_encoding = "rank_pool",
                         selection = "bottom", species_budget = 3L))
dim(rarest$pool_weights())
#> [1] 150   3
```

``` python
rarest = load_with(species_encoding=rc.SpeciesEncodingMode.RankPool,
                   selection=rc.SelectionMode.Bottom, species_budget=3)
rarest.pool_weights.shape
#> torch.Size([150, 3])
```

`representation = "presence_absence"` sets every species’ value to 1 in
every encoding: the hash and sparse values, the pool weights, the
ranking of the genus and family slots and the share of unknown species.
The selection still ranks by recorded abundance, so presence-absence
under `"top"` keeps the dominant species.

`aggregation` ranks the genus and family slots of hash, sparse and
embed: `"abundance"` (default) by the summed abundance of their species,
`"count"` by how many of the plot’s species they hold. Rank pool and
transformer carry each species’ own genus and family; under `"count"`
each species counts once in its plot’s genus and family embeddings,
instead of at its pool weight.

## Encoders above the encoding

The *encoder* is the network that reads the encoded plot, together with
the coordinates and covariates, and produces the plot’s latent
representation, the vector every target is predicted from. The default
is a multilayer perceptron (MLP), a stack of fully connected layers
whose widths `hiddenDims` / `hidden_dims` sets. `encoder_architecture`
replaces it with a network from the literature on tabular data, each
with its own settings under the name in the table.

| `encoder_architecture` | Network |
|----|----|
| `"mlp"` (default) | Fully connected layers |
| `"ft_transformer"` | FT-Transformer: one token per input feature, with self-attention across features |
| `"tabnet"` | TabNet: sequential attention that selects features step by step |
| `"saint"` | SAINT: attention across features and across rows; at prediction, rows attend to a fixed set of training rows |
| `"excelformer"` | ExcelFormer: attention guided by learned feature importance |
| `"trait_net"` | TraitNet: combines the plot’s environment with a trait table per species |
| `"gnn"` | A graph network over plots, joined by coordinates, taxonomy or composition; trains on all plots at once |
| `"heterogeneous_gnn"` | A graph network over species, joined by shared genus, shared family and co-occurrence; needs the sparse encoding |

Every encoding works with every architecture except where the table
names a requirement; a pair that cannot work is refused when the model
is built or first run. The graph network over plots joins them by
coordinates by default, so it needs the latitude and longitude roles.

``` r

fit <- resolve.train.dataset(
  hashed, maxEpochs = 50L, verbose = FALSE,
  modelConfig = list(encoder_architecture = "ft_transformer")
)
round(fit$result$final_metrics$habitat$accuracy, 3)
#> [1] 0.9
```

``` python
model_config = rc.ModelConfig()
model_config.encoder_architecture = rc.EncoderArchitecture.FTTransformer
model = rc.ResolveModel(hashed.schema,
                        rc.model_config_for_dataset(hashed, model_config))
train_config = rc.TrainConfig()
train_config.max_epochs = 50
trainer = rc.Trainer(model, train_config)
trainer.prepare_data(hashed)
round(trainer.fit().final_metrics["habitat"]["accuracy"], 3)
#> 0.8
```

[Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) lists the
settings of each architecture and the mixture of experts that can sit
above any of them.

## Comparing the encodings

The loop below fits the survey under each encoding with the default
settings and prints the held-out mean absolute error of elevation, in
metres. The *held-out* plots are the share training sets aside and never
fits on.

``` r

encodings <- c("hash", "embed", "sparse", "rank_pool", "transformer")
mae <- sapply(encodings, function(encoding) {
  fit <- resolve.train.dataset(load_with(list(species_encoding = encoding)),
                               maxEpochs = 100L, verbose = FALSE)
  fit$result$final_metrics$elevation$mae
})
round(mae)
#>        hash       embed      sparse   rank_pool transformer 
#>         231         238         232         148         193
```

``` python
def held_out_mae(encoding):
    dataset = load_with(species_encoding=encoding)
    model = rc.ResolveModel(dataset.schema,
                            rc.model_config_for_dataset(dataset, rc.ModelConfig()))
    train_config = rc.TrainConfig()
    train_config.max_epochs = 100
    trainer = rc.Trainer(model, train_config)
    trainer.prepare_data(dataset)
    return trainer.fit().final_metrics["elevation"]["mae"]


encodings = ["Hash", "Embed", "Sparse", "RankPool", "Transformer"]
{name: round(held_out_mae(getattr(rc.SpeciesEncodingMode, name)))
 for name in encodings}
#> {'Hash': 218, 'Embed': 222, 'Sparse': 225, 'RankPool': 170, 'Transformer': 149}
```

These numbers say little about the encodings. The survey has 150 plots
and 25 species, the held-out fold is 30 plots, and the starting weights
differ from run to run, so the R and the Python run can rank the
encodings differently. A comparison on your own data runs each encoding
under the same cross-validation folds; [Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) shows how.

## Choosing

| Situation | Start with |
|----|----|
| First run on a new dataset | `hash` with `hash_dim = 64` and `selection = "all"` |
| A vocabulary of tens of thousands of species | `hash` or `rank_pool` |
| A small vocabulary, with the input read as a community matrix | `sparse` |
| A few dominant species carry the signal | `embed` |
| Richness varies widely, rare species matter | `rank_pool` |
| Interactions between species matter and the data is large | `transformer` |

## Settings that travel with the encoding

| Setting | hash, embed, sparse | rank_pool | transformer |
|----|----|----|----|
| `lr` | default `1e-3` | default `1e-3` | lower, `3e-4` as a start |
| `use_amp` | can be on | can be on | off when the loss turns into NaN |
| `pool_weighting` | not read | read | read |
| `pool_species_cap` | not read | read | read, and matters more |
| `selection` | read, with `top_k` (hash) or `top_k_species` (embed); sparse with `species_budget` | read, with `species_budget` | read, with `species_budget` |
| `species_budget` | read by sparse only | read | read |

## Next steps

- [Training and reading the
  fit](https://gillescolling.com/resolve/articles/training.md): the
  training settings, the held-out plots and cross-validation.
- [Latent and species
  embeddings](https://gillescolling.com/resolve/articles/embeddings.md):
  reading what the encoders learned.
- [Large data and
  GPUs](https://gillescolling.com/resolve/articles/performance.md):
  speed and memory of each encoding.
