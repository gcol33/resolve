# RESOLVE

*predicting what a sample is from what's in it*

[![Tests](https://github.com/gcol33/resolve/actions/workflows/tests.yml/badge.svg)](https://github.com/gcol33/resolve/actions/workflows/tests.yml)
[![R-CMD-check](https://github.com/gcol33/resolve/actions/workflows/r-cmd-check.yml/badge.svg)](https://github.com/gcol33/resolve/actions/workflows/r-cmd-check.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://gillescolling.com/resolve)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Predict sample-level outcomes from compositional data, with a C++ engine on libtorch.**

Hand RESOLVE two tables: one row per sample, and one row per entity observed in a
sample. It learns a single representation of the composition and reads several
attributes off it at once, so one encoder answers a continuous question and a
categorical one from the same latent state. The engine is a standalone C++ library;
Python, R, and the command line are thin bindings over it, so a model trained from
one is identical to a model trained from any other.

```python
import resolve_core as rc

roles = rc.RoleMapping()
roles.plot_id    = "PlotID"
roles.species_id = "Species"
roles.abundance  = "Cover"

dataset = rc.ResolveDataset.from_csv(
    "plots.csv", "species.csv", roles,
    [rc.TargetSpec.regression("Area", rc.TransformType.Log1p),
     rc.TargetSpec.classification("Habitat", 9)],
)

trainer = rc.Trainer(rc.ResolveModel(dataset.schema, rc.ModelConfig()), rc.TrainConfig())
trainer.prepare_data(dataset, test_size=0.2, seed=42)
trainer.fit()
```

## Set-valued input, not a design matrix

A sample's composition is a variable-length set: this plot holds nine species, the
next holds sixty, and the identities differ. RESOLVE takes that set directly. Entity
effects are pooled linearly, weighted by abundance, before the encoder mixes them, so
each entity contributes additively to the latent signal and the pooling stays readable
after training.

Five ways to encode the set are available, selected on the dataset config:

| Mode | What it encodes |
|------|-----------------|
| `Hash` | Feature hashing over the full entity list, fixed width regardless of vocabulary |
| `Embed` | Learned embeddings for the dominant entities |
| `Sparse` | Explicit abundance vector over the known vocabulary |
| `RankPool` | Shared entity/genus/family tables, abundance-weighted mean pooling |
| `Transformer` | Additive tokens with self-attention and attention or CLS pooling |

```python
cfg = rc.DatasetConfig()
cfg.species_encoding = rc.SpeciesEncodingMode.RankPool
cfg.pool_weighting   = rc.PoolWeighting.Log1p
```

Above the entity encoder, the sample encoder itself is swappable: `MLP`,
`FTTransformer`, `TabNet`, `SAINT`, `ExcelFormer`, `TraitNet`, and graph encoders
(`GNN`, `HeterogeneousGNN`) with spatial, taxonomic, or co-occurrence graph
construction.

## More than the entity list

Coordinates, numeric covariates, and string covariates join the composition in the
same encoder. Taxonomy gives an entity a fallback identity, so a species the model
has never seen still lands near its genus and family:

```python
roles.latitude     = "Latitude"
roles.longitude    = "Longitude"
roles.genus        = "Genus"
roles.family       = "Family"
roles.covariates   = ["elevation", "slope"]
roles.categoricals = ["bedrock", "country"]
```

String columns are factorized at load time into their own embedding tables, and the
vocabulary travels in the checkpoint so raw CSVs score correctly at inference.
An entity the model never saw is encoded on the reserved unknown row rather than
dropped, so the sample still gets a prediction. How novel the sample is travels
with it: the share of its abundance coming from entities outside the training
vocabulary, and how many such entities it holds, are both encoder inputs.

## Reading the model after it trains

The held-out fold stays reachable from the trainer, with per-sample output for both
task types and the fold indices to tie any of it back to your own table:

```python
resid = trainer.compute_residuals("Area")             # per-sample residuals, skew, kurtosis
cls   = trainer.compute_classification_predictions("Habitat")
cls.predicted_classes, cls.probabilities, cls.actuals

trainer.test_plot_ids()                                # which samples were held out
trainer.compute_calibration("Habitat")                 # reliability bins
trainer.cross_validate(n_folds=5)                      # or cross_validate_spatial(...)
```

Latent representations come off a fitted predictor, which is where the encoder earns
its keep outside prediction: cluster the samples, project them, or feed them to a
downstream model.

```python
pred = rc.Predictor.load("model.pt", device="cpu")
out  = pred.predict_dataset(dataset, return_latent=True)
out.latent                                             # one row per sample
pred.get_species_embeddings()                          # the learned entity space
```

## Learning before the labels arrive

Labelled samples are usually the scarce part. Self-supervised pretext tasks train the
encoder on composition alone, then hand the weights to a supervised fit:

```python
# `continuous` is the encoder's continuous block: coordinates, covariates, the
# unknown-mass columns, and the hash embedding in hash mode, concatenated.
jepa = rc.JEPAPretrainer(model, rc.PretrainConfig())    # joint-embedding prediction
jepa.pretrain(continuous,
              genus_ids=dataset.genus_ids,
              family_ids=dataset.family_ids,
              species_ids=dataset.species_ids)

scarf = rc.SCARFPretrainer(model, rc.PretrainConfig())  # contrastive corruption
vae   = rc.VAEPretrainer(dataset.species_vector.shape[1], rc.VAEConfig())
vae.pretrain(dataset.species_vector)                    # variational reconstruction
```

Masked-entity modelling is available alongside them, masking entity identities so the
pretext task cannot read its own answer.

## From the command line

The engine runs without a language runtime:

```bash
resolve train --header plots.csv --species species.csv \
              --plot-id PlotID --species-id Species --abundance Cover \
              --target Area:regression --target Habitat:classification:9 \
              --encoding rank_pool --cuda --output model.pt

resolve predict --model model.pt --header new_plots.csv --species new_species.csv \
                --output predictions.csv

resolve info --model model.pt
```

## From R

The R package speaks to the same engine through a C ABI, so it needs no libtorch
headers of its own:

```r
library(resolve)

dataset <- resolve.dataset.csv(
  header  = "plots.csv",
  species = "species.csv",
  roles   = list(plot_id = "PlotID", species_id = "Species", abundance = "Cover"),
  targets = list(
    area    = list(column = "Area", task = "regression", transform = "log1p"),
    habitat = list(column = "Habitat", task = "classification", num_classes = 9)
  ),
  config  = list(species_encoding = "rank_pool", hash_dim = 64)
)

trainer <- resolve.train.dataset(dataset, maxEpochs = 200L, device = "cuda",
                                 savePath = "model.pt")

predictor <- resolve.load("model.pt")
preds     <- resolve.predict.dataset(predictor, dataset)
```

## The same two tables, other fields

The shape RESOLVE takes is a sample and the set of things observed in it, which is not
specific to ecology:

| Field | Entities | Sample | Predictions |
|-------|----------|--------|-------------|
| Ecology | Plant species | Vegetation plot | Plot area, habitat type, elevation |
| Medicine | Symptoms, conditions | Patient | Diagnosis, severity, treatment response |
| Retail | Products | Basket | Customer segment, churn risk |
| Genomics | Genes, variants | Sample | Phenotype, disease risk |
| Text | Words, n-grams | Document | Topic, sentiment, author |

Role names carry the ecological vocabulary the package grew up with; they map onto any
of these by pointing `plot_id` and `species_id` at your own columns.

## Running on a GPU

CUDA is used when available, with hash embedding kernels written for it. Training
recovers from a device that runs out of memory on its own: the trainer releases its
caches, halves the batch size, and restarts, down to a floor you set. On a GPU shared
with a desktop, `vram_fraction` leaves headroom for the compositor.

```python
cfg = rc.TrainConfig()
cfg.batch_size     = 16384
cfg.vram_fraction  = 0.80    # sharing the GPU; 1.0 (default) for a dedicated job
```

## Installation

The engine and its Python bindings build from source with CMake and an installed
PyTorch:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve/src/core/python
pip install .
```

R installs without a backend and fetches one afterwards:

```r
install.packages("pak")
pak::pak("gcol33/resolve/r")

library(resolve)
resolve.install_backend()                  # CPU
resolve.install_backend(variant = "cuda")
```

## Documentation

- [Installation](https://gillescolling.com/resolve/tutorials/installation/)
- [Quick Start](https://gillescolling.com/resolve/tutorials/quickstart/)
- [Data Preparation](https://gillescolling.com/resolve/tutorials/data-preparation/)
- [Training Models](https://gillescolling.com/resolve/tutorials/training/)
- [Encoding Modes](https://gillescolling.com/resolve/tutorials/encoding-modes/)
- [Performance Tuning](https://gillescolling.com/resolve/tutorials/performance-tuning/)
- [Making Predictions](https://gillescolling.com/resolve/tutorials/prediction/)
- [Understanding Embeddings](https://gillescolling.com/resolve/tutorials/embeddings/)
- [API Reference](https://gillescolling.com/resolve/api/dataset/)

## Support

> "Software is like sex: it's better when it's free." — Linus Torvalds

I'm a PhD student who builds packages in my free time because I believe good tools
should be free and open. I started these projects for my own work and figured others
might find them useful too.

If this package saved you some time, buying me a coffee is a nice way to say thanks.
It helps with my coffee addiction.

[![Buy Me A Coffee](https://img.shields.io/badge/-Buy%20me%20a%20coffee-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/gcol33)

## License

MIT (see the LICENSE.md file)

## Citation

```bibtex
@software{resolve,
  author = {Colling, Gilles},
  title  = {RESOLVE: Predicting Sample Outcomes from Compositional Data},
  year   = {2026},
  url    = {https://github.com/gcol33/resolve}
}
```
