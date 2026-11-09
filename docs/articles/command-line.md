# The command line

The `resolve` executable trains a model, imputes new plots and describes
a saved model, all from CSV files and without R or Python. It runs the
same RESOLVE engine as the two packages, so a checkpoint written by
`resolve train` loads in R with
[`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md)
and in Python with `Predictor.load()`, and a checkpoint saved from
either package scores with `resolve predict`.

This article runs each command on the survey that ships with the R
package: training with the column flags and targets, covariates and
coordinates, the model settings, the held-out fold and the seed,
cross-validation, checkpoints and resuming, devices, imputation and
reading a checkpoint.

## Installation

``` r

# install.packages("pak")
pak::pak("gcol33/resolve/r")

# The RESOLVE engine library, downloaded once
library(resolveR)
resolve.install_backend()
```

``` bash
pip install resolve-core
```

The packages do not install the command line. It is built from source by
the RESOLVE engine’s CMake project; [Installing the RESOLVE
engine](https://gillescolling.com/resolve/articles/installation.md)
gives the build. The commands below assume `resolve` is on the `PATH`:

``` bash
resolve version
#> resolve 1.1.0
```

`resolve` alone prints every command with its flags, and `--help` after
a command prints that command’s flags only, with their defaults. That
listing is the full reference; this article covers the flags a run
usually needs.

## The two tables

The survey has 150 plots in the Alps. `plots.csv` holds one row per
plot: its identifier, latitude, longitude, elevation in metres and
habitat. `species.csv` holds one row per species recorded on a plot,
with the species’ genus and family and its cover in percent. Copy both
into the directory you run the commands from.

``` r

csv <- system.file("extdata", c("plots.csv", "species.csv"), package = "resolveR")
file.copy(csv, ".")
#> [1] TRUE TRUE
```

``` bash
data=https://raw.githubusercontent.com/gcol33/resolve/master/r/inst/extdata
curl -O "$data/plots.csv" -O "$data/species.csv"
```

``` bash
head -n 3 plots.csv species.csv
#> ==> plots.csv <==
#> "plot_id","lat","lon","elevation","habitat"
#> "A001",46.6042,11.4453,2188,"alpine"
#> "A002",47.0674,10.5684,568,"grassland"
#> 
#> ==> species.csv <==
#> "plot_id","species","genus","family","cover"
#> "A001","Rhododendron ferrugineum","Rhododendron","Ericaceae",54
#> "A001","Carex curvula","Carex","Cyperaceae",51
```

## Training a model

`resolve train` reads the two tables, fits a model and saves it. The
*roles* name the columns: which one identifies the plot, which one the
species, and which optional columns hold the abundance and the taxonomy.
Each *target* is a column to impute, with its task: elevation is a
regression target and habitat a classification target with four classes.

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --genus genus --family family \
  --target elevation:regression --target habitat:classification:4 \
  --max-epochs 100 --output model.pt
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Target: habitat (classification, 4 classes)
#> Using cpu
#> 
#> Loading data...
#>   Encoded 'habitat' as Int64 (auto, 4 classes, 0 null)
#> Loaded 150 plots
#> Species vocabulary size: 26
#> Genera: 23
#> Families: 11
#> 
#> Creating model...
#> 
#> Training...
#> Epoch 0 - Train: 2.46106 Test: 2.18601 | by target (train/test): elevation 0.945828/0.795493, habitat 1.51523/1.39052
#> Epoch 10 - Train: 1.05485 Test: 1.76346 | by target (train/test): elevation 0.339726/0.549583, habitat 0.715122/1.21388
#> Epoch 20 - Train: 0.682665 Test: 1.26906 | by target (train/test): elevation 0.301509/0.445917, habitat 0.381155/0.823145
#> Epoch 30 - Train: 0.475883 Test: 1.24873 | by target (train/test): elevation 0.266623/0.455152, habitat 0.20926/0.793583
#> Epoch 40 - Train: 0.336204 Test: 1.28291 | by target (train/test): elevation 0.22092/0.4841, habitat 0.115285/0.798808
#> Epoch 50 - Train: 0.334085 Test: 1.20692 | by target (train/test): elevation 0.251459/0.402074, habitat 0.0826259/0.804848
#> Epoch 60 - Train: 0.297208 Test: 1.21962 | by target (train/test): elevation 0.232828/0.383811, habitat 0.0643794/0.835807
#> Epoch 70 - Train: 0.248395 Test: 1.2439 | by target (train/test): elevation 0.201501/0.401017, habitat 0.0468942/0.842885
#> Epoch 80 - Train: 0.257394 Test: 1.25083 | by target (train/test): elevation 0.214747/0.395713, habitat 0.042647/0.855119
#> Epoch 90 - Train: 0.208502 Test: 1.24722 | by target (train/test): elevation 0.175122/0.3771, habitat 0.0333792/0.870123
#> 
#> ================
#> Training complete!
#> Best epoch: 50
#> Training time: 2.09139s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   elevation:
#>     band_25: 0.433333
#>     band_10: 0.333333
#>     mae: 260.756
#>     rmse: 301.784
#>     r2: 0.720619
#>     smape: 0.278262
#>     band_50: 0.866667
#>   habitat:
#>     recall_1: 0.818182
#>     accuracy: 0.8
#>     f1_2: 0.842105
#>     macro_f1: 0.796024
#>     precision_0: 0.75
#>     weighted_f1: 0.793901
#>     f1_0: 0.857143
#>     recall_0: 1
#>     precision_1: 0.818182
#>     f1_1: 0.818182
#>     precision_2: 0.8
#>     recall_2: 0.888889
#>     precision_3: 0.8
#>     recall_3: 0.571429
#>     f1_3: 0.666667
#> 
#> Saving model to: model.pt
#> Done!
```

The run prints the targets it read, what the loader found in the tables
and a line every ten epochs with the training and held-out loss, in
total and per target. It ends with the best epoch, the time it took and
the metrics of the best epoch on the held-out plots, the 20 % of plots
it did not train on. For elevation, `mae` and `rmse` are in metres, `r2`
is the share of the variance the imputed values explain and `band_10`,
`band_25` and `band_50` are the shares of plots imputed within 10, 25
and 50 % of their elevation. For habitat, `accuracy` is the share of
plots assigned their recorded habitat, followed by the precision, recall
and F1 score of each class code and their macro and weighted means. The
metrics are printed in no fixed order.

`--output` names the checkpoint, and a JSON file beside it describes the
run in plain text:

``` bash
ls model.*
#> model.json
#> model.pt
```

## Columns

Each role has its own flag. A column flag takes the column’s name in the
table.

| Flag | Column | Default |
|----|----|----|
| `--plot-id` | Plot identifier, in both tables | `plot_id` |
| `--species-id` | Species name, in the species table | `species_id` |
| `--abundance` | Abundance of the species on the plot, such as cover | not read |
| `--genus`, `--family` | Genus and family of the species | not read |
| `--lat`, `--lon` | Coordinates of the plot, in the header table | not read |
| `--covariate` | A numeric plot-level column; repeat the flag for each one | none |
| `--categorical` | A plot-level column of labels; repeat the flag for each one | none |

The survey’s plot column is already called `plot_id`, so the run above
leaves out `--plot-id`. A column the loader cannot find is an error that
names the column:

``` bash
resolve train --header plots.csv --species species.csv \
  --target elevation --output model.pt
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Error loading data: species_id column 'species_id' not found in the species table
```

Without `--header`, the species table is read alone, and its rows also
carry the plot’s targets. Covariates and categorical columns are
plot-level columns, so they need the header table.

## Targets

`--target` takes a column and its task, separated by colons:
`COL:regression` for a continuous target, `COL:classification:N` for a
target with `N` classes. A bare `COL` is a regression target. The fields
after the task are modifiers in any order:

- **`log1p`** fits a regression target on `log(1 + y)` and returns
  imputed values in the column’s own units.
- **`mask`** keeps a plot that has no label for this target: it trains
  the other targets only. The default, `drop`, leaves the plot out of
  the run.
- **An integer** sets the number of classes of a classification target.

`resolve train --help` gives the same grammar:

``` bash
resolve train --help | grep -A 8 -e "--target SPEC"
#>   --target SPEC               Target column and type. Repeat per target:
#>                                 COL:regression          continuous target
#>                                 COL:regression:log1p    fit on log1p(y),
#>                                                         predictions inverted
#>                                 COL:classification:N    N-class target
#>                               A bare COL means COL:regression. Append :mask to
#>                               keep plots that lack this label: they train the
#>                               other targets only (default :drop leaves them out).
#>   --encoding V                Species encoding: hash, embed, sparse, rank_pool, transformer (default: hash)
```

A classification target needs its class count:

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --target habitat:classification --output model.pt
#> RESOLVE Training
#> ================
#> Error: --target habitat:classification: classification needs a class count >= 2, e.g. habitat:classification:9
```

A label column may hold text, as `habitat` does: the loader codes the
labels as integers, and the checkpoint keeps the labels, so imputed
values come back as habitat names.

## Covariates and coordinates

Coordinates and covariates enter the same encoder as the species. This
run imputes habitat from the species, the plot’s coordinates and its
elevation as a covariate. `grep -v` leaves out the epoch lines.

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --genus genus --family family \
  --lat lat --lon lon --covariate elevation \
  --target habitat:classification:4 \
  --max-epochs 100 --output habitat.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Covariates: elevation
#> Target: habitat (classification, 4 classes)
#> Using cpu
#> 
#> Loading data...
#>   Encoded 'habitat' as Int64 (auto, 4 classes, 0 null)
#> Loaded 150 plots
#> Species vocabulary size: 26
#> Genera: 23
#> Families: 11
#> 
#> Creating model...
#> 
#> Training...
#> Early stopping at epoch 69
#> 
#> ================
#> Training complete!
#> Best epoch: 19
#> Training time: 1.43249s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   habitat:
#>     recall_1: 0.909091
#>     accuracy: 0.833333
#>     f1_2: 0.888889
#>     macro_f1: 0.820566
#>     precision_0: 0.75
#>     weighted_f1: 0.826777
#>     f1_0: 0.857143
#>     recall_0: 1
#>     precision_1: 0.833333
#>     f1_1: 0.869565
#>     precision_2: 0.888889
#>     recall_2: 0.888889
#>     precision_3: 0.8
#>     recall_3: 0.571429
#>     f1_3: 0.666667
#> 
#> Saving model to: habitat.pt
#> Done!
```

A `--categorical` column is coded at load time and gets its own
embedding table, for example `--categorical bedrock` for a header column
of rock types. A column cannot be both a covariate and a categorical
column. A covariate or coordinate cell left empty is filled with the
training plots’ mean and marked in a column of its own;
`--missing-values zero` fills it with zero instead.

## Model and encoding settings

Every field of the loader, model and training configurations is a flag
of `resolve train`, named after the field: `--encoding` sets the species
encoding, `--hidden-dims` the widths of the encoder’s layers, `--lr` the
learning rate, and so on. [Encoding the species
set](https://gillescolling.com/resolve/articles/encoding.md) describes
the encodings and [Training and reading the
fit](https://gillescolling.com/resolve/articles/training.md) the
training settings. This run encodes the species by rank pooling, which
pools learned species, genus and family embeddings by a weighted mean:

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --genus genus --family family \
  --target elevation --encoding rank_pool --pool-weighting log1p \
  --max-epochs 100 --output pooled.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Loaded 150 plots
#> Species vocabulary size: 26
#> Genera: 23
#> Families: 11
#> 
#> Creating model...
#> 
#> Training...
#> 
#> ================
#> Training complete!
#> Best epoch: 99
#> Training time: 2.12728s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   elevation:
#>     band_25: 0.733333
#>     band_10: 0.4
#>     mae: 170.535
#>     rmse: 208.097
#>     r2: 0.867158
#>     smape: 0.183215
#>     band_50: 0.933333
#> 
#> Saving model to: pooled.pt
#> Done!
```

A flag whose field the run does not read is an error, so a setting
cannot be dropped without notice. The transformer’s token width, for
example, applies to the transformer encoding only:

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --target elevation --d-model 64 --output model.pt
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Error: --d-model applies only with --encoding transformer
```

A flag that is not declared at all, or a number that does not parse
whole, is refused the same way.

## The held-out fold and the seed

By default a run holds out 20 % of the plots, drawn at random, and stops
when the loss on them has not improved for `--patience` epochs (50 by
default). `--test-size` sets the fraction.

`--seed` (42 by default) seeds the starting weights, the held-out draw,
the order the training plots are drawn in and the cross-validation
folds. On the CPU, two runs with the same flags and seed train the same
model, so their imputed values match:

``` bash
for run in a b; do
  resolve train --header plots.csv --species species.csv \
    --species-id species --abundance cover --target elevation \
    --max-epochs 100 --seed 7 --output seed_$run.pt > /dev/null
  resolve predict --model seed_$run.pt --header plots.csv \
    --species species.csv --species-id species --abundance cover \
    --output seed_$run.csv > /dev/null
done
cmp seed_a.csv seed_b.csv && echo "same predictions"
#> same predictions
```

On a GPU, also pass `--no-cudnn-benchmark`: it turns off the cuDNN
autotuner, which is on by default and can make two runs differ.

`--stopping-plots` holds out the plots a file names instead, one
identifier per line: a spatial block, another survey, or the fold a
published protocol selects its epoch on. The best epoch and the final
metrics are then read on exactly these plots. Here the held-out plots
are those north of 47° latitude:

``` bash
awk -F, 'NR > 1 && $2 > 47 { gsub(/"/, "", $1); print $1 }' \
  plots.csv > north.txt
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --target elevation \
  --stopping-plots north.txt --max-epochs 100 \
  --output north.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Loaded 150 plots
#> Species vocabulary size: 26
#> 
#> Creating model...
#> Stopping fold: 43 plots from north.txt
#> 
#> Training...
#> Early stopping at epoch 91
#> 
#> ================
#> Training complete!
#> Best epoch: 41
#> Training time: 1.88025s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   elevation:
#>     band_25: 0.837209
#>     band_10: 0.604651
#>     mae: 164.553
#>     rmse: 216.001
#>     r2: 0.892192
#>     smape: 0.141527
#>     band_50: 0.930233
#> 
#> Saving model to: north.pt
#> Done!
```

`--stopping-plots` and `--test-size` exclude each other.

To refit on every plot, fix the length of the run in advance and hold
nothing out: `--fixed-epochs N --test-size 0` trains exactly `N` epochs
of the `--max-epochs` schedule and keeps the final weights. With nothing
held out, the run reports no metrics.

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --target elevation \
  --max-epochs 100 --fixed-epochs 50 --test-size 0 \
  --output refit.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Loaded 150 plots
#> Species vocabulary size: 26
#> 
#> Creating model...
#> 
#> Training...
#> 
#> ================
#> Training complete!
#> Best epoch: 49
#> Training time: 0.777753s
#> Effective batch size: 4096
#> 
#> Final metrics:
#> 
#> Saving model to: refit.pt
#> Done!
```

## Cross-validation

`--cv-folds N` runs an `N`-fold cross-validation before the final fit
and prints each fold’s metrics with their mean and standard deviation
across folds. Each fold starts from the untrained weights, and the final
fit afterwards is the same run as one without the flag. `--cv-spatial`
makes the folds blocks of `--cv-lat-size` by `--cv-lon-size` degrees,
which needs the coordinates; `--cv-balance` assigns the blocks to folds
so the folds hold similar numbers of plots.

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --lat lat --lon lon \
  --target elevation --max-epochs 100 \
  --cv-folds 3 --cv-spatial --cv-lat-size 0.25 --cv-lon-size 0.25 \
  --output cv.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Loaded 150 plots
#> Species vocabulary size: 26
#> 
#> Creating model...
#> 
#> Cross-validating...
#> Spatial block CV: 3 folds, block_size=0.250000x0.250000 deg
#> Cross-validation fold 1/3
#>   Train: 104 plots, Test: 46 plots
#> Early stopping at epoch 92
#> Cross-validation fold 2/3
#>   Train: 99 plots, Test: 51 plots
#> Early stopping at epoch 74
#> Cross-validation fold 3/3
#>   Train: 97 plots, Test: 53 plots
#> Spatial CV complete (3 folds, 5.371973s)
#> 
#> Cross-validation (3 folds, 5.37197s):
#>   Fold 1 (best epoch 42):
#>     elevation:
#>       mae: 179.673
#>       band_10: 0.630435
#>       band_25: 0.847826
#>       rmse: 302.294
#>       r2: 0.751443
#>       smape: 0.146256
#>       band_50: 0.934783
#>   Fold 2 (best epoch 24):
#>     elevation:
#>       band_25: 0.705882
#>       band_10: 0.490196
#>       mae: 227.528
#>       rmse: 311.261
#>       r2: 0.779468
#>       smape: 0.203085
#>       band_50: 0.823529
#>   Fold 3 (best epoch 54):
#>     elevation:
#>       mae: 219.156
#>       band_10: 0.490566
#>       band_25: 0.773585
#>       rmse: 307.56
#>       r2: 0.770781
#>       smape: 0.18207
#>       band_50: 0.90566
#>   Across folds (mean +/- std):
#>     elevation:
#>       band_25: 0.775764 +/- 0.0579688
#>       band_10: 0.537066 +/- 0.0660221
#>       mae: 208.785 +/- 20.8677
#>       rmse: 307.039 +/- 3.6792
#>       r2: 0.767231 +/- 0.0117133
#>       smape: 0.177137 +/- 0.0234611
#>       band_50: 0.887991 +/- 0.0471061
#> 
#> Training...
#> 
#> ================
#> Training complete!
#> Best epoch: 73
#> Training time: 1.99067s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   elevation:
#>     band_25: 0.6
#>     band_10: 0.366667
#>     mae: 267.264
#>     rmse: 359.022
#>     r2: 0.604591
#>     smape: 0.253193
#>     band_50: 0.8
#> 
#> Saving model to: cv.pt
#> Done!
```

Under each fold, the `Train:` and `Test:` counts give how many plots it
trained and tested on; with spatial blocks the folds differ in size.

## Checkpoints and resuming

`--checkpoint-dir` makes the run write its best model and a progress
file into a directory as it trains, and `--checkpoint-every N` adds a
checkpoint every `N` epochs:

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --target elevation \
  --max-epochs 100 --checkpoint-dir ckpt --checkpoint-every 25 \
  --output model_ckpt.pt > /dev/null
ls ckpt
#> best.pt
#> checkpoint.json
#> checkpoint.pt
#> checkpoint_25.pt
#> checkpoint_50.pt
#> progress.json
```

`--resume` continues the run that wrote a checkpoint at the epoch after
it. Pass the run’s own data, model and training flags and its seed, so
the held-out plots and the settings are the ones it trained with:

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --target elevation \
  --max-epochs 100 --checkpoint-dir ckpt --checkpoint-every 25 \
  --resume ckpt/checkpoint_25.pt --output model_ckpt.pt | grep -v "^Epoch"
#> RESOLVE Training
#> ================
#> Target: elevation (regression)
#> Using cpu
#> 
#> Loading data...
#> Loaded 150 plots
#> Species vocabulary size: 26
#> 
#> Creating model...
#> 
#> Resuming from ckpt/checkpoint_25.pt...
#> Early stopping at epoch 68
#> Resumed at epoch 25
#> 
#> ================
#> Training complete!
#> Best epoch: 18
#> Training time: 2.42732s
#> Effective batch size: 4096
#> 
#> Final metrics:
#>   elevation:
#>     mae: 274.867
#>     band_10: 0.366667
#>     band_25: 0.6
#>     rmse: 364.646
#>     r2: 0.592106
#>     smape: 0.2687
#>     band_50: 0.8
#> 
#> Saving model to: model_ckpt.pt
#> Done!
```

`--resume` and `--cv-folds` exclude each other.

## Devices

`--device` selects where the run trains: `cpu` (the default), `cuda` for
the current GPU or `cuda:N` for GPU `N`. A CUDA device the machine does
not have is an error; the run never falls back to the CPU on its own.
`resolve predict` takes the same flag and also defaults to the CPU.

``` bash
resolve train --header plots.csv --species species.csv \
  --species-id species --abundance cover --target elevation \
  --device cuda --vram-fraction 0.8 --output model.pt
```

On a GPU, a batch that does not fit in memory halves the batch size and
restarts the run, down to `--batch-size-floor`. `--vram-fraction` caps
the share of GPU memory the run may use, which leaves room for a desktop
on the same card; it takes effect in a build with the RESOLVE engine’s
CUDA kernels. `--amp` turns on mixed precision. [Large data and
GPUs](https://gillescolling.com/resolve/articles/performance.md) covers
these settings.

## Imputing

`resolve predict` scores a header and a species table with a saved model
and writes one row per plot. Here the model imputes the survey’s own
plots again.

``` bash
resolve predict --model model.pt --header plots.csv --species species.csv \
  --species-id species --abundance cover --genus genus --family family \
  --output predictions.csv
head -n 4 predictions.csv
#> RESOLVE Prediction
#> ==================
#> Loading model from: model.pt
#> Using cpu
#> Loading data...
#> Loaded 150 plots
#> Making predictions (batch_size=4096)...
#> Writing predictions to: predictions.csv
#> Done! Wrote 150 predictions.
#> Classification targets are written as two columns: '<target>' (class label) and '<target>_code' (integer code).
#> plot_id,elevation,habitat,habitat_code
#> A001,2247.743,alpine,0
#> A002,1002.3265,grassland,2
#> A003,2104.989,wetland,3
```

The first column is the plot identifier, followed by one column per
target. A classification target gets two columns: `habitat` holds the
class label and `habitat_code` the integer code the model imputed.
`--probabilities` adds one column per class with the probability behind
the code; the columns of one target sum to one in every row:

``` bash
resolve predict --model model.pt --header plots.csv --species species.csv \
  --species-id species --abundance cover --genus genus --family family \
  --probabilities --output probabilities.csv > /dev/null
head -n 4 probabilities.csv
#> plot_id,elevation,habitat,habitat_code,habitat_prob_alpine,habitat_prob_forest,habitat_prob_grassland,habitat_prob_wetland
#> A001,2247.743,alpine,0,0.95639765,0.010085367,0.016828658,0.016688375
#> A002,1002.3265,grassland,2,0.0067752292,0.008118316,0.9660513,0.01905505
#> A003,2104.989,wetland,3,0.030167382,0.024352005,0.037282534,0.9081981
```

The checkpoint carries the species, genus and family vocabularies of its
training data, so the new tables are encoded against the same species.
It also records the covariate and categorical columns, so a covariate
model imputes without `--covariate` flags; pass them only to rename the
columns, one flag per column the model was trained on. The abundance,
coordinate and taxonomy columns are read only when their flags are
given, so pass `resolve predict` the same flags for them as
`resolve train`.

`--predict-batch-size` sets how many plots go through the model at once
(4096 by default). The retrieval flags blend each imputed value with the
labels of its nearest plots from a reference set, in the model’s latent
representation; [Imputing new
plots](https://gillescolling.com/resolve/articles/prediction.md)
describes them.

A *model suite* is a directory of several trained models with a
`manifest.json` that lists them. `resolve predict --suite DIR` scores
every target of the suite, combines its members’ imputed values and
reports how far they agree and how much of each plot’s species list the
target’s vocabulary recognises. `--target` limits the output to some
targets, `--members` adds every member’s own imputed value and
`--no-verify` skips checking each member file against the size and
SHA-256 checksum in the manifest. `resolve info --suite DIR` prints the
manifest and checks every member. [Imputing new
plots](https://gillescolling.com/resolve/articles/prediction.md) covers
suites.

## Reading a checkpoint

`resolve info` prints what a checkpoint holds: how the run split the
plots, the schema of the data it was trained on, the model, loader and
training settings, and the final metrics. The first part reads:

``` bash
resolve info --model north.pt | head -n 40
#> RESOLVE Model Information
#> =========================
#> Model: north.pt
#> 
#> Run:
#>   resolve_version: 1.1.0
#>   created_at: 2026-11-09T22:48:25Z
#>   completed_at: 2026-11-09T22:48:27Z
#>   train_time_seconds: 1.880247
#>   n_plots_train: 107
#>   n_plots_test: 43
#>   hold_out: explicit
#>   test_size: 0
#>   split_seed: 42
#>   best_epoch: 41
#>   total_epochs: 92
#> 
#> Schema:
#>   n_plots: 150
#>   n_species: 25
#>   n_species_vocab: 26
#>   has_coordinates: no
#>   has_abundance: yes
#>   has_taxonomy: no
#>   n_genera: 0
#>   n_families: 0
#>   covariates: []
#>   categoricals: []
#>   categorical_embed_dim: 8
#>   targets:
#>     - name: elevation
#>       task: regression
#>       transform: none
#>       num_classes: 0
#>       weight: 1
#>       missing: drop
#>       class_names: []
#> 
#> Model Configuration:
#>   Model species_encoding: hash
```

`hold_out` is `explicit` for a run with `--stopping-plots` and `random`
for one with `--test-size`. The schema lists each target with its task,
its transform, its missing-label policy and, for a classification
target, its class labels. The listing continues with every setting of
the run and ends with the number of parameters.

The checkpoint loads in R and in Python like one saved from a package:

``` r

predictor <- resolve.load("model.pt")
dataset <- resolve.dataset.csv(
  header = "plots.csv",
  species = "species.csv",
  roles = list(plot_id = "plot_id", species_id = "species",
               abundance = "cover", genus = "genus", family = "family"),
  vocabs = predictor$vocabs(),
  config = predictor$dataset_config()
)
preds <- resolve.predict.dataset(predictor, dataset)
head(round(preds$predictions$elevation))
#> [1] 2248 1002 2105 2319 2004 1950
```

``` python
import resolve_core as rc

predictor = rc.Predictor.load("model.pt")
roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"

dataset = rc.ResolveDataset.from_csv_with_vocabs(
    "plots.csv", "species.csv", roles, [],
    predictor.external_vocabs, predictor.dataset_config)
preds = predictor.predict_dataset(dataset)
print(preds.predictions["elevation"][:6].round())
```

The imputed elevations match the `elevation` column of `predictions.csv`
above.
