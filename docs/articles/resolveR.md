# Imputing plot attributes from species lists

A vegetation plot’s species list says a lot about the place it was
recorded: alpine species grow high up, wetland species where the ground
stays wet. RESOLVE learns this from data. It encodes the species
observed in each sample into one latent representation and reads several
attributes off it, here the elevation and the habitat of a plot.

This article runs the whole path on a small survey that ships with the R
package: loading two tables, training a model, reading how well it
imputes plots it did not train on, and imputing new plots.

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

[Installing the RESOLVE
engine](https://gillescolling.com/resolve/articles/installation.html)
covers the GPU build, building from source and the command line.

## Two tables

The survey has 150 plots in the Alps. `plots` holds one row per plot:
its identifier, coordinates, elevation in metres and habitat. `species`
holds one row per species recorded on a plot, with the species’ genus
and family and its cover in percent. The plot identifier links the two
tables.

``` r

library(resolveR)

plots <- read.csv(system.file("extdata", "plots.csv", package = "resolveR"))
species <- read.csv(system.file("extdata", "species.csv", package = "resolveR"))
head(plots, 3)
#>   plot_id     lat     lon elevation   habitat
#> 1    A001 46.6042 11.4453      2188    alpine
#> 2    A002 47.0674 10.5684       568 grassland
#> 3    A003 46.6674 11.3040      2129   wetland
head(species, 3)
#>   plot_id                  species        genus     family cover
#> 1    A001 Rhododendron ferrugineum Rhododendron  Ericaceae    54
#> 2    A001            Carex curvula        Carex Cyperaceae    51
#> 3    A002          Quercus petraea      Quercus   Fagaceae    16
```

``` python
import pandas as pd
import resolve_core as rc

data = "https://raw.githubusercontent.com/gcol33/resolve/master/r/inst/extdata/"
plots = pd.read_csv(data + "plots.csv")
species = pd.read_csv(data + "species.csv")
print(plots.head(3))
#>   plot_id      lat      lon  elevation    habitat
#> 0    A001  46.6042  11.4453       2188     alpine
#> 1    A002  47.0674  10.5684        568  grassland
#> 2    A003  46.6674  11.3040       2129    wetland
print(species.head(3))
#>   plot_id                   species         genus      family  cover
#> 0    A001  Rhododendron ferrugineum  Rhododendron   Ericaceae     54
#> 1    A001             Carex curvula         Carex  Cyperaceae     51
#> 2    A002           Quercus petraea       Quercus    Fagaceae     16
```

## Loading a dataset

The *roles* name the columns: which one identifies the plot, which one
the species, and which optional columns hold the abundance, the taxonomy
and the coordinates. Each *target* is a column to impute, with its task.
Elevation is a regression target and habitat a classification target.

``` r

dataset <- resolve.dataset.frame(
  header = plots,
  species = species,
  roles = list(plot_id = "plot_id", species_id = "species",
               abundance = "cover", genus = "genus", family = "family"),
  targets = list(
    elevation = list(column = "elevation", task = "regression"),
    habitat = list(column = "habitat", task = "classification")
  )
)
```

``` python
roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"

targets = [rc.TargetSpec.regression("elevation"),
           rc.TargetSpec.classification("habitat", 4)]
dataset = rc.ResolveDataset.from_pandas(plots, species, roles, targets)
```

The coordinates stay out of the roles here, so the imputed values come
from the species alone.

## Training

Training holds out 20 % of the plots, drawn at random with a fixed seed,
and stops when the loss on them no longer improves. The metrics are
computed on these held-out plots.

``` r

resolve.manual_seed(42)
fit <- resolve.train.dataset(dataset, verbose = FALSE)
metrics <- fit$result$final_metrics
round(unlist(metrics$elevation[c("mae", "rmse", "r2", "band_25")]), 3)
#>     mae    rmse      r2 band_25 
#> 184.887 263.782   0.835   0.800
round(metrics$habitat$accuracy, 3)
#> [1] 0.8
```

``` python
rc.manual_seed(42)
model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
trainer = rc.Trainer(model, rc.TrainConfig())
trainer.prepare_data(dataset)
result = trainer.fit()
metrics = result.final_metrics
{name: round(metrics["elevation"][name], 3) for name in ["mae", "rmse", "r2", "band_25"]}
#> {'mae': 260.756, 'rmse': 301.784, 'r2': 0.721, 'band_25': 0.433}
round(metrics["habitat"]["accuracy"], 3)
#> 0.8
```

For elevation, `mae` and `rmse` are in metres, `r2` is the share of the
variance the imputed values explain, and `band_25` is the share of plots
imputed within 25 % of their elevation; `band_10` and `band_50` give the
same at 10 and 50 %. For habitat, `accuracy` is the share of plots
assigned their recorded habitat, and the per-class precision, recall and
F1 scores sit beside it in the same list. Training’s own seed fixes the
held-out plots and the order the training plots are drawn in, and
[`resolve.manual_seed()`](https://gillescolling.com/resolve/reference/resolve.manual_seed.md)
(`rc.manual_seed()` in Python) fixes the starting weights, so the same
run on the same machine ends at the same numbers. The R and the Python
run here still end at slightly different numbers, since each reaches the
RESOLVE engine through its own torch build.

## Imputing new plots

A saved model carries the species vocabulary it was trained with, so a
new table is encoded against the same species. Here the model imputes
the survey’s own plots again.

``` r

path <- file.path(tempdir(), "model.pt")
resolve.save(fit$trainer, path)

predictor <- resolve.load(path)
preds <- resolve.predict.dataset(predictor, dataset)
head(data.frame(plot = preds$plot_ids,
                elevation = round(preds$predictions$elevation),
                observed = plots$elevation[match(preds$plot_ids, plots$plot_id)]))
#>   plot elevation observed
#> 1 A001      2245     2188
#> 2 A002       589      568
#> 3 A003      2170     2129
#> 4 A004      2300     2267
#> 5 A005      2031     2145
#> 6 A006      1753     1799
```

``` python
import os, tempfile

path = os.path.join(tempfile.gettempdir(), "model.pt")
trainer.save(path)

predictor = rc.Predictor.load(path)
preds = predictor.predict_dataset(dataset)
out = pd.DataFrame({"plot": preds.plot_ids,
                    "elevation": preds.predictions["elevation"].numpy().round().astype(int)})
out.merge(plots[["plot_id", "elevation"]], left_on="plot", right_on="plot_id",
          suffixes=("", "_observed")).drop(columns="plot_id").head(6)
#>    plot  elevation  elevation_observed
#> 0  A001       2248                2188
#> 1  A002       1002                 568
#> 2  A003       2105                2129
#> 3  A004       2319                2267
#> 4  A005       2004                2145
#> 5  A006       1950                1799
```

A plot with a species the model never saw still gets an imputed value:
the unknown species is encoded through its genus and family where those
are known.

## Next steps

- [Preparing the two
  tables](https://gillescolling.com/resolve/articles/data-preparation.html):
  every role, missing values, targets labelled on some plots and reusing
  a training vocabulary.

- [Encoding the species
  set](https://gillescolling.com/resolve/articles/encoding.html): the
  five encodings and when each fits.

- [Training and reading the
  fit](https://gillescolling.com/resolve/articles/training.html): the
  training settings, the held-out fold, cross-validation and
  checkpoints.

- [Imputing new
  plots](https://gillescolling.com/resolve/articles/prediction.html):
  output formats, class probabilities, new species and model suites.
