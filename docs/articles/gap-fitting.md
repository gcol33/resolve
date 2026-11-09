# Filling a dataset's gaps

A dataset rarely lacks a variable on every plot. Most plots carry a
survey year, coordinates or an elevation, and some do not. The plots
that carry it are labelled examples from the same surveys, recorded by
the same people, as the plots that lack it. *Gap fitting* reads them: on
the plots where the value is recorded (the *complete plots*) it chooses
how to combine the model’s imputed value with the dataset’s own
neighbours, and applies that choice to the plots where it is missing
(the *gaps*). The model’s imputed value is never replaced; the fitted
value comes back beside it, with a report of what was chosen and why.

This article covers the candidates gap fitting chooses among, how the
choice is tested at the distance the gaps actually are, fitting a single
model’s imputed values, reading the report, model suites and the command
line, the settings, and the gaps the dataset cannot inform.

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

## What gap fitting chooses among

Seven candidates impute a gap, from the dataset alone, from the model
alone, or from both:

- `local_constant`: the complete plots’ mean (their median on a log
  scale, their mean direction for a bearing, their geographic median for
  a location);
- `resolve`: the model’s imputed value as it stands;
- `offset_slope`: the model’s imputed value calibrated on the complete
  plots, `y = mean(y) + b (raw - mean(raw))`, with `b` fitted by least
  squares; `b = 1` shifts the imputed value by its mean error there,
  `b = 0` is the local constant;
- `knn_species`: the median recorded value of the `k` complete plots
  most similar in species composition (the Ochiai similarity of their
  species lists);
- `knn_spatial`: the median recorded value of the `k` nearest complete
  plots by great-circle distance;
- `stack_species` and `stack_species_spatial`: the model’s imputed value
  and those neighbour medians combined by least squares on the complete
  plots.

`k` is chosen from 1, 3, 5, 10, 20 and 50 by how well the neighbours
impute the complete plots.

## Testing the choice at the gaps’ distance

A complete plot’s nearest other complete plots are as near as complete
plots are to one another. A gap can be much farther from them: a region
nobody georeferenced, or the oldest surveys. Judging the neighbours by
how well they impute a complete plot from its own nearest plots would
trust them more than they deserve at the gaps.

Gap fitting therefore measures each gap’s distance to its nearest
complete plot on every axis it can be measured on: species similarity,
km where the gaps carry coordinates, and survey years where they carry a
year. Each complete plot is then imputed with the complete plots nearer
to it than a gap is to its nearest complete plot hidden, one gap
distance drawn per complete plot and axis (nearest-neighbour distance
matching, Milà et al. 2022). A complete plot left with no neighbour at
that distance counts as imputed by the local constant, which is all the
dataset says about it there. Each candidate is scored on the complete
plots by the target’s own error: the mean absolute error, on a log scale
that of log1p, the circular error of a bearing, the median distance in
km for a location. The best candidate imputes the gaps.

Gaps with coordinates and gaps without them choose apart: the second
group only among the candidates that read no position.

## The survey and its gaps

The examples use the survey of 150 plots in the Alps that ships with the
R package. A model trained on another survey imputes plots it never saw,
so the first 75 plots train it and the other 75 are the dataset whose
elevation has gaps: every fifth of them lacks its recorded elevation.

``` r

plots <- read.csv(system.file("extdata", "plots.csv", package = "resolveR"))
species <- read.csv(system.file("extdata", "species.csv", package = "resolveR"))

is_training <- seq_len(nrow(plots)) <= 75
train_plots <- plots[is_training, ]
train_species <- species[species$plot_id %in% train_plots$plot_id, ]
my_plots <- plots[!is_training, ]
my_species <- species[species$plot_id %in% my_plots$plot_id, ]

is_gap <- seq_len(nrow(my_plots)) %% 5 == 0
my_plots$elevation_recorded <- ifelse(is_gap, NA, my_plots$elevation)
c(complete = sum(!is_gap), gaps = sum(is_gap))
#> complete     gaps 
#>       60       15
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

is_training = plots.index < 75
train_plots = plots[is_training]
train_species = species[species.plot_id.isin(train_plots.plot_id)]
my_plots = plots[~is_training].reset_index(drop=True)
my_species = species[species.plot_id.isin(my_plots.plot_id)]

is_gap = (my_plots.index + 1) % 5 == 0
my_plots["elevation_recorded"] = my_plots.elevation.where(~is_gap)
{"complete": int((~is_gap).sum()), "gaps": int(is_gap.sum())}
#> {'complete': 60, 'gaps': 15}
```

The model imputes elevation from the species alone and is saved to a
checkpoint.

``` r

roles <- list(plot_id = "plot_id", species_id = "species",
              abundance = "cover", genus = "genus", family = "family")
targets <- list(elevation = list(column = "elevation", task = "regression"))
dataset <- resolve.dataset.frame(train_plots, train_species,
                                 roles = roles, targets = targets)
path <- file.path(tempdir(), "elevation.pt")
fit <- resolve.train.dataset(dataset, maxEpochs = 100L,
                             savePath = path, verbose = FALSE)
```

``` python
roles = rc.RoleMapping()
roles.plot_id = "plot_id"
roles.species_id = "species"
roles.abundance = "cover"
roles.genus = "genus"
roles.family = "family"
dataset = rc.ResolveDataset.from_pandas(train_plots, train_species, roles,
                                        [rc.TargetSpec.regression("elevation")])

model = rc.ResolveModel(dataset.schema, rc.model_config_for_dataset(dataset))
train_config = rc.TrainConfig()
train_config.max_epochs = 100
trainer = rc.Trainer(model, train_config)
trainer.prepare_data(dataset)
result = trainer.fit()
path = os.path.join(tempfile.gettempdir(), "elevation.pt")
trainer.save(path, trainer.run_metadata)
```

The dataset’s plots are encoded with the model’s vocabularies and
imputed, all of them: the complete plots are what the choice is tested
on.

``` r

predictor <- resolve.load(path)
my_data <- resolve.dataset.frame(my_plots[c("plot_id", "lat", "lon")], my_species,
                                 roles = roles, vocabs = predictor$vocabs(),
                                 config = predictor$dataset_config())
preds <- resolve.predict.dataset(predictor, my_data)
```

``` python
predictor = rc.Predictor.load(path)
header_csv = os.path.join(tempfile.gettempdir(), "my_plots.csv")
species_csv = os.path.join(tempfile.gettempdir(), "my_species.csv")
my_plots[["plot_id", "lat", "lon"]].to_csv(header_csv, index=False)
my_species.to_csv(species_csv, index=False)
my_data = rc.ResolveDataset.from_csv_with_vocabs(
    header_csv, species_csv, roles, [],
    predictor.external_vocabs, predictor.dataset_config)
preds = predictor.predict_dataset(my_data)
```

## Fitting the gaps of a single model

Gap fitting reads the dataset’s own tables: the species records, for
composition over every species they name, whatever the model’s
vocabulary holds; the coordinates, for the spatial neighbours; and the
column holding the recorded value, empty on the gaps. `targets` names
each target to fit and how its values lie, here elevation on its own
scale.

``` r

columns <- list(plot_id = "plot_id", species_id = "species",
                latitude = "lat", longitude = "lon",
                recorded = list(elevation = "elevation_recorded"))
fits <- resolve.fit_prediction_gaps(preds, my_species, my_plots, columns = columns,
                                    targets = list(elevation = list(scale = "linear")))
elevation <- fits$elevation
table(elevation$method)
#> 
#>              recorded stack_species_spatial 
#>                    60                    15
```

``` python
columns = {"plot_id": "plot_id", "species_id": "species",
           "latitude": "lat", "longitude": "lon",
           "recorded": {"elevation": "elevation_recorded"}}
fits = rc.fit_prediction_gaps(preds, my_species, my_plots, columns=columns,
                              targets={"elevation": {"scale": "linear"}})
elevation = fits["elevation"]
pd.Series(elevation.method).value_counts()
#> recorded                 60
#> stack_species_spatial    15
#> Name: count, dtype: int64
```

A complete plot keeps its recorded value (`recorded`); every gap carries
the method that imputed it. The gaps’ recorded elevation is known here,
so the fitted values can be checked against it beside the model’s own
imputed value.

``` r

order <- match(preds$plot_ids, my_plots$plot_id)
truth <- my_plots$elevation[order]
gaps <- is.na(my_plots$elevation_recorded[order])
c(model = mean(abs(preds$predictions$elevation[gaps] - truth[gaps])),
  fitted = mean(abs(elevation$fitted[gaps] - truth[gaps])))
#>    model   fitted 
#> 274.8799 162.6798
```

``` python
order = pd.Index(my_plots.plot_id.astype(str)).get_indexer(list(preds.plot_ids))
truth = my_plots.elevation.to_numpy()[order]
gaps = my_plots.elevation_recorded.isna().to_numpy()[order]
raw = preds.predictions["elevation"].numpy()
fitted = elevation.fitted.numpy()
{"model": float(np.abs(raw[gaps] - truth[gaps]).mean()),
 "fitted": float(np.abs(fitted[gaps] - truth[gaps]).mean())}
#> {'model': 252.3594970703125, 'fitted': 141.37301726696327}
```

## Reading the report

The report says what was chosen and why: the number of complete plots
and gaps, the gaps’ distance to their nearest complete plot on each
axis, the `k` each kind of neighbour chose, and per group of gaps every
candidate’s error on the complete plots at the gaps’ distance, the
method chosen and what it fitted.

``` r

located <- elevation$report$groups$located
located$method
#> [1] "stack_species_spatial"
round(unlist(located$matched_error), 1)
#>        local_constant               resolve          offset_slope 
#>                 559.9                 158.6                 155.9 
#>           knn_species         stack_species           knn_spatial 
#>                 173.0                 135.1                 609.3 
#> stack_species_spatial 
#>                 128.8
elevation$report$axes$km$gap_distance$median
#> [1] 7.379892
```

``` python
located = elevation.report["groups"]["located"]
located["method"]
#> stack_species_spatial
{m: round(e, 1) for m, e in located["matched_error"].items()}
#> {'local_constant': 559.9, 'resolve': 167.7, 'offset_slope': 167.2, 'knn_species': 173.0, 'stack_species': 134.9, 'knn_spatial': 609.3, 'stack_species_spatial': 131.0}
elevation.report["axes"]["km"]["gap_distance"]["median"]
#> 7.379891561267665
```

A chosen least-squares method also reports its coefficients
(`coefficients`, one per term: `resolve`, `species`, `spatial`), the
complete plots it was fitted on (`n_fitted`) and the `k` of each
neighbour term. For a bearing a coefficient is a complex number,
reported as its `modulus` (how much it shrinks the term) and its
`angle_deg` (how far it turns it).

## Model suites and the command line

A suite fits its targets after combining its members. `columns` names
the recorded column of each target to fit (two, latitude and longitude,
for a location) and optionally the survey-year column; the imputation
table then carries `<target>_fitted` and `<target>_fit_method` beside
`<target>`, and each fitted target a `gap_fit` with its report. The
scale follows the target: a bearing is fitted as a direction, a location
in a plane, a target the members learned on log1p on that scale.

``` r

suite <- resolve.load_suite("resolve-suite")
preds <- resolve.predict.suite(
  suite, species = my_species, header = my_plots,
  columns = list(recorded = list(year = "Year", location = c("Latitude", "Longitude")),
                 survey_year = "Year"),
  gapFitting = TRUE)
head(as.data.frame(preds)[c("plot_id", "year", "year_fitted", "year_fit_method")])
preds$targets$year$gap_fit$report$groups
```

``` python
suite = rc.SuitePredictor.load("resolve-suite")
preds = suite.predict(
    my_species, my_plots,
    columns={"recorded": {"year": "Year", "location": ["Latitude", "Longitude"]},
             "survey_year": "Year"},
    gap_fitting=True)
preds.to_pandas()[["plot_id", "year", "year_fitted", "year_fit_method"]].head()
preds.target("year").gap_fit.report["groups"]
```

These lines are shown and not run here. On the command line,
`--fit-gaps` with one `--recorded` per target does the same for
`--suite` and for `--model`, and writes the report beside the imputed
values:

``` bash
resolve predict --suite resolve-suite --header plots.csv --species species.csv \
  --fit-gaps --recorded year=Year --recorded location=Latitude,Longitude \
  --survey-year Year --output predictions.csv
# predictions.csv: <target>_fitted and <target>_fit_method beside each target
# predictions.csv.gap_fitting.json: the report (--gap-report names another path)
```

With `--model`, a target is the prediction head of its name, on a log
scale when the model learned it on log1p;
`--gap-target aspect=circular:360:asp_sin,asp_cos` or
`--gap-target location=geographic:lat,lon` names the scale and the heads
otherwise.

`<target>_fit_method` reads `recorded` on the plots that record the
target, so it also tells which values were imputed. Without gap fitting,
the recorded columns alone (`columns$recorded` in R,
`columns["recorded"]` in Python, `--recorded` without `--fit-gaps`) add
`<target>_recorded` instead: `TRUE` where the plot records the target,
`FALSE` where `<target>` is imputed.

## Settings

Four settings change the choice (`gapFitting` in R, `gap_fitting` in
Python, the `--gap-*` flags on the command line):

- `candidates`: the methods to choose among, by name; the gaps without
  coordinates need at least one that reads no position;
- `k_grid`: the neighbour counts `k` is chosen from;
- `spatial = FALSE` (`--no-spatial`): composition only, no spatial
  neighbours;
- `matching_seed`: the seed of the draw of one gap distance per complete
  plot.

``` r

composition_only <- resolve.fit_prediction_gaps(
  preds, my_species, my_plots, columns = columns,
  targets = list(elevation = list(scale = "linear")),
  gapFitting = list(spatial = FALSE, k_grid = c(1, 3, 5, 10)))
composition_only$elevation$report$groups$unlocated$method
#> [1] "stack_species"
```

``` python
config = rc.GapFittingConfig()
config.spatial = False
config.k_grid = [1, 3, 5, 10]
composition_only = rc.fit_prediction_gaps(preds, my_species, my_plots, columns=columns,
                                          targets={"elevation": {"scale": "linear"}},
                                          gap_fitting=config)
composition_only["elevation"].report["groups"]["unlocated"]["method"]
#> stack_species
```

[`resolve.fit_gaps()`](https://gillescolling.com/resolve/reference/resolve.fit_gaps.md)
(R) and `rc.fit_gaps()` (Python) take the same choice on vectors: the
raw imputed value, the recorded values, a matrix of species and
optionally the coordinates and survey years, for a value imputed
anywhere.

## Gaps the dataset cannot inform

The complete plots show how far off a gap’s candidates are only where
the gaps resemble them. Two kinds of gap do not:

- **Undated plots that are probably older than the dated ones.** When
  the undated plots are the oldest surveys, nothing on the dated plots
  shows how far off they are, and the model’s own imputed value is the
  better estimate: keep `<target>` rather than `<target>_fitted` for
  them. For other targets an old survey is visible through the
  survey-year axis, and gap fitting allows for it.
- **Plots without coordinates that probably come from an area with no
  georeferenced plots.** A location’s gaps have no position, so the test
  hides plots by species and survey year only, and a missing region
  still looks near in species terms. Such plots are better served by the
  model’s imputed value, or by the species stack alone.
