# RESOLVE App

Predict a vegetation plot’s EUNIS habitat, altitude, slope, plot area,
survey year and location from the species recorded on it, with the
released models trained on the European Vegetation Archive’s plots. The
models run in your browser: your files are read inside this tab and are
not uploaded anywhere. The page downloads the models and the elevation
model from Zenodo ([models](https://doi.org/10.5281/zenodo.24077609),
[elevation model](https://doi.org/10.5281/zenodo.24075925)), the map
from OpenStreetMap and, for a map download, the country boundaries from
Eurostat GISCO.

![](../logo.svg)![RESOLVE](../logo.svg)

Reading the model list…

Data

Species table: one row per species record, with plot ID, species name
(World Flora Online), and optionally cover (%), genus and family.
Columns are matched by name (Darwin Core names too); change them below
if needed.

Header table (optional): one row per plot, with plot ID, latitude and
longitude (WGS84), and year and month or a date. Where the header
records survey year, coordinates, altitude, slope or plot area on only
some plots, the missing values are estimated from the plots that have
them.

Correct misspelled species names off up to 1 letter off up to 2 letters
off

Use the example data again

Models

Terrain correction of altitude and slope

Predict

Workers

plots

variables predicted

species names not recognised

to predict

Plots

Download map (SVG)

Habitats

Predictions

Download all predictions (CSV)

&nbsp;

&nbsp;

Each variable is predicted by five models, trained in the same way from
different random starting points (seeds), and the page combines their
five predictions: habitat by majority vote (seed agreement is the share
of the five naming the habitat), numeric variables by their mean and
standard deviation, and location by the spherical mean and the median
distance of the five from it.

Altitude and slope can also be corrected with an elevation model
(Copernicus GLO-30). The correction weighs the prediction against the
elevation model's value at the plot and over the square kilometre around
it, depending on the plot's location uncertainty, the spread of the five
predictions and how many of the plot's species the models know. The
table gives the elevation model's own values beside the corrected ones.

Terrain values produced using Copernicus WorldDEM-30 © DLR e.V.
2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved. The
organisations in charge of the Copernicus programme by law or by
delegation do not incur any liability for any use of the Copernicus
WorldDEM-30.

runs on [ONNX Runtime Web](https://onnxruntime.ai/) ·
