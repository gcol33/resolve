# RESOLVE App

Impute the habitat
([EUNIS](https://www.eea.europa.eu/en/datahub/datahubitem-view/123d0c9a-a6fa-4f2d-b887-5d8e5468faed)),
site metadata (altitude, slope, location) and survey metadata (plot
area, survey year) of vegetation plots from their species composition
and cover-abundance, with the released RESOLVE models trained on the
plots of the European Vegetation Archive (EVA). The models run locally
in your browser; your files are not uploaded. The page downloads the
models and the elevation model from Zenodo
([models](https://doi.org/10.5281/zenodo.24077609), [elevation
model](https://doi.org/10.5281/zenodo.24075925)), the map from
OpenStreetMap and, for a map download, the country boundaries from
Eurostat GISCO.

![](../logo.svg)![RESOLVE](../logo.svg)

Reading the model list…

Data

Species table: one row per species record, with plot ID, species name
(World Flora Online), and optionally cover-abundance (%), genus and
family. Columns are matched by name (Darwin Core names too); change them
below if needed.

Header table (optional): one row per plot, with plot ID, latitude and
longitude (WGS84), and year and month or a date. A header with gaps in
some columns is calibrated on its own plots (opt-out).

Fuzzy match species names off up to 1 edit up to 2 edits up to 3 edits

Use the example data again

Models

Terrain correction of altitude and slope

Calibrate gaps to input data

Impute

Workers

![](data:image/svg+xml;base64,PHN2ZyBpZD0iYnVzeS1wdWxzZSIgdmlld2JveD0iLTYgLTYgMTIgMTIiIGFyaWEtaGlkZGVuPSJ0cnVlIj48cGF0aCBkPSJNMCAtNC42VjQuNk0tNCAtMi4zTDQgMi4zTS00IDIuM0w0IC0yLjMiIC8+PC9zdmc+)
·

plots

variables imputed

species names not recognised

to impute

Plots

Points Grid

Download map (SVG)

‹

›

Imputed values

«

‹

›

»

Download all imputed values (CSV)

&nbsp;

&nbsp;

Each variable is imputed by five models, trained in the same way from
different random starting points (seeds), and the page combines their
five values: habitat by majority vote (seed agreement is the share of
the five naming the habitat), numeric variables by their mean and
standard deviation, and location by the spherical mean and the median
distance of the five from it.

Altitude and slope can also be corrected with an elevation model
(Copernicus GLO-30). The correction weighs the imputed value against the
elevation model's value at the plot and over the square kilometre around
it, depending on the plot's location uncertainty, the spread of the five
values and how many of the plot's species the models know. The table
gives the elevation model's own values beside the corrected ones.

Terrain values produced using Copernicus WorldDEM-30 © DLR e.V.
2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved. The
organisations in charge of the Copernicus programme by law or by
delegation do not incur any liability for any use of the Copernicus
WorldDEM-30.

runs on [ONNX Runtime Web](https://onnxruntime.ai/) ·
