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

## 1. Your plots

Species table

One row per species record: plot ID, species name (World Flora Online),
and optionally cover (%), genus and family. Columns are recognised by
name, Darwin Core terms included; change any choice below.

Header table (optional)

One row per plot: plot ID, latitude and longitude (WGS84), and year and
month or a date. Each variable uses the coordinates and date where the
tables give them; section 2 names what each is predicted from. A
variable the header records on some plots (survey year, coordinates,
altitude, slope, plot area) has its gaps fitted on the plots that record
it.

Misspelled species names: read as the nearest known name within no edits
(names as written) 1 edit 2 edits (can also rename species the models
never saw)

Use the example data again

## 2. Models

Predict

Workers Weights half precision (half the download) full precision (the
engine's own)

Terrain correction of altitude and slope

## 3. Predictions

&nbsp;

Names are matched exactly against the World Flora Online names the
models were trained on; check the spelling and the accepted name of
each.

The models know these species by the name the European Vegetation
Archive resolved its records to. Each name is matched through EVA's own
records, the synonymy of a taxonomic backbone, or its spelling, named
beside it.

Download all predictions (CSV)

Download map (SVG)

Habitat combines its five seeds by majority vote (agreement is the share
of seeds naming the class); numeric variables by their mean and standard
deviation across seeds; location by the spherical mean and the median
distance of the seeds from it. "Species recognised" is the share of the
plot's cover held by species the models know. A terrain-corrected
altitude or slope weighs the prediction against the elevation model
(GLO-30) at the plot and over the square kilometre around it, by the
plot's location uncertainty, the seeds' spread and how many of its
species the models know; the elevation model's own values stand beside
it.

Terrain values produced using Copernicus WorldDEM-30 © DLR e.V.
2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under
COPERNICUS by the European Union and ESA; all rights reserved. The
organisations in charge of the Copernicus programme by law or by
delegation do not incur any liability for any use of the Copernicus
WorldDEM-30.
