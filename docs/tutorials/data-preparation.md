# Data Preparation

This guide covers how to prepare your data for RESOLVE.

## Data Structure

RESOLVE uses two data files:

### Header File (plots.csv)

One row per plot with plot-level attributes:

```csv
PlotID,Latitude,Longitude,Area,Elevation,Habitat
P001,45.2,6.1,100,1200,forest
P002,46.1,7.2,250,800,grassland
P003,44.8,5.9,50,1500,forest
```

### Species File (species.csv)

One row per species occurrence:

```csv
PlotID,Species,Genus,Family,Cover
P001,Quercus robur,Quercus,Fagaceae,25
P001,Fagus sylvatica,Fagus,Fagaceae,40
P001,Pinus sylvestris,Pinus,Pinaceae,10
P002,Festuca rubra,Festuca,Poaceae,60
P002,Trifolium repens,Trifolium,Fabaceae,15
```

## Required Fields

At minimum, you need:

1. **Plot identifier** in both files (for joining)
2. **Species identifier** in species file
3. **At least one target variable** in header file

## Optional Fields

### Coordinates

Latitude and longitude enable spatial awareness:

```python
roles = {
    "coords_lat": "Latitude",
    "coords_lon": "Longitude",
}
```

If omitted, RESOLVE uses `coord_mode="none"`.

### Abundance

Species abundance values (cover, count, biomass):

```python
roles = {
    "abundance": "Cover",
}
```

If omitted, presence/absence (1.0) is assumed.

### Taxonomy

Genus and family enable learned embeddings:

```python
roles = {
    "taxonomy_genus": "Genus",
    "taxonomy_family": "Family",
}
```

These significantly improve model performance when available.

### Covariates

Additional predictor variables from the header:

```python
roles = {
    "covariates": ["Temperature", "Precipitation", "SoilpH"],
}
```

## Handling Missing Data

### Target Variables

Filter plots with missing target values before loading:

```python
import pandas as pd

header = pd.read_csv("plots.csv")
header = header.dropna(subset=["Area", "Habitat"])
```

### Covariates

RESOLVE does not impute missing covariate values. Either:
1. Filter plots with missing covariates
2. Impute before loading
3. Exclude problematic covariates

## Data Validation

After loading, check the schema:

```python
dataset = resolve.ResolveDataset.from_csv(...)

schema = dataset.schema
print(f"Plots: {schema.n_plots}")
print(f"Species: {schema.n_species}")
print(f"Genera: {schema.n_genera}")
print(f"Families: {schema.n_families}")
print(f"Covariates: {schema.covariate_names}")
print(f"Targets: {list(schema.targets.keys())}")
```

## Loading from DataFrames

You can also load from pandas DataFrames:

```python
import pandas as pd

header = pd.read_csv("plots.csv")
species = pd.read_csv("species.csv")

dataset = resolve.ResolveDataset(
    header=header,
    species=species,
    roles=roles,
    targets=targets,
)
```

## Next Steps

- [Training Models](training.md): Configure and train your model
- [Quick Start](quickstart.md): Complete workflow example
