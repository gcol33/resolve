# RESOLVE Examples

Notebooks and scripts demonstrating RESOLVE usage.

## Contents

- **getting_started.ipynb** — builds a small dataset, trains a multi-target
  model, reads the held-out fold, saves a checkpoint, and scores a new survey
  through the training vocabularies.

## Running the examples

The examples use `resolve_core`, the Python binding over the C++ engine. It
builds from source against the PyTorch you already have installed:

```bash
git clone https://github.com/gcol33/resolve.git
cd resolve/src/core/python
pip install .
```

Then, for the notebook:

```bash
pip install jupyter matplotlib
jupyter notebook examples/getting_started.ipynb
```

See [Installation](https://gillescolling.com/resolve/tutorials/installation/)
for the R package and the command line.

## Data format

RESOLVE reads two tables, as CSV paths or as pandas DataFrames:

1. **Header table**: one row per plot
   - Required: plot ID column
   - Optional: coordinates, numeric covariates, string covariates, target columns

2. **Species table**: one row per species occurrence
   - Required: species ID and plot ID columns
   - Optional: abundance, genus, family

A single long table works too, with the targets carried inline; see
`ResolveDataset.from_species_csv`.
