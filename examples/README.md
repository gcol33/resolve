# RESOLVE Examples

Example notebooks and scripts demonstrating RESOLVE usage.

## Contents

- **getting_started.ipynb** - Quick start tutorial covering data loading, training, and prediction

## Running the Examples

```bash
# Install dependencies
pip install resolve-ml jupyter matplotlib

# Launch Jupyter
jupyter notebook
```

## Data Format

RESOLVE expects two CSV files:

1. **Header file**: Plot-level data (one row per plot)
   - Required: plot ID column
   - Optional: coordinates, covariates, target columns

2. **Species file**: Species occurrences (one row per species-plot combination)
   - Required: species ID, plot ID columns
   - Optional: abundance, taxonomy (genus, family)

See the notebooks for detailed examples.
