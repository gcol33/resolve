# RESOLVE

*predicting sample attributes from the species or items observed in it*

![Species and cover of held-out example plot A139 pooled into one latent vector, read out as elevation (predicted 713 m, observed 679 m; R squared 0.85 over 30 held-out plots) and habitat (observed forest, probability 0.89)](assets/hero-light.svg#only-light)
![Species and cover of held-out example plot A139 pooled into one latent vector, read out as elevation (predicted 713 m, observed 679 m; R squared 0.85 over 30 held-out plots) and habitat (observed forest, probability 0.89)](assets/hero-dark.svg#only-dark)

RESOLVE predicts attributes of a sample from the set of things observed in it, such as the area, elevation or habitat type of a vegetation plot from its species list.

The engine is a C++ library on libtorch, the C++ core of [PyTorch](https://pytorch.org/); the Python package, the R package and the command line are bindings over it, so a model trained from one loads in any other. A sample's composition is a variable-length set, and the model pools the effect of each entity, weighted by its abundance, into one latent representation of the sample. Every target is read off that representation, so a continuous target and a categorical one are fitted together. The fit holds out a share of the samples and reports metrics per target, so you can compare encodings and inputs on your own data and choose the model to keep.

RESOLVE supports:

- **Several targets from one encoder.** Regression and classification heads share the latent representation. A target labelled on only some of the samples keeps the rest in the fit when its missing-label policy is set to mask ([Data Preparation](tutorials/data-preparation.md)).
- **Five encodings of the species set.** Feature hashing, learned embeddings, an explicit abundance vector, rank pooling over shared species, genus and family tables, and a transformer over species tokens ([Encoding Modes](tutorials/encoding-modes.md)).
- **Covariates, coordinates and taxonomy.** Numeric covariates, string covariates and coordinates enter the same encoder. Genus and family give a species the model never saw a fallback identity.
- **Pretraining without labels.** Joint-embedding, contrastive, variational and masked-species pretext tasks train the encoder on composition alone before a supervised fit.
- **Diagnostics on the held-out fold.** Residuals, class probabilities, calibration bins and cross-validation, with spatial blocks when the samples have coordinates, come off the trainer ([Training Models](tutorials/training.md)).
- **GPU training.** The engine runs on CUDA when the installed torch has it, and a run that exhausts device memory halves its batch size and retries ([Performance Tuning](tutorials/performance-tuning.md)).

Every interface calls the same compiled engine. The [tests workflow](https://github.com/gcol33/resolve/actions/workflows/tests.yml) builds the engine and runs its C++ tests, the command-line end-to-end tests and the Python binding tests on every push, and the R package goes through R CMD check. The feature-hashing encoder follows [Weinberger et al. (2009)](https://doi.org/10.1145/1553374.1553516) and the transformer encoder [Vaswani et al. (2017)](https://doi.org/10.48550/arXiv.1706.03762), which a methods section can cite alongside the package.

## Installation

=== "Python"

    ```bash
    pip install resolve-core
    ```

    The wheels cover CPython 3.10 to 3.13 on Linux x86-64, Windows x86-64 and macOS on Apple Silicon, and link the libtorch inside the `torch` package that pip installs with them.

=== "R"

    ```r
    # install.packages("pak")
    pak::pak("gcol33/resolve/r")

    library(resolveR)
    resolve.install_backend()
    ```

    The package installs without the engine; `resolve.install_backend()` downloads a prebuilt build, and `variant = "cuda"` the GPU build.

[Installation](tutorials/installation.md) covers the source build, the CUDA kernels and the command-line executable.

## Usage

The example builds 120 synthetic plots whose target `y` is the sum of a fixed value per species, so a model that learns the composition recovers it. Each plot is a row of the header table, and each species observed in it a row of the species table:

=== "Python"

    ```python
    import math
    import pandas as pd
    import resolve_core as rc

    values = {f"sp{s}": math.sin(s * 1.7) * 2.0 for s in range(12)}
    header, species = [], []
    for i in range(120):
        picks = [(i + k * (i // 12 + 1)) % 12 for k in range(3)]
        header.append({"plot": f"P{i}", "y": sum(values[f"sp{s}"] for s in picks)})
        species += [{"plot": f"P{i}", "species": f"sp{s}"} for s in picks]
    plots = pd.DataFrame(header)
    records = pd.DataFrame(species)

    roles = rc.RoleMapping()
    roles.plot_id = "plot"
    roles.species_id = "species"

    targets = [rc.TargetSpec.regression("y")]
    dataset = rc.ResolveDataset.from_pandas(plots, records, roles, targets)

    model = rc.ResolveModel(dataset.schema)
    trainer = rc.Trainer(model, rc.TrainConfig())
    trainer.prepare_data(dataset)
    result = trainer.fit()
    result.final_metrics
    ```

=== "R"

    ```r
    library(resolveR)

    values <- setNames(sin(0:11 * 1.7) * 2, paste0("sp", 0:11))
    picks <- lapply(0:119, function(i) (i + 0:2 * (i %/% 12 + 1)) %% 12)
    y <- vapply(picks, function(s) sum(values[s + 1]), numeric(1))

    plots <- data.frame(plot = paste0("P", 0:119), y = y)
    records <- data.frame(
      plot = rep(plots$plot, each = 3),
      species = paste0("sp", unlist(picks))
    )

    dataset <- resolve.dataset.frame(
      header = plots,
      species = records,
      roles = list(plot_id = "plot", species_id = "species"),
      targets = list(y = list(column = "y", task = "regression"))
    )

    fit <- resolve.train.dataset(dataset)
    fit$result$final_metrics
    ```

The fit holds out 20 % of the plots for early stopping, and `final_metrics` holds, per target, the mean absolute error, root mean square error, R², symmetric mean absolute percentage error and the share of held-out predictions within 10, 25 and 50 % of the truth. [Quick Start](tutorials/quickstart.md) continues from two CSV files to a saved model and its predictions, in Python, R and on the command line.

## Documentation

Start with [Quick Start](tutorials/quickstart.md): one path from two CSV files to a saved model and its predictions.

- [Installation](tutorials/installation.md): requirements, the source build, the R backend and the CLI build.
- [Data Preparation](tutorials/data-preparation.md): the two tables, roles, targets, missing values and reusing a training vocabulary.
- [Training Models](tutorials/training.md): the configuration structs, results, the held-out fold, cross-validation and checkpoints.
- [Encoding Modes](tutorials/encoding-modes.md): the five species encodings and the sample encoders.
- [Performance Tuning](tutorials/performance-tuning.md): learning rate, batch size, mixed precision and GPU settings.
- [Making Predictions](tutorials/prediction.md): loading a model, output formats, new species and batch scoring.
- [Understanding Embeddings](tutorials/embeddings.md): the latent and species spaces.
- [API Reference](api/dataset.md): the dataset, model, trainer and predictor classes.
- [resolveR](https://gillescolling.com/resolve/r/): the R package.

## Getting help

Questions and bug reports go to the [GitHub issues](https://github.com/gcol33/resolve/issues). A bug report is easiest to act on with a small reproducible example and the output of `resolve_core.__version__`.

## License

RESOLVE is licensed under MIT (see [LICENSE.md](https://github.com/gcol33/resolve/blob/master/LICENSE.md)).
