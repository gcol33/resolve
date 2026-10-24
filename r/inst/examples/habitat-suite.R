# Builds a model suite in a temporary directory from the example data shipped
# with resolveR: two habitat classifiers trained under different seeds, and the
# manifest.json that names them. The manifest is left unsealed;
# resolve.seal_suite() fills in each member's size and SHA-256.

plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
dataset <- resolve.dataset.csv(
  plots, species, roles = roles,
  targets = list(habitat = list(column = "habitat", task = "classification"))
)

suite_dir <- file.path(tempdir(), "habitat_suite")
dir.create(suite_dir, showWarnings = FALSE)
members <- lapply(1:2, function(seed) {
  file <- sprintf("seed_%d.pt", seed)
  resolve.train.dataset(dataset, maxEpochs = 20L, seed = seed,
                        savePath = file.path(suite_dir, file), verbose = FALSE)
  list(seed = seed, file = file)
})

manifest <- list(
  format = "resolve-suite", format_version = 1L, name = "habitat_example",
  description = "Habitat class from species composition",
  licence = "CC-BY-4.0", engine_version = resolve.version(),
  taxonomy = "as in the example data", training_scope = "150 synthetic Alpine plots",
  inputs = list(plot_id = "plot_id", species = "species", abundance = "cover",
                genus = "genus", family = "family", latitude = "lat", longitude = "lon"),
  targets = list(list(name = "habitat", task = "classification", combine = "vote",
                      outputs = list("habitat"), status = "released", members = members))
)
jsonlite::write_json(manifest, file.path(suite_dir, "manifest.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = NA)
