# Writes inst/extdata/plots.csv and inst/extdata/species.csv, the synthetic
# vegetation survey the help-page examples load: 150 plots along an Alpine
# elevation gradient, each with a habitat class and the species recorded in it.
# Every species has an elevation optimum and a habitat it favours; a plot holds
# a species with a probability that falls off with the distance between the
# plot's elevation and that optimum and drops further outside the favoured
# habitat. Run from the package root: Rscript data-raw/example_data.R

set.seed(20261024)

pool <- data.frame(
  species = c(
    "Fagus sylvatica", "Quercus robur", "Quercus petraea", "Picea abies",
    "Abies alba", "Larix decidua", "Pinus cembra", "Acer pseudoplatanus",
    "Vaccinium myrtillus", "Rhododendron ferrugineum", "Arrhenatherum elatius",
    "Dactylis glomerata", "Festuca rubra", "Nardus stricta", "Sesleria caerulea",
    "Trifolium pratense", "Trifolium repens", "Ranunculus acris",
    "Carex nigra", "Eriophorum angustifolium", "Caltha palustris",
    "Molinia caerulea", "Carex curvula", "Salix herbacea", "Silene acaulis"
  ),
  family = c(
    "Fagaceae", "Fagaceae", "Fagaceae", "Pinaceae",
    "Pinaceae", "Pinaceae", "Pinaceae", "Sapindaceae",
    "Ericaceae", "Ericaceae", "Poaceae",
    "Poaceae", "Poaceae", "Poaceae", "Poaceae",
    "Fabaceae", "Fabaceae", "Ranunculaceae",
    "Cyperaceae", "Cyperaceae", "Ranunculaceae",
    "Poaceae", "Cyperaceae", "Salicaceae", "Caryophyllaceae"
  ),
  optimum = c(
    900, 500, 600, 1400,
    1100, 1900, 2000, 1000,
    1600, 2000, 500,
    700, 1200, 1900, 1700,
    800, 900, 900,
    1500, 1600, 1000,
    1000, 2400, 2500, 2400
  ),
  habitat = c(
    "forest", "forest", "forest", "forest",
    "forest", "forest", "forest", "forest",
    "forest", "alpine", "grassland",
    "grassland", "grassland", "grassland", "grassland",
    "grassland", "grassland", "grassland",
    "wetland", "wetland", "wetland",
    "wetland", "alpine", "alpine", "alpine"
  ),
  stringsAsFactors = FALSE
)
pool$genus <- sub(" .*", "", pool$species)

n_plots <- 150L
elevation <- round(stats::runif(n_plots, 400, 2600))
habitat <- ifelse(
  elevation > 2150, "alpine",
  sample(c("forest", "grassland", "wetland"), n_plots, replace = TRUE,
         prob = c(0.5, 0.35, 0.15))
)
plots <- data.frame(
  plot_id = sprintf("A%03d", seq_len(n_plots)),
  lat = round(stats::runif(n_plots, 46.5, 47.2), 4),
  lon = round(stats::runif(n_plots, 10.5, 12.0), 4),
  elevation = elevation,
  habitat = habitat,
  stringsAsFactors = FALSE
)

records <- lapply(seq_len(n_plots), function(i) {
  p <- 0.95 * exp(-((plots$elevation[i] - pool$optimum) / 450)^2) *
    ifelse(pool$habitat == plots$habitat[i], 1, 0.25)
  present <- which(stats::runif(nrow(pool)) < p)
  if (length(present) < 2L) present <- order(p, decreasing = TRUE)[1:2]
  data.frame(
    plot_id = plots$plot_id[i],
    species = pool$species[present],
    genus = pool$genus[present],
    family = pool$family[present],
    cover = round(stats::runif(length(present), 1, 60)),
    stringsAsFactors = FALSE
  )
})
species <- do.call(rbind, records)

dir.create("inst/extdata", recursive = TRUE, showWarnings = FALSE)
utils::write.csv(plots, "inst/extdata/plots.csv", row.names = FALSE)
utils::write.csv(species, "inst/extdata/species.csv", row.names = FALSE)
