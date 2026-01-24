#!/usr/bin/env Rscript
#' RESOLVE Simple R Example
#'
#' This example demonstrates the formula interface for RESOLVE using the
#' hash() and embed() special functions for species composition encoding.

library(resolveR)

set.seed(42)

cat("============================================================\n")
cat("RESOLVE Formula API Example\n")
cat("============================================================\n\n")

# Create sample data
cat("Creating sample data...\n")

# Species occurrence data (long format)
species_df <- data.frame(
  plot_id = rep(paste0("plot_", 1:100), each = 10),
  species = paste0("sp_", sample(1:50, 1000, replace = TRUE)),
  genus = sample(c("Quercus", "Pinus", "Acer", "Betula", "Fagus"), 1000, replace = TRUE),
  family = sample(c("Fagaceae", "Pinaceae", "Sapindaceae", "Betulaceae"), 1000, replace = TRUE),
  abundance = rexp(1000, rate = 0.1)
)

# Plot-level data with targets and covariates
plots_df <- data.frame(
  plot_id = paste0("plot_", 1:100),
  latitude = runif(100, 45, 55),
  longitude = runif(100, 5, 15),
  elevation = runif(100, 100, 1500),
  ph = runif(100, 4.5, 7.5),
  nitrogen = rexp(100, rate = 5),
  carbon = rexp(100, rate = 0.3)
)

cat(sprintf("  %d plots, %d species records\n\n", nrow(plots_df), nrow(species_df)))

# Split data
train_idx <- 1:80
test_idx <- 81:100

train_plots <- plots_df[train_idx, ]
test_plots <- plots_df[test_idx, ]
train_species <- species_df[species_df$plot_id %in% train_plots$plot_id, ]
test_species <- species_df[species_df$plot_id %in% test_plots$plot_id, ]

# =============================================================================
# FORMULA API: hash() for species, embed() for taxonomy
# =============================================================================

cat("Example 1: Top 5 species by abundance\n")
cat("--------------------------------------\n")

# hash(species, top = 5, by = "abundance") - use top 5 most abundant species
fit1 <- resolve(
  ph ~ latitude + longitude + elevation + hash(species, top = 5, by = "abundance"),
  data = train_plots,
  species = train_species,
  epochs = 50L,
  verbose = TRUE
)

# Print model summary
print(fit1)

# Predict on new data
pred1 <- predict(fit1, newdata = test_plots, species = test_species)
cat("\nPredictions:\n")
print(head(pred1))

# Compute metrics
metrics1 <- compute_metrics(pred1$ph, test_plots$ph)
cat(sprintf("\nTest metrics: MAE = %.3f, RMSE = %.3f\n\n",
            metrics1$mae, metrics1$rmse))


cat("Example 2: Top + bottom species (dominant + rare)\n")
cat("--------------------------------------------------\n")

# Use both top and bottom species - rare species can be informative!
fit2 <- resolve(
  ph ~ latitude + elevation + hash(species, top = 3, bottom = 3, by = "abundance"),
  data = train_plots,
  species = train_species,
  epochs = 50L,
  verbose = TRUE
)

summary(fit2)


cat("\nExample 3: Add taxonomic embeddings\n")
cat("-------------------------------------\n")

# Combine species hashing with learned genus/family embeddings
fit3 <- resolve(
  ph ~ latitude + hash(species, top = 5, by = "abundance") + embed(genus, family),
  data = train_plots,
  species = train_species,
  epochs = 50L,
  verbose = TRUE
)

summary(fit3)


cat("\nExample 4: Custom dimensions\n")
cat("-----------------------------\n")

# Specify dimensions for hash and embed
fit4 <- resolve(
  ph ~ elevation + hash(species, top = 5, by = "abundance", dim = 64) + embed(genus, dim = 32),
  data = train_plots,
  species = train_species,
  epochs = 50L,
  verbose = TRUE
)

summary(fit4)


cat("\nExample 5: Multiple targets with transformations\n")
cat("-------------------------------------------------\n")

# Multiple targets using cbind()
fit5 <- resolve(
  cbind(ph, nitrogen, carbon) ~ latitude + elevation + hash(species, top = 5, by = "abundance") + embed(genus),
  data = train_plots,
  species = train_species,
  transform = list(nitrogen = "log1p", carbon = "log1p"),
  epochs = 50L,
  verbose = TRUE
)

# Summary shows all targets
summary(fit5)

# Predictions include all targets
pred5 <- predict(fit5, newdata = test_plots, species = test_species)
cat("\nMulti-target predictions:\n")
print(head(pred5))


cat("\nExample 6: Species composition only (no covariates)\n")
cat("----------------------------------------------------\n")

# Species features only, no environmental covariates
fit6 <- resolve(
  ph ~ hash(species, top = 5, by = "abundance") + embed(genus, family),
  data = train_plots,
  species = train_species,
  epochs = 50L,
  verbose = TRUE
)

pred6 <- predict(fit6, newdata = test_plots, species = test_species)
metrics6 <- compute_metrics(pred6$ph, test_plots$ph)
cat(sprintf("Species-only model: MAE = %.3f, RMSE = %.3f\n\n",
            metrics6$mae, metrics6$rmse))


# =============================================================================
# API Summary
# =============================================================================

cat("============================================================\n")
cat("Formula Syntax Reference\n")
cat("============================================================\n\n")

cat("Species selection (use in hash()):\n")
cat("  hash(species, top = 5, by = \"cover\")      - Top 5 by cover\n")
cat("  hash(species, bottom = 5, by = \"cover\")   - Bottom 5 (rarest)\n")
cat("  hash(species, top = 3, bottom = 3, by = \"abundance\")  - Both\n\n")

cat("Embedding functions:\n")
cat("  hash(species, ...)      - Feature hashing for species (scalable)\n")
cat("  embed(genus, family)    - Learned embeddings for taxonomy\n")
cat("  onehot(soil_type)       - One-hot encoding (small vocabularies)\n\n")

cat("Optional parameters:\n")
cat("  dim = 64                - Custom embedding dimension\n")
cat("  top = N                 - Use top N species by 'by' column\n")
cat("  bottom = N              - Use bottom N species (rarest)\n")
cat("  by = \"column\"           - Column to rank species by\n\n")

cat("Example formulas:\n")
cat("  ph ~ latitude + hash(species, top = 5, by = \"cover\")\n")
cat("  ph ~ hash(species, top = 3, bottom = 3, by = \"abundance\") + embed(genus)\n")
cat("  cbind(ph, nitrogen) ~ elevation + hash(species, top = 5, by = \"cover\", dim = 64)\n\n")

cat("============================================================\n")
cat("Example complete!\n")
cat("============================================================\n")
