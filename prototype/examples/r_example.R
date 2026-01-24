#!/usr/bin/env Rscript
#' RESOLVE R Example
#'
#' Relational Encoding via Structured Observation Learning with Vector Embeddings
#'
#' Demonstrates the formula-based API with data/obs pattern.

library(resolveR)
set.seed(42)

cat("============================================================\n")
cat("RESOLVE R Example\n")
cat("============================================================\n\n")

# Create synthetic sample data
create_sample_data <- function(n_plots = 100, n_species = 50) {
  genera <- c("Quercus", "Pinus", "Acer", "Betula", "Fagus", "Abies", "Picea")
  families <- c("Fagaceae", "Pinaceae", "Sapindaceae", "Betulaceae")

  # Pre-allocate observation data
  obs_list <- vector("list", n_plots)
  for (i in seq_len(n_plots)) {
    n_sp <- sample(3:15, 1)
    genus_idx <- sample(seq_along(genera), n_sp, replace = TRUE)
    obs_list[[i]] <- data.frame(
      plot_id = sprintf("plot_%04d", i - 1),
      species_id = sprintf("species_%03d", sample(0:(n_species - 1), n_sp, replace = TRUE)),
      genus = genera[genus_idx],
      family = families[((genus_idx - 1) %% length(families)) + 1],
      abundance = rexp(n_sp, rate = 0.1),
      stringsAsFactors = FALSE
    )
  }
  obs <- do.call(rbind, obs_list)

  # Plot-level data
  data <- data.frame(
    plot_id = sprintf("plot_%04d", 0:(n_plots - 1)),
    latitude = runif(n_plots, 40, 60),
    longitude = runif(n_plots, -10, 30),
    elevation = runif(n_plots, 0, 2000),
    ph = runif(n_plots, 4, 8),
    nitrogen = rexp(n_plots, rate = 5),
    carbon = rexp(n_plots, rate = 1/3),
    stringsAsFactors = FALSE
  )

  list(data = data, obs = obs)
}

# Create and split data
cat("1. Creating sample data...\n")
sample_data <- create_sample_data(n_plots = 200, n_species = 100)
data <- sample_data$data
obs <- sample_data$obs

train_data <- data[1:160, ]
test_data <- data[161:200, ]
train_obs <- obs[obs$plot_id %in% train_data$plot_id, ]
test_obs <- obs[obs$plot_id %in% test_data$plot_id, ]

cat(sprintf("   data: %d plots | obs: %d observations\n", nrow(data), nrow(obs)))
cat(sprintf("   Train: %d | Test: %d\n", nrow(train_data), nrow(test_data)))

# Show the API
cat("\n2. Formula API:\n")
cat("   fit <- resolve(\n")
cat("     ph ~ latitude + elevation + hash(species_id, top = 5, by = \"abundance\"),\n")
cat("     data = train_data,\n")
cat("     obs = train_obs,\n")
cat("     by = \"plot_id\"\n")
cat("   )\n\n")

cat("3. data/obs pattern:\n")
cat("   | Domain     | data           | obs                 | by          |\n")
cat("   |------------|----------------|---------------------|-------------|\n")
cat("   | Ecology    | plots          | species occurrences | plot_id     |\n")
cat("   | E-commerce | customers      | purchases           | customer_id |\n")
cat("   | Healthcare | patients       | diagnoses           | patient_id  |\n\n")

cat("4. Encoding types:\n")
cat("   - bare variable: numeric from data (auto-scaled)\n")
cat("   - hash(col, top=K): feature hashing from obs\n")
cat("   - embed(col, dim=D): learned embeddings from obs\n")
cat("   - onehot(col): one-hot encoding\n")
cat("   - raw(col): passthrough\n\n")

cat("5. Formula examples:\n")
cat("   ph ~ lat + hash(species_id, top = 5, by = \"abundance\")\n")
cat("   ph ~ lat + hash(species_id, top = 3, bottom = 3)\n")
cat("   ph ~ lat + hash(species_id) + embed(genus) + embed(family)\n")
cat("   cbind(ph, nitrogen) ~ lat + hash(species_id, top = 5)\n\n")

cat("============================================================\n")
cat("Done!\n")
cat("============================================================\n")
