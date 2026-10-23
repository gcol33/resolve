# The species graph a heterogeneous GNN passes messages on, through R: the
# builder, the taxonomy of the dataset's vocabulary, and the model accessors
# that take a graph handed in from R.

graph_csvs <- function(env = parent.frame()) {
  n_plots <- 12
  header_data <- data.frame(
    plot_id = paste0("p", seq_len(n_plots)),
    lon = 9 + 0.3 * seq_len(n_plots),
    lat = 46 + 0.3 * seq_len(n_plots),
    elev = 150 + 11 * seq_len(n_plots),
    y = 1 + 0.2 * seq_len(n_plots),
    stringsAsFactors = FALSE
  )

  # Two plot groups with disjoint composition, so co-occurrence is a different
  # relation from taxonomy.
  groups <- list(c(0, 1, 3), c(2, 4, 5))
  genus_of <- c(0, 0, 1, 1, 2, 2)
  family_of <- c(0, 0, 0, 0, 1, 1)
  rows <- list()
  for (plot in seq_len(n_plots)) {
    members <- groups[[if (plot <= n_plots / 2) 1 else 2]]
    for (j in seq_along(members)) {
      s <- members[j]
      rows[[length(rows) + 1]] <- data.frame(
        plot_id = paste0("p", plot),
        species_id = paste0("sp_", s),
        cover = j,
        genus = paste0("gen_", genus_of[s + 1]),
        family = paste0("fam_", family_of[s + 1]),
        stringsAsFactors = FALSE
      )
    }
  }
  local_csv_pair(header_data, do.call(rbind, rows), env = env)
}

graph_dataset <- function(files, use_taxonomy = TRUE) {
  roles <- list(
    plot_id = "plot_id",
    species_id = "species_id",
    abundance = "cover",
    longitude = "lon",
    latitude = "lat"
  )
  if (use_taxonomy) {
    roles$genus <- "genus"
    roles$family <- "family"
  }
  resolve.dataset.csv(
    header = files$header,
    species = files$species,
    roles = roles,
    targets = list(y = list(column = "y", task = "regression")),
    config = list(species_encoding = "sparse", use_taxonomy = use_taxonomy)
  )
}


test_that("resolve.species_graph joins species by taxonomy and co-occurrence", {
  skip_if_no_backend()
  skip_on_cran()

  files <- graph_csvs()
  dataset <- graph_dataset(files)

  graph <- resolve.species_graph(dataset)
  expect_true(is.list(graph))
  expect_equal(nrow(graph$edge_index), 2)
  expect_equal(ncol(graph$edge_index), graph$n_edges)
  expect_equal(length(graph$edge_type), graph$n_edges)
  expect_equal(graph$n_species, dataset$schema()$n_species_vocab)

  # Three genera of two species (6 ordered pairs), one family of four and one
  # of two (14), and every ordered pair inside the two plot groups (12).
  expect_equal(sum(graph$edge_type == 0), 6)
  expect_equal(sum(graph$edge_type == 1), 14)
  expect_equal(sum(graph$edge_type == 2), 12)
  # <UNK> (species code 0) is a node with no edges.
  expect_true(min(graph$edge_index) > 0)
})


test_that("each relation of the species graph can be switched off", {
  skip_if_no_backend()
  skip_on_cran()

  files <- graph_csvs()
  dataset <- graph_dataset(files)

  taxonomy <- resolve.species_graph(dataset, cooccurrenceEdges = FALSE)
  shared <- resolve.species_graph(dataset, taxonomicEdges = FALSE)
  expect_equal(sort(unique(taxonomy$edge_type)), c(0, 1))
  expect_equal(unique(shared$edge_type), 2)

  # A threshold no pair reaches leaves no co-occurrence edges.
  none <- resolve.species_graph(dataset, taxonomicEdges = FALSE,
                                cooccurrenceThreshold = 1)
  expect_equal(none$n_edges, 0)

  # Neither relation leaves nothing to connect the species by.
  expect_error(resolve.species_graph(dataset, taxonomicEdges = FALSE,
                                     cooccurrenceEdges = FALSE))
})


test_that("the dataset reports the taxonomy of its own vocabulary", {
  skip_if_no_backend()
  skip_on_cran()

  files <- graph_csvs()
  dataset <- graph_dataset(files)

  genus <- dataset$species_genus_ids()
  family <- dataset$species_family_ids()
  n_vocab <- dataset$schema()$n_species_vocab
  expect_equal(length(genus), n_vocab)
  expect_equal(length(family), n_vocab)
  # <UNK> belongs to nothing, every real species to something.
  expect_equal(genus[1], 0)
  expect_true(all(genus[-1] > 0))
  expect_equal(length(unique(genus[-1])), 3)
  expect_equal(length(unique(family[-1])), 2)

  plain <- graph_dataset(files, use_taxonomy = FALSE)
  expect_null(plain$species_genus_ids())
})


test_that("only a heterogeneous GNN reads a species graph", {
  skip_if_no_backend()
  skip_on_cran()

  files <- graph_csvs()
  dataset <- graph_dataset(files)
  graph <- resolve.species_graph(dataset)

  hetero <- new(.resolve_module()$ResolveModel, dataset$schema(), list(
    species_encoding = "sparse",
    uses_explicit_vector = TRUE,
    hidden_dims = c(12, 8),
    encoder_architecture = "heterogeneous_gnn",
    heterogeneous_gnn = list(hidden_dim = 8, output_dim = 6, n_layers = 1,
                             n_heads = 2, dropout = 0)
  ))
  expect_true(hetero$requires_species_graph())
  expect_false(hetero$has_species_graph())
  hetero$set_species_graph(graph$edge_index, graph$edge_type)
  expect_true(hetero$has_species_graph())
  expect_equal(nrow(hetero$species_graph_edge_index()), 2)
  expect_equal(length(hetero$species_graph_edge_type()), graph$n_edges)

  mlp <- new(.resolve_module()$ResolveModel, dataset$schema(), list(
    species_encoding = "sparse",
    uses_explicit_vector = TRUE,
    hidden_dims = c(12, 8)
  ))
  expect_false(mlp$requires_species_graph())
  expect_error(mlp$set_species_graph(graph$edge_index, graph$edge_type))
})

test_that("a sparse dataset returns its species vector as sparse rows", {
  skip_if_no_backend()
  skip_on_cran()

  dataset <- graph_dataset(graph_csvs())
  rows <- dataset$species_vector()
  expect_setequal(names(rows), c("row_offsets", "species", "values", "n_species"))
  n_plots <- dataset$n_plots()
  expect_equal(length(rows$row_offsets), n_plots + 1)
  expect_equal(rows$n_species, dataset$schema()$n_species_vocab)
  # Every plot records three species.
  expect_equal(diff(rows$row_offsets), rep(3, n_plots))

  dense <- dataset$species_vector_dense()
  expect_equal(dim(dense), c(n_plots, rows$n_species))
  rebuilt <- matrix(0, n_plots, rows$n_species)
  for (p in seq_len(n_plots)) {
    idx <- seq.int(rows$row_offsets[p] + 1, rows$row_offsets[p + 1])
    rebuilt[p, rows$species[idx] + 1] <- rows$values[idx]
  }
  expect_equal(dense, rebuilt)
})
