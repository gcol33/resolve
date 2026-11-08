# Name resolution through the R front door: the routes resolve.names()
# reports against a vocabulary, the release's names.json read by path, and the
# arguments it refuses. The suite and dataset paths are in test-suite.R.

.vocabulary <- c("<UNK>", "Quercus robur", "Anemone altaica", "Carex curvicollis")

.names_file <- function(env = parent.frame()) {
  path <- local_csv("", env = env)
  writeLines(jsonlite::toJSON(list(
    source = "test",
    aliases = list(`Anemone nemorosa` = c("Anemone altaica", "eva"),
                   `Carex viridula` = c("Carex curvicollis", "eva"))
  ), auto_unbox = TRUE), path)
  path
}

test_that("each name takes the first route that reaches a token", {
  skip_if_no_backend()
  out <- resolve.names(c("Quercus robur", "Quercus robru", "Anemone nemorosa", "Carex viridul",
                         "Poa annua"), .vocabulary, table = .names_file())
  expect_equal(names(out), c("name", "token", "route", "source", "via", "distance"))
  expect_equal(out$route, c("exact", "spelling", "alias", "spelling", "unresolved"))
  expect_equal(out$token, c("Quercus robur", "Quercus robur", "Anemone altaica",
                            "Carex curvicollis", "Poa annua"))
  expect_equal(out$source, c("", "", "eva", "eva", ""))
  expect_equal(out$via[4], "Carex viridula")
  expect_equal(out$distance, c(0L, 1L, 0L, 1L, 0L))
})

test_that("the table is optional and maxEditDistance bounds the spelling step", {
  skip_if_no_backend()
  expect_equal(resolve.names("Anemone nemorosa", .vocabulary)$route, "unresolved")
  expect_equal(resolve.names("Quercus robru", .vocabulary, maxEditDistance = 0)$route,
               "unresolved")
  expect_equal(resolve.names("Qurecus robru", .vocabulary, maxEditDistance = 2)$distance, 2L)
  expect_error(resolve.names("Quercus robru", .vocabulary, maxEditDistance = 9),
               "max_edit_distance")
  expect_error(resolve.names("x", .vocabulary, maxEditDistance = -1), "maxEditDistance")
  expect_error(resolve.names("x", .vocabulary, table = "no/such/names.json"), "does not exist")
  expect_error(resolve.names(1, .vocabulary), "character")
  expect_error(resolve.names("x", list()), "model must be")
})
