# The staged install of resolve.install_backend(): a build is put together
# beside its destination and takes the destination's place whole. These run
# without a backend.

local_backend_parent <- function(env = parent.frame()) {
  parent <- local_dir("resolve_backend_", env)
  file.path(parent, "resolve_c")
}

write_build <- function(dir, label, complete = TRUE) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(label, file.path(dir, .resolve_backend_libname()))
  writeLines(label, file.path(dir, paste0("runtime_", label, ".txt")))
  if (complete) file.create(file.path(dir, .resolve_staged_marker))
  invisible(dir)
}

test_that("without a completed staged build the installed backend stays", {
  dir <- local_backend_parent()
  write_build(dir, "old", complete = FALSE)
  expect_false(.resolve_swap_staged(dir))

  # An unfinished download carries no marker and is not swapped in.
  write_build(.resolve_staged_dir(dir), "new", complete = FALSE)
  expect_false(.resolve_swap_staged(dir))
  expect_identical(readLines(file.path(dir, .resolve_backend_libname())), "old")
  expect_true(dir.exists(.resolve_staged_dir(dir)))
})

test_that("a completed staged build takes the installed backend's place whole", {
  dir <- local_backend_parent()
  write_build(dir, "old", complete = FALSE)
  write_build(.resolve_staged_dir(dir), "new")
  expect_true(.resolve_swap_staged(dir))
  expect_identical(readLines(file.path(dir, .resolve_backend_libname())), "new")
  expect_setequal(list.files(dir), c(.resolve_backend_libname(), "runtime_new.txt"))
  expect_false(dir.exists(.resolve_staged_dir(dir)))
  expect_length(Sys.glob(paste0(.resolve_sibling_dir(dir, ".retired"), "*")), 0L)
})

test_that("a staged build is installed where no backend was", {
  dir <- local_backend_parent()
  write_build(.resolve_staged_dir(dir), "new")
  expect_true(.resolve_swap_staged(dir))
  expect_setequal(list.files(dir), c(.resolve_backend_libname(), "runtime_new.txt"))
})

test_that("install_backend refuses a directory holding other files", {
  dir <- local_dir("resolve_other_")
  writeLines("mine", file.path(dir, "notes.txt"))
  expect_error(resolve.install_backend(dir = dir, quiet = TRUE), "holds other files")
  expect_identical(list.files(dir), "notes.txt")
  expect_false(dir.exists(.resolve_staged_dir(dir)))
})
