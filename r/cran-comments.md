## Resubmission

This is a resubmission of resolveR 0.10.0 (submitted 2026-09-09). The version
is now 1.0.0 because development of the engine continued while the submission
waited, and the package's version follows the engine's:
`resolve.install_backend()` downloads the engine build of the same version.

Response to the review of 2026-09-17:

* "Please replace \dontrun with \donttest. Please unwrap the examples if they
  are executable in < 5 sec": no example uses `\dontrun{}` any more. The
  package now ships a small example survey (`inst/extdata`, 150 synthetic
  plots), and every example runs on it, each in under 2.1 s elapsed. Apart from
  `resolve.available()` and `resolve.progress()`, the examples call the
  compiled engine, which `resolve.install_backend()` downloads on request and
  which is therefore absent on the check machines. They are guarded with
  roxygen's `@examplesIf resolve.available()`, so they run wherever the engine
  is installed and are skipped where it is not. The example of
  `resolve.install_backend()` itself downloads the engine into the user data
  directory, so it runs only in interactive sessions.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Package name

The package is `resolveR` rather than `resolve` because Bioconductor already
carries a package named RESOLVE. The functions keep their `resolve.` prefix.

## The compiled engine is not part of this package

The package is a thin client over `resolve_c`, a shared library that bundles
'libtorch'. That library is not compiled or linked at install time: the
package installs and checks with no backend present, and every engine verb
raises a clear error until the backend is loaded. `resolve.install_backend()`
downloads a prebuilt build into `tools::R_user_dir("resolveR", "data")`, and
only when the user calls it; nothing is written there by default, at load, or
by any example, test or vignette. This is the same delayed-binary model the
'torch' package uses. The dataset, training and prediction tests skip on CRAN
(`skip_if_no_backend()`), so the check exercises the pure-R and Rcpp layers.

## Test environments

* local: Windows 11, R 4.6.1, without the backend (as on CRAN) and with it
  (every example and test executed)
* win-builder: R-devel, R-release
* GitHub Actions: Ubuntu (R-release, R-devel), macOS (R-release), Windows (R-release)

## Downstream dependencies

None: this is a new package.
