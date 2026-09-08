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

## Examples

`resolve.available()` and `resolve.progress()` run without the backend and are
unwrapped. Every other example needs the downloaded engine, which cannot be
present on the check machines, so those are wrapped in `\dontrun{}`.

## Test environments

* local: Windows 11, R 4.6.0, with and without the backend
* win-builder: R-devel, R-release
* GitHub Actions: Ubuntu (R-release, R-devel), macOS (R-release), Windows (R-release)

## Downstream dependencies

None: this is a new package.
