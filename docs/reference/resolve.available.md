# Is the resolve_c backend available?

The R package is a thin client over the `resolve_c` shared library (the
C ABI over the RESOLVE engine, which bundles libtorch). That library is
loaded at runtime rather than linked at build time, so the package
installs and checks without it. This returns `TRUE` once the backend has
been located and bound (at load, or after
[`resolve.install_backend()`](https://gillescolling.com/resolve/reference/resolve.install_backend.md)).
The dataset / training / prediction verbs require it; metric helpers and
input validation do not.

## Usage

``` r
resolve.available()
```

## Value

`TRUE` if the resolve_c backend is loaded, else `FALSE`.

## See also

[`resolve.install_backend()`](https://gillescolling.com/resolve/reference/resolve.install_backend.md)

## Examples

``` r
resolve.available()
```
