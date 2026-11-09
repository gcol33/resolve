# Download and install the resolve_c backend

Fetch a prebuilt `resolve_c` shared library (with its bundled libtorch
runtime) and load it, so the dataset / training / prediction verbs
become available. This mirrors
[`torch::install_torch()`](https://torch.mlverse.org/docs/reference/install_torch.html):
the CRAN package is small and backend-free, and the heavy binary is
fetched on first use into the user's data directory
(`tools::R_user_dir("resolveR", "data")`).

## Usage

``` r
resolve.install_backend(
  version = NULL,
  variant = "cpu",
  url = NULL,
  dir = NULL,
  quiet = FALSE,
  force = FALSE
)
```

## Arguments

- version:

  Release version to fetch (default: the installed package version).
  Ignored when `url` is given.

- variant:

  One of `"cpu"` (default), `"cu128"`, `"cu130"`, or `"cuda"`. `"cuda"`
  auto-selects the CUDA line from the installed NVIDIA driver (`cu130`
  for CUDA \>= 13, else `cu128`). CUDA variants are Windows/Linux x86_64
  only (no CUDA on macOS).

- url:

  Explicit download URL for a `.zip` containing the library and its
  runtime dependencies. Overrides `version` / `variant`.

- dir:

  Destination directory (default: the package's user data dir).

- quiet:

  Suppress progress messages (default `FALSE`).

- force:

  Reinstall even if the backend is already present (default `FALSE`).

## Value

(Invisibly) the directory the backend was installed into.

## Details

The default download URL points at a per-platform asset on the package's
GitHub Releases; supply `url` to install from a local or private mirror.

The `cpu` variant is self-contained (its libtorch is bundled in the
download). The CUDA variants ship only the small `resolve_c` library on
the GitHub release and fetch the matching official libtorch (2-4 GB)
from `download.pytorch.org` on first install, pinned to the exact
version `resolve_c` was built against so the ABI matches. GPU training
then runs through the ordinary `device = "cuda"` path of
[`resolve.train.dataset()`](https://gillescolling.com/resolve/reference/resolve.train.dataset.md)
/
[`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md).

## See also

[`resolve.available()`](https://gillescolling.com/resolve/reference/resolve.available.md)

## Examples

``` r
if (FALSE) { # interactive()
resolve.install_backend()
}
```
