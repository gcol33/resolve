# Configure the PyTorch CUDA caching allocator

Set the `PYTORCH_CUDA_ALLOC_CONF` environment variable to a
platform-aware default (Linux/macOS get an `expandable_segments:True,`
prefix; Windows omits it). Best-effort: if torch already initialized its
allocator, the change may not take effect for the running process.

## Usage

``` r
resolve_configure_cuda_allocator(force = FALSE)
```

## Arguments

- force:

  If `TRUE`, overwrite any existing `PYTORCH_CUDA_ALLOC_CONF`; default
  `FALSE` only sets when unset.

## Value

The resulting value of `PYTORCH_CUDA_ALLOC_CONF` as a string.
