# Cap the GPU Memory the RESOLVE Engine May Use

Cap the PyTorch CUDA caching allocator at a fraction of a device's VRAM,
so a run leaves headroom for the desktop or other programs on a shared
GPU.
[`resolve.load()`](https://gillescolling.com/resolve/reference/resolve.load.md)
and the `vram_fraction` training setting apply the same cap per call;
this sets it for the session.

## Usage

``` r
resolve.set_vram_fraction(fraction, deviceIndex = -1L)
```

## Arguments

- fraction:

  Fraction of the device's VRAM, in (0, 1\]; 1 removes the cap.

- deviceIndex:

  CUDA device; -1 (default) is the current device.

## Value

No return value. Nothing happens on a CPU-only backend or a machine
without a CUDA device.

## Examples

``` r
resolve.set_vram_fraction(0.8)
```
