# Pretrain a Variational Autoencoder on Species Composition

Train a variational autoencoder to reconstruct each plot's species
vector, without labels: a compressed representation of composition
learned from the plots alone.

## Usage

``` r
resolve.pretrain_vae(dataset, config = list())
```

## Arguments

- dataset:

  A ResolveDataset loaded with
  `config = list(species_encoding = "sparse")`, which carries each
  plot's species vector.

- config:

  Named list of VAE settings (`latent_dim`, `encoder_dims`,
  `decoder_dims`, `dropout`, `kl_weight`, `kl_anneal_epochs`,
  `pretrain_epochs`, `pretrain_lr`, `pretrain_weight_decay`,
  `batch_size`, `device`, `seed`); a field left out takes the RESOLVE
  engine's default
  ([`resolve.default_config()`](https://gillescolling.com/resolve/reference/resolve.default_config.md)
  `"vae"`).

## Value

A list with the per-epoch `loss_history`, `recon_loss_history` and
`kl_loss_history`, `total_time_seconds`, `epochs_completed`,
`latent_dim` and `projection_weights`, the encoder's first layer (one
row per unit of its first hidden layer, one column per species of the
vocabulary).

## Examples

``` r
plots <- system.file("extdata", "plots.csv", package = "resolveR")
species <- system.file("extdata", "species.csv", package = "resolveR")
roles <- list(plot_id = "plot_id", species_id = "species", abundance = "cover",
              genus = "genus", family = "family", latitude = "lat", longitude = "lon")
targets <- list(elevation = list(column = "elevation", task = "regression"),
                habitat = list(column = "habitat", task = "classification"))
dataset <- resolve.dataset.csv(plots, species, roles = roles, targets = targets,
                               config = list(species_encoding = "sparse"))
vae <- resolve.pretrain_vae(dataset, list(latent_dim = 4, pretrain_epochs = 20))
plot(vae$loss_history, type = "l")
```
