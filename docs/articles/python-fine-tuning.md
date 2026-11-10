# Python reference: fine-tuning

New heads for your own targets on the encoder of a trained model, a
checkpoint or a member of a suite, and the fine-tuned models released as
a suite of their own. [Fine-tuning a trained
model](https://gillescolling.com/resolve/articles/fine-tuning.html?lang=python)
runs every name below on the survey that ships with the R package.

## `ParentModel`

``` python
ParentModel.open(source, config=InitFromConfig())
```

The trained model a fine-tuned model starts from, its *parent*. `source`
is a checkpoint, or a suite directory whose member `config.suite_target`
and `config.suite_seed` name; a suite member’s size and SHA-256 are
checked against the manifest before it is read. A checkpoint that
carries no species vocabulary, only its size, is refused.

- `vocabs()` (`ExternalVocabs`) and `dataset_config()`
  (`DatasetConfig`): what the fine-tuning plots are loaded with, so each
  species is read through the row of the embedding table the parent
  gives it. The vocabularies also replay the classes of a target the
  parent shares.
- `model_config` (`ModelConfig`): the configuration the fine-tuned model
  takes. Only the fields in `FINE_TUNE_MODEL_FIELDS` may be changed.
- `schema` (`ResolveSchema`), `scalers` (`Scalers`) and
  `categorical_vocab` (`CategoricalVocab`): the parent’s.
- `record` (`ParentRecord`): what a fine-tuned checkpoint keeps of it;
  `reuse`, `n_tensors` and `heads` are set by `Trainer.init_from`.

``` python
config = rc.InitFromConfig()
config.suite_target, config.suite_seed = "habitat", 1
parent = rc.ParentModel.open(suite_dir, config)
plots = rc.ResolveDataset.from_pandas(header, species, roles, targets,
                                     config=parent.dataset_config(), vocabs=parent.vocabs())
model_config = parent.model_config
model_config.freeze_encoder = True
trainer = rc.Trainer(rc.ResolveModel(plots.schema,
                                     rc.model_config_for_dataset(plots, model_config)),
                     rc.TrainConfig())
trainer.init_from(parent)
```

## `InitFromConfig`

Where a warm start reads its parent from and what it carries over.

- `reuse` (`InitReuse`, default `Encoder`): read by `Trainer.init_from`.
- `suite_target` (`str`): with a suite directory, the target whose
  member to start from; may stay empty for a suite of one target.
- `suite_seed` (`int`, default `-1`): that member’s seed; `-1` takes the
  target’s first member.

## `InitReuse`

What `Trainer.init_from` carries over besides every parameter and buffer
outside the task heads: `Encoder` (every head starts fresh) or
`EncoderHeads` (also the head of each target the parent shares under the
same name, task, transform and classes, with its target scaler). In a
config they are spelled `encoder` and `encoder_heads`.

## `ParentRecord`

The parent a fine-tuned checkpoint records (`RunMetadata.parent`,
`Trainer.parent`).

- `checkpoint` (`str`), `sha256` (`str`, lower-case hex) and `bytes`
  (`int`): the parent file as opened.
- `resolve_version` (`str`): the engine that trained the parent.
- `reuse` (`InitReuse`), `n_tensors` (`int`, the parameters and buffers
  copied) and `heads` (`list[str]`, the parent heads carried over).
- `suite`, `suite_target` (`str`) and `suite_seed` (`int`): the suite
  member it came from; `suite` is empty for a checkpoint opened
  directly. `licence` and `training_scope` (`str`): that suite’s terms.

## `fine_tune_mismatches()`

``` python
fine_tune_mismatches(parent, model_config, schema)
```

Every way a fine-tuned model’s config and data schema depart from what
its parent fixes, as messages: an encoder-shaping config field with
another value, other loader settings, other vocabularies, covariates,
categorical columns, coordinates, taxonomy or abundance. Empty when the
model reads the plots as its parent does. `Trainer.init_from` raises
`ValueError` listing them.

## `FINE_TUNE_MODEL_FIELDS`

The `ModelConfig` fields a fine-tuned model may set apart from its
parent’s: the heads (`head_hidden_dims`, `head_activation`,
`head_dropout`), the dropout rates and what is frozen
(`freeze_composition`, `freeze_encoder`).

## `write_derived_suite()`

``` python
write_derived_suite(dir, spec)
```

Releases fine-tuned checkpoints as a suite in `dir`, which must not hold
a manifest yet. Each member is copied to
`members/<target>/seed_<seed>.pt`; the manifest takes its input contract
from the parent suite (the part of it the targets’ members read), names
for each target the parent suite and target its members started from
(`SuiteTarget.derived_from`) and per member the SHA-256 of its parent
checkpoint (`SuiteMember.parent_sha256`), and lists a vote’s classes.
The manifest is sealed and the suite loaded once, so every member is
checked against it. Returns the `SuiteManifest`. A member that records
no parent, one started from another suite or from a checkpoint that is
not the parent suite’s member of the seed it records, and members of one
target started from different parent targets raise `ValueError`.

## `DerivedSuiteSpec`

The suite `write_derived_suite` writes: `parent` (the directory of the
suite the members started from), `name`, `description`, `licence`,
`training_scope` (the plots the fine-tuned members were trained on) and
`targets` (`list[DerivedSuiteTarget]`).

## `DerivedSuiteTarget`

One target of a derived suite: `name`, `members`
(`list[DerivedSuiteMember]`), and optionally `outputs` (the heads read;
empty reads the head named `name`), `combine` (`SuiteCombine` or `None`:
a vote for a classification head, a mean for a regression head),
`period`, `units`, `status`, `limit`, `training` and `validation`.

## `DerivedSuiteMember`

``` python
DerivedSuiteMember(checkpoint, seed)
```

One fine-tuned checkpoint of a derived target and its seed.
