/* resolve_capi.h - Flat C ABI over the RESOLVE C++ engine.
 *
 * The R package is compiled with whatever C++ toolchain R uses (Rtools
 * mingw-w64 on Windows), while libtorch, and so the engine, is built with
 * another (MSVC on Windows); the two C++ ABIs do not cross. A C ABI does. This
 * header exposes the engine through opaque handles and primitive types ONLY --
 * no torch::Tensor, no std::string, no STL, no Rcpp in any signature -- so such
 * a client can call the `resolve_c` shared library. This is the "lantern"
 * pattern of the mlverse/torch R package.
 *
 * Marshaling primitive: `resolve_value_t`, a heap tagged tree (scalars, arrays,
 * row-major double/int matrices, ordered maps, lists). It carries BOTH
 * structured input (roles / targets / config / forward inputs) and structured
 * output (results, accessor tensors). Tensors are DOUBLE_MATRIX / INT_MATRIX
 * nodes; an absent optional input is simply a key missing from the input map.
 *
 * Ownership: allocations and frees happen on the SAME side of the boundary.
 *   - Value trees are allocated by this library (builder calls / return values)
 *     and freed by it (`resolve_value_free`). The caller only fills / reads.
 *   - Raw buffers the caller passes into a builder are COPIED; the library
 *     never frees caller memory.
 * No foreign `free()` crosses the C runtime boundary.
 *
 * Errors: functions returning a pointer return NULL on failure (an allocation
 * failure included); functions returning `int` return 0 on success and -1 on
 * failure. In both cases a human-readable message is available from
 * `resolve_last_error()` (thread-local). C++ exceptions never cross the
 * boundary.
 *
 * Configuration maps (a DatasetConfig, ModelConfig, TrainConfig and the
 * per-call settings below) are read strictly: every key is a field name of the
 * struct, at every level of nesting, and a key that is not -- a misspelling --
 * fails the call naming the accepted keys. A value of the wrong kind (a string
 * where a number is read) fails too. An absent key keeps the engine's default,
 * which resolve_default_config() reports.
 *
 * Devices are named as libtorch spells them: "cpu", "cuda" (the current CUDA
 * device) or "cuda:N"; anything else fails.
 *
 * Runtime binding (the CRAN R package): a client that defines
 * RESOLVE_CAPI_DYNLOAD before including this header gets ONLY the opaque handle
 * typedefs and the resolve_value_kind_t enum -- every function PROTOTYPE is
 * suppressed. Such a client supplies its own dynamically-loaded forwarders
 * (see r/src/resolve_capi_dynload.h) so it can `dlopen`/`LoadLibrary` the
 * resolve_c shared library at runtime instead of linking an import library at
 * build time. The engine build (RESOLVE_CAPI_BUILD) never defines
 * RESOLVE_CAPI_DYNLOAD, so it always sees the prototypes it must define.
 */
#ifndef RESOLVE_CAPI_H
#define RESOLVE_CAPI_H

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#  if defined(RESOLVE_CAPI_BUILD)
#    define RESOLVE_CAPI __declspec(dllexport)
#  else
#    define RESOLVE_CAPI __declspec(dllimport)
#  endif
#else
#  if defined(RESOLVE_CAPI_BUILD)
#    define RESOLVE_CAPI __attribute__((visibility("default")))
#  else
#    define RESOLVE_CAPI
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Opaque handles and the value-tree tag enum                                 */
/*                                                                            */
/* These types are needed by BOTH the engine build and a dynamic-loading      */
/* client, so they live above the RESOLVE_CAPI_DYNLOAD prototype guard.       */
/* ========================================================================== */

/* The single marshaling primitive: a heap tagged tree. */
typedef struct resolve_value resolve_value_t;

/* Engine object handles (defined only inside the library). */
typedef struct resolve_dataset   resolve_dataset_t;
typedef struct resolve_model     resolve_model_t;
typedef struct resolve_trainer   resolve_trainer_t;
typedef struct resolve_predictor resolve_predictor_t;
typedef struct resolve_suite     resolve_suite_t;
typedef struct resolve_fuzzy     resolve_fuzzy_t;

/* Tag returned by resolve_value_kind(). */
typedef enum {
    RESOLVE_VALUE_NULL = 0,
    RESOLVE_VALUE_BOOL,
    RESOLVE_VALUE_INT,          /* int64 scalar       */
    RESOLVE_VALUE_DOUBLE,       /* double scalar      */
    RESOLVE_VALUE_STRING,
    RESOLVE_VALUE_INT_ARRAY,    /* 1-D int64          */
    RESOLVE_VALUE_DOUBLE_ARRAY, /* 1-D double         */
    RESOLVE_VALUE_STRING_ARRAY,
    RESOLVE_VALUE_DOUBLE_MATRIX,/* row-major nrow x ncol double */
    RESOLVE_VALUE_INT_MATRIX,   /* row-major nrow x ncol int64  */
    RESOLVE_VALUE_MAP,          /* ordered key -> value         */
    RESOLVE_VALUE_LIST          /* ordered values               */
} resolve_value_kind_t;

#ifndef RESOLVE_CAPI_DYNLOAD

/* ========================================================================== */
/* Error reporting                                                            */
/* ========================================================================== */

/* Message for the most recent failure on the calling thread. Valid until the
 * next failing call on the same thread. Never NULL (empty string if no error). */
RESOLVE_CAPI const char* resolve_last_error(void);

/* ========================================================================== */
/* resolve_value_t - the single marshaling primitive                          */
/* ========================================================================== */

/* ---- construction --------------------------------------------------------- */
RESOLVE_CAPI resolve_value_t* resolve_value_new_null(void);
RESOLVE_CAPI resolve_value_t* resolve_value_new_map(void);
RESOLVE_CAPI resolve_value_t* resolve_value_new_list(void);
/* Free a value tree (recursively). Safe on NULL. */
RESOLVE_CAPI void resolve_value_free(resolve_value_t* v);

/* Standalone scalar / array / matrix node factories. Arrays and matrices copy
 * their input (NULL data is accepted only for zero elements). Returned node is
 * owned by the caller until appended to a list (resolve_list_append_value) or
 * set into a map (resolve_map_set_value), which transfer ownership. NULL on
 * failure. */
RESOLVE_CAPI resolve_value_t* resolve_value_new_bool  (int v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int   (int64_t v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double(double v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_string(const char* v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int_array   (const int64_t* v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double_array(const double*  v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_string_array(const char* const* v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double_matrix(const double*  v, int64_t nrow, int64_t ncol);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int_matrix   (const int64_t* v, int64_t nrow, int64_t ncol);

/* ---- map setters (key is copied; arrays/matrices are copied) --------------
 * Return 0, or -1 for a NULL or non-map `map`, a NULL key, NULL data for a
 * non-empty array or an allocation failure (message in resolve_last_error()). */
RESOLVE_CAPI int resolve_map_set_null   (resolve_value_t* map, const char* key);
RESOLVE_CAPI int resolve_map_set_bool   (resolve_value_t* map, const char* key, int v);
RESOLVE_CAPI int resolve_map_set_int    (resolve_value_t* map, const char* key, int64_t v);
RESOLVE_CAPI int resolve_map_set_double (resolve_value_t* map, const char* key, double v);
RESOLVE_CAPI int resolve_map_set_string (resolve_value_t* map, const char* key, const char* v);
RESOLVE_CAPI int resolve_map_set_int_array   (resolve_value_t* map, const char* key, const int64_t* v, int64_t n);
RESOLVE_CAPI int resolve_map_set_double_array(resolve_value_t* map, const char* key, const double*  v, int64_t n);
RESOLVE_CAPI int resolve_map_set_string_array(resolve_value_t* map, const char* key, const char* const* v, int64_t n);
RESOLVE_CAPI int resolve_map_set_double_matrix(resolve_value_t* map, const char* key, const double*  v, int64_t nrow, int64_t ncol);
RESOLVE_CAPI int resolve_map_set_int_matrix   (resolve_value_t* map, const char* key, const int64_t* v, int64_t nrow, int64_t ncol);
/* Transfers ownership of `child` into `map` under `key`; do not free `child`,
 * even on failure (it is freed then). 0 / -1. */
RESOLVE_CAPI int resolve_map_set_value  (resolve_value_t* map, const char* key, resolve_value_t* child);

/* ---- list builders -------------------------------------------------------- */
/* Transfers ownership of `child` into `list`; do not free `child`, even on
 * failure (it is freed then). 0 / -1. */
RESOLVE_CAPI int resolve_list_append_value(resolve_value_t* list, resolve_value_t* child);

/* ---- readers -------------------------------------------------------------- */
RESOLVE_CAPI resolve_value_kind_t resolve_value_kind(const resolve_value_t* v);

/* scalars */
RESOLVE_CAPI int         resolve_value_as_bool  (const resolve_value_t* v);
RESOLVE_CAPI int64_t     resolve_value_as_int   (const resolve_value_t* v);
RESOLVE_CAPI double      resolve_value_as_double (const resolve_value_t* v);
RESOLVE_CAPI const char* resolve_value_as_string (const resolve_value_t* v);

/* arrays / matrices: returned pointer is owned by `v` (valid until freed) */
RESOLVE_CAPI const int64_t* resolve_value_as_int_array   (const resolve_value_t* v, int64_t* n);
RESOLVE_CAPI const double*  resolve_value_as_double_array (const resolve_value_t* v, int64_t* n);
RESOLVE_CAPI const double*  resolve_value_as_double_matrix(const resolve_value_t* v, int64_t* nrow, int64_t* ncol);
RESOLVE_CAPI const int64_t* resolve_value_as_int_matrix   (const resolve_value_t* v, int64_t* nrow, int64_t* ncol);
RESOLVE_CAPI int64_t        resolve_value_string_array_size(const resolve_value_t* v);
RESOLVE_CAPI const char*    resolve_value_string_at        (const resolve_value_t* v, int64_t i);

/* map */
RESOLVE_CAPI int64_t                resolve_map_size    (const resolve_value_t* map);
RESOLVE_CAPI const char*            resolve_map_key_at  (const resolve_value_t* map, int64_t i);
RESOLVE_CAPI const resolve_value_t* resolve_map_value_at(const resolve_value_t* map, int64_t i);
/* Returns NULL if key absent (distinct from a present NULL-kind value). */
RESOLVE_CAPI const resolve_value_t* resolve_map_get     (const resolve_value_t* map, const char* key);

/* list */
RESOLVE_CAPI int64_t                resolve_list_size(const resolve_value_t* list);
RESOLVE_CAPI const resolve_value_t* resolve_list_at  (const resolve_value_t* list, int64_t i);

/* ========================================================================== */
/* Free functions                                                             */
/* ========================================================================== */

/* Engine version string (static storage; do not free). */
RESOLVE_CAPI const char* resolve_capi_version(void);

/* Cap the PyTorch CUDA caching allocator at `fraction` of device VRAM.
 * device_index = -1 uses the current device. No-op on CPU builds. */
RESOLVE_CAPI int resolve_capi_set_vram_fraction(double fraction, int device_index);

/* Set PYTORCH_CUDA_ALLOC_CONF to a platform-aware default. Returns the
 * resulting value as a freshly-allocated value tree (STRING kind). */
RESOLVE_CAPI resolve_value_t* resolve_capi_configure_cuda_allocator(int force);

/* The default of every field of a configuration struct, as the MAP its reader
 * takes. `which`: dataset model train spatial_block retrieval pretrain
 * masked_species vae. NULL for an unknown name. */
RESOLVE_CAPI resolve_value_t* resolve_default_config(const char* which);

/* Pin libtorch's intra-op / inter-op thread pools (<=0 keeps the default).
 * Best-effort: removes the worker threads whose teardown join crashes the
 * Rscript.exe launcher on Windows. Call at startup. */
RESOLVE_CAPI int resolve_capi_set_thread_pools(int intraop_threads, int interop_threads);

/* Install the Windows crash handler: turn an unhandled native fault into an
 * immediate TerminateProcess instead of a JIT-debugger hang or a
 * teardown access violation. No-op off Windows. Exits with
 * `shutdown_exit_code` once resolve_capi_signal_work_complete() has run, else a
 * non-zero failure code derived from the fault. */
RESOLVE_CAPI int resolve_capi_install_crash_handler(int shutdown_exit_code);

/* Mark all engine work complete: a subsequent native fault is treated as a
 * benign teardown artifact and exits with the shutdown code. Wire to the
 * binding's normal-shutdown hook (R on-exit finalizer, Python atexit). */
RESOLVE_CAPI int resolve_capi_signal_work_complete(void);

/* ========================================================================== */
/* Metrics (flat vectors in, scalar out via out-param; return 0/-1)           */
/* ========================================================================== */

RESOLVE_CAPI int resolve_metric_band_accuracy(const double* pred, const double* target, int64_t n, double threshold, double* out);
RESOLVE_CAPI int resolve_metric_mae          (const double* pred, const double* target, int64_t n, double* out);
RESOLVE_CAPI int resolve_metric_rmse         (const double* pred, const double* target, int64_t n, double* out);
RESOLVE_CAPI int resolve_metric_smape        (const double* pred, const double* target, int64_t n, double eps, double* out);
RESOLVE_CAPI int resolve_metric_accuracy     (const double* pred, const double* target, int64_t n, double* out);
RESOLVE_CAPI int resolve_metric_r_squared    (const double* pred, const double* target, int64_t n, double* out);

/* ========================================================================== */
/* Dataset                                                                    */
/* ========================================================================== */

/* Load a dataset. `species` (required) and `header` (NULL for a single table,
 * where each plot's plot-level columns are read from its first species row)
 * are each a STRING, the path of a CSV file, or a MAP of column name ->
 * STRING_ARRAY, an in-memory table with every cell a string (missing = "").
 * `roles`, `targets` and `config` are MAP value trees. A vocabulary source
 * encodes the dataset in a trained model's integer-code namespace instead of
 * fitting its own: `schema_source`, another dataset, or `vocabs`, the tree
 * resolve_predictor_get(p, "vocabs") or resolve_dataset_get(ds, "vocabs")
 * returns (a schema tree plus a "categorical_vocab" map of {column -> {name ->
 * code}}); at most one. Every non-hash encoder indexes an embedding table by a
 * code that is a function of the file its vocab was fitted on, so data scored
 * by a trained model must be loaded with its vocabularies. NULL on error,
 * including when `vocabs` carries no species vocabulary (a checkpoint that
 * stores only the vocabulary sizes). */
RESOLVE_CAPI resolve_dataset_t* resolve_dataset_load(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config,
    const resolve_dataset_t* schema_source, const resolve_value_t* vocabs);

RESOLVE_CAPI void resolve_dataset_free(resolve_dataset_t* ds);

/* String-dispatched accessor. Returns a freshly-allocated value tree (caller
 * frees), or NULL on error. `what` is one of:
 *   coordinates covariates hash_embedding species_ids species_vector
 *   species_vector_dense genus_ids family_ids unknown_fraction
 *   unknown_count categorical_ids
 *   categorical_vocab targets schema vocabs plot_ids species_vocab n_plots
 *   config has_raw_species_data raw_species_ids raw_weights plot_offsets
 *   taxonomy_vocab pool_genus_ids pool_family_ids pool_weights pool_mask
 *   pool_has_cover has_pool_data
 * species_vector is a MAP of the sparse rows {row_offsets, species,
 * values, n_species}: plot p holds species[row_offsets[p] ..
 * row_offsets[p + 1]) with their values; species_vector_dense is the
 * (n_plots, n_species) matrix.
 * Optional-tensor accessors return a NULL-kind value when the tensor is
 * undefined/empty. */
RESOLVE_CAPI resolve_value_t* resolve_dataset_get(const resolve_dataset_t* ds, const char* what);

/* ========================================================================== */
/* Model                                                                      */
/* ========================================================================== */

/* `schema` and `config` are MAP value trees. Return NULL on error. */
RESOLVE_CAPI resolve_model_t* resolve_model_create(
    const resolve_value_t* schema, const resolve_value_t* config);

/* A model for the plots of `ds`: its schema, and `config` (a ModelConfig MAP)
 * with the fields that size a model to the data -- species_encoding, hash_dim,
 * top_k, top_k_species and the taxonomy slot count -- taken from the dataset.
 * NULL on error. */
RESOLVE_CAPI resolve_model_t* resolve_model_create_for_dataset(
    const resolve_dataset_t* ds, const resolve_value_t* config);

RESOLVE_CAPI void resolve_model_free(resolve_model_t* m);

/* Input-taking methods. `inputs` is a MAP carrying the relevant tensors
 * (continuous required; genus_ids/family_ids/species_ids/species_vector/
 * pool_genus_ids/pool_family_ids/pool_weights/pool_mask/pool_has_cover/
 * categorical_ids optional, plus "target" string for forward_single).
 * `method` is one of: forward get_latent forward_with_aux forward_single
 * encode_with_activations get_gate_probs. Returns value tree / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_model_call(
    resolve_model_t* m, const char* method, const resolve_value_t* inputs);

/* Zero-arg accessor. `what` is one of: latent_dim species_encoding config
 * uses_explicit_vector uses_moe n_experts genus_weights family_weights
 * species_weights composition_parameters (a list of matrices, the species,
 * genus and family tables the species encoder reads composition through)
 * aux_loss_names (the auxiliary objective terms forward_with_aux reports
 * under "aux_losses") row_reference_rows (SAINT's inference reference rows)
 * requires_species_graph has_species_graph species_graph_edge_index
 * species_graph_edge_type (sorted by edge type).
 * Returns value tree / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_model_get(const resolve_model_t* m, const char* what);

/* State mutations. Return 0 / -1. mode!=0 => train, mode==0 => eval. */
RESOLVE_CAPI int resolve_model_set_train(resolve_model_t* m, int mode);
RESOLVE_CAPI int resolve_model_to_device(resolve_model_t* m, const char* device);
RESOLVE_CAPI int resolve_model_set_traits(resolve_model_t* m, const resolve_value_t* traits);
/* The species graph a HeterogeneousGNN passes messages on. `edge_index` is a
 * (2, n_edges) INT_MATRIX (row 0 source, row 1 target), `edge_type` an
 * INT_ARRAY of the same length. Trainer::prepare_data builds and sets this from
 * the dataset, and a checkpoint carries it, so a caller only needs it to supply
 * a graph of its own. */
RESOLVE_CAPI int resolve_model_set_species_graph(resolve_model_t* m,
                                                const resolve_value_t* edge_index,
                                                const resolve_value_t* edge_type);

/* Build the species graph from a dataset's taxonomy and co-occurrence, the way
 * a HeterogeneousGNNConfig MAP (`config`, or NULL for the defaults) asks for.
 * Returns a MAP of edge_index / edge_type / n_species / n_edges. */
RESOLVE_CAPI resolve_value_t* resolve_build_species_graph(
    const resolve_dataset_t* ds, const resolve_value_t* config);

/* Self-supervised pretraining of the model's encoder, in place, on the plots
 * of `ds` (their model inputs, the continuous block unstandardised). `method`:
 * "jepa" or "scarf" (`config` a PretrainConfig MAP) or "masked_species" (a
 * MLMPretrainConfig MAP; a transformer-encoded model). NULL `config` takes the
 * defaults. Returns a MAP of loss_history / total_time_seconds /
 * epochs_completed, NULL on error. */
RESOLVE_CAPI resolve_value_t* resolve_model_pretrain(
    resolve_model_t* m, const char* method, const resolve_dataset_t* ds,
    const resolve_value_t* config);

/* Train a variational autoencoder on the species vectors of `ds` (a dataset
 * loaded with species_encoding "sparse"); `config` a VAEConfig MAP or NULL.
 * Returns a MAP of loss_history / recon_loss_history / kl_loss_history /
 * total_time_seconds / epochs_completed / latent_dim and projection_weights,
 * the encoder's first layer (DOUBLE_MATRIX), NULL on error. */
RESOLVE_CAPI resolve_value_t* resolve_vae_pretrain(
    const resolve_dataset_t* ds, const resolve_value_t* config);

/* ========================================================================== */
/* Trainer                                                                    */
/* ========================================================================== */

/* `config` is a MAP value tree. The model is shared (TORCH_MODULE holder). */
RESOLVE_CAPI resolve_trainer_t* resolve_trainer_create(
    resolve_model_t* model, const resolve_value_t* config);

RESOLVE_CAPI void resolve_trainer_free(resolve_trainer_t* t);

/* Raw-tensor prepare_data. `inputs` MAP carries coordinates/covariates/
 * hash_embedding/species_ids/species_vector/genus_ids/family_ids/
 * unknown_fraction/unknown_count/categorical_ids/pool fields and a "targets"
 * sub-MAP (name -> double array). Covers both the hash/embed/sparse and pool
 * paths (pool fields present-or-absent). Returns 0 / -1. */
RESOLVE_CAPI int resolve_trainer_prepare_data(
    resolve_trainer_t* t, const resolve_value_t* inputs, double test_size, int seed);

RESOLVE_CAPI int resolve_trainer_prepare_data_from_dataset(
    resolve_trainer_t* t, const resolve_dataset_t* ds, double test_size, int seed);

/* prepare_data with the held-out fold early stopping watches named by the
 * caller instead of drawn at random. `stopping` is an INT_ARRAY of 0-based
 * positions in the dataset's plot order, or a STRING_ARRAY of plot IDs. Every
 * other plot is a fitting row; `seed` drives the per-epoch shuffle. An empty
 * fold, an unknown plot, a repeated plot, a fold of every plot, or a fold with
 * no labelled plot for a masked target fails. Returns 0 / -1. */
RESOLVE_CAPI int resolve_trainer_prepare_data_stopping_fold(
    resolve_trainer_t* t, const resolve_dataset_t* ds,
    const resolve_value_t* stopping, int seed);

/* Train. Returns a MAP value tree (TrainResult) / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_fit(resolve_trainer_t* t);

/* Continue a run from a checkpoint fit() wrote while training (best.pt or
 * checkpoint_<N>.pt under the checkpoint directory) at the epoch after it.
 * The trainer must hold the run's architecture and recipe and be prepared on
 * the run's data with its seed. Returns the whole run's TrainResult MAP, its
 * resumed_from_epoch naming where it continued / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_resume(
    resolve_trainer_t* t, const char* checkpoint_path);

/* Save checkpoint. `metadata` may be NULL (or a NULL-kind value). 0 / -1. */
RESOLVE_CAPI int resolve_trainer_save(
    resolve_trainer_t* t, const char* path, const resolve_value_t* metadata);

/* Load weights/scalers/vocab into this trainer in place. 0 / -1. */
RESOLVE_CAPI int resolve_trainer_load_state(
    resolve_trainer_t* t, const char* path, const char* device, double vram_fraction);

/* Zero-arg accessor. `what`: scalers config test_indices train_indices
 * test_plot_ids train_plot_ids categorical_vocab effective_batch_size
 * run_metadata (a MAP shaped like resolve_load_run_metadata's: the split
 * prepare_data made and, after fit, the run's timing, epochs and metrics; pass
 * it to resolve_trainer_save to record it in the checkpoint).
 * Returns value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_get(const resolve_trainer_t* t, const char* what);

/* Evaluation. `kind`: diagnostics calibration residuals
 * classification_predictions. `args` MAP carries "target_name" and optional
 * "n_bins". Returns value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_compute(
    resolve_trainer_t* t, const char* kind, const resolve_value_t* args);

RESOLVE_CAPI resolve_value_t* resolve_trainer_cross_validate(
    resolve_trainer_t* t, int n_folds, int seed);

/* `spatial_cfg` is a SpatialBlockConfig MAP (lat_size / lon_size / balance). */
RESOLVE_CAPI resolve_value_t* resolve_trainer_cross_validate_spatial(
    resolve_trainer_t* t, const resolve_value_t* spatial_cfg, int n_folds, int seed);

/* predict_from_trainer. `inputs` MAP as for resolve_model_call("forward").
 * Returns a MAP (name -> double array) / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_predict(
    resolve_trainer_t* t, const resolve_value_t* inputs);

/* Read persisted config / run metadata from a checkpoint without loading the
 * model. Returns a MAP value tree / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_load_train_config(const char* path);
RESOLVE_CAPI resolve_value_t* resolve_load_run_metadata(const char* path);

/* ========================================================================== */
/* Predictor                                                                  */
/* ========================================================================== */

RESOLVE_CAPI resolve_predictor_t* resolve_predictor_load(
    const char* path, const char* device, double vram_fraction);

RESOLVE_CAPI void resolve_predictor_free(resolve_predictor_t* p);

/* Raw-tensor predict. `inputs` MAP carries coordinates/covariates/
 * hash_embedding + optional species_ids/species_vector/genus_ids/family_ids/
 * unknown_fraction/unknown_count/pool fields/categorical_ids. Returns a MAP
 * with "predictions" (name -> DOUBLE_ARRAY), "probabilities" (one
 * DOUBLE_MATRIX `(n_plots, n_classes)` of softmax rows per classification
 * target; empty MAP when there is none), "targets", "plot_ids" and (if
 * return_latent) "latent". */
RESOLVE_CAPI resolve_value_t* resolve_predictor_predict(
    resolve_predictor_t* p, const resolve_value_t* inputs, int return_latent);

/* Predict on a dataset. batch_size = -1 keeps the one-shot path. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_predict_dataset(
    resolve_predictor_t* p, const resolve_dataset_t* ds, int return_latent, int64_t batch_size);

/* get_embeddings. `inputs` MAP carries the same keys as
 * resolve_predictor_predict. Returns a DOUBLE_MATRIX value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_get_embeddings(
    resolve_predictor_t* p, const resolve_value_t* inputs);

RESOLVE_CAPI int resolve_predictor_optimize_for_inference(resolve_predictor_t* p);

/* Retrieval augmentation (TabR): encode the plots of `reference` (built in the
 * model's namespace, carrying every target's labels, normally the training
 * plots) in chunks of `batch_size`, and from then on blend every prediction
 * with the labels of the plot's nearest reference plots in latent space.
 * `config` is a RetrievalConfig MAP (k_neighbors / alpha / temperature) or
 * NULL. 0 / -1. */
RESOLVE_CAPI int resolve_predictor_set_retrieval(
    resolve_predictor_t* p, const resolve_dataset_t* reference,
    const resolve_value_t* config, int64_t batch_size);

/* Back to the model's own predictions. 0 / -1. */
RESOLVE_CAPI int resolve_predictor_clear_retrieval(resolve_predictor_t* p);

/* Zero-arg accessor. `what`: device scalers categorical_vocab genus_embeddings
 * family_embeddings species_embeddings schema vocabs species_vocab genus_vocab
 * family_vocab dataset_config model_config retrieval (the RetrievalConfig MAP
 * plus n_reference, NULL-kind when none is set). Returns value / NULL.
 *
 * "vocabs" + "dataset_config" are what a caller needs to score new data
 * correctly: pass the first to resolve_dataset_load as `vocabs` and the
 * second as its `config`. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_get(const resolve_predictor_t* p, const char* what);

/* ========================================================================== */
/* Model suite                                                                */
/* ========================================================================== */

/* Load the model suite in directory `dir` (a manifest.json and the weight files
 * it names). `options` is a MAP or NULL:
 *   device        STRING  "cpu" (default), "cuda" or "cuda:N"
 *   vram_fraction DOUBLE  default 1.0
 *   verify        BOOL    compare each member's size and SHA-256 with the
 *                         manifest before loading (default true)
 *   targets       STRING_ARRAY  load only these suite targets (default all)
 * NULL on error, including a checksum mismatch or a member that does not match
 * the manifest's input contract. */
RESOLVE_CAPI resolve_suite_t* resolve_suite_load(const char* dir, const resolve_value_t* options);

RESOLVE_CAPI void resolve_suite_free(resolve_suite_t* s);

/* Zero-arg accessor. `what`: manifest (the manifest as a value tree: MAPs,
 * LISTs, STRINGs, DOUBLEs, BOOLs and NULLs, key order kept), targets
 * (STRING_ARRAY of the loaded targets), n_encodings (INT), directory (STRING).
 * Returns value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_suite_get(const resolve_suite_t* s, const char* what);

/* Score plots. `header` (may be NULL) and `species` are column tables as for
 * resolve_dataset_load. `options` is a MAP or NULL:
 *   columns       MAP  renames the manifest's input columns; keys as in a
 *                      roles tree (plot_id species_id abundance latitude
 *                      longitude genus family covariates categoricals)
 *   batch_size    INT  per-member forward chunk (default 4096)
 *   keep_members  BOOL keep every member's own prediction (default false)
 * Returns a MAP: "plot_ids" (STRING_ARRAY) and "targets", an ordered MAP of
 * target name -> MAP with name task combine status limit units
 * dispersion_units class_names member_seeds value (INT_ARRAY class codes for a
 * vote; for a spherical mean a DOUBLE_MATRIX of n_plots rows, latitude and
 * longitude in degrees; DOUBLE_ARRAY otherwise) agreement probabilities
 * dispersion members (members x plots; for a spherical mean a MAP of
 * "latitude" and "longitude", each members x plots) n_species n_recognised
 * recognised_share recognised_abundance_share. A statistic the target's
 * combine rule does not define is NULL-kind; NaN marks an undefined value.
 * "columns" is the same prediction as an ordered MAP of column name -> array,
 * one entry per plot (STRING_ARRAY labels, INT_ARRAY codes, DOUBLE_ARRAY
 * numbers), the columns `resolve predict --suite --probabilities --members`
 * writes; "probability_columns" (STRING_ARRAY) names its class-probability
 * columns. */
RESOLVE_CAPI resolve_value_t* resolve_suite_predict_dataframe(
    resolve_suite_t* s, const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* options);

/* The same over CSV files; `header_path` may be NULL or empty. */
RESOLVE_CAPI resolve_value_t* resolve_suite_predict_csv(
    resolve_suite_t* s, const char* header_path, const char* species_path,
    const resolve_value_t* options);

/* Read and validate the manifest in `dir` without loading any member, then
 * compare every member file with it. Returns a MAP: "manifest" (value tree as
 * above) and "problems" (STRING_ARRAY, empty when the suite is intact). NULL
 * when the manifest itself cannot be read or is invalid. */
RESOLVE_CAPI resolve_value_t* resolve_suite_verify(const char* dir);

/* Seal a suite being built: read `dir`/manifest.json, whose members may lack
 * their sha256 and bytes, fill both from the files under `dir`, validate, and
 * write the manifest back. Returns the sealed manifest (value tree as above) /
 * NULL. */
RESOLVE_CAPI resolve_value_t* resolve_suite_seal(const char* dir);

/* ========================================================================== */
/* Fuzzy string index                                                         */
/* ========================================================================== */

/* A Damerau-Levenshtein index over `entries` (a STRING_ARRAY). `options` is a
 * MAP or NULL: max_edit_distance (INT, the largest a query may ask for),
 * case_insensitive (BOOL), damerau (BOOL, transpositions cost 1), buckets
 * (STRING_ARRAY, one bucket per entry, which a query's hint searches first).
 * NULL on error. */
RESOLVE_CAPI resolve_fuzzy_t* resolve_fuzzy_build(const resolve_value_t* entries,
                                                  const resolve_value_t* options);

RESOLVE_CAPI void resolve_fuzzy_free(resolve_fuzzy_t* f);

/* The best matches of each of `needles` (a STRING_ARRAY). `options` is a MAP
 * or NULL: max_edit_distance (INT), top_n (INT), bucket_hint (STRING, for
 * every needle) or bucket_hints (STRING_ARRAY, one per needle). Returns a MAP
 * of equal-length arrays, one row per match, by needle then distance: query
 * (INT_ARRAY, the needle's 0-based position), entry (STRING_ARRAY), id
 * (INT_ARRAY, the entry's 0-based position) and distance (INT_ARRAY); NULL on
 * error. */
RESOLVE_CAPI resolve_value_t* resolve_fuzzy_query(const resolve_fuzzy_t* f,
                                                  const resolve_value_t* needles,
                                                  const resolve_value_t* options);

/* Zero-arg accessor. `what`: size bucket_count max_supported_distance. */
RESOLVE_CAPI resolve_value_t* resolve_fuzzy_get(const resolve_fuzzy_t* f, const char* what);

#endif /* !RESOLVE_CAPI_DYNLOAD */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RESOLVE_CAPI_H */
