/* resolve_capi.h - Flat C ABI facade over the RESOLVE C++ engine.
 *
 * Purpose (issue #17): let the R package build on Windows. R packages there are
 * compiled with Rtools mingw-w64; the only Windows libtorch is MSVC-built, and
 * the two C++ ABIs do not cross. A C ABI does. This header exposes the engine
 * through opaque handles and primitive types ONLY -- no torch::Tensor, no
 * std::string, no STL, no Rcpp in any signature -- so a mingw client can call an
 * MSVC-built `resolve_c` shared library. This is the "lantern" pattern used by
 * the mlverse/torch R package.
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
 * Errors: functions returning a pointer return NULL on failure; functions
 * returning `int` return 0 on success and -1 on failure. In both cases a
 * human-readable message is available from `resolve_last_error()` (thread-local).
 * C++ exceptions never cross the boundary.
 *
 * Runtime binding (issue: CRAN R package): a client that defines
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
 * their input. Returned node is owned by the caller until appended to a list
 * (resolve_list_append_value) or set into a map (resolve_map_set_value), which
 * transfer ownership. */
RESOLVE_CAPI resolve_value_t* resolve_value_new_bool  (int v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int   (int64_t v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double(double v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_string(const char* v);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int_array   (const int64_t* v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double_array(const double*  v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_string_array(const char* const* v, int64_t n);
RESOLVE_CAPI resolve_value_t* resolve_value_new_double_matrix(const double*  v, int64_t nrow, int64_t ncol);
RESOLVE_CAPI resolve_value_t* resolve_value_new_int_matrix   (const int64_t* v, int64_t nrow, int64_t ncol);

/* ---- map setters (key is copied; arrays/matrices are copied) -------------- */
RESOLVE_CAPI void resolve_map_set_null   (resolve_value_t* map, const char* key);
RESOLVE_CAPI void resolve_map_set_bool   (resolve_value_t* map, const char* key, int v);
RESOLVE_CAPI void resolve_map_set_int    (resolve_value_t* map, const char* key, int64_t v);
RESOLVE_CAPI void resolve_map_set_double (resolve_value_t* map, const char* key, double v);
RESOLVE_CAPI void resolve_map_set_string (resolve_value_t* map, const char* key, const char* v);
RESOLVE_CAPI void resolve_map_set_int_array   (resolve_value_t* map, const char* key, const int64_t* v, int64_t n);
RESOLVE_CAPI void resolve_map_set_double_array(resolve_value_t* map, const char* key, const double*  v, int64_t n);
RESOLVE_CAPI void resolve_map_set_string_array(resolve_value_t* map, const char* key, const char* const* v, int64_t n);
RESOLVE_CAPI void resolve_map_set_double_matrix(resolve_value_t* map, const char* key, const double*  v, int64_t nrow, int64_t ncol);
RESOLVE_CAPI void resolve_map_set_int_matrix   (resolve_value_t* map, const char* key, const int64_t* v, int64_t nrow, int64_t ncol);
/* Transfers ownership of `child` into `map` under `key`; do not free `child`. */
RESOLVE_CAPI void resolve_map_set_value  (resolve_value_t* map, const char* key, resolve_value_t* child);

/* ---- list builders -------------------------------------------------------- */
/* Transfers ownership of `child` into `list`; do not free `child`. */
RESOLVE_CAPI void resolve_list_append_value(resolve_value_t* list, resolve_value_t* child);

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

/* Pin libtorch's intra-op / inter-op thread pools (<=0 keeps the default).
 * Best-effort: removes the worker threads whose teardown join crashes the
 * Rscript.exe launcher on Windows (issue #18). Call at startup. */
RESOLVE_CAPI int resolve_capi_set_thread_pools(int intraop_threads, int interop_threads);

/* Install the Windows crash handler: turn an unhandled native fault into an
 * immediate TerminateProcess instead of a JIT-debugger hang (issue #19) or a
 * teardown access violation (issue #18). No-op off Windows. Exits with
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

/* `roles`, `targets`, `config` are MAP value trees (see r/src marshaling).
 * Return NULL on error. */
RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_csv(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_csv_with_schema(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_dataset_t* schema_source, const resolve_value_t* config);

/* Vocabulary-reusing loaders (issue #102). `vocabs` is a MAP value tree: a
 * schema tree (species_vocab / genus_vocab / family_vocab / targets, as emitted
 * by resolve_predictor_get(p, "vocabs") or resolve_dataset_get(ds, "vocabs"))
 * plus an optional "categorical_vocab" sub-map of {column -> {name -> code}}.
 *
 * The resulting dataset is encoded in the TRAINING model's integer-code
 * namespace: every non-hash encoder indexes an embedding table by a code that
 * is a function of the file its vocab was fitted on, so a plain from_csv on new
 * data looks up the wrong rows and predicts wrongly with no error. NULL on
 * error, including when `vocabs` carries no species vocabulary (a checkpoint
 * written before issue #102). */
RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_csv_with_vocabs(
    const char* header_path, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_species_csv(
    const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_species_csv_with_vocabs(
    const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config);

/* In-memory (DataFrame) loaders (issue #22). `header` / `species` are MAP value
 * trees: an ordered (column name -> STRING_ARRAY) mapping, i.e. a data.frame
 * with every cell stringified (missing value = empty string). Identical result
 * to the from_csv* verbs on the equivalent CSV; no disk round-trip. NULL on
 * error. */
RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_dataframe(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_dataframe_header(
    const resolve_value_t* header, const char* species_path,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_species_dataframe(
    const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_dataframe_with_schema(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_dataset_t* schema_source, const resolve_value_t* config);

/* In-memory counterparts of the vocabulary-reusing CSV loaders above. */
RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_dataframe_with_vocabs(
    const resolve_value_t* header, const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config);

RESOLVE_CAPI resolve_dataset_t* resolve_dataset_from_species_dataframe_with_vocabs(
    const resolve_value_t* species,
    const resolve_value_t* roles, const resolve_value_t* targets,
    const resolve_value_t* vocabs, const resolve_value_t* config);

RESOLVE_CAPI void resolve_dataset_free(resolve_dataset_t* ds);

/* String-dispatched accessor. Returns a freshly-allocated value tree (caller
 * frees), or NULL on error. `what` is one of:
 *   coordinates covariates hash_embedding species_ids species_vector
 *   genus_ids family_ids unknown_fraction unknown_count categorical_ids
 *   categorical_vocab targets schema vocabs plot_ids species_vocab n_plots
 *   config has_raw_species_data raw_species_ids raw_weights plot_offsets
 *   taxonomy_vocab pool_genus_ids pool_family_ids pool_weights pool_mask
 *   pool_has_cover has_pool_data
 * Optional-tensor accessors return a NULL-kind value when the tensor is
 * undefined/empty. */
RESOLVE_CAPI resolve_value_t* resolve_dataset_get(const resolve_dataset_t* ds, const char* what);

/* ========================================================================== */
/* Model                                                                      */
/* ========================================================================== */

/* `schema` and `config` are MAP value trees. Return NULL on error. */
RESOLVE_CAPI resolve_model_t* resolve_model_create(
    const resolve_value_t* schema, const resolve_value_t* config);

RESOLVE_CAPI void resolve_model_free(resolve_model_t* m);

/* Input-taking methods. `inputs` is a MAP carrying the relevant tensors
 * (continuous required; genus_ids/family_ids/species_ids/species_vector/
 * pool_genus_ids/pool_family_ids/pool_weights/pool_mask/pool_has_cover/
 * categorical_ids optional, plus "target" string for forward_single).
 * `method` is one of: forward get_latent forward_with_aux forward_single
 * encode_with_activations get_gate_probs. Returns value tree / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_model_call(
    resolve_model_t* m, const char* method, const resolve_value_t* inputs);

/* Zero-arg accessor. `what` is one of: latent_dim species_encoding
 * uses_explicit_vector uses_moe n_experts genus_weights family_weights
 * species_weights. Returns value tree / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_model_get(const resolve_model_t* m, const char* what);

/* State mutations. Return 0 / -1. mode!=0 => train, mode==0 => eval. */
RESOLVE_CAPI int resolve_model_set_train(resolve_model_t* m, int mode);
RESOLVE_CAPI int resolve_model_to_device(resolve_model_t* m, const char* device);
RESOLVE_CAPI int resolve_model_set_traits(resolve_model_t* m, const resolve_value_t* traits);

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

/* Train. Returns a MAP value tree (TrainResult) / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_fit(resolve_trainer_t* t);

/* Save checkpoint. `metadata` may be NULL (or a NULL-kind value). 0 / -1. */
RESOLVE_CAPI int resolve_trainer_save(
    resolve_trainer_t* t, const char* path, const resolve_value_t* metadata);

/* Load weights/scalers/vocab into this trainer in place. 0 / -1. */
RESOLVE_CAPI int resolve_trainer_load_state(
    resolve_trainer_t* t, const char* path, const char* device, double vram_fraction);

/* Zero-arg accessor. `what`: scalers config test_indices train_indices
 * test_plot_ids train_plot_ids categorical_vocab effective_batch_size.
 * Returns value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_get(const resolve_trainer_t* t, const char* what);

/* Evaluation. `kind`: diagnostics calibration residuals
 * classification_predictions. `args` MAP carries "target_name" and optional
 * "n_bins". Returns value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_trainer_compute(
    resolve_trainer_t* t, const char* kind, const resolve_value_t* args);

RESOLVE_CAPI resolve_value_t* resolve_trainer_cross_validate(
    resolve_trainer_t* t, int n_folds, int seed);

/* `spatial_cfg` MAP carries lat_size/lon_size/balance. */
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
 * with "predictions", "targets", "plot_ids" and (if return_latent) "latent". */
RESOLVE_CAPI resolve_value_t* resolve_predictor_predict(
    resolve_predictor_t* p, const resolve_value_t* inputs, int return_latent);

/* Predict on a dataset. batch_size = -1 keeps the one-shot path. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_predict_dataset(
    resolve_predictor_t* p, const resolve_dataset_t* ds, int return_latent, int64_t batch_size);

/* get_embeddings. `inputs` MAP carries coordinates/covariates/hash_embedding +
 * optional genus_ids/family_ids. Returns a DOUBLE_MATRIX value / NULL. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_get_embeddings(
    resolve_predictor_t* p, const resolve_value_t* inputs);

RESOLVE_CAPI int resolve_predictor_optimize_for_inference(resolve_predictor_t* p);

/* Zero-arg accessor. `what`: device scalers categorical_vocab genus_embeddings
 * family_embeddings species_embeddings schema vocabs species_vocab genus_vocab
 * family_vocab dataset_config. Returns value / NULL.
 *
 * "vocabs" + "dataset_config" are what a caller needs to score new data
 * correctly (issue #102): pass the first to a resolve_dataset_from_*_with_vocabs
 * loader and the second as that loader's `config`. */
RESOLVE_CAPI resolve_value_t* resolve_predictor_get(const resolve_predictor_t* p, const char* what);

#endif /* !RESOLVE_CAPI_DYNLOAD */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* RESOLVE_CAPI_H */
