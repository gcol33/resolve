/* resolve_capi_symbols.inc - the single source of truth for every symbol in the
 * resolve_c C ABI (resolve_capi.h).
 *
 * X-macro list. The includer defines TWO macros before including this file and
 * undefines them after:
 *
 *   RESOLVE_SYM(ret, name, params, args)   - a function returning `ret`
 *   RESOLVE_SYM_VOID(name, params, args)   - a function returning void
 *
 * where `params` is the FULL parenthesized parameter list (types + names) and
 * `args` is the parenthesized forwarding list (names only). This one list drives
 * both the function-pointer table (member declarations), the runtime symbol
 * binding loop, and the identically-named forwarders in resolve_capi_dynload.h,
 * so a new C-ABI function is added in exactly one place.
 *
 * Keep this list byte-for-byte in step with the prototypes in resolve_capi.h.
 * The r-cmd-check `vendor-drift` CI job diffs the vendored copy against this
 * canonical one; extend that guard to cover this file too.
 */

/* ---- Error reporting ------------------------------------------------------ */
RESOLVE_SYM(const char*, resolve_last_error, (void), ())

/* ---- resolve_value_t: construction ---------------------------------------- */
RESOLVE_SYM(resolve_value_t*, resolve_value_new_null, (void), ())
RESOLVE_SYM(resolve_value_t*, resolve_value_new_map,  (void), ())
RESOLVE_SYM(resolve_value_t*, resolve_value_new_list, (void), ())
RESOLVE_SYM_VOID(resolve_value_free, (resolve_value_t* v), (v))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_bool,   (int v), (v))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_int,    (int64_t v), (v))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_double, (double v), (v))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_string, (const char* v), (v))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_int_array,    (const int64_t* v, int64_t n), (v, n))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_double_array, (const double* v, int64_t n), (v, n))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_string_array, (const char* const* v, int64_t n), (v, n))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_double_matrix, (const double* v, int64_t nrow, int64_t ncol), (v, nrow, ncol))
RESOLVE_SYM(resolve_value_t*, resolve_value_new_int_matrix,    (const int64_t* v, int64_t nrow, int64_t ncol), (v, nrow, ncol))

/* ---- resolve_value_t: map setters (all void) ------------------------------ */
RESOLVE_SYM_VOID(resolve_map_set_null,   (resolve_value_t* map, const char* key), (map, key))
RESOLVE_SYM_VOID(resolve_map_set_bool,   (resolve_value_t* map, const char* key, int v), (map, key, v))
RESOLVE_SYM_VOID(resolve_map_set_int,    (resolve_value_t* map, const char* key, int64_t v), (map, key, v))
RESOLVE_SYM_VOID(resolve_map_set_double, (resolve_value_t* map, const char* key, double v), (map, key, v))
RESOLVE_SYM_VOID(resolve_map_set_string, (resolve_value_t* map, const char* key, const char* v), (map, key, v))
RESOLVE_SYM_VOID(resolve_map_set_int_array,    (resolve_value_t* map, const char* key, const int64_t* v, int64_t n), (map, key, v, n))
RESOLVE_SYM_VOID(resolve_map_set_double_array, (resolve_value_t* map, const char* key, const double* v, int64_t n), (map, key, v, n))
RESOLVE_SYM_VOID(resolve_map_set_string_array, (resolve_value_t* map, const char* key, const char* const* v, int64_t n), (map, key, v, n))
RESOLVE_SYM_VOID(resolve_map_set_double_matrix, (resolve_value_t* map, const char* key, const double* v, int64_t nrow, int64_t ncol), (map, key, v, nrow, ncol))
RESOLVE_SYM_VOID(resolve_map_set_int_matrix,    (resolve_value_t* map, const char* key, const int64_t* v, int64_t nrow, int64_t ncol), (map, key, v, nrow, ncol))
RESOLVE_SYM_VOID(resolve_map_set_value,  (resolve_value_t* map, const char* key, resolve_value_t* child), (map, key, child))

/* ---- resolve_value_t: list builders --------------------------------------- */
RESOLVE_SYM_VOID(resolve_list_append_value, (resolve_value_t* list, resolve_value_t* child), (list, child))

/* ---- resolve_value_t: readers --------------------------------------------- */
RESOLVE_SYM(resolve_value_kind_t, resolve_value_kind, (const resolve_value_t* v), (v))
RESOLVE_SYM(int,         resolve_value_as_bool,   (const resolve_value_t* v), (v))
RESOLVE_SYM(int64_t,     resolve_value_as_int,    (const resolve_value_t* v), (v))
RESOLVE_SYM(double,      resolve_value_as_double, (const resolve_value_t* v), (v))
RESOLVE_SYM(const char*, resolve_value_as_string, (const resolve_value_t* v), (v))
RESOLVE_SYM(const int64_t*, resolve_value_as_int_array,    (const resolve_value_t* v, int64_t* n), (v, n))
RESOLVE_SYM(const double*,  resolve_value_as_double_array, (const resolve_value_t* v, int64_t* n), (v, n))
RESOLVE_SYM(const double*,  resolve_value_as_double_matrix, (const resolve_value_t* v, int64_t* nrow, int64_t* ncol), (v, nrow, ncol))
RESOLVE_SYM(const int64_t*, resolve_value_as_int_matrix,    (const resolve_value_t* v, int64_t* nrow, int64_t* ncol), (v, nrow, ncol))
RESOLVE_SYM(int64_t,     resolve_value_string_array_size, (const resolve_value_t* v), (v))
RESOLVE_SYM(const char*, resolve_value_string_at,         (const resolve_value_t* v, int64_t i), (v, i))

/* ---- resolve_value_t: map readers ----------------------------------------- */
RESOLVE_SYM(int64_t,                resolve_map_size,     (const resolve_value_t* map), (map))
RESOLVE_SYM(const char*,            resolve_map_key_at,   (const resolve_value_t* map, int64_t i), (map, i))
RESOLVE_SYM(const resolve_value_t*, resolve_map_value_at, (const resolve_value_t* map, int64_t i), (map, i))
RESOLVE_SYM(const resolve_value_t*, resolve_map_get,      (const resolve_value_t* map, const char* key), (map, key))

/* ---- resolve_value_t: list readers ---------------------------------------- */
RESOLVE_SYM(int64_t,                resolve_list_size, (const resolve_value_t* list), (list))
RESOLVE_SYM(const resolve_value_t*, resolve_list_at,   (const resolve_value_t* list, int64_t i), (list, i))

/* ---- Free functions ------------------------------------------------------- */
RESOLVE_SYM(const char*, resolve_capi_version, (void), ())
RESOLVE_SYM(int, resolve_capi_set_vram_fraction, (double fraction, int device_index), (fraction, device_index))
RESOLVE_SYM(resolve_value_t*, resolve_capi_configure_cuda_allocator, (int force), (force))
RESOLVE_SYM(int, resolve_capi_set_thread_pools, (int intraop_threads, int interop_threads), (intraop_threads, interop_threads))
RESOLVE_SYM(int, resolve_capi_install_crash_handler, (int shutdown_exit_code), (shutdown_exit_code))
RESOLVE_SYM(int, resolve_capi_signal_work_complete, (void), ())

/* ---- Metrics -------------------------------------------------------------- */
RESOLVE_SYM(int, resolve_metric_band_accuracy, (const double* pred, const double* target, int64_t n, double threshold, double* out), (pred, target, n, threshold, out))
RESOLVE_SYM(int, resolve_metric_mae,           (const double* pred, const double* target, int64_t n, double* out), (pred, target, n, out))
RESOLVE_SYM(int, resolve_metric_rmse,          (const double* pred, const double* target, int64_t n, double* out), (pred, target, n, out))
RESOLVE_SYM(int, resolve_metric_smape,         (const double* pred, const double* target, int64_t n, double eps, double* out), (pred, target, n, eps, out))
RESOLVE_SYM(int, resolve_metric_accuracy,      (const double* pred, const double* target, int64_t n, double* out), (pred, target, n, out))
RESOLVE_SYM(int, resolve_metric_r_squared,     (const double* pred, const double* target, int64_t n, double* out), (pred, target, n, out))

/* ---- Dataset -------------------------------------------------------------- */
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_csv, (const char* header_path, const char* species_path, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_value_t* config), (header_path, species_path, roles, targets, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_csv_with_schema, (const char* header_path, const char* species_path, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_dataset_t* schema_source, const resolve_value_t* config), (header_path, species_path, roles, targets, schema_source, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_species_csv, (const char* species_path, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_value_t* config), (species_path, roles, targets, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_dataframe, (const resolve_value_t* header, const resolve_value_t* species, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_value_t* config), (header, species, roles, targets, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_dataframe_header, (const resolve_value_t* header, const char* species_path, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_value_t* config), (header, species_path, roles, targets, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_species_dataframe, (const resolve_value_t* species, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_value_t* config), (species, roles, targets, config))
RESOLVE_SYM(resolve_dataset_t*, resolve_dataset_from_dataframe_with_schema, (const resolve_value_t* header, const resolve_value_t* species, const resolve_value_t* roles, const resolve_value_t* targets, const resolve_dataset_t* schema_source, const resolve_value_t* config), (header, species, roles, targets, schema_source, config))
RESOLVE_SYM_VOID(resolve_dataset_free, (resolve_dataset_t* ds), (ds))
RESOLVE_SYM(resolve_value_t*, resolve_dataset_get, (const resolve_dataset_t* ds, const char* what), (ds, what))

/* ---- Model ---------------------------------------------------------------- */
RESOLVE_SYM(resolve_model_t*, resolve_model_create, (const resolve_value_t* schema, const resolve_value_t* config), (schema, config))
RESOLVE_SYM_VOID(resolve_model_free, (resolve_model_t* m), (m))
RESOLVE_SYM(resolve_value_t*, resolve_model_call, (resolve_model_t* m, const char* method, const resolve_value_t* inputs), (m, method, inputs))
RESOLVE_SYM(resolve_value_t*, resolve_model_get, (const resolve_model_t* m, const char* what), (m, what))
RESOLVE_SYM(int, resolve_model_set_train, (resolve_model_t* m, int mode), (m, mode))
RESOLVE_SYM(int, resolve_model_to_device, (resolve_model_t* m, const char* device), (m, device))
RESOLVE_SYM(int, resolve_model_set_traits, (resolve_model_t* m, const resolve_value_t* traits), (m, traits))

/* ---- Trainer -------------------------------------------------------------- */
RESOLVE_SYM(resolve_trainer_t*, resolve_trainer_create, (resolve_model_t* model, const resolve_value_t* config), (model, config))
RESOLVE_SYM_VOID(resolve_trainer_free, (resolve_trainer_t* t), (t))
RESOLVE_SYM(int, resolve_trainer_prepare_data, (resolve_trainer_t* t, const resolve_value_t* inputs, double test_size, int seed), (t, inputs, test_size, seed))
RESOLVE_SYM(int, resolve_trainer_prepare_data_from_dataset, (resolve_trainer_t* t, const resolve_dataset_t* ds, double test_size, int seed), (t, ds, test_size, seed))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_fit, (resolve_trainer_t* t), (t))
RESOLVE_SYM(int, resolve_trainer_save, (resolve_trainer_t* t, const char* path, const resolve_value_t* metadata), (t, path, metadata))
RESOLVE_SYM(int, resolve_trainer_load_state, (resolve_trainer_t* t, const char* path, const char* device, double vram_fraction), (t, path, device, vram_fraction))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_get, (const resolve_trainer_t* t, const char* what), (t, what))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_compute, (resolve_trainer_t* t, const char* kind, const resolve_value_t* args), (t, kind, args))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_cross_validate, (resolve_trainer_t* t, int n_folds, int seed), (t, n_folds, seed))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_cross_validate_spatial, (resolve_trainer_t* t, const resolve_value_t* spatial_cfg, int n_folds, int seed), (t, spatial_cfg, n_folds, seed))
RESOLVE_SYM(resolve_value_t*, resolve_trainer_predict, (resolve_trainer_t* t, const resolve_value_t* inputs), (t, inputs))
RESOLVE_SYM(resolve_value_t*, resolve_load_train_config, (const char* path), (path))
RESOLVE_SYM(resolve_value_t*, resolve_load_run_metadata, (const char* path), (path))

/* ---- Predictor ------------------------------------------------------------ */
RESOLVE_SYM(resolve_predictor_t*, resolve_predictor_load, (const char* path, const char* device, double vram_fraction), (path, device, vram_fraction))
RESOLVE_SYM_VOID(resolve_predictor_free, (resolve_predictor_t* p), (p))
RESOLVE_SYM(resolve_value_t*, resolve_predictor_predict, (resolve_predictor_t* p, const resolve_value_t* inputs, int return_latent), (p, inputs, return_latent))
RESOLVE_SYM(resolve_value_t*, resolve_predictor_predict_dataset, (resolve_predictor_t* p, const resolve_dataset_t* ds, int return_latent, int64_t batch_size), (p, ds, return_latent, batch_size))
RESOLVE_SYM(resolve_value_t*, resolve_predictor_get_embeddings, (resolve_predictor_t* p, const resolve_value_t* inputs), (p, inputs))
RESOLVE_SYM(int, resolve_predictor_optimize_for_inference, (resolve_predictor_t* p), (p))
RESOLVE_SYM(resolve_value_t*, resolve_predictor_get, (const resolve_predictor_t* p, const char* what), (p, what))
