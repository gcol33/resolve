"""
Resolve Core - the Python bindings of the RESOLVE C++ engine.

    from resolve_core import ResolveDataset, ResolveModel, Trainer, Predictor, ...

The CUDA caching allocator reads ``PYTORCH_CUDA_ALLOC_CONF`` when it first
allocates on a device, not when ``torch`` is imported, so importing this
package sets the engine's default (``configure_cuda_allocator``) in time for
any run that has not yet touched the GPU. A value already set is kept.
"""

import os as _os

# _resolve_core wraps engine tensors as torch.Tensor objects through
# torch_python, which needs torch's Python tensor class registered first, and
# on Windows torch's DLL directory added to the search path: both happen when
# torch is imported.
import torch  # noqa: F401

# Every public name the compiled extension registers is re-exported here, and
# listed in __all__ below. `resolve_core.X` is the supported import path for all
# of them; reaching into `resolve_core._resolve_core` is never necessary.
# tests/core/test_bindings_surface.py enforces both halves of that, so a new
# binding that skips this block fails CI rather than being reachable only
# through the private module.
try:
    from ._resolve_core import (
        # Enums
        TaskType,
        TransformType,
        SpeciesEncodingMode,
        LossConfigMode,
        SelectionMode,
        RepresentationMode,
        NormalizationMode,
        AggregationMode,
        MissingValuePolicy,
        TargetMissingPolicy,
        HoldOutMode,
        MoERoutingType,
        MoEPlacement,
        LRSchedulerType,
        SelectionMetric,
        ActivationType,
        NormLayerType,
        # Pool-style species encoder weighting (rank_pool / transformer modes)
        PoolWeighting,
        # Architecture enums
        EncoderArchitecture,
        GNNType,
        GraphConstructionMode,
        SpeciesEdgeType,
        SpeciesGraph,
        build_species_graph,
        TraitInteractionMode,
        TabMAggregation,
        TransformerPooling,
        ParallelAggregation,
        # Config structs
        TargetConfig,
        ResolveSchema,
        ModelConfig,
        TrainConfig,
        TrainResult,
        RunMetadata,
        BaselineMetrics,
        LayerDiagnostics,
        NetworkDiagnostics,
        ResolvePredictions,
        Scalers,
        DatasetConfig,
        SpatialBlockConfig,
        ModelForwardResult,
        RetrievalConfig,
        # Architecture configs
        FTTransformerConfig,
        TabNetConfig,
        SAINTConfig,
        GNNConfig,
        TraitNetConfig,
        ExcelFormerConfig,
        HeterogeneousGNNConfig,
        TabMConfig,
        ParallelBranchConfig,
        ParallelLayersConfig,
        # Calibration and residual analysis
        CalibrationBin,
        CalibrationResult,
        ResidualAnalysis,
        ClassificationPredictions,
        CrossValidationResult,
        # Species encoding
        TaxonomyVocab,
        CategoricalVocab,
        SpeciesVocab,
        EncodingVocab,
        SpeciesRecord,
        RankPoolEncoder,
        RankPoolEncodedData,
        EmbeddingEncoder,
        EmbeddingEncodedData,
        UnknownSpeciesStats,
        compute_unknown_species_stats,
        # Training-time vocabularies + checkpoint -> DatasetConfig
        ExternalVocabs,
        external_vocabs_from_schema,
        dataset_config_from_checkpoint,
        effective_selection,
        model_config_for_dataset,
        target_specs_from_schema,
        # Model
        ResolveModel,
        # Training
        Trainer,
        # Inference
        Predictor,
        # Metrics
        Metrics,
        ClassificationMetrics,
        ConfidenceMetrics,
        # Role mapping
        RoleMapping,
        TargetSpec,
        # Dataset
        ResolveDataset,
        # Pretraining
        MaskStrategy,
        PretrainConfig,
        PretrainResult,
        JEPAPretrainer,
        SCARFPretrainer,
        MLMPretrainConfig,
        MaskedSpeciesPretrainer,
        # VAE
        VAEConfig,
        VAEPretrainResult,
        VAEPretrainer,
        # Native fuzzy-string index submodule (fuzzy.FuzzyIndex, fuzzy.Match)
        fuzzy,
        # Model suites: a released set of checkpoints scored as one model
        SuiteCombine,
        SuiteTargetStatus,
        SuiteInputContract,
        SuiteTargetInputs,
        SuiteMember,
        SuiteTarget,
        SuiteManifest,
        SuiteColumns,
        SuitePredictor,
        SuitePredictions,
        SuiteTargetPrediction,
        CombinedPrediction,
        combine_vote,
        combine_mean,
        combine_circular,
        combine_spherical,
        EARTH_RADIUS_KM,
        bearing_from_components,
        SpeciesRecognition,
        compute_species_recognition,
        sha256_file,
        suite_prediction_columns,
        # Storage failures (an OSError), GPU memory and process hardening
        StorageError,
        default_cuda_alloc_conf,
        set_vram_fraction,
        set_thread_pools,
        install_crash_handler,
        _configure_cuda_allocator,
        _signal_work_complete,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import resolve_core C++ extension: {e}\n"
        "Make sure the package was built with CMake and libtorch is available."
    ) from e


def configure_cuda_allocator(force: bool = False) -> str:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` to the engine's default and return it.

    The default (``default_cuda_alloc_conf()``) sets the garbage-collection
    threshold and the split-size limit that keep Windows WDDM fragmentation in
    check, and prefixes ``expandable_segments:True`` off Windows (the
    cuMemMap-backed allocator does not exist there). A value already set is
    kept unless ``force`` is true; the return value is the setting in effect.
    It takes effect when it runs before the process's first CUDA allocation;
    importing ``resolve_core`` calls it.
    """
    value = _configure_cuda_allocator(force)
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    return value


configure_cuda_allocator()

# The extension installs the Windows crash handler at import, so a native
# fault in a headless worker fails fast with its NTSTATUS instead of hanging on
# the JIT debugger. A clean interpreter shutdown after a successful run marks
# the work complete, so a late libtorch teardown fault then exits 0 rather than
# being reported as a crash.
import atexit as _atexit  # noqa: E402

_atexit.register(_signal_work_complete)

# Attach the high-level in-memory loaders ResolveDataset.from_pandas /
# .from_dataframe, wrapping the low-level from_columns* bindings.
from . import _from_pandas as _from_pandas_mod

_from_pandas_mod.install()

# Attach SuitePredictor.predict (pandas DataFrames or CSV paths) and
# SuitePredictions.to_pandas over the low-level suite bindings.
from . import _suite as _suite_mod

_suite_mod.install()

# Report the version of the engine actually loaded, not a literal that can go
# stale against the .pyd. bindings.cpp sets this from resolve::VERSION, which
# tools/version.py keeps equal to the repo-root VERSION file.
from ._resolve_core import __version__ as __version__  # noqa: E402

__all__ = [
    # Enums
    "TaskType",
    "TransformType",
    "SpeciesEncodingMode",
    "LossConfigMode",
    "SelectionMode",
    "RepresentationMode",
    "NormalizationMode",
    "AggregationMode",
    "MissingValuePolicy",
    "TargetMissingPolicy",
    "HoldOutMode",
    "MoERoutingType",
    "MoEPlacement",
    "LRSchedulerType",
    "SelectionMetric",
    "ActivationType",
    "NormLayerType",
    "PoolWeighting",
    # Architecture enums
    "EncoderArchitecture",
    "GNNType",
    "GraphConstructionMode",
    "SpeciesEdgeType",
    "SpeciesGraph",
    "build_species_graph",
    "TraitInteractionMode",
    "TabMAggregation",
    "TransformerPooling",
    "ParallelAggregation",
    # Config structs
    "TargetConfig",
    "ResolveSchema",
    "ModelConfig",
    "TrainConfig",
    "TrainResult",
    "RunMetadata",
    "BaselineMetrics",
    "LayerDiagnostics",
    "NetworkDiagnostics",
    "ResolvePredictions",
    "Scalers",
    "DatasetConfig",
    "SpatialBlockConfig",
    "ModelForwardResult",
    "RetrievalConfig",
    # Architecture configs
    "FTTransformerConfig",
    "TabNetConfig",
    "SAINTConfig",
    "GNNConfig",
    "TraitNetConfig",
    "ExcelFormerConfig",
    "HeterogeneousGNNConfig",
    "TabMConfig",
    "ParallelBranchConfig",
    "ParallelLayersConfig",
    # Calibration and residual analysis
    "CalibrationBin",
    "CalibrationResult",
    "ResidualAnalysis",
    "ClassificationPredictions",
    "CrossValidationResult",
    # Species encoding
    "TaxonomyVocab",
    "CategoricalVocab",
    "SpeciesVocab",
    "EncodingVocab",
    "SpeciesRecord",
    "RankPoolEncoder",
    "RankPoolEncodedData",
    "EmbeddingEncoder",
    "EmbeddingEncodedData",
    "UnknownSpeciesStats",
    "compute_unknown_species_stats",
    # Training-time vocabularies + checkpoint -> DatasetConfig
    "ExternalVocabs",
    "external_vocabs_from_schema",
    "dataset_config_from_checkpoint",
    "effective_selection",
    "model_config_for_dataset",
    "target_specs_from_schema",
    # Model
    "ResolveModel",
    # Training
    "Trainer",
    # Inference
    "Predictor",
    # Metrics
    "Metrics",
    "ClassificationMetrics",
    "ConfidenceMetrics",
    # Role mapping
    "RoleMapping",
    "TargetSpec",
    # Dataset
    "ResolveDataset",
    # Pretraining
    "MaskStrategy",
    "PretrainConfig",
    "PretrainResult",
    "JEPAPretrainer",
    "SCARFPretrainer",
    "MLMPretrainConfig",
    "MaskedSpeciesPretrainer",
    # VAE
    "VAEConfig",
    "VAEPretrainResult",
    "VAEPretrainer",
    # Native fuzzy-string index submodule
    "fuzzy",
    # Model suites
    "SuiteCombine",
    "SuiteTargetStatus",
    "SuiteInputContract",
    "SuiteTargetInputs",
    "SuiteMember",
    "SuiteTarget",
    "SuiteManifest",
    "SuiteColumns",
    "SuitePredictor",
    "SuitePredictions",
    "SuiteTargetPrediction",
    "CombinedPrediction",
    "combine_vote",
    "combine_mean",
    "combine_circular",
    "combine_spherical",
    "EARTH_RADIUS_KM",
    "bearing_from_components",
    "SpeciesRecognition",
    "compute_species_recognition",
    "sha256_file",
    "suite_prediction_columns",
    # Storage failures
    "StorageError",
    # GPU memory management
    "set_vram_fraction",
    "configure_cuda_allocator",
    "default_cuda_alloc_conf",
    # Process hardening (Windows teardown and crashes)
    "set_thread_pools",
    "install_crash_handler",
]
