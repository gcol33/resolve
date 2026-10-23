#pragma once

// The RESOLVE CLI's flag tables: one CommandSpec per subcommand.
//
// This is the single place a flag is declared. The parser accepts exactly what
// is listed here (anything else exits non-zero), `resolve help` is rendered
// from these rows, and each command reads its values back by the same name.
// The DatasetConfig, ModelConfig, TrainConfig and architecture flags of
// `train` are generated from the field registry (config_flags.hpp).
//
// Header-only and torch-free so the tables can be exercised from a unit test
// without spawning a process.

#include "arg_parser.hpp"
#include "config_flags.hpp"

namespace resolve_cli {

// Roles shared by `train` and `predict`: both read the same two CSVs with the
// same column semantics.
inline void append_role_flags(std::vector<FlagSpec>& flags) {
    flags.push_back({"--plot-id", Arity::Value, "COL", "plot_id",
                     "Column name for plot ID"});
    flags.push_back({"--species-id", Arity::Value, "COL", "species_id",
                     "Column name for species ID"});
    flags.push_back({"--abundance", Arity::Value, "COL", "",
                     "Column name for abundance (optional)"});
    flags.push_back({"--lon", Arity::Value, "COL", "",
                     "Column name for longitude (optional)"});
    flags.push_back({"--lat", Arity::Value, "COL", "",
                     "Column name for latitude (optional)"});
    flags.push_back({"--genus", Arity::Value, "COL", "",
                     "Column name for genus (optional)"});
    flags.push_back({"--family", Arity::Value, "COL", "",
                     "Column name for family (optional)"});
    flags.push_back({"--covariate", Arity::Repeatable, "COL", "",
                     "Header column to use as a continuous covariate.\n"
                     "Repeat for each one; order defines the model's\n"
                     "covariate order."});
    flags.push_back({"--categorical", Arity::Repeatable, "COL", "",
                     "Header column to use as a categorical covariate.\n"
                     "Factorized at load time and embedded. Repeat for\n"
                     "each one. Must not also be a --covariate."});
}

inline const CommandSpec& train_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--header", Arity::Value, "PATH", "",
                         "Path to header CSV file (plot-level data)"});
        flags.push_back({"--species", Arity::Value, "PATH", "",
                         "Path to species CSV file (species occurrences)"});
        flags.push_back({"--output", Arity::Value, "PATH", "model.pt",
                         "Output path for the trained model"});
        append_role_flags(flags);
        flags.push_back({"--target", Arity::Repeatable, "SPEC", "",
                         "Target column and type. Repeat per target:\n"
                         "  COL:regression          continuous target\n"
                         "  COL:regression:log1p    fit on log1p(y),\n"
                         "                          predictions inverted\n"
                         "  COL:classification:N    N-class target\n"
                         "A bare COL means COL:regression. Append :mask to\n"
                         "keep plots that lack this label: they train the\n"
                         "other targets only (default :drop leaves them out)."});

        // Every DatasetConfig, ModelConfig and TrainConfig field, then every
        // field of every architecture sub-config, from the field registry.
        append_top_level_flags(flags);
        append_architecture_flags(flags);
        flags.push_back({"--parallel-branch", Arity::Repeatable, "SPEC", "",
                         "One branch of the parallel block. Repeat per\n"
                         "branch; the order is the branch order:\n"
                         "  DIMS                    widths, e.g. 256,128\n"
                         "  DIMS:ACTIVATION         gelu, relu, silu, ...\n"
                         "  DIMS:ACT:NORMALIZATION  batch_norm, layer_norm,\n"
                         "                          group_norm, rms_norm, none\n"
                         "  DIMS:ACT:NORM:DROPOUT   rate inside the branch\n"
                         "  DIMS:ACT:NORM:DROP:WEIGHT  what the branch is\n"
                         "                          worth in the aggregation\n"
                         "Needs --parallel-enabled; the block then replaces\n"
                         "the encoder's MLP tail."});

        flags.push_back({"--resume", Arity::Value, "PATH", "",
                         "Continue the run that wrote PATH (best.pt or\n"
                         "checkpoint_<N>.pt under its --checkpoint-dir) at\n"
                         "the epoch after it. Pass the run's own data,\n"
                         "model and training flags and --seed, so the split\n"
                         "and recipe are the ones it trained on. Not\n"
                         "combined with --cv-folds."});
        flags.push_back({"--test-size", Arity::Value, "FLOAT", "0.2",
                         "Held-out fraction early stopping watches, drawn\n"
                         "at random with --seed. 0 holds nothing out and\n"
                         "needs --fixed-epochs."});
        flags.push_back({"--stopping-plots", Arity::Value, "PATH", "",
                         "File of plot IDs, one per line, to hold out for\n"
                         "early stopping instead of a random --test-size\n"
                         "fraction: a withheld spatial block or database,\n"
                         "or the fold a published protocol selects its\n"
                         "epoch on. Best epoch, final metrics and baselines\n"
                         "are read on exactly these plots; every other plot\n"
                         "trains. Not combined with --test-size."});
        flags.push_back({"--seed", Arity::Value, "N", "42",
                         "Random seed. Seeds the global torch RNG before the\n"
                         "model is constructed (weight init) and drives the\n"
                         "train/test split, the epoch shuffle, and the\n"
                         "cross-validation folds, so two identical runs\n"
                         "produce the same model."});
        flags.push_back({"--device", Arity::Value, "DEVICE", "cpu",
                         "Device to train on: cpu, cuda (the current CUDA\n"
                         "device) or cuda:N (CUDA device N). A CUDA device\n"
                         "that is not available is an error."});

        // Cross-validation
        flags.push_back({"--cv-folds", Arity::Value, "N", "0",
                         "Run N-fold cross-validation BEFORE the final fit\n"
                         "and report per-fold plus mean +/- std metrics\n"
                         "(0 = skip). Each fold starts from the untrained\n"
                         "weights; the trainer's split and weights are\n"
                         "restored afterwards, so the saved model is the\n"
                         "same one a run without this flag produces."});
        flags.push_back({"--cv-spatial", Arity::Flag, "", "",
                         "Make the folds spatial blocks instead of random\n"
                         "rows. Requires --lon / --lat."});
        flags.push_back({"--cv-lat-size", Arity::Value, "FLOAT", "1.0",
                         "Spatial block height in degrees latitude"});
        flags.push_back({"--cv-lon-size", Arity::Value, "FLOAT", "1.0",
                         "Spatial block width in degrees longitude"});
        flags.push_back({"--cv-balance", Arity::Flag, "", "",
                         "Assign spatial blocks to folds by greedy\n"
                         "bin-packing instead of round-robin"});

        return CommandSpec("train", "Train Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& predict_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--model", Arity::Value, "PATH", "",
                         "Path to the trained model"});
        flags.push_back({"--suite", Arity::Value, "DIR", "",
                         "Directory of a model suite (a manifest.json and\n"
                         "the weight files it names), scored instead of\n"
                         "--model. Each target is the combined prediction\n"
                         "of its members, with their agreement and the\n"
                         "share of each plot's species the target's\n"
                         "vocabulary recognises. The manifest names the\n"
                         "input columns; the role flags below rename them."});
        flags.push_back({"--target", Arity::Repeatable, "NAME", "",
                         "With --suite: score only this target. Repeat per\n"
                         "target; default every target of the suite."});
        flags.push_back({"--members", Arity::Flag, "", "",
                         "With --suite: also write every member's own\n"
                         "prediction, '<target>_seed<N>'."});
        flags.push_back({"--no-verify", Arity::Flag, "", "",
                         "With --suite: skip comparing each member's size\n"
                         "and SHA-256 with the manifest before loading."});
        flags.push_back({"--header", Arity::Value, "PATH", "",
                         "Path to header CSV file"});
        flags.push_back({"--species", Arity::Value, "PATH", "",
                         "Path to species CSV file"});
        flags.push_back({"--output", Arity::Value, "PATH", "predictions.csv",
                         "Output path for predictions"});
        append_role_flags(flags);
        flags.push_back({"--device", Arity::Value, "DEVICE", "cpu",
                         "Device to predict on: cpu, cuda or cuda:N (a CUDA\n"
                         "device that is not available is an error). Predict on a\n"
                         "5M-param MLP over 300k plots is ~12s on CPU vs ~1s\n"
                         "on GPU; the OOM and bookkeeping cost of GPU\n"
                         "predict on 16 GiB-class cards usually outweighs\n"
                         "the speedup."});
        flags.push_back({"--predict-batch-size", Arity::Value, "N", "4096",
                         "Forward-pass batch size for inference. Pass -1 to\n"
                         "disable chunking (one forward over the entire\n"
                         "dataset; can OOM on >150k plots at typical hidden\n"
                         "sizes)."});
        flags.push_back({"--vram-fraction", Arity::Value, "FLOAT", "1.0",
                         "Fraction of GPU VRAM the PyTorch caching allocator\n"
                         "may use"});
        flags.push_back({"--retrieval-header", Arity::Value, "PATH", "",
                         "Header CSV of the retrieval reference plots (with\n"
                         "--retrieval-species): normally the training plots,\n"
                         "carrying a column for each target, named as the\n"
                         "target. Each prediction is then blended with the\n"
                         "labels of its nearest reference plots in the\n"
                         "model's latent space."});
        flags.push_back({"--retrieval-species", Arity::Value, "PATH", "",
                         "Species CSV of the retrieval reference plots; alone,\n"
                         "a single table that also carries the targets."});
        append_config_flags(flags, "Retrieval", resolve::RetrievalConfig{},
                            FlagNaming::members(kRetrievalConfigFlags));
        flags.push_back({"--probabilities", Arity::Flag, "", "",
                         "Also write each classification target's class\n"
                         "probabilities: one column per class,\n"
                         "'<target>_prob_<class>', after '<target>_code'.\n"
                         "The columns of one target sum to one per row and\n"
                         "the largest is the predicted class."});
        return CommandSpec("predict", "Predict Options:", std::move(flags));
    }();
    return spec;
}

inline const CommandSpec& info_spec() {
    static const CommandSpec spec = [] {
        std::vector<FlagSpec> flags;
        flags.push_back({"--model", Arity::Value, "PATH", "",
                         "Path to the trained model"});
        flags.push_back({"--suite", Arity::Value, "DIR", "",
                         "Directory of a model suite: print its manifest and\n"
                         "check every member's size and SHA-256"});
        return CommandSpec("info", "Info Options:", std::move(flags));
    }();
    return spec;
}

}  // namespace resolve_cli
